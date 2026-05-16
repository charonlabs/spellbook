"""Model generation service.

The ``Generator`` is a thin service that takes a list of IR blocks and
produces an ``IRGeneration`` — the model's response as IR. It owns its
own ``RequestSurfaceBuilder`` (constructed at init from the backend,
config, and tool registry) and delegates the actual API call to the
``ModelBackend``.

The contract is narrow on purpose: ``run(blocks, cancel_token)`` in,
``IRGeneration`` out. Streaming mechanics, provider translation, and
surface assembly live below this layer. The loop above just alternates
generate and execute.
"""

import asyncio
import logging
from contextlib import suppress

from spellbook.round_lifecycle import RoundLifecycle

from .backends.model_backend import ModelBackend
from .cancel_token import CancelToken
from .config import SpellbookConfig
from .ir_types import IRBlock, IRGeneration
from .surface_builder import RequestSurfaceBuilder

logger = logging.getLogger(__name__)


class Generator:
    def __init__(
        self,
        *,
        backend: ModelBackend,
        config: SpellbookConfig,
        surface_builder: RequestSurfaceBuilder,
    ):
        self.builder = surface_builder
        self.backend = backend
        self._config = config

    async def run(
        self,
        blocks: list[IRBlock],
        cancel_token: CancelToken,
        lifecycle: RoundLifecycle,
    ) -> IRGeneration:
        surface = self.builder.build(blocks)
        logger.info(
            "generator.surface_built model=%s blocks=%s messages=%s tools=%s max_output_tokens=%s",
            surface.model,
            len(blocks),
            len(surface.messages),
            len(surface.tools),
            surface.max_output_tokens,
        )
        logger.info(
            "generator.stream_enter model=%s messages=%s tools=%s",
            surface.model,
            len(surface.messages),
            len(surface.tools),
        )
        first_event_seen = False
        async with self.backend.stream(surface, cancel_token) as stream:
            while True:
                next_event = asyncio.create_task(stream.__anext__())
                cancelled = asyncio.create_task(cancel_token.wait_cancelled())

                done, _ = await asyncio.wait(
                    {next_event, cancelled}, return_when=asyncio.FIRST_COMPLETED
                )

                if cancelled in done:
                    next_event.cancel()
                    with suppress(asyncio.CancelledError):
                        await next_event
                    return stream.get_current_response(stop_reason="cancelled")

                cancelled.cancel()
                with suppress(asyncio.CancelledError):
                    await cancelled

                try:
                    event = next_event.result()
                except StopAsyncIteration:
                    logger.info(
                        "generator.stream_exhausted model=%s first_event_seen=%s",
                        surface.model,
                        first_event_seen,
                    )
                    return await stream.get_final_response()

                if not first_event_seen:
                    first_event_seen = True
                    logger.info(
                        "generator.first_stream_event model=%s event=%s kind=%s",
                        surface.model,
                        type(event).__name__,
                        getattr(event, "kind", None),
                    )
                await lifecycle.on_stream_event(event)

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
import random
from collections.abc import Callable, Coroutine, Mapping
from contextlib import suppress
from typing import Any

from spellbook.round_lifecycle import RoundLifecycle

from .backends.model_backend import ModelBackend, RequestSurface
from .cancel_token import CancelToken
from .config import SpellbookConfig
from .ir_types import IRBlock, IRGeneration
from .surface_builder import RequestSurfaceBuilder

logger = logging.getLogger(__name__)

_RETRYABLE_API_ERROR_TYPES = {"api_error", "overloaded_error"}
_RETRYABLE_API_STATUS_CODES = {500, 529}
_DEFAULT_MAX_RETRY_ATTEMPTS = 5
_DEFAULT_RETRY_BASE_DELAY_SECONDS = 1.0
_DEFAULT_RETRY_FACTOR = 2.0
_DEFAULT_RETRY_CAP_SECONDS = 30.0
_DEFAULT_RETRY_JITTER_FRACTION = 0.25


class Generator:
    def __init__(
        self,
        *,
        backend: ModelBackend,
        config: SpellbookConfig,
        surface_builder: RequestSurfaceBuilder,
        max_retry_attempts: int = _DEFAULT_MAX_RETRY_ATTEMPTS,
        retry_base_delay_seconds: float = _DEFAULT_RETRY_BASE_DELAY_SECONDS,
        retry_factor: float = _DEFAULT_RETRY_FACTOR,
        retry_cap_seconds: float = _DEFAULT_RETRY_CAP_SECONDS,
        retry_jitter_seconds: Callable[[float], float] | None = None,
        sleep: Callable[[float], Coroutine[Any, Any, None]] = asyncio.sleep,
    ):
        if max_retry_attempts < 1:
            raise ValueError("max_retry_attempts must be at least 1")
        self.builder = surface_builder
        self.backend = backend
        self._config = config
        self._max_retry_attempts = max_retry_attempts
        self._retry_base_delay_seconds = retry_base_delay_seconds
        self._retry_factor = retry_factor
        self._retry_cap_seconds = retry_cap_seconds
        self._retry_jitter_seconds = retry_jitter_seconds or _retry_jitter_seconds
        self._sleep = sleep

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
        attempt = 1
        while True:
            try:
                return await self._run_once(surface, cancel_token, lifecycle)
            except Exception as exc:
                if (
                    not _is_retryable_generation_error(exc)
                    or attempt >= self._max_retry_attempts
                ):
                    raise

                delay = self._retry_delay_seconds(attempt)
                logger.warning(
                    "generator.transient_api_error_retry "
                    "model=%s attempt=%s next_attempt=%s max_attempts=%s "
                    "delay_seconds=%.3f messages=%s tools=%s status_code=%s "
                    "error_type=%s request_id=%s exc_type=%s",
                    surface.model,
                    attempt,
                    attempt + 1,
                    self._max_retry_attempts,
                    delay,
                    len(surface.messages),
                    len(surface.tools),
                    _api_status_code(exc),
                    _api_error_type(exc),
                    getattr(exc, "request_id", None),
                    type(exc).__name__,
                )
                if await self._sleep_or_cancel(delay, cancel_token):
                    return IRGeneration(
                        model=surface.model,
                        blocks=[],
                        stop_reason="cancelled",
                        usage=None,
                    )
                attempt += 1

    async def _run_once(
        self,
        surface: RequestSurface,
        cancel_token: CancelToken,
        lifecycle: RoundLifecycle,
    ) -> IRGeneration:
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

    def _retry_delay_seconds(self, failed_attempt: int) -> float:
        raw_delay = self._retry_base_delay_seconds * (
            self._retry_factor ** (failed_attempt - 1)
        )
        capped_delay = min(raw_delay, self._retry_cap_seconds)
        jitter = max(0.0, self._retry_jitter_seconds(capped_delay))
        return min(capped_delay + jitter, self._retry_cap_seconds)

    async def _sleep_or_cancel(
        self,
        delay_seconds: float,
        cancel_token: CancelToken,
    ) -> bool:
        if cancel_token.cancelled:
            return True

        sleep_task = asyncio.create_task(self._sleep(delay_seconds))
        cancelled = asyncio.create_task(cancel_token.wait_cancelled())
        done, pending = await asyncio.wait(
            {sleep_task, cancelled}, return_when=asyncio.FIRST_COMPLETED
        )

        for task in pending:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

        if cancelled in done:
            with suppress(asyncio.CancelledError):
                await sleep_task
            return True

        cancelled.cancel()
        with suppress(asyncio.CancelledError):
            await cancelled
        await sleep_task
        return False


def _retry_jitter_seconds(delay_seconds: float) -> float:
    jitter_cap = min(
        delay_seconds * _DEFAULT_RETRY_JITTER_FRACTION,
        _DEFAULT_RETRY_CAP_SECONDS - delay_seconds,
    )
    if jitter_cap <= 0:
        return 0.0
    return random.uniform(0.0, jitter_cap)


def _is_retryable_generation_error(exc: Exception) -> bool:
    if _looks_like_api_status_error(exc):
        status_code = _api_status_code(exc)
        if status_code is not None and 400 <= status_code < 500:
            return False
        return (
            _api_error_type(exc) in _RETRYABLE_API_ERROR_TYPES
            or status_code in _RETRYABLE_API_STATUS_CODES
        )
    return _looks_like_api_connection_error(exc)


def _looks_like_api_status_error(exc: Exception) -> bool:
    return "APIStatusError" in _exception_class_names(exc)


def _looks_like_api_connection_error(exc: Exception) -> bool:
    names = _exception_class_names(exc)
    if "APIConnectionError" in names or "APITimeoutError" in names:
        return True
    if "TimeoutException" in names:
        return True
    return isinstance(exc, TimeoutError)


def _exception_class_names(exc: Exception) -> set[str]:
    return {cls.__name__ for cls in type(exc).__mro__}


def _api_status_code(exc: Exception) -> int | None:
    value = getattr(exc, "status_code", None)
    if isinstance(value, int):
        return value
    return None


def _api_error_type(exc: Exception) -> str | None:
    value = getattr(exc, "type", None)
    if isinstance(value, str):
        return value

    body = getattr(exc, "body", None)
    if not isinstance(body, Mapping):
        return None

    error = body.get("error")
    if isinstance(error, Mapping):
        nested_type = error.get("type")
        if isinstance(nested_type, str):
            return nested_type

    top_level_type = body.get("type")
    if isinstance(top_level_type, str):
        return top_level_type
    return None

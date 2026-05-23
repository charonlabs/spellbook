"""Small in-process SDK for casting Spellbook entities from Python code."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from spellbook.app.event_bus import AppEventBus, AppEventSubscription
from spellbook.app.protocol import (
    AwarenessResponse,
    ContextBlockAddedEvent,
    HealthResponse,
    ServerEvent,
    StreamEvent,
    TurnEndedEvent,
    TurnStartedEvent,
)
from spellbook.app.runtime import CoreAppRuntime
from spellbook.config import SpellbookConfig
from spellbook.custom import CustomSurface
from spellbook.ir_types import (
    IRAssistantTextBlock,
    IRBlock,
    IRInboundMessage,
    IRLoopResult,
    IRStreamEvent,
    IRUserTextBlock,
    StopReason,
)

SDK_REQUEST_ID_KEY = "spellbook_sdk_request_id"
SDK_SOURCE = "sdk"

RuntimeFactory = Callable[
    [Path, SpellbookConfig | None, CustomSurface | None, AppEventBus], CoreAppRuntime
]


@dataclass(frozen=True)
class TurnResult:
    """Result of one `Entity.send(...)` call."""

    text: str
    blocks: list[IRBlock]
    turn_id: str
    stop_reason: StopReason
    loop_result: IRLoopResult
    stream_events: list[IRStreamEvent] = field(default_factory=list)


class Spell:
    """An inert recipe for casting Spellbook entities."""

    def __init__(
        self,
        *,
        config: SpellbookConfig,
        transcript_path: Path | None = None,
        custom_surface: CustomSurface | None = None,
        _runtime_factory: RuntimeFactory | None = None,
    ) -> None:
        self.config = config
        self.transcript_path = transcript_path
        self.custom_surface = custom_surface
        self._runtime_factory = _runtime_factory or _default_runtime_factory

    def cast(self, *, transcript_path: Path | None = None) -> EntityCast:
        """Create an async context manager for one live entity runtime."""
        resolved_path = self._resolve_transcript_path(transcript_path)
        config = None if resolved_path.exists() else self.config
        return EntityCast(
            config=config,
            transcript_path=resolved_path,
            custom_surface=self.custom_surface,
            runtime_factory=self._runtime_factory,
        )

    async def once(
        self,
        message: str | IRInboundMessage,
        *,
        transcript_path: Path | None = None,
        metadata: dict | None = None,
    ) -> TurnResult:
        """Cast one entity, send one message, and shut it down."""
        async with self.cast(transcript_path=transcript_path) as entity:
            return await entity.send(message, metadata=metadata)

    def _resolve_transcript_path(self, override: Path | None) -> Path:
        path = override or self.transcript_path
        if path is None:
            return _new_transcript_path()
        return path.expanduser().resolve()


class EntityCast:
    """Async context manager returned by `Spell.cast()`."""

    def __init__(
        self,
        *,
        config: SpellbookConfig | None,
        transcript_path: Path,
        custom_surface: CustomSurface | None,
        runtime_factory: RuntimeFactory,
    ) -> None:
        self._config = config
        self._transcript_path = transcript_path
        self._custom_surface = custom_surface
        self._runtime_factory = runtime_factory
        self._runtime: CoreAppRuntime | None = None

    async def __aenter__(self) -> Entity:
        self._transcript_path.parent.mkdir(parents=True, exist_ok=True)
        bus = AppEventBus()
        runtime = self._runtime_factory(
            self._transcript_path,
            self._config,
            self._custom_surface,
            bus,
        )
        self._runtime = runtime
        await runtime.startup()
        return Entity(runtime=runtime)

    async def __aexit__(self, *exc: object) -> None:
        if self._runtime is not None:
            await self._runtime.shutdown()
            self._runtime = None


class Entity:
    """A live Spellbook entity runtime."""

    def __init__(self, *, runtime: CoreAppRuntime) -> None:
        self._runtime = runtime
        self._send_lock = asyncio.Lock()

    @property
    def transcript_path(self) -> Path:
        return self._runtime.transcript_path

    async def send(
        self,
        message: str | IRInboundMessage,
        *,
        metadata: dict | None = None,
    ) -> TurnResult:
        """Submit one turn and wait for the matching turn result."""
        stream = self.stream(message, metadata=metadata)
        async for _event in stream:
            pass
        return await stream.result()

    def stream(
        self,
        message: str | IRInboundMessage,
        *,
        metadata: dict | None = None,
    ) -> EntityStream:
        """Submit one turn and yield live IR stream events."""
        return EntityStream(entity=self, message=message, metadata=metadata)

    async def interrupt(self) -> bool:
        """Interrupt the active turn, if any."""
        return await self._runtime.interrupt()

    def health(self) -> HealthResponse:
        return self._runtime.build_health()

    def awareness(self) -> AwarenessResponse:
        return self._runtime.build_awareness()

    async def _next_event_or_session_exit(
        self, subscription: AppEventSubscription
    ) -> ServerEvent:
        event_task = asyncio.create_task(subscription.__anext__())
        task = self._runtime._session_task
        wait_for: set[asyncio.Task] = {event_task}
        if task is not None:
            wait_for.add(task)

        done, _ = await asyncio.wait(wait_for, return_when=asyncio.FIRST_COMPLETED)
        if event_task in done:
            try:
                return event_task.result()
            except StopAsyncIteration as e:
                raise RuntimeError(
                    "Spellbook entity event stream closed before the turn ended."
                ) from e

        event_task.cancel()
        try:
            await event_task
        except asyncio.CancelledError:
            pass

        if task is None:
            raise RuntimeError("Spellbook entity session is not running.")
        if task.cancelled():
            raise RuntimeError("Spellbook entity session was cancelled.")
        exc = task.exception()
        if exc is not None:
            raise RuntimeError("Spellbook entity session crashed.") from exc
        raise RuntimeError("Spellbook entity session exited before the turn ended.")


class EntityStream:
    """Async iterator returned by `Entity.stream(...)`."""

    def __init__(
        self,
        *,
        entity: Entity,
        message: str | IRInboundMessage,
        metadata: dict | None,
    ) -> None:
        self._entity = entity
        self._message = message
        self._metadata = metadata
        self._request_id = f"sdk_{uuid4().hex}"
        self._subscription: AppEventSubscription | None = None
        self._turn_id: str | None = None
        self._blocks: list[IRBlock] = []
        self._stream_events: list[IRStreamEvent] = []
        self._result: TurnResult | None = None
        self._started = False
        self._closed = False
        self._lock_acquired = False

    def __aiter__(self) -> EntityStream:
        return self

    async def __anext__(self) -> IRStreamEvent:
        if self._result is not None:
            raise StopAsyncIteration
        await self._start()
        assert self._subscription is not None

        try:
            while True:
                event = await self._entity._next_event_or_session_exit(
                    self._subscription
                )
                match event:
                    case TurnStartedEvent():
                        if _event_request_id(event) == self._request_id:
                            self._turn_id = event.turn_id
                    case StreamEvent():
                        if self._turn_id is not None:
                            self._stream_events.append(event.event)
                            return event.event
                    case ContextBlockAddedEvent():
                        if self._turn_id is not None:
                            self._blocks.append(event.block)
                    case TurnEndedEvent():
                        if self._turn_id is not None and event.turn_id == self._turn_id:
                            self._result = TurnResult(
                                text=_blocks_text(self._blocks),
                                blocks=self._blocks,
                                turn_id=self._turn_id,
                                stop_reason=event.result.stop_reason,
                                loop_result=event.result,
                                stream_events=self._stream_events,
                            )
                            self._close()
                            raise StopAsyncIteration
                    case _:
                        continue
        except Exception:
            self._close()
            raise

    async def result(self) -> TurnResult:
        """Return the final turn result, draining the stream if needed."""
        async for _event in self:
            pass
        if self._result is None:
            raise RuntimeError("Spellbook entity stream ended without a result.")
        return self._result

    async def _start(self) -> None:
        if self._started:
            return
        self._started = True
        await self._entity._send_lock.acquire()
        self._lock_acquired = True
        inbound = _coerce_inbound_message(
            self._message,
            metadata=self._metadata,
            request_id=self._request_id,
        )
        self._subscription = self._entity._runtime.bus.subscribe()
        try:
            await self._entity._runtime.submit_message(inbound)
        except Exception:
            self._close()
            raise

    def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._subscription is not None:
            self._subscription.close()
            self._subscription = None
        if self._lock_acquired:
            self._entity._send_lock.release()
            self._lock_acquired = False


def _default_runtime_factory(
    transcript_path: Path,
    config: SpellbookConfig | None,
    custom_surface: CustomSurface | None,
    bus: AppEventBus,
) -> CoreAppRuntime:
    return CoreAppRuntime(
        transcript_path=transcript_path,
        config=config,
        custom_surface=custom_surface,
        bus=bus,
    )


def _new_transcript_path() -> Path:
    sessions_dir = Path.home() / ".spellbook" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    return (sessions_dir / f"sdk_{uuid4().hex}.jsonl").resolve()


def _coerce_inbound_message(
    message: str | IRInboundMessage,
    *,
    metadata: dict | None,
    request_id: str,
) -> IRInboundMessage:
    source_metadata = dict(metadata or {})
    source_metadata.setdefault("source", SDK_SOURCE)
    source_metadata[SDK_REQUEST_ID_KEY] = request_id
    if isinstance(message, str):
        return IRInboundMessage(
            blocks=[IRUserTextBlock(text=message, origin="human")],
            source_metadata=source_metadata,
            delivery="turn",
        )

    merged_metadata = dict(message.source_metadata)
    merged_metadata.update(source_metadata)
    return message.model_copy(update={"source_metadata": merged_metadata})


def _event_request_id(event: TurnStartedEvent) -> str | None:
    raw = event.message.source_metadata.get(SDK_REQUEST_ID_KEY)
    return raw if isinstance(raw, str) else None


def _blocks_text(blocks: list[IRBlock]) -> str:
    return "\n\n".join(
        block.text for block in blocks if isinstance(block, IRAssistantTextBlock)
    )

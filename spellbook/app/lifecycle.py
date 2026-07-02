from collections.abc import Awaitable, Callable

from spellbook.app.event_bus import AppEventBus
from spellbook.app.protocol import (
    ContextBlockAddedEvent,
    RuntimeStateEvent,
    StreamEvent,
    SystemResponseEvent,
    TurnEndedEvent,
    TurnStartedEvent,
)
from spellbook.ir_types import (
    IRExecution,
    IRGeneration,
    IRLoopResult,
    IRStreamEvent,
)
from spellbook.round_lifecycle import RoundContext, RoundLifecycle
from spellbook.session_lifecycle import SessionContext, SessionLifecycle
from spellbook.system_response import SystemResponse

TurnStartedHook = Callable[[SessionContext, str], Awaitable[None]]
TurnEndedHook = Callable[[SessionContext, IRLoopResult, str], Awaitable[None]]


class AppRoundLifecycle(RoundLifecycle):
    """Round lifecycle for the App Server."""

    def __init__(self, bus: AppEventBus):
        self._bus = bus

    async def on_stream_event(self, event: IRStreamEvent) -> None:
        self._bus.publish(event=StreamEvent(event=event))

    async def after_generate(
        self,
        ctx: RoundContext,
        generation: IRGeneration,
    ) -> None:
        for block in generation.blocks:
            self._bus.publish(event=ContextBlockAddedEvent(block=block))

    async def after_execute(
        self,
        ctx: RoundContext,
        execution: IRExecution,
    ) -> None:
        for block in execution.blocks:
            self._bus.publish(event=ContextBlockAddedEvent(block=block))


class AppSessionLifecycle(SessionLifecycle):
    """Session lifecycle for the App Server."""

    def __init__(
        self,
        bus: AppEventBus,
        *,
        before_turn_started: TurnStartedHook | None = None,
        after_turn_ended: TurnEndedHook | None = None,
    ):
        self._bus = bus
        self._before_turn_started = before_turn_started
        self._after_turn_ended = after_turn_ended

    async def on_enter_idle(self, ctx: SessionContext) -> None:
        self._bus.publish(event=RuntimeStateEvent(state="idle"))

    async def on_exit_idle(self, ctx: SessionContext, reason: str) -> None:
        if reason == "message":
            self._bus.publish(event=RuntimeStateEvent(state="running"))

    async def on_turn_started(self, ctx: SessionContext, turn_id: str) -> None:
        assert ctx.inbound is not None
        if self._before_turn_started is not None:
            await self._before_turn_started(ctx, turn_id)

        self._bus.publish(
            event=TurnStartedEvent(
                turn=ctx.turn_idx, turn_id=turn_id, message=ctx.inbound
            )
        )

    async def on_turn_ended(
        self, ctx: SessionContext, result: IRLoopResult, turn_id: str
    ) -> None:
        if self._after_turn_ended is not None:
            await self._after_turn_ended(ctx, result, turn_id)

        self._bus.publish(
            event=TurnEndedEvent(turn=ctx.turn_idx, turn_id=turn_id, result=result)
        )

    async def on_system_response(
        self, ctx: SessionContext, response: SystemResponse
    ) -> None:
        self._bus.publish(
            event=SystemResponseEvent(
                command=response.command,
                content=response.content,
                plaintext=response.plaintext,
                metadata=response.metadata,
            )
        )

    async def on_shutdown(self, ctx: SessionContext) -> None:
        self._bus.publish(event=RuntimeStateEvent(state="suspended"))

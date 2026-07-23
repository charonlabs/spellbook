from dataclasses import dataclass
from typing import Literal, Protocol

from .ir_types import IRInboundMessage, IRLoopResult
from .system_response import SystemResponse

DreamingOutcome = Literal["completed", "refused", "failed"]


class DreamingRuntime(Protocol):
    """The narrow session-state seam available to the Sleep tool."""

    async def enter_dreaming(self) -> None: ...

    async def exit_dreaming(self, outcome: DreamingOutcome) -> None: ...


@dataclass  # # not pydantic because mutable, internal state
class SessionContext:
    # TODO: not sure yet exactly what to put here
    session_id: str
    turn_idx: int
    inbound: IRInboundMessage | None = None


class SessionLifecycle:
    """Session-level hooks. Distinct from RoundLifecycle."""

    async def on_enter_idle(self, ctx: SessionContext) -> None:
        """Manager transitioned to idle. Hearth crackles, ambient behaviors,
        idle-time subsystems start their work."""

    async def on_exit_idle(self, ctx: SessionContext, reason: str) -> None:
        """About to leave idle. Reason: 'message' | 'rest' | 'shutdown'."""

    async def on_turn_started(self, ctx: SessionContext, turn_id: str) -> None:
        """About to invoke run_loop for a new turn."""

    async def on_turn_ended(
        self, ctx: SessionContext, result: IRLoopResult, turn_id: str
    ) -> None:
        """run_loop returned. Post-turn analysis: tiredness accumulation,
        rest decision, any subsystem work triggered by turn boundaries."""

    async def on_enter_dreaming(self, ctx: SessionContext) -> None:
        """Self-triggered or forced Sleep entered its synchronous dream window."""

    async def on_exit_dreaming(
        self,
        ctx: SessionContext,
        outcome: DreamingOutcome,
    ) -> None:
        """Sleep returned to the interrupted waking turn, with an honest outcome."""

    async def on_system_response(
        self, ctx: SessionContext, response: SystemResponse
    ) -> None:
        """A system-to-user response was recorded outside the model turn path."""

    async def on_shutdown(self, ctx: SessionContext) -> None:
        """Shutdown requested."""


class CompositeSessionLifecycle(SessionLifecycle):
    """Session lifecycles serially executed in list order."""

    def __init__(self, lifecycles: list[SessionLifecycle]):
        self._lifecycles = lifecycles

    async def on_enter_idle(self, ctx: SessionContext) -> None:
        for lifecycle in self._lifecycles:
            await lifecycle.on_enter_idle(ctx)

    async def on_exit_idle(self, ctx: SessionContext, reason: str) -> None:
        for lifecycle in self._lifecycles:
            await lifecycle.on_exit_idle(ctx, reason)

    async def on_turn_started(self, ctx: SessionContext, turn_id: str) -> None:
        for lifecycle in self._lifecycles:
            await lifecycle.on_turn_started(ctx, turn_id)

    async def on_turn_ended(
        self, ctx: SessionContext, result: IRLoopResult, turn_id: str
    ) -> None:
        for lifecycle in self._lifecycles:
            await lifecycle.on_turn_ended(ctx, result, turn_id)

    async def on_enter_dreaming(self, ctx: SessionContext) -> None:
        for lifecycle in self._lifecycles:
            await lifecycle.on_enter_dreaming(ctx)

    async def on_exit_dreaming(
        self,
        ctx: SessionContext,
        outcome: DreamingOutcome,
    ) -> None:
        for lifecycle in self._lifecycles:
            await lifecycle.on_exit_dreaming(ctx, outcome)

    async def on_system_response(
        self, ctx: SessionContext, response: SystemResponse
    ) -> None:
        for lifecycle in self._lifecycles:
            await lifecycle.on_system_response(ctx, response)

    async def on_shutdown(self, ctx: SessionContext) -> None:
        for lifecycle in self._lifecycles:
            await lifecycle.on_shutdown(ctx)

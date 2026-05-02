"""Round lifecycle hooks and shared round state.

``RoundContext`` is the mutable shared state threaded through one round
of the inner loop. ``blocks`` is the full accumulated block stream;
``blocks_this_round`` is the subset added during the current round
(reset at the top of each iteration). ``cancel_token`` is the same
token used by the Generator and Executor. It's a dataclass not a
Pydantic model — mutation is expected, and the internal-state
discipline doesn't need Pydantic's validation machinery.

``RoundLifecycle`` is the hook surface. Subsystems that need to fire
at round boundaries (TokenMeter, BlockDetector, TTLRegistry, Planner,
FooterController) will subscribe via this interface. The base class
has no-op defaults so testable loop tests don't need to implement
everything.
"""

from dataclasses import dataclass

from .cancel_token import CancelToken
from .ir_types import IRBlock, IRExecution, IRGeneration, IRStreamEvent, StopReason


@dataclass  # not pydantic because mutable, internal state that needs to carry CancelToken
class RoundContext:
    blocks: list[IRBlock]
    round_number: int
    cancel_token: CancelToken
    blocks_this_round: list[IRBlock]


class RoundLifecycle:
    """Hooks wired into the loop. All async — hooks can await I/O."""

    async def before_round(self, ctx: RoundContext) -> None:
        """Fire before generate. Last chance to inject footers or modify
        what the model will see in this round."""

    async def on_stream_event(self, event: IRStreamEvent) -> None:
        """Fire inside the streaming loop on each event."""

    async def after_generate(
        self,
        ctx: RoundContext,
        generation: IRGeneration,
    ) -> None:
        """Fire after generate, before execute. Observe usage telemetry."""

    async def after_execute(
        self,
        ctx: RoundContext,
        execution: IRExecution,
    ) -> None:
        """Fire after tool execution. Register auto-TTLs, feed block detection."""

    async def between_rounds(self, ctx: RoundContext) -> None:
        """Fire at the round boundary, after the full round completes.
        Compaction decisions, TTL ticks, pressure checks, block analysis.
        Queue footers for the next round."""

    async def on_loop_exit(self, ctx: RoundContext, stop_reason: StopReason) -> None:
        """Fire when the loop terminates. Reason is one of:
        'end_turn' | 'cancelled' | 'error'."""


class CompositeRoundLifecycle(RoundLifecycle):
    """Lifecycles are serially executed IN LIST ORDER."""

    def __init__(self, lifecycles: list[RoundLifecycle]):
        self._lifecycles = lifecycles

    def add_pre(self, lifecycle: RoundLifecycle) -> None:
        self._lifecycles.insert(0, lifecycle)

    def add_post(self, lifecycle: RoundLifecycle) -> None:
        self._lifecycles.append(lifecycle)

    async def before_round(self, ctx: RoundContext) -> None:
        for lifecycle in self._lifecycles:
            await lifecycle.before_round(ctx)

    async def on_stream_event(self, event: IRStreamEvent) -> None:
        for lifecycle in self._lifecycles:
            await lifecycle.on_stream_event(event)

    async def after_generate(
        self,
        ctx: RoundContext,
        generation: IRGeneration,
    ) -> None:
        for lifecycle in self._lifecycles:
            await lifecycle.after_generate(ctx, generation)

    async def after_execute(
        self,
        ctx: RoundContext,
        execution: IRExecution,
    ) -> None:
        for lifecycle in self._lifecycles:
            await lifecycle.after_execute(ctx, execution)

    async def between_rounds(self, ctx: RoundContext) -> None:
        for lifecycle in self._lifecycles:
            await lifecycle.between_rounds(ctx)

    async def on_loop_exit(self, ctx: RoundContext, stop_reason: StopReason) -> None:
        for lifecycle in self._lifecycles:
            await lifecycle.on_loop_exit(ctx, stop_reason)

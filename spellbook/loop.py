"""The inner loop.

``run_loop`` is the heartbeat of a Spellbook entity. It alternates
generate and execute until the model stops calling tools or cancellation
fires. Five lifecycle hooks fire at fixed points:

- ``before_round``      — last chance to inject context for this round
- ``after_generate``    — observe usage, register TTLs on the assistant msg
- ``after_execute``     — register auto-TTLs for tool results, feed block detection
- ``between_rounds``    — compaction, TTL tick, pressure check, block analysis
- ``on_loop_exit``      — loop terminated, for any stop reason

The loop is round-centric: every API call is a round, hooks fire every
round. Turns are a higher-level concept owned by the session manager;
the loop doesn't know the word "turn."

Returns an ``IRLoopResult`` with full fidelity (every generation, every
execution, the full block stream at termination) so the outer layer
can record without loss.
"""

from .cancel_token import CancelToken
from .executor import Executor
from .generator import Generator
from .ir_types import IRBlock, IRExecution, IRGeneration, IRLoopResult
from .round_lifecycle import RoundContext, RoundLifecycle


async def run_loop(
    *,
    generator: Generator,
    executor: Executor,
    lifecycle: RoundLifecycle,
    initial_blocks: list[IRBlock],
    cancel_token: CancelToken,
) -> IRLoopResult:
    """Alternate generate and execute until the model stops calling tools
    or cancellation fires."""
    ctx = RoundContext(
        blocks=list(initial_blocks),
        round_number=0,
        cancel_token=cancel_token,
        blocks_this_round=[],
    )
    generations: list[IRGeneration] = []
    executions: list[IRExecution] = []

    while not cancel_token.cancelled:
        ctx.round_number += 1
        ctx.blocks_this_round = []

        await lifecycle.before_round(ctx)
        generation = await generator.run(ctx.blocks, cancel_token, lifecycle)
        ctx.blocks.extend(generation.blocks)
        ctx.blocks_this_round.extend(generation.blocks)
        generations.append(generation)

        await lifecycle.after_generate(ctx, generation)

        if generation.stop_reason != "tool_use":
            await lifecycle.on_loop_exit(ctx, generation.stop_reason)
            return IRLoopResult(
                blocks=ctx.blocks,
                generations=generations,
                executions=executions,
                stop_reason=generation.stop_reason,
                rounds=ctx.round_number,
            )

        execution = await executor.run(generation.tool_calls, cancel_token)
        ctx.blocks.extend(execution.blocks)
        ctx.blocks_this_round.extend(execution.blocks)
        executions.append(execution)

        await lifecycle.after_execute(ctx, execution)

        if execution.cancelled_early:
            await lifecycle.on_loop_exit(ctx, "cancelled")
            return IRLoopResult(
                blocks=ctx.blocks,
                generations=generations,
                executions=executions,
                stop_reason="cancelled",
                rounds=ctx.round_number,
            )

        await lifecycle.between_rounds(ctx)

    await lifecycle.on_loop_exit(ctx, "cancelled")
    return IRLoopResult(
        blocks=ctx.blocks,
        generations=generations,
        executions=executions,
        stop_reason="cancelled",
        rounds=ctx.round_number,
    )

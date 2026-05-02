"""Tests for run_loop — the inner loop itself.

Uses fake Generator and Executor implementations to verify:
- Round lifecycle hooks fire in the right order at every round
- Single-round end_turn terminates cleanly
- Multi-round tool use alternates generate and execute correctly
- Non-tool_use stop reasons (max_tokens, error, etc.) exit the loop
- Cancellation between rounds exits cleanly
- Cancellation after generate exits before execute runs
- Cancellation during execute exits immediately
- IRLoopResult carries full fidelity: all generations, all executions, final blocks
"""

from __future__ import annotations

from typing import Any

import pytest

from spellbook.cancel_token import CancelToken
from spellbook.ir_types import (
    IRAssistantTextBlock,
    IRBlock,
    IRExecution,
    IRGeneration,
    IRToolCallBlock,
    IRToolResultBlock,
    IRToolTextBlock,
    IRUsage,
    IRUserTextBlock,
    StopReason,
)
from spellbook.loop import run_loop
from spellbook.round_lifecycle import RoundLifecycle

# --- Fakes ---


class _FakeGenerator:
    """Returns queued IRGenerations in order. Records the block stream at each call."""

    def __init__(self, responses: list[IRGeneration]):
        self._queue = list(responses)
        self.calls_seen: list[list[IRBlock]] = []

    async def run(
        self,
        blocks: list[IRBlock],
        cancel_token: CancelToken,
        lifecycle: RoundLifecycle,
    ) -> IRGeneration:
        self.calls_seen.append(list(blocks))
        if not self._queue:
            raise RuntimeError("FakeGenerator ran out of responses")
        return self._queue.pop(0)


class _FakeExecutor:
    """Returns queued IRExecutions in order."""

    def __init__(self, responses: list[IRExecution]):
        self._queue = list(responses)
        self.calls_received: list[list[IRToolCallBlock]] = []

    async def run(self, calls, cancel_token) -> IRExecution:
        self.calls_received.append(list(calls))
        if not self._queue:
            raise RuntimeError("FakeExecutor ran out of responses")
        return self._queue.pop(0)


class _RecordingLifecycle(RoundLifecycle):
    """Records every hook invocation in order, so tests can verify sequencing."""

    def __init__(self) -> None:
        self.events: list[tuple[Any, ...]] = []

    async def before_round(self, ctx) -> None:
        self.events.append(("before_round", ctx.round_number))

    async def after_generate(self, ctx, generation) -> None:
        self.events.append(("after_generate", ctx.round_number, generation.stop_reason))

    async def after_execute(self, ctx, execution) -> None:
        self.events.append(
            ("after_execute", ctx.round_number, execution.cancelled_early)
        )

    async def between_rounds(self, ctx) -> None:
        self.events.append(("between_rounds", ctx.round_number))

    async def on_loop_exit(self, ctx, stop_reason) -> None:
        self.events.append(("on_loop_exit", ctx.round_number, stop_reason))


def _gen(
    blocks: list[IRBlock] | None = None,
    stop_reason: StopReason = "end_turn",
) -> IRGeneration:
    return IRGeneration(
        model="test",
        blocks=blocks or [],
        stop_reason=stop_reason,
        usage=IRUsage(),
    )


def _tool_call(
    call_id: str, tool: str = "Echo", input: dict | None = None
) -> IRToolCallBlock:
    return IRToolCallBlock(
        origin="model",
        call_id=call_id,
        tool=tool,
        input=input or {},
    )


def _tool_result(call_id: str, text: str = "ok") -> IRToolResultBlock:
    return IRToolResultBlock(
        call_id=call_id,
        tool="Echo",
        content=[IRToolTextBlock(text=text)],
    )


def _initial() -> list[IRBlock]:
    return [IRUserTextBlock(text="hi", origin="human")]


# --- Single-round tests ---


class TestSingleRound:
    @pytest.mark.asyncio
    async def test_end_turn_exits_after_one_round(self) -> None:
        """Model says end_turn on first response; loop exits after one round."""
        gen = _FakeGenerator(
            [
                _gen(
                    blocks=[IRAssistantTextBlock(text="done", origin="model")],
                    stop_reason="end_turn",
                )
            ]
        )
        ex = _FakeExecutor([])  # never called
        lifecycle = _RecordingLifecycle()

        result = await run_loop(
            generator=gen,  # type: ignore
            executor=ex,  # type: ignore
            lifecycle=lifecycle,
            initial_blocks=_initial(),
            cancel_token=CancelToken(),
        )

        assert result.stop_reason == "end_turn"
        assert result.rounds == 1
        assert len(result.generations) == 1
        assert len(result.executions) == 0

    @pytest.mark.asyncio
    async def test_end_turn_hook_sequence(self) -> None:
        """end_turn fires: before_round → after_generate → on_loop_exit. No execute path."""
        gen = _FakeGenerator([_gen(stop_reason="end_turn")])
        ex = _FakeExecutor([])
        lifecycle = _RecordingLifecycle()

        await run_loop(
            generator=gen,  # type: ignore
            executor=ex,  # type: ignore
            lifecycle=lifecycle,
            initial_blocks=_initial(),
            cancel_token=CancelToken(),
        )

        kinds = [e[0] for e in lifecycle.events]
        assert kinds == [
            "before_round",
            "after_generate",
            "on_loop_exit",
        ]

    @pytest.mark.asyncio
    async def test_max_tokens_exits_loop(self) -> None:
        """max_tokens is a non-tool_use stop reason; loop exits without calling execute."""
        gen = _FakeGenerator([_gen(stop_reason="max_tokens")])
        ex = _FakeExecutor([])
        lifecycle = _RecordingLifecycle()

        result = await run_loop(
            generator=gen,  # type: ignore
            executor=ex,  # type: ignore
            lifecycle=lifecycle,
            initial_blocks=_initial(),
            cancel_token=CancelToken(),
        )

        assert result.stop_reason == "max_tokens"
        assert result.rounds == 1
        # executor never called
        assert ex.calls_received == []

    @pytest.mark.asyncio
    async def test_error_stop_reason_exits_loop(self) -> None:
        gen = _FakeGenerator([_gen(stop_reason="error")])
        ex = _FakeExecutor([])
        lifecycle = _RecordingLifecycle()

        result = await run_loop(
            generator=gen,  # type: ignore
            executor=ex,  # type: ignore
            lifecycle=lifecycle,
            initial_blocks=_initial(),
            cancel_token=CancelToken(),
        )

        assert result.stop_reason == "error"
        # loop_exit received the correct reason
        exit_event = [e for e in lifecycle.events if e[0] == "on_loop_exit"][0]
        assert exit_event[2] == "error"


# --- Multi-round tests ---


class TestMultiRound:
    @pytest.mark.asyncio
    async def test_tool_use_then_end_turn(self) -> None:
        """Round 1: tool_use. Round 2: end_turn after execute."""
        gen = _FakeGenerator(
            [
                _gen(
                    blocks=[_tool_call("toolu_1")],
                    stop_reason="tool_use",
                ),
                _gen(
                    blocks=[IRAssistantTextBlock(text="done", origin="model")],
                    stop_reason="end_turn",
                ),
            ]
        )
        ex = _FakeExecutor(
            [
                IRExecution(blocks=[_tool_result("toolu_1")]),
            ]
        )
        lifecycle = _RecordingLifecycle()

        result = await run_loop(
            generator=gen,  # type: ignore
            executor=ex,  # type: ignore
            lifecycle=lifecycle,
            initial_blocks=_initial(),
            cancel_token=CancelToken(),
        )

        assert result.stop_reason == "end_turn"
        assert result.rounds == 2
        assert len(result.generations) == 2
        assert len(result.executions) == 1

    @pytest.mark.asyncio
    async def test_full_hook_sequence_over_two_rounds(self) -> None:
        """Verify the exact hook sequence for a tool_use → end_turn session."""
        gen = _FakeGenerator(
            [
                _gen(blocks=[_tool_call("toolu_1")], stop_reason="tool_use"),
                _gen(stop_reason="end_turn"),
            ]
        )
        ex = _FakeExecutor([IRExecution(blocks=[_tool_result("toolu_1")])])
        lifecycle = _RecordingLifecycle()

        await run_loop(
            generator=gen,  # type: ignore
            executor=ex,  # type: ignore
            lifecycle=lifecycle,
            initial_blocks=_initial(),
            cancel_token=CancelToken(),
        )

        kinds_only = [e[0] for e in lifecycle.events]
        assert kinds_only == [
            # Round 1: tool_use path, all four "active" hooks fire
            "before_round",
            "after_generate",
            "after_execute",
            "between_rounds",
            # Round 2: end_turn path, no execute
            "before_round",
            "after_generate",
            "on_loop_exit",
        ]

    @pytest.mark.asyncio
    async def test_blocks_accumulate_across_rounds(self) -> None:
        """Each round extends the block stream with generation + execution output."""
        initial = _initial()
        gen = _FakeGenerator(
            [
                _gen(
                    blocks=[
                        IRAssistantTextBlock(text="thinking", origin="model"),
                        _tool_call("toolu_1"),
                    ],
                    stop_reason="tool_use",
                ),
                _gen(
                    blocks=[IRAssistantTextBlock(text="final", origin="model")],
                    stop_reason="end_turn",
                ),
            ]
        )
        ex = _FakeExecutor([IRExecution(blocks=[_tool_result("toolu_1", "output")])])

        result = await run_loop(
            generator=gen,  # type: ignore
            executor=ex,  # type: ignore
            lifecycle=RoundLifecycle(),
            initial_blocks=initial,
            cancel_token=CancelToken(),
        )

        # Final block stream: initial user + round1 gen + round1 exec + round2 gen
        # 1 user + (1 text + 1 tool_call) + 1 tool_result + 1 text = 5 blocks
        assert len(result.blocks) == 5
        # First block is the initial user message
        assert isinstance(result.blocks[0], IRUserTextBlock)
        # Round 2's generator saw the full stream through round 1
        assert (
            len(gen.calls_seen[1]) == 4
        )  # initial + gen1 text + gen1 tool_call + exec1 result

    @pytest.mark.asyncio
    async def test_executor_receives_tool_calls_from_generation(self) -> None:
        """The Executor's input is exactly the tool_calls property of the preceding generation."""
        gen = _FakeGenerator(
            [
                _gen(
                    blocks=[
                        IRAssistantTextBlock(text="thinking", origin="model"),
                        _tool_call("toolu_a", tool="Echo"),
                        _tool_call("toolu_b", tool="Read"),
                    ],
                    stop_reason="tool_use",
                ),
                _gen(stop_reason="end_turn"),
            ]
        )
        ex = _FakeExecutor(
            [
                IRExecution(
                    blocks=[
                        _tool_result("toolu_a"),
                        _tool_result("toolu_b"),
                    ]
                ),
            ]
        )

        await run_loop(
            generator=gen,  # type: ignore
            executor=ex,  # type: ignore
            lifecycle=RoundLifecycle(),
            initial_blocks=_initial(),
            cancel_token=CancelToken(),
        )

        # Executor saw only the tool_call blocks, not the text
        assert len(ex.calls_received) == 1
        assert [c.call_id for c in ex.calls_received[0]] == ["toolu_a", "toolu_b"]


# --- Cancellation tests ---


class TestCancellation:
    @pytest.mark.asyncio
    async def test_cancelled_token_before_first_round(self) -> None:
        """If the token is already cancelled, loop exits immediately with cancelled."""
        gen = _FakeGenerator([])
        ex = _FakeExecutor([])
        lifecycle = _RecordingLifecycle()
        token = CancelToken()
        token.cancel()

        result = await run_loop(
            generator=gen,  # type: ignore
            executor=ex,  # type: ignore
            lifecycle=lifecycle,
            initial_blocks=_initial(),
            cancel_token=token,
        )

        assert result.stop_reason == "cancelled"
        assert result.rounds == 0
        # Only on_loop_exit fires; no rounds ran
        kinds = [e[0] for e in lifecycle.events]
        assert kinds == ["on_loop_exit"]

    @pytest.mark.asyncio
    async def test_generator_returns_cancelled(self) -> None:
        """If the generator returns stop_reason=cancelled, loop exits without executing."""
        gen = _FakeGenerator([_gen(stop_reason="cancelled")])
        ex = _FakeExecutor([])
        lifecycle = _RecordingLifecycle()

        result = await run_loop(
            generator=gen,  # type: ignore
            executor=ex,  # type: ignore
            lifecycle=lifecycle,
            initial_blocks=_initial(),
            cancel_token=CancelToken(),
        )

        assert result.stop_reason == "cancelled"
        assert ex.calls_received == []  # executor never called
        kinds = [e[0] for e in lifecycle.events]
        assert kinds == ["before_round", "after_generate", "on_loop_exit"]

    @pytest.mark.asyncio
    async def test_executor_reports_cancelled_early(self) -> None:
        """If the executor returns cancelled_early, loop exits with cancelled."""
        gen = _FakeGenerator(
            [
                _gen(blocks=[_tool_call("toolu_1")], stop_reason="tool_use"),
            ]
        )
        ex = _FakeExecutor(
            [
                IRExecution(blocks=[_tool_result("toolu_1")], cancelled_early=True),
            ]
        )
        lifecycle = _RecordingLifecycle()

        result = await run_loop(
            generator=gen,  # type: ignore
            executor=ex,  # type: ignore
            lifecycle=lifecycle,
            initial_blocks=_initial(),
            cancel_token=CancelToken(),
        )

        assert result.stop_reason == "cancelled"
        kinds = [e[0] for e in lifecycle.events]
        # between_rounds NOT fired since we exited early
        assert "between_rounds" not in kinds
        assert kinds[-1] == "on_loop_exit"

    @pytest.mark.asyncio
    async def test_cancellation_between_rounds(self) -> None:
        """Token cancelled during between_rounds: next round doesn't start."""
        token = CancelToken()

        class _CancellingLifecycle(_RecordingLifecycle):
            async def between_rounds(self, ctx) -> None:
                await super().between_rounds(ctx)
                token.cancel()

        gen = _FakeGenerator(
            [
                # Round 1 would be tool_use → execute → between_rounds (cancels) → exit
                _gen(blocks=[_tool_call("toolu_1")], stop_reason="tool_use"),
            ]
        )
        ex = _FakeExecutor(
            [
                IRExecution(blocks=[_tool_result("toolu_1")]),
            ]
        )
        lifecycle = _CancellingLifecycle()

        result = await run_loop(
            generator=gen,  # type: ignore
            executor=ex,  # type: ignore
            lifecycle=lifecycle,
            initial_blocks=_initial(),
            cancel_token=token,
        )

        assert result.stop_reason == "cancelled"
        assert result.rounds == 1  # ran one round then exited
        kinds = [e[0] for e in lifecycle.events]
        assert kinds == [
            "before_round",
            "after_generate",
            "after_execute",
            "between_rounds",
            "on_loop_exit",
        ]


# --- IRLoopResult fidelity ---


class TestLoopResultFidelity:
    @pytest.mark.asyncio
    async def test_result_carries_all_generations_and_executions(self) -> None:
        """Every generation and execution appears in the result, in order."""
        gen_a = _gen(blocks=[_tool_call("a")], stop_reason="tool_use")
        gen_b = _gen(blocks=[_tool_call("b")], stop_reason="tool_use")
        gen_c = _gen(stop_reason="end_turn")
        exec_a = IRExecution(blocks=[_tool_result("a", "A")])
        exec_b = IRExecution(blocks=[_tool_result("b", "B")])

        gen = _FakeGenerator([gen_a, gen_b, gen_c])
        ex = _FakeExecutor([exec_a, exec_b])

        result = await run_loop(
            generator=gen,  # type: ignore
            executor=ex,  # type: ignore
            lifecycle=RoundLifecycle(),
            initial_blocks=_initial(),
            cancel_token=CancelToken(),
        )

        assert result.rounds == 3
        assert len(result.generations) == 3
        assert len(result.executions) == 2
        # The usage/stop_reason of each generation is preserved
        assert [g.stop_reason for g in result.generations] == [
            "tool_use",
            "tool_use",
            "end_turn",
        ]

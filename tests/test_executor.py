"""Tests for the Executor — tool dispatch with mock tools.

Verifies the 1:1 call→result invariant: every input call produces exactly
one IRToolResultBlock in the output, in declared order. Errors (unknown
tool, validation failure, tool-raised ToolError) all become error result
blocks — never exceptions bubbling out of run().
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from spellbook.cancel_token import CancelToken
from spellbook.config import SpellbookConfig
from spellbook.executor import Executor
from spellbook.ir_types import IRToolCallBlock, IRToolResultBlock, IRToolTextBlock
from spellbook.tools.common import (
    Tool,
    ToolError,
    ToolExecutionResult,
    ToolMetadata,
)
from spellbook.tools.registry import ToolRegistry

# --- Test tool: Echo — predictable, no side effects ---


class _EchoInput(BaseModel):
    message: str = Field(description="The message to echo back")


async def _exec_echo(meta: ToolMetadata, input: _EchoInput) -> ToolExecutionResult:
    return ToolExecutionResult(
        content=[IRToolTextBlock(text=f"echoed: {input.message}")],
    )


ECHO_TOOL: Tool[_EchoInput] = Tool(
    name="Echo",
    input_model=_EchoInput,
    exec=_exec_echo,
    category="filesystem",
)


# --- Test tool: Boom — always raises ToolError ---


class _BoomInput(BaseModel):
    reason: str = "because"


async def _exec_boom(meta: ToolMetadata, input: _BoomInput) -> ToolExecutionResult:
    raise ToolError(f"boom: {input.reason}")


BOOM_TOOL: Tool[_BoomInput] = Tool(
    name="Boom",
    input_model=_BoomInput,
    exec=_exec_boom,
    category="filesystem",
)


# --- Fixtures ---


def _make_executor(*tools: Tool) -> Executor:
    registry = ToolRegistry(tools=list(tools))
    config = SpellbookConfig(cwd=".")  # type: ignore
    return Executor(config, Path(), registry)


def _result_text(block: IRToolResultBlock, content_idx: int = 0) -> str:
    content = block.content[content_idx]
    assert isinstance(content, IRToolTextBlock)
    return content.text


# --- Tests ---


class TestSuccessfulDispatch:
    @pytest.mark.asyncio
    async def test_single_call_returns_single_result(self) -> None:
        executor = _make_executor(ECHO_TOOL)
        calls = [
            IRToolCallBlock(
                origin="model",
                call_id="toolu_1",
                tool="Echo",
                input={"message": "hello"},
            ),
        ]
        result = await executor.run(calls, CancelToken())
        assert len(result.blocks) == 1
        assert result.blocks[0].call_id == "toolu_1"
        assert result.blocks[0].tool == "Echo"
        assert result.blocks[0].is_error is False
        assert _result_text(result.blocks[0]) == "echoed: hello"
        assert result.cancelled_early is False

    @pytest.mark.asyncio
    async def test_multiple_calls_preserve_order(self) -> None:
        executor = _make_executor(ECHO_TOOL)
        calls = [
            IRToolCallBlock(
                origin="model",
                call_id="toolu_1",
                tool="Echo",
                input={"message": "first"},
            ),
            IRToolCallBlock(
                origin="model",
                call_id="toolu_2",
                tool="Echo",
                input={"message": "second"},
            ),
            IRToolCallBlock(
                origin="model",
                call_id="toolu_3",
                tool="Echo",
                input={"message": "third"},
            ),
        ]
        result = await executor.run(calls, CancelToken())
        assert len(result.blocks) == 3
        assert result.blocks[0].call_id == "toolu_1"
        assert result.blocks[1].call_id == "toolu_2"
        assert result.blocks[2].call_id == "toolu_3"
        assert _result_text(result.blocks[0]) == "echoed: first"
        assert _result_text(result.blocks[1]) == "echoed: second"
        assert _result_text(result.blocks[2]) == "echoed: third"

    @pytest.mark.asyncio
    async def test_empty_calls_returns_empty_result(self) -> None:
        executor = _make_executor(ECHO_TOOL)
        result = await executor.run([], CancelToken())
        assert result.blocks == []
        assert result.cancelled_early is False


class TestErrorPaths:
    """All error paths produce IRToolResultBlock with is_error=True.
    Nothing raises out of run()."""

    @pytest.mark.asyncio
    async def test_unknown_tool_becomes_error_block(self) -> None:
        executor = _make_executor(ECHO_TOOL)
        calls = [
            IRToolCallBlock(
                origin="model",
                call_id="toolu_1",
                tool="DoesNotExist",
                input={},
            ),
        ]
        result = await executor.run(calls, CancelToken())
        assert len(result.blocks) == 1
        assert result.blocks[0].is_error is True
        assert result.blocks[0].call_id == "toolu_1"
        assert result.blocks[0].tool == "DoesNotExist"
        assert "DoesNotExist" in _result_text(result.blocks[0])

    @pytest.mark.asyncio
    async def test_validation_failure_becomes_error_block(self) -> None:
        executor = _make_executor(ECHO_TOOL)
        calls = [
            IRToolCallBlock(
                origin="model",
                call_id="toolu_1",
                tool="Echo",
                input={},  # missing required "message" field
            ),
        ]
        result = await executor.run(calls, CancelToken())
        assert len(result.blocks) == 1
        assert result.blocks[0].is_error is True
        error_text = _result_text(result.blocks[0])
        assert "validation" in error_text.lower() or "Field required" in error_text

    @pytest.mark.asyncio
    async def test_tool_error_becomes_error_block(self) -> None:
        executor = _make_executor(BOOM_TOOL)
        calls = [
            IRToolCallBlock(
                origin="model",
                call_id="toolu_1",
                tool="Boom",
                input={"reason": "testing"},
            ),
        ]
        result = await executor.run(calls, CancelToken())
        assert len(result.blocks) == 1
        assert result.blocks[0].is_error is True
        assert "boom: testing" in _result_text(result.blocks[0])

    @pytest.mark.asyncio
    async def test_mix_of_success_and_error_preserves_order(self) -> None:
        executor = _make_executor(ECHO_TOOL, BOOM_TOOL)
        calls = [
            IRToolCallBlock(
                origin="model",
                call_id="toolu_1",
                tool="Echo",
                input={"message": "ok"},
            ),
            IRToolCallBlock(
                origin="model",
                call_id="toolu_2",
                tool="Boom",
                input={"reason": "fail"},
            ),
            IRToolCallBlock(
                origin="model",
                call_id="toolu_3",
                tool="Unknown",
                input={},
            ),
            IRToolCallBlock(
                origin="model",
                call_id="toolu_4",
                tool="Echo",
                input={"message": "still ok"},
            ),
        ]
        result = await executor.run(calls, CancelToken())
        assert len(result.blocks) == 4
        assert result.blocks[0].is_error is False
        assert result.blocks[1].is_error is True
        assert result.blocks[2].is_error is True
        assert result.blocks[3].is_error is False
        # Order is preserved
        assert [b.call_id for b in result.blocks] == [
            "toolu_1",
            "toolu_2",
            "toolu_3",
            "toolu_4",
        ]


class TestOneToOneInvariant:
    """zip(calls, result.blocks) is safe — the invariant holds."""

    @pytest.mark.asyncio
    async def test_one_result_per_call_always(self) -> None:
        """No matter what mix of success/error, blocks length == calls length."""
        executor = _make_executor(ECHO_TOOL, BOOM_TOOL)
        # All combinations — 10 calls, mix of success and all error types
        calls = [
            IRToolCallBlock(
                origin="model",
                call_id=f"toolu_{i}",
                tool="Echo" if i % 2 else "Boom",
                input={"message": "x"} if i % 2 else {"reason": "x"},
            )
            for i in range(10)
        ]
        result = await executor.run(calls, CancelToken())
        assert len(result.blocks) == len(calls)
        # Every result's call_id matches its corresponding call
        for call, block in zip(calls, result.blocks):
            assert call.call_id == block.call_id

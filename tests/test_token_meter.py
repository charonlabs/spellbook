"""Tests for Homunculus token metering."""

from __future__ import annotations

import pytest

from spellbook.backends.model_backend import RequestSurface
from spellbook.config import HomunculusConfig
from spellbook.homunculus.token_meter import TokenMeter
from spellbook.ir_types import (
    IRAssistantTextBlock,
    IRBlock,
    IRImageBlock,
    IRThinkingBlock,
    IRToolCallBlock,
    IRToolResultBlock,
    IRUserTextBlock,
)

pytestmark = pytest.mark.asyncio


class _FakeTokenCounter:
    def __init__(self, *, max_blocks: int | None = None) -> None:
        self.frame_calls = 0
        self.block_calls: list[list[IRBlock]] = []
        self.max_blocks = max_blocks

    async def count_block_content(self, block: IRBlock) -> int | None:
        return 10

    async def count_blocks(self, blocks: list[IRBlock]) -> int | None:
        self.block_calls.append(list(blocks))
        if self.max_blocks is not None and len(blocks) > self.max_blocks:
            return None
        if blocks and self._role(blocks[0]) != "user":
            return None

        pending: set[str] = set()
        for block in blocks:
            match block:
                case IRToolCallBlock():
                    pending.add(block.call_id)
                case IRToolResultBlock():
                    if block.call_id not in pending:
                        return None
                    pending.discard(block.call_id)
        if pending:
            return None
        return len(blocks) * 10

    async def count_frame(self) -> int | None:
        self.frame_calls += 1
        return 100

    async def count_surface(self, surface: RequestSurface) -> int | None:
        return None

    def _role(self, block: IRBlock) -> str:
        match block:
            case IRUserTextBlock() | IRImageBlock() | IRToolResultBlock():
                return "user"
            case IRAssistantTextBlock() | IRThinkingBlock() | IRToolCallBlock():
                return "assistant"


def _meter(
    counter: _FakeTokenCounter,
    *,
    slice_chunk_blocks: int = 128,
) -> TokenMeter:
    return TokenMeter(
        config=HomunculusConfig(),
        tok_counter=counter,
        slice_chunk_blocks=slice_chunk_blocks,
    )


async def test_frame_tokens_are_cached() -> None:
    counter = _FakeTokenCounter()
    meter = _meter(counter)

    assert await meter.frame_tokens() == 100
    assert await meter.frame_tokens() == 100

    assert counter.frame_calls == 1


async def test_invalidate_clears_cached_prefix_counts() -> None:
    counter = _FakeTokenCounter()
    meter = _meter(counter)

    assert await meter.frame_tokens() == 100
    meter.observe_generation_usage(
        input_end=0,
        generation_end=1,
        total_input_tokens=100,
        total_tokens=125,
    )
    assert meter.prefix_counts

    meter.invalidate()

    assert meter.prefix_counts == {}
    assert await meter.frame_tokens() == 100
    assert counter.frame_calls == 2


async def test_count_range_uses_half_open_prefix_boundaries() -> None:
    counter = _FakeTokenCounter()
    meter = _meter(counter)
    blocks: list[IRBlock] = [
        IRUserTextBlock(text="a", origin="human"),
        IRAssistantTextBlock(text="b", origin="model"),
        IRUserTextBlock(text="c", origin="human"),
    ]

    count = await meter.count_range(blocks, 1, 3)

    assert count is not None
    assert count.tokens == 20
    assert count.exact is True
    assert count.method == "prefix_delta"


async def test_count_slice_counts_only_requested_slice() -> None:
    counter = _FakeTokenCounter()
    meter = _meter(counter)
    blocks: list[IRBlock] = [
        IRUserTextBlock(text="a", origin="human"),
        IRAssistantTextBlock(text="b", origin="model"),
        IRUserTextBlock(text="c", origin="human"),
    ]

    count = await meter.count_slice(blocks, 1, 3)

    assert count is not None
    assert count.tokens == 30
    assert count.exact is False
    assert count.method == "repaired_count_blocks"
    assert counter.frame_calls == 0
    assert counter.block_calls[0] == blocks[1:3]
    assert isinstance(counter.block_calls[-1][0], IRUserTextBlock)


async def test_count_slice_direct_success_is_exact_api_count() -> None:
    counter = _FakeTokenCounter()
    meter = _meter(counter)
    blocks: list[IRBlock] = [
        IRAssistantTextBlock(text="outside", origin="model"),
        IRUserTextBlock(text="a", origin="human"),
        IRAssistantTextBlock(text="b", origin="model"),
    ]

    count = await meter.count_slice(blocks, 1, 3)

    assert count is not None
    assert count.tokens == 20
    assert count.exact is True
    assert count.method == "api"
    assert counter.frame_calls == 0
    assert counter.block_calls == [blocks[1:3]]


async def test_count_slice_chunks_when_full_slice_is_too_large() -> None:
    counter = _FakeTokenCounter(max_blocks=2)
    meter = _meter(counter, slice_chunk_blocks=2)
    blocks: list[IRBlock] = [
        IRUserTextBlock(text="a", origin="human"),
        IRAssistantTextBlock(text="b", origin="model"),
        IRUserTextBlock(text="c", origin="human"),
        IRAssistantTextBlock(text="d", origin="model"),
        IRUserTextBlock(text="e", origin="human"),
    ]

    count = await meter.count_slice(blocks, 0, 5)

    assert count is not None
    assert count.tokens == 50
    assert count.exact is False
    assert count.method == "chunked_count_blocks"
    assert counter.block_calls[0] == blocks
    assert counter.block_calls[1:] == [blocks[0:2], blocks[2:4], blocks[4:5]]


async def test_observed_generation_usage_anchors_input_boundary() -> None:
    counter = _FakeTokenCounter()
    meter = _meter(counter)
    blocks: list[IRBlock] = [
        IRUserTextBlock(text="a", origin="human"),
        IRAssistantTextBlock(text="b", origin="model"),
    ]
    meter.observe_generation_usage(
        input_end=2,
        generation_end=3,
        total_input_tokens=145,
        total_tokens=170,
    )

    count = await meter.count_range(blocks, 0, 2)

    assert count is not None
    assert count.tokens == 45
    assert count.exact is True
    assert counter.block_calls == []


async def test_observed_generation_total_is_cached_as_inexact_prefix() -> None:
    counter = _FakeTokenCounter()
    meter = _meter(counter)
    meter.observe_generation_usage(
        input_end=1,
        generation_end=2,
        total_input_tokens=120,
        total_tokens=160,
    )

    cached = meter.prefix_counts[2]

    assert cached.tokens == 160
    assert cached.exact is False
    assert cached.method == "observed_generation_total"


async def test_exact_prefix_is_not_overwritten_by_observed_generation_total() -> None:
    counter = _FakeTokenCounter()
    meter = _meter(counter)
    blocks: list[IRBlock] = [IRUserTextBlock(text="a", origin="human")]

    count = await meter.count_range(blocks, 0, 1)
    assert count is not None
    assert meter.prefix_counts[1].exact is True

    meter.observe_generation_usage(
        input_end=0,
        generation_end=1,
        total_input_tokens=100,
        total_tokens=999,
    )

    cached = meter.prefix_counts[1]
    assert cached.tokens == 110
    assert cached.exact is True
    assert cached.method == "count_blocks"


async def test_inexact_prefix_is_upgraded_when_exact_count_succeeds() -> None:
    counter = _FakeTokenCounter()
    meter = _meter(counter)
    blocks: list[IRBlock] = [IRUserTextBlock(text="a", origin="human")]
    meter.observe_generation_usage(
        input_end=0,
        generation_end=1,
        total_input_tokens=100,
        total_tokens=160,
    )

    count = await meter.count_range(blocks, 0, 1)

    assert count is not None
    assert count.tokens == 10
    assert count.exact is True
    assert meter.prefix_counts[1].tokens == 110
    assert meter.prefix_counts[1].exact is True
    assert meter.prefix_counts[1].method == "count_blocks"


async def test_dirty_prefix_boundary_is_repaired_with_stub_tool_result() -> None:
    counter = _FakeTokenCounter()
    meter = _meter(counter)
    blocks: list[IRBlock] = [
        IRUserTextBlock(text="a", origin="human"),
        IRToolCallBlock(call_id="toolu_1", tool="Bash", input={"command": "pwd"}),
    ]

    count = await meter.count_range(blocks, 0, 2)

    assert count is not None
    assert count.tokens == 30
    assert count.exact is False
    assert count.method == "prefix_delta_repaired_boundary"
    assert isinstance(counter.block_calls[-1][-1], IRToolResultBlock)


async def test_assistant_start_is_repaired_with_stub_user_message() -> None:
    counter = _FakeTokenCounter()
    meter = _meter(counter)
    blocks: list[IRBlock] = [IRAssistantTextBlock(text="hello", origin="model")]

    count = await meter.count_range(blocks, 0, 1)

    assert count is not None
    assert count.tokens == 20
    assert count.exact is False
    assert count.method == "prefix_delta_repaired_boundary"
    assert isinstance(counter.block_calls[-1][0], IRUserTextBlock)


async def test_leading_tool_result_is_repaired_with_stub_tool_use() -> None:
    counter = _FakeTokenCounter()
    meter = _meter(counter)
    blocks: list[IRBlock] = [IRToolResultBlock(call_id="toolu_1", tool="Bash")]

    count = await meter.count_range(blocks, 0, 1)

    assert count is not None
    assert count.tokens == 30
    assert count.exact is False
    assert count.method == "prefix_delta_repaired_boundary"
    assert isinstance(counter.block_calls[-1][0], IRUserTextBlock)
    assert isinstance(counter.block_calls[-1][1], IRToolCallBlock)
    assert counter.block_calls[-1][1].call_id == "toolu_1"


async def test_trailing_thinking_is_repaired_with_stub_assistant_text() -> None:
    counter = _FakeTokenCounter()
    meter = _meter(counter)
    blocks: list[IRBlock] = [IRThinkingBlock(text="thinking", signature="sig")]

    count = await meter.count_range(blocks, 0, 1)

    assert count is not None
    assert count.tokens == 30
    assert count.exact is False
    assert count.method == "prefix_delta_repaired_boundary"
    assert isinstance(counter.block_calls[-1][0], IRUserTextBlock)
    assert isinstance(counter.block_calls[-1][1], IRThinkingBlock)
    assert isinstance(counter.block_calls[-1][2], IRAssistantTextBlock)

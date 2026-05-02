from __future__ import annotations

from spellbook.backends.model_backend import TokenCounter
from spellbook.config import HomunculusConfig
from spellbook.ir_types import (
    IRAssistantTextBlock,
    IRBlock,
    IRImageBlock,
    IRThinkingBlock,
    IRTokenPrefixCount,
    IRTokenRangeCount,
    IRToolCallBlock,
    IRToolResultBlock,
    IRToolTextBlock,
    IRUserTextBlock,
    RangeCountMethod,
)


class TokenMeter:
    """Central service for token counting and count caching.

    Prefix cache semantics are half-open:
    - prefix_counts[0] is the frame-only count before block 0
    - prefix_counts[38] is the count through blocks[:38]
    - prefix_counts[83] is the count through blocks[:83]
    """

    def __init__(
        self,
        config: HomunculusConfig,
        tok_counter: TokenCounter,
        *,
        slice_chunk_blocks: int = 512,
    ):
        if slice_chunk_blocks < 1:
            raise ValueError("`slice_chunk_blocks` must be >= 1.")
        self._config = config
        self.tok_counter = tok_counter
        self._slice_chunk_blocks = slice_chunk_blocks
        self.prefix_counts: dict[int, IRTokenPrefixCount] = {}

    def invalidate(self) -> None:
        self.prefix_counts = {}

    async def frame_tokens(self) -> int | None:
        prefix = await self._count_prefix([], 0)
        if prefix is None:
            return None
        return prefix.tokens

    def observe_generation_usage(
        self,
        *,
        input_end: int,
        generation_end: int,
        total_input_tokens: int,
        total_tokens: int,
    ) -> None:
        self._store_prefix(
            input_end,
            IRTokenPrefixCount(
                tokens=total_input_tokens,
                method="observed_input",
                exact=True,
            ),
        )
        self._store_prefix(
            generation_end,
            IRTokenPrefixCount(
                tokens=total_tokens,
                method="observed_generation_total",
                exact=False,
            ),
        )

    async def count_range(
        self, blocks: list[IRBlock], start: int, end: int
    ) -> IRTokenRangeCount | None:
        if start < 0 or end < start or end > len(blocks):
            raise ValueError(
                f"Invalid token count range {start}:{end} for {len(blocks)} blocks."
            )
        if start == end:
            return IRTokenRangeCount(tokens=0, method="empty", exact=True)

        start_count = await self._count_prefix(blocks, start)
        end_count = await self._count_prefix(blocks, end)
        if start_count is None or end_count is None:
            return None

        exact = start_count.exact and end_count.exact
        method: RangeCountMethod
        if exact:
            method = "prefix_delta"
        elif (
            start_count.method == "repaired_count_blocks"
            or end_count.method == "repaired_count_blocks"
        ):
            method = "prefix_delta_repaired_boundary"
        else:
            method = "prefix_delta_approximate"

        return IRTokenRangeCount(
            tokens=max(0, end_count.tokens - start_count.tokens),
            method=method,
            exact=exact,
        )

    async def count_slice(
        self, blocks: list[IRBlock], start: int, end: int
    ) -> IRTokenRangeCount | None:
        """Count one block slice directly.

        This is the right API for semantic block metrics: it does not count
        full prefixes, does not touch prefix caches, and avoids asking the
        provider to count an ever-growing historical transcript.
        """
        if start < 0 or end < start or end > len(blocks):
            raise ValueError(
                f"Invalid token count slice {start}:{end} for {len(blocks)} blocks."
            )
        if start == end:
            return IRTokenRangeCount(tokens=0, method="empty", exact=True)

        return await self._count_block_slice(list(blocks[start:end]))

    async def _count_prefix(
        self, blocks: list[IRBlock], end: int
    ) -> IRTokenPrefixCount | None:
        cached = self.prefix_counts.get(end)
        if cached is not None and cached.exact:
            return cached

        if end == 0:
            frame_count = await self.tok_counter.count_frame()
            if frame_count is None:
                return cached
            count = IRTokenPrefixCount(tokens=frame_count, method="frame", exact=True)
            self._store_prefix(0, count)
            return count

        frame_count = await self._count_prefix(blocks, 0)
        if frame_count is None:
            return cached

        prefix_blocks = list(blocks[:end])
        message_count = await self.tok_counter.count_blocks(prefix_blocks)
        if message_count is not None:
            count = IRTokenPrefixCount(
                tokens=frame_count.tokens + message_count,
                method="count_blocks",
                exact=True,
            )
            self._store_prefix(end, count)
            return count

        repaired_blocks = self._repair_prefix_for_count(prefix_blocks)
        if repaired_blocks != prefix_blocks:
            repaired_count = await self.tok_counter.count_blocks(repaired_blocks)
            if repaired_count is not None:
                count = IRTokenPrefixCount(
                    tokens=frame_count.tokens + repaired_count,
                    method="repaired_count_blocks",
                    exact=False,
                )
                self._store_prefix(end, count)
                return count

        return cached

    async def _count_block_slice(
        self,
        blocks: list[IRBlock],
    ) -> IRTokenRangeCount | None:
        count = await self.tok_counter.count_blocks(blocks)
        if count is not None:
            return IRTokenRangeCount(tokens=count, method="api", exact=True)

        repaired_blocks = self._repair_prefix_for_count(blocks)
        if repaired_blocks != blocks:
            repaired_count = await self.tok_counter.count_blocks(repaired_blocks)
            if repaired_count is not None:
                return IRTokenRangeCount(
                    tokens=repaired_count,
                    method="repaired_count_blocks",
                    exact=False,
                )

        return await self._count_block_slice_in_chunks(blocks)

    async def _count_block_slice_in_chunks(
        self,
        blocks: list[IRBlock],
    ) -> IRTokenRangeCount | None:
        if len(blocks) <= self._slice_chunk_blocks:
            return None

        total = 0
        for start in range(0, len(blocks), self._slice_chunk_blocks):
            chunk = blocks[start : start + self._slice_chunk_blocks]
            count = await self.tok_counter.count_blocks(chunk)
            if count is None:
                repaired_chunk = self._repair_prefix_for_count(chunk)
                if repaired_chunk == chunk:
                    return None
                count = await self.tok_counter.count_blocks(repaired_chunk)
                if count is None:
                    return None
            total += count

        return IRTokenRangeCount(
            tokens=total,
            method="chunked_count_blocks",
            exact=False,
        )

    def _store_prefix(self, end: int, count: IRTokenPrefixCount) -> None:
        existing = self.prefix_counts.get(end)
        if existing is not None and existing.exact and not count.exact:
            return
        self.prefix_counts[end] = count

    def _repair_prefix_for_count(self, blocks: list[IRBlock]) -> list[IRBlock]:
        if not blocks:
            return blocks

        repaired: list[IRBlock] = []
        pending: dict[str, str] = {}

        for block in blocks:
            match block:
                case IRToolResultBlock():
                    if block.call_id not in pending:
                        if not repaired:
                            repaired.append(IRUserTextBlock(text=".", origin="human"))
                        repaired.append(
                            IRToolCallBlock(
                                call_id=block.call_id,
                                tool=block.tool,
                                input={},
                            )
                        )
                    else:
                        pending.pop(block.call_id)
                    repaired.append(block)
                case IRToolCallBlock():
                    if not repaired:
                        repaired.append(IRUserTextBlock(text=".", origin="human"))
                    pending[block.call_id] = block.tool
                    repaired.append(block)
                case IRAssistantTextBlock() | IRThinkingBlock():
                    if not repaired:
                        repaired.append(IRUserTextBlock(text=".", origin="human"))
                    repaired.append(block)
                case IRUserTextBlock() | IRImageBlock():
                    repaired.append(block)

        if not pending:
            if isinstance(repaired[-1], IRThinkingBlock):
                repaired.append(IRAssistantTextBlock(text=".", origin="model"))
            return repaired

        for call_id, tool in pending.items():
            repaired.append(
                IRToolResultBlock(
                    call_id=call_id,
                    tool=tool,
                    content=[IRToolTextBlock(text=".")],
                )
            )
        return repaired

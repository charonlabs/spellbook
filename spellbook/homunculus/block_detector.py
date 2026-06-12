"""Block detector state and prompt rendering for fork-backed semantic grouping.

This module is the Homunculus-side coordinator for semantic block detection.

The important state model is:

- `_accumulated` is the canonical detector-visible source slice collected so far
- `_accumulated_start_block_id` is the global block id of `_accumulated[0]`
- `_semantic_buffer` holds still-buffered semantic blocks returned by the latest
  detector fork
- `completed_blocks` holds semantic blocks that are finalized
- `_context_buffer` is derived state: the remaining ungrouped suffix of
  `_accumulated`
- `_start_block_id` is the global block id of `_context_buffer[0]`

This distinction is load-bearing: the detector should mutate semantic ranges and
recompute the remaining buffer, not treat the buffer itself as canonical state.

`build_inbound_block()` renders the detector prompt surface as structured
XML-ish text. That rendering is a protocol surface, not a throwaway string.
If you change it:

- preserve parseability and section structure
- keep block ids and semantic ranges explicit
- escape user/model content before embedding it
- keep completed, buffered, and raw-context sections conceptually distinct

The detector currently relies on global block ids threaded in from Homunculus so
that semantic ranges remain stable across batches.
"""

from dataclasses import dataclass
from html import escape
from typing import Sequence

from spellbook.config import HomunculusConfig
from spellbook.fork import (
    BlockDetectorConfig,
    BlockDetectorResult,
    ForkRunner,
    PreparedFork,
)
from spellbook.homunculus.common import render_context_block
from spellbook.ir_types import (
    IRBlock,
    IRSemanticBlockRange,
    IRUserTextBlock,
)
from spellbook.recorder import Recorder
from spellbook.rehydrator import RehydrationResult


@dataclass(frozen=True, slots=True)
class BlockDetectorIntegration:
    result: BlockDetectorResult
    completed: list[IRSemanticBlockRange]
    deferred_completed: list[IRSemanticBlockRange]
    discarded_ranges: list[IRSemanticBlockRange]

    @property
    def partial(self) -> bool:
        return bool(self.deferred_completed or self.discarded_ranges)


class BlockDetector:
    def __init__(
        self,
        *,
        config: HomunculusConfig,
        fork_runner: ForkRunner,
        recorder: Recorder,
    ):
        self._config = config
        self._detect_interval = config.detect_interval
        self._fork_runner = fork_runner
        self._recorder = recorder
        self._counter = 0
        self.completed_blocks: list[IRSemanticBlockRange] = []
        self._accumulated: list[IRBlock] = []
        self._accumulated_start_block_id: int | None = None
        self._context_buffer: list[IRBlock] = []
        self._semantic_buffer: list[IRSemanticBlockRange] = []
        self._start_block_id: int = 0

    def rehydrate(self, rehydrated: RehydrationResult) -> None:
        self.completed_blocks = list(rehydrated.completed_semantic_block_ranges)
        self._semantic_buffer = list(rehydrated.buffered_semantic_block_ranges)
        self._accumulated = list(rehydrated.blocks)
        self._accumulated_start_block_id = 0 if self._accumulated else None
        self._counter = len(self._accumulated) % self._detect_interval
        self.build_context_buffer()

    @property
    def buffered_blocks(self) -> list[IRSemanticBlockRange]:
        return list(self._semantic_buffer)

    @property
    def has_pending_blocks(self) -> bool:
        self.build_context_buffer()
        return bool(self._semantic_buffer or self._context_buffer)

    async def maybe_detect(
        self, blocks: Sequence[IRBlock], first_block_id: int
    ) -> PreparedFork | None:
        if not blocks:
            return None

        if self._accumulated_start_block_id is None:
            self._accumulated_start_block_id = first_block_id

        self._accumulated.extend(blocks)
        self._counter += len(blocks)
        if self._counter >= self._detect_interval:
            self.build_context_buffer()
            self._counter -= self._detect_interval
            return await self._run_detection_fork(finalize=False)
        return None

    async def force_detect(self, *, finalize: bool = False) -> PreparedFork | None:
        """Run detection immediately when buffered/raw context remains."""

        self.build_context_buffer()
        if not self.has_pending_blocks:
            return None
        return await self._run_detection_fork(finalize=finalize)

    async def _run_detection_fork(self, *, finalize: bool) -> PreparedFork:
        fc = BlockDetectorConfig(
            prev_semantic_blocks=list(self.completed_blocks),
            full_context_blocks=list(self._accumulated),
            context_block_buffer=list(self._context_buffer),
            context_block_start_id=self._start_block_id,
            semantic_block_buffer=list(self._semantic_buffer),
            inbound_block=self.build_inbound_block(finalize=finalize),
        )
        return await self._fork_runner.run_fork(fork_config=fc)

    def integrate_result(
        self, result: BlockDetectorResult, fork_id: str
    ) -> list[IRSemanticBlockRange]:
        integration = self.simulate_result(result)
        return self.integrate_prepared_result(integration, fork_id)

    def simulate_result(self, result: BlockDetectorResult) -> BlockDetectorIntegration:
        completed = self._sort_ranges(result.completed)
        still_buffered = self._sort_ranges(result.still_buffered)
        self._validate_ranges_in_known_context([*completed, *still_buffered])

        accepted, deferred = self._split_completed_prefix(completed)
        buffer_start = self._next_block_after(accepted)
        safe_buffer, discarded = self._split_safe_buffer(
            [*still_buffered, *deferred],
            start_block=buffer_start,
        )

        if result.completed or result.still_buffered:
            if not accepted and not safe_buffer:
                raise ValueError(
                    "Detector result did not contain any contiguous semantic range "
                    "that could be integrated safely."
                )

        sanitized = result.model_copy(
            update={
                "completed": accepted,
                "still_buffered": safe_buffer,
            }
        )
        return BlockDetectorIntegration(
            result=sanitized,
            completed=accepted,
            deferred_completed=deferred,
            discarded_ranges=discarded,
        )

    def integrate_prepared_result(
        self, integration: BlockDetectorIntegration, fork_id: str
    ) -> list[IRSemanticBlockRange]:
        self._fork_runner.integrate_result(fork_id)
        self._recorder.detect_blocks(integration.result)
        self.completed_blocks.extend(integration.completed)
        self._semantic_buffer = integration.result.still_buffered
        self.build_context_buffer()
        return integration.completed

    def discard_result(self, fork_id: str) -> None:
        self._fork_runner.integrate_result(fork_id)

    def _sort_ranges(
        self, ranges: Sequence[IRSemanticBlockRange]
    ) -> list[IRSemanticBlockRange]:
        return sorted(ranges, key=lambda r: (r.start_block, r.end_block))

    def _next_expected_completed_start(self) -> int:
        if self.completed_blocks:
            return self.completed_blocks[-1].end_block + 1
        if self._accumulated_start_block_id is not None:
            return self._accumulated_start_block_id
        return 0

    def _next_block_after(self, ranges: Sequence[IRSemanticBlockRange]) -> int:
        if ranges:
            return ranges[-1].end_block + 1
        return self._next_expected_completed_start()

    def _validate_ranges_in_known_context(
        self, ranges: Sequence[IRSemanticBlockRange]
    ) -> None:
        if not ranges:
            return
        if self._accumulated_start_block_id is None:
            raise ValueError("Detector result returned ranges with no known context.")

        first = self._accumulated_start_block_id
        last = first + len(self._accumulated) - 1
        for block in ranges:
            if block.start_block < first or block.end_block > last:
                raise ValueError(
                    "Detector result returned a semantic range outside the known "
                    f"context: {block.start_block}-{block.end_block}, expected "
                    f"{first}-{last}."
                )

    def _split_completed_prefix(
        self, completed: Sequence[IRSemanticBlockRange]
    ) -> tuple[list[IRSemanticBlockRange], list[IRSemanticBlockRange]]:
        accepted: list[IRSemanticBlockRange] = []
        expected_start = self._next_expected_completed_start()

        for idx, block in enumerate(completed):
            if block.start_block < expected_start:
                raise ValueError(
                    "Detector result returned overlapping completed ranges. "
                    f'Range "{block.title}" starts at {block.start_block}, '
                    f"expected at least {expected_start}."
                )
            if block.start_block != expected_start:
                return accepted, list(completed[idx:])
            accepted.append(block)
            expected_start = block.end_block + 1
        return accepted, []

    def _split_safe_buffer(
        self,
        ranges: Sequence[IRSemanticBlockRange],
        *,
        start_block: int,
    ) -> tuple[list[IRSemanticBlockRange], list[IRSemanticBlockRange]]:
        safe: list[IRSemanticBlockRange] = []
        discarded: list[IRSemanticBlockRange] = []
        expected_start = start_block

        for block in self._sort_ranges(ranges):
            if block.start_block < expected_start:
                raise ValueError(
                    "Detector result returned overlapping buffered ranges. "
                    f'Range "{block.title}" starts at {block.start_block}, '
                    f"expected at least {expected_start}."
                )
            if block.start_block != expected_start:
                discarded.append(block)
                continue
            safe.append(block)
            expected_start = block.end_block + 1
        return safe, discarded

    def build_inbound_block(self, *, finalize: bool = False) -> IRUserTextBlock:
        payload = "\n".join(
            [
                "<block_detector_context>",
                self._render_instructions(finalize=finalize),
                self._render_completed_semantic_blocks(),
                self._render_buffered_semantic_blocks(),
                self._render_context_block_buffer(),
                "</block_detector_context>",
            ]
        )
        return IRUserTextBlock(text=payload, origin="system")

    def _render_instructions(self, *, finalize: bool) -> str:
        instructions = """<instructions>
Completed semantic blocks are finalized history.
Buffered semantic blocks are draft groupings from the current detection window and may be amended or completed.
Context block buffer contains the remaining ungrouped context blocks.
Use block ids and ranges exactly as rendered.
"""
        if finalize:
            instructions += """This is an EOF finalization pass for an offline replay artifact.
No future context blocks will arrive from this source transcript.
Complete any stable buffered semantic blocks that can now be finalized.
If context block buffer still contains raw context, propose final semantic block(s) that cover it.
Blocks proposed or amended in this detector session still cannot be completed until a later finalization pass.
"""
        return instructions + "</instructions>"

    def _render_completed_semantic_blocks(self) -> str:
        if not self.completed_blocks:
            return "<completed_semantic_blocks />"
        rendered = ["<completed_semantic_blocks>"]
        for block in self.completed_blocks:
            rendered.append(
                f'<completed_semantic_block title="{escape(block.title)}" range="{block.start_block}-{block.end_block}" />'
            )
        rendered.append("</completed_semantic_blocks>")
        return "\n".join(rendered)

    def _render_buffered_semantic_blocks(self) -> str:
        if not self._semantic_buffer:
            return "<buffered_semantic_blocks />"
        rendered = ["<buffered_semantic_blocks>"]
        for block in self._semantic_buffer:
            rendered.append(
                f'<buffered_semantic_block title="{escape(block.title)}" range="{block.start_block}-{block.end_block}">'
            )
            for block_id, ctx_block in self._iter_blocks_for_range(
                block.start_block, block.end_block
            ):
                rendered.append(render_context_block(ctx_block, block_id))
            rendered.append("</buffered_semantic_block>")
        rendered.append("</buffered_semantic_blocks>")
        return "\n".join(rendered)

    def _render_context_block_buffer(self) -> str:
        if not self._context_buffer:
            return "<context_block_buffer />"
        rendered = ["<context_block_buffer>"]
        for i, block in enumerate(self._context_buffer):
            block_id = self._start_block_id + i
            rendered.append(render_context_block(block, block_id))
        rendered.append("</context_block_buffer>")
        return "\n".join(rendered)

    def _iter_blocks_for_range(
        self, start_block: int, end_block: int
    ) -> list[tuple[int, IRBlock]]:
        if self._accumulated_start_block_id is None:
            return []
        start_offset = start_block - self._accumulated_start_block_id
        end_offset = end_block - self._accumulated_start_block_id
        if start_offset < 0 or end_offset < start_offset:
            return []
        sliced = self._accumulated[start_offset : end_offset + 1]
        return [(start_block + i, block) for i, block in enumerate(sliced)]

    def build_context_buffer(self) -> None:
        if self._semantic_buffer:
            self._start_block_id = self._semantic_buffer[-1].end_block + 1
        elif self.completed_blocks:
            self._start_block_id = self.completed_blocks[-1].end_block + 1
        elif self._accumulated_start_block_id is not None:
            self._start_block_id = self._accumulated_start_block_id
        else:
            self._start_block_id = 0

        if self._accumulated_start_block_id is None:
            self._context_buffer = []
            return

        accumulated_offset = self._start_block_id - self._accumulated_start_block_id
        if accumulated_offset < 0:
            accumulated_offset = 0
        if accumulated_offset >= len(self._accumulated):
            self._context_buffer = []
            return

        self._context_buffer = self._accumulated[accumulated_offset:]

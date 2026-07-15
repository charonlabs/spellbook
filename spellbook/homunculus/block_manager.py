import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from spellbook.config import HomunculusConfig
from spellbook.footer import FooterController
from spellbook.fork import (
    BlockDetectorResult,
    BlockSummarizerResult,
    ForkRunner,
    PreparedFork,
)
from spellbook.homunculus.block_detector import BlockDetector, BlockDetectorIntegration
from spellbook.homunculus.block_summarizer import BlockSummarizer
from spellbook.homunculus.common import render_context_block, render_summary
from spellbook.homunculus.token_meter import TokenMeter
from spellbook.ir_types import (
    IRAssistantTextBlock,
    IRBlock,
    IRImageBlock,
    IRRefusalBlock,
    IRSemanticBlock,
    IRSemanticBlockFacet,
    IRSemanticBlockPairNarrative,
    IRSemanticBlockPairNarrativeChild,
    IRSemanticBlockPin,
    IRSemanticBlockRange,
    IRSemanticBlockSummary,
    IRThinkingBlock,
    IRTokenRangeCount,
    IRToolCallBlock,
    IRToolResultBlock,
    IRUsage,
    IRUserTextBlock,
    SemanticBlockApplyModeSource,
    SemanticBlockMode,
)
from spellbook.nursery import Nursery, NurseryJob, NurseryJobResult
from spellbook.recorder import Recorder
from spellbook.rehydrator import RehydrationResult
from spellbook.runtime_breadcrumb import spellbook_runtime_breadcrumb

if TYPE_CHECKING:
    from spellbook.debug_visibility import DebugNoticeLevel, DebugEmitter

logger = logging.getLogger(__name__)

ForgetBlockStatus = Literal[
    "compacted",
    "boundary_invalid",
    "summary_queued",
    "summary_in_flight",
    "summary_unavailable",
]

_DetectionIntegrationOutcome = Literal[
    "accepted",
    "partially_deferred",
    "discarded",
    "boundary_invalid",
]


@dataclass(frozen=True, slots=True)
class ForgetBlockResult:
    status: ForgetBlockStatus
    message: str

    @property
    def compacted(self) -> bool:
        return self.status == "compacted"


@dataclass(frozen=True, slots=True)
class _ToolClosureEnvelope:
    start_block: int
    end_block: int
    call_ids: tuple[str, ...]
    complete: bool


@dataclass(frozen=True, slots=True)
class _ToolClosureViolation:
    block: IRSemanticBlock
    envelope: _ToolClosureEnvelope


class BlockManager:
    def __init__(
        self,
        *,
        config: HomunculusConfig,
        fork_runner: ForkRunner,
        footer_c: FooterController,
        nursery: Nursery,
        recorder: Recorder,
        token_meter: TokenMeter,
        context_projector: Callable[[Sequence[IRBlock]], list[IRBlock]] | None = None,
        enable_block_metrics: bool = True,
        enable_block_detection: bool = True,
        debug_emitter: "DebugEmitter | None" = None,
    ):
        """Manager for semantic and context blocks. Public instance vars can be
        read and mutated in the Homunculus itself. At present, `context_blocks` and
        `next_block_id` are both directly mutated externally."""
        self._context_projector = context_projector or _identity_context_projector
        self._enable_block_metrics = enable_block_metrics
        self._enable_block_detection = enable_block_detection
        self._detector = BlockDetector(
            config=config,
            fork_runner=fork_runner,
            recorder=recorder,
            context_projector=self._context_projector,
        )
        self._fork_runner = fork_runner
        self._meter = token_meter
        self._recorder = recorder
        transcript_path = getattr(recorder, "transcript_path", None)
        self._detector_forensics_path = (
            transcript_path.with_name("detector_forensics.jsonl")
            if isinstance(transcript_path, Path)
            else None
        )
        self._debug = debug_emitter
        self._footer_c = footer_c
        self._nursery = nursery
        self._summarizer = BlockSummarizer(
            config=config, fork_runner=fork_runner, recorder=recorder
        )
        self.semantic_blocks: list[IRSemanticBlock] = []  # This should stay ordered
        self.context_blocks: list[IRBlock] = []
        self.next_block_id = 0
        self._accept_background_work = True

    @property
    def proposed_semantic_blocks(self) -> list[IRSemanticBlockRange]:
        return self._detector.buffered_blocks

    @property
    def has_unfinalized_detection(self) -> bool:
        return self._detector.has_pending_blocks

    def rehydrate(self, rehydrated: RehydrationResult) -> None:
        self._detector.rehydrate(rehydrated)
        self.semantic_blocks = rehydrated.semantic_blocks
        self._validate_semantic_blocks(validate_tool_closure=False)

    async def append_context_blocks(
        self, blocks: Sequence[IRBlock], *, usage: IRUsage | None = None
    ) -> int:
        """Extend context_blocks and run block detection.
        Returns the start_id of the appended batch."""
        batch_start_id = self.next_block_id
        self.context_blocks.extend(blocks)
        self.next_block_id += len(blocks)
        if usage:
            self._meter.observe_generation_usage(
                input_end=batch_start_id,
                generation_end=self.next_block_id,
                total_input_tokens=usage.total_input_tokens,
                total_tokens=usage.total_tokens,
            )
        await self.maybe_detect(blocks, batch_start_id)
        return batch_start_id

    def render_block(
        self,
        *,
        semantic_block: IRSemanticBlock | None = None,
        id: str | None = None,
        idx: int | None = None,
    ) -> list[IRBlock]:
        """Render a semantic block given the block itself, block id, or block idx."""
        if semantic_block is None and id is None and idx is None:
            raise ValueError(
                "`render_block` needs one of its params to work! They can't all be None!"
            )
        if semantic_block is not None:
            block = semantic_block
        elif idx is not None:
            if not -1 < idx < len(self.semantic_blocks):
                raise ValueError(
                    f"Can't render block at idx={idx} when there's only {len(self.semantic_blocks)} blocks!"
                )
            block = self.semantic_blocks[idx]
        elif id is not None:
            maybe_block = next((b for b in self.semantic_blocks if b.id == id), None)
            if maybe_block is None:
                raise ValueError(f"`{id} is not a valid block id.")
            block = maybe_block

        match block.mode:
            case "summary":
                # ASSUMES THE ARTIFACTS ARE THERE AND VALID - VALIDATION OCCURS ON MODE FLIP, NOT RENDER
                if block.facet_pins:
                    return self._render_summary_with_pinned_facets(block)
                return [render_summary(block)]
            case "full":
                return self._context_blocks_in_block(block)
            case "pair_narrative":
                return self._render_pair_narrative(block)
            case _:
                raise NotImplementedError(f"Mode `{block.mode} not yet supported.")

    def render_summary_preview(self, idx: int) -> str:
        block = self._get_block_by_idx(idx)
        if not any(artifact.type == "summary" for artifact in block.artifacts):
            raise ValueError(f"Block {idx} has no summary artifact yet.")

        preview_block = block.model_copy(update={"mode": "summary"})
        rendered = self.render_block(semantic_block=preview_block)
        parts: list[str] = [
            f'# Block {block.idx}: "{block.title}"',
            "",
            f"Range: {block.range.start_block}-{block.range.end_block}.",
            "Preview mode: summary.",
            "",
        ]

        if len(rendered) > 1:
            parts.append(
                f"This preview renders as {len(rendered)} content blocks "
                "because pinned facets are preserved as original conversation."
            )
            for rendered_idx, rendered_block in enumerate(rendered, start=1):
                parts.extend(
                    [
                        "",
                        f"## Rendered Block {rendered_idx}",
                        "",
                        self._render_preview_block(rendered_block),
                    ]
                )
        else:
            parts.append(self._render_preview_block(rendered[0]))
        return "\n".join(parts)

    def _render_preview_block(self, block: IRBlock) -> str:
        if isinstance(block, IRUserTextBlock) and block.origin == "memory":
            return block.text
        return render_context_block(block)

    def _context_blocks_in_block(self, block: IRSemanticBlock) -> list[IRBlock]:
        return self.context_blocks[block.range.start_block : block.range.end_block + 1]

    def _render_summary_with_pinned_facets(
        self, block: IRSemanticBlock
    ) -> list[IRBlock]:
        artifact = self._summary_artifact(block)
        pinned_facets = self._pinned_facets(block, artifact)
        if not pinned_facets:
            return [render_summary(block)]

        intervals = self._expanded_pinned_facet_intervals(block, pinned_facets)
        if not intervals:
            return [render_summary(block)]

        rendered: list[IRBlock] = [
            self._render_pinned_summary_opening(block, artifact, pinned_facets)
        ]
        for start, end in intervals:
            rendered.extend(self.context_blocks[start : end + 1])
        rendered.append(self._render_pinned_summary_closing(block, artifact))
        return rendered

    def _summary_artifact(self, block: IRSemanticBlock) -> IRSemanticBlockSummary:
        artifact = next(
            (a for a in block.artifacts if isinstance(a, IRSemanticBlockSummary)),
            None,
        )
        if artifact is None:
            raise ValueError(f"Block {block.idx} has no summary artifact.")
        return artifact

    def _pair_narrative_artifact(
        self, block: IRSemanticBlock
    ) -> IRSemanticBlockPairNarrative | IRSemanticBlockPairNarrativeChild:
        artifact = next(
            (a for a in reversed(block.artifacts) if a.mode == "pair_narrative"),
            None,
        )
        if artifact is None:
            raise ValueError(f"Block {block.idx} has no pair narrative artifact.")
        if not isinstance(
            artifact, IRSemanticBlockPairNarrative | IRSemanticBlockPairNarrativeChild
        ):
            raise ValueError(
                f"Block {block.idx} has an invalid pair narrative artifact."
            )
        return artifact

    def _render_pair_narrative(self, block: IRSemanticBlock) -> list[IRBlock]:
        artifact = self._pair_narrative_artifact(block)
        self._validate_pair_narrative_render(block, artifact)
        if isinstance(artifact, IRSemanticBlockPairNarrativeChild):
            return []
        return artifact.blocks

    def _validate_pair_narrative_render(
        self,
        block: IRSemanticBlock,
        artifact: IRSemanticBlockPairNarrative | IRSemanticBlockPairNarrativeChild,
    ) -> None:
        expected_idx = (
            artifact.pair[0]
            if isinstance(artifact, IRSemanticBlockPairNarrative)
            else artifact.pair[1]
        )
        if block.idx != expected_idx:
            raise ValueError(
                f"Block {block.idx} has a pair narrative artifact for pair "
                f"{artifact.pair}."
            )

        other_idx = (
            artifact.pair[1] if block.idx == artifact.pair[0] else artifact.pair[0]
        )
        if not -1 < other_idx < len(self.semantic_blocks):
            raise ValueError(
                f"Block {block.idx} pair narrative references missing block {other_idx}."
            )
        other = self.semantic_blocks[other_idx]
        if other.mode != "pair_narrative":
            raise ValueError(
                f"Block {block.idx} pair narrative is active but block {other_idx} "
                f"is in mode {other.mode}."
            )
        other_artifact = self._pair_narrative_artifact(other)
        if other_artifact.narrative_id != artifact.narrative_id:
            raise ValueError(
                f"Block {block.idx} pair narrative does not match block {other_idx}."
            )
        if isinstance(artifact, IRSemanticBlockPairNarrativeChild):
            if artifact.parent_block_id != other.id:
                raise ValueError(
                    f"Block {block.idx} pair narrative child references a different parent."
                )
        elif not isinstance(other_artifact, IRSemanticBlockPairNarrativeChild):
            raise ValueError(
                f"Block {block.idx} pair narrative parent has no child artifact."
            )

    def _pinned_facets(
        self, block: IRSemanticBlock, artifact: IRSemanticBlockSummary
    ) -> list[IRSemanticBlockFacet]:
        pinned_ids = {pin.facet_id for pin in block.facet_pins}
        return [facet for facet in artifact.facets if facet.id in pinned_ids]

    def _render_pinned_summary_opening(
        self,
        block: IRSemanticBlock,
        artifact: IRSemanticBlockSummary,
        pinned_facets: list[IRSemanticBlockFacet],
    ) -> IRUserTextBlock:
        pinned_ids = {facet.id for facet in pinned_facets}
        parts: list[str] = [
            f'<spellbook-memory block_idx="{block.idx}" mode="summary" turns="{block.range.start_block}-{block.range.end_block}">',
            f"# {artifact.headline}",
            "",
            artifact.text,
        ]

        unpinned_facets = [
            facet for facet in artifact.facets if facet.id not in pinned_ids
        ]
        if unpinned_facets:
            parts.append("")
            parts.append("## Facets")
            for facet in unpinned_facets:
                parts.append(
                    f"- {facet.title} (blocks {facet.start_block}-{facet.end_block})"
                )
                parts.append(f"  {facet.description}")
                if facet.resources:
                    parts.append(f"  Resources: {'; '.join(facet.resources)}")

        pinned_names = ", ".join(f'"{facet.title}"' for facet in pinned_facets)
        parts.append("")
        parts.append(f"Pinned facets follow as original conversation: {pinned_names}")
        parts.append("</spellbook-memory>")
        return IRUserTextBlock(text="\n".join(parts), origin="memory")

    def _render_pinned_summary_closing(
        self,
        block: IRSemanticBlock,
        artifact: IRSemanticBlockSummary,
    ) -> IRUserTextBlock:
        parts: list[str] = [
            f'<spellbook-memory block_idx="{block.idx}" mode="summary" continues="true">',
            "End of pinned facets.",
        ]
        if artifact.open_thread:
            parts.append("")
            parts.append(f"Open thread: {artifact.open_thread}")
        parts.append("</spellbook-memory>")
        return IRUserTextBlock(text="\n".join(parts), origin="memory")

    def _expanded_pinned_facet_intervals(
        self,
        block: IRSemanticBlock,
        facets: list[IRSemanticBlockFacet],
    ) -> list[tuple[int, int]]:
        intervals: list[tuple[int, int]] = []
        for facet in facets:
            start = max(facet.start_block, block.range.start_block)
            end = min(facet.end_block, block.range.end_block)
            if start > end:
                continue
            expanded = self._expand_interval_to_valid_sequence(
                start,
                end,
                min_start=block.range.start_block,
                max_end=block.range.end_block,
            )
            if expanded is None:
                return []
            intervals.append(expanded)
        return self._merge_intervals(intervals)

    def _expand_interval_to_valid_sequence(
        self,
        start: int,
        end: int,
        *,
        min_start: int,
        max_end: int,
    ) -> tuple[int, int] | None:
        while True:
            previous = (start, end)
            start, end = self._expand_same_role_edges(start, end, min_start, max_end)
            expanded = self._expand_tool_pairs(start, end, min_start, max_end)
            if expanded is None:
                return None
            start, end = expanded
            if (start, end) == previous:
                return start, end

    def _expand_same_role_edges(
        self,
        start: int,
        end: int,
        min_start: int,
        max_end: int,
    ) -> tuple[int, int]:
        start_role = self._provider_role(self.context_blocks[start])
        while (
            start > min_start
            and self._provider_role(self.context_blocks[start - 1]) == start_role
        ):
            start -= 1

        end_role = self._provider_role(self.context_blocks[end])
        while (
            end < max_end
            and self._provider_role(self.context_blocks[end + 1]) == end_role
        ):
            end += 1
        return start, end

    def _expand_tool_pairs(
        self,
        start: int,
        end: int,
        min_start: int,
        max_end: int,
    ) -> tuple[int, int] | None:
        new_start = start
        new_end = end
        for idx in range(start, end + 1):
            block = self.context_blocks[idx]
            match block:
                case IRToolResultBlock():
                    call_idx = self._find_tool_call(
                        block.call_id, min_start=min_start, max_end=max_end
                    )
                    if call_idx is None:
                        return None
                    new_start = min(new_start, call_idx)
                    new_end = max(new_end, call_idx)
                case IRToolCallBlock():
                    result_idx = self._find_tool_result(
                        block.call_id, min_start=min_start, max_end=max_end
                    )
                    if result_idx is None:
                        return None
                    new_start = min(new_start, result_idx)
                    new_end = max(new_end, result_idx)
                case _:
                    pass
        return new_start, new_end

    def _find_tool_call(
        self,
        call_id: str,
        *,
        min_start: int,
        max_end: int,
    ) -> int | None:
        for idx in range(min_start, max_end + 1):
            block = self.context_blocks[idx]
            if isinstance(block, IRToolCallBlock) and block.call_id == call_id:
                return idx
        return None

    def _find_tool_result(
        self,
        call_id: str,
        *,
        min_start: int,
        max_end: int,
    ) -> int | None:
        for idx in range(min_start, max_end + 1):
            block = self.context_blocks[idx]
            if isinstance(block, IRToolResultBlock) and block.call_id == call_id:
                return idx
        return None

    def _merge_intervals(
        self, intervals: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        if not intervals:
            return []
        ordered = sorted(intervals)
        merged = [ordered[0]]
        for start, end in ordered[1:]:
            prev_start, prev_end = merged[-1]
            if start <= prev_end + 1:
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))
        return merged

    def _provider_role(self, block: IRBlock) -> str:
        match block:
            case IRUserTextBlock() | IRImageBlock() | IRToolResultBlock():
                return "user"
            case (
                IRAssistantTextBlock()
                | IRRefusalBlock()
                | IRThinkingBlock()
                | IRToolCallBlock()
            ):
                return "assistant"
            case _:
                raise TypeError(f"Unsupported IR block type: {type(block)}")

    def render_tail(self) -> list[IRBlock]:
        if not self.semantic_blocks:
            return list(self.context_blocks)
        return self.context_blocks[self.semantic_blocks[-1].range.end_block + 1 :]

    async def _integrate_detection(
        self, result: BlockDetectorResult, fork_id: str
    ) -> None:
        try:
            integration = self._detector.simulate_result(result)
        except ValueError as exc:
            self._write_detection_forensics(
                fork_id=fork_id,
                outcome="discarded",
                raw=result,
                sanitized=None,
                error=exc,
            )
            self._discard_detection_result(fork_id=fork_id, result=result, error=exc)
            return

        self._log_detection_range_sanitization(
            fork_id=fork_id,
            raw=result,
            sanitized=integration,
        )
        try:
            new_blocks = self._semantic_blocks_for_completed(integration.completed)
            candidate_blocks = [*self.semantic_blocks, *new_blocks]
            self._validate_semantic_blocks(
                candidate_blocks,
                validate_tool_closure=True,
            )
        except ValueError as exc:
            self._write_detection_forensics(
                fork_id=fork_id,
                outcome="boundary_invalid",
                raw=result,
                sanitized=integration,
                error=exc,
            )
            self._discard_detection_result(fork_id=fork_id, result=result, error=exc)
            return

        self._write_detection_forensics(
            fork_id=fork_id,
            outcome="partially_deferred" if integration.partial else "accepted",
            raw=result,
            sanitized=integration,
            error=None,
        )

        self._debug_event(
            subsystem="block_detector",
            event="boundaries_proposed",
            title="Block detector returned boundaries",
            metadata={
                "fork_id": fork_id,
                "completed": len(result.completed),
                "still_buffered": len(result.still_buffered),
                "accepted": len(integration.completed),
                "deferred": len(integration.deferred_completed),
                "discarded": len(integration.discarded_ranges),
            },
        )

        if integration.partial:
            logger.warning(
                "block_detector.result_partially_deferred fork_id=%s "
                "accepted=%s deferred=%s discarded=%s",
                fork_id,
                len(integration.completed),
                len(integration.deferred_completed),
                len(integration.discarded_ranges),
            )
            self._alert_detection_failure(
                fork_id=fork_id,
                event="result_partially_deferred",
                title="Block detector result was partially deferred",
                level="warning",
                metadata={
                    "completed": len(result.completed),
                    "still_buffered": len(result.still_buffered),
                    "accepted": len(integration.completed),
                    "deferred": len(integration.deferred_completed),
                    "discarded": len(integration.discarded_ranges),
                },
            )
            self._footer_c.queue_footer(
                text="a detection result failed validation and was partially deferred",
                footer_type="notif",
                source="detector",
                key="detector:partial_deferred",
            )

        self._detector.integrate_prepared_result(integration, fork_id)

        # semantic_blocks may be in a partial-merge state during this loop.
        for new_block in new_blocks:
            self.semantic_blocks.append(new_block)
            self._recorder.write_semantic_block(new_block)
            self._debug_event(
                subsystem="block_detector",
                event="block_crystallized",
                title=f'New block crystallized: "{new_block.title}"',
                metadata={
                    "fork_id": fork_id,
                    "block_id": new_block.id,
                    "block_idx": new_block.idx,
                    "title": new_block.title,
                    "start_block": new_block.range.start_block,
                    "end_block": new_block.range.end_block,
                },
            )
            self._footer_c.queue_footer(
                text=f'New block crystallized: "{new_block.title}"',
                footer_type="notif",
                source="detector",
                key=new_block.id,
            )
            if self._accept_background_work and self._enable_block_metrics:
                key = f"metrics:{new_block.id}"
                if self._nursery.get_by_key(key) is not None:
                    continue
                coro = self._count_semantic_block(block=new_block.range)
                self._nursery.submit(
                    coro=coro,
                    kind="block_metrics",
                    source="block_manager",
                    key=key,
                    metadata={
                        "block_id": new_block.id,
                        "block_idx": new_block.idx,
                    },
                )
        if self._accept_background_work and new_blocks:
            await self.generate_next_summary()

    def _semantic_blocks_for_completed(
        self, completed_ranges: Sequence[IRSemanticBlockRange]
    ) -> list[IRSemanticBlock]:
        new_blocks: list[IRSemanticBlock] = []
        for completed in completed_ranges:
            idx = len(self.semantic_blocks) + len(new_blocks)
            new_block = IRSemanticBlock(
                idx=idx,
                title=completed.title,
                range=completed,
                toks=None,
                full_toks=None,
            )
            new_blocks.append(new_block)
        return new_blocks

    def _discard_detection_result(
        self,
        *,
        fork_id: str,
        result: BlockDetectorResult,
        error: ValueError,
    ) -> None:
        logger.exception(
            "block_detector.result_rejected fork_id=%s completed=%s "
            "still_buffered=%s reason=%s",
            fork_id,
            len(result.completed),
            len(result.still_buffered),
            error,
        )
        self._alert_detection_failure(
            fork_id=fork_id,
            event="result_rejected",
            title="Block detector result was rejected",
            error=error,
            metadata={
                "completed": len(result.completed),
                "still_buffered": len(result.still_buffered),
            },
        )
        self._detector.discard_result(fork_id)
        self._footer_c.queue_footer(
            text="a detection pass failed validation and was discarded",
            footer_type="notif",
            source="detector",
            key="detector:validation_failed",
        )

    def _integrate_block_metrics(
        self, toks: IRTokenRangeCount | None, block_idx: int, block_id: str
    ) -> None:
        if toks is None:
            return
        if not -1 < block_idx < len(self.semantic_blocks):
            return
        block = self.semantic_blocks[block_idx]
        if not block.id == block_id:
            return
        counted_block = block.model_copy(update={"full_toks": toks})
        if counted_block.mode == "full":
            counted_block = counted_block.model_copy(update={"toks": toks})
        self.semantic_blocks[block_idx] = counted_block
        self._recorder.write_block_metrics(toks=toks, block_id=block_id)

    async def maybe_detect(
        self, blocks: Sequence[IRBlock], first_block_id: int
    ) -> None:
        if not self._enable_block_detection:
            return
        maybe_prepared = await self._detector.maybe_detect(blocks, first_block_id)

        if maybe_prepared is not None:
            self._submit_detection(maybe_prepared)

    async def force_detect(self, *, finalize: bool = False) -> bool:
        if not self._enable_block_detection:
            return False
        maybe_prepared = await self._detector.force_detect(finalize=finalize)
        if maybe_prepared is None:
            return False
        self._submit_detection(maybe_prepared)
        return True

    def _submit_detection(self, prepared: PreparedFork) -> None:
        if not self._accept_background_work:
            return
        job = self._nursery.submit(
            prepared.coro,
            kind="detect_blocks",
            source="block_manager",
            metadata={"fork_id": prepared.fork_id},
        )
        self._debug_event(
            subsystem="block_detector",
            event="detection_scheduled",
            title="Block detection scheduled",
            metadata={"job_id": job.id, "fork_id": prepared.fork_id},
        )

    async def _integrate_summarizer_result(
        self,
        *,
        result: BlockSummarizerResult,
        block_idx: int,
        block_id: str,
        fork_id: str,
    ) -> None:
        self._summarizer.integrate_result(fork_id)
        if not -1 < block_idx < len(self.semantic_blocks):
            self._debug_event(
                subsystem="summarizer",
                event="summary_discarded",
                title="Summary result discarded",
                metadata={
                    "fork_id": fork_id,
                    "block_id": block_id,
                    "block_idx": block_idx,
                    "reason": "block_idx_missing",
                },
            )
            return
        block = self.semantic_blocks[block_idx]
        if block.id != block_id:
            self._debug_event(
                subsystem="summarizer",
                event="summary_discarded",
                title="Summary result discarded",
                metadata={
                    "fork_id": fork_id,
                    "block_id": block_id,
                    "block_idx": block_idx,
                    "current_block_id": block.id,
                    "reason": "block_id_mismatch",
                },
            )
            return
        if self._summary_ready(block):
            self._debug_event(
                subsystem="summarizer",
                event="summary_discarded",
                title="Summary result discarded",
                metadata={
                    "fork_id": fork_id,
                    "block_id": block_id,
                    "block_idx": block_idx,
                    "reason": "summary_already_available",
                },
            )
            return
        available_modes = block.available_modes
        if "summary" not in available_modes:
            available_modes = block.available_modes + ["summary"]
        new_block = block.model_copy(
            update={
                "artifacts": block.artifacts + [result.summary],
                "available_modes": available_modes,
            }
        )
        summary_toks = await self._meter.tok_counter.count_blocks(
            [render_summary(new_block)]
        )  # this should be quick because short content
        if summary_toks is not None:
            new_toks = IRTokenRangeCount(tokens=summary_toks, method="api", exact=True)
            new_summary = result.summary.model_copy(update={"toks": new_toks})
            new_block = new_block.model_copy(
                update={"artifacts": block.artifacts + [new_summary]}
            )
        else:
            new_summary = result.summary
        # we're mutating the list while iterating it here...
        # prob should do something a bit cleaner in the future
        self.semantic_blocks[block.idx] = new_block
        self._recorder.write_block_artifact(new_summary, block.id)
        self._debug_event(
            subsystem="summarizer",
            event="summary_applied",
            title=f'Summary applied for block {block.idx}: "{block.title}"',
            metadata={
                "fork_id": fork_id,
                "block_id": block.id,
                "block_idx": block.idx,
                "title": block.title,
                "headline": new_summary.headline,
                "token_count": new_summary.toks.tokens
                if new_summary.toks is not None
                else None,
            },
        )
        await self.generate_next_summary()

    async def generate_next_summary(self) -> None:
        """Generate summary for the next completed block without a summary artifact."""
        if not self._accept_background_work or not self._enable_block_detection:
            return
        prev: list[IRSemanticBlock] = []
        for block in self.semantic_blocks:
            if self._summary_ready(block):
                prev.append(block)
            else:
                await self._schedule_summary_for_block(
                    block=block,
                    prev_semantic_blocks=prev,
                )
                return
            if len(prev) > 5:
                prev.pop(0)

    async def _schedule_summary_for_block(
        self,
        *,
        block: IRSemanticBlock,
        prev_semantic_blocks: list[IRSemanticBlock],
    ) -> bool:
        if not self._accept_background_work or not self._enable_block_detection:
            return False
        key = self._summary_key(block)
        if self._nursery.get_by_key(key) is not None:
            return False
        prepared = await self._summarizer.summarize(
            semantic_block=block,
            context_block_slice=self._project_blocks(
                self._context_blocks_in_block(block)
            ),
            prev_semantic_blocks=prev_semantic_blocks,
        )
        job = self._nursery.submit(
            prepared.coro,
            kind="summarize_block",
            source="block_manager",
            key=key,
            metadata={
                "block_id": block.id,
                "block_idx": block.idx,
                "fork_id": prepared.fork_id,
            },
        )
        self._debug_event(
            subsystem="summarizer",
            event="summary_scheduled",
            title=f'Summary scheduled for block {block.idx}: "{block.title}"',
            metadata={
                "job_id": job.id,
                "fork_id": prepared.fork_id,
                "block_id": block.id,
                "block_idx": block.idx,
                "title": block.title,
            },
        )
        return True

    def _summary_ready(self, block: IRSemanticBlock) -> bool:
        return "summary" in block.available_modes and any(
            isinstance(artifact, IRSemanticBlockSummary) for artifact in block.artifacts
        )

    def _previous_summary_context(self, target_idx: int) -> list[IRSemanticBlock]:
        prev: list[IRSemanticBlock] = []
        for block in self.semantic_blocks[:target_idx]:
            if not self._summary_ready(block):
                continue
            prev.append(block)
            if len(prev) > 5:
                prev.pop(0)
        return prev

    def _summary_key(self, block: IRSemanticBlock) -> str:
        return f"summary:{block.id}"

    async def _queue_summary_for_forget(
        self, block: IRSemanticBlock
    ) -> ForgetBlockResult:
        if not self._enable_block_detection:
            return ForgetBlockResult(
                status="summary_unavailable",
                message=(
                    f"Block {block.idx}'s summary is unavailable because this "
                    "session profile does not run summary forks."
                ),
            )
        key = self._summary_key(block)
        if self._nursery.get_by_key(key) is not None:
            return ForgetBlockResult(
                status="summary_in_flight",
                message=(
                    f"Block {block.idx}'s summary is still baking; a summary job is "
                    "already in flight. Try Forget again in a moment."
                ),
            )

        scheduled = await self._schedule_summary_for_block(
            block=block,
            prev_semantic_blocks=self._previous_summary_context(block.idx),
        )
        if scheduled:
            return ForgetBlockResult(
                status="summary_queued",
                message=(
                    f"Block {block.idx}'s summary is still baking; I queued it just "
                    "now. Try Forget again in a moment."
                ),
            )
        return ForgetBlockResult(
            status="summary_unavailable",
            message=(
                f"Block {block.idx}'s summary is still baking, but summary generation "
                "is not available right now. Try Forget again in a moment."
            ),
        )

    async def forget_block(
        self, idx: int, confirm: bool, source: SemanticBlockApplyModeSource = "model"
    ) -> ForgetBlockResult:
        block = self._get_block_by_idx(idx)
        self._ensure_not_pair_narrative(block, operation="Forget")
        if not confirm and block.pin is not None:
            raise ValueError(
                (
                    f"Block {idx} is currently pinned because: {block.pin.reason}. If you really want "
                    "to forget it, call this tool again with (confirm=true)"
                )
            )
        match block.mode:
            case "summary":
                raise ValueError(f"Block {idx} is already at the lowest possible mode.")
            case "full":
                refusal = self._tool_closure_refusal(block)
                if refusal is not None:
                    self._queue_tool_closure_refusal_footer(block, refusal)
                    return ForgetBlockResult(
                        status="boundary_invalid",
                        message=refusal,
                    )
                if not self._summary_ready(block):
                    return await self._queue_summary_for_forget(block)
                new_block = self._apply_mode(block, "summary", source)
                self.semantic_blocks[idx] = new_block
                return ForgetBlockResult(
                    status="compacted",
                    message=f"Block {idx} successfully compacted.",
                )
            case _:
                raise NotImplementedError(f"Forget does not support mode {block.mode}.")

    def pin_block(self, idx: int, reason: str) -> bool:
        """Returns True when the pin invalidates the prefix."""
        block = self._get_block_by_idx(idx)
        self._ensure_not_pair_narrative(block, operation="Pin")
        new_pin = IRSemanticBlockPin(kind="block", reason=reason)
        if block.pin is not None and block.pin.kind == "block":
            raise ValueError(f"Block {idx} is already pinned!")
        should_invalidate = False
        if not block.mode == "full":
            block = self._apply_mode(block, "full", "model")
            should_invalidate = True
        new_block = block.model_copy(update={"pin": new_pin})
        self._recorder.apply_block_pin(new_pin, block.id)
        self.semantic_blocks[idx] = new_block
        return should_invalidate

    def pin_facet(self, idx: int, facet_id: str, reason: str) -> bool:
        """Returns True when the pin invalidates the rendered prefix."""
        block = self._get_block_by_idx(idx)
        self._ensure_not_pair_narrative(block, operation="Pin")
        artifact = self._summary_artifact(block)
        facet = next((facet for facet in artifact.facets if facet.id == facet_id), None)
        if facet is None:
            raise ValueError(f'Block {idx} has no summary facet with id="{facet_id}".')
        if any(pin.facet_id == facet_id for pin in block.facet_pins):
            raise ValueError(f'Facet "{facet_id}" in block {idx} is already pinned!')

        new_pin = IRSemanticBlockPin(
            kind="facet",
            reason=reason,
            facet_id=facet_id,
        )
        new_block = block.model_copy(
            update={"facet_pins": block.facet_pins + [new_pin]}
        )
        self._recorder.apply_block_pin(new_pin, block.id)
        self.semantic_blocks[idx] = new_block
        return block.mode == "summary"

    def recall_block(self, idx: int) -> str:
        # TODO: make this TTL or something like that
        block = self._get_block_by_idx(idx)
        if block.mode == "pair_narrative":
            chapter = self._pair_narrative_artifact(block).chapter_number
            raise ValueError(
                f"Block {idx} is rendered as chapter {chapter} pair narrative memory. "
                f"Use Recall(chapter={chapter}) to recall the original source blocks."
            )
        if block.mode == "full":
            raise ValueError(
                f"Block {idx} is already entirely in context and cannot be recalled further."
            )
        context_blocks = self._context_blocks_in_block(block)
        output: list[str] = [f'# Block {idx} - "{block.title}"\n']
        for block_id, b in enumerate(context_blocks, start=block.range.start_block):
            output.append(render_context_block(b, block_id))

        return "\n".join(output)

    def recall_chapter(self, chapter_number: int) -> str:
        if chapter_number < 1:
            raise ValueError(f"Chapter number must be positive: {chapter_number}.")
        first_idx = (chapter_number - 1) * 2
        second_idx = first_idx + 1
        if second_idx >= len(self.semantic_blocks):
            raise ValueError(
                f"Chapter {chapter_number} maps to blocks {first_idx}-{second_idx}, "
                f"but there are only {len(self.semantic_blocks)} semantic blocks."
            )

        first = self._get_block_by_idx(first_idx)
        second = self._get_block_by_idx(second_idx)
        output: list[str] = [
            f"# Chapter {chapter_number:02d} source blocks",
            "",
            f'## Block {first.idx} - "{first.title}"',
            "",
        ]
        for block_id, context_block in enumerate(
            self._context_blocks_in_block(first),
            start=first.range.start_block,
        ):
            output.append(render_context_block(context_block, block_id))

        output.extend(["", f'## Block {second.idx} - "{second.title}"', ""])
        for block_id, context_block in enumerate(
            self._context_blocks_in_block(second),
            start=second.range.start_block,
        ):
            output.append(render_context_block(context_block, block_id))

        return "\n".join(output)

    def _ensure_not_pair_narrative(
        self, block: IRSemanticBlock, *, operation: str
    ) -> None:
        if block.mode != "pair_narrative":
            return
        artifact = self._pair_narrative_artifact(block)
        raise ValueError(
            f"{operation} cannot operate on block {block.idx} because it is part "
            f"of chapter {artifact.chapter_number} pair narrative memory."
        )

    def _get_block_by_idx(self, idx: int) -> IRSemanticBlock:
        if not -1 < idx < len(self.semantic_blocks):
            raise ValueError(
                f"{idx} is not a valid block id - there's only {len(self.semantic_blocks)} blocks!"
            )
        return self.semantic_blocks[idx]

    async def check_nursery(
        self,
        *,
        wait_for_all: bool = False,
    ) -> None:
        if wait_for_all:
            await self._wait_for_jobs()

        ready = self._nursery.collect_ready(source="block_manager")
        for result in ready:
            await self._integrate_nursery_result(result)

        await self.generate_next_summary()
        if wait_for_all:
            await self._wait_for_jobs()

    async def shutdown_nursery(self) -> None:
        self._accept_background_work = False
        results = await self._nursery.shutdown(cancel=True)
        for result in results:
            await self._integrate_nursery_result(result)

    async def _wait_for_jobs(self) -> None:
        while True:
            jobs = self._nursery.jobs(source="block_manager")
            if not jobs:
                return
            for job in jobs:
                result = await self._nursery.wait(job.id)
                if result is not None:
                    await self._integrate_nursery_result(result)

    async def _integrate_nursery_result(self, result: NurseryJobResult[Any]) -> None:
        match result.job.kind:
            case "detect_blocks":
                fork_id = self._get_fork_id(result.job)
                if result.cancelled or result.error is not None:
                    self._fork_runner.integrate_result(fork_id)
                elif result.result is not None:
                    assert isinstance(result.result, BlockDetectorResult)
                    await self._integrate_detection(result.result, fork_id)
                else:
                    self._fork_runner.integrate_result(fork_id)
            case "summarize_block":
                fork_id = self._get_fork_id(result.job)
                if result.cancelled or result.error is not None:
                    self._fork_runner.integrate_result(fork_id)
                elif result.result is not None:
                    block_idx = self._get_block_idx(result.job)
                    block_id = self._get_block_id(result.job)
                    assert isinstance(result.result, BlockSummarizerResult)
                    await self._integrate_summarizer_result(
                        result=result.result,
                        block_idx=block_idx,
                        block_id=block_id,
                        fork_id=fork_id,
                    )
                else:
                    self._fork_runner.integrate_result(fork_id)
            case "block_metrics":
                if not result.cancelled and result.result is not None:
                    block_idx = self._get_block_idx(result.job)
                    block_id = self._get_block_id(result.job)
                    assert isinstance(result.result, IRTokenRangeCount | None)
                    self._integrate_block_metrics(
                        toks=result.result, block_idx=block_idx, block_id=block_id
                    )

    def _get_fork_id(self, job: NurseryJob[Any]) -> str:
        fork_id = job.metadata.get("fork_id", None)
        if not isinstance(fork_id, str):
            raise ValueError(
                f'fork_id is None for job(id="{job.id}") when it really shouldn\'t be.'
            )
        return fork_id

    def _get_block_idx(self, job: NurseryJob[Any]) -> int:
        block_idx = job.metadata.get("block_idx")
        if not isinstance(block_idx, int):
            raise ValueError(
                f'block_idx is missing for job(id="{job.id}") when it really shouldn\'t be.'
            )
        return block_idx

    def _alert_detection_failure(
        self,
        *,
        fork_id: str,
        event: str,
        title: str,
        error: BaseException | None = None,
        metadata: dict[str, Any] | None = None,
        level: "DebugNoticeLevel" = "error",
    ) -> None:
        if self._debug is None:
            return
        event_metadata: dict[str, Any] = {
            "fork_id": fork_id,
            "error_type": type(error).__name__ if error is not None else None,
            "error_message": str(error) if error is not None else None,
        }
        if metadata is not None:
            event_metadata.update(metadata)
        self._debug.alert(
            subsystem="block_detector",
            event=event,
            title=title,
            content=_render_detection_failure(
                fork_id=fork_id,
                event=event,
                title=title,
                error=error,
                metadata=metadata or {},
            ),
            plaintext=_detection_failure_plaintext(
                fork_id=fork_id,
                title=title,
                error=error,
            ),
            metadata=event_metadata,
            level=level,
        )

    def _debug_event(
        self,
        *,
        subsystem: str,
        event: str,
        title: str,
        metadata: dict[str, object],
        content: str | None = None,
    ) -> None:
        if self._debug is None:
            return
        self._debug.debug(
            subsystem=subsystem,
            event=event,
            title=title,
            content=content,
            plaintext=title,
            metadata=metadata,
        )

    def _get_block_id(self, job: NurseryJob[Any]) -> str:
        block_id = job.metadata.get("block_id")
        if not isinstance(block_id, str):
            raise ValueError(
                f'block_id is missing for job(id="{job.id}") when it really shouldn\'t be.'
            )
        return block_id

    def _apply_mode(
        self,
        block: IRSemanticBlock,
        mode: SemanticBlockMode,
        source: SemanticBlockApplyModeSource,
    ) -> IRSemanticBlock:
        if mode == block.mode:
            return block
        if mode not in block.available_modes:
            raise ValueError(
                f"Block {block.idx} currently has no {mode} artifact. Please wait and try again later."
            )
        if mode == "summary":
            refusal = self._tool_closure_refusal(block)
            if refusal is not None:
                self._queue_tool_closure_refusal_footer(block, refusal)
                raise ValueError(refusal)
        match mode:
            case "full":
                new_block = block.model_copy(
                    update={"mode": "full", "toks": block.full_toks}
                )
            case "summary":
                artifact = next(a for a in block.artifacts if a.mode == "summary")
                new_block = block.model_copy(
                    update={"mode": "summary", "toks": artifact.toks}
                )
            case "pair_narrative":
                artifact = self._pair_narrative_artifact(block)
                new_block = block.model_copy(
                    update={"mode": "pair_narrative", "toks": artifact.toks}
                )
            case _:
                raise NotImplementedError(
                    f'`_apply_mode` for mode="{mode}" is not implemented.'
                )
        self._recorder.apply_semantic_block_mode(
            mode, block.id, source
        )  # this only gets recorded when successful
        return new_block

    async def _count_semantic_block(
        self, block: IRSemanticBlockRange
    ) -> IRTokenRangeCount | None:
        count = await self._meter.count_slice(
            self.context_blocks,
            block.start_block,
            block.end_block + 1,
        )
        return count

    def _project_blocks(self, blocks: Sequence[IRBlock]) -> list[IRBlock]:
        projected = self._context_projector(blocks)
        if len(projected) != len(blocks):
            raise ValueError("Context projector must preserve block count.")
        return projected

    def _validate_semantic_blocks(
        self,
        semantic_blocks: Sequence[IRSemanticBlock] | None = None,
        *,
        validate_tool_closure: bool = False,
    ) -> None:
        """Semantic blocks must render as a gapless prefix of context_blocks."""
        blocks = self.semantic_blocks if semantic_blocks is None else semantic_blocks
        expected_start = 0
        for expected_idx, block in enumerate(blocks):
            if block.idx != expected_idx:
                raise ValueError(
                    "Semantic blocks must have ordered, gapless idx values. "
                    f"Expected idx={expected_idx}, got idx={block.idx}."
                )
            if block.range.start_block != expected_start:
                raise ValueError(
                    "Semantic blocks must form a gapless prefix of context blocks. "
                    f'Block "{block.title}" starts at {block.range.start_block}, '
                    f"expected {expected_start}."
                )
            if block.range.end_block >= len(self.context_blocks):
                raise ValueError(
                    "Semantic blocks must stay within the known context blocks. "
                    f'Block "{block.title}" ends at {block.range.end_block}, '
                    f"but there are only {len(self.context_blocks)} context blocks."
                )
            expected_start = block.range.end_block + 1
        if validate_tool_closure:
            violation = self._first_tool_closure_violation(blocks)
            if violation is not None:
                raise ValueError(self._tool_closure_validation_message(violation))

    def _first_tool_closure_violation(
        self, blocks: Sequence[IRSemanticBlock]
    ) -> _ToolClosureViolation | None:
        envelopes = self._tool_closure_envelopes()
        if not envelopes:
            return None
        for block in blocks:
            for envelope in envelopes:
                if not _ranges_overlap(
                    block.range.start_block,
                    block.range.end_block,
                    envelope.start_block,
                    envelope.end_block,
                ):
                    continue
                if (
                    not envelope.complete
                    or block.range.start_block > envelope.start_block
                    or block.range.end_block < envelope.end_block
                ):
                    return _ToolClosureViolation(block=block, envelope=envelope)
        return None

    def _tool_closure_refusal(self, block: IRSemanticBlock) -> str | None:
        violation = self._first_tool_closure_violation([block])
        if violation is None:
            return None
        return self._tool_closure_refusal_message(violation)

    def _tool_closure_envelopes(self) -> list[_ToolClosureEnvelope]:
        call_positions: dict[str, int] = {}
        result_positions: dict[str, int] = {}
        for idx, block in enumerate(self.context_blocks):
            if isinstance(block, IRToolCallBlock):
                call_positions[block.call_id] = idx
            elif isinstance(block, IRToolResultBlock):
                result_positions[block.call_id] = idx

        envelopes: list[_ToolClosureEnvelope] = []
        covered_call_ids: set[str] = set()
        idx = 0
        while idx < len(self.context_blocks):
            if self._provider_role(self.context_blocks[idx]) != "assistant":
                idx += 1
                continue

            segment_call_ids: list[str] = []
            segment_call_positions: list[int] = []
            while (
                idx < len(self.context_blocks)
                and self._provider_role(self.context_blocks[idx]) == "assistant"
            ):
                block = self.context_blocks[idx]
                if isinstance(block, IRToolCallBlock):
                    segment_call_ids.append(block.call_id)
                    segment_call_positions.append(idx)
                    covered_call_ids.add(block.call_id)
                idx += 1

            if not segment_call_ids:
                continue

            result_blocks = [
                result_positions[call_id]
                for call_id in segment_call_ids
                if call_id in result_positions
            ]
            start_block = min([*segment_call_positions, *result_blocks])
            end_block = max([*segment_call_positions, *result_blocks])
            complete = len(result_blocks) == len(segment_call_ids) and all(
                result_positions[call_id] > call_positions[call_id]
                for call_id in segment_call_ids
                if call_id in result_positions
            )
            envelopes.append(
                _ToolClosureEnvelope(
                    start_block=start_block,
                    end_block=end_block,
                    call_ids=tuple(segment_call_ids),
                    complete=complete,
                )
            )

        for call_id, result_block in result_positions.items():
            if call_id in covered_call_ids:
                continue
            envelopes.append(
                _ToolClosureEnvelope(
                    start_block=result_block,
                    end_block=result_block,
                    call_ids=(call_id,),
                    complete=False,
                )
            )
        return envelopes

    def _tool_closure_validation_message(self, violation: _ToolClosureViolation) -> str:
        block = violation.block
        envelope = violation.envelope
        return (
            "Semantic block validation rejected a tool-closure split. "
            f'Block {block.idx} "{block.title}" covers '
            f"{block.range.start_block}-{block.range.end_block}, but tool "
            f"call/result envelope {_format_call_ids(envelope.call_ids)} spans "
            f"{envelope.start_block}-{envelope.end_block}. Repair semantic block "
            "boundaries so every tool call batch and matching tool result batch "
            "are contained in one semantic block."
        )

    def _tool_closure_refusal_message(self, violation: _ToolClosureViolation) -> str:
        block = violation.block
        envelope = violation.envelope
        return (
            f'Block {block.idx} "{block.title}" cannot be compacted because its '
            f"semantic range {block.range.start_block}-{block.range.end_block} "
            "splits a tool call/result envelope "
            f"({_format_call_ids(envelope.call_ids)}, envelope "
            f"{envelope.start_block}-{envelope.end_block}). Repair the "
            "transcript's semantic block boundaries so each tool call batch and "
            "matching tool result batch are contained in the same semantic block, "
            "then try Forget again."
        )

    def _queue_tool_closure_refusal_footer(
        self, block: IRSemanticBlock, message: str
    ) -> None:
        self._footer_c.queue_footer(
            text=message,
            footer_type="notif",
            source="runtime",
            key=f"forget:tool_closure:{block.id}",
        )

    def _log_detection_range_sanitization(
        self,
        *,
        fork_id: str,
        raw: BlockDetectorResult,
        sanitized: BlockDetectorIntegration,
    ) -> None:
        raw_completed = _range_specs(raw.completed)
        raw_buffered = _range_specs(raw.still_buffered)
        sanitized_completed = _range_specs(sanitized.completed)
        sanitized_buffered = _range_specs(sanitized.result.still_buffered)
        logger.debug(
            "block_detector.ranges_sanitized fork_id=%s raw_completed=%s "
            "raw_buffered=%s sanitized_completed=%s sanitized_buffered=%s "
            "deferred=%s discarded=%s",
            fork_id,
            raw_completed,
            raw_buffered,
            sanitized_completed,
            sanitized_buffered,
            _range_specs(sanitized.deferred_completed),
            _range_specs(sanitized.discarded_ranges),
        )
        if raw_completed == sanitized_completed and raw_buffered == sanitized_buffered:
            return
        logger.info(
            "block_detector.ranges_changed fork_id=%s completed=%s->%s "
            "buffered=%s->%s deferred=%s discarded=%s",
            fork_id,
            raw_completed,
            sanitized_completed,
            raw_buffered,
            sanitized_buffered,
            _range_specs(sanitized.deferred_completed),
            _range_specs(sanitized.discarded_ranges),
        )

    def _write_detection_forensics(
        self,
        *,
        fork_id: str,
        outcome: _DetectionIntegrationOutcome,
        raw: BlockDetectorResult,
        sanitized: BlockDetectorIntegration | None,
        error: ValueError | None,
    ) -> None:
        path = self._detector_forensics_path
        if path is None:
            return
        try:
            breadcrumb = spellbook_runtime_breadcrumb()
            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "fork_id": fork_id,
                "outcome": outcome,
                "raw": {
                    "completed": _range_payloads(raw.completed),
                    "still_buffered": _range_payloads(raw.still_buffered),
                },
                "sanitized": (
                    {
                        "completed": _range_payloads(sanitized.completed),
                        "still_buffered": _range_payloads(
                            sanitized.result.still_buffered
                        ),
                        "deferred_completed": _range_payloads(
                            sanitized.deferred_completed
                        ),
                        "discarded_ranges": _range_payloads(sanitized.discarded_ranges),
                    }
                    if sanitized is not None
                    else None
                ),
                "validation_errors": (
                    [{"type": type(error).__name__, "message": str(error)}]
                    if error is not None
                    else []
                ),
                "runtime_breadcrumb": {
                    "package_path": breadcrumb.package_path,
                    "git_sha": breadcrumb.git_sha,
                    "git_state": breadcrumb.git_state,
                },
            }
            with path.open("a", encoding="utf-8") as sink:
                sink.write(json.dumps(payload, separators=(",", ":")) + "\n")
        except Exception:  # noqa: BLE001 - diagnostics must never affect integration.
            logger.exception(
                "block_detector.forensics_write_failed fork_id=%s outcome=%s path=%s",
                fork_id,
                outcome,
                path,
            )


def _identity_context_projector(blocks: Sequence[IRBlock]) -> list[IRBlock]:
    return list(blocks)


def _ranges_overlap(
    left_start: int,
    left_end: int,
    right_start: int,
    right_end: int,
) -> bool:
    return left_start <= right_end and right_start <= left_end


def _format_call_ids(call_ids: tuple[str, ...]) -> str:
    if len(call_ids) == 1:
        return f"call_id={call_ids[0]}"
    return "call_ids=" + ",".join(call_ids)


def _range_specs(ranges: Sequence[IRSemanticBlockRange]) -> list[str]:
    return [f"{block.start_block}-{block.end_block}:{block.title}" for block in ranges]


def _range_payloads(
    ranges: Sequence[IRSemanticBlockRange],
) -> list[dict[str, Any]]:
    return [block.model_dump(mode="json") for block in ranges]


def _render_detection_failure(
    *,
    fork_id: str,
    event: str,
    title: str,
    error: BaseException | None,
    metadata: dict[str, Any],
) -> str:
    rows = [
        f"# {title}",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Fork | `{fork_id}` |",
        f"| Event | `{event}` |",
    ]
    for key, value in sorted(metadata.items()):
        rows.append(f"| {key} | `{_escape_table(value)}` |")
    if error is not None:
        rows.append(
            f"| Error | `{type(error).__name__}: {_escape_table(str(error))}` |"
        )
    return "\n".join(rows)


def _detection_failure_plaintext(
    *, fork_id: str, title: str, error: BaseException | None
) -> str:
    if error is None:
        return f"{title}: {fork_id}."
    return f"{title}: {fork_id}: {error}"


def _escape_table(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")

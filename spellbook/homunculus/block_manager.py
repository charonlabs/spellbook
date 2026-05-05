from typing import Any, Sequence

from spellbook.config import HomunculusConfig
from spellbook.footer import FooterController
from spellbook.fork import (
    BlockDetectorResult,
    BlockSummarizerResult,
    ForkRunner,
    PreparedFork,
)
from spellbook.homunculus.block_detector import BlockDetector
from spellbook.homunculus.block_summarizer import BlockSummarizer
from spellbook.homunculus.common import render_context_block, render_summary
from spellbook.homunculus.token_meter import TokenMeter
from spellbook.ir_types import (
    IRAssistantTextBlock,
    IRBlock,
    IRImageBlock,
    IRSemanticBlock,
    IRSemanticBlockFacet,
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
    ):
        """Manager for semantic and context blocks. Public instance vars can be
        read and mutated in the Homunculus itself. At present, `context_blocks` and
        `next_block_id` are both directly mutated externally."""
        self._detector = BlockDetector(
            config=config, fork_runner=fork_runner, recorder=recorder
        )
        self._fork_runner = fork_runner
        self._meter = token_meter
        self._recorder = recorder
        self._footer_c = footer_c
        self._nursery = nursery
        self._summarizer = BlockSummarizer(
            config=config, fork_runner=fork_runner, recorder=recorder
        )
        self.semantic_blocks: list[IRSemanticBlock] = []  # This should stay ordered
        self.context_blocks: list[IRBlock] = []
        self.next_block_id = 0

    @property
    def proposed_semantic_blocks(self) -> list[IRSemanticBlockRange]:
        return self._detector.buffered_blocks

    @property
    def has_unfinalized_detection(self) -> bool:
        return self._detector.has_pending_blocks

    def rehydrate(self, rehydrated: RehydrationResult) -> None:
        self._detector.rehydrate(rehydrated)
        self.semantic_blocks = rehydrated.semantic_blocks
        self._validate_semantic_blocks()

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
        artifact = next((a for a in block.artifacts if a.type == "summary"), None)
        if artifact is None:
            raise ValueError(f"Block {block.idx} has no summary artifact.")
        return artifact

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
            case IRAssistantTextBlock() | IRThinkingBlock() | IRToolCallBlock():
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
        new_completed = self._detector.integrate_result(result, fork_id)
        new_blocks: list[IRSemanticBlock] = []
        for completed in new_completed:
            idx = len(self.semantic_blocks) + len(new_blocks)
            new_block = IRSemanticBlock(
                idx=idx,
                title=completed.title,
                range=completed,
                toks=None,
                full_toks=None,
            )
            new_blocks.append(new_block)

        candidate_blocks = [*self.semantic_blocks, *new_blocks]
        self._validate_semantic_blocks(candidate_blocks)

        # semantic_blocks may be in a partial-merge state during this loop.
        for new_block in new_blocks:
            self.semantic_blocks.append(new_block)
            self._recorder.write_semantic_block(new_block)
            self._footer_c.queue_footer(
                text=f'New block crystallized: "{new_block.title}"',
                footer_type="notif",
                source="detector",
                key=new_block.id,
            )
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
        if new_blocks:
            await self.generate_next_summary()

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
        maybe_prepared = await self._detector.maybe_detect(blocks, first_block_id)

        if maybe_prepared is not None:
            self._submit_detection(maybe_prepared)

    async def force_detect(self, *, finalize: bool = False) -> bool:
        maybe_prepared = await self._detector.force_detect(finalize=finalize)
        if maybe_prepared is None:
            return False
        self._submit_detection(maybe_prepared)
        return True

    def _submit_detection(self, prepared: PreparedFork) -> None:
        self._nursery.submit(
            prepared.coro,
            kind="detect_blocks",
            source="block_manager",
            metadata={"fork_id": prepared.fork_id},
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
            return
        block = self.semantic_blocks[block_idx]
        if block.id != block_id or "summary" in block.available_modes:
            return
        new_block = block.model_copy(
            update={
                "artifacts": block.artifacts + [result.summary],
                "available_modes": block.available_modes + ["summary"],
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
        await self.generate_next_summary()

    async def generate_next_summary(self) -> None:
        """Generate summary for the next completed block without a summary artifact."""
        prev: list[IRSemanticBlock] = []
        for block in self.semantic_blocks:
            if "summary" in block.available_modes:
                prev.append(block)
            else:
                key = f"summary:{block.id}"
                if self._nursery.get_by_key(key) is not None:
                    return
                prepared = await self._summarizer.summarize(
                    semantic_block=block,
                    context_block_slice=self._context_blocks_in_block(block),
                    prev_semantic_blocks=prev,
                )
                self._nursery.submit(
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
                return
            if len(prev) > 5:
                prev.pop(0)

    def forget_block(
        self, idx: int, confirm: bool, source: SemanticBlockApplyModeSource = "model"
    ) -> None:
        block = self._get_block_by_idx(idx)
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
                new_block = self._apply_mode(block, "summary", source)
                self.semantic_blocks[idx] = new_block

    def pin_block(self, idx: int, reason: str) -> bool:
        """Returns True when the pin invalidates the prefix."""
        block = self._get_block_by_idx(idx)
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
        if block.mode == "full":
            raise ValueError(
                f"Block {idx} is already entirely in context and cannot be recalled further."
            )
        context_blocks = self._context_blocks_in_block(block)
        output: list[str] = [f'# Block {idx} - "{block.title}"\n']
        for block_id, b in enumerate(context_blocks, start=block.range.start_block):
            output.append(render_context_block(b, block_id))

        return "\n".join(output)

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

    def _validate_semantic_blocks(
        self, semantic_blocks: Sequence[IRSemanticBlock] | None = None
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

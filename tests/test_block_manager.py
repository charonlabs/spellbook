from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Sequence, cast

import pytest

from spellbook.config import HomunculusConfig, SpellbookConfig
from spellbook.footer import FooterController
from spellbook.fork import (
    BlockDetectorResult,
    BlockSummarizerResult,
    ForkRunner,
    PreparedFork,
)
from spellbook.homunculus.block_manager import BlockManager
from spellbook.homunculus.token_meter import TokenMeter
from spellbook.ir_types import (
    IRAssistantTextBlock,
    IRBlock,
    IRSemanticBlock,
    IRSemanticBlockFacet,
    IRSemanticBlockPin,
    IRSemanticBlockRange,
    IRSemanticBlockSummary,
    IRSkillCatalog,
    IRTokenRangeCount,
    IRToolCallBlock,
    IRToolResultBlock,
    IRToolTextBlock,
    IRUserTextBlock,
    SemanticBlockApplyModeSource,
)
from spellbook.nursery import Nursery
from spellbook.recorder import Recorder
from spellbook.rehydrator import RehydrationResult


def _user(text: str) -> IRUserTextBlock:
    return IRUserTextBlock(text=text, origin="human")


async def _settle() -> None:
    await asyncio.sleep(0)
    await asyncio.sleep(0)


def _user_blocks(*texts: str) -> list[IRBlock]:
    return [_user(text) for text in texts]


def _semantic_block(
    *,
    idx: int,
    start: int,
    end: int,
    title: str,
) -> IRSemanticBlock:
    semantic_range = IRSemanticBlockRange(
        title=title,
        start_block=start,
        end_block=end,
        completed=True,
    )
    return IRSemanticBlock(
        idx=idx,
        title=title,
        range=semantic_range,
        toks=None,
        full_toks=None,
    )


def _rehydrated(
    *,
    tmp_path: Path,
    blocks: list[IRBlock],
    semantic_blocks: list[IRSemanticBlock],
) -> RehydrationResult:
    return RehydrationResult(
        session_id="session_test",
        records=[],
        config=SpellbookConfig(cwd=tmp_path),
        blocks=blocks,
        tools=[],
        last_completed_turn=0,
        pending_footers={},
        completed_semantic_block_ranges=[block.range for block in semantic_blocks],
        buffered_semantic_block_ranges=[],
        semantic_blocks=semantic_blocks,
        plan_proposal=None,
        skill_catalog=IRSkillCatalog(),
    )


class _FakeMeter:
    def __init__(self) -> None:
        self.tok_counter = self

    async def count_range(
        self, blocks: list[IRBlock], start: int, end: int
    ) -> IRTokenRangeCount | None:
        return IRTokenRangeCount(
            tokens=end - start,
            method="prefix_delta",
            exact=True,
        )

    async def count_slice(
        self, blocks: list[IRBlock], start: int, end: int
    ) -> IRTokenRangeCount | None:
        return IRTokenRangeCount(
            tokens=end - start,
            method="api",
            exact=True,
        )

    async def count_blocks(self, blocks: list[IRBlock]) -> int | None:
        return len(blocks) * 10


class _FakeRecorder:
    def __init__(self) -> None:
        self.semantic_blocks: list[IRSemanticBlock] = []
        self.applied_modes: list[tuple[str, str]] = []
        self.artifacts: list[tuple[IRSemanticBlockSummary, str]] = []
        self.detected: list[BlockDetectorResult] = []
        self.metrics: list[tuple[IRTokenRangeCount, str]] = []
        self.pins: list[tuple[IRSemanticBlockPin, str]] = []

    def write_semantic_block(self, block: IRSemanticBlock) -> None:
        self.semantic_blocks.append(block)

    def apply_semantic_block_mode(
        self, mode: str, block_id: str, source: SemanticBlockApplyModeSource
    ) -> None:
        self.applied_modes.append((mode, block_id))

    def write_block_artifact(
        self, artifact: IRSemanticBlockSummary, block_id: str
    ) -> None:
        self.artifacts.append((artifact, block_id))

    def detect_blocks(self, result: BlockDetectorResult) -> None:
        self.detected.append(result)

    def write_block_metrics(self, toks: IRTokenRangeCount, block_id: str) -> None:
        self.metrics.append((toks, block_id))

    def apply_block_pin(self, pin: IRSemanticBlockPin, block_id: str) -> None:
        self.pins.append((pin, block_id))


class _FakeFooter:
    def __init__(self) -> None:
        self.queued: list[dict[str, object]] = []

    def queue_footer(self, **kwargs: object) -> None:
        self.queued.append(kwargs)


class _FakeDetector:
    def __init__(self, completed: list[IRSemanticBlockRange]):
        self.completed = completed
        self.integrated_forks: list[str] = []

    def rehydrate(self, rehydrated: RehydrationResult) -> None:
        return None

    async def maybe_detect(
        self, blocks: Sequence[IRBlock], first_block_id: int
    ) -> PreparedFork:
        async def _run() -> BlockDetectorResult:
            return BlockDetectorResult(completed=self.completed, still_buffered=[])

        return PreparedFork(coro=_run(), fork_id="detector_test")

    def integrate_result(
        self, result: BlockDetectorResult, fork_id: str
    ) -> list[IRSemanticBlockRange]:
        self.integrated_forks.append(fork_id)
        return result.completed


class _FakeSummarizer:
    def __init__(self) -> None:
        self.integrated_forks: list[str] = []
        self.calls: list[str] = []

    async def summarize(
        self,
        *,
        semantic_block: IRSemanticBlock,
        context_block_slice: list[IRBlock],
        prev_semantic_blocks: list[IRSemanticBlock],
    ) -> PreparedFork:
        self.calls.append(semantic_block.id)

        async def _run() -> BlockSummarizerResult:
            return BlockSummarizerResult(
                summary=IRSemanticBlockSummary(
                    headline=f"Summary for {semantic_block.title}",
                    text="Summary text.",
                    facets=[],
                    open_thread=None,
                    toks=None,
                )
            )

        return PreparedFork(coro=_run(), fork_id=f"summarizer_{semantic_block.idx}")

    def integrate_result(self, fork_id: str) -> None:
        self.integrated_forks.append(fork_id)


class _FakeForkRunner:
    def __init__(self) -> None:
        self.integrated_forks: list[str] = []

    def integrate_result(self, fork_id: str) -> None:
        self.integrated_forks.append(fork_id)


def _manager() -> tuple[BlockManager, _FakeRecorder, _FakeFooter, _FakeForkRunner]:
    recorder = _FakeRecorder()
    footer = _FakeFooter()
    fork_runner = _FakeForkRunner()
    manager = BlockManager(
        config=HomunculusConfig(),
        fork_runner=cast(ForkRunner, fork_runner),
        footer_c=cast(FooterController, footer),
        nursery=Nursery(config=SpellbookConfig(cwd=Path.cwd())),
        recorder=cast(Recorder, recorder),
        token_meter=cast(TokenMeter, _FakeMeter()),
    )
    return manager, recorder, footer, fork_runner


def _count(tokens: int) -> IRTokenRangeCount:
    return IRTokenRangeCount(tokens=tokens, method="prefix_delta", exact=True)


def _summary(
    headline: str = "Summary headline",
    text: str = "Summary text.",
    toks: IRTokenRangeCount | None = None,
    facets: list[IRSemanticBlockFacet] | None = None,
) -> IRSemanticBlockSummary:
    return IRSemanticBlockSummary(
        headline=headline,
        text=text,
        facets=facets or [],
        open_thread=None,
        toks=toks,
    )


def _rehydrate_manager(
    tmp_path: Path,
    *,
    blocks: list[IRBlock],
    semantic_blocks: list[IRSemanticBlock],
) -> BlockManager:
    manager, _, _, _ = _manager()
    manager.context_blocks = blocks
    manager.rehydrate(
        _rehydrated(
            tmp_path=tmp_path,
            blocks=blocks,
            semantic_blocks=semantic_blocks,
        )
    )
    return manager


def test_empty_semantic_blocks_render_full_tail(tmp_path: Path) -> None:
    blocks = _user_blocks("a", "b")
    manager = _rehydrate_manager(tmp_path, blocks=blocks, semantic_blocks=[])

    assert manager.render_tail() == blocks


def test_render_block_accepts_zero_idx(tmp_path: Path) -> None:
    blocks = _user_blocks("a", "b")
    semantic_blocks = [
        _semantic_block(idx=0, start=0, end=0, title="First"),
    ]
    manager = _rehydrate_manager(
        tmp_path,
        blocks=blocks,
        semantic_blocks=semantic_blocks,
    )

    assert manager.render_block(idx=0) == [blocks[0]]


def test_gapless_prefix_renders_blocks_then_tail_in_order(tmp_path: Path) -> None:
    blocks = _user_blocks("a", "b", "c", "d")
    semantic_blocks = [
        _semantic_block(idx=0, start=0, end=1, title="First"),
        _semantic_block(idx=1, start=2, end=2, title="Second"),
    ]
    manager = _rehydrate_manager(
        tmp_path,
        blocks=blocks,
        semantic_blocks=semantic_blocks,
    )

    rendered: list[IRBlock] = []
    for block in manager.semantic_blocks:
        rendered.extend(manager.render_block(semantic_block=block))
    rendered.extend(manager.render_tail())

    assert rendered == blocks


def test_rehydrate_rejects_gap_in_semantic_prefix(tmp_path: Path) -> None:
    blocks = _user_blocks("a", "b", "c")
    semantic_blocks = [
        _semantic_block(idx=0, start=0, end=0, title="First"),
        _semantic_block(idx=1, start=2, end=2, title="Gap"),
    ]

    with pytest.raises(ValueError, match="gapless prefix"):
        _rehydrate_manager(tmp_path, blocks=blocks, semantic_blocks=semantic_blocks)


def test_rehydrate_rejects_overlapping_semantic_ranges(tmp_path: Path) -> None:
    blocks = _user_blocks("a", "b", "c")
    semantic_blocks = [
        _semantic_block(idx=0, start=0, end=1, title="First"),
        _semantic_block(idx=1, start=1, end=2, title="Overlap"),
    ]

    with pytest.raises(ValueError, match="gapless prefix"):
        _rehydrate_manager(tmp_path, blocks=blocks, semantic_blocks=semantic_blocks)


def test_rehydrate_rejects_out_of_bounds_semantic_range(tmp_path: Path) -> None:
    blocks = _user_blocks("a", "b")
    semantic_blocks = [
        _semantic_block(idx=0, start=0, end=2, title="Too Far"),
    ]

    with pytest.raises(ValueError, match="within the known context"):
        _rehydrate_manager(tmp_path, blocks=blocks, semantic_blocks=semantic_blocks)


def test_rehydrate_rejects_bad_semantic_idx_order(tmp_path: Path) -> None:
    blocks = _user_blocks("a", "b")
    semantic_blocks = [
        _semantic_block(idx=1, start=0, end=0, title="Wrong Idx"),
    ]

    with pytest.raises(ValueError, match="gapless idx"):
        _rehydrate_manager(tmp_path, blocks=blocks, semantic_blocks=semantic_blocks)


@pytest.mark.asyncio
async def test_detected_blocks_are_validated_before_recording() -> None:
    manager, recorder, footer, _ = _manager()
    manager.context_blocks = _user_blocks("a", "b")
    manager._detector = cast(  # noqa: SLF001 - test swaps collaborator
        Any,
        _FakeDetector([IRSemanticBlockRange(title="Gap", start_block=1, end_block=1)]),
    )

    await manager.maybe_detect([manager.context_blocks[1]], first_block_id=1)
    await _settle()
    with pytest.raises(ValueError, match="gapless prefix"):
        await manager.check_nursery()

    assert manager.semantic_blocks == []
    assert recorder.semantic_blocks == []
    assert footer.queued == []


@pytest.mark.asyncio
async def test_detector_jobs_are_best_effort() -> None:
    manager, _, _, _ = _manager()
    manager.context_blocks = _user_blocks("a")
    manager._detector = cast(  # noqa: SLF001 - test swaps collaborator
        Any,
        _FakeDetector(
            [IRSemanticBlockRange(title="First", start_block=0, end_block=0)]
        ),
    )
    manager._summarizer = cast(  # noqa: SLF001 - test swaps collaborator
        Any,
        _FakeSummarizer(),
    )

    await manager.maybe_detect([manager.context_blocks[0]], first_block_id=0)

    jobs = manager._nursery.jobs(  # noqa: SLF001 - assert job handoff metadata
        source="block_manager",
        kind="detect_blocks",
    )
    assert len(jobs) == 1
    assert jobs[0].mode == "best_effort"
    await manager.check_nursery(wait_for_all=True)


@pytest.mark.asyncio
async def test_check_nursery_does_not_wait_for_pending_detection() -> None:
    manager, _, _, _ = _manager()
    manager.context_blocks = _user_blocks("a")
    never = asyncio.Event()

    class _SlowDetector(_FakeDetector):
        async def maybe_detect(
            self, blocks: Sequence[IRBlock], first_block_id: int
        ) -> PreparedFork:
            async def _run() -> BlockDetectorResult:
                await never.wait()
                return BlockDetectorResult(completed=self.completed, still_buffered=[])

            return PreparedFork(coro=_run(), fork_id="detector_slow")

    manager._detector = cast(  # noqa: SLF001 - test swaps collaborator
        Any,
        _SlowDetector(
            [IRSemanticBlockRange(title="First", start_block=0, end_block=0)]
        ),
    )

    await manager.maybe_detect([manager.context_blocks[0]], first_block_id=0)
    await asyncio.wait_for(manager.check_nursery(), timeout=0.1)

    assert manager.semantic_blocks == []
    await manager._nursery.shutdown(cancel=True)  # noqa: SLF001 - cleanup pending job


def test_semantic_block_range_rejects_inverted_bounds() -> None:
    with pytest.raises(ValueError, match="start_block"):
        IRSemanticBlockRange(title="Bad", start_block=2, end_block=1)


@pytest.mark.asyncio
async def test_block_metrics_update_full_mode_toks_and_full_toks() -> None:
    manager, recorder, _, _ = _manager()
    manager.context_blocks = _user_blocks("a")
    block = _semantic_block(idx=0, start=0, end=0, title="First")
    manager.semantic_blocks = [block]
    count = _count(17)

    async def _count_job() -> IRTokenRangeCount:
        return count

    manager._nursery.submit(  # noqa: SLF001 - exercise boundary integration
        _count_job(),
        kind="block_metrics",
        source="block_manager",
        metadata={"block_id": block.id, "block_idx": block.idx},
    )

    await _settle()
    await manager.check_nursery()

    counted = manager.semantic_blocks[0]
    assert counted.toks == count
    assert counted.full_toks == count
    assert recorder.metrics == [(count, block.id)]


@pytest.mark.asyncio
async def test_completed_detection_starts_summary_generation() -> None:
    manager, recorder, footer, _ = _manager()
    manager.context_blocks = _user_blocks("a")
    summarizer = _FakeSummarizer()
    manager._detector = cast(  # noqa: SLF001 - test swaps collaborator
        Any,
        _FakeDetector(
            [IRSemanticBlockRange(title="First", start_block=0, end_block=0)]
        ),
    )
    manager._summarizer = cast(Any, summarizer)  # noqa: SLF001 - test swaps collaborator

    await manager.maybe_detect([manager.context_blocks[0]], first_block_id=0)
    await _settle()
    await manager.check_nursery()
    await _settle()
    await manager.check_nursery()

    assert [block.title for block in manager.semantic_blocks] == ["First"]
    assert [block.title for block in recorder.semantic_blocks] == ["First"]
    assert footer.queued[0]["text"] == 'New block crystallized: "First"'
    assert summarizer.calls == [manager.semantic_blocks[0].id]
    assert manager.semantic_blocks[0].available_modes == ["full", "summary"]


@pytest.mark.asyncio
async def test_generated_summary_adds_summary_available_mode(tmp_path: Path) -> None:
    blocks = _user_blocks("a")
    semantic_blocks = [
        _semantic_block(idx=0, start=0, end=0, title="First"),
    ]
    manager = _rehydrate_manager(
        tmp_path,
        blocks=blocks,
        semantic_blocks=semantic_blocks,
    )
    summarizer = _FakeSummarizer()
    manager._summarizer = cast(Any, summarizer)  # noqa: SLF001 - test swaps collaborator

    await manager.generate_next_summary()
    await _settle()
    await manager.check_nursery()

    assert manager.semantic_blocks[0].available_modes == ["full", "summary"]
    assert "summmary" not in manager.semantic_blocks[0].available_modes
    assert manager.semantic_blocks[0].artifacts[0].headline == "Summary for First"
    assert manager.semantic_blocks[0].artifacts[0].toks == IRTokenRangeCount(
        tokens=10,
        method="api",
        exact=True,
    )
    assert summarizer.integrated_forks == ["summarizer_0"]
    assert summarizer.calls == [manager.semantic_blocks[0].id]


@pytest.mark.asyncio
async def test_generate_next_summary_does_not_duplicate_in_flight_summary(
    tmp_path: Path,
) -> None:
    blocks = _user_blocks("a")
    semantic_blocks = [
        _semantic_block(idx=0, start=0, end=0, title="First"),
    ]
    manager = _rehydrate_manager(
        tmp_path,
        blocks=blocks,
        semantic_blocks=semantic_blocks,
    )
    never = asyncio.Event()

    class _SlowSummarizer(_FakeSummarizer):
        async def summarize(
            self,
            *,
            semantic_block: IRSemanticBlock,
            context_block_slice: list[IRBlock],
            prev_semantic_blocks: list[IRSemanticBlock],
        ) -> PreparedFork:
            self.calls.append(semantic_block.id)

            async def _run() -> BlockSummarizerResult:
                await never.wait()
                return BlockSummarizerResult(summary=_summary())

            return PreparedFork(coro=_run(), fork_id=f"summarizer_{semantic_block.idx}")

    summarizer = _SlowSummarizer()
    manager._summarizer = cast(Any, summarizer)  # noqa: SLF001 - test swaps collaborator

    await manager.generate_next_summary()
    await manager.generate_next_summary()

    assert summarizer.calls == [manager.semantic_blocks[0].id]
    never.set()
    await manager.check_nursery(wait_for_all=True)


def test_forget_block_compacts_to_summary_and_records_mode() -> None:
    manager, recorder, _, _ = _manager()
    full_count = _count(40)
    summary_count = _count(7)
    block = _semantic_block(idx=0, start=0, end=0, title="First").model_copy(
        update={
            "toks": full_count,
            "full_toks": full_count,
            "available_modes": ["full", "summary"],
            "artifacts": [_summary(toks=summary_count)],
        }
    )
    manager.context_blocks = _user_blocks("full text")
    manager.semantic_blocks = [block]

    manager.forget_block(0, confirm=False)

    compacted = manager.semantic_blocks[0]
    assert compacted.mode == "summary"
    assert compacted.toks == summary_count
    assert recorder.applied_modes == [("summary", block.id)]
    rendered = manager.render_block(idx=0)
    assert len(rendered) == 1
    assert isinstance(rendered[0], IRUserTextBlock)
    assert rendered[0].origin == "memory"
    assert "Summary headline" in rendered[0].text


def test_forget_block_requires_existing_summary_artifact() -> None:
    manager, recorder, _, _ = _manager()
    block = _semantic_block(idx=0, start=0, end=0, title="First")
    manager.context_blocks = _user_blocks("full text")
    manager.semantic_blocks = [block]

    with pytest.raises(ValueError, match="no summary artifact"):
        manager.forget_block(0, confirm=False)

    assert recorder.applied_modes == []


def test_recall_summary_block_returns_full_context_with_original_ids() -> None:
    manager, _, _, _ = _manager()
    first = _semantic_block(idx=0, start=0, end=1, title="First")
    second = _semantic_block(idx=1, start=2, end=3, title="Second").model_copy(
        update={
            "mode": "summary",
            "available_modes": ["full", "summary"],
            "artifacts": [_summary()],
        }
    )
    manager.context_blocks = _user_blocks(
        "first a",
        "first b",
        "second <detail> a",
        "second b",
    )
    manager.semantic_blocks = [first, second]

    result = manager.recall_block(1)

    assert '# Block 1 - "Second"' in result
    assert '<context_block id="2">' in result
    assert '<context_block id="3">' in result
    assert "second &lt;detail&gt; a" in result
    assert "second b" in result
    assert '<context_block id="0">' not in result
    assert "first a" not in result


def test_recall_block_does_not_mutate_mode_or_record_apply_mode() -> None:
    manager, recorder, _, _ = _manager()
    summary_count = _count(7)
    block = _semantic_block(idx=0, start=0, end=0, title="First").model_copy(
        update={
            "mode": "summary",
            "toks": summary_count,
            "available_modes": ["full", "summary"],
            "artifacts": [_summary(toks=summary_count)],
        }
    )
    manager.context_blocks = _user_blocks("full text")
    manager.semantic_blocks = [block]

    result = manager.recall_block(0)

    assert "full text" in result
    assert manager.semantic_blocks[0] == block
    assert recorder.applied_modes == []


def test_recall_full_block_errors_without_recording_mode_change() -> None:
    manager, recorder, _, _ = _manager()
    block = _semantic_block(idx=0, start=0, end=0, title="First")
    manager.context_blocks = _user_blocks("full text")
    manager.semantic_blocks = [block]

    with pytest.raises(ValueError, match="already entirely in context"):
        manager.recall_block(0)

    assert manager.semantic_blocks[0] == block
    assert recorder.applied_modes == []


def test_pin_full_block_records_pin_without_rerendering() -> None:
    manager, recorder, _, _ = _manager()
    block = _semantic_block(idx=0, start=0, end=0, title="First")
    manager.context_blocks = _user_blocks("full text")
    manager.semantic_blocks = [block]

    should_rerender = manager.pin_block(0, "This is important.")

    pinned = manager.semantic_blocks[0]
    assert should_rerender is False
    assert pinned.mode == "full"
    assert pinned.pin is not None
    assert pinned.pin.reason == "This is important."
    assert recorder.pins == [(pinned.pin, block.id)]
    assert recorder.applied_modes == []


def test_pin_summary_block_restores_full_mode_and_records_pin() -> None:
    manager, recorder, _, _ = _manager()
    full_count = _count(40)
    summary_count = _count(7)
    block = _semantic_block(idx=0, start=0, end=0, title="First").model_copy(
        update={
            "mode": "summary",
            "toks": summary_count,
            "full_toks": full_count,
            "available_modes": ["full", "summary"],
            "artifacts": [_summary(toks=summary_count)],
        }
    )
    manager.context_blocks = _user_blocks("full text")
    manager.semantic_blocks = [block]

    should_rerender = manager.pin_block(0, "Needs exact wording.")

    pinned = manager.semantic_blocks[0]
    assert should_rerender is True
    assert pinned.mode == "full"
    assert pinned.toks == full_count
    assert pinned.pin is not None
    assert pinned.pin.reason == "Needs exact wording."
    assert recorder.applied_modes == [("full", block.id)]
    assert recorder.pins == [(pinned.pin, block.id)]


def test_pin_summary_facet_records_facet_pin_without_mode_change() -> None:
    manager, recorder, _, _ = _manager()
    summary_count = _count(7)
    facet = IRSemanticBlockFacet(
        id="facet_design",
        title="Design moment",
        description="They found the shape.",
        start_block=0,
        end_block=1,
        resources=[],
    )
    block = _semantic_block(idx=0, start=0, end=1, title="First").model_copy(
        update={
            "mode": "summary",
            "toks": summary_count,
            "available_modes": ["full", "summary"],
            "artifacts": [_summary(toks=summary_count, facets=[facet])],
        }
    )
    manager.context_blocks = _user_blocks("a", "b")
    manager.semantic_blocks = [block]

    should_rerender = manager.pin_facet(0, "facet_design", "This was the turn.")

    pinned = manager.semantic_blocks[0]
    assert should_rerender is True
    assert pinned.mode == "summary"
    assert pinned.facet_pins == [
        IRSemanticBlockPin(
            kind="facet",
            reason="This was the turn.",
            facet_id="facet_design",
        )
    ]
    assert recorder.applied_modes == []
    assert recorder.pins == [(pinned.facet_pins[0], block.id)]


def test_pin_unknown_summary_facet_errors_without_recording() -> None:
    manager, recorder, _, _ = _manager()
    facet = IRSemanticBlockFacet(
        id="facet_design",
        title="Design moment",
        description="They found the shape.",
        start_block=0,
        end_block=0,
        resources=[],
    )
    block = _semantic_block(idx=0, start=0, end=0, title="First").model_copy(
        update={
            "available_modes": ["full", "summary"],
            "artifacts": [_summary(facets=[facet])],
        }
    )
    manager.context_blocks = _user_blocks("a")
    manager.semantic_blocks = [block]

    with pytest.raises(ValueError, match="no summary facet"):
        manager.pin_facet(0, "facet_missing", "Nope.")

    assert manager.semantic_blocks[0] == block
    assert recorder.pins == []


def test_render_summary_with_pinned_facet_inlines_original_context_blocks() -> None:
    manager, _, _, _ = _manager()
    context_blocks: list[IRBlock] = [
        _user("Ryan asks the key question."),
        IRAssistantTextBlock(text="The answer lands.", origin="model"),
        _user("A later unpinned note."),
    ]
    pinned_facet = IRSemanticBlockFacet(
        id="facet_decision",
        title="Decision moment",
        description="The exact exchange should stay vivid.",
        start_block=0,
        end_block=1,
        resources=[],
    )
    unpinned_facet = IRSemanticBlockFacet(
        id="facet_followup",
        title="Follow-up",
        description="A later note can stay summarized.",
        start_block=2,
        end_block=2,
        resources=[],
    )
    pin = IRSemanticBlockPin(
        kind="facet",
        reason="Keep the original exchange.",
        facet_id="facet_decision",
    )
    block = _semantic_block(idx=0, start=0, end=2, title="First").model_copy(
        update={
            "mode": "summary",
            "available_modes": ["full", "summary"],
            "artifacts": [_summary(facets=[pinned_facet, unpinned_facet])],
            "facet_pins": [pin],
        }
    )
    manager.context_blocks = context_blocks
    manager.semantic_blocks = [block]

    rendered = manager.render_block(idx=0)

    assert rendered[1:3] == context_blocks[0:2]
    assert isinstance(rendered[0], IRUserTextBlock)
    assert (
        'Pinned facets follow as original conversation: "Decision moment"'
        in rendered[0].text
    )
    assert "- Follow-up (blocks 2-2)" in rendered[0].text
    assert "- Decision moment (blocks 0-1)" not in rendered[0].text
    assert isinstance(rendered[-1], IRUserTextBlock)
    assert "End of pinned facets." in rendered[-1].text


def test_render_pinned_facet_expands_to_include_matching_tool_pair() -> None:
    manager, _, _, _ = _manager()
    context_blocks: list[IRBlock] = [
        _user("Please inspect the file."),
        IRToolCallBlock(call_id="toolu_1", tool="Read", input={"file_path": "a.py"}),
        IRToolResultBlock(
            call_id="toolu_1",
            tool="Read",
            content=[IRToolTextBlock(text="file contents")],
        ),
        IRAssistantTextBlock(text="The file matters.", origin="model"),
    ]
    facet = IRSemanticBlockFacet(
        id="facet_tool",
        title="Tool evidence",
        description="The evidence should keep its tool pair.",
        start_block=2,
        end_block=2,
        resources=[],
    )
    pin = IRSemanticBlockPin(
        kind="facet",
        reason="The tool result needs its call.",
        facet_id="facet_tool",
    )
    block = _semantic_block(idx=0, start=0, end=3, title="Tool block").model_copy(
        update={
            "mode": "summary",
            "available_modes": ["full", "summary"],
            "artifacts": [_summary(facets=[facet])],
            "facet_pins": [pin],
        }
    )
    manager.context_blocks = context_blocks
    manager.semantic_blocks = [block]

    rendered = manager.render_block(idx=0)

    assert rendered[1:3] == context_blocks[1:3]
    assert isinstance(rendered[1], IRToolCallBlock)
    assert isinstance(rendered[2], IRToolResultBlock)


def test_forget_pinned_block_requires_confirm() -> None:
    manager, recorder, _, _ = _manager()
    summary_count = _count(7)
    block = _semantic_block(idx=0, start=0, end=0, title="First").model_copy(
        update={
            "available_modes": ["full", "summary"],
            "artifacts": [_summary(toks=summary_count)],
            "pin": IRSemanticBlockPin(kind="block", reason="Keep this vivid."),
        }
    )
    manager.context_blocks = _user_blocks("full text")
    manager.semantic_blocks = [block]

    with pytest.raises(ValueError, match="currently pinned"):
        manager.forget_block(0, confirm=False)

    assert manager.semantic_blocks[0].mode == "full"
    assert recorder.applied_modes == []


def test_forget_pinned_block_with_confirm_compacts() -> None:
    manager, recorder, _, _ = _manager()
    summary_count = _count(7)
    pin = IRSemanticBlockPin(kind="block", reason="Keep this vivid.")
    block = _semantic_block(idx=0, start=0, end=0, title="First").model_copy(
        update={
            "available_modes": ["full", "summary"],
            "artifacts": [_summary(toks=summary_count)],
            "pin": pin,
        }
    )
    manager.context_blocks = _user_blocks("full text")
    manager.semantic_blocks = [block]

    manager.forget_block(0, confirm=True)

    compacted = manager.semantic_blocks[0]
    assert compacted.mode == "summary"
    assert compacted.toks == summary_count
    assert compacted.pin == pin
    assert recorder.applied_modes == [("summary", block.id)]


@pytest.mark.asyncio
async def test_stale_summary_result_is_discarded_but_fork_is_shutdown(
    tmp_path: Path,
) -> None:
    blocks = _user_blocks("a")
    semantic_blocks = [
        _semantic_block(idx=0, start=0, end=0, title="First"),
    ]
    manager = _rehydrate_manager(
        tmp_path,
        blocks=blocks,
        semantic_blocks=semantic_blocks,
    )
    summarizer = _FakeSummarizer()
    manager._summarizer = cast(Any, summarizer)  # noqa: SLF001 - test swaps collaborator

    await manager.generate_next_summary()
    stale_summary = _summary(headline="Already summarized")
    manager.semantic_blocks[0] = manager.semantic_blocks[0].model_copy(
        update={
            "artifacts": [stale_summary],
            "available_modes": ["full", "summary"],
        }
    )
    await _settle()
    await manager.check_nursery()

    assert [artifact.headline for artifact in manager.semantic_blocks[0].artifacts] == [
        "Already summarized"
    ]
    assert summarizer.integrated_forks == ["summarizer_0"]

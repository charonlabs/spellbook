from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

from spellbook.backends.model_backend import RequestSurface, TokenCounter
from spellbook.config import HomunculusConfig, SpellbookConfig
from spellbook.footer import FooterController
from spellbook.fork import BlockDetectorResult, ForkRunner
from spellbook.homunculus import Homunculus
from spellbook.homunculus.common import render_context_block
from spellbook.inbound import InboundMessageQueue
from spellbook.ir_types import (
    IRBlock,
    IRGeneration,
    IRSemanticBlock,
    IRSemanticBlockApplyModeRecord,
    IRSemanticBlockPairNarrative,
    IRSemanticBlockPairNarrativeChild,
    IRSemanticBlockPin,
    IRSemanticBlockRange,
    IRSemanticBlockSummary,
    IRSkillCatalog,
    IRTokenRangeCount,
    IRToolTextBlock,
    IRUsage,
    IRUserTextBlock,
    SemanticBlockApplyModeSource,
    SemanticBlockMode,
)
from spellbook.nursery import Nursery
from spellbook.recorder import Recorder
from spellbook.rehydrator import Rehydrator
from spellbook.session_lifecycle import DreamingOutcome
from spellbook.tools.common import ToolMetadata
from spellbook.tools.registry import DEFAULT_TOOL_REGISTRY
from spellbook.tools.sleep import SleepExecutionError, SleepInput, exec_sleep

pytestmark = pytest.mark.asyncio


class _TokenCounter:
    async def count_block_content(self, block: IRBlock) -> int | None:
        return _block_size(block)

    async def count_blocks(self, blocks: list[IRBlock]) -> int | None:
        return sum(_block_size(block) for block in blocks)

    async def count_frame(self) -> int | None:
        return 0

    async def count_surface(self, surface: RequestSurface) -> int | None:
        return None


class _DreamingRuntime:
    def __init__(self) -> None:
        self.state = "running"
        self.events: list[tuple[str, DreamingOutcome | None]] = []

    async def enter_dreaming(self) -> None:
        assert self.state == "running"
        self.state = "dreaming"
        self.events.append(("enter", None))

    async def exit_dreaming(self, outcome: DreamingOutcome) -> None:
        assert self.state == "dreaming"
        self.state = "running"
        self.events.append(("exit", outcome))


@dataclass
class _World:
    homunculus: Homunculus
    recorder: Recorder
    transcript: Path
    runtime: _DreamingRuntime

    @property
    def meta(self) -> ToolMetadata:
        return ToolMetadata(
            cwd=self.transcript.parent,
            transcript_path=self.transcript,
            homunculus=self.homunculus,
            dreaming_runtime=self.runtime,
        )


def _count(tokens: int) -> IRTokenRangeCount:
    return IRTokenRangeCount(tokens=tokens, method="api", exact=True)


def _summary(idx: int, tokens: int = 10) -> IRSemanticBlockSummary:
    return IRSemanticBlockSummary(
        headline=f"Summary {idx}",
        text=f"What mattered in block {idx}.",
        facets=[],
        open_thread=None,
        toks=_count(tokens),
    )


def _block(
    idx: int,
    *,
    summary: bool = True,
    pinned: bool = False,
) -> IRSemanticBlock:
    artifact = _summary(idx) if summary else None
    return IRSemanticBlock(
        id=f"block_{idx}",
        idx=idx,
        title=f"Block {idx}",
        range=IRSemanticBlockRange(
            id=f"range_{idx}",
            title=f"Block {idx}",
            start_block=idx,
            end_block=idx,
            completed=True,
        ),
        toks=_count(100 + idx * 10),
        full_toks=_count(100 + idx * 10),
        available_modes=["full", "summary"] if artifact is not None else ["full"],
        artifacts=[artifact] if artifact is not None else [],
        pin=(
            IRSemanticBlockPin(kind="block", reason="Keep this exact exchange.")
            if pinned
            else None
        ),
    )


def _add_pair(blocks: list[IRSemanticBlock], first: int = 0) -> None:
    second = first + 1
    chapter = first // 2 + 1
    narrative_id = f"narrative_{chapter}"
    parent = IRSemanticBlockPairNarrative(
        narrative_id=narrative_id,
        pair=(first, second),
        chapter_number=chapter,
        title=f"Chapter {chapter}",
        blocks=[IRUserTextBlock(text="Woven memory.", origin="memory")],
        toks=_count(30),
        source_chapter_path=f"dreams/chapter-{chapter:02d}.md",
        compiled_json_path=f"dreams/chapter-{chapter:02d}.compiled.json",
    )
    child = IRSemanticBlockPairNarrativeChild(
        narrative_id=narrative_id,
        parent_block_idx=first,
        parent_block_id=blocks[first].id,
        pair=(first, second),
        chapter_number=chapter,
    )
    for idx, artifact in ((first, parent), (second, child)):
        block = blocks[idx]
        blocks[idx] = block.model_copy(
            update={
                "available_modes": [*block.available_modes, "pair_narrative"],
                "artifacts": [*block.artifacts, artifact],
            }
        )


async def _build_world(
    tmp_path: Path,
    semantic_blocks: list[IRSemanticBlock],
    *,
    sleep_enabled: bool = False,
    homunculus_config: HomunculusConfig | None = None,
) -> _World:
    transcript = tmp_path / "transcript.jsonl"
    config = SpellbookConfig(
        cwd=tmp_path,
        sleep_enabled=sleep_enabled,
        hom_config=homunculus_config or HomunculusConfig(soft_threshold=500),
    )
    recorder = Recorder(config, transcript, "session_sleep", DEFAULT_TOOL_REGISTRY)
    recorder.write_session_record(skill_catalog=IRSkillCatalog())
    context_blocks = [
        IRUserTextBlock(
            text=f"Full source {idx}: " + "detail " * 80,
            origin="human",
        )
        for idx in range(len(semantic_blocks))
    ]
    recorder.start_turn("turn_1", context_blocks)
    recorder.detect_blocks(
        BlockDetectorResult(
            completed=[block.range for block in semantic_blocks],
            still_buffered=[],
        )
    )
    for block in semantic_blocks:
        recorder.write_semantic_block(block)
        for artifact in block.artifacts:
            recorder.write_block_artifact(artifact, block.id)
        if block.pin is not None:
            recorder.apply_block_pin(block.pin, block.id)
    recorder.end_turn()
    recorder.start_turn("turn_2", [])

    rehydrated = Rehydrator(transcript).run()
    footer = FooterController(
        inbound_queue=InboundMessageQueue(),
        recorder=recorder,
    )
    homunculus = Homunculus(
        config=config.hom_config,
        footer_c=footer,
        recorder=recorder,
        token_counter=cast(TokenCounter, _TokenCounter()),
        nursery=Nursery(config=config),
        fork_runner=cast(ForkRunner, object()),
        sleep_enabled=sleep_enabled,
    )
    await homunculus.rehydrate(rehydrated)
    return _World(
        homunculus=homunculus,
        recorder=recorder,
        transcript=transcript,
        runtime=_DreamingRuntime(),
    )


def _result_text(result: Any) -> str:
    content = result.content[0]
    assert isinstance(content, IRToolTextBlock)
    return content.text


def _render_size(blocks: list[IRBlock]) -> int:
    return sum(_block_size(block) for block in blocks)


def _block_size(block: IRBlock) -> int:
    text = getattr(block, "text", None)
    return len(text) if isinstance(text, str) else len(render_context_block(block))


async def test_sleep_lands_actual_modes_manifest_debts_and_smaller_render(
    tmp_path: Path,
) -> None:
    semantic_blocks = [_block(idx) for idx in range(8)]
    _add_pair(semantic_blocks)
    semantic_blocks[2] = _block(2, pinned=True)
    world = await _build_world(tmp_path, semantic_blocks)
    before_render = await world.homunculus.render_context([])

    result = await exec_sleep(world.meta, SleepInput())

    after_render = await world.homunculus.maybe_rerender()
    assert after_render is not None
    assert _render_size(after_render) < _render_size(before_render)
    modes = [block.mode for block in world.homunculus.build_awareness().semantic_blocks]
    assert modes == [
        "pair_narrative",
        "pair_narrative",
        "full",
        "summary",
        "full",
        "full",
        "full",
        "full",
    ]
    text = _result_text(result)
    assert "Deltas\n" in text
    assert "Debts\n" in text
    assert "Policy\n- Target: calm below 500 tokens" in text
    assert "projected result 780 tokens" in text
    assert "kept 4 recent blocks full" in text
    assert 'Block 0 "Block 0": full -> narrative (chapter 1)' in text
    assert "because its block pin is absolute" in text
    assert "The dream itself is kept, if you ever want to hold it: forks/." in text
    assert result.display["status"] == "completed"
    assert len(result.display["deltas"]) == 3
    assert result.display["debts"] == ["pinned"]
    assert result.display["projection"] == {
        "target_tokens": 500,
        "projected_render_tokens": 780,
        "kept_full_blocks": 4,
        "estimate_quality": "conservative",
        "outcome": "floor_reached",
    }
    assert world.runtime.events == [("enter", None), ("exit", "completed")]

    rehydrated = Rehydrator(world.transcript).run()
    records = rehydrated.records
    mode_records = [
        record
        for record in records
        if isinstance(record, IRSemanticBlockApplyModeRecord)
    ]
    assert [(record.block_id, record.mode) for record in mode_records] == [
        ("block_0", "pair_narrative"),
        ("block_1", "pair_narrative"),
        ("block_3", "summary"),
    ]
    assert all(record.source == "model" for record in mode_records)


async def test_sleep_refusal_names_missing_summary_and_mutates_nothing(
    tmp_path: Path,
) -> None:
    semantic_blocks = [_block(idx) for idx in range(6)]
    semantic_blocks[1] = _block(1, summary=False)
    world = await _build_world(tmp_path, semantic_blocks)
    before = world.transcript.read_bytes()

    result = await exec_sleep(world.meta, SleepInput())

    assert world.transcript.read_bytes() == before
    assert all(
        block.mode == "full"
        for block in world.homunculus.build_awareness().semantic_blocks
    )
    assert "whole frontier advance was refused rather than guessing" in _result_text(
        result
    )
    assert result.display["status"] == "refused"
    assert result.display["deltas"] == []
    assert result.display["debts"] == ["missing_summary"]
    assert world.runtime.events == [("enter", None), ("exit", "refused")]


async def test_sleep_keeps_block_pins_at_full_resolution(tmp_path: Path) -> None:
    semantic_blocks = [_block(idx) for idx in range(5)]
    semantic_blocks[0] = _block(0, pinned=True)
    world = await _build_world(tmp_path, semantic_blocks)
    before = world.transcript.read_bytes()

    result = await exec_sleep(world.meta, SleepInput())

    assert world.transcript.read_bytes() == before
    assert world.homunculus.build_awareness().semantic_blocks[0].mode == "full"
    assert result.display["deltas"] == []
    assert result.display["debts"] == ["pinned"]


async def test_sleep_pair_preflight_failure_lands_neither_half(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    semantic_blocks = [_block(idx) for idx in range(6)]
    _add_pair(semantic_blocks)
    world = await _build_world(tmp_path, semantic_blocks)
    manager = cast(Any, world.homunculus)._block_manager
    original_prepare = manager._prepare_mode_update

    def fail_between_pair_halves(
        block: IRSemanticBlock,
        mode: SemanticBlockMode,
        *,
        queue_summary_refusal: bool,
    ) -> IRSemanticBlock:
        prepared = original_prepare(
            block,
            mode,
            queue_summary_refusal=queue_summary_refusal,
        )
        if block.idx == 1:
            raise RuntimeError("induced failure between parent and child")
        return prepared

    monkeypatch.setattr(manager, "_prepare_mode_update", fail_between_pair_halves)
    before = world.transcript.read_bytes()

    with pytest.raises(SleepExecutionError, match="No block modes moved"):
        await exec_sleep(world.meta, SleepInput())

    assert world.transcript.read_bytes() == before
    assert [
        block.mode for block in world.homunculus.build_awareness().semantic_blocks[:2]
    ] == ["full", "full"]
    assert world.runtime.events == [("enter", None), ("exit", "failed")]


async def test_sleep_dry_run_returns_plan_and_forecast_without_entering_dreaming(
    tmp_path: Path,
) -> None:
    world = await _build_world(tmp_path, [_block(idx) for idx in range(6)])
    before = world.transcript.read_bytes()

    result = await exec_sleep(world.meta, SleepInput(dry_run=True))

    assert world.transcript.read_bytes() == before
    assert world.runtime.events == []
    assert all(
        block.mode == "full"
        for block in world.homunculus.build_awareness().semantic_blocks
    )
    text = _result_text(result)
    assert "Sleep dry run" in text
    assert "Sleep: ~seconds" in text
    assert "Target: calm below 500 tokens" in text
    assert "projected result 560 tokens" in text
    assert "kept 4 recent blocks full" in text
    assert 'Block 0 "Block 0": full -> summary' in text
    assert result.display["status"] == "preview"
    assert result.display["dry_run"] is True
    assert len(result.display["transitions"]) == 2
    assert result.display["projection"] == {
        "target_tokens": 500,
        "current_render_tokens": 750,
        "projected_render_tokens": 560,
        "estimated_tokens_freed": 190,
        "kept_full_blocks": 4,
        "estimate_quality": "conservative",
        "outcome": "floor_reached",
    }


async def test_sleep_partial_failure_reports_only_already_landed_deltas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = await _build_world(tmp_path, [_block(idx) for idx in range(6)])
    original_apply = world.recorder.apply_semantic_block_modes
    calls = 0

    def fail_second_group(
        applications: Sequence[tuple[SemanticBlockMode, str]],
        *,
        source: SemanticBlockApplyModeSource,
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("induced append failure")
        original_apply(applications, source=source)

    monkeypatch.setattr(
        world.recorder,
        "apply_semantic_block_modes",
        fail_second_group,
    )

    with pytest.raises(SleepExecutionError) as raised:
        await exec_sleep(world.meta, SleepInput())

    message = str(raised.value)
    assert 'Block 0 "Block 0": full -> summary' in message
    assert 'Block 1 "Block 1" did not make its planned full -> summary' in message
    assert [
        block.mode for block in world.homunculus.build_awareness().semantic_blocks[:2]
    ] == ["summary", "full"]
    rehydrated = Rehydrator(world.transcript).run()
    records = rehydrated.records
    mode_records = [
        record
        for record in records
        if isinstance(record, IRSemanticBlockApplyModeRecord)
    ]
    assert [(record.block_id, record.mode) for record in mode_records] == [
        ("block_0", "summary")
    ]
    assert world.runtime.events == [("enter", None), ("exit", "failed")]


async def test_forced_sleep_uses_planner_source_manifest_history_and_preserves_pin(
    tmp_path: Path,
) -> None:
    semantic_blocks = [_block(idx) for idx in range(8)]
    semantic_blocks[0] = _block(0, pinned=True)
    world = await _build_world(
        tmp_path,
        semantic_blocks,
        sleep_enabled=True,
    )

    await world.homunculus.integrate_generation(
        IRGeneration(
            model="test-model",
            blocks=[],
            stop_reason="end_turn",
            usage=IRUsage(input_tokens=900_000),
        )
    )
    world.homunculus.check_sleep_pressure()
    assert world.homunculus.take_forced_sleep_plan() is None
    planner_footers = [
        footer
        for footer in world.homunculus._footer_c.peek_pending()  # noqa: SLF001
        if footer.source == "planner"
    ]
    assert len(planner_footers) == 1
    assert "Sleep now:" in planner_footers[0].text
    assert "Sleep pre-warning: at 95%" in planner_footers[0].text

    await world.homunculus.integrate_generation(
        IRGeneration(
            model="test-model",
            blocks=[],
            stop_reason="end_turn",
            usage=IRUsage(input_tokens=950_000),
        )
    )
    world.homunculus.check_sleep_pressure()
    forced = world.homunculus.take_forced_sleep_plan()

    assert forced is not None
    world.recorder.end_turn()
    manifest = world.homunculus.execute_forced_sleep(forced)

    assert manifest.forced is True
    assert manifest.prewarning_tokens == 900_000
    assert manifest.floor_tokens == 950_000
    assert "forced=true" in manifest.render()
    assert "Pre-warning history: issued at 900,000 tokens" in manifest.render()
    assert world.homunculus.build_awareness().semantic_blocks[0].mode == "full"

    rehydrated = Rehydrator(world.transcript).run()
    records = rehydrated.records
    mode_records = [
        record
        for record in records
        if isinstance(record, IRSemanticBlockApplyModeRecord)
    ]
    assert mode_records
    assert all(record.source == "planner" for record in mode_records)
    assert rehydrated.semantic_blocks[0].mode == "full"
    assert any(
        "forced=true" in footer.text for footer in rehydrated.pending_footers.values()
    )


async def test_forced_sleep_empty_frontier_stands_down_once_to_existing_warning(
    tmp_path: Path,
) -> None:
    world = await _build_world(
        tmp_path,
        [_block(idx) for idx in range(4)],
        sleep_enabled=True,
    )

    await world.homunculus.integrate_generation(
        IRGeneration(
            model="test-model",
            blocks=[],
            stop_reason="end_turn",
            usage=IRUsage(input_tokens=900_000),
        )
    )
    world.homunculus.check_sleep_pressure()
    await world.homunculus.integrate_generation(
        IRGeneration(
            model="test-model",
            blocks=[],
            stop_reason="end_turn",
            usage=IRUsage(input_tokens=950_000),
        )
    )
    world.homunculus.check_sleep_pressure()

    assert world.homunculus.take_forced_sleep_plan() is None
    assert world.homunculus.take_forced_sleep_plan() is None
    assert all(
        block.mode == "full"
        for block in world.homunculus.build_awareness().semantic_blocks
    )
    pending = world.homunculus._footer_c.peek_pending()  # noqa: SLF001
    assert any(footer.type == "gas_gauge" for footer in pending)
    records = Rehydrator(world.transcript).run().records
    assert not any(
        isinstance(record, IRSemanticBlockApplyModeRecord) for record in records
    )


async def test_sleep_disabled_preserves_existing_pressure_behavior(
    tmp_path: Path,
) -> None:
    world = await _build_world(tmp_path, [_block(idx) for idx in range(6)])

    for input_tokens in (900_000, 950_000):
        await world.homunculus.integrate_generation(
            IRGeneration(
                model="test-model",
                blocks=[],
                stop_reason="end_turn",
                usage=IRUsage(input_tokens=input_tokens),
            )
        )
        world.homunculus.check_sleep_pressure()

    assert world.homunculus.take_forced_sleep_plan() is None
    pending = world.homunculus._footer_c.peek_pending()  # noqa: SLF001
    assert all(footer.source != "planner" for footer in pending)
    records = Rehydrator(world.transcript).run().records
    assert not any(
        isinstance(record, IRSemanticBlockApplyModeRecord) for record in records
    )

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import TypeAdapter

from scripts.apply_initial_block_modes import apply_initial_block_modes
from spellbook.config import SpellbookConfig
from spellbook.fork import BlockDetectorResult
from spellbook.ir_types import (
    IRRecord,
    IRSemanticBlock,
    IRSemanticBlockApplyModeRecord,
    IRSemanticBlockRange,
    IRSemanticBlockSummary,
    IRSkillCatalog,
    IRTokenRangeCount,
    IRUserTextBlock,
)
from spellbook.recorder import Recorder
from spellbook.rehydrator import Rehydrator
from spellbook.tools.registry import DEFAULT_TOOL_REGISTRY


def _make_prepared_transcript(
    tmp_path: Path,
    *,
    block_count: int = 4,
    skip_summary_idx: int | None = None,
) -> tuple[Path, list[IRSemanticBlock]]:
    transcript = tmp_path / "transcript.jsonl"
    config = SpellbookConfig(model="claude-sonnet-4-6", cwd=tmp_path)
    recorder = Recorder(config, transcript, "session_test", DEFAULT_TOOL_REGISTRY)
    recorder.write_session_record(skill_catalog=IRSkillCatalog())

    for idx in range(block_count):
        recorder.start_turn(
            f"turn_{idx + 1}",
            [IRUserTextBlock(text=f"Block {idx} full text.", origin="human")],
        )
        recorder.end_turn()

    recorder.start_turn("turn_semantic", [])
    semantic_ranges: list[IRSemanticBlockRange] = []
    semantic_blocks: list[IRSemanticBlock] = []
    for idx in range(block_count):
        full_toks = IRTokenRangeCount(
            tokens=100 + idx,
            method="prefix_delta",
            exact=True,
        )
        semantic_range = IRSemanticBlockRange(
            title=f"Block {idx}",
            start_block=idx,
            end_block=idx,
            completed=True,
        )
        semantic_block = IRSemanticBlock(
            idx=idx,
            title=f"Block {idx}",
            range=semantic_range,
            toks=full_toks,
            full_toks=full_toks,
        )
        semantic_ranges.append(semantic_range)
        semantic_blocks.append(semantic_block)

    recorder.detect_blocks(
        BlockDetectorResult(completed=semantic_ranges, still_buffered=[])
    )
    for block in semantic_blocks:
        recorder.write_semantic_block(block)
        if block.idx == skip_summary_idx:
            continue
        summary_toks = IRTokenRangeCount(
            tokens=10 + block.idx,
            method="api",
            exact=True,
        )
        recorder.write_block_artifact(
            IRSemanticBlockSummary(
                headline=f"Summary {block.idx}",
                text=f"Summary text {block.idx}.",
                facets=[],
                open_thread=None,
                toks=summary_toks,
            ),
            block.id,
        )
    recorder.end_turn()
    return transcript, semantic_blocks


def _read_records(path: Path) -> list[IRRecord]:
    adapter = TypeAdapter(IRRecord)
    return [
        adapter.validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_apply_initial_block_modes_appends_summary_prefix(tmp_path: Path) -> None:
    transcript, semantic_blocks = _make_prepared_transcript(tmp_path)

    report = apply_initial_block_modes(
        transcript,
        keep_newest_full=1,
        apply=True,
        write_report=False,
    )

    assert report.dry_run is False
    assert report.target_summary_blocks == 3
    assert report.target_full_blocks == 1
    assert report.records_appended == 3
    assert report.skipped_unchanged == 1
    assert report.backup_path is not None
    assert report.backup_path.exists()

    rehydrated = Rehydrator(transcript).run()
    assert [block.mode for block in rehydrated.semantic_blocks] == [
        "summary",
        "summary",
        "summary",
        "full",
    ]

    records = _read_records(transcript)
    mode_records = [
        record
        for record in records
        if isinstance(record, IRSemanticBlockApplyModeRecord)
    ]
    assert [record.block_id for record in mode_records] == [
        block.id for block in semantic_blocks[:3]
    ]
    assert {record.source for record in mode_records} == {"planner"}


def test_apply_initial_block_modes_dry_run_does_not_mutate(tmp_path: Path) -> None:
    transcript, _semantic_blocks = _make_prepared_transcript(tmp_path)
    before = transcript.read_text(encoding="utf-8")

    report = apply_initial_block_modes(
        transcript,
        keep_newest_full=2,
        apply=False,
        write_report=False,
    )

    assert transcript.read_text(encoding="utf-8") == before
    assert report.dry_run is True
    assert report.records_appended == 0
    assert report.records_to_append == 2
    assert report.skipped_unchanged == 2


def test_apply_initial_block_modes_requires_summary_artifacts(tmp_path: Path) -> None:
    transcript, _semantic_blocks = _make_prepared_transcript(
        tmp_path,
        skip_summary_idx=0,
    )

    with pytest.raises(ValueError, match="summary artifacts"):
        apply_initial_block_modes(
            transcript,
            keep_newest_full=1,
            apply=True,
            write_report=False,
        )

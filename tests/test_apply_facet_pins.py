from __future__ import annotations

from pathlib import Path

import pytest

from scripts.apply_facet_pins import apply_facet_pins
from spellbook.config import SpellbookConfig
from spellbook.fork import BlockDetectorResult
from spellbook.ir_types import (
    IRSemanticBlock,
    IRSemanticBlockFacet,
    IRSemanticBlockPin,
    IRSemanticBlockPinRecord,
    IRSemanticBlockRange,
    IRSemanticBlockSummary,
    IRSkillCatalog,
    IRUserTextBlock,
)
from spellbook.recorder import Recorder
from spellbook.rehydrator import Rehydrator
from spellbook.tools.registry import DEFAULT_TOOL_REGISTRY


def _write_transcript(
    tmp_path: Path,
    *,
    facets: list[IRSemanticBlockFacet],
    existing_pins: list[IRSemanticBlockPin] | None = None,
) -> Path:
    transcript = tmp_path / "transcript.jsonl"
    config = SpellbookConfig(model="claude-sonnet-4-6", cwd=tmp_path)
    recorder = Recorder(config, transcript, "session_test", DEFAULT_TOOL_REGISTRY)
    recorder.write_session_record(skill_catalog=IRSkillCatalog())
    recorder.start_turn("turn_1", [IRUserTextBlock(text="hello", origin="human")])

    semantic_range = IRSemanticBlockRange(
        id="range_1",
        title="Completed block",
        start_block=0,
        end_block=0,
        completed=True,
    )
    semantic_block = IRSemanticBlock(
        id="block_1",
        idx=0,
        title="Completed block",
        range=semantic_range,
        toks=None,
        full_toks=None,
    )
    recorder.detect_blocks(
        BlockDetectorResult(completed=[semantic_range], still_buffered=[])
    )
    recorder.write_semantic_block(semantic_block)
    recorder.write_block_artifact(
        IRSemanticBlockSummary(
            id="summary_1",
            headline="Completed block",
            text="Summary text.",
            facets=facets,
            open_thread=None,
            toks=None,
        ),
        semantic_block.id,
    )
    for pin in existing_pins or []:
        recorder.apply_block_pin(pin, semantic_block.id)
    recorder.end_turn()
    return transcript


def _facet(facet_id: str, title: str) -> IRSemanticBlockFacet:
    return IRSemanticBlockFacet(
        id=facet_id,
        title=title,
        description=f"{title} description.",
        start_block=0,
        end_block=0,
        resources=[],
    )


def test_apply_facet_pins_resolves_prefix_and_appends_pin_record(
    tmp_path: Path,
) -> None:
    transcript = _write_transcript(
        tmp_path,
        facets=[
            _facet("facet_477e5c97abcdef", "Target facet"),
            _facet("facet_12345678abcdef", "Other facet"),
        ],
    )

    report = apply_facet_pins(
        transcript,
        ["facet_477e5c97"],
        reason="Keep this facet exact.",
        backup=False,
    )

    assert report.records_appended == 1
    assert report.resolved[0].facet_id == "facet_477e5c97abcdef"
    result = Rehydrator(transcript).run()
    assert result.semantic_blocks[0].facet_pins == [
        IRSemanticBlockPin(
            kind="facet",
            reason="Keep this facet exact.",
            facet_id="facet_477e5c97abcdef",
        )
    ]
    pin_records = [
        record for record in result.records if isinstance(record, IRSemanticBlockPinRecord)
    ]
    assert len(pin_records) == 1
    assert pin_records[0].turn == 1
    assert pin_records[0].turn_id == "turn_1"


def test_apply_facet_pins_dry_run_does_not_mutate(tmp_path: Path) -> None:
    transcript = _write_transcript(
        tmp_path,
        facets=[_facet("facet_477e5c97abcdef", "Target facet")],
    )
    before = transcript.read_text(encoding="utf-8")

    report = apply_facet_pins(
        transcript,
        ["facet_477e5c97"],
        dry_run=True,
    )

    assert report.records_appended == 0
    assert transcript.read_text(encoding="utf-8") == before
    assert Rehydrator(transcript).run().semantic_blocks[0].facet_pins == []


def test_apply_facet_pins_rejects_ambiguous_prefix(tmp_path: Path) -> None:
    transcript = _write_transcript(
        tmp_path,
        facets=[
            _facet("facet_477e5c97abcdef", "Target facet"),
            _facet("facet_477e5c97123456", "Neighbor facet"),
        ],
    )

    with pytest.raises(ValueError, match="ambiguous"):
        apply_facet_pins(transcript, ["facet_477e5c97"])


def test_apply_facet_pins_skips_duplicate_inputs_and_existing_pin(
    tmp_path: Path,
) -> None:
    existing_pin = IRSemanticBlockPin(
        kind="facet",
        reason="Already important.",
        facet_id="facet_477e5c97abcdef",
    )
    transcript = _write_transcript(
        tmp_path,
        facets=[_facet("facet_477e5c97abcdef", "Target facet")],
        existing_pins=[existing_pin],
    )
    before = transcript.read_text(encoding="utf-8")

    report = apply_facet_pins(
        transcript,
        ["facet_477e5c97", "facet_477e5c97"],
        backup=False,
    )

    assert report.records_appended == 0
    assert report.duplicate_inputs == ["facet_477e5c97"]
    assert report.resolved[0].already_pinned is True
    assert transcript.read_text(encoding="utf-8") == before

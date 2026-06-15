from __future__ import annotations

import json
from pathlib import Path

import pytest

from core_scripts.apply_dream_narratives import (
    _parse_chapters,
    apply_dream_narratives,
)
from core_scripts.merge_stats import _build_block_manager
from spellbook.config import SpellbookConfig
from spellbook.fork import BlockDetectorResult
from spellbook.ir_types import (
    IRAssistantTextBlock,
    IRBlock,
    IRSemanticBlock,
    IRSemanticBlockPairNarrative,
    IRSemanticBlockPairNarrativeChild,
    IRSemanticBlockRange,
    IRSkillCatalog,
    IRTokenRangeCount,
    IRToolCallBlock,
    IRToolResultBlock,
    IRToolTextBlock,
    IRUserTextBlock,
)
from spellbook.recorder import Recorder
from spellbook.rehydrator import Rehydrator
from spellbook.tools.registry import DEFAULT_TOOL_REGISTRY


def _count(tokens: int) -> IRTokenRangeCount:
    return IRTokenRangeCount(tokens=tokens, method="api", exact=True)


def _write_transcript(
    tmp_path: Path,
    *,
    block_count: int = 2,
    unfinished: bool = False,
) -> Path:
    transcript = tmp_path / "transcript.jsonl"
    config = SpellbookConfig(cwd=tmp_path, model="claude-opus-4-6")
    recorder = Recorder(config, transcript, "session_test", DEFAULT_TOOL_REGISTRY)
    recorder.write_session_record(skill_catalog=IRSkillCatalog())

    recorder.start_turn(
        "turn_1",
        [
            IRUserTextBlock(text=f"source block {idx}", origin="human")
            for idx in range(block_count)
        ],
    )
    recorder.end_turn()

    ranges = [
        IRSemanticBlockRange(
            title=f"Block {idx}",
            start_block=idx,
            end_block=idx,
            completed=True,
        )
        for idx in range(block_count)
    ]
    recorder.detect_blocks(BlockDetectorResult(completed=ranges, still_buffered=[]))
    for idx, semantic_range in enumerate(ranges):
        recorder.write_semantic_block(
            IRSemanticBlock(
                idx=idx,
                title=semantic_range.title,
                range=semantic_range,
                toks=_count(100 + idx),
                full_toks=_count(100 + idx),
            )
        )

    if unfinished:
        recorder.start_turn(
            "turn_2", [IRUserTextBlock(text="dangling", origin="human")]
        )

    return transcript


def _write_compiled(
    chapter_dir: Path,
    *,
    chapter: int = 1,
    pair: tuple[int, int] = (0, 1),
    token_count: int = 37,
    blocks: list[IRBlock] | None = None,
) -> Path:
    chapter_dir.mkdir(parents=True, exist_ok=True)
    source_path = chapter_dir / f"chapter-{chapter:02d}-test.md"
    source_path.write_text("# Test Chapter\n", encoding="utf-8")
    output_path = chapter_dir / f"chapter-{chapter:02d}-test.compiled.json"
    ir_blocks = blocks or [
        IRUserTextBlock(text="Narrative memory opening.", origin="memory"),
        IRAssistantTextBlock(text="Narrative assistant beat."),
    ]
    payload = {
        "schema_version": "dream_compiled_chapter.v1",
        "chapter": {
            "number": chapter,
            "title": f"Chapter {chapter:02d}",
            "source_path": str(source_path),
        },
        "target": {
            "pair": list(pair),
            "semantic_block_indices": list(pair),
        },
        "parse": {
            "beat_count": len(ir_blocks),
            "beats": [],
        },
        "compiled": {
            "ir_block_count": len(ir_blocks),
            "ir_blocks": [block.model_dump(mode="json") for block in ir_blocks],
            "rendered_ir_block_count": len(ir_blocks),
            "provider_message_count": len(ir_blocks),
            "token_count": token_count,
            "token_count_scope": "ttl_collapsed_render",
            "expanded_pins": [],
        },
        "sanity": {
            "status": "skipped",
            "review": None,
            "transcript_path": None,
            "error": None,
        },
        "artifacts": {
            "compiled_json": str(output_path),
            "compiled_markdown": str(output_path.with_suffix(".md")),
        },
    }
    output_path.write_text(json.dumps(payload), encoding="utf-8")
    return output_path


def test_apply_dream_narratives_dry_run_does_not_mutate(tmp_path: Path) -> None:
    transcript = _write_transcript(tmp_path)
    chapter_dir = tmp_path / "chapters"
    _write_compiled(chapter_dir)
    before = transcript.read_text(encoding="utf-8")

    report = apply_dream_narratives(
        transcript_path=transcript,
        chapter_dir=chapter_dir,
        chapters=(1,),
        apply=False,
        write_report=False,
    )

    assert transcript.read_text(encoding="utf-8") == before
    assert report.dry_run is True
    assert report.records_to_append == 4
    assert report.records_appended == 0
    assert len(report.entries) == 1
    assert report.entries[0].status == "would_append"
    assert report.entries[0].artifact_records == 2
    assert report.entries[0].mode_records == 2
    assert report.entries[0].narrative_id == "dream_chapter_01_blocks_0_1"


def test_apply_dream_narratives_appends_artifacts_and_modes(tmp_path: Path) -> None:
    transcript = _write_transcript(tmp_path)
    chapter_dir = tmp_path / "chapters"
    _write_compiled(chapter_dir)
    before = transcript.read_text(encoding="utf-8")

    report = apply_dream_narratives(
        transcript_path=transcript,
        chapter_dir=chapter_dir,
        chapters=(1,),
        apply=True,
        backup=True,
        write_report=False,
    )

    assert report.dry_run is False
    assert report.records_appended == 4
    assert report.entries[0].status == "appended"
    assert report.backup_path is not None
    assert report.backup_path.exists()
    assert report.backup_path.read_text(encoding="utf-8") == before

    rehydrated = Rehydrator(transcript).run()
    parent = rehydrated.semantic_blocks[0]
    child = rehydrated.semantic_blocks[1]
    assert parent.mode == "pair_narrative"
    assert child.mode == "pair_narrative"
    assert parent.toks is not None
    assert parent.toks.tokens == 37
    assert child.toks is not None
    assert child.toks.tokens == 0
    assert any(
        isinstance(artifact, IRSemanticBlockPairNarrative)
        for artifact in parent.artifacts
    )
    assert any(
        isinstance(artifact, IRSemanticBlockPairNarrativeChild)
        for artifact in child.artifacts
    )

    manager = _build_block_manager(rehydrated)
    rendered_parent = manager.render_block(semantic_block=parent)
    rendered_child = manager.render_block(semantic_block=child)
    assert rendered_child == []
    assert any(
        isinstance(block, IRUserTextBlock) and block.text == "Narrative memory opening."
        for block in rendered_parent
    )


def test_apply_dream_narratives_skips_existing_pair_narrative(
    tmp_path: Path,
) -> None:
    transcript = _write_transcript(tmp_path)
    chapter_dir = tmp_path / "chapters"
    _write_compiled(chapter_dir)
    apply_dream_narratives(
        transcript_path=transcript,
        chapter_dir=chapter_dir,
        chapters=(1,),
        apply=True,
        backup=False,
        write_report=False,
    )
    before = transcript.read_text(encoding="utf-8")

    report = apply_dream_narratives(
        transcript_path=transcript,
        chapter_dir=chapter_dir,
        chapters=(1,),
        apply=False,
        write_report=False,
    )

    assert transcript.read_text(encoding="utf-8") == before
    assert report.records_to_append == 0
    assert report.skipped_existing == 1
    assert report.entries[0].status == "skipped_existing"


def test_apply_dream_narratives_rejects_compiled_pair_mismatch(
    tmp_path: Path,
) -> None:
    transcript = _write_transcript(tmp_path, block_count=3)
    chapter_dir = tmp_path / "chapters"
    _write_compiled(chapter_dir, pair=(0, 2))

    with pytest.raises(ValueError, match="targets pair"):
        apply_dream_narratives(
            transcript_path=transcript,
            chapter_dir=chapter_dir,
            chapters=(1,),
            apply=False,
            write_report=False,
        )


def test_apply_dream_narratives_rejects_wrong_tool_result_pair(
    tmp_path: Path,
) -> None:
    transcript = _write_transcript(tmp_path)
    chapter_dir = tmp_path / "chapters"
    _write_compiled(
        chapter_dir,
        blocks=[
            IRToolCallBlock(call_id="toolu_1", tool="Bash", input={}),
            IRToolResultBlock(
                call_id="toolu_1",
                tool="Read",
                content=[IRToolTextBlock(text="wrong tool")],
            ),
        ],
    )

    with pytest.raises(ValueError, match="expected Bash"):
        apply_dream_narratives(
            transcript_path=transcript,
            chapter_dir=chapter_dir,
            chapters=(1,),
            apply=False,
            write_report=False,
        )


def test_apply_dream_narratives_rejects_unfinished_transcript(
    tmp_path: Path,
) -> None:
    transcript = _write_transcript(tmp_path, unfinished=True)
    chapter_dir = tmp_path / "chapters"
    _write_compiled(chapter_dir)

    with pytest.raises(ValueError, match="unfinished turn"):
        apply_dream_narratives(
            transcript_path=transcript,
            chapter_dir=chapter_dir,
            chapters=(1,),
            apply=False,
            write_report=False,
        )


def test_parse_chapters_deduplicates_ranges() -> None:
    assert _parse_chapters("1-3,2,5") == (1, 2, 3, 5)

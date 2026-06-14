from __future__ import annotations

from pathlib import Path

from core_scripts.dream_director import (
    DirectorReviewInput,
    DirectorReviewReport,
    DirectorReviewRequest,
    Pair,
    _director_user_message,
    _expand_pin_markers,
    _parse_pair,
    _source_blocks_from_render,
    _summaries_markdown,
)
from core_scripts.merge_stats import MergeStatsReport
from spellbook.config import SpellbookConfig
from spellbook.fork import BlockDetectorResult
from spellbook.ir_types import (
    IRSemanticBlock,
    IRSemanticBlockFacet,
    IRSemanticBlockRange,
    IRSemanticBlockSummary,
    IRSkillCatalog,
    IRTokenRangeCount,
    IRUserTextBlock,
)
from spellbook.recorder import Recorder
from spellbook.rehydrator import Rehydrator
from spellbook.tools.registry import DEFAULT_TOOL_REGISTRY


def _count(tokens: int) -> IRTokenRangeCount:
    return IRTokenRangeCount(tokens=tokens, method="api", exact=True)


def _write_transcript(tmp_path: Path) -> Path:
    transcript = tmp_path / "transcript.jsonl"
    config = SpellbookConfig(cwd=tmp_path, model="claude-sonnet-4-6")
    recorder = Recorder(config, transcript, "session_test", DEFAULT_TOOL_REGISTRY)
    recorder.write_session_record(skill_catalog=IRSkillCatalog())
    recorder.start_turn("turn_1", [IRUserTextBlock(text="source", origin="human")])
    ranges = [
        IRSemanticBlockRange(
            title="Block zero",
            start_block=0,
            end_block=0,
            completed=True,
        ),
        IRSemanticBlockRange(
            title="Block one",
            start_block=1,
            end_block=1,
            completed=True,
        ),
    ]
    for semantic_range in ranges:
        recorder.detect_blocks(
            BlockDetectorResult(completed=[semantic_range], still_buffered=[])
        )

    blocks = [
        IRSemanticBlock(
            idx=0,
            title="Block zero",
            range=ranges[0],
            toks=_count(10),
            full_toks=_count(10),
        ),
        IRSemanticBlock(
            idx=1,
            title="Block one",
            range=ranges[1],
            toks=_count(20),
            full_toks=_count(20),
        ),
    ]
    facet = IRSemanticBlockFacet(
        id="facet_decision",
        title="Decision",
        description="A load-bearing decision.",
        start_block=0,
        end_block=0,
        resources=["src/file.py", "commit abc123"],
    )
    for block in blocks:
        recorder.write_semantic_block(block)
    recorder.write_block_artifact(
        IRSemanticBlockSummary(
            headline="Summary zero",
            text="They decided the important thing.",
            facets=[facet],
            open_thread="Follow up later.",
            toks=_count(5),
        ),
        blocks[0].id,
    )
    recorder.write_block_artifact(
        IRSemanticBlockSummary(
            headline="Summary one",
            text="They shipped the next piece.",
            facets=[],
            open_thread=None,
            toks=_count(6),
        ),
        blocks[1].id,
    )
    recorder.end_turn()
    return transcript


def test_parse_pair_requires_adjacent_indices() -> None:
    pair = _parse_pair("2-3")

    assert pair == Pair(first=2, second=3)
    assert pair.chapter_number == 2


def test_summaries_markdown_renders_summary_artifacts(tmp_path: Path) -> None:
    rehydrated = Rehydrator(_write_transcript(tmp_path)).run()

    rendered = _summaries_markdown(rehydrated, Pair(first=0, second=1))

    assert "### Block 0: Summary zero" in rendered
    assert "They decided the important thing." in rendered
    assert (
        "Decision (context blocks 0-0). Resources: src/file.py; commit abc123"
        in rendered
    )
    assert "Follow up later." in rendered
    assert "### Block 1: Summary one" in rendered


def test_source_blocks_from_render_skips_director_prompt(tmp_path: Path) -> None:
    render = tmp_path / "render.md"
    render.write_text(
        "# Prompt\n\nignore me\n\n# Source Blocks\n\nkeep me\n", encoding="utf-8"
    )

    assert _source_blocks_from_render(render) == "# Source Blocks\n\nkeep me\n"


def test_expand_pin_markers_inserts_matching_source_pin() -> None:
    result = _expand_pin_markers(
        chapter_markdown=(
            "# Chapter\n\n"
            "A narrative beat.\n\n"
            "<!-- pin: Three-Body Architecture Decision -->\n"
        ),
        source_blocks_markdown=(
            "# Source Blocks\n\n"
            '<pin kind="facet" block_idx="1" '
            'title="Three-Body Architecture Decision: Loom as Universal Transcript Intelligence" '
            'reason="Keep exact exchange." facet_id="facet_1">\n'
            "Pinned exact source material.\n"
            "</pin>\n"
        ),
    )

    assert result.matched_markers == ("Three-Body Architecture Decision",)
    assert result.unmatched_markers == ()
    assert result.unused_pins == ()
    assert "Pinned exact source material." in result.chapter_markdown
    assert "Matched chapter pin" in result.notes[0]


def test_director_user_message_contains_budget_and_content(tmp_path: Path) -> None:
    pair = Pair(first=0, second=1)
    request = DirectorReviewRequest(
        pair=pair,
        transcript_path=tmp_path / "transcript.jsonl",
        render_path=tmp_path / "render.md",
        chapter_path=tmp_path / "chapter.md",
        chapter_tokens=100,
        chapter_chars=500,
        ceiling_tokens=120,
        stats=MergeStatsReport(
            transcript_path=tmp_path / "transcript.jsonl",
            model="claude-opus-4-6",
            block_lines=[],
            full_count=_count(200),
            summary_count=_count(120),
        ),
        summaries_markdown="summary facts",
        source_blocks_markdown=(
            "# Source Blocks\n"
            '<pin kind="facet" block_idx="1" title="Decision" '
            'reason="Pinned reason." facet_id="facet_decision">\n'
            "source pin\n"
            "</pin>\n"
        ),
        chapter_markdown="# Chapter\nchapter\n\n<!-- pin: Decision -->",
    )

    message = _director_user_message(request)

    assert "Exact chapter tokens from mdtoks: 100" in message
    assert "Token ceiling from two summary renderings: 120" in message
    assert "summary facts" in message
    assert "## Pin Expansion Notes" in message
    assert "Matched chapter markers: 1" in message
    assert "# Source Blocks\n<pin" in message
    assert "## Dream Chapter (Pins Expanded For Review)" in message
    assert "# Chapter\nchapter" in message
    assert "source pin" in message


def test_director_review_report_json_shape(tmp_path: Path) -> None:
    pair = Pair(first=0, second=1)
    request = DirectorReviewRequest(
        pair=pair,
        transcript_path=tmp_path / "transcript.jsonl",
        render_path=tmp_path / "render.md",
        chapter_path=tmp_path / "chapter.md",
        chapter_tokens=100,
        chapter_chars=500,
        ceiling_tokens=120,
        stats=MergeStatsReport(
            transcript_path=tmp_path / "transcript.jsonl",
            model="claude-opus-4-6",
            block_lines=[],
            full_count=_count(200),
            summary_count=_count(120),
        ),
        summaries_markdown="summary",
        source_blocks_markdown="source",
        chapter_markdown="chapter",
    )
    review = DirectorReviewInput(
        verdict="pass",
        coverage_present=["decision"],
        coverage_missing=[],
        accuracy_flags=[],
        budget_token_count=100,
        budget_ceiling=120,
        budget_passed=True,
        budget_notes="ok",
        quality_notes="good",
        general_feedback="clean",
    )

    report = DirectorReviewReport(
        pair=pair,
        review=review,
        output_path=tmp_path / "review.json",
        director_model="claude-opus-4-8",
        director_transcript_path=tmp_path / "director.jsonl",
        request=request,
        turn_text="done",
    )

    data = report.to_json_dict()

    assert data["pair"] == [0, 1]
    assert data["chapter_number"] == 1
    assert data["chapter_tokens"] == 100
    assert data["ceiling_tokens"] == 120
    assert data["budget_passed"] is True
    assert data["review"] == review.model_dump(mode="json")

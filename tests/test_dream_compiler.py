from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.dreaming.dream_compiler import (
    ChapterBeat,
    CompileOptions,
    SanityCheckInput,
    SanityRunResult,
    SourcePin,
    _chapter_paths,
    _pair_for_chapter,
    _print_run_summary,
    compile_beats_to_ir_blocks,
    parse_chapter_markdown,
    run_compile,
)
from spellbook.config import SpellbookConfig
from spellbook.fork import BlockDetectorResult
from spellbook.ir_types import (
    IRAssistantTextBlock,
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
)
from spellbook.recorder import Recorder
from spellbook.tools.registry import DEFAULT_TOOL_REGISTRY


def _count(tokens: int) -> IRTokenRangeCount:
    return IRTokenRangeCount(tokens=tokens, method="api", exact=True)


def _write_transcript_with_pin(tmp_path: Path) -> Path:
    transcript = tmp_path / "transcript.jsonl"
    config = SpellbookConfig(cwd=tmp_path, model="claude-opus-4-6")
    recorder = Recorder(config, transcript, "session_test", DEFAULT_TOOL_REGISTRY)
    recorder.write_session_record(skill_catalog=IRSkillCatalog())

    recorder.start_turn("turn_1", [IRUserTextBlock(text="Pinned user", origin="human")])
    recorder.write_block(IRAssistantTextBlock(text="Pinned assistant"))
    recorder.write_block(
        IRToolCallBlock(call_id="toolu_pinned_big", tool="Read", input={})
    )
    recorder.write_block(
        IRToolResultBlock(
            call_id="toolu_pinned_big",
            tool="Read",
            content=[IRToolTextBlock(text="large pinned output\n" * 5)],
        )
    )
    recorder.write_tool_result_ttl(
        call_id="toolu_pinned_big",
        replace_content="[Read: pinned output collapsed by TTL]",
        ttl=0,
        trigger="end_turn",
        delivered_turn=1,
        output_ref="tool-outputs/toolu_pinned_big.txt",
    )
    recorder.end_turn()
    recorder.start_turn(
        "turn_2", [IRUserTextBlock(text="Unpinned user", origin="human")]
    )
    recorder.write_block(IRAssistantTextBlock(text="Unpinned assistant"))
    recorder.end_turn()

    range_zero = IRSemanticBlockRange(
        title="Block zero",
        start_block=0,
        end_block=3,
        completed=True,
    )
    range_one = IRSemanticBlockRange(
        title="Block one",
        start_block=4,
        end_block=5,
        completed=True,
    )
    recorder.detect_blocks(
        BlockDetectorResult(completed=[range_zero, range_one], still_buffered=[])
    )
    block_zero = IRSemanticBlock(
        idx=0,
        title="Block zero",
        range=range_zero,
        toks=_count(20),
        full_toks=_count(20),
    )
    block_one = IRSemanticBlock(
        idx=1,
        title="Block one",
        range=range_one,
        toks=_count(20),
        full_toks=_count(20),
    )
    recorder.write_semantic_block(block_zero)
    recorder.write_semantic_block(block_one)
    facet = IRSemanticBlockFacet(
        id="facet_decision",
        title="Decision Moment",
        description="Pinned source exchange.",
        start_block=0,
        end_block=3,
        resources=[],
    )
    recorder.write_block_artifact(
        IRSemanticBlockSummary(
            headline="Summary zero",
            text="Summary text.",
            facets=[facet],
            open_thread=None,
            toks=_count(5),
        ),
        block_zero.id,
    )
    recorder.write_block_artifact(
        IRSemanticBlockSummary(
            headline="Summary one",
            text="Next summary.",
            facets=[],
            open_thread=None,
            toks=_count(5),
        ),
        block_one.id,
    )
    recorder.apply_block_pin(
        IRSemanticBlockPin(
            kind="facet",
            facet_id="facet_decision",
            reason="Keep exact exchange.",
        ),
        block_zero.id,
    )
    return transcript


def test_parse_chapter_markdown_recognizes_dialogue_tools_and_pins() -> None:
    beats = parse_chapter_markdown(
        "# Chapter\n\n"
        "[A marginal note.]\n\n"
        "Ryan: Hello\n\n"
        "Claude: Hi\n\n"
        "  -> Bash: echo hi - completed\n"
        "  <- hi\n\n"
        "<!-- pin: Decision Moment -->\n"
    )

    assert [beat.kind for beat in beats] == [
        "assistant_text",
        "assistant_text",
        "user_text",
        "assistant_text",
        "tool_call",
        "tool_result",
        "pin",
    ]
    assert beats[4].tool_name == "Bash"
    assert beats[6].marker == "Decision Moment"


def test_parse_chapter_markdown_keeps_narrative_arrow_bullets_as_text() -> None:
    beats = parse_chapter_markdown(
        "# Chapter\n\n"
        "  -> Bash: echo hi - completed\n"
        "  -> The reviewer traced the daemon log to 16:29:36\n"
        "  -> sandbox.py — ~173 lines. Docker launch.\n"
        "  -> Write: status/meta.json — reset state\n"
        "  <- done\n"
    )

    tool_names = [beat.tool_name for beat in beats if beat.kind == "tool_call"]
    assistant_text = "\n".join(beat.text or "" for beat in beats)

    assert tool_names == ["Bash", "Write"]
    assert "The reviewer traced the daemon log" in assistant_text
    assert "sandbox.py" in assistant_text


def test_parse_chapter_markdown_merges_open_bracket_marginalia() -> None:
    beats = parse_chapter_markdown(
        "[A converged priority order emerged:\n\n"
        "1. Split homunculus.py\n"
        "2. Deduplicate utilities]\n\n"
        "Ryan: Yes\n"
    )

    assert [beat.kind for beat in beats] == ["assistant_text", "user_text"]
    assert beats[0].line_start == 1
    assert beats[0].line_end == 4
    assert "1. Split homunculus.py" in (beats[0].text or "")


def test_compile_beats_starts_with_memory_opener_and_expands_pin() -> None:
    source_pin = SourcePin(
        title="Decision Moment",
        block_idx=0,
        kind="facet",
        start_context_block=10,
        end_context_block=11,
        blocks=(
            IRUserTextBlock(text="Exact Ryan", origin="human"),
            IRAssistantTextBlock(text="Exact Claude"),
        ),
        facet_id="facet_decision",
    )
    beats = (
        ChapterBeat(
            kind="user_text",
            line_start=1,
            line_end=1,
            source="Ryan: Hello",
            text="Hello",
        ),
        ChapterBeat(
            kind="tool_call",
            line_start=2,
            line_end=2,
            source="-> Bash: echo hi",
            tool_name="Bash",
            tool_input={"command": "echo hi"},
            call_id="dream_toolu_0001",
        ),
        ChapterBeat(
            kind="pin",
            line_start=3,
            line_end=3,
            source="<!-- pin: Decision Moment -->",
            marker="Decision Moment",
        ),
    )

    blocks, expanded = compile_beats_to_ir_blocks(
        beats=beats,
        chapter_number=1,
        pair=_pair_for_chapter(1),
        source_pins=(source_pin,),
    )

    assert isinstance(blocks[0], IRUserTextBlock)
    assert blocks[0].origin == "memory"
    assert "not an exact transcript" in blocks[0].text
    assert any(
        getattr(block, "type", None) == "tool_result"
        and getattr(block, "call_id", None) == "dream_toolu_0001"
        for block in blocks
    )
    assert [pin.title for pin in expanded] == ["Decision Moment"]
    assert any(
        isinstance(block, IRUserTextBlock) and block.text == "Exact Ryan"
        for block in blocks
    )


def test_compile_beats_only_adds_memory_opener_to_chapter_one() -> None:
    beats = (
        ChapterBeat(
            kind="assistant_text",
            line_start=1,
            line_end=1,
            source="# Chapter 2",
            text="Chapter 2",
        ),
        ChapterBeat(
            kind="user_text",
            line_start=2,
            line_end=2,
            source="Ryan: Next",
            text="Next",
        ),
    )

    blocks, expanded = compile_beats_to_ir_blocks(
        beats=beats,
        chapter_number=2,
        pair=_pair_for_chapter(2),
        source_pins=(),
    )

    assert expanded == ()
    assert isinstance(blocks[0], IRAssistantTextBlock)
    assert blocks[0].text == "Chapter 2"
    assert all(
        not (
            isinstance(block, IRUserTextBlock)
            and "not an exact transcript" in block.text
        )
        for block in blocks
    )


@pytest.mark.asyncio
async def test_run_compile_writes_json_and_markdown_with_fake_sanity(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    transcript = _write_transcript_with_pin(tmp_path)
    chapter_dir = tmp_path / "chapters"
    chapter_dir.mkdir()
    chapter = chapter_dir / "chapter-01-test.md"
    chapter.write_text(
        "# Chapter 1: Test\n\n"
        "Ryan: Hello\n\n"
        "Claude: Hi\n\n"
        "<!-- pin: Decision Moment -->\n",
        encoding="utf-8",
    )
    review_dir = tmp_path / "reviews"
    review_dir.mkdir()
    (review_dir / "review-01.json").write_text(
        json.dumps({"ceiling_tokens": 100}),
        encoding="utf-8",
    )

    async def fake_count(blocks: list[object]) -> int:
        tool_result_texts = [
            content.text
            for block in blocks
            if isinstance(block, IRToolResultBlock)
            for content in block.content
            if isinstance(content, IRToolTextBlock)
        ]
        assert "[Read: pinned output collapsed by TTL]" in tool_result_texts
        assert "large pinned output\n" * 5 not in tool_result_texts
        return len(blocks) * 10

    async def fake_sanity(compiled: object) -> SanityRunResult:
        return SanityRunResult(
            review=SanityCheckInput(
                verdict="pass",
                notes="Coherent.",
                boundary_notes="Boundaries look fine.",
                tool_pairing_notes="No dangling tools.",
                malformed_items=[],
            ),
            transcript_path=tmp_path / "sanity.jsonl",
        )

    report = await run_compile(
        CompileOptions(
            chapter_dir=chapter_dir,
            transcript_path=transcript,
            review_dir=review_dir,
            out_dir=tmp_path / "compiled",
            show_progress=False,
        ),
        token_count_fn=fake_count,  # type: ignore[arg-type]
        sanity_runner=fake_sanity,  # type: ignore[arg-type]
    )

    result = report.results[0]
    assert result.status == "compiled"
    assert result.sanity_status == "pass"
    assert result.token_count is not None
    assert result.ceiling_tokens == 100
    assert result.savings_tokens == 100 - result.token_count
    assert result.output_path is not None
    data = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert data["schema_version"] == "dream_compiled_chapter.v1"
    assert data["compiled"]["token_count_scope"] == "ttl_collapsed_render"
    assert data["compiled"]["expanded_pins"][0]["title"] == "Decision Moment"
    assert data["compiled"]["ir_blocks"][0]["origin"] == "memory"
    raw_json = json.dumps(data["compiled"]["ir_blocks"])
    assert "large pinned output" in raw_json
    preview = result.preview_path.read_text(encoding="utf-8")  # type: ignore[union-attr]
    assert "Pinned user" in preview
    assert "[Read: pinned output collapsed by TTL]" in preview
    assert "large pinned output" not in preview

    _print_run_summary(report)
    output = capsys.readouterr().out
    assert "Ceiling" in output
    assert "Savings" in output
    assert "Totals:" in output
    assert "100 original ceiling" in output


@pytest.mark.asyncio
async def test_run_compile_reports_unmatched_pin_with_nearby_lines(
    tmp_path: Path,
) -> None:
    transcript = _write_transcript_with_pin(tmp_path)
    chapter_dir = tmp_path / "chapters"
    chapter_dir.mkdir()
    chapter = chapter_dir / "chapter-01-test.md"
    chapter.write_text(
        "# Chapter 1: Test\n\nRyan: Hello\n\n<!-- pin: Missing Pin -->\n",
        encoding="utf-8",
    )

    async def fake_count(blocks: list[object]) -> int:
        return len(blocks)

    report = await run_compile(
        CompileOptions(
            chapter_dir=chapter_dir,
            transcript_path=transcript,
            review_dir=tmp_path / "reviews",
            out_dir=tmp_path / "compiled",
            skip_sanity=True,
            show_progress=False,
        ),
        token_count_fn=fake_count,  # type: ignore[arg-type]
    )

    result = report.results[0]
    assert result.status == "failed"
    assert result.error is not None
    assert "did not match source pins" in result.error.message
    assert result.error.line_start == 5
    assert any("Missing Pin" in line for line in result.error.nearby_lines)


def test_chapter_paths_excludes_compiled_markdown(tmp_path: Path) -> None:
    chapter_dir = tmp_path / "chapters"
    chapter_dir.mkdir()
    source = chapter_dir / "chapter-01-test.md"
    compiled = chapter_dir / "chapter-01-test.compiled.md"
    source.write_text("# Source\n", encoding="utf-8")
    compiled.write_text("# Preview\n", encoding="utf-8")

    assert _chapter_paths(chapter_dir, None) == [source.resolve()]

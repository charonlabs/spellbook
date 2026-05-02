from __future__ import annotations

from pathlib import Path

from scripts import replay_context_markdown
from spellbook.config import HomunculusConfig, SpellbookConfig
from spellbook.fork import BlockDetectorResult
from spellbook.ir_types import (
    IRSemanticBlock,
    IRSemanticBlockFacet,
    IRSemanticBlockRange,
    IRSemanticBlockSummary,
    IRSkillCatalog,
    IRUserTextBlock,
)
from spellbook.recorder import Recorder
from spellbook.tools.registry import DEFAULT_TOOL_REGISTRY


def _write_replay_transcript(tmp_path: Path) -> Path:
    transcript = tmp_path / "replay.jsonl"
    config = SpellbookConfig(
        model="claude-sonnet-4-6",
        cwd=tmp_path,
        hom_config=HomunculusConfig(detect_interval=2),
    )
    recorder = Recorder(config, transcript, "session_replay", DEFAULT_TOOL_REGISTRY)
    recorder.write_session_record(skill_catalog=IRSkillCatalog())
    recorder.start_turn("turn_original", [])
    recorder.write_block(IRUserTextBlock(text="hello", origin="human"))
    recorder.summon_fork(
        fork_id="detector_test",
        fork_type="block_detector",
        child_transcript_path=str(tmp_path / "forks" / "detector_test.jsonl"),
    )

    proposed = IRSemanticBlockRange(
        title="Draft planning",
        start_block=0,
        end_block=0,
    )
    completed_range = IRSemanticBlockRange(
        title="Completed setup",
        start_block=0,
        end_block=0,
        completed=True,
    )
    recorder.detect_blocks(
        BlockDetectorResult(
            completed=[completed_range],
            still_buffered=[proposed],
        )
    )
    completed_block = IRSemanticBlock(
        idx=0,
        title="Completed setup",
        range=completed_range,
        toks=None,
        full_toks=None,
    )
    recorder.write_semantic_block(completed_block)
    summary = IRSemanticBlockSummary(
        headline="Setup summary",
        text="Ryan and Codex set up the replay pipeline.",
        facets=[
            IRSemanticBlockFacet(
                title="Replay scripts",
                description="Scripts can inspect detector and summarizer outputs.",
                start_block=0,
                end_block=0,
                resources=["scripts/replay_context.py"],
            )
        ],
        open_thread="Use the markdown report for future agent review.",
        toks=None,
    )
    recorder.write_block_artifact(summary, completed_block.id)
    recorder.shutdown_fork("detector_test")
    recorder.end_turn()
    return transcript


def test_render_replay_markdown_includes_replay_events(tmp_path: Path) -> None:
    transcript = _write_replay_transcript(tmp_path)

    markdown = replay_context_markdown.render_replay_markdown(transcript)

    assert "# Replay Context Report" in markdown
    assert "- Session: `session_replay`" in markdown
    assert "- Detect interval: `2`" in markdown
    assert "### Fork Summoned: `detector_test`" in markdown
    assert "### Proposed Blocks After Detection 1" in markdown
    assert "Draft planning" in markdown
    assert "### Completed Block 0: Completed setup" in markdown
    assert '### Summary for Block 0: "Completed setup"' in markdown
    assert "#### Setup summary" in markdown
    assert "Ryan and Codex set up the replay pipeline." in markdown
    assert "Replay scripts" in markdown
    assert "Use the markdown report for future agent review." in markdown
    assert "### Fork Shutdown: `detector_test`" in markdown


def test_main_writes_markdown_output(tmp_path: Path, capsys) -> None:
    transcript = _write_replay_transcript(tmp_path)
    output = tmp_path / "report.md"

    replay_context_markdown.main(
        [
            "--transcript",
            str(transcript),
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert f"Wrote {output.resolve()}" in captured.out
    assert output.read_text(encoding="utf-8").startswith("# Replay Context Report")

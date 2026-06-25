from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
from scripts.dreaming.dream_restore import (
    KeepItem,
    RestoreOptions,
    RestoreTask,
    _load_keep_items,
    _pair_for_chapter,
    _parse_chapters,
    _print_run_summary,
    _restore_user_message,
    run_restore,
)
from scripts.dreaming.merge_stats import BlockLine, MergeStatsReport
from spellbook.ir_types import IRTokenRangeCount


def _count(tokens: int) -> IRTokenRangeCount:
    return IRTokenRangeCount(tokens=tokens, method="api", exact=True)


def _stats(transcript_path: Path, render_path: Path) -> MergeStatsReport:
    return MergeStatsReport(
        transcript_path=transcript_path,
        model="claude-opus-4-6",
        block_lines=[
            BlockLine(
                idx=4,
                title="Transcript IR Debug Path Fix",
                start_block=100,
                end_block=150,
            ),
            BlockLine(
                idx=5,
                title="Historical Endpoint Interface",
                start_block=151,
                end_block=200,
            ),
        ],
        full_count=_count(12_000),
        summary_count=_count(2_400),
        pins=[],
        render_path=render_path,
    )


def _write_triage(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "chapter_03": {
                    "keep": [
                        {
                            "item": "Ryan's explicit affirmation that the direction brings things to a whole new level",
                            "category": "relational",
                            "reason": "This is the emotional beat of the block.",
                        }
                    ],
                    "compress": [],
                    "drop": [],
                },
                "chapter_04": {
                    "keep": [],
                    "compress": [],
                    "drop": [],
                },
            }
        ),
        encoding="utf-8",
    )


def _write_tool_call(transcript_path: Path, tool: str) -> None:
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(
        json.dumps({"event": {"type": "tool_call", "tool": tool}}) + "\n",
        encoding="utf-8",
    )


def test_load_keep_items_skips_chapters_without_keep_items(tmp_path: Path) -> None:
    triage_path = tmp_path / "triage.json"
    _write_triage(triage_path)

    keep = _load_keep_items(triage_path)

    assert list(keep) == [3]
    assert keep[3][0].category == "relational"


def test_pair_for_chapter_and_parse_chapter_filter() -> None:
    assert _pair_for_chapter(1).slug == "0-1"
    assert _pair_for_chapter(26).slug == "50-51"
    assert _parse_chapters("3,5-7,5") == (3, 5, 6, 7)

    with pytest.raises(argparse.ArgumentTypeError):
        _parse_chapters("7-5")


def test_restore_user_message_includes_keep_items_render_and_chapter_path(
    tmp_path: Path,
) -> None:
    render_path = tmp_path / "render.md"
    render_path.write_text("# Source\n\nrender body", encoding="utf-8")
    chapter_path = tmp_path / "chapter-03-test.md"
    item = KeepItem(
        chapter_id="chapter_03",
        chapter_number=3,
        item="fold realism coined",
        category="philosophical",
        reason="A naming moment should survive.",
    )

    message = _restore_user_message(
        chapter_path=chapter_path,
        keep_items=(item,),
        pair=_pair_for_chapter(3),
        render_path=render_path,
        stats_text="Merge pair: Block 4 + Block 5",
    )

    assert str(chapter_path) in message
    assert "fold realism coined" in message
    assert "Triage reason: A naming moment should survive." in message
    assert "render body" in message
    assert "Do not use Write" in message
    assert "End your turn with a quick summary/overview" in message


@pytest.mark.asyncio
async def test_run_restore_with_fake_edit_runner(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    triage_path = tmp_path / "triage.json"
    _write_triage(triage_path)
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text('{"ir":"session"}\n', encoding="utf-8")
    chapter_dir = tmp_path / "chapters"
    chapter_dir.mkdir()
    chapter = chapter_dir / "chapter-03-the-composition.md"
    chapter.write_text("# Chapter\n\nBody.\n\n## Anchors\n", encoding="utf-8")

    async def fake_merge_stats(**kwargs: object) -> MergeStatsReport:
        render_path = Path(str(kwargs["render_path"]))
        render_path.parent.mkdir(parents=True, exist_ok=True)
        render_path.write_text("# Render\n\nsource blocks", encoding="utf-8")
        return _stats(transcript, render_path.resolve())

    async def fake_restore(task: RestoreTask) -> str:
        assert "source blocks" in task.plan.user_message
        text = task.plan.chapter_path.read_text(encoding="utf-8")
        task.plan.chapter_path.write_text(
            text.replace("## Anchors", "Restored beat.\n\n## Anchors"),
            encoding="utf-8",
        )
        _write_tool_call(task.plan.transcript_path, "Edit")
        return "Inserted the Ryan affirmation near the closing beat."

    report = await run_restore(
        RestoreOptions(
            triage_path=triage_path,
            transcript_path=transcript,
            chapter_dir=chapter_dir,
            work_dir=tmp_path / "work",
            show_progress=False,
        ),
        merge_stats_fn=fake_merge_stats,
        restore_task_runner=fake_restore,
    )

    assert len(report.plans) == 1
    assert report.plans[0].pair.slug == "4-5"
    assert report.results[0].status == "restored"
    assert report.results[0].edit_calls == 1
    assert "Restored beat." in chapter.read_text(encoding="utf-8")

    _print_run_summary(report)
    output = capsys.readouterr().out
    assert "Keep items:" in output
    assert "Ryan's explicit affirmation" in output
    assert "Reason: This is the emotional beat of the block." in output
    assert "Opus summary:" in output
    assert "Inserted the Ryan affirmation near the closing beat." in output


@pytest.mark.asyncio
async def test_run_restore_flags_write_usage(tmp_path: Path) -> None:
    triage_path = tmp_path / "triage.json"
    _write_triage(triage_path)
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text('{"ir":"session"}\n', encoding="utf-8")
    chapter_dir = tmp_path / "chapters"
    chapter_dir.mkdir()
    chapter = chapter_dir / "chapter-03-the-composition.md"
    chapter.write_text("# Chapter\n", encoding="utf-8")

    async def fake_merge_stats(**kwargs: object) -> MergeStatsReport:
        render_path = Path(str(kwargs["render_path"]))
        render_path.parent.mkdir(parents=True, exist_ok=True)
        render_path.write_text("# Render\n", encoding="utf-8")
        return _stats(transcript, render_path.resolve())

    async def fake_restore(task: RestoreTask) -> str:
        task.plan.chapter_path.write_text("# Chapter\n\nchanged\n", encoding="utf-8")
        _write_tool_call(task.plan.transcript_path, "Write")
        return "wrote"

    report = await run_restore(
        RestoreOptions(
            triage_path=triage_path,
            transcript_path=transcript,
            chapter_dir=chapter_dir,
            work_dir=tmp_path / "work",
            show_progress=False,
        ),
        merge_stats_fn=fake_merge_stats,
        restore_task_runner=fake_restore,
    )

    result = report.results[0]
    assert result.status == "failed"
    assert result.write_calls == 1
    assert "Model used Write" in str(result.error)

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
from scripts.dreaming.dream_director import Pair
from scripts.dreaming.dream_merge import (
    QUANTUM_DICE_CONTRACT,
    DirectorResult,
    DirectorTask,
    DreamMergeOptions,
    DreamTask,
    _clone_transcript_workspace,
    _dream_user_message,
    _parse_pairs,
    _write_stats_files,
    run_dream_merge,
)
from scripts.dreaming.mdtoks import MarkdownTokenReport
from scripts.dreaming.merge_stats import BlockLine, MergeStatsReport, PinCount
from spellbook.ir_types import IRTokenRangeCount


def _count(tokens: int) -> IRTokenRangeCount:
    return IRTokenRangeCount(tokens=tokens, method="api", exact=True)


def _stats(transcript_path: Path, render_path: Path | None = None) -> MergeStatsReport:
    return MergeStatsReport(
        transcript_path=transcript_path,
        model="claude-opus-4-6",
        block_lines=[
            BlockLine(
                idx=0,
                title="Session 38 Startup",
                start_block=0,
                end_block=67,
            ),
            BlockLine(
                idx=1,
                title="Design Common IR Schema",
                start_block=68,
                end_block=135,
            ),
        ],
        full_count=_count(48_200),
        summary_count=_count(3_400),
        pins=[
            PinCount(
                block_idx=1,
                block_title="Design Common IR Schema",
                kind="facet",
                title="Three-Body Architecture Decision",
                tokens=2_231,
                exact=True,
                method="api",
                reason="Important exact exchange.",
                facet_id="facet_1",
            )
        ],
        render_path=render_path,
    )


def test_parse_pairs_requires_adjacent_unique_pairs() -> None:
    assert _parse_pairs("0-1, 2-3") == (
        Pair(first=0, second=1),
        Pair(first=2, second=3),
    )

    with pytest.raises(argparse.ArgumentTypeError):
        _parse_pairs("0-2")
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_pairs("0-1,0-1")


def test_clone_transcript_workspace_copies_parallel_asset_dirs(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    transcript = source_dir / "transcript.jsonl"
    transcript.write_text('{"ir":"session"}\n', encoding="utf-8")
    (source_dir / "blobs").mkdir()
    (source_dir / "blobs" / "image.bin").write_text("blob", encoding="utf-8")
    (source_dir / "tool-outputs").mkdir()
    (source_dir / "tool-outputs" / "tool.txt").write_text("tool", encoding="utf-8")

    fork_transcript = _clone_transcript_workspace(
        source_transcript=transcript,
        fork_dir=tmp_path / "fork",
    )

    assert fork_transcript.read_text(encoding="utf-8") == '{"ir":"session"}\n'
    assert (fork_transcript.parent / "blobs" / "image.bin").read_text(
        encoding="utf-8"
    ) == "blob"
    assert (fork_transcript.parent / "tool-outputs" / "tool.txt").read_text(
        encoding="utf-8"
    ) == "tool"


def test_clone_transcript_workspace_replaces_existing_fork_dir(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    transcript = source_dir / "transcript.jsonl"
    transcript.write_text("fresh\n", encoding="utf-8")
    fork_dir = tmp_path / "fork"
    fork_dir.mkdir()
    (fork_dir / "transcript.jsonl").write_text("stale turn\n", encoding="utf-8")
    (fork_dir / "tool-outputs").mkdir()
    (fork_dir / "tool-outputs" / "stale.txt").write_text("stale", encoding="utf-8")

    fork_transcript = _clone_transcript_workspace(
        source_transcript=transcript,
        fork_dir=fork_dir,
    )

    assert fork_transcript.read_text(encoding="utf-8") == "fresh\n"
    assert not (fork_dir / "tool-outputs" / "stale.txt").exists()


def test_dream_user_message_includes_contract_stats_pins_and_render_path(
    tmp_path: Path,
) -> None:
    render_path = tmp_path / "render.md"
    render_path.write_text(
        "# Director Opening\n\nFull dream message from render.\n\n# Source Blocks\nsource",
        encoding="utf-8",
    )
    pair_run = _pair_run_for_test(tmp_path, render_path=render_path)

    message = _dream_user_message(pair_run)

    assert QUANTUM_DICE_CONTRACT in message
    assert "Save the chapter to:" in message
    assert str(pair_run.chapter_path) in message
    assert "Full fidelity (both): 48,200 tokens" in message
    assert "Three-Body Architecture Decision" in message
    assert "## Rendered Markdown Path" in message
    assert str(render_path) in message
    assert "Full dream message from render." not in message


@pytest.mark.asyncio
async def test_run_dream_merge_with_fake_runners(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    transcript = source_dir / "transcript.jsonl"
    transcript.write_text('{"ir":"session"}\n', encoding="utf-8")
    (source_dir / "blobs").mkdir()
    (source_dir / "blobs" / "asset.txt").write_text("asset", encoding="utf-8")
    (source_dir / "tool-outputs").mkdir()
    (source_dir / "tool-outputs" / "output.txt").write_text("output", encoding="utf-8")
    dream_messages: list[str] = []

    async def fake_merge_stats(**kwargs: object) -> MergeStatsReport:
        render_path = Path(str(kwargs["render_path"]))
        render_path.parent.mkdir(parents=True, exist_ok=True)
        render_path.write_text(
            "# Director Opening\n\nFULL RENDER BODY\n\n# Source Blocks\nsource",
            encoding="utf-8",
        )
        return _stats(transcript, render_path=render_path.resolve())

    async def fake_count(path: Path, **_: object) -> MarkdownTokenReport:
        return MarkdownTokenReport(
            path=path,
            model="claude-opus-4-6",
            tokens=42,
            characters=len(path.read_text(encoding="utf-8")),
        )

    async def fake_dream(task: DreamTask) -> str:
        dream_messages.append(task.dream_message)
        assert (
            task.pair_run.fork_transcript_path.parent / "blobs" / "asset.txt"
        ).exists()
        assert (
            task.pair_run.fork_transcript_path.parent / "tool-outputs" / "output.txt"
        ).exists()
        alternate = task.pair_run.chapter_path.parent / "chapter-03-the-composition.md"
        alternate.parent.mkdir(parents=True, exist_ok=True)
        alternate.write_text("# Dream\n", encoding="utf-8")
        return "wrote chapter"

    async def fake_director(task: DirectorTask) -> DirectorResult:
        assert task.pair_run.chapter_path.exists()
        return DirectorResult(
            status="reviewed",
            verdict="pass",
            review_path=task.pair_run.review_path,
        )

    report = await run_dream_merge(
        DreamMergeOptions(
            pairs=(Pair(first=4, second=5),),
            transcript_path=transcript,
            chapter_dir=tmp_path / "chapters",
            review_dir=tmp_path / "reviews",
            work_dir=tmp_path / "work",
            concurrency=2,
            show_progress=False,
        ),
        merge_stats_fn=fake_merge_stats,
        count_markdown_fn=fake_count,
        dream_task_runner=fake_dream,
        director_task_runner=fake_director,
    )

    pair_run = report.pair_runs[0]
    assert pair_run.dream_result is not None
    assert pair_run.dream_result.status == "written"
    assert pair_run.chapter_path.name == "chapter-03-the-composition.md"
    assert pair_run.dream_result.chapter_tokens == 42
    assert pair_run.director_result is not None
    assert pair_run.director_result.verdict == "pass"
    assert "FULL RENDER BODY" not in dream_messages[0]
    assert str(pair_run.render_path) in dream_messages[0]
    assert pair_run.stats_text_path.exists()
    assert pair_run.stats_json_path.exists()


@pytest.mark.asyncio
async def test_run_dream_merge_can_reuse_preflight_stats(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text('{"ir":"session"}\n', encoding="utf-8")
    work_dir = tmp_path / "work"
    pair = Pair(first=0, second=1)
    render_path = work_dir / "renders" / "merge-blocks-0-1.md"
    render_path.parent.mkdir(parents=True)
    render_path.write_text("# Render\n", encoding="utf-8")
    stats = _stats(transcript, render_path=render_path.resolve())
    _write_stats_files(
        work_dir=work_dir,
        pair=pair,
        stats=stats,
        stats_text="cached stats",
    )

    async def fail_merge_stats(**_: object) -> MergeStatsReport:
        raise AssertionError("merge_stats should not run when preflight is reused")

    report = await run_dream_merge(
        DreamMergeOptions(
            pairs=(pair,),
            transcript_path=transcript,
            chapter_dir=tmp_path / "chapters",
            review_dir=tmp_path / "reviews",
            work_dir=work_dir,
            dry_run=True,
            reuse_preflight=True,
            show_progress=False,
        ),
        merge_stats_fn=fail_merge_stats,
    )

    assert report.pair_runs[0].stats.full_count.tokens == 48_200
    assert report.pair_runs[0].render_path == render_path.resolve()


def _pair_run_for_test(tmp_path: Path, *, render_path: Path):
    from scripts.dreaming.dream_merge import PairRun

    pair = Pair(first=0, second=1)
    stats = _stats(tmp_path / "transcript.jsonl", render_path=render_path)
    return PairRun(
        pair=pair,
        stats=stats,
        stats_text=(
            "Merge pair: Block 0 + Block 1\n"
            "Full fidelity (both): 48,200 tokens\n"
            "Summaries (both):      3,400 tokens (target ceiling)"
        ),
        stats_text_path=tmp_path / "stats.txt",
        stats_json_path=tmp_path / "stats.json",
        render_path=render_path,
        chapter_path=tmp_path / "chapter.md",
        review_path=tmp_path / "review.json",
        fork_dir=tmp_path / "fork",
        fork_transcript_path=tmp_path / "fork" / "transcript.jsonl",
    )

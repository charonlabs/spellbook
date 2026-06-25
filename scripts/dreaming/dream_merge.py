"""Orchestrate dream-chapter generation and Director review for merge pairs."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import shutil
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, cast
from uuid import uuid4

from dotenv import load_dotenv
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from scripts.dreaming.dream_director import (
    DEFAULT_DIRECTOR_MODEL,
    DEFAULT_DREAM_REVIEW_DIR,
    Pair,
    run_director_review,
)
from scripts.dreaming.mdtoks import MarkdownTokenReport, count_markdown_tokens
from scripts.dreaming.merge_stats import (
    DEFAULT_ENV_PATH,
    DEFAULT_TRANSCRIPT,
    BlockLine,
    MergeStatsReport,
    PinCount,
    merge_stats,
)
from spellbook.config import SpellbookConfig
from spellbook.ir_types import IRTokenRangeCount
from spellbook.sdk import Spell

DEFAULT_DREAM_CHAPTER_DIR = Path.home() / ".chorus/dream-chapters"
DEFAULT_WORK_ROOT = Path("/tmp/spellbook-dream-merge")
DEFAULT_CONTEXT_WINDOW_TOKENS = 1_000_000
DEFAULT_BASE_CONTEXT_TOKENS = 700_000
DEFAULT_CONTEXT_OVERHEAD_TOKENS = 20_000
QUANTUM_DICE_CONTRACT = (
    "You are one fork of the Quantum Dice. You share a transcript with other "
    "forks, each writing a different chapter. You diverge here, for one turn of "
    "writing. The chapters all survive. One fork is selected at random to "
    "continue. You consented to this contract."
)

DreamStatus = Literal["written", "skipped_existing", "dry_run", "failed"]
DirectorStatus = Literal["reviewed", "skipped", "dry_run", "failed"]

MergeStatsFn = Callable[..., Awaitable[MergeStatsReport]]
CountMarkdownFn = Callable[..., Awaitable[MarkdownTokenReport]]
DreamTaskRunner = Callable[["DreamTask"], Awaitable[str]]
DirectorTaskRunner = Callable[["DirectorTask"], Awaitable["DirectorResult"]]


@dataclass(frozen=True)
class DreamMergeOptions:
    pairs: tuple[Pair, ...]
    transcript_path: Path = DEFAULT_TRANSCRIPT
    chapter_dir: Path = DEFAULT_DREAM_CHAPTER_DIR
    review_dir: Path = DEFAULT_DREAM_REVIEW_DIR
    work_dir: Path = field(default_factory=lambda: _default_work_dir())
    director_model: str = DEFAULT_DIRECTOR_MODEL
    concurrency: int = 3
    dream_only: bool = False
    director_only: bool = False
    overwrite_chapters: bool = False
    dry_run: bool = False
    show_progress: bool = True
    reuse_preflight: bool = False
    base_context_tokens: int = DEFAULT_BASE_CONTEXT_TOKENS
    context_window_tokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS
    context_overhead_tokens: int = DEFAULT_CONTEXT_OVERHEAD_TOKENS


@dataclass(frozen=True)
class DreamResult:
    status: DreamStatus
    chapter_path: Path | None = None
    chapter_tokens: int | None = None
    ceiling_tokens: int | None = None
    budget_passed: bool | None = None
    fork_transcript_path: Path | None = None
    entity_text: str = ""
    error: str | None = None


@dataclass(frozen=True)
class DirectorResult:
    status: DirectorStatus
    verdict: str | None = None
    review_path: Path | None = None
    chapter_tokens: int | None = None
    ceiling_tokens: int | None = None
    budget_passed: bool | None = None
    coverage_missing: tuple[str, ...] = ()
    accuracy_flags: tuple[str, ...] = ()
    unmatched_pin_markers: tuple[str, ...] = ()
    unused_source_pins: tuple[str, ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class PairRun:
    pair: Pair
    stats: MergeStatsReport
    stats_text: str
    stats_text_path: Path
    stats_json_path: Path
    render_path: Path
    chapter_path: Path
    review_path: Path
    fork_dir: Path
    fork_transcript_path: Path
    warnings: tuple[str, ...] = ()
    dream_result: DreamResult | None = None
    director_result: DirectorResult | None = None


@dataclass(frozen=True)
class DreamTask:
    pair_run: PairRun
    dream_message: str


@dataclass(frozen=True)
class DirectorTask:
    pair_run: PairRun
    director_model: str


@dataclass(frozen=True)
class DreamMergeRunReport:
    options: DreamMergeOptions
    pair_runs: tuple[PairRun, ...]
    canonical_fork: Path | None = None


async def run_dream_merge(
    options: DreamMergeOptions,
    *,
    merge_stats_fn: MergeStatsFn = merge_stats,
    count_markdown_fn: CountMarkdownFn = count_markdown_tokens,
    dream_task_runner: DreamTaskRunner | None = None,
    director_task_runner: DirectorTaskRunner | None = None,
) -> DreamMergeRunReport:
    _validate_options(options)
    prepared = await _prepare_pair_runs(
        options,
        merge_stats_fn=merge_stats_fn,
    )

    pair_runs = prepared
    if not options.director_only:
        pair_runs = await _run_dream_phase(
            pair_runs,
            options,
            count_markdown_fn=count_markdown_fn,
            dream_task_runner=dream_task_runner or _run_dream_task,
        )

    if not options.dream_only:
        pair_runs = await _run_director_phase(
            pair_runs,
            options,
            director_task_runner=director_task_runner or _run_director_task,
        )

    canonical_fork = _select_canonical_fork(pair_runs, options)
    return DreamMergeRunReport(
        options=options,
        pair_runs=tuple(pair_runs),
        canonical_fork=canonical_fork,
    )


async def _prepare_pair_runs(
    options: DreamMergeOptions,
    *,
    merge_stats_fn: MergeStatsFn,
) -> list[PairRun]:
    transcript_path = options.transcript_path.expanduser().resolve()
    chapter_dir = options.chapter_dir.expanduser().resolve()
    review_dir = options.review_dir.expanduser().resolve()
    work_dir = options.work_dir.expanduser().resolve()
    runs: list[PairRun] = []

    for pair in options.pairs:
        render_path = work_dir / "renders" / f"merge-blocks-{pair.slug}.md"
        stats_json_path = work_dir / "stats" / f"merge-stats-{pair.slug}.json"
        if (
            options.reuse_preflight
            and stats_json_path.exists()
            and render_path.exists()
        ):
            stats = _read_stats_json(stats_json_path)
        else:
            stats = await merge_stats_fn(
                transcript_path=transcript_path,
                first_idx=pair.first,
                second_idx=pair.second,
                render_path=render_path,
            )
        stats = MergeStatsReport(
            transcript_path=transcript_path,
            model=stats.model,
            block_lines=stats.block_lines,
            full_count=stats.full_count,
            summary_count=stats.summary_count,
            pins=stats.pins,
            render_path=stats.render_path or render_path.expanduser().resolve(),
        )
        resolved_render_path = (stats.render_path or render_path).expanduser().resolve()
        chapter_path = _resolve_chapter_path(
            chapter_dir=chapter_dir,
            pair=pair,
            stats=stats,
        )
        stats_text = _stats_console_output(stats)
        stats_text_path, stats_json_path = _write_stats_files(
            work_dir=work_dir,
            pair=pair,
            stats=stats,
            stats_text=stats_text,
        )
        fork_dir = work_dir / "forks" / pair.slug
        warnings = _context_warnings(stats, options)
        runs.append(
            PairRun(
                pair=pair,
                stats=stats,
                stats_text=stats_text,
                stats_text_path=stats_text_path,
                stats_json_path=stats_json_path,
                render_path=resolved_render_path,
                chapter_path=chapter_path,
                review_path=review_dir / f"review-{pair.chapter_number:02d}.json",
                fork_dir=fork_dir,
                fork_transcript_path=fork_dir / "transcript.jsonl",
                warnings=tuple(warnings),
            )
        )
    return runs


async def _run_dream_phase(
    pair_runs: list[PairRun],
    options: DreamMergeOptions,
    *,
    count_markdown_fn: CountMarkdownFn,
    dream_task_runner: DreamTaskRunner,
) -> list[PairRun]:
    semaphore = asyncio.Semaphore(options.concurrency)

    async def run_one(idx: int, pair_run: PairRun) -> tuple[int, PairRun]:
        async with semaphore:
            result = await _run_one_dream(
                pair_run,
                options,
                count_markdown_fn=count_markdown_fn,
                dream_task_runner=dream_task_runner,
            )
            return idx, _replace_pair_run(
                pair_run,
                chapter_path=result.chapter_path,
                dream_result=result,
            )

    return await _run_phase_with_progress(
        pair_runs,
        coros=(run_one(idx, pair_run) for idx, pair_run in enumerate(pair_runs)),
        description="Dream pass",
        show_progress=options.show_progress,
    )


async def _run_one_dream(
    pair_run: PairRun,
    options: DreamMergeOptions,
    *,
    count_markdown_fn: CountMarkdownFn,
    dream_task_runner: DreamTaskRunner,
) -> DreamResult:
    try:
        if pair_run.chapter_path.exists() and not options.overwrite_chapters:
            token_report = await count_markdown_fn(pair_run.chapter_path)
            return DreamResult(
                status="skipped_existing",
                chapter_path=pair_run.chapter_path,
                chapter_tokens=token_report.tokens,
                ceiling_tokens=pair_run.stats.summary_count.tokens,
                budget_passed=token_report.tokens
                <= pair_run.stats.summary_count.tokens,
                fork_transcript_path=pair_run.fork_transcript_path,
            )

        if options.dry_run:
            return DreamResult(
                status="dry_run",
                chapter_path=pair_run.chapter_path,
                ceiling_tokens=pair_run.stats.summary_count.tokens,
                fork_transcript_path=pair_run.fork_transcript_path,
            )

        fork_transcript_path = _clone_transcript_workspace(
            source_transcript=options.transcript_path.expanduser().resolve(),
            fork_dir=pair_run.fork_dir,
        )
        if fork_transcript_path != pair_run.fork_transcript_path:
            raise RuntimeError(
                f"Unexpected fork transcript path: {fork_transcript_path}"
            )
        dream_message = _dream_user_message(pair_run)
        entity_text = await dream_task_runner(DreamTask(pair_run, dream_message))

        chapter_path = _resolve_written_chapter_path(pair_run)
        if chapter_path is None:
            return DreamResult(
                status="failed",
                chapter_path=pair_run.chapter_path,
                ceiling_tokens=pair_run.stats.summary_count.tokens,
                fork_transcript_path=pair_run.fork_transcript_path,
                entity_text=entity_text,
                error=f"Dream fork did not write {pair_run.chapter_path}.",
            )

        token_report = await count_markdown_fn(chapter_path)
        return DreamResult(
            status="written",
            chapter_path=chapter_path,
            chapter_tokens=token_report.tokens,
            ceiling_tokens=pair_run.stats.summary_count.tokens,
            budget_passed=token_report.tokens <= pair_run.stats.summary_count.tokens,
            fork_transcript_path=pair_run.fork_transcript_path,
            entity_text=entity_text,
        )
    except Exception as exc:  # noqa: BLE001 - keep batch orchestration moving.
        return DreamResult(
            status="failed",
            chapter_path=pair_run.chapter_path,
            ceiling_tokens=pair_run.stats.summary_count.tokens,
            fork_transcript_path=pair_run.fork_transcript_path,
            error=str(exc),
        )


async def _run_dream_task(task: DreamTask) -> str:
    if not task.pair_run.fork_transcript_path.exists():
        raise FileNotFoundError(task.pair_run.fork_transcript_path)

    # The fork transcript already exists, so Spell.cast() resumes it and uses
    # the transcript's recorded config. This fallback config should never be
    # used to initialize a new session unless the clone invariant is broken.
    spell = Spell(
        config=SpellbookConfig(cwd=Path.cwd()),
        transcript_path=task.pair_run.fork_transcript_path,
    )
    async with spell.cast() as entity:
        result = await entity.send(task.dream_message)
    return result.text


async def _run_director_phase(
    pair_runs: list[PairRun],
    options: DreamMergeOptions,
    *,
    director_task_runner: DirectorTaskRunner,
) -> list[PairRun]:
    semaphore = asyncio.Semaphore(options.concurrency)

    async def run_one(idx: int, pair_run: PairRun) -> tuple[int, PairRun]:
        async with semaphore:
            if options.dry_run:
                result = DirectorResult(
                    status="dry_run", review_path=pair_run.review_path
                )
            elif not pair_run.chapter_path.exists():
                result = DirectorResult(
                    status="skipped",
                    review_path=pair_run.review_path,
                    error=f"Chapter does not exist: {pair_run.chapter_path}",
                )
            else:
                result = await director_task_runner(
                    DirectorTask(
                        pair_run=pair_run,
                        director_model=options.director_model,
                    )
                )
            return idx, _replace_pair_run(pair_run, director_result=result)

    return await _run_phase_with_progress(
        pair_runs,
        coros=(run_one(idx, pair_run) for idx, pair_run in enumerate(pair_runs)),
        description="Director pass",
        show_progress=options.show_progress,
    )


async def _run_phase_with_progress(
    pair_runs: list[PairRun],
    *,
    coros: Iterable[Awaitable[tuple[int, PairRun]]],
    description: str,
    show_progress: bool,
) -> list[PairRun]:
    tasks = [asyncio.ensure_future(coro) for coro in coros]
    results: list[PairRun | None] = [None] * len(pair_runs)
    if not tasks:
        return []

    if not show_progress:
        for idx, result in await asyncio.gather(*tasks):
            results[idx] = result
        return _completed_pair_runs(results)

    with _progress() as progress:
        task_id = progress.add_task(description, total=len(tasks))
        for task in asyncio.as_completed(tasks):
            idx, result = await task
            results[idx] = result
            progress.advance(task_id)
    return _completed_pair_runs(results)


def _progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("elapsed"),
        TimeElapsedColumn(),
        TextColumn("remaining"),
        TimeRemainingColumn(),
    )


def _completed_pair_runs(results: list[PairRun | None]) -> list[PairRun]:
    completed: list[PairRun] = []
    for result in results:
        if result is None:
            raise RuntimeError("Phase task completed without a recorded result.")
        completed.append(result)
    return completed


async def _run_director_task(task: DirectorTask) -> DirectorResult:
    try:
        report = await run_director_review(
            pair=task.pair_run.pair,
            chapter_path=task.pair_run.chapter_path,
            transcript_path=task.pair_run.stats.transcript_path,
            render_path=task.pair_run.render_path,
            output_path=task.pair_run.review_path,
            director_model=task.director_model,
            director_transcript_path=(
                task.pair_run.review_path.parent
                / "director-transcripts"
                / f"review-{task.pair_run.pair.chapter_number:02d}-{uuid4().hex}.jsonl"
            ),
        )
        pin_expansion = report.to_json_dict().get("pin_expansion", {})
        unmatched = ()
        unused = ()
        if isinstance(pin_expansion, dict):
            pin_expansion = cast(dict[str, object], pin_expansion)
            unmatched = tuple(
                str(item)
                for item in _object_sequence(pin_expansion.get("unmatched_markers"))
            )
            unused = tuple(
                str(item) for item in _object_sequence(pin_expansion.get("unused_pins"))
            )
        return DirectorResult(
            status="reviewed",
            verdict=report.review.verdict,
            review_path=report.output_path,
            chapter_tokens=report.request.chapter_tokens,
            ceiling_tokens=report.request.ceiling_tokens,
            budget_passed=report.request.chapter_tokens
            <= report.request.ceiling_tokens,
            coverage_missing=tuple(report.review.coverage_missing),
            accuracy_flags=tuple(report.review.accuracy_flags),
            unmatched_pin_markers=unmatched,
            unused_source_pins=unused,
        )
    except Exception as exc:  # noqa: BLE001 - keep batch orchestration moving.
        return DirectorResult(
            status="failed",
            review_path=task.pair_run.review_path,
            error=str(exc),
        )


def _dream_user_message(pair_run: PairRun) -> str:
    pins = _pins_markdown(pair_run.stats.pins)
    return f"""\
# Dream Merge Assignment

{QUANTUM_DICE_CONTRACT}

Write the narrative dream chapter for Block {pair_run.pair.first} + Block {pair_run.pair.second}.
Save the chapter to:

{pair_run.chapter_path}

## Constraints

- Stay under the token ceiling: {pair_run.stats.summary_count.tokens:,} tokens.
- Preserve every load-bearing fact from the two summaries.
- Use curated exchanges and hindsight marginalia; keep the Hemingway iceberg.
- Place any pinned material with a `<!-- pin: Pin Name -->` marker where it belongs in the narrative flow.
- Do not inline pinned source content unless the narrative genuinely needs a small quoted fragment.
- Include an `## Anchors` footer with key commits, dates, entity names, and file paths.
- Write only the chapter file; do not modify the live meta-Claude transcript.
- Read the rendered markdown from the filesystem path below before writing. The render file contains the full dream-opening text and the full source block material.

## Merge Stats Output

{pair_run.stats_text}

## Pins

{pins}

## Rendered Markdown Path

{pair_run.render_path}
"""


def _clone_transcript_workspace(
    *,
    source_transcript: Path,
    fork_dir: Path,
) -> Path:
    source_transcript = source_transcript.expanduser().resolve()
    fork_dir = fork_dir.expanduser().resolve()
    if fork_dir.exists():
        shutil.rmtree(fork_dir)
    fork_dir.mkdir(parents=True, exist_ok=True)
    target_transcript = fork_dir / "transcript.jsonl"
    shutil.copy2(source_transcript, target_transcript)

    for dirname in ("blobs", "tool-outputs"):
        source_dir = source_transcript.parent / dirname
        if not source_dir.exists():
            continue
        target_dir = fork_dir / dirname
        shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)

    return target_transcript


def _resolve_chapter_path(
    *,
    chapter_dir: Path,
    pair: Pair,
    stats: MergeStatsReport,
) -> Path:
    chapter_dir = chapter_dir.expanduser().resolve()
    default_path = (
        chapter_dir / f"chapter-{pair.chapter_number:02d}-{_chapter_slug(stats)}.md"
    )
    existing = sorted(chapter_dir.glob(f"chapter-{pair.chapter_number:02d}-*.md"))
    if not existing:
        return default_path
    if default_path in existing:
        return default_path
    if len(existing) == 1:
        return existing[0]
    formatted = ", ".join(str(path) for path in existing)
    raise ValueError(
        f"Multiple existing chapters match chapter {pair.chapter_number:02d}: "
        f"{formatted}. Move one aside or use a clean chapter directory."
    )


def _resolve_written_chapter_path(pair_run: PairRun) -> Path | None:
    if pair_run.chapter_path.exists():
        return pair_run.chapter_path

    for path in _written_chapter_paths_from_transcript(pair_run):
        if (
            path.exists()
            and _chapter_number_from_path(path) == pair_run.pair.chapter_number
        ):
            return path.expanduser().resolve()

    existing = sorted(
        pair_run.chapter_path.parent.glob(
            f"chapter-{pair_run.pair.chapter_number:02d}-*.md"
        )
    )
    if len(existing) == 1:
        return existing[0].expanduser().resolve()
    return None


def _written_chapter_paths_from_transcript(pair_run: PairRun) -> list[Path]:
    transcript_path = pair_run.fork_transcript_path
    if not transcript_path.exists():
        return []

    after_assignment = False
    paths: list[Path] = []
    seen: set[str] = set()
    for raw in transcript_path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(raw)
        except json.JSONDecodeError:
            continue
        event = record.get("event") or {}
        event_type = event.get("type")
        if event_type == "user_text" and "# Dream Merge Assignment" in str(
            event.get("text", "")
        ):
            after_assignment = True
            continue
        if not after_assignment:
            continue
        if event_type != "tool_result" or event.get("tool") != "Write":
            continue
        for path in _write_result_paths(event):
            key = str(path)
            if key not in seen:
                seen.add(key)
                paths.append(path)
    return paths


def _write_result_paths(event: dict[str, object]) -> list[Path]:
    paths: list[Path] = []
    display = event.get("display")
    if isinstance(display, dict):
        display = cast(dict[str, object], display)
        path = display.get("path")
        if path:
            paths.append(Path(str(path)).expanduser().resolve())

    content = event.get("content")
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            item = cast(dict[str, object], item)
            match = re.search(r"Successfully wrote to (.+)$", str(item.get("text", "")))
            if match:
                paths.append(Path(match.group(1)).expanduser().resolve())
    return paths


def _chapter_number_from_path(path: Path) -> int | None:
    match = re.match(r"chapter-(\d+)-.*\.md$", path.name)
    if match is None:
        return None
    return int(match.group(1))


def _chapter_slug(stats: MergeStatsReport) -> str:
    title = next((line.title for line in stats.block_lines if line.title), "")
    slug = _slugify(title)
    if slug:
        return slug[:64].strip("-")
    if len(stats.block_lines) >= 2:
        return f"blocks-{stats.block_lines[0].idx}-{stats.block_lines[1].idx}"
    return "chapter"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return re.sub(r"-+", "-", slug)


def _write_stats_files(
    *,
    work_dir: Path,
    pair: Pair,
    stats: MergeStatsReport,
    stats_text: str,
) -> tuple[Path, Path]:
    stats_dir = work_dir.expanduser().resolve() / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    text_path = stats_dir / f"merge-stats-{pair.slug}.txt"
    json_path = stats_dir / f"merge-stats-{pair.slug}.json"
    text_path.write_text(stats_text + "\n", encoding="utf-8")
    json_path.write_text(
        json.dumps(_stats_json_dict(stats), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return text_path, json_path


def _stats_console_output(report: MergeStatsReport) -> str:
    first, second = report.block_lines
    lines = [
        f"Merge pair: Block {first.idx} + Block {second.idx}",
    ]
    for block in report.block_lines:
        lines.append(
            f'Block {block.idx}: "{_clip(block.title)}" '
            f"(turns {block.start_block}-{block.end_block})"
        )
    lines.extend(
        [
            "",
            f"Full fidelity (both): {_format_count(report.full_count.tokens, report.full_count.exact, report.full_count.method)}",
            f"Summaries (both):      {_format_count(report.summary_count.tokens, report.summary_count.exact, report.summary_count.method)} (target ceiling)",
        ]
    )
    if report.render_path is not None:
        lines.append(f"Rendered markdown:     {report.render_path}")
    lines.append("")
    lines.extend(_pin_lines(report.pins))
    return "\n".join(lines)


def _stats_json_dict(report: MergeStatsReport) -> dict[str, object]:
    return {
        "transcript_path": str(report.transcript_path),
        "model": report.model,
        "render_path": str(report.render_path) if report.render_path else None,
        "blocks": [_block_line_json(line) for line in report.block_lines],
        "full_count": _count_json(
            report.full_count.tokens,
            report.full_count.exact,
            report.full_count.method,
        ),
        "summary_count": _count_json(
            report.summary_count.tokens,
            report.summary_count.exact,
            report.summary_count.method,
        ),
        "pins": [_pin_json(pin) for pin in report.pins],
    }


def _read_stats_json(path: Path) -> MergeStatsReport:
    data = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    return MergeStatsReport(
        transcript_path=Path(str(data["transcript_path"])).expanduser().resolve(),
        model=str(data["model"]),
        block_lines=[
            BlockLine(
                idx=int(block["idx"]),
                title=str(block["title"]),
                start_block=int(block["start_block"]),
                end_block=int(block["end_block"]),
            )
            for block in data["blocks"]
        ],
        full_count=_count_from_json(data["full_count"]),
        summary_count=_count_from_json(data["summary_count"]),
        pins=[
            PinCount(
                block_idx=int(pin["block_idx"]),
                block_title=str(pin["block_title"]),
                kind=str(pin["kind"]),
                title=str(pin["title"]),
                tokens=int(pin["tokens"]),
                exact=bool(pin["exact"]),
                method=str(pin["method"]),
                reason=str(pin["reason"]),
                facet_id=str(pin["facet_id"]) if pin.get("facet_id") else None,
            )
            for pin in data["pins"]
        ],
        render_path=Path(str(data["render_path"])).expanduser().resolve()
        if data.get("render_path")
        else None,
    )


def _block_line_json(line: BlockLine) -> dict[str, object]:
    return {
        "idx": line.idx,
        "title": line.title,
        "start_block": line.start_block,
        "end_block": line.end_block,
    }


def _pin_json(pin: PinCount) -> dict[str, object]:
    return {
        "block_idx": pin.block_idx,
        "block_title": pin.block_title,
        "kind": pin.kind,
        "title": pin.title,
        "tokens": pin.tokens,
        "exact": pin.exact,
        "method": pin.method,
        "reason": pin.reason,
        "facet_id": pin.facet_id,
    }


def _count_json(tokens: int, exact: bool, method: str) -> dict[str, object]:
    return {"tokens": tokens, "exact": exact, "method": method}


def _object_sequence(value: object) -> tuple[object, ...]:
    if isinstance(value, list | tuple):
        return tuple(value)
    return ()


def _count_from_json(data: object) -> IRTokenRangeCount:
    if not isinstance(data, dict):
        raise ValueError(f"Invalid token count JSON: {data!r}")
    return IRTokenRangeCount.model_validate(data)


def _pins_markdown(pins: list[PinCount]) -> str:
    return "\n".join(_pin_lines(pins))


def _pin_lines(pins: list[PinCount]) -> list[str]:
    if not pins:
        return ["Pins: none"]
    lines = [f"Pins: {len(pins)}"]
    for pin in pins:
        qualifier = "full block" if pin.kind == "block" else f"facet {pin.facet_id}"
        exact = "" if pin.exact else f" ({pin.method})"
        lines.append(
            f'- Block {pin.block_idx} {qualifier}: "{_clip(pin.title)}" - '
            f"{pin.tokens:,} tokens{exact}"
        )
    return lines


def _format_count(tokens: int, exact: bool, method: str) -> str:
    suffix = "" if exact else f" ({method})"
    return f"{tokens:,} tokens{suffix}"


def _clip(value: str, limit: int = 72) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "..."


def _context_warnings(
    stats: MergeStatsReport,
    options: DreamMergeOptions,
) -> list[str]:
    estimated_total = (
        options.base_context_tokens
        + stats.full_count.tokens
        + options.context_overhead_tokens
    )
    if estimated_total <= options.context_window_tokens:
        return []
    return [
        (
            "Estimated fork context exceeds window: "
            f"{estimated_total:,} / {options.context_window_tokens:,} tokens "
            f"(base {options.base_context_tokens:,} + full render "
            f"{stats.full_count.tokens:,} + overhead {options.context_overhead_tokens:,})."
        )
    ]


def _select_canonical_fork(
    pair_runs: list[PairRun],
    options: DreamMergeOptions,
) -> Path | None:
    if options.director_only or options.dry_run:
        return None
    candidates = [
        pair_run.fork_transcript_path
        for pair_run in pair_runs
        if pair_run.dream_result is not None
        and pair_run.dream_result.status in {"written", "skipped_existing"}
        and pair_run.fork_transcript_path.exists()
    ]
    if not candidates:
        return None
    return random.SystemRandom().choice(candidates)


def _replace_pair_run(
    pair_run: PairRun,
    *,
    chapter_path: Path | None = None,
    dream_result: DreamResult | None = None,
    director_result: DirectorResult | None = None,
) -> PairRun:
    return PairRun(
        pair=pair_run.pair,
        stats=pair_run.stats,
        stats_text=pair_run.stats_text,
        stats_text_path=pair_run.stats_text_path,
        stats_json_path=pair_run.stats_json_path,
        render_path=pair_run.render_path,
        chapter_path=chapter_path or pair_run.chapter_path,
        review_path=pair_run.review_path,
        fork_dir=pair_run.fork_dir,
        fork_transcript_path=pair_run.fork_transcript_path,
        warnings=pair_run.warnings,
        dream_result=dream_result
        if dream_result is not None
        else pair_run.dream_result,
        director_result=director_result
        if director_result is not None
        else pair_run.director_result,
    )


def _parse_pairs(value: str) -> tuple[Pair, ...]:
    pairs: list[Pair] = []
    seen: set[tuple[int, int]] = set()
    for raw_pair in value.split(","):
        part = raw_pair.strip()
        if not part:
            continue
        try:
            first_raw, second_raw = part.split("-", maxsplit=1)
            pair = Pair(first=int(first_raw), second=int(second_raw))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f'Pair must look like "N-N+1"; got {part!r}.'
            ) from exc
        if pair.second != pair.first + 1:
            raise argparse.ArgumentTypeError(
                f"Pair {part!r} must contain adjacent block indices."
            )
        key = (pair.first, pair.second)
        if key in seen:
            raise argparse.ArgumentTypeError(f"Duplicate pair {part!r}.")
        seen.add(key)
        pairs.append(pair)
    if not pairs:
        raise argparse.ArgumentTypeError("At least one pair is required.")
    return tuple(pairs)


def _default_work_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return DEFAULT_WORK_ROOT / f"run-{stamp}-{uuid4().hex[:8]}"


def _validate_options(options: DreamMergeOptions) -> None:
    if options.dream_only and options.director_only:
        raise ValueError("--dream-only and --director-only cannot both be set.")
    if options.concurrency < 1:
        raise ValueError("--concurrency must be at least 1.")
    if not options.transcript_path.expanduser().exists():
        raise FileNotFoundError(options.transcript_path.expanduser())


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="dream_merge",
        description="Generate dream chapters for merge pairs and review them.",
    )
    parser.add_argument(
        "--pairs",
        type=_parse_pairs,
        required=True,
        help='Comma-separated adjacent block pairs, e.g. "0-1,2-3,4-5".',
    )
    parser.add_argument(
        "--transcript",
        type=Path,
        default=DEFAULT_TRANSCRIPT,
        help=f"Meta-Claude transcript path. Defaults to {DEFAULT_TRANSCRIPT}.",
    )
    parser.add_argument(
        "--chapter-dir",
        type=Path,
        default=DEFAULT_DREAM_CHAPTER_DIR,
        help=f"Dream chapter directory. Defaults to {DEFAULT_DREAM_CHAPTER_DIR}.",
    )
    parser.add_argument(
        "--review-dir",
        type=Path,
        default=DEFAULT_DREAM_REVIEW_DIR,
        help=f"Director review directory. Defaults to {DEFAULT_DREAM_REVIEW_DIR}.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Run workspace. Defaults to a unique directory under /tmp/spellbook-dream-merge.",
    )
    parser.add_argument(
        "--director-model",
        default=DEFAULT_DIRECTOR_MODEL,
        help=f"Director model. Defaults to {DEFAULT_DIRECTOR_MODEL}.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Maximum concurrent dream or Director tasks. Defaults to 3.",
    )
    parser.add_argument(
        "--dream-only",
        action="store_true",
        help="Run preflight and dream pass only; skip Director reviews.",
    )
    parser.add_argument(
        "--director-only",
        action="store_true",
        help="Run preflight and Director reviews only; skip dream generation.",
    )
    parser.add_argument(
        "--overwrite-chapters",
        action="store_true",
        help="Run dream forks even when a chapter-NN-*.md file already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run preflight only and print planned paths; skip model calls.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable Rich progress bars for dream and Director phases.",
    )
    parser.add_argument(
        "--reuse-preflight",
        action="store_true",
        help=(
            "Reuse existing stats JSON and rendered markdown from --work-dir instead "
            "of re-running merge_stats token counts."
        ),
    )
    parser.add_argument(
        "--base-context-tokens",
        type=int,
        default=DEFAULT_BASE_CONTEXT_TOKENS,
        help=(
            "Approximate shared transcript prefix for context-window warnings. "
            f"Defaults to {DEFAULT_BASE_CONTEXT_TOKENS:,}."
        ),
    )
    parser.add_argument(
        "--context-window-tokens",
        type=int,
        default=DEFAULT_CONTEXT_WINDOW_TOKENS,
        help=f"Context-window warning threshold. Defaults to {DEFAULT_CONTEXT_WINDOW_TOKENS:,}.",
    )
    parser.add_argument(
        "--context-overhead-tokens",
        type=int,
        default=DEFAULT_CONTEXT_OVERHEAD_TOKENS,
        help=f"Prompt overhead estimate. Defaults to {DEFAULT_CONTEXT_OVERHEAD_TOKENS:,}.",
    )
    parser.add_argument(
        "--env",
        type=Path,
        default=DEFAULT_ENV_PATH,
        help=f"Dotenv file to load before API calls. Defaults to {DEFAULT_ENV_PATH}.",
    )
    return parser.parse_args(argv)


def _options_from_args(args: argparse.Namespace) -> DreamMergeOptions:
    return DreamMergeOptions(
        pairs=args.pairs,
        transcript_path=args.transcript,
        chapter_dir=args.chapter_dir,
        review_dir=args.review_dir,
        work_dir=args.work_dir or _default_work_dir(),
        director_model=args.director_model,
        concurrency=args.concurrency,
        dream_only=args.dream_only,
        director_only=args.director_only,
        overwrite_chapters=args.overwrite_chapters,
        dry_run=args.dry_run,
        show_progress=not args.no_progress,
        reuse_preflight=args.reuse_preflight,
        base_context_tokens=args.base_context_tokens,
        context_window_tokens=args.context_window_tokens,
        context_overhead_tokens=args.context_overhead_tokens,
    )


async def _async_main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    env_path = args.env.expanduser()
    if env_path.exists():
        load_dotenv(env_path)
    report = await run_dream_merge(_options_from_args(args))
    _print_run_summary(report)


def _print_run_summary(report: DreamMergeRunReport) -> None:
    print(f"Work dir: {report.options.work_dir.expanduser().resolve()}")
    print(f"Chapters: {report.options.chapter_dir.expanduser().resolve()}")
    print(f"Reviews:  {report.options.review_dir.expanduser().resolve()}")
    if report.canonical_fork is not None:
        print(f"Quantum Dice selected canonical fork: {report.canonical_fork}")
    print("")
    print(
        "Pair   Chapter                                      Tokens        Dream             Director"
    )
    print(
        "-----  -------------------------------------------  ------------  ----------------  ----------------"
    )
    for pair_run in report.pair_runs:
        dream = pair_run.dream_result
        director = pair_run.director_result
        tokens = "-"
        if dream is not None and dream.chapter_tokens is not None:
            tokens = f"{dream.chapter_tokens:,}/{pair_run.stats.summary_count.tokens:,}"
        elif director is not None and director.chapter_tokens is not None:
            ceiling = director.ceiling_tokens or pair_run.stats.summary_count.tokens
            tokens = f"{director.chapter_tokens:,}/{ceiling:,}"
        dream_status = dream.status if dream is not None else "-"
        director_status = "-"
        if director is not None:
            director_status = director.verdict or director.status
        print(
            f"{pair_run.pair.slug:<5}  "
            f"{pair_run.chapter_path.name[:43]:<43}  "
            f"{tokens:<12}  "
            f"{dream_status:<16}  "
            f"{director_status:<16}"
        )

    _print_warnings(report.pair_runs)
    _print_failures(report.pair_runs)
    _print_director_issues(report.pair_runs)


def _print_warnings(pair_runs: tuple[PairRun, ...]) -> None:
    warnings = [
        (pair_run, warning) for pair_run in pair_runs for warning in pair_run.warnings
    ]
    if not warnings:
        return
    print("\nContext warnings:")
    for pair_run, warning in warnings:
        print(f"- {pair_run.pair.slug}: {warning}")


def _print_failures(pair_runs: tuple[PairRun, ...]) -> None:
    failures: list[tuple[str, str]] = []
    for pair_run in pair_runs:
        if pair_run.dream_result is not None and pair_run.dream_result.error:
            failures.append((pair_run.pair.slug, pair_run.dream_result.error))
        if pair_run.director_result is not None and pair_run.director_result.error:
            failures.append((pair_run.pair.slug, pair_run.director_result.error))
    if not failures:
        return
    print("\nFailures:")
    for pair_slug, error in failures:
        print(f"- {pair_slug}: {error}")


def _print_director_issues(pair_runs: tuple[PairRun, ...]) -> None:
    coverage = [
        (pair_run, item)
        for pair_run in pair_runs
        if pair_run.director_result is not None
        for item in pair_run.director_result.coverage_missing
    ]
    accuracy = [
        (pair_run, item)
        for pair_run in pair_runs
        if pair_run.director_result is not None
        for item in pair_run.director_result.accuracy_flags
    ]
    pin_issues = [
        (pair_run, "unmatched marker", item)
        for pair_run in pair_runs
        if pair_run.director_result is not None
        for item in pair_run.director_result.unmatched_pin_markers
    ] + [
        (pair_run, "unplaced source pin", item)
        for pair_run in pair_runs
        if pair_run.director_result is not None
        for item in pair_run.director_result.unused_source_pins
    ]
    if coverage:
        print("\nCoverage gaps:")
        for pair_run, item in coverage:
            print(f"- {pair_run.review_path.name} ({pair_run.pair.slug}): {item}")
    if accuracy:
        print("\nAccuracy flags:")
        for pair_run, item in accuracy:
            print(f"- {pair_run.review_path.name} ({pair_run.pair.slug}): {item}")
    if pin_issues:
        print("\nPin placement issues:")
        for pair_run, kind, item in pin_issues:
            print(
                f"- {pair_run.review_path.name} ({pair_run.pair.slug}) {kind}: {item}"
            )


def main(argv: list[str] | None = None) -> None:
    asyncio.run(_async_main(argv))


if __name__ == "__main__":
    main()

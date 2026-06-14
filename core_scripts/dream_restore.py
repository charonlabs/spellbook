"""Restore triaged keep items into dream chapters with surgical Edit passes."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from dotenv import load_dotenv

from core_scripts.dream_director import DEFAULT_ENV_PATH, Pair
from core_scripts.dream_merge import (
    DEFAULT_DREAM_CHAPTER_DIR,
    _progress,
    _resolve_chapter_path,
    _stats_console_output,
)
from core_scripts.dream_triage import DEFAULT_TRIAGE_OUTPUT
from core_scripts.merge_stats import DEFAULT_TRANSCRIPT, MergeStatsReport, merge_stats
from spellbook.backends import infer_provider_for_model
from spellbook.config import HomunculusConfig, SpellbookConfig
from spellbook.sdk import Spell

DEFAULT_RESTORE_MODEL = "claude-opus-4-6"
DEFAULT_WORK_ROOT = Path("/tmp/spellbook-dream-restore")

RestoreStatus = Literal["restored", "dry_run", "failed"]
MergeStatsFn = Callable[..., Awaitable[MergeStatsReport]]
RestoreTaskRunner = Callable[["RestoreTask"], Awaitable[str]]

RESTORE_SYSTEM_PROMPT = """\
You are restoring missing keep items into existing narrative dream chapters.

This is a surgical restoration pass. Preserve the existing chapter's voice,
shape, pacing, title, and good paragraphs. Do not rewrite sections that already
work. Weave each missing item into the place it naturally belongs in the
chronology.

Use the Read tool to inspect the target chapter, then use the Edit tool to
modify it in place. Do not use Write. Modify only the target chapter file.

For each keep item:
- Verify it against the rendered source blocks.
- Insert it as a near-verbatim exchange, hindsight marginalia, narrated tool
  call, architectural beat, or Anchors entry, whichever fits the chapter.
- Prefer near-verbatim source exchanges when the source has a strong quote.
- Preserve the first-person dream voice and bracketed marginalia style.
- Keep the Hemingway iceberg: include enough to restore meaning without turning
  the chapter into a flat summary.

Rendered source blocks may contain <pin>...</pin> sections. If a keep item is
best represented by pinned material, you may add or move the relevant
<!-- pin: Name --> marker where it belongs instead of restating the full pin.

End your turn with a quick summary/overview of the changes you made, organized
by keep item when possible. The file edits are the output that matters, but the
final message should help Ryan quickly inspect what landed.
"""


@dataclass(frozen=True)
class KeepItem:
    chapter_id: str
    chapter_number: int
    item: str
    category: str
    reason: str


@dataclass(frozen=True)
class RestoreOptions:
    triage_path: Path = DEFAULT_TRIAGE_OUTPUT
    transcript_path: Path = DEFAULT_TRANSCRIPT
    chapter_dir: Path = DEFAULT_DREAM_CHAPTER_DIR
    work_dir: Path = field(default_factory=lambda: _default_work_dir())
    model: str = DEFAULT_RESTORE_MODEL
    concurrency: int = 3
    dry_run: bool = False
    show_progress: bool = True
    chapters: tuple[int, ...] | None = None


@dataclass(frozen=True)
class RestorePlan:
    chapter_id: str
    chapter_number: int
    pair: Pair
    keep_items: tuple[KeepItem, ...]
    stats: MergeStatsReport
    stats_text: str
    render_path: Path
    chapter_path: Path
    transcript_path: Path
    user_message: str


@dataclass(frozen=True)
class RestoreResult:
    status: RestoreStatus
    chapter_path: Path
    transcript_path: Path | None = None
    keep_count: int = 0
    edit_calls: int = 0
    write_calls: int = 0
    changed: bool = False
    entity_text: str = ""
    error: str | None = None


@dataclass(frozen=True)
class RestoreRunReport:
    options: RestoreOptions
    plans: tuple[RestorePlan, ...]
    results: tuple[RestoreResult, ...]


@dataclass(frozen=True)
class RestoreTask:
    plan: RestorePlan
    model: str


async def run_restore(
    options: RestoreOptions,
    *,
    merge_stats_fn: MergeStatsFn = merge_stats,
    restore_task_runner: RestoreTaskRunner | None = None,
) -> RestoreRunReport:
    _validate_options(options)
    plans = await _prepare_restore_plans(options, merge_stats_fn=merge_stats_fn)
    runner = restore_task_runner or _run_restore_task
    results = await _run_restore_phase(plans, options, restore_task_runner=runner)
    return RestoreRunReport(
        options=options,
        plans=tuple(plans),
        results=tuple(results),
    )


async def _prepare_restore_plans(
    options: RestoreOptions,
    *,
    merge_stats_fn: MergeStatsFn,
) -> list[RestorePlan]:
    keep_by_chapter = _load_keep_items(options.triage_path)
    if options.chapters is not None:
        keep_by_chapter = {
            chapter_number: items
            for chapter_number, items in keep_by_chapter.items()
            if chapter_number in options.chapters
        }

    transcript_path = options.transcript_path.expanduser().resolve()
    chapter_dir = options.chapter_dir.expanduser().resolve()
    work_dir = options.work_dir.expanduser().resolve()
    plans: list[RestorePlan] = []
    for chapter_number in sorted(keep_by_chapter):
        pair = _pair_for_chapter(chapter_number)
        render_path = work_dir / "renders" / f"restore-blocks-{pair.slug}.md"
        stats = await merge_stats_fn(
            transcript_path=transcript_path,
            first_idx=pair.first,
            second_idx=pair.second,
            render_path=render_path,
        )
        resolved_render_path = (stats.render_path or render_path).expanduser().resolve()
        stats = MergeStatsReport(
            transcript_path=transcript_path,
            model=stats.model,
            block_lines=stats.block_lines,
            full_count=stats.full_count,
            summary_count=stats.summary_count,
            pins=stats.pins,
            render_path=resolved_render_path,
        )
        chapter_path = _resolve_chapter_path(
            chapter_dir=chapter_dir,
            pair=pair,
            stats=stats,
        )
        transcript = (
            work_dir
            / "restore-transcripts"
            / f"chapter-{chapter_number:02d}-{uuid4().hex}.jsonl"
        )
        stats_text = _stats_console_output(stats)
        user_message = _restore_user_message(
            chapter_path=chapter_path,
            keep_items=tuple(keep_by_chapter[chapter_number]),
            pair=pair,
            render_path=resolved_render_path,
            stats_text=stats_text,
        )
        plans.append(
            RestorePlan(
                chapter_id=f"chapter_{chapter_number:02d}",
                chapter_number=chapter_number,
                pair=pair,
                keep_items=tuple(keep_by_chapter[chapter_number]),
                stats=stats,
                stats_text=stats_text,
                render_path=resolved_render_path,
                chapter_path=chapter_path,
                transcript_path=transcript,
                user_message=user_message,
            )
        )
    return plans


async def _run_restore_phase(
    plans: list[RestorePlan],
    options: RestoreOptions,
    *,
    restore_task_runner: RestoreTaskRunner,
) -> list[RestoreResult]:
    semaphore = asyncio.Semaphore(options.concurrency)

    async def run_one(idx: int, plan: RestorePlan) -> tuple[int, RestoreResult]:
        async with semaphore:
            result = await _run_one_restore(
                plan,
                options,
                restore_task_runner=restore_task_runner,
            )
            return idx, result

    return await _run_results_with_progress(
        plans,
        coros=(run_one(idx, plan) for idx, plan in enumerate(plans)),
        description="Restore pass",
        show_progress=options.show_progress,
    )


async def _run_results_with_progress(
    plans: list[RestorePlan],
    *,
    coros: Iterable[Awaitable[tuple[int, RestoreResult]]],
    description: str,
    show_progress: bool,
) -> list[RestoreResult]:
    tasks = [asyncio.create_task(coro) for coro in coros]
    results: list[RestoreResult | None] = [None] * len(plans)
    if not tasks:
        return []

    if not show_progress:
        for idx, result in await asyncio.gather(*tasks):
            results[idx] = result
        return _completed_results(results)

    with _progress() as progress:
        task_id = progress.add_task(description, total=len(tasks))
        for task in asyncio.as_completed(tasks):
            idx, result = await task
            results[idx] = result
            progress.advance(task_id)
    return _completed_results(results)


def _completed_results(results: list[RestoreResult | None]) -> list[RestoreResult]:
    completed: list[RestoreResult] = []
    for result in results:
        if result is None:
            raise RuntimeError("Restore task completed without a recorded result.")
        completed.append(result)
    return completed


async def _run_one_restore(
    plan: RestorePlan,
    options: RestoreOptions,
    *,
    restore_task_runner: RestoreTaskRunner,
) -> RestoreResult:
    if not plan.chapter_path.exists():
        return RestoreResult(
            status="failed",
            chapter_path=plan.chapter_path,
            transcript_path=plan.transcript_path,
            keep_count=len(plan.keep_items),
            error=f"Chapter does not exist: {plan.chapter_path}",
        )
    if options.dry_run:
        return RestoreResult(
            status="dry_run",
            chapter_path=plan.chapter_path,
            transcript_path=plan.transcript_path,
            keep_count=len(plan.keep_items),
        )

    try:
        before_hash = _file_hash(plan.chapter_path)
        entity_text = await restore_task_runner(
            RestoreTask(plan=plan, model=options.model)
        )
        calls = _tool_call_counts(plan.transcript_path)
        after_hash = _file_hash(plan.chapter_path)
    except Exception as exc:  # noqa: BLE001 - keep batch orchestration moving.
        return RestoreResult(
            status="failed",
            chapter_path=plan.chapter_path,
            transcript_path=plan.transcript_path,
            keep_count=len(plan.keep_items),
            error=str(exc),
        )

    changed = before_hash != after_hash
    edit_calls = calls.get("Edit", 0)
    write_calls = calls.get("Write", 0)
    errors: list[str] = []
    if write_calls:
        errors.append(f"Model used Write {write_calls} time(s); expected Edit only.")
    if edit_calls == 0:
        errors.append("Model did not call Edit.")
    if not changed:
        errors.append("Chapter file did not change.")

    if errors:
        return RestoreResult(
            status="failed",
            chapter_path=plan.chapter_path,
            transcript_path=plan.transcript_path,
            keep_count=len(plan.keep_items),
            edit_calls=edit_calls,
            write_calls=write_calls,
            changed=changed,
            entity_text=entity_text,
            error=" ".join(errors),
        )
    return RestoreResult(
        status="restored",
        chapter_path=plan.chapter_path,
        transcript_path=plan.transcript_path,
        keep_count=len(plan.keep_items),
        edit_calls=edit_calls,
        write_calls=write_calls,
        changed=changed,
        entity_text=entity_text,
    )


async def _run_restore_task(task: RestoreTask) -> str:
    spell = Spell(
        config=_restore_config(task.model),
        transcript_path=task.plan.transcript_path,
    )
    result = await spell.once(task.plan.user_message)
    return result.text


def _restore_config(model: str) -> SpellbookConfig:
    return SpellbookConfig(
        provider=infer_provider_for_model(model),
        model=model,
        cwd=Path.cwd(),
        max_output_tokens=32_000,
        system_prompt=RESTORE_SYSTEM_PROMPT,
        hom_config=HomunculusConfig(detect_interval=10_000),
    )


def _restore_user_message(
    *,
    chapter_path: Path,
    keep_items: tuple[KeepItem, ...],
    pair: Pair,
    render_path: Path,
    stats_text: str,
) -> str:
    render_text = render_path.read_text(encoding="utf-8")
    keep_lines: list[str] = []
    for idx, item in enumerate(keep_items, start=1):
        keep_lines.extend(
            [
                f"{idx}. [{item.category}] {item.item}",
                f"   Triage reason: {item.reason}",
            ]
        )
    keep_markdown = "\n".join(keep_lines)
    return f"""\
# Dream Chapter Restoration

Here is a dream chapter that needs a few items restored. Below are the items to
weave in, the full source blocks for reference, and the chapter file path.

Use the Read tool to inspect the chapter file, then use the Edit tool to insert
each item where it naturally belongs in the narrative: as a near-verbatim
exchange, a marginalia note, a narrated tool call, or an Anchors entry, whatever
fits. Preserve the existing voice and flow. Do not rewrite sections that are
already good. Do not use Write.

End your turn with a quick summary/overview of the changes you made, organized
by keep item when possible. That final message will be printed alongside the
original triage keep items for human review.

## Chapter File Path

{chapter_path}

## Block Pair

Block {pair.first} + Block {pair.second}

## Keep Items

{keep_markdown}

## Merge Stats

{stats_text}

## Rendered Source Blocks

Source path: {render_path}

<rendered_source_blocks>
{render_text}
</rendered_source_blocks>
"""


def _load_keep_items(triage_path: Path) -> dict[int, list[KeepItem]]:
    data = json.loads(triage_path.expanduser().read_text(encoding="utf-8"))
    keep_by_chapter: dict[int, list[KeepItem]] = {}
    for chapter_id, groups in sorted(data.items()):
        chapter_number = _chapter_number_from_id(chapter_id)
        keep_items = groups.get("keep", []) if isinstance(groups, dict) else []
        if not keep_items:
            continue
        keep_by_chapter[chapter_number] = [
            KeepItem(
                chapter_id=chapter_id,
                chapter_number=chapter_number,
                item=str(item["item"]),
                category=str(item["category"]),
                reason=str(item["reason"]),
            )
            for item in keep_items
        ]
    return keep_by_chapter


def _chapter_number_from_id(chapter_id: str) -> int:
    prefix = "chapter_"
    if not chapter_id.startswith(prefix):
        raise ValueError(f"Invalid chapter id: {chapter_id!r}")
    return int(chapter_id.removeprefix(prefix))


def _pair_for_chapter(chapter_number: int) -> Pair:
    if chapter_number < 1:
        raise ValueError(f"Chapter number must be positive: {chapter_number}")
    first = (chapter_number - 1) * 2
    return Pair(first=first, second=first + 1)


def _parse_chapters(value: str) -> tuple[int, ...]:
    chapters: list[int] = []
    seen: set[int] = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", maxsplit=1)
            start = int(start_raw)
            end = int(end_raw)
            if end < start:
                raise argparse.ArgumentTypeError(
                    f"Chapter range {part!r} must be ascending."
                )
            values = range(start, end + 1)
        else:
            values = (int(part),)
        for chapter in values:
            if chapter < 1:
                raise argparse.ArgumentTypeError("Chapter numbers must be positive.")
            if chapter not in seen:
                seen.add(chapter)
                chapters.append(chapter)
    if not chapters:
        raise argparse.ArgumentTypeError("At least one chapter is required.")
    return tuple(chapters)


def _tool_call_counts(transcript_path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not transcript_path.exists():
        return counts
    for raw in transcript_path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(raw)
        except json.JSONDecodeError:
            continue
        event = record.get("event") or {}
        if event.get("type") != "tool_call":
            continue
        tool = str(event.get("tool", ""))
        counts[tool] = counts.get(tool, 0) + 1
    return counts


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _default_work_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return DEFAULT_WORK_ROOT / f"run-{stamp}-{uuid4().hex[:8]}"


def _validate_options(options: RestoreOptions) -> None:
    if options.concurrency < 1:
        raise ValueError("--concurrency must be at least 1.")
    if not options.triage_path.expanduser().exists():
        raise FileNotFoundError(options.triage_path.expanduser())
    if not options.transcript_path.expanduser().exists():
        raise FileNotFoundError(options.transcript_path.expanduser())


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="dream_restore",
        description="Weave triaged keep items back into dream chapters with Edit.",
    )
    parser.add_argument(
        "--triage",
        type=Path,
        default=DEFAULT_TRIAGE_OUTPUT,
        help=f"Triage JSON path. Defaults to {DEFAULT_TRIAGE_OUTPUT}.",
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
        "--work-dir",
        type=Path,
        default=None,
        help="Run workspace. Defaults to a unique directory under /tmp/spellbook-dream-restore.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_RESTORE_MODEL,
        help=f"Restoration model. Defaults to {DEFAULT_RESTORE_MODEL}.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Maximum concurrent restoration tasks. Defaults to 3.",
    )
    parser.add_argument(
        "--chapters",
        type=_parse_chapters,
        default=None,
        help='Optional chapter filter, e.g. "3,5-8". Defaults to all keep chapters.',
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render and print planned restorations, but skip model calls.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the Rich progress bar.",
    )
    parser.add_argument(
        "--env",
        type=Path,
        default=DEFAULT_ENV_PATH,
        help=f"Dotenv file to load before API calls. Defaults to {DEFAULT_ENV_PATH}.",
    )
    return parser.parse_args(argv)


def _options_from_args(args: argparse.Namespace) -> RestoreOptions:
    return RestoreOptions(
        triage_path=args.triage,
        transcript_path=args.transcript,
        chapter_dir=args.chapter_dir,
        work_dir=args.work_dir or _default_work_dir(),
        model=args.model,
        concurrency=args.concurrency,
        dry_run=args.dry_run,
        show_progress=not args.no_progress,
        chapters=args.chapters,
    )


async def _async_main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    env_path = args.env.expanduser()
    if env_path.exists():
        load_dotenv(env_path)
    report = await run_restore(_options_from_args(args))
    _print_run_summary(report)


def _print_run_summary(report: RestoreRunReport) -> None:
    print(f"Work dir: {report.options.work_dir.expanduser().resolve()}")
    print(f"Triage:   {report.options.triage_path.expanduser().resolve()}")
    print(f"Chapters: {report.options.chapter_dir.expanduser().resolve()}")
    print("")
    print("Chapter     Pair   Keep  Edits  Status    File")
    print(
        "----------  -----  ----  -----  --------  -------------------------------------------"
    )
    for plan, result in zip(report.plans, report.results, strict=True):
        edits = "-" if result.status == "dry_run" else str(result.edit_calls)
        print(
            f"{plan.chapter_id:<10}  "
            f"{plan.pair.slug:<5}  "
            f"{len(plan.keep_items):>4}  "
            f"{edits:>5}  "
            f"{result.status:<8}  "
            f"{plan.chapter_path.name[:43]:<43}"
        )

    failures = [
        (plan, result)
        for plan, result in zip(report.plans, report.results, strict=True)
        if result.error
    ]
    if failures:
        print("")
        print("Failures:")
        for plan, result in failures:
            print(f"- {plan.chapter_id} ({plan.pair.slug}): {result.error}")

    _print_restoration_details(report)


def _print_restoration_details(report: RestoreRunReport) -> None:
    if not report.plans:
        return
    print("")
    print("Restoration details:")
    for plan, result in zip(report.plans, report.results, strict=True):
        print("")
        print(f"{plan.chapter_id} ({plan.pair.slug}) - {plan.chapter_path}")
        print("Keep items:")
        for idx, item in enumerate(plan.keep_items, start=1):
            print(f"{idx}. [{item.category}] {item.item}")
            print(f"   Reason: {item.reason}")
        print("")
        print("Opus summary:")
        summary = result.entity_text.strip()
        if summary:
            print(summary)
        elif result.status == "dry_run":
            print("(dry run; no model call)")
        else:
            print("(no final message captured)")


def main(argv: list[str] | None = None) -> None:
    asyncio.run(_async_main(argv))


if __name__ == "__main__":
    main()

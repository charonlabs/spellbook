"""Drain an existing transcript's raw context tail through detection and summaries."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from core_scripts.merge_stats import DEFAULT_ENV_PATH, DEFAULT_TRANSCRIPT
from spellbook.backends import build_backend
from spellbook.config import SpellbookConfig
from spellbook.footer import FooterController
from spellbook.fork import ForkRunner
from spellbook.homunculus.block_manager import BlockManager
from spellbook.homunculus.token_meter import TokenMeter
from spellbook.homunculus.tool_result_ttl import (
    AUTO_TTL_SKIP_TOOLS,
    ToolResultTTLRegistry,
    tool_result_ttl_content,
)
from spellbook.inbound import InboundMessageQueue
from spellbook.ir_types import (
    IRBlock,
    IRRecord,
    IRSemanticBlock,
    IRSemanticBlockSummary,
    IRToolResultBlock,
    IRTurnEndRecord,
)
from spellbook.nursery import Nursery, NurseryJobKind, NurseryJobResult
from spellbook.recorder import Recorder
from spellbook.rehydrator import RehydrationResult, Rehydrator
from spellbook.session_manager import SessionManager
from spellbook.surface_builder import RequestSurfaceBuilder
from spellbook.tools.registry import ToolRegistry


RuntimeBuilder = Callable[
    [SpellbookConfig, Path, Recorder, Callable[[Sequence[IRBlock]], list[IRBlock]]],
    "DrainRuntime",
]
BufferedPolicy = Literal["discard", "preserve"]


@dataclass(frozen=True)
class DrainTickReport:
    tick: int
    start_block: int
    end_block: int
    semantic_blocks: int
    summaries: int

    def to_dict(self) -> dict[str, object]:
        return {
            "tick": self.tick,
            "start_block": self.start_block,
            "end_block": self.end_block,
            "semantic_blocks": self.semantic_blocks,
            "summaries": self.summaries,
        }


@dataclass(frozen=True)
class LargeUntrackedToolResult:
    block_idx: int
    call_id: str
    tool: str
    chars: int
    lines: int
    images: int = 0
    image_bytes: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "block_idx": self.block_idx,
            "call_id": self.call_id,
            "tool": self.tool,
            "chars": self.chars,
            "lines": self.lines,
            "images": self.images,
            "image_bytes": self.image_bytes,
        }


@dataclass
class DrainBacklogReport:
    transcript_path: Path
    backup_path: Path | None
    dry_run: bool
    buffered_policy: BufferedPolicy
    source_blocks: int
    start_block: int
    target_end_block: int
    chunk_size: int
    interval: int
    starting_semantic_blocks: int
    starting_summaries: int
    starting_buffered_blocks: int
    ignored_buffered_blocks: int
    processed_blocks: int = 0
    detector_passes: int = 0
    finalization_passes: int = 0
    final_semantic_blocks: int | None = None
    final_summaries: int | None = None
    final_buffered_blocks: int | None = None
    large_untracked: list[LargeUntrackedToolResult] = field(default_factory=list)
    ticks: list[DrainTickReport] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def target_blocks(self) -> int:
        return max(0, self.target_end_block - self.start_block)

    @property
    def new_semantic_blocks(self) -> int:
        final = self.final_semantic_blocks
        if final is None:
            return 0
        return final - self.starting_semantic_blocks

    @property
    def new_summaries(self) -> int:
        final = self.final_summaries
        if final is None:
            return 0
        return final - self.starting_summaries

    def to_dict(self) -> dict[str, object]:
        return {
            "transcript_path": str(self.transcript_path),
            "backup_path": str(self.backup_path) if self.backup_path else None,
            "dry_run": self.dry_run,
            "buffered_policy": self.buffered_policy,
            "source_blocks": self.source_blocks,
            "start_block": self.start_block,
            "target_end_block": self.target_end_block,
            "target_blocks": self.target_blocks,
            "chunk_size": self.chunk_size,
            "interval": self.interval,
            "starting_semantic_blocks": self.starting_semantic_blocks,
            "starting_summaries": self.starting_summaries,
            "starting_buffered_blocks": self.starting_buffered_blocks,
            "ignored_buffered_blocks": self.ignored_buffered_blocks,
            "processed_blocks": self.processed_blocks,
            "detector_passes": self.detector_passes,
            "finalization_passes": self.finalization_passes,
            "final_semantic_blocks": self.final_semantic_blocks,
            "final_summaries": self.final_summaries,
            "final_buffered_blocks": self.final_buffered_blocks,
            "new_semantic_blocks": self.new_semantic_blocks,
            "new_summaries": self.new_summaries,
            "large_untracked": [item.to_dict() for item in self.large_untracked],
            "ticks": [tick.to_dict() for tick in self.ticks],
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class DrainRuntime:
    block_manager: BlockManager
    nursery: Nursery


@dataclass
class _DrainEventPrinter:
    console: Console
    progress: Progress | None = None
    printed_block_ids: set[str] = field(default_factory=set)
    printed_summary_ids: set[str] = field(default_factory=set)

    def prime(self, blocks: Sequence[IRSemanticBlock]) -> None:
        for block in blocks:
            self.printed_block_ids.add(block.id)
            for artifact in block.artifacts:
                if artifact.type == "summary":
                    self.printed_summary_ids.add(artifact.id)

    def emit_new(self, manager: BlockManager) -> None:
        new_blocks = [
            block
            for block in manager.semantic_blocks
            if block.id not in self.printed_block_ids
        ]
        if new_blocks:
            self._print(_render_completed_blocks(new_blocks))
            self.printed_block_ids.update(block.id for block in new_blocks)

        for block in manager.semantic_blocks:
            for artifact in block.artifacts:
                if artifact.type != "summary":
                    continue
                if artifact.id in self.printed_summary_ids:
                    continue
                self._print(_render_summary(block, artifact))
                self.printed_summary_ids.add(artifact.id)

    def _print(self, renderable: object) -> None:
        if self.progress is None:
            self.console.print(renderable)
            return

        self.progress.stop()
        try:
            self.console.print(renderable)
        finally:
            self.progress.start()


async def drain_block_backlog(
    *,
    transcript_path: Path = DEFAULT_TRANSCRIPT,
    chunk_size: int | None = None,
    interval: int | None = None,
    max_blocks: int | None = None,
    max_finalize_passes: int = 4,
    apply: bool = False,
    backup: bool = True,
    allow_unfinished: bool = False,
    allow_large_untracked: bool = False,
    buffered_policy: BufferedPolicy = "discard",
    show_progress: bool = True,
    stream_events: bool = True,
    report_json: Path | None = None,
    write_report: bool = True,
    runtime_builder: RuntimeBuilder | None = None,
) -> DrainBacklogReport:
    """Run detector and summarizer forks over the current raw context tail."""

    if chunk_size is not None and chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if interval is not None and interval <= 0:
        raise ValueError("interval must be greater than zero.")
    if max_blocks is not None and max_blocks < 0:
        raise ValueError("max_blocks must be greater than or equal to zero.")
    if max_finalize_passes < 0:
        raise ValueError("max_finalize_passes must be greater than or equal to zero.")
    if buffered_policy not in ("discard", "preserve"):
        raise ValueError("buffered_policy must be 'discard' or 'preserve'.")

    transcript_path = transcript_path.expanduser().resolve()
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript not found: {transcript_path}")

    source = Rehydrator(transcript_path).run()
    if source.is_unfinished_turn and not allow_unfinished:
        raise ValueError(
            "Transcript has an unfinished turn. Pass --allow-unfinished only if "
            "you intentionally want to append detection records after it."
        )

    drain_interval = interval or source.config.hom_config.detect_interval
    drain_chunk_size = chunk_size or drain_interval
    start_block = _append_start_block(source, buffered_policy=buffered_policy)
    target_end = _target_end_block(
        source_blocks=len(source.blocks),
        start_block=start_block,
        max_blocks=max_blocks,
    )
    large_untracked = _large_untracked_tool_results(
        source,
        transcript_path=transcript_path,
        start_block=start_block,
        target_end_block=target_end,
    )

    report = DrainBacklogReport(
        transcript_path=transcript_path,
        backup_path=None,
        dry_run=not apply,
        buffered_policy=buffered_policy,
        source_blocks=len(source.blocks),
        start_block=start_block,
        target_end_block=target_end,
        chunk_size=drain_chunk_size,
        interval=drain_interval,
        starting_semantic_blocks=len(source.semantic_blocks),
        starting_summaries=_summary_count(source.semantic_blocks),
        starting_buffered_blocks=len(source.buffered_semantic_block_ranges),
        ignored_buffered_blocks=(
            len(source.buffered_semantic_block_ranges)
            if buffered_policy == "discard"
            else 0
        ),
        large_untracked=large_untracked,
    )

    if not apply:
        _finish_report_from_rehydrated(report, source)
        if write_report:
            _write_report(report, report_json or _default_report_path(transcript_path))
        _print_report(report)
        return report

    if large_untracked and not allow_large_untracked:
        preview = ", ".join(item.call_id for item in large_untracked[:5])
        raise ValueError(
            f"{len(large_untracked)} large tool result(s) in the drain range do not "
            f"have TTL records. Run scripts/repair_tool_result_ttls.py first, or "
            f"pass --allow-large-untracked. Examples: {preview}"
        )

    backup_path = _backup_path(transcript_path) if backup else None
    if backup_path is not None:
        shutil.copy2(transcript_path, backup_path)
        report.backup_path = backup_path

    config = source.config.model_copy(
        update={
            "hom_config": source.config.hom_config.model_copy(
                update={"detect_interval": drain_interval}
            )
        }
    )
    recorder = _build_recorder(
        config=config,
        transcript_path=transcript_path,
        source=source,
    )
    ttl_registry = ToolResultTTLRegistry(config=config.hom_config, recorder=recorder)
    ttl_registry.rehydrate(
        source.tool_result_ttls,
        last_completed_turn=source.last_completed_turn,
        config_records=source.runtime_config_updates,
    )
    build_runtime = runtime_builder or _build_runtime
    runtime = build_runtime(
        config,
        transcript_path,
        recorder,
        ttl_registry.collapse_blocks,
    )
    manager = runtime.block_manager
    manager.context_blocks = list(source.blocks[:start_block])
    manager.next_block_id = start_block
    manager.rehydrate(
        source.model_copy(
            update={
                "blocks": source.blocks[:start_block],
                "buffered_semantic_block_ranges": (
                    []
                    if buffered_policy == "discard"
                    else source.buffered_semantic_block_ranges
                ),
            }
        )
    )

    try:
        await _run_drain(
            source=source,
            runtime=runtime,
            report=report,
            start_block=start_block,
            target_end=target_end,
            chunk_size=drain_chunk_size,
            max_finalize_passes=max_finalize_passes,
            show_progress=show_progress,
            stream_events=stream_events,
        )
        Rehydrator(transcript_path).run()
    except Exception as exc:
        report.errors.append(str(exc))
        if write_report:
            _write_report(report, report_json or _default_report_path(transcript_path))
        raise
    finally:
        await runtime.nursery.shutdown(cancel=True)

    if write_report:
        _write_report(report, report_json or _default_report_path(transcript_path))
    _print_report(report)
    return report


async def _run_drain(
    *,
    source: RehydrationResult,
    runtime: DrainRuntime,
    report: DrainBacklogReport,
    start_block: int,
    target_end: int,
    chunk_size: int,
    max_finalize_passes: int,
    show_progress: bool,
    stream_events: bool,
) -> None:
    manager = runtime.block_manager
    progress = _progress() if show_progress else None
    event_printer = _DrainEventPrinter(console=Console()) if stream_events else None
    if event_printer is not None:
        event_printer.prime(manager.semantic_blocks)

    if progress is None:
        await _run_drain_without_progress(
            source=source,
            runtime=runtime,
            report=report,
            start_block=start_block,
            target_end=target_end,
            chunk_size=chunk_size,
            max_finalize_passes=max_finalize_passes,
            event_printer=event_printer,
        )
        return

    with progress:
        if event_printer is not None:
            event_printer.console = progress.console
            event_printer.progress = progress
        backlog_task = progress.add_task(
            "Backlog",
            total=max(0, target_end - start_block),
        )
        detector_task = progress.add_task("Detector passes", total=None)
        summary_task = progress.add_task(
            "Summaries",
            total=max(1, len(manager.semantic_blocks)),
            completed=_summary_count(manager.semantic_blocks),
        )

        await _run_drain_loop(
            source=source,
            runtime=runtime,
            report=report,
            start_block=start_block,
            target_end=target_end,
            chunk_size=chunk_size,
            max_finalize_passes=max_finalize_passes,
            progress=progress,
            backlog_task=backlog_task,
            detector_task=detector_task,
            summary_task=summary_task,
            event_printer=event_printer,
        )


async def _run_drain_without_progress(
    *,
    source: RehydrationResult,
    runtime: DrainRuntime,
    report: DrainBacklogReport,
    start_block: int,
    target_end: int,
    chunk_size: int,
    max_finalize_passes: int,
    event_printer: "_DrainEventPrinter | None",
) -> None:
    await _run_drain_loop(
        source=source,
        runtime=runtime,
        report=report,
        start_block=start_block,
        target_end=target_end,
        chunk_size=chunk_size,
        max_finalize_passes=max_finalize_passes,
        progress=None,
        backlog_task=None,
        detector_task=None,
        summary_task=None,
        event_printer=event_printer,
    )


async def _run_drain_loop(
    *,
    source: RehydrationResult,
    runtime: DrainRuntime,
    report: DrainBacklogReport,
    start_block: int,
    target_end: int,
    chunk_size: int,
    max_finalize_passes: int,
    progress: Progress | None,
    backlog_task: int | None,
    detector_task: int | None,
    summary_task: int | None,
    event_printer: "_DrainEventPrinter | None",
) -> None:
    manager = runtime.block_manager
    nursery = runtime.nursery
    tick = 0

    for chunk_start in range(start_block, target_end, chunk_size):
        chunk_end = min(chunk_start + chunk_size, target_end)
        chunk = source.blocks[chunk_start:chunk_end]
        await _collect_ready_jobs(
            manager,
            nursery,
            kind="summarize_block",
            event_printer=event_printer,
        )
        await manager.append_context_blocks(chunk)
        report.processed_blocks += len(chunk)
        if progress is not None and backlog_task is not None:
            progress.update(backlog_task, advance=len(chunk))

        if nursery.jobs(source="block_manager", kind="detect_blocks"):
            report.detector_passes += 1
            if progress is not None and detector_task is not None:
                progress.update(
                    detector_task,
                    description=f"Detector passes ({report.detector_passes})",
                )
            await _wait_for_jobs(
                manager,
                nursery,
                kind="detect_blocks",
                event_printer=event_printer,
            )

        await _collect_ready_jobs(
            manager,
            nursery,
            kind="summarize_block",
            event_printer=event_printer,
        )
        tick += 1
        report.ticks.append(
            DrainTickReport(
                tick=tick,
                start_block=chunk_start,
                end_block=chunk_end - 1,
                semantic_blocks=len(manager.semantic_blocks),
                summaries=_summary_count(manager.semantic_blocks),
            )
        )
        _update_summary_progress(progress, summary_task, manager)

    for _ in range(max_finalize_passes):
        await _collect_ready_jobs(
            manager,
            nursery,
            kind="summarize_block",
            event_printer=event_printer,
        )
        started = await manager.force_detect(finalize=True)
        if not started:
            break
        report.detector_passes += 1
        report.finalization_passes += 1
        if progress is not None and detector_task is not None:
            progress.update(
                detector_task,
                description=f"Detector passes ({report.detector_passes})",
            )
        await _wait_for_jobs(
            manager,
            nursery,
            kind="detect_blocks",
            event_printer=event_printer,
        )
        await _collect_ready_jobs(
            manager,
            nursery,
            kind="summarize_block",
            event_printer=event_printer,
        )
        _update_summary_progress(progress, summary_task, manager)
        if not manager.has_unfinalized_detection:
            break

    await _drain_summaries(
        manager=manager,
        nursery=nursery,
        progress=progress,
        summary_task=summary_task,
        event_printer=event_printer,
    )
    report.final_semantic_blocks = len(manager.semantic_blocks)
    report.final_summaries = _summary_count(manager.semantic_blocks)
    report.final_buffered_blocks = len(manager.proposed_semantic_blocks)


async def _drain_summaries(
    *,
    manager: BlockManager,
    nursery: Nursery,
    progress: Progress | None,
    summary_task: int | None,
    event_printer: "_DrainEventPrinter | None",
) -> None:
    while True:
        before = _summary_count(manager.semantic_blocks)
        await manager.generate_next_summary()
        jobs = nursery.jobs(source="block_manager", kind="summarize_block")
        if not jobs:
            break
        await _wait_for_jobs(
            manager,
            nursery,
            kind="summarize_block",
            event_printer=event_printer,
        )
        _update_summary_progress(progress, summary_task, manager)
        after = _summary_count(manager.semantic_blocks)
        if after == before and not nursery.jobs(source="block_manager"):
            break


async def _wait_for_jobs(
    manager: BlockManager,
    nursery: Nursery,
    *,
    kind: NurseryJobKind,
    event_printer: "_DrainEventPrinter | None",
) -> list[NurseryJobResult[Any]]:
    results: list[NurseryJobResult[Any]] = []
    jobs = list(nursery.jobs(source="block_manager", kind=kind))
    for job in jobs:
        result = await nursery.wait(job.id)
        if result is None:
            continue
        await manager._integrate_nursery_result(result)
        if result.error is not None:
            raise RuntimeError(f"{kind} job failed: {result.error}") from result.error
        if result.cancelled:
            raise RuntimeError(f"{kind} job was cancelled.")
        if event_printer is not None:
            event_printer.emit_new(manager)
        results.append(result)
    return results


async def _collect_ready_jobs(
    manager: BlockManager,
    nursery: Nursery,
    *,
    kind: NurseryJobKind,
    event_printer: "_DrainEventPrinter | None",
) -> list[NurseryJobResult[Any]]:
    results = nursery.collect_ready(source="block_manager", kind=kind)
    for result in results:
        await manager._integrate_nursery_result(result)
        if result.error is not None:
            raise RuntimeError(f"{kind} job failed: {result.error}") from result.error
        if result.cancelled:
            raise RuntimeError(f"{kind} job was cancelled.")
        if event_printer is not None:
            event_printer.emit_new(manager)
    return results


def _update_summary_progress(
    progress: Progress | None,
    summary_task: int | None,
    manager: BlockManager,
) -> None:
    if progress is None or summary_task is None:
        return
    progress.update(
        summary_task,
        total=max(1, len(manager.semantic_blocks)),
        completed=_summary_count(manager.semantic_blocks),
    )


def _build_runtime(
    config: SpellbookConfig,
    transcript_path: Path,
    recorder: Recorder,
    context_projector: Callable[[Sequence[IRBlock]], list[IRBlock]],
) -> DrainRuntime:
    tool_registry = ToolRegistry.build(
        config.tool_categories, surface=config.session_type
    )
    backend = build_backend(config)
    surface_builder = RequestSurfaceBuilder.from_config(
        backend=backend,
        config=config,
        tool_registry=tool_registry,
    )
    token_counter = backend.build_token_counter(
        config=config,
        surface_builder=surface_builder,
    )
    nursery = Nursery(config=config)
    fork_runner = ForkRunner(
        parent_config=config,
        parent_transcript_path=transcript_path,
        recorder=recorder,
        session_builder=SessionManager.build,
    )
    footer_controller = FooterController(
        inbound_queue=InboundMessageQueue(),
        recorder=recorder,
    )
    manager = BlockManager(
        config=config.hom_config,
        fork_runner=fork_runner,
        footer_c=footer_controller,
        nursery=nursery,
        recorder=recorder,
        token_meter=TokenMeter(config=config.hom_config, tok_counter=token_counter),
        context_projector=context_projector,
        enable_block_metrics=False,
    )
    return DrainRuntime(block_manager=manager, nursery=nursery)


def _build_recorder(
    *,
    config: SpellbookConfig,
    transcript_path: Path,
    source: RehydrationResult,
) -> Recorder:
    tool_registry = ToolRegistry.build(
        config.tool_categories, surface=config.session_type
    )
    recorder = Recorder(
        config=config,
        transcript_path=transcript_path,
        session_id=source.session_id,
        tool_registry=tool_registry,
    )
    recorder.set_state(
        turn_id=_last_completed_turn_id(source.records, source.last_completed_turn),
        turn=source.last_completed_turn,
        seq=(source.last_seq + 1) if source.last_seq is not None else 0,
    )
    return recorder


def _append_start_block(
    source: RehydrationResult,
    *,
    buffered_policy: BufferedPolicy,
) -> int:
    if buffered_policy == "preserve" and source.buffered_semantic_block_ranges:
        return source.buffered_semantic_block_ranges[-1].end_block + 1
    if source.semantic_blocks:
        return source.semantic_blocks[-1].range.end_block + 1
    return 0


def _target_end_block(
    *,
    source_blocks: int,
    start_block: int,
    max_blocks: int | None,
) -> int:
    if max_blocks is None:
        return source_blocks
    return min(source_blocks, start_block + max_blocks)


def _large_untracked_tool_results(
    source: RehydrationResult,
    *,
    transcript_path: Path,
    start_block: int,
    target_end_block: int,
) -> list[LargeUntrackedToolResult]:
    existing_ttls = {record.call_id for record in source.tool_result_ttls}
    threshold = source.config.hom_config.tool_result_ttl_char_threshold
    results: list[LargeUntrackedToolResult] = []
    for idx, block in enumerate(
        source.blocks[start_block:target_end_block], start=start_block
    ):
        if not isinstance(block, IRToolResultBlock):
            continue
        if (
            block.call_id in existing_ttls
            or block.is_error
            or block.tool in AUTO_TTL_SKIP_TOOLS
        ):
            continue
        content = tool_result_ttl_content(
            block,
            transcript_path=transcript_path,
        )
        if not content.should_auto_register(threshold):
            continue
        results.append(
            LargeUntrackedToolResult(
                block_idx=idx,
                call_id=block.call_id,
                tool=block.tool,
                chars=content.chars or 0,
                lines=content.lines or 0,
                images=content.image_count,
                image_bytes=content.image_bytes,
            )
        )
    return results


def _summary_count(blocks: Sequence[IRSemanticBlock]) -> int:
    return sum(
        1 for block in blocks if any(a.type == "summary" for a in block.artifacts)
    )


def _finish_report_from_rehydrated(
    report: DrainBacklogReport,
    source: RehydrationResult,
) -> None:
    report.final_semantic_blocks = len(source.semantic_blocks)
    report.final_summaries = _summary_count(source.semantic_blocks)
    report.final_buffered_blocks = len(source.buffered_semantic_block_ranges)


def _last_completed_turn_id(
    records: Sequence[IRRecord], last_completed_turn: int
) -> str:
    for record in reversed(records):
        if isinstance(record, IRTurnEndRecord) and record.turn == last_completed_turn:
            return record.turn_id
    if last_completed_turn == 0:
        return ""
    raise ValueError("Could not find last completed turn_id.")


def _backup_path(transcript_path: Path) -> Path:
    candidate = transcript_path.with_suffix(transcript_path.suffix + ".block-drain.bak")
    if not candidate.exists():
        return candidate
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return transcript_path.with_suffix(
        transcript_path.suffix + f".block-drain.bak.{timestamp}"
    )


def _default_report_path(transcript_path: Path) -> Path:
    return transcript_path.with_suffix(".block-drain-report.json")


def _write_report(report: DrainBacklogReport, report_path: Path) -> None:
    report_path = report_path.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )


def _render_completed_blocks(blocks: list[IRSemanticBlock]) -> Table:
    table = Table(title="Completed Blocks", show_lines=True)
    table.add_column("Idx", justify="right")
    table.add_column("Title", style="green")
    table.add_column("Range", justify="right")
    table.add_column("Tokens", justify="right")
    for block in blocks:
        table.add_row(
            str(block.idx),
            block.title,
            _format_range(block.range.start_block, block.range.end_block),
            _format_tokens(block.full_toks.tokens if block.full_toks else None),
        )
    return table


def _render_summary(
    block: IRSemanticBlock,
    summary: IRSemanticBlockSummary,
) -> Panel:
    title = f'Summary: [Block {block.idx}] "{block.title}"'
    return Panel(
        Markdown(_format_summary_markdown(summary)),
        title=title,
        border_style="magenta",
    )


def _format_summary_markdown(summary: IRSemanticBlockSummary) -> str:
    parts = [f"# {summary.headline}", "", summary.text]
    if summary.facets:
        parts.extend(["", "## Facets"])
        for facet in summary.facets:
            parts.append(
                f"- **{facet.title}** "
                f"({_format_range(facet.start_block, facet.end_block)})"
            )
            parts.append(f"  {facet.description}")
            if facet.resources:
                parts.append(f"  Resources: {'; '.join(facet.resources)}")
    if summary.open_thread:
        parts.extend(["", f"Open thread: {summary.open_thread}"])
    return "\n".join(parts)


def _format_range(start: int, end: int) -> str:
    return f"{start}-{end}"


def _format_tokens(tokens: int | None) -> str:
    if tokens is None:
        return "unknown"
    return f"{tokens:,}"


def _print_report(report: DrainBacklogReport) -> None:
    console = Console()
    title = "Block Backlog Drain Dry Run" if report.dry_run else "Block Backlog Drain"
    table = Table(title=title)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Transcript blocks", f"{report.source_blocks:,}")
    table.add_row("Buffered policy", report.buffered_policy)
    table.add_row("Start block", f"{report.start_block:,}")
    table.add_row("Target blocks", f"{report.target_blocks:,}")
    table.add_row("Processed", f"{report.processed_blocks:,}")
    table.add_row("Detector passes", f"{report.detector_passes:,}")
    table.add_row("Finalize passes", f"{report.finalization_passes:,}")
    table.add_row(
        "Semantic blocks",
        _delta(report.starting_semantic_blocks, report.final_semantic_blocks),
    )
    table.add_row(
        "Summaries", _delta(report.starting_summaries, report.final_summaries)
    )
    table.add_row("Buffered ranges", str(report.final_buffered_blocks))
    table.add_row("Ignored buffered ranges", f"{report.ignored_buffered_blocks:,}")
    table.add_row("Large untracked tools", f"{len(report.large_untracked):,}")
    if report.backup_path is not None:
        table.add_row("Backup", str(report.backup_path))
    console.print(table)


def _delta(start: int, final: int | None) -> str:
    if final is None:
        return f"{start:,}"
    change = final - start
    sign = "+" if change >= 0 else ""
    return f"{final:,} ({sign}{change:,})"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="drain_block_backlog",
        description="Append-only drain of an existing raw context tail through block detection and summaries.",
    )
    parser.add_argument(
        "--transcript",
        type=Path,
        default=DEFAULT_TRANSCRIPT,
        help=f"Transcript path. Defaults to {DEFAULT_TRANSCRIPT}.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Existing context blocks to feed per tick. Defaults to --interval.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        help="Detector interval for this drain. Defaults to transcript config.",
    )
    parser.add_argument(
        "--max-blocks",
        type=int,
        default=None,
        help="Process at most this many raw tail blocks.",
    )
    parser.add_argument(
        "--max-finalize-passes",
        type=int,
        default=4,
        help="Maximum EOF detector passes after feeding the tail.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually run forks and append records. Without this, only reports.",
    )
    parser.add_argument(
        "--allow-unfinished",
        action="store_true",
        help="Allow appending after an unfinished turn.",
    )
    parser.add_argument(
        "--allow-large-untracked",
        action="store_true",
        help="Allow detector/summarizer prompts to include large tool results without TTLs.",
    )
    parser.add_argument(
        "--buffered-policy",
        choices=("discard", "preserve"),
        default="discard",
        help=(
            "How to treat existing buffered detector proposals. "
            "'discard' replays from the last completed semantic block; "
            "'preserve' keeps the old behavior and starts after buffered proposals."
        ),
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a transcript backup before applying.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable Rich progress bars.",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Do not print completed block tables and summary panels as they arrive.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional report JSON path. Defaults beside the transcript.",
    )
    return parser.parse_args(argv)


async def _async_main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    load_dotenv(DEFAULT_ENV_PATH)
    await drain_block_backlog(
        transcript_path=args.transcript,
        chunk_size=args.chunk_size,
        interval=args.interval,
        max_blocks=args.max_blocks,
        max_finalize_passes=args.max_finalize_passes,
        apply=args.apply,
        backup=not args.no_backup,
        allow_unfinished=args.allow_unfinished,
        allow_large_untracked=args.allow_large_untracked,
        buffered_policy=args.buffered_policy,
        show_progress=not args.no_progress,
        stream_events=not args.no_stream,
        report_json=args.report_json,
    )


def main(argv: list[str] | None = None) -> None:
    asyncio.run(_async_main(argv))


if __name__ == "__main__":
    main()

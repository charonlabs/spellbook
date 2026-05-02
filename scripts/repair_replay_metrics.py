"""Repair legacy tool-result ordering in a core replay and backfill block metrics.

This is for replay artifacts that already have good semantic blocks/summaries but
were imported from legacy transcripts whose event ordering is provider-invalid
for token counting.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from pydantic import TypeAdapter

from scripts.legacy_tool_result_repair import (
    ToolResultOrderRepairReport,
    repair_tool_result_order_records,
)
from spellbook.backends.anthropic import AnthropicBackend
from spellbook.backends.model_backend import TokenCounter
from spellbook.config import SpellbookConfig
from spellbook.homunculus.token_meter import TokenMeter
from spellbook.ir_types import (
    IRRecord,
    IRSemanticBlock,
    IRSemanticBlockMetricsRecord,
    IRSemanticBlockRecord,
    IRTokenRangeCount,
)
from spellbook.rehydrator import RehydrationResult, Rehydrator
from spellbook.surface_builder import RequestSurfaceBuilder
from spellbook.tools.registry import ToolRegistry

DEFAULT_ENV_PATH = Path.home() / ".chorus" / ".env"


@dataclass(frozen=True)
class MetricBackfillEntry:
    block_id: str
    block_idx: int
    title: str
    status: str
    tokens: int | None = None
    method: str | None = None
    exact: bool | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "block_id": self.block_id,
            "block_idx": self.block_idx,
            "title": self.title,
            "status": self.status,
            "tokens": self.tokens,
            "method": self.method,
            "exact": self.exact,
        }


@dataclass(frozen=True)
class ReplayRepairReport:
    transcript_path: Path
    backup_path: Path | None
    dry_run: bool
    records_read: int
    records_written: int
    order_repair: ToolResultOrderRepairReport
    metrics: list[MetricBackfillEntry] = field(default_factory=list)

    @property
    def metrics_written(self) -> int:
        return sum(1 for metric in self.metrics if metric.status == "written")

    @property
    def metrics_failed(self) -> int:
        return sum(1 for metric in self.metrics if metric.status == "failed")

    @property
    def metrics_skipped_existing(self) -> int:
        return sum(1 for metric in self.metrics if metric.status == "skipped_existing")

    def to_dict(self) -> dict[str, object]:
        return {
            "transcript_path": str(self.transcript_path),
            "backup_path": str(self.backup_path) if self.backup_path else None,
            "dry_run": self.dry_run,
            "records_read": self.records_read,
            "records_written": self.records_written,
            "tool_result_order": self.order_repair.to_dict(),
            "metrics_written": self.metrics_written,
            "metrics_failed": self.metrics_failed,
            "metrics_skipped_existing": self.metrics_skipped_existing,
            "metrics": [metric.to_dict() for metric in self.metrics],
        }


async def repair_replay_transcript(
    *,
    transcript_path: Path,
    backup: bool = True,
    dry_run: bool = False,
    overwrite_metrics: bool = False,
    report_json: Path | None = None,
    token_counter: TokenCounter | None = None,
) -> ReplayRepairReport:
    transcript_path = transcript_path.expanduser().resolve()
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript not found: {transcript_path}")

    original_records = _read_records(transcript_path)
    original_rehydrated = Rehydrator(transcript_path).run()
    repaired_records, order_report = repair_tool_result_order_records(
        original_records,
        semantic_ranges=[
            semantic_block.range
            for semantic_block in original_rehydrated.semantic_blocks
        ],
    )

    repair_rehydrated = _rehydrate_records_via_temp(
        transcript_path,
        repaired_records,
        suffix=".order-repair-tmp",
    )
    metrics = await _build_metric_records(
        rehydrated=repair_rehydrated,
        records=repaired_records,
        overwrite_metrics=overwrite_metrics,
        token_counter=token_counter,
    )
    final_records = [*repaired_records, *metrics]
    final_rehydrated = _rehydrate_records_via_temp(
        transcript_path,
        final_records,
        suffix=".final-repair-tmp",
    )

    metric_entries = _metric_entries(
        semantic_blocks=final_rehydrated.semantic_blocks,
        metric_records=metrics,
        original_blocks=repair_rehydrated.semantic_blocks,
        overwrite_metrics=overwrite_metrics,
    )
    backup_path: Path | None = None
    if not dry_run:
        backup_path = _backup_path(transcript_path) if backup else None
        _atomic_write_records(
            transcript_path=transcript_path,
            records=final_records,
            backup_path=backup_path,
        )

    report = ReplayRepairReport(
        transcript_path=transcript_path,
        backup_path=backup_path,
        dry_run=dry_run,
        records_read=len(original_records),
        records_written=len(final_records),
        order_repair=order_report,
        metrics=metric_entries,
    )
    _write_report(report, report_json or _default_report_path(transcript_path))
    return report


async def _build_metric_records(
    *,
    rehydrated: RehydrationResult,
    records: list[IRRecord],
    overwrite_metrics: bool,
    token_counter: TokenCounter | None,
) -> list[IRSemanticBlockMetricsRecord]:
    meter = TokenMeter(
        config=rehydrated.config.hom_config,
        tok_counter=token_counter or _build_token_counter(rehydrated.config),
    )
    semantic_records = _semantic_block_records_by_id(records)
    metrics: list[IRSemanticBlockMetricsRecord] = []
    for block in rehydrated.semantic_blocks:
        if block.full_toks is not None and not overwrite_metrics:
            continue
        count = await meter.count_slice(
            rehydrated.blocks,
            block.range.start_block,
            block.range.end_block + 1,
        )
        if count is None:
            continue
        semantic_record = semantic_records.get(block.id)
        metrics.append(
            IRSemanticBlockMetricsRecord(
                session_id=rehydrated.session_id,
                block_id=block.id,
                toks=count,
                turn=semantic_record.turn if semantic_record else 0,
                turn_id=semantic_record.turn_id if semantic_record else "",
            )
        )
    return metrics


def _metric_entries(
    *,
    semantic_blocks: list[IRSemanticBlock],
    metric_records: list[IRSemanticBlockMetricsRecord],
    original_blocks: list[IRSemanticBlock],
    overwrite_metrics: bool,
) -> list[MetricBackfillEntry]:
    metric_by_block_id = {record.block_id: record.toks for record in metric_records}
    original_by_block_id = {block.id: block for block in original_blocks}
    entries: list[MetricBackfillEntry] = []
    for block in semantic_blocks:
        metric = metric_by_block_id.get(block.id)
        if metric is not None:
            entries.append(_metric_entry(block, "written", metric))
            continue
        original = original_by_block_id.get(block.id)
        if (
            original is not None
            and original.full_toks is not None
            and not overwrite_metrics
        ):
            entries.append(_metric_entry(block, "skipped_existing", original.full_toks))
        else:
            entries.append(_metric_entry(block, "failed", None))
    return entries


def _metric_entry(
    block: IRSemanticBlock,
    status: str,
    toks: IRTokenRangeCount | None,
) -> MetricBackfillEntry:
    return MetricBackfillEntry(
        block_id=block.id,
        block_idx=block.idx,
        title=block.title,
        status=status,
        tokens=toks.tokens if toks else None,
        method=toks.method if toks else None,
        exact=toks.exact if toks else None,
    )


def _build_token_counter(config: SpellbookConfig) -> TokenCounter:
    match config.provider:
        case "anthropic":
            backend = AnthropicBackend()
        case _:
            raise NotImplementedError(
                f"Metrics repair does not support provider `{config.provider}` yet."
            )
    tool_registry = ToolRegistry.build(
        config.tool_categories,
        surface=config.session_type,
    )
    surface_builder = RequestSurfaceBuilder.from_config(
        backend=backend,
        config=config,
        tool_registry=tool_registry,
    )
    return backend.build_token_counter(config=config, surface_builder=surface_builder)


def _semantic_block_records_by_id(
    records: list[IRRecord],
) -> dict[str, IRSemanticBlockRecord]:
    return {
        record.id: record
        for record in records
        if isinstance(record, IRSemanticBlockRecord)
    }


def _read_records(path: Path) -> list[IRRecord]:
    adapter = TypeAdapter(IRRecord)
    records: list[IRRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(adapter.validate_json(line))
    return records


def _write_records(path: Path, records: list[IRRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(record.model_dump_json() + "\n")


def _rehydrate_records_via_temp(
    transcript_path: Path,
    records: list[IRRecord],
    *,
    suffix: str,
) -> RehydrationResult:
    tmp_path = transcript_path.with_name(transcript_path.name + suffix)
    try:
        _write_records(tmp_path, records)
        return Rehydrator(tmp_path).run()
    finally:
        tmp_path.unlink(missing_ok=True)


def _atomic_write_records(
    *,
    transcript_path: Path,
    records: list[IRRecord],
    backup_path: Path | None,
) -> None:
    tmp_path = transcript_path.with_name(transcript_path.name + ".tmp")
    try:
        _write_records(tmp_path, records)
        Rehydrator(tmp_path).run()
        if backup_path is not None:
            shutil.copy2(transcript_path, backup_path)
        tmp_path.replace(transcript_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _backup_path(transcript_path: Path) -> Path:
    candidate = transcript_path.with_suffix(transcript_path.suffix + ".bak")
    if not candidate.exists():
        return candidate
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return transcript_path.with_suffix(transcript_path.suffix + f".bak.{timestamp}")


def _default_report_path(transcript_path: Path) -> Path:
    return transcript_path.with_suffix(".repair-report.json")


def _write_report(report: ReplayRepairReport, report_path: Path) -> None:
    report_path = report_path.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="scripts.repair_replay_metrics",
        description=(
            "Repair legacy tool-result ordering in a core replay transcript and "
            "append missing semantic_block_metrics records."
        ),
    )
    parser.add_argument("transcript", type=Path, help="Core replay transcript path.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute repairs and metrics without rewriting the transcript.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Rewrite without creating transcript.jsonl.bak.",
    )
    parser.add_argument(
        "--overwrite-metrics",
        action="store_true",
        help="Append new metrics even when a block already has full_toks.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        help="Path for the repair report. Defaults beside the transcript.",
    )
    parser.add_argument(
        "--env",
        type=Path,
        default=DEFAULT_ENV_PATH,
        help=f"Dotenv file to load before counting. Defaults to {DEFAULT_ENV_PATH}.",
    )
    return parser.parse_args(argv)


async def _async_main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    env_path = args.env.expanduser()
    if env_path.exists():
        load_dotenv(env_path)
    report = await repair_replay_transcript(
        transcript_path=args.transcript,
        backup=not args.no_backup,
        dry_run=args.dry_run,
        overwrite_metrics=args.overwrite_metrics,
        report_json=args.report_json,
    )
    print(
        "Repair complete: "
        f"{report.order_repair.records_moved} moved; "
        f"{report.metrics_written} metrics written; "
        f"{report.metrics_failed} failed; "
        f"{report.metrics_skipped_existing} skipped existing."
    )
    if report.backup_path is not None:
        print(f"Backup: {report.backup_path}")
    print(f"Report: {args.report_json or _default_report_path(args.transcript)}")


def main(argv: list[str] | None = None) -> None:
    asyncio.run(_async_main(argv))


if __name__ == "__main__":
    main()

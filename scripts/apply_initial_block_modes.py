"""Append semantic block mode records for an initial transplant context plan."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from spellbook.ir_types import (
    IRRecord,
    IRSemanticBlock,
    IRSemanticBlockApplyModeRecord,
    IRTurnEndRecord,
    SemanticBlockApplyModeSource,
    SemanticBlockMode,
)
from spellbook.rehydrator import Rehydrator

ApplyStatus = Literal["appended", "would_append", "skipped_unchanged"]


@dataclass(frozen=True)
class BlockModeApplyEntry:
    block_id: str
    block_idx: int
    title: str
    previous_mode: SemanticBlockMode
    target_mode: SemanticBlockMode
    status: ApplyStatus

    def to_dict(self) -> dict[str, object]:
        return {
            "block_id": self.block_id,
            "block_idx": self.block_idx,
            "title": self.title,
            "previous_mode": self.previous_mode,
            "target_mode": self.target_mode,
            "status": self.status,
        }


@dataclass(frozen=True)
class InitialBlockModeApplyReport:
    transcript_path: Path
    backup_path: Path | None
    dry_run: bool
    keep_newest_full: int
    semantic_blocks: int
    target_full_blocks: int
    target_summary_blocks: int
    source: SemanticBlockApplyModeSource
    entries: list[BlockModeApplyEntry] = field(default_factory=list)

    @property
    def records_appended(self) -> int:
        return sum(1 for entry in self.entries if entry.status == "appended")

    @property
    def records_to_append(self) -> int:
        return sum(
            1
            for entry in self.entries
            if entry.status == "appended" or entry.status == "would_append"
        )

    @property
    def skipped_unchanged(self) -> int:
        return sum(1 for entry in self.entries if entry.status == "skipped_unchanged")

    def to_dict(self) -> dict[str, object]:
        return {
            "transcript_path": str(self.transcript_path),
            "backup_path": str(self.backup_path) if self.backup_path else None,
            "dry_run": self.dry_run,
            "keep_newest_full": self.keep_newest_full,
            "semantic_blocks": self.semantic_blocks,
            "target_full_blocks": self.target_full_blocks,
            "target_summary_blocks": self.target_summary_blocks,
            "source": self.source,
            "records_appended": self.records_appended,
            "records_to_append": self.records_to_append,
            "skipped_unchanged": self.skipped_unchanged,
            "entries": [entry.to_dict() for entry in self.entries],
        }


def apply_initial_block_modes(
    transcript_path: Path,
    keep_newest_full: int,
    *,
    apply: bool = False,
    backup: bool = True,
    validate: bool = True,
    force: bool = False,
    allow_unfinished: bool = False,
    source: SemanticBlockApplyModeSource = "planner",
    report_json: Path | None = None,
    write_report: bool = True,
) -> InitialBlockModeApplyReport:
    """Append mode records so all but the newest N semantic blocks render summary."""

    if keep_newest_full < 0:
        raise ValueError("keep_newest_full must be greater than or equal to zero.")

    transcript_path = transcript_path.expanduser().resolve()
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript not found: {transcript_path}")

    rehydrated = Rehydrator(transcript_path).run()
    if rehydrated.is_unfinished_turn and not allow_unfinished:
        raise ValueError(
            "Transcript has an unfinished turn. Pass --allow-unfinished if you "
            "really want to append mode records after it."
        )

    target_modes = _target_modes(rehydrated.semantic_blocks, keep_newest_full)
    _validate_target_modes(rehydrated.semantic_blocks, target_modes)

    turn_id = _last_completed_turn_id(
        rehydrated.records,
        rehydrated.last_completed_turn,
    )
    mode_records: list[IRSemanticBlockApplyModeRecord] = []
    entries: list[BlockModeApplyEntry] = []
    dry_run = not apply
    for block in rehydrated.semantic_blocks:
        target_mode = target_modes[block.id]
        should_append = force or block.mode != target_mode
        status: ApplyStatus
        if should_append:
            status = "would_append" if dry_run else "appended"
            mode_records.append(
                IRSemanticBlockApplyModeRecord(
                    session_id=rehydrated.session_id,
                    block_id=block.id,
                    mode=target_mode,
                    source=source,
                    turn=rehydrated.last_completed_turn,
                    turn_id=turn_id,
                )
            )
        else:
            status = "skipped_unchanged"
        entries.append(
            BlockModeApplyEntry(
                block_id=block.id,
                block_idx=block.idx,
                title=block.title,
                previous_mode=block.mode,
                target_mode=target_mode,
                status=status,
            )
        )

    backup_path: Path | None = None
    if mode_records and apply:
        backup_path = _backup_path(transcript_path) if backup else None
        _append_records(
            transcript_path=transcript_path,
            records=mode_records,
            backup_path=backup_path,
            validate=validate,
        )

    target_full_blocks = sum(1 for mode in target_modes.values() if mode == "full")
    report = InitialBlockModeApplyReport(
        transcript_path=transcript_path,
        backup_path=backup_path,
        dry_run=dry_run,
        keep_newest_full=keep_newest_full,
        semantic_blocks=len(rehydrated.semantic_blocks),
        target_full_blocks=target_full_blocks,
        target_summary_blocks=len(target_modes) - target_full_blocks,
        source=source,
        entries=entries,
    )
    if write_report:
        _write_report(report, report_json or _default_report_path(transcript_path))
    return report


def _target_modes(
    semantic_blocks: list[IRSemanticBlock],
    keep_newest_full: int,
) -> dict[str, SemanticBlockMode]:
    full_start = max(0, len(semantic_blocks) - keep_newest_full)
    return {
        block.id: "full" if idx >= full_start else "summary"
        for idx, block in enumerate(semantic_blocks)
    }


def _validate_target_modes(
    semantic_blocks: list[IRSemanticBlock],
    target_modes: dict[str, SemanticBlockMode],
) -> None:
    missing_summary = [
        f'{block.idx} "{block.title}"'
        for block in semantic_blocks
        if target_modes[block.id] == "summary"
        and not any(artifact.mode == "summary" for artifact in block.artifacts)
    ]
    if missing_summary:
        preview = "\n".join(f"- {line}" for line in missing_summary[:20])
        suffix = "\n..." if len(missing_summary) > 20 else ""
        raise ValueError(
            "Cannot apply summary mode because some target blocks do not have "
            f"summary artifacts:\n{preview}{suffix}"
        )


def _last_completed_turn_id(records: list[IRRecord], last_completed_turn: int) -> str:
    for record in reversed(records):
        if isinstance(record, IRTurnEndRecord) and record.turn == last_completed_turn:
            return record.turn_id
    raise ValueError(
        "Could not find the last completed turn_id needed for appended mode records."
    )


def _append_records(
    *,
    transcript_path: Path,
    records: list[IRSemanticBlockApplyModeRecord],
    backup_path: Path | None,
    validate: bool,
) -> None:
    original_text = transcript_path.read_text(encoding="utf-8")
    append_text = "".join(record.model_dump_json() + "\n" for record in records)
    tmp_path = transcript_path.with_name(transcript_path.name + ".tmp")
    try:
        tmp_path.write_text(
            _ensure_trailing_newline(original_text) + append_text,
            encoding="utf-8",
        )
        if validate:
            Rehydrator(tmp_path).run()
        if backup_path is not None:
            shutil.copy2(transcript_path, backup_path)
        tmp_path.replace(transcript_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _ensure_trailing_newline(text: str) -> str:
    if not text:
        return ""
    return text if text.endswith("\n") else text + "\n"


def _backup_path(transcript_path: Path) -> Path:
    candidate = transcript_path.with_suffix(transcript_path.suffix + ".block-modes.bak")
    if not candidate.exists():
        return candidate
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return transcript_path.with_suffix(
        transcript_path.suffix + f".block-modes.bak.{timestamp}"
    )


def _default_report_path(transcript_path: Path) -> Path:
    return transcript_path.with_suffix(".block-modes-report.json")


def _write_report(
    report: InitialBlockModeApplyReport,
    report_path: Path,
) -> None:
    report_path = report_path.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="scripts.apply_initial_block_modes",
        description=(
            "Append semantic_block_apply_mode records so all but the newest N "
            "semantic blocks render in summary mode."
        ),
    )
    parser.add_argument("transcript", type=Path, help="Prepared core transcript JSONL.")
    parser.add_argument(
        "keep_newest_full",
        type=int,
        help="Number of newest semantic blocks to keep at full fidelity.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Rewrite the transcript. Without this, only report the planned records.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Append a mode record for every block, even when it already has the target mode.",
    )
    parser.add_argument(
        "--source",
        choices=["planner", "model"],
        default="planner",
        help="Source stored on appended mode records. Defaults to planner.",
    )
    parser.add_argument(
        "--allow-unfinished",
        action="store_true",
        help="Allow appending after a transcript with an unfinished turn.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Append without creating transcript.jsonl.block-modes.bak.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validating the rewritten transcript with the core Rehydrator.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional path for the report JSON. Defaults beside the transcript.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    report_path = args.report_json or _default_report_path(
        args.transcript.expanduser().resolve()
    )
    report = apply_initial_block_modes(
        args.transcript,
        args.keep_newest_full,
        apply=args.apply,
        backup=not args.no_backup,
        validate=not args.no_validate,
        force=args.force,
        allow_unfinished=args.allow_unfinished,
        source=args.source,
        report_json=args.report_json,
    )
    _print_report(report, report_path)


def _print_report(
    report: InitialBlockModeApplyReport,
    report_path: Path,
) -> None:
    mode = "Dry run" if report.dry_run else "Applied block modes"
    print(f"{mode}: {report.transcript_path}")
    print(f"Semantic blocks: {report.semantic_blocks}")
    print(
        f"Target: {report.target_summary_blocks} summary, {report.target_full_blocks} full"
    )
    print(f"Keep newest full: {report.keep_newest_full}")
    print(f"Source: {report.source}")
    print(
        "Records: "
        f"{report.records_appended} appended; "
        f"{report.records_to_append} to append; "
        f"{report.skipped_unchanged} unchanged"
    )
    if report.backup_path is not None:
        print(f"Backup: {report.backup_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()

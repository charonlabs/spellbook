"""Append newly completed legacy turns onto an existing core transcript.

This is for transplant artifacts that were imported from a legacy transcript,
then later need to catch up with conversation that happened after the import.
It preserves the existing core transcript and appends only converted content
turn records from the legacy source. Legacy Spellbook-internal markers are
skipped in the same way as the full importer.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.legacy_tool_result_repair import repair_tool_result_order_records
from scripts.parse_legacy_context import (
    _convert_event_record,
    _convert_turn_end_record,
    _convert_turn_start_record,
    _legacy_event_type,
)
from spellbook.ir_types import IRRecord
from spellbook.rehydrator import Rehydrator

DEFAULT_LEGACY_TRANSCRIPT = (
    Path.home()
    / ".chorus"
    / "spellbook"
    / "sessions"
    / "meta-claude"
    / "transcript.jsonl"
)
DEFAULT_TARGET_TRANSCRIPT = (
    Path(__file__).resolve().parents[1]
    / "archive"
    / "core_imports"
    / "meta-claude-core.jsonl"
)


@dataclass
class LegacyTailAppendReport:
    legacy_path: Path
    target_path: Path
    backup_path: Path | None
    dry_run: bool
    after_turn: int
    latest_completed_legacy_turn: int | None
    records_read: int = 0
    records_appended: int = 0
    blocks_appended: int = 0
    turns_appended: int = 0
    converted_event_types: Counter[str] = field(default_factory=Counter)
    skipped_ir: Counter[str] = field(default_factory=Counter)
    skipped_event_types: Counter[str] = field(default_factory=Counter)
    skipped_open_turns: list[int] = field(default_factory=list)
    tool_result_order_repairs: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "legacy_path": str(self.legacy_path),
            "target_path": str(self.target_path),
            "backup_path": str(self.backup_path) if self.backup_path else None,
            "dry_run": self.dry_run,
            "after_turn": self.after_turn,
            "latest_completed_legacy_turn": self.latest_completed_legacy_turn,
            "records_read": self.records_read,
            "records_appended": self.records_appended,
            "blocks_appended": self.blocks_appended,
            "turns_appended": self.turns_appended,
            "converted_event_types": dict(self.converted_event_types),
            "skipped_ir": dict(self.skipped_ir),
            "skipped_event_types": dict(self.skipped_event_types),
            "skipped_open_turns": self.skipped_open_turns,
            "tool_result_order_repairs": self.tool_result_order_repairs,
        }


def append_legacy_tail(
    *,
    legacy_path: Path,
    target_path: Path,
    after_turn: int | None = None,
    backup: bool = True,
    dry_run: bool = False,
    validate: bool = True,
) -> LegacyTailAppendReport:
    legacy_path = legacy_path.expanduser().resolve()
    target_path = target_path.expanduser().resolve()
    if not legacy_path.exists():
        raise FileNotFoundError(f"Legacy transcript not found: {legacy_path}")
    if not target_path.exists():
        raise FileNotFoundError(f"Target core transcript not found: {target_path}")

    target = Rehydrator(target_path).run()
    if target.is_unfinished_turn and after_turn is None:
        raise ValueError(
            "Target transcript has an unfinished turn; pass --after-turn explicitly "
            "if you really want to append after it."
        )
    append_after = target.last_completed_turn if after_turn is None else after_turn

    legacy_records = _read_legacy_records(legacy_path)
    completed_turns = _completed_turns_after(legacy_records, append_after)
    skipped_open_turns = sorted(
        {
            turn
            for turn in _turns_after(legacy_records, append_after)
            if turn not in completed_turns
        }
    )
    append_records, report = _convert_tail_records(
        legacy_records=legacy_records,
        session_id=target.session_id,
        after_turn=append_after,
        completed_turns=completed_turns,
        skipped_open_turns=skipped_open_turns,
        legacy_path=legacy_path,
        target_path=target_path,
        dry_run=dry_run,
    )
    repaired_records, repair_report = repair_tool_result_order_records(append_records)
    report.records_appended = len(repaired_records)
    report.tool_result_order_repairs = repair_report.records_moved

    backup_path: Path | None = None
    if repaired_records and not dry_run:
        backup_path = _backup_path(target_path) if backup else None
        _append_records(
            target_path=target_path,
            records=repaired_records,
            backup_path=backup_path,
            validate=validate,
        )
        report.backup_path = backup_path

    return report


def _convert_tail_records(
    *,
    legacy_records: list[dict[str, Any]],
    session_id: str,
    after_turn: int,
    completed_turns: set[int],
    skipped_open_turns: list[int],
    legacy_path: Path,
    target_path: Path,
    dry_run: bool,
) -> tuple[list[IRRecord], LegacyTailAppendReport]:
    report = LegacyTailAppendReport(
        legacy_path=legacy_path,
        target_path=target_path,
        backup_path=None,
        dry_run=dry_run,
        after_turn=after_turn,
        latest_completed_legacy_turn=max(completed_turns) if completed_turns else None,
        skipped_open_turns=skipped_open_turns,
    )
    records: list[IRRecord] = []
    current_turn_id: str | None = None
    fallback_seq_by_turn: dict[int, int] = {}

    for legacy_record in legacy_records:
        report.records_read += 1
        turn = legacy_record.get("turn")
        if not isinstance(turn, int) or turn <= after_turn:
            continue
        if turn not in completed_turns:
            report.skipped_ir[str(legacy_record.get("ir"))] += 1
            continue

        match legacy_record.get("ir"):
            case "turn_start":
                record = _convert_turn_start_record(legacy_record, session_id)
                current_turn_id = record.turn_id
                records.append(record)
                report.turns_appended += 1
            case "turn_end":
                record = _convert_turn_end_record(legacy_record, session_id)
                current_turn_id = None
                records.append(record)
            case "event":
                converted = _convert_event_record(
                    legacy_record,
                    session_id=session_id,
                    current_turn_id=current_turn_id,
                    fallback_seq_by_turn=fallback_seq_by_turn,
                )
                if converted is None:
                    report.skipped_event_types[_legacy_event_type(legacy_record)] += 1
                    continue
                records.append(converted)
                report.blocks_appended += 1
                report.converted_event_types[_legacy_event_type(legacy_record)] += 1
            case _:
                report.skipped_ir[str(legacy_record.get("ir"))] += 1

    return records, report


def _read_legacy_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as src:
        for raw_line in src:
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            if isinstance(record, dict):
                records.append(record)
    return records


def _completed_turns_after(
    legacy_records: list[dict[str, Any]],
    after_turn: int,
) -> set[int]:
    turns: set[int] = set()
    for record in legacy_records:
        if record.get("ir") != "turn_end":
            continue
        turn = record.get("turn")
        if isinstance(turn, int) and turn > after_turn:
            turns.add(turn)
    return turns


def _turns_after(
    legacy_records: list[dict[str, Any]],
    after_turn: int,
) -> set[int]:
    turns: set[int] = set()
    for record in legacy_records:
        turn = record.get("turn")
        if isinstance(turn, int) and turn > after_turn:
            turns.add(turn)
    return turns


def _append_records(
    *,
    target_path: Path,
    records: list[IRRecord],
    backup_path: Path | None,
    validate: bool,
) -> None:
    original_text = target_path.read_text(encoding="utf-8")
    append_text = "".join(record.model_dump_json() + "\n" for record in records)
    tmp_path = target_path.with_name(target_path.name + ".tmp")
    try:
        tmp_path.write_text(
            _ensure_trailing_newline(original_text) + append_text, encoding="utf-8"
        )
        if validate:
            Rehydrator(tmp_path).run()
        if backup_path is not None:
            shutil.copy2(target_path, backup_path)
        tmp_path.replace(target_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _ensure_trailing_newline(text: str) -> str:
    if not text:
        return ""
    return text if text.endswith("\n") else text + "\n"


def _backup_path(target_path: Path) -> Path:
    candidate = target_path.with_suffix(target_path.suffix + ".tail.bak")
    if not candidate.exists():
        return candidate
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return target_path.with_suffix(target_path.suffix + f".tail.bak.{timestamp}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="scripts.append_legacy_tail",
        description=(
            "Append newly completed content turns from a legacy transcript onto "
            "an existing core transcript."
        ),
    )
    parser.add_argument(
        "--legacy",
        type=Path,
        default=DEFAULT_LEGACY_TRANSCRIPT,
        help=f"Legacy transcript path. Defaults to {DEFAULT_LEGACY_TRANSCRIPT}.",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET_TRANSCRIPT,
        help=f"Core transcript to append to. Defaults to {DEFAULT_TARGET_TRANSCRIPT}.",
    )
    parser.add_argument(
        "--after-turn",
        type=int,
        default=None,
        help=(
            "Only append legacy turns greater than this turn. Defaults to the "
            "target transcript's last completed turn."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be appended without rewriting the target transcript.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Append without creating transcript.jsonl.tail.bak.",
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
        help="Optional path for the append report JSON.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    report = append_legacy_tail(
        legacy_path=args.legacy,
        target_path=args.target,
        after_turn=args.after_turn,
        backup=not args.no_backup,
        dry_run=args.dry_run,
        validate=not args.no_validate,
    )
    if args.report_json is not None:
        report_path = args.report_json.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    _print_report(report)


def _print_report(report: LegacyTailAppendReport) -> None:
    mode = "Dry run" if report.dry_run else "Appended legacy tail"
    print(f"{mode}: {report.legacy_path}")
    print(f"Target: {report.target_path}")
    print(f"After turn: {report.after_turn}")
    print(f"Latest completed legacy turn: {report.latest_completed_legacy_turn}")
    print(
        "Appended: "
        f"{report.records_appended} records; "
        f"{report.blocks_appended} content blocks; "
        f"{report.turns_appended} turns"
    )
    if report.backup_path is not None:
        print(f"Backup: {report.backup_path}")
    if report.converted_event_types:
        print(f"Converted events: {dict(report.converted_event_types)}")
    if report.skipped_event_types:
        print(f"Skipped events: {dict(report.skipped_event_types)}")
    if report.skipped_open_turns:
        print(f"Skipped open turns: {report.skipped_open_turns}")
    if report.tool_result_order_repairs:
        print(f"Tool result order repairs: {report.tool_result_order_repairs}")


if __name__ == "__main__":
    main()

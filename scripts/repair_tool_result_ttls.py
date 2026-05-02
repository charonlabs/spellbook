"""Append retroactive TTL records for oversized historical tool results."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Sequence

from spellbook.homunculus.tool_result_ttl import (
    AUTO_TTL_SKIP_TOOLS,
    build_tool_result_ttl_replacement,
    tool_result_text_content,
)
from spellbook.ir_types import (
    IRBlockRecord,
    IRRecord,
    IRToolResultBlock,
    IRToolResultTTLRecord,
    IRTurnStartRecord,
)
from spellbook.rehydrator import Rehydrator


@dataclass(frozen=True)
class ToolResultTTLRepairCandidate:
    call_id: str
    tool: str
    turn: int
    turn_id: str
    seq: int
    chars: int
    lines: int
    output_ref: str
    replace_content: str

    def to_dict(self) -> dict[str, object]:
        return {
            "call_id": self.call_id,
            "tool": self.tool,
            "turn": self.turn,
            "turn_id": self.turn_id,
            "seq": self.seq,
            "chars": self.chars,
            "lines": self.lines,
            "output_ref": self.output_ref,
            "replace_content": self.replace_content,
        }


@dataclass(frozen=True)
class ToolResultTTLRepairReport:
    transcript_path: Path
    backup_path: Path | None
    dry_run: bool
    min_chars: int
    ttl_turns: int
    records_scanned: int
    candidates: list[ToolResultTTLRepairCandidate] = field(default_factory=list)
    skipped: Counter[str] = field(default_factory=Counter)

    @property
    def records_appended(self) -> int:
        return 0 if self.dry_run else len(self.candidates)

    def to_dict(self) -> dict[str, object]:
        return {
            "transcript_path": str(self.transcript_path),
            "backup_path": str(self.backup_path) if self.backup_path else None,
            "dry_run": self.dry_run,
            "min_chars": self.min_chars,
            "ttl_turns": self.ttl_turns,
            "records_scanned": self.records_scanned,
            "records_appended": self.records_appended,
            "candidate_count": len(self.candidates),
            "skipped": dict(self.skipped),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def repair_tool_result_ttls(
    transcript_path: Path,
    *,
    apply: bool = False,
    backup: bool = True,
    min_chars: int | None = None,
    ttl_turns: int | None = None,
    validate: bool = True,
    report_json: Path | None = None,
    write_report: bool = True,
) -> ToolResultTTLRepairReport:
    """Scan a core transcript and append missing TTL records for large tool results."""

    transcript_path = transcript_path.expanduser().resolve()
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript not found: {transcript_path}")

    rehydrated = Rehydrator(transcript_path).run()
    threshold = (
        rehydrated.config.hom_config.tool_result_ttl_char_threshold
        if min_chars is None
        else min_chars
    )
    ttl = (
        rehydrated.config.hom_config.tool_result_ttl_turns
        if ttl_turns is None
        else ttl_turns
    )
    candidates, skipped = _find_candidates(
        rehydrated.records,
        transcript_path=transcript_path,
        min_chars=threshold,
    )
    dry_run = not apply
    backup_path: Path | None = None
    if candidates and apply:
        backup_path = _backup_path(transcript_path) if backup else None
        ttl_records = [
            _ttl_record(
                session_id=rehydrated.session_id,
                candidate=candidate,
                ttl=ttl,
            )
            for candidate in candidates
        ]
        _append_records(
            transcript_path=transcript_path,
            records=ttl_records,
            candidates=candidates,
            backup_path=backup_path,
            validate=validate,
        )

    report = ToolResultTTLRepairReport(
        transcript_path=transcript_path,
        backup_path=backup_path,
        dry_run=dry_run,
        min_chars=threshold,
        ttl_turns=ttl,
        records_scanned=len(rehydrated.records),
        candidates=candidates,
        skipped=skipped,
    )
    if write_report:
        _write_report(report, report_json or _default_report_path(transcript_path))
    return report


def _find_candidates(
    records: Sequence[IRRecord],
    *,
    transcript_path: Path,
    min_chars: int,
) -> tuple[list[ToolResultTTLRepairCandidate], Counter[str]]:
    existing_ttls = {
        record.call_id
        for record in records
        if isinstance(record, IRToolResultTTLRecord)
    }
    turn_ids = _turn_ids_by_turn(records)
    skipped: Counter[str] = Counter()
    candidates: list[ToolResultTTLRepairCandidate] = []

    for record in records:
        if not isinstance(record, IRBlockRecord):
            continue
        block = record.event
        if not isinstance(block, IRToolResultBlock):
            continue
        if block.call_id in existing_ttls:
            skipped["already_has_ttl"] += 1
            continue
        if block.is_error:
            skipped["error_result"] += 1
            continue
        if block.tool in AUTO_TTL_SKIP_TOOLS:
            skipped["memory_or_skill_tool"] += 1
            continue

        output = tool_result_text_content(block)
        if output is None:
            skipped["non_text_result"] += 1
            continue
        if len(output) < min_chars:
            skipped["below_threshold"] += 1
            continue

        turn_id = block.turn_id or turn_ids.get(record.turn)
        if turn_id is None:
            skipped["missing_turn_id"] += 1
            continue

        output_ref = f"tool-outputs/{_safe_filename(block.call_id)}.txt"
        candidates.append(
            ToolResultTTLRepairCandidate(
                call_id=block.call_id,
                tool=block.tool,
                turn=record.turn,
                turn_id=turn_id,
                seq=record.seq,
                chars=len(output),
                lines=_line_count(output),
                output_ref=output_ref,
                replace_content=build_tool_result_ttl_replacement(
                    tool=block.tool,
                    output=output,
                    display=block.display,
                    output_ref=output_ref,
                ),
            )
        )

    return candidates, skipped


def _turn_ids_by_turn(records: Sequence[IRRecord]) -> dict[int, str]:
    return {
        record.turn: record.turn_id
        for record in records
        if isinstance(record, IRTurnStartRecord)
    }


def _ttl_record(
    *,
    session_id: str,
    candidate: ToolResultTTLRepairCandidate,
    ttl: int,
) -> IRToolResultTTLRecord:
    return IRToolResultTTLRecord(
        session_id=session_id,
        call_id=candidate.call_id,
        replace_content=candidate.replace_content,
        ttl=ttl,
        trigger="end_turn",
        delivered_turn=candidate.turn,
        source="repair",
        output_ref=candidate.output_ref,
        turn=candidate.turn,
        turn_id=candidate.turn_id,
    )


def _append_records(
    *,
    transcript_path: Path,
    records: list[IRToolResultTTLRecord],
    candidates: list[ToolResultTTLRepairCandidate],
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
        _write_tool_outputs(transcript_path, candidates)
        tmp_path.replace(transcript_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _write_tool_outputs(
    transcript_path: Path,
    candidates: list[ToolResultTTLRepairCandidate],
) -> None:
    records_by_call_id = _tool_results_by_call_id(transcript_path)
    output_dir = transcript_path.parent / "tool-outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    for candidate in candidates:
        block = records_by_call_id[candidate.call_id]
        output = tool_result_text_content(block)
        if output is None:
            continue
        (transcript_path.parent / candidate.output_ref).write_text(
            output,
            encoding="utf-8",
        )


def _tool_results_by_call_id(transcript_path: Path) -> dict[str, IRToolResultBlock]:
    rehydrated = Rehydrator(transcript_path).run()
    blocks: dict[str, IRToolResultBlock] = {}
    for record in rehydrated.records:
        if isinstance(record, IRBlockRecord) and isinstance(
            record.event, IRToolResultBlock
        ):
            blocks[record.event.call_id] = record.event
    return blocks


def _ensure_trailing_newline(text: str) -> str:
    if not text:
        return ""
    return text if text.endswith("\n") else text + "\n"


def _backup_path(transcript_path: Path) -> Path:
    candidate = transcript_path.with_suffix(transcript_path.suffix + ".tool-ttls.bak")
    if not candidate.exists():
        return candidate
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return transcript_path.with_suffix(
        transcript_path.suffix + f".tool-ttls.bak.{timestamp}"
    )


def _default_report_path(transcript_path: Path) -> Path:
    return transcript_path.with_suffix(".tool-result-ttl-repair-report.json")


def _write_report(report: ToolResultTTLRepairReport, report_path: Path) -> None:
    report_path = report_path.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _line_count(text: str) -> int:
    if text == "":
        return 0
    return len(text.splitlines()) or 1


def _safe_filename(value: str) -> str:
    return "".join(c if c.isalnum() or c in "_.-" else "_" for c in value)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="scripts.repair_tool_result_ttls",
        description=(
            "Append explicit source=repair TTL records for oversized historical "
            "tool results in a core transcript."
        ),
    )
    parser.add_argument("transcript", type=Path, help="Core transcript path.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Rewrite the transcript. Without this flag, only reports candidates.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=None,
        help="Override the transcript HomunculusConfig TTL char threshold.",
    )
    parser.add_argument(
        "--ttl-turns",
        type=int,
        default=None,
        help="Override the transcript HomunculusConfig TTL turn count.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Apply without creating transcript.jsonl.tool-ttls.bak.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validating the repaired transcript with Rehydrator.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Report JSON path. Defaults beside the transcript.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    report = repair_tool_result_ttls(
        args.transcript,
        apply=args.apply,
        backup=not args.no_backup,
        min_chars=args.min_chars,
        ttl_turns=args.ttl_turns,
        validate=not args.no_validate,
        report_json=args.report_json,
    )
    mode = "Applied" if args.apply else "Dry run"
    print(
        f"{mode}: {len(report.candidates)} TTL candidate(s), "
        f"{report.records_appended} record(s) appended."
    )
    if report.backup_path is not None:
        print(f"Backup: {report.backup_path}")
    print(f"Report: {args.report_json or _default_report_path(args.transcript)}")
    if report.skipped:
        print("Skipped: " + ", ".join(f"{k}={v}" for k, v in report.skipped.items()))


if __name__ == "__main__":
    main()

"""Append facet pin records to a core transcript by facet id prefix."""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from spellbook.ir_types import (
    IRRecord,
    IRSemanticBlock,
    IRSemanticBlockPin,
    IRSemanticBlockPinRecord,
    IRTurnEndRecord,
    IRTurnStartRecord,
)
from spellbook.rehydrator import RehydrationResult, Rehydrator

DEFAULT_REASON = "Pinned during bulk facet pin import."


@dataclass(frozen=True)
class FacetPinResolution:
    requested: str
    facet_id: str
    facet_title: str
    block_id: str
    block_idx: int
    block_title: str
    already_pinned: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "requested": self.requested,
            "facet_id": self.facet_id,
            "facet_title": self.facet_title,
            "block_id": self.block_id,
            "block_idx": self.block_idx,
            "block_title": self.block_title,
            "already_pinned": self.already_pinned,
        }


@dataclass(frozen=True)
class FacetPinApplyReport:
    transcript_path: Path
    backup_path: Path | None
    dry_run: bool
    reason: str
    requested_count: int
    resolved: list[FacetPinResolution] = field(default_factory=list)
    duplicate_inputs: list[str] = field(default_factory=list)

    @property
    def records_appended(self) -> int:
        if self.dry_run:
            return 0
        return sum(1 for resolution in self.resolved if not resolution.already_pinned)

    def to_dict(self) -> dict[str, object]:
        return {
            "transcript_path": str(self.transcript_path),
            "backup_path": str(self.backup_path) if self.backup_path else None,
            "dry_run": self.dry_run,
            "reason": self.reason,
            "requested_count": self.requested_count,
            "records_appended": self.records_appended,
            "duplicate_inputs": self.duplicate_inputs,
            "resolved": [resolution.to_dict() for resolution in self.resolved],
        }


def apply_facet_pins(
    transcript_path: Path,
    facet_id_prefixes: list[str],
    *,
    reason: str = DEFAULT_REASON,
    dry_run: bool = False,
    backup: bool = True,
    validate: bool = True,
    report_json: Path | None = None,
    write_report: bool = False,
) -> FacetPinApplyReport:
    """Resolve facet id prefixes and append facet pin records."""

    transcript_path = transcript_path.expanduser().resolve()
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript not found: {transcript_path}")

    rehydrated = Rehydrator(transcript_path).run()
    requested, duplicate_inputs = _dedupe_requested(facet_id_prefixes)
    resolved = [
        _resolve_facet_prefix(rehydrated.semantic_blocks, requested_prefix)
        for requested_prefix in requested
    ]
    if not reason.strip():
        raise ValueError("Pin reason must not be empty.")
    resolved = [
        _with_existing_pin_status(rehydrated.semantic_blocks, resolution)
        for resolution in resolved
    ]

    records_to_append = [
        _pin_record(rehydrated, resolution, reason=reason.strip())
        for resolution in resolved
        if not resolution.already_pinned
    ]
    backup_path: Path | None = None
    if records_to_append and not dry_run:
        backup_path = _backup_path(transcript_path) if backup else None
        _append_records(
            transcript_path=transcript_path,
            records=records_to_append,
            backup_path=backup_path,
            validate=validate,
        )

    report = FacetPinApplyReport(
        transcript_path=transcript_path,
        backup_path=backup_path,
        dry_run=dry_run,
        reason=reason.strip(),
        requested_count=len(facet_id_prefixes),
        resolved=resolved,
        duplicate_inputs=duplicate_inputs,
    )
    if write_report:
        _write_report(report, report_json or _default_report_path(transcript_path))
    return report


def _dedupe_requested(values: list[str]) -> tuple[list[str], list[str]]:
    seen: set[str] = set()
    requested: list[str] = []
    duplicates: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized:
            continue
        if normalized in seen:
            duplicates.append(normalized)
            continue
        seen.add(normalized)
        requested.append(normalized)
    if not requested:
        raise ValueError("At least one facet id prefix is required.")
    return requested, duplicates


def _resolve_facet_prefix(
    blocks: list[IRSemanticBlock],
    requested: str,
) -> FacetPinResolution:
    matches: list[FacetPinResolution] = []
    for block in blocks:
        for artifact in block.artifacts:
            if artifact.type != "summary":
                continue
            for facet in artifact.facets:
                if not facet.id.startswith(requested):
                    continue
                matches.append(
                    FacetPinResolution(
                        requested=requested,
                        facet_id=facet.id,
                        facet_title=facet.title,
                        block_id=block.id,
                        block_idx=block.idx,
                        block_title=block.title,
                    )
                )

    if not matches:
        raise ValueError(
            f'Facet id prefix "{requested}" did not match any summary facets.\n\n'
            f"Available facets:\n{_facet_candidates(blocks)}"
        )
    if len(matches) > 1:
        match_lines = "\n".join(
            f'- {match.facet_id} in block {match.block_idx} "{match.block_title}"'
            for match in matches
        )
        raise ValueError(
            f'Facet id prefix "{requested}" is ambiguous.\n\n'
            f"Matches:\n{match_lines}"
        )
    return matches[0]


def _with_existing_pin_status(
    blocks: list[IRSemanticBlock],
    resolution: FacetPinResolution,
) -> FacetPinResolution:
    block = next(block for block in blocks if block.id == resolution.block_id)
    already_pinned = any(
        pin.kind == "facet" and pin.facet_id == resolution.facet_id
        for pin in block.facet_pins
    )
    return FacetPinResolution(
        requested=resolution.requested,
        facet_id=resolution.facet_id,
        facet_title=resolution.facet_title,
        block_id=resolution.block_id,
        block_idx=resolution.block_idx,
        block_title=resolution.block_title,
        already_pinned=already_pinned,
    )


def _pin_record(
    rehydrated: RehydrationResult,
    resolution: FacetPinResolution,
    *,
    reason: str,
) -> IRSemanticBlockPinRecord:
    turn, turn_id = _pin_turn_stamp(rehydrated)
    return IRSemanticBlockPinRecord(
        session_id=rehydrated.session_id,
        block_id=resolution.block_id,
        pin=IRSemanticBlockPin(
            kind="facet",
            reason=reason,
            facet_id=resolution.facet_id,
        ),
        turn=turn,
        turn_id=turn_id,
    )


def _pin_turn_stamp(rehydrated: RehydrationResult) -> tuple[int, str]:
    if rehydrated.in_progress_turn is not None and rehydrated.current_turn_id:
        return rehydrated.in_progress_turn, rehydrated.current_turn_id

    for record in reversed(rehydrated.records):
        if isinstance(record, IRTurnEndRecord | IRTurnStartRecord):
            return record.turn, record.turn_id

    return rehydrated.last_completed_turn, "manual_facet_pin"


def _append_records(
    *,
    transcript_path: Path,
    records: Sequence[IRRecord],
    backup_path: Path | None,
    validate: bool,
) -> None:
    existing = transcript_path.read_text(encoding="utf-8")
    appended = "".join(record.model_dump_json() + "\n" for record in records)
    tmp_path = transcript_path.with_name(transcript_path.name + ".tmp")
    try:
        tmp_path.write_text(existing.rstrip("\n") + "\n" + appended, encoding="utf-8")
        if validate:
            Rehydrator(tmp_path).run()
        if backup_path is not None:
            shutil.copy2(transcript_path, backup_path)
        tmp_path.replace(transcript_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _facet_candidates(blocks: list[IRSemanticBlock]) -> str:
    lines: list[str] = []
    for block in blocks:
        for artifact in block.artifacts:
            if artifact.type != "summary":
                continue
            for facet in artifact.facets:
                lines.append(
                    f'- block {block.idx} "{block.title}": {facet.id} - {facet.title}'
                )
    return "\n".join(lines) if lines else "(no summary facets found)"


def _backup_path(transcript_path: Path) -> Path:
    candidate = transcript_path.with_suffix(transcript_path.suffix + ".facet-pins.bak")
    if not candidate.exists():
        return candidate
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return transcript_path.with_suffix(
        transcript_path.suffix + f".facet-pins.bak.{timestamp}"
    )


def _default_report_path(transcript_path: Path) -> Path:
    return transcript_path.with_suffix(".facet-pins-report.json")


def _write_report(report: FacetPinApplyReport, report_path: Path) -> None:
    report_path = report_path.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="scripts.apply_facet_pins",
        description="Append facet pin records to a core transcript by facet id prefix.",
    )
    parser.add_argument("transcript", type=Path, help="Core transcript path.")
    parser.add_argument(
        "facet_ids",
        nargs="+",
        help="Facet id prefixes to pin, for example facet_477e5c97.",
    )
    parser.add_argument(
        "--reason",
        default=DEFAULT_REASON,
        help=f"Reason stored on each facet pin. Defaults to: {DEFAULT_REASON}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and report without rewriting the transcript.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Rewrite without creating transcript.jsonl.facet-pins.bak.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validating the rewritten transcript with the core Rehydrator.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        help="Optionally write the report as JSON.",
    )
    return parser.parse_args(argv)


def _print_report(report: FacetPinApplyReport) -> None:
    action = "Would append" if report.dry_run else "Appended"
    print(f"{action} {report.records_appended} facet pin record(s).")
    print(f"Transcript: {report.transcript_path}")
    if report.backup_path is not None:
        print(f"Backup: {report.backup_path}")
    if report.duplicate_inputs:
        print(f"Duplicate inputs skipped: {', '.join(report.duplicate_inputs)}")
    for resolution in report.resolved:
        status = "already pinned" if resolution.already_pinned else "pinned"
        print(
            f"- {status}: {resolution.requested} -> {resolution.facet_id} "
            f'in block {resolution.block_idx} "{resolution.block_title}"'
        )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    report = apply_facet_pins(
        args.transcript,
        args.facet_ids,
        reason=args.reason,
        dry_run=args.dry_run,
        backup=not args.no_backup,
        validate=not args.no_validate,
        report_json=args.report_json,
        write_report=args.report_json is not None,
    )
    _print_report(report)


if __name__ == "__main__":
    main()

"""Repair a core session record's tool surface to the current registry."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from spellbook.config import SpellbookConfig
from spellbook.ir_types import IRSessionRecord, IRToolRecord
from spellbook.rehydrator import Rehydrator
from spellbook.tools.registry import ToolRegistry

RepairStatus = Literal["updated", "refreshed", "unchanged"]


@dataclass(frozen=True)
class SessionToolSurfaceRepairReport:
    transcript_path: Path
    backup_path: Path | None
    dry_run: bool
    status: RepairStatus
    previous_tool_count: int
    repaired_tool_count: int
    previous_tools: list[str]
    repaired_tools: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "transcript_path": str(self.transcript_path),
            "backup_path": str(self.backup_path) if self.backup_path else None,
            "dry_run": self.dry_run,
            "status": self.status,
            "previous_tool_count": self.previous_tool_count,
            "repaired_tool_count": self.repaired_tool_count,
            "previous_tools": self.previous_tools,
            "repaired_tools": self.repaired_tools,
        }


def repair_session_tool_surface(
    transcript_path: Path,
    *,
    backup: bool = True,
    dry_run: bool = False,
    force: bool = False,
    report_json: Path | None = None,
    write_report: bool = True,
) -> SessionToolSurfaceRepairReport:
    """Refresh the first session record's tools to the current main surface."""

    transcript_path = transcript_path.expanduser().resolve()
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript not found: {transcript_path}")

    lines = transcript_path.read_text(encoding="utf-8").splitlines()
    session_idx = _first_record_index(lines)
    session_data = _read_session_dict(lines[session_idx])
    config = SpellbookConfig.model_validate(session_data["config"])
    repaired_tools = current_main_tool_surface(config)
    previous_tools = _read_tool_records(session_data)

    status = _repair_status(
        previous_tools=previous_tools,
        repaired_tools=repaired_tools,
        force=force,
    )
    backup_path: Path | None = None
    if status != "unchanged" and not dry_run:
        backup_path = _backup_path(transcript_path) if backup else None
        updated_session = {
            **session_data,
            "tools": [tool.model_dump(mode="json") for tool in repaired_tools],
        }
        IRSessionRecord.model_validate(updated_session)
        updated_lines = list(lines)
        updated_lines[session_idx] = json.dumps(updated_session, separators=(",", ":"))
        _atomic_write_lines(
            transcript_path=transcript_path,
            lines=updated_lines,
            backup_path=backup_path,
        )

    report = SessionToolSurfaceRepairReport(
        transcript_path=transcript_path,
        backup_path=backup_path,
        dry_run=dry_run,
        status=status,
        previous_tool_count=len(previous_tools),
        repaired_tool_count=len(repaired_tools),
        previous_tools=[tool.name for tool in previous_tools],
        repaired_tools=[tool.name for tool in repaired_tools],
    )
    if write_report:
        _write_report(report, report_json or _default_report_path(transcript_path))
    return report


def current_main_tool_surface(config: SpellbookConfig) -> list[IRToolRecord]:
    registry = ToolRegistry.build(config.tool_categories, surface="main")
    return registry.records


def _repair_status(
    *,
    previous_tools: list[IRToolRecord],
    repaired_tools: list[IRToolRecord],
    force: bool,
) -> RepairStatus:
    if force:
        return "refreshed"
    if previous_tools == repaired_tools:
        return "unchanged"
    return "updated"


def _first_record_index(lines: list[str]) -> int:
    for idx, line in enumerate(lines):
        if line.strip():
            return idx
    raise ValueError("Transcript is empty.")


def _read_session_dict(line: str) -> dict:
    data = json.loads(line)
    if not isinstance(data, dict) or data.get("ir") != "session":
        raise ValueError("First transcript record must be a session record.")
    if "config" not in data:
        raise ValueError("Session record is missing config.")
    return data


def _read_tool_records(session_data: dict) -> list[IRToolRecord]:
    tools = session_data.get("tools")
    if not isinstance(tools, list):
        return []
    records: list[IRToolRecord] = []
    for raw_tool in tools:
        if not isinstance(raw_tool, dict):
            continue
        records.append(IRToolRecord.model_validate(raw_tool))
    return records


def _atomic_write_lines(
    *,
    transcript_path: Path,
    lines: list[str],
    backup_path: Path | None,
) -> None:
    tmp_path = transcript_path.with_name(transcript_path.name + ".tmp")
    try:
        tmp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        Rehydrator(tmp_path).run()
        if backup_path is not None:
            shutil.copy2(transcript_path, backup_path)
        tmp_path.replace(transcript_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _backup_path(transcript_path: Path) -> Path:
    candidate = transcript_path.with_suffix(transcript_path.suffix + ".tools.bak")
    if not candidate.exists():
        return candidate
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return transcript_path.with_suffix(
        transcript_path.suffix + f".tools.bak.{timestamp}"
    )


def _default_report_path(transcript_path: Path) -> Path:
    return transcript_path.with_suffix(".tool-surface-repair-report.json")


def _write_report(
    report: SessionToolSurfaceRepairReport,
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
        prog="scripts.repair_session_tool_surface",
        description=(
            "Refresh a core transcript session record's tools to the current "
            "main ToolRegistry surface."
        ),
    )
    parser.add_argument("transcript", type=Path, help="Core transcript path.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report without rewriting the transcript.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rewrite even if the tool surface already matches.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Rewrite without creating transcript.jsonl.tools.bak.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        help="Path for the repair report. Defaults beside the transcript.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    report = repair_session_tool_surface(
        args.transcript,
        backup=not args.no_backup,
        dry_run=args.dry_run,
        force=args.force,
        report_json=args.report_json,
    )
    print(
        "Tool surface repair: "
        f"{report.status}; "
        f"{report.previous_tool_count} -> {report.repaired_tool_count} tools."
    )
    if report.backup_path is not None:
        print(f"Backup: {report.backup_path}")
    print(f"Report: {args.report_json or _default_report_path(args.transcript)}")


if __name__ == "__main__":
    main()

"""Repair missing or empty skill catalogs on core session records."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from spellbook.config import SpellbookConfig
from spellbook.ir_types import IRSessionRecord, IRSkillCatalog
from spellbook.rehydrator import Rehydrator
from spellbook.skills.manager import SkillManager

RepairStatus = Literal["populated_missing", "populated_empty", "refreshed", "unchanged"]


@dataclass(frozen=True)
class SessionSkillCatalogRepairReport:
    transcript_path: Path
    backup_path: Path | None
    dry_run: bool
    status: RepairStatus
    cwd: Path
    previous_skill_count: int | None
    discovered_skill_count: int
    skills: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "transcript_path": str(self.transcript_path),
            "backup_path": str(self.backup_path) if self.backup_path else None,
            "dry_run": self.dry_run,
            "status": self.status,
            "cwd": str(self.cwd),
            "previous_skill_count": self.previous_skill_count,
            "discovered_skill_count": self.discovered_skill_count,
            "skills": self.skills,
        }


def repair_session_skill_catalog(
    transcript_path: Path,
    *,
    backup: bool = True,
    dry_run: bool = False,
    force: bool = False,
    report_json: Path | None = None,
    write_report: bool = True,
) -> SessionSkillCatalogRepairReport:
    """Populate the first session record's skill catalog from its config cwd."""

    transcript_path = transcript_path.expanduser().resolve()
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript not found: {transcript_path}")

    lines = transcript_path.read_text(encoding="utf-8").splitlines()
    session_idx = _first_record_index(lines)
    session_data = _read_session_dict(lines[session_idx])
    config = SpellbookConfig.model_validate(session_data["config"])
    catalog = discover_skill_catalog(config)
    previous_skill_count = _skill_count(session_data)

    status = _repair_status(
        has_catalog="skill_catalog" in session_data,
        previous_skill_count=previous_skill_count,
        force=force,
    )
    backup_path: Path | None = None
    if status != "unchanged" and not dry_run:
        backup_path = _backup_path(transcript_path) if backup else None
        updated_session = {
            **session_data,
            "skill_catalog": catalog.model_dump(mode="json"),
        }
        IRSessionRecord.model_validate(updated_session)
        updated_lines = list(lines)
        updated_lines[session_idx] = json.dumps(updated_session, separators=(",", ":"))
        _atomic_write_lines(
            transcript_path=transcript_path,
            lines=updated_lines,
            backup_path=backup_path,
        )

    report = SessionSkillCatalogRepairReport(
        transcript_path=transcript_path,
        backup_path=backup_path,
        dry_run=dry_run,
        status=status,
        cwd=config.cwd,
        previous_skill_count=previous_skill_count,
        discovered_skill_count=len(catalog.skills),
        skills=sorted(catalog.skills),
    )
    if write_report:
        _write_report(report, report_json or _default_report_path(transcript_path))
    return report


def discover_skill_catalog(config: SpellbookConfig) -> IRSkillCatalog:
    manager = SkillManager(config=config)
    return manager.discover_skills()


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


def _skill_count(session_data: dict) -> int | None:
    catalog = session_data.get("skill_catalog")
    if not isinstance(catalog, dict):
        return None
    skills = catalog.get("skills")
    if not isinstance(skills, dict):
        return None
    return len(skills)


def _repair_status(
    *,
    has_catalog: bool,
    previous_skill_count: int | None,
    force: bool,
) -> RepairStatus:
    if force:
        return "refreshed"
    if not has_catalog:
        return "populated_missing"
    if previous_skill_count == 0:
        return "populated_empty"
    return "unchanged"


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
    candidate = transcript_path.with_suffix(transcript_path.suffix + ".skill.bak")
    if not candidate.exists():
        return candidate
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return transcript_path.with_suffix(
        transcript_path.suffix + f".skill.bak.{timestamp}"
    )


def _default_report_path(transcript_path: Path) -> Path:
    return transcript_path.with_suffix(".skill-catalog-repair-report.json")


def _write_report(
    report: SessionSkillCatalogRepairReport,
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
        prog="scripts.repair_session_skill_catalog",
        description=(
            "Populate a core transcript session record's missing or empty "
            "skill_catalog from its configured cwd."
        ),
    )
    parser.add_argument("transcript", type=Path, help="Core transcript path.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover and report without rewriting the transcript.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refresh the session skill catalog even if it is already populated.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Rewrite without creating transcript.jsonl.skill.bak.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        help="Path for the repair report. Defaults beside the transcript.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    report = repair_session_skill_catalog(
        args.transcript,
        backup=not args.no_backup,
        dry_run=args.dry_run,
        force=args.force,
        report_json=args.report_json,
    )
    print(
        "Skill catalog repair: "
        f"{report.status}; "
        f"{report.discovered_skill_count} discovered from cwd {report.cwd}."
    )
    if report.backup_path is not None:
        print(f"Backup: {report.backup_path}")
    print(f"Report: {args.report_json or _default_report_path(args.transcript)}")


if __name__ == "__main__":
    main()

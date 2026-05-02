"""Repair meta-Claude core session records with the current meta frame prompt."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from spellbook.config import SpellbookConfig
from spellbook.frame_lite import FRAME_ADDENDUM_INTRO, discover_claude_md_paths
from spellbook.ir_types import IRSessionRecord
from spellbook.orientation import build_core_orientation
from spellbook.rehydrator import Rehydrator
from spellbook.skills.manager import SkillManager

RepairStatus = Literal["populated_empty", "refreshed", "unchanged"]


@dataclass(frozen=True)
class MetaSystemPromptRepairReport:
    transcript_path: Path
    backup_path: Path | None
    dry_run: bool
    status: RepairStatus
    cwd: Path
    previous_prompt_chars: int
    repaired_prompt_chars: int
    sections: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "transcript_path": str(self.transcript_path),
            "backup_path": str(self.backup_path) if self.backup_path else None,
            "dry_run": self.dry_run,
            "status": self.status,
            "cwd": str(self.cwd),
            "previous_prompt_chars": self.previous_prompt_chars,
            "repaired_prompt_chars": self.repaired_prompt_chars,
            "sections": self.sections,
        }


def repair_meta_system_prompt(
    transcript_path: Path,
    *,
    backup: bool = True,
    dry_run: bool = False,
    force: bool = False,
    report_json: Path | None = None,
    write_report: bool = True,
) -> MetaSystemPromptRepairReport:
    """Populate a meta-Claude core transcript's session system prompt."""

    transcript_path = transcript_path.expanduser().resolve()
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript not found: {transcript_path}")

    lines = transcript_path.read_text(encoding="utf-8").splitlines()
    session_idx = _first_record_index(lines)
    session_data = _read_session_dict(lines[session_idx])
    config = SpellbookConfig.model_validate(session_data["config"])
    if not is_meta_cwd(config.cwd):
        raise ValueError(
            f"Meta prompt repair only applies to cwd {meta_cwd()}, got {config.cwd}."
        )

    prompt, sections = build_meta_system_prompt(config)
    previous_prompt = config.system_prompt
    status = _repair_status(previous_prompt=previous_prompt, force=force)
    backup_path: Path | None = None
    if status != "unchanged" and not dry_run:
        backup_path = _backup_path(transcript_path) if backup else None
        updated_config = config.model_copy(update={"system_prompt": prompt})
        updated_session = {
            **session_data,
            "config": updated_config.model_dump(mode="json"),
        }
        IRSessionRecord.model_validate(updated_session)
        updated_lines = list(lines)
        updated_lines[session_idx] = json.dumps(updated_session, separators=(",", ":"))
        _atomic_write_lines(
            transcript_path=transcript_path,
            lines=updated_lines,
            backup_path=backup_path,
        )

    report = MetaSystemPromptRepairReport(
        transcript_path=transcript_path,
        backup_path=backup_path,
        dry_run=dry_run,
        status=status,
        cwd=config.cwd,
        previous_prompt_chars=len(previous_prompt),
        repaired_prompt_chars=len(prompt),
        sections=sections,
    )
    if write_report:
        _write_report(report, report_json or _default_report_path(transcript_path))
    return report


def build_meta_system_prompt(config: SpellbookConfig) -> tuple[str, list[str]]:
    """Render the old meta frame shape with the current core orientation."""

    sections: list[tuple[str, str]] = []
    section_names: list[str] = []
    for path in discover_claude_md_paths(config.cwd):
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        title = _first_heading(content) or path.name
        sections.append((title, content))
        section_names.append(title)

    skill_manager = SkillManager(config=config)
    skill_manager.discover_skills()
    if skill_manager.num_skills > 0:
        sections.append(("Available skills", skill_manager.render_prompt_addendum()))
        section_names.append("Available skills")

    meta_identity = config.cwd / "prompts" / "meta-claude.md"
    if meta_identity.is_file():
        sections.append(("Meta-Claude", meta_identity.read_text(encoding="utf-8")))
        section_names.append("Meta-Claude")

    working_memory = meta_working_memory_path(config.cwd)
    if working_memory is not None:
        sections.append(
            (
                "Meta-Claude Working Memory",
                working_memory.read_text(encoding="utf-8"),
            )
        )
        section_names.append("Meta-Claude Working Memory")

    parts = [build_core_orientation(config.model, cwd=config.cwd).rstrip()]
    if sections:
        parts.append(FRAME_ADDENDUM_INTRO)
        for title, content in sections:
            content = content.strip()
            if content:
                parts.append(f"## {title}\n\n{content}")
    return "\n\n".join(part for part in parts if part).strip(), section_names


def is_meta_cwd(cwd: Path) -> bool:
    return cwd.expanduser().resolve() == meta_cwd()


def meta_cwd() -> Path:
    return (Path.home() / ".chorus").expanduser().resolve()


def meta_working_memory_path(cwd: Path) -> Path | None:
    projects_dir = Path.home() / ".claude" / "projects"
    if not projects_dir.is_dir():
        return None

    encoded = str(cwd.expanduser().resolve()).replace("/", "-").replace(".", "-")
    candidates = [projects_dir / encoded / "memory" / "MEMORY.md"]
    candidates.extend(
        directory / "memory" / "MEMORY.md"
        for directory in sorted(projects_dir.iterdir())
        if directory.is_dir() and directory.name.endswith("--chorus")
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _repair_status(*, previous_prompt: str, force: bool) -> RepairStatus:
    if force:
        return "refreshed"
    if previous_prompt.strip():
        return "unchanged"
    return "populated_empty"


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


def _first_heading(markdown: str) -> str | None:
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        title = stripped.lstrip("#").strip()
        if title:
            return title
    return None


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
    candidate = transcript_path.with_suffix(transcript_path.suffix + ".meta-prompt.bak")
    if not candidate.exists():
        return candidate
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return transcript_path.with_suffix(
        transcript_path.suffix + f".meta-prompt.bak.{timestamp}"
    )


def _default_report_path(transcript_path: Path) -> Path:
    return transcript_path.with_suffix(".meta-prompt-repair-report.json")


def _write_report(report: MetaSystemPromptRepairReport, report_path: Path) -> None:
    report_path = report_path.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="scripts.repair_meta_system_prompt",
        description=(
            "Populate a meta-Claude core transcript session record with the "
            "current meta system prompt."
        ),
    )
    parser.add_argument("transcript", type=Path, help="Core transcript path.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render and report without rewriting the transcript.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refresh the prompt even if config.system_prompt is already populated.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Rewrite without creating transcript.jsonl.meta-prompt.bak.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        help="Path for the repair report. Defaults beside the transcript.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    report = repair_meta_system_prompt(
        args.transcript,
        backup=not args.no_backup,
        dry_run=args.dry_run,
        force=args.force,
        report_json=args.report_json,
    )
    print(
        "Meta system prompt repair: "
        f"{report.status}; "
        f"{report.repaired_prompt_chars} chars; "
        f"{len(report.sections)} supplemental sections."
    )
    if report.backup_path is not None:
        print(f"Backup: {report.backup_path}")
    print(f"Report: {args.report_json or _default_report_path(args.transcript)}")


if __name__ == "__main__":
    main()

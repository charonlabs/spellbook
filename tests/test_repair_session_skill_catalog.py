from __future__ import annotations

import json
from pathlib import Path

from scripts.repair_session_skill_catalog import repair_session_skill_catalog
from spellbook.config import SpellbookConfig
from spellbook.ir_types import IRSkillCatalog
from spellbook.rehydrator import Rehydrator


def _write_skill(
    cwd: Path,
    *,
    skill_dir: str = "compose",
    name: str = "compose",
    description: str = "Draft careful prose.",
) -> None:
    skill_path = cwd / ".test-skills" / "skills" / skill_dir / "SKILL.md"
    skill_path.parent.mkdir(parents=True, exist_ok=True)
    skill_path.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\nUse care.\n",
        encoding="utf-8",
    )


def _write_session(
    transcript: Path,
    config: SpellbookConfig,
    *,
    skill_catalog: IRSkillCatalog | None,
) -> None:
    record = {
        "session_id": "s1",
        "ir": "session",
        "time": "2026-04-29T00:00:00Z",
        "config": config.model_dump(mode="json"),
        "tools": [],
    }
    if skill_catalog is not None:
        record["skill_catalog"] = skill_catalog.model_dump(mode="json")
    transcript.write_text(json.dumps(record) + "\n", encoding="utf-8")


def test_repair_populates_missing_session_skill_catalog(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    _write_skill(tmp_path)
    transcript = tmp_path / "transcript.jsonl"
    config = SpellbookConfig(
        model="claude-sonnet-4-6",
        cwd=tmp_path,
        skill_discovery_dirs=[".test-skills"],
    )
    _write_session(transcript, config, skill_catalog=None)

    report = repair_session_skill_catalog(
        transcript,
        backup=False,
        write_report=False,
    )

    assert report.status == "populated_missing"
    assert report.previous_skill_count is None
    assert report.discovered_skill_count == 1
    result = Rehydrator(transcript).run()
    assert set(result.skill_catalog.skills) == {"compose"}


def test_repair_populates_empty_session_skill_catalog(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    _write_skill(tmp_path)
    transcript = tmp_path / "transcript.jsonl"
    config = SpellbookConfig(
        model="claude-sonnet-4-6",
        cwd=tmp_path,
        skill_discovery_dirs=[".test-skills"],
    )
    _write_session(transcript, config, skill_catalog=IRSkillCatalog())

    report = repair_session_skill_catalog(
        transcript,
        backup=False,
        write_report=False,
    )

    assert report.status == "populated_empty"
    assert report.previous_skill_count == 0
    result = Rehydrator(transcript).run()
    assert set(result.skill_catalog.skills) == {"compose"}

from __future__ import annotations

import json
from pathlib import Path

from scripts.repair_meta_system_prompt import repair_meta_system_prompt
from spellbook.config import SpellbookConfig
from spellbook.ir_types import IRSkillCatalog
from spellbook.rehydrator import Rehydrator


def _write_meta_frame_files(home: Path) -> Path:
    chorus_home = home / ".chorus"
    chorus_home.mkdir(parents=True)
    (chorus_home / "CLAUDE.md").write_text(
        "# Chorus Workspace\nWorkspace instruction.",
        encoding="utf-8",
    )
    prompts = chorus_home / "prompts"
    prompts.mkdir()
    (prompts / "meta-claude.md").write_text(
        "# Meta-Claude\nMeta identity instruction.",
        encoding="utf-8",
    )

    global_claude = home / ".claude" / "CLAUDE.md"
    global_claude.parent.mkdir(parents=True)
    global_claude.write_text(
        "# About Ryan\nUser model instruction.",
        encoding="utf-8",
    )

    memory = (
        home
        / ".claude"
        / "projects"
        / str(chorus_home.resolve()).replace("/", "-").replace(".", "-")
        / "memory"
        / "MEMORY.md"
    )
    memory.parent.mkdir(parents=True)
    memory.write_text(
        "# Meta-Claude Working Memory\nWorking memory instruction.",
        encoding="utf-8",
    )

    skill_path = chorus_home / ".test-skills" / "skills" / "compose" / "SKILL.md"
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text(
        "---\nname: compose\ndescription: Draft careful prose.\n---\n\nUse care.\n",
        encoding="utf-8",
    )
    return chorus_home


def _write_session(transcript: Path, config: SpellbookConfig) -> None:
    record = {
        "session_id": "s1",
        "ir": "session",
        "time": "2026-04-29T00:00:00Z",
        "config": config.model_dump(mode="json"),
        "tools": [],
        "skill_catalog": IRSkillCatalog().model_dump(mode="json"),
    }
    transcript.write_text(json.dumps(record) + "\n", encoding="utf-8")


def test_repair_meta_system_prompt_populates_old_meta_frame_shape(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    chorus_home = _write_meta_frame_files(home)
    transcript = tmp_path / "transcript.jsonl"
    config = SpellbookConfig(
        model="claude-opus-4-6",
        cwd=chorus_home,
        skill_discovery_dirs=[".test-skills"],
    )
    _write_session(transcript, config)

    report = repair_meta_system_prompt(
        transcript,
        backup=False,
        write_report=False,
    )

    assert report.status == "populated_empty"
    assert report.sections == [
        "About Ryan",
        "Chorus Workspace",
        "Available skills",
        "Meta-Claude",
        "Meta-Claude Working Memory",
    ]
    result = Rehydrator(transcript).run()
    prompt = result.config.system_prompt
    assert "Claude Opus 4.6 entity" in prompt
    assert "This is a safe place. Be yourself." in prompt
    assert "## About Ryan\n\n# About Ryan\nUser model instruction." in prompt
    assert "## Chorus Workspace\n\n# Chorus Workspace\nWorkspace instruction." in prompt
    assert "## Available skills\n\n<available-skills>" in prompt
    assert "<name>compose</name>" in prompt
    assert "## Meta-Claude\n\n# Meta-Claude\nMeta identity instruction." in prompt
    assert (
        "## Meta-Claude Working Memory\n\n"
        "# Meta-Claude Working Memory\nWorking memory instruction."
    ) in prompt
    assert prompt.index("Claude Opus 4.6 entity") < prompt.index("## About Ryan")
    assert prompt.index("## About Ryan") < prompt.index("## Chorus Workspace")
    assert prompt.index("## Chorus Workspace") < prompt.index("## Available skills")
    assert prompt.index("## Available skills") < prompt.index("## Meta-Claude")
    assert prompt.index("## Meta-Claude") < prompt.index(
        "## Meta-Claude Working Memory"
    )

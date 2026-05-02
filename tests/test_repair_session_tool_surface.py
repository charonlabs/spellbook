from __future__ import annotations

import json
from pathlib import Path

from scripts.repair_session_tool_surface import repair_session_tool_surface
from spellbook.config import SpellbookConfig
from spellbook.ir_types import IRSkillCatalog
from spellbook.rehydrator import Rehydrator
from spellbook.tools.registry import ToolRegistry


def _write_session(
    transcript: Path,
    config: SpellbookConfig,
    *,
    tools: list[dict],
) -> None:
    record = {
        "session_id": "s1",
        "ir": "session",
        "time": "2026-04-30T00:00:00Z",
        "config": config.model_dump(mode="json"),
        "tools": tools,
        "skill_catalog": IRSkillCatalog().model_dump(mode="json"),
    }
    transcript.write_text(json.dumps(record) + "\n", encoding="utf-8")


def test_repair_session_tool_surface_updates_old_tools(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.jsonl"
    config = SpellbookConfig(model="claude-sonnet-4-6", cwd=tmp_path)
    _write_session(
        transcript,
        config,
        tools=[
            {
                "name": "Read",
                "input_schema": {"type": "object", "properties": {}},
                "category": "filesystem",
            }
        ],
    )

    report = repair_session_tool_surface(
        transcript,
        backup=False,
        write_report=False,
    )

    expected_names = [
        record.name
        for record in ToolRegistry.build(config.tool_categories, surface="main").records
    ]
    result = Rehydrator(transcript).run()
    assert report.status == "updated"
    assert report.previous_tools == ["Read"]
    assert report.repaired_tools == expected_names
    assert [tool.name for tool in result.tools] == expected_names


def test_repair_session_tool_surface_is_unchanged_when_current(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.jsonl"
    config = SpellbookConfig(model="claude-sonnet-4-6", cwd=tmp_path)
    registry = ToolRegistry.build(config.tool_categories, surface="main")
    _write_session(
        transcript,
        config,
        tools=[record.model_dump(mode="json") for record in registry.records],
    )

    report = repair_session_tool_surface(
        transcript,
        backup=False,
        write_report=False,
    )

    assert report.status == "unchanged"
    assert report.previous_tool_count == len(registry.records)
    assert report.repaired_tool_count == len(registry.records)


def test_repair_session_tool_surface_dry_run_does_not_rewrite(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.jsonl"
    config = SpellbookConfig(model="claude-sonnet-4-6", cwd=tmp_path)
    _write_session(transcript, config, tools=[])
    before = transcript.read_text(encoding="utf-8")

    report = repair_session_tool_surface(
        transcript,
        dry_run=True,
        write_report=False,
    )

    assert report.status == "updated"
    assert report.dry_run is True
    assert transcript.read_text(encoding="utf-8") == before

from __future__ import annotations

import json
from pathlib import Path

from scripts.append_legacy_tail import append_legacy_tail
from spellbook.config import SpellbookConfig
from spellbook.ir_types import (
    IRAssistantTextBlock,
    IRBlockRecord,
    IRRecord,
    IRSessionRecord,
    IRSkillCatalog,
    IRTurnEndRecord,
    IRTurnStartRecord,
    IRUserTextBlock,
)
from spellbook.rehydrator import Rehydrator
from spellbook.tools.registry import ToolRegistry


def _write_records(path: Path, records: list[IRRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(record.model_dump_json() for record in records) + "\n",
        encoding="utf-8",
    )


def _write_target(path: Path) -> None:
    config = SpellbookConfig(model="claude-sonnet-4-6", cwd=path.parent)
    registry = ToolRegistry.build(config.tool_categories, surface=config.session_type)
    _write_records(
        path,
        [
            IRSessionRecord(
                session_id="core_session",
                config=config,
                tools=registry.records,
                skill_catalog=IRSkillCatalog(),
            ),
            IRTurnStartRecord(
                session_id="core_session",
                turn=1,
                turn_id="turn_1",
            ),
            IRBlockRecord(
                session_id="core_session",
                turn=1,
                seq=0,
                event=IRUserTextBlock(
                    text="already imported",
                    origin="human",
                    turn_id="turn_1",
                ),
            ),
            IRTurnEndRecord(
                session_id="core_session",
                turn=1,
                turn_id="turn_1",
                stop_reason="end_turn",
            ),
        ],
    )


def _write_legacy(path: Path) -> None:
    records = [
        _session_record(path.parent),
        _turn_start(1),
        _event(1, 0, "user_message", {"text": "already imported", "source": "tui"}),
        _turn_end(1),
        _turn_start(2),
        _event(2, 0, "user_message", {"text": "new user", "source": "tui"}),
        _event(2, 1, "system", {"kind": "request_anchor"}),
        _event(2, 2, "assistant_text", {"text": "new assistant"}),
        _turn_end(2),
        _turn_start(3),
        _event(3, 0, "user_message", {"text": "open turn", "source": "tui"}),
    ]
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def _session_record(tmp_path: Path) -> dict:
    return {
        "ir": "session",
        "session": "legacy_session",
        "data": {
            "id": "legacy_session",
            "model": "claude-opus-4-7",
            "start_time": "2026-04-25T05:02:00+00:00",
            "cwd": str(tmp_path),
            "extra": {"spellbook": {"provider": "anthropic", "effort": "high"}},
        },
    }


def _turn_start(turn: int) -> dict:
    return {
        "ir": "turn_start",
        "session": "legacy_session",
        "turn": turn,
        "turn_id": f"turn_{turn}",
        "time": f"2026-04-25T05:0{turn}:00+00:00",
    }


def _turn_end(turn: int) -> dict:
    return {
        "ir": "turn_end",
        "session": "legacy_session",
        "turn": turn,
        "turn_id": f"turn_{turn}",
        "time": f"2026-04-25T05:0{turn}:30+00:00",
    }


def _event(turn: int, seq: int, event_type: str, data: dict) -> dict:
    return {
        "ir": "event",
        "session": "legacy_session",
        "turn": turn,
        "turn_id": f"turn_{turn}",
        "seq": seq,
        "event": {
            "type": event_type,
            "seq": seq,
            "time": f"2026-04-25T05:0{turn}:{seq:02d}+00:00",
            "layer": "narrative"
            if event_type in {"user_message", "assistant_text"}
            else "activity",
            "data": data,
        },
    }


def test_append_legacy_tail_imports_only_new_completed_content_turns(tmp_path: Path) -> None:
    target = tmp_path / "target.jsonl"
    legacy = tmp_path / "legacy.jsonl"
    _write_target(target)
    _write_legacy(legacy)

    report = append_legacy_tail(
        legacy_path=legacy,
        target_path=target,
        backup=True,
    )

    result = Rehydrator(target).run()
    assert report.after_turn == 1
    assert report.latest_completed_legacy_turn == 2
    assert report.turns_appended == 1
    assert report.blocks_appended == 2
    assert report.records_appended == 4
    assert report.skipped_event_types == {"system": 1}
    assert report.skipped_open_turns == [3]
    assert report.backup_path is not None
    assert report.backup_path.exists()
    assert result.last_completed_turn == 2
    assert result.session_id == "core_session"
    assert len(result.blocks) == 3
    assert isinstance(result.blocks[1], IRUserTextBlock)
    assert result.blocks[1].text == "new user"
    assert result.blocks[1].turn_id == "turn_2"
    assert isinstance(result.blocks[2], IRAssistantTextBlock)
    assert result.blocks[2].text == "new assistant"


def test_append_legacy_tail_dry_run_does_not_rewrite(tmp_path: Path) -> None:
    target = tmp_path / "target.jsonl"
    legacy = tmp_path / "legacy.jsonl"
    _write_target(target)
    _write_legacy(legacy)
    before = target.read_text(encoding="utf-8")

    report = append_legacy_tail(
        legacy_path=legacy,
        target_path=target,
        dry_run=True,
    )

    assert report.dry_run is True
    assert report.records_appended == 4
    assert report.backup_path is None
    assert target.read_text(encoding="utf-8") == before

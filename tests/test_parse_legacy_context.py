from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import parse_legacy_context
from spellbook.ir_types import (
    IRAssistantTextBlock,
    IRBlockRecord,
    IRThinkingBlock,
    IRToolCallBlock,
    IRToolResultBlock,
    IRToolTextBlock,
    IRUserTextBlock,
)
from spellbook.rehydrator import Rehydrator


def _write_legacy(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def _session_record(tmp_path: Path, *, cwd: Path | None = None) -> dict:
    return {
        "ir": "session",
        "session": "legacy_session",
        "data": {
            "id": "legacy_session",
            "model": "claude-opus-4-7",
            "start_time": "2026-04-25T05:02:00+00:00",
            "cwd": str(cwd or tmp_path / "legacy-cwd"),
            "extra": {
                "spellbook": {
                    "provider": "anthropic",
                    "effort": "high",
                }
            },
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


def _write_skill(cwd: Path) -> None:
    skill_path = cwd / ".claude" / "skills" / "compose" / "SKILL.md"
    skill_path.parent.mkdir(parents=True, exist_ok=True)
    skill_path.write_text(
        "---\nname: compose\ndescription: Draft careful prose.\n---\n\nUse care.\n",
        encoding="utf-8",
    )


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
    global_claude.write_text("# About Ryan\nUser instruction.", encoding="utf-8")
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
        "# Meta-Claude Working Memory\nMemory instruction.",
        encoding="utf-8",
    )
    return chorus_home


def test_imports_legacy_conversation_into_core_transcript(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    _write_skill(tmp_path / "legacy-cwd")
    legacy = tmp_path / "legacy.jsonl"
    output = tmp_path / "core" / "transcript.jsonl"
    _write_legacy(
        legacy,
        [
            _session_record(tmp_path),
            _turn_start(1),
            _event(
                1,
                0,
                "user_message",
                {"text": "hello", "source": "tui", "origin": "direct"},
            ),
            _event(
                1,
                1,
                "thinking",
                {"text": "thinking text", "signature": "sig"},
            ),
            _event(1, 2, "assistant_text", {"text": "assistant text"}),
            _event(
                1,
                3,
                "tool_call",
                {
                    "call_id": "toolu_1",
                    "tool": "Read",
                    "input": {"file_path": "/tmp/a.txt"},
                },
            ),
            _event(1, 4, "system", {"kind": "request_anchor"}),
            _event(
                1,
                5,
                "tool_result",
                {
                    "call_id": "toolu_1",
                    "tool": "Read",
                    "output": "file contents",
                    "summary": "1 line",
                    "is_error": False,
                },
            ),
            _turn_end(1),
        ],
    )

    report = parse_legacy_context.import_legacy_transcript(
        input_path=legacy,
        output_path=output,
    )
    result = Rehydrator(output).run()

    assert report.records_read == 9
    assert report.records_written == 8
    assert report.blocks_written == 5
    assert report.turns_written == 1
    assert report.skipped_event_types == {"system": 1}
    assert result.session_id == "legacy_session"
    assert result.config.model == "claude-opus-4-7"
    assert result.config.cwd == tmp_path / "legacy-cwd"
    assert set(result.skill_catalog.skills) == {"compose"}
    assert result.last_completed_turn == 1
    assert {tool.name for tool in result.tools} >= {"Read", "Write", "Edit", "Bash"}

    user, thinking, assistant, tool_call, tool_result = result.blocks
    assert isinstance(user, IRUserTextBlock)
    assert user.origin == "human"
    assert user.turn_id == "turn_1"
    assert isinstance(thinking, IRThinkingBlock)
    assert thinking.text == "thinking text"
    assert thinking.signature == "sig"
    assert isinstance(assistant, IRAssistantTextBlock)
    assert assistant.text == "assistant text"
    assert isinstance(tool_call, IRToolCallBlock)
    assert tool_call.input == {"file_path": "/tmp/a.txt"}
    assert isinstance(tool_result, IRToolResultBlock)
    tool_result_text = tool_result.content[0]
    assert isinstance(tool_result_text, IRToolTextBlock)
    assert tool_result_text.text == "file contents"
    assert tool_result.display["summary"] == "1 line"


def test_import_populates_meta_system_prompt_for_chorus_cwd(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    chorus_home = _write_meta_frame_files(home)
    legacy = tmp_path / "legacy.jsonl"
    output = tmp_path / "core.jsonl"
    _write_legacy(legacy, [_session_record(tmp_path, cwd=chorus_home)])

    parse_legacy_context.import_legacy_transcript(
        input_path=legacy,
        output_path=output,
    )

    result = Rehydrator(output).run()
    prompt = result.config.system_prompt
    assert "Claude Opus 4.7 entity" in prompt
    assert "## Meta-Claude\n\n# Meta-Claude\nMeta identity instruction." in prompt
    assert (
        "## Meta-Claude Working Memory\n\n"
        "# Meta-Claude Working Memory\nMemory instruction."
    ) in prompt


def test_import_maps_conduit_user_messages(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.jsonl"
    output = tmp_path / "core.jsonl"
    _write_legacy(
        legacy,
        [
            _session_record(tmp_path),
            _turn_start(1),
            _event(
                1,
                0,
                "user_message",
                {"text": "from chorus", "source": "chorus", "origin": "conduit"},
            ),
            _turn_end(1),
        ],
    )

    parse_legacy_context.import_legacy_transcript(input_path=legacy, output_path=output)
    result = Rehydrator(output).run()

    assert isinstance(result.blocks[0], IRUserTextBlock)
    assert result.blocks[0].origin == "conduit"


def test_import_repairs_legacy_user_text_before_tool_results(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.jsonl"
    output = tmp_path / "core.jsonl"
    _write_legacy(
        legacy,
        [
            _session_record(tmp_path),
            _turn_start(1),
            _event(1, 0, "user_message", {"text": "please inspect", "source": "human"}),
            _event(
                1,
                1,
                "tool_call",
                {"call_id": "toolu_1", "tool": "Bash", "input": {"command": "pwd"}},
            ),
            _event(
                1,
                2,
                "user_message",
                {"text": "also, your cwd is already right", "source": "human"},
            ),
            _event(
                1,
                3,
                "tool_result",
                {"call_id": "toolu_1", "tool": "Bash", "output": "/tmp"},
            ),
            _turn_end(1),
        ],
    )

    report = parse_legacy_context.import_legacy_transcript(
        input_path=legacy,
        output_path=output,
    )
    result = Rehydrator(output).run()

    assert report.tool_result_order_repairs == 1
    assert [
        record.seq for record in result.records if isinstance(record, IRBlockRecord)
    ] == [
        0,
        1,
        2,
        3,
    ]
    user, tool_call, tool_result, interleaved_text = result.blocks
    assert isinstance(user, IRUserTextBlock)
    assert isinstance(tool_call, IRToolCallBlock)
    assert isinstance(tool_result, IRToolResultBlock)
    assert tool_result.call_id == "toolu_1"
    assert isinstance(interleaved_text, IRUserTextBlock)
    assert interleaved_text.text == "also, your cwd is already right"


def test_import_refuses_to_overwrite_without_force(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.jsonl"
    output = tmp_path / "core.jsonl"
    _write_legacy(legacy, [_session_record(tmp_path)])
    output.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError):
        parse_legacy_context.import_legacy_transcript(
            input_path=legacy,
            output_path=output,
        )

    parse_legacy_context.import_legacy_transcript(
        input_path=legacy,
        output_path=output,
        force=True,
    )

    assert Rehydrator(output).run().session_id == "legacy_session"


def test_import_can_limit_turns(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.jsonl"
    output = tmp_path / "core.jsonl"
    _write_legacy(
        legacy,
        [
            _session_record(tmp_path),
            _turn_start(1),
            _event(1, 0, "user_message", {"text": "one", "source": "human"}),
            _turn_end(1),
            _turn_start(2),
            _event(2, 0, "user_message", {"text": "two", "source": "human"}),
            _turn_end(2),
        ],
    )

    report = parse_legacy_context.import_legacy_transcript(
        input_path=legacy,
        output_path=output,
        limit_turns=1,
    )
    result = Rehydrator(output).run()

    assert report.truncated_after_turn == 1
    assert report.skipped_ir == {"turn_start": 1, "event": 1, "turn_end": 1}
    assert [
        block.text for block in result.blocks if isinstance(block, IRUserTextBlock)
    ] == ["one"]

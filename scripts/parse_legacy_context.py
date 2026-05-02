"""Import a legacy Spellbook transcript into the current core transcript IR.

This is intentionally a lossy boundary importer. It preserves conversation,
thinking, tool calls/results, and turn boundaries, but skips legacy
Spellbook-internal system markers such as old block artifacts, request anchors,
footer events, and TTL records.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, TextIO

from scripts.legacy_tool_result_repair import repair_tool_result_order_records
from scripts.repair_meta_system_prompt import build_meta_system_prompt, is_meta_cwd
from scripts.repair_session_skill_catalog import discover_skill_catalog
from spellbook.config import SpellbookConfig
from spellbook.ir_types import (
    IRAssistantTextBlock,
    IRBlock,
    IRBlockRecord,
    IRRecord,
    IRSessionRecord,
    IRThinkingBlock,
    IRToolCallBlock,
    IRToolResultBlock,
    IRToolTextBlock,
    IRTurnEndRecord,
    IRTurnStartRecord,
    IRUserTextBlock,
)
from spellbook.rehydrator import Rehydrator
from spellbook.tools.registry import ToolRegistry

CONVERTED_EVENT_TYPES = {
    "assistant_text",
    "thinking",
    "assistant_thinking",
    "tool_call",
    "tool_result",
    "user_message",
}

UserTextOrigin = Literal["human", "conduit", "system", "memory"]


@dataclass
class ImportReport:
    input_path: Path
    output_path: Path
    legacy_session_id: str | None = None
    core_session_id: str | None = None
    records_read: int = 0
    records_written: int = 0
    blocks_written: int = 0
    turns_written: int = 0
    converted_event_types: Counter[str] = field(default_factory=Counter)
    skipped_ir: Counter[str] = field(default_factory=Counter)
    skipped_event_types: Counter[str] = field(default_factory=Counter)
    warnings: list[str] = field(default_factory=list)
    truncated_after_turn: int | None = None
    tool_result_order_repairs: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_path": str(self.input_path),
            "output_path": str(self.output_path),
            "legacy_session_id": self.legacy_session_id,
            "core_session_id": self.core_session_id,
            "records_read": self.records_read,
            "records_written": self.records_written,
            "blocks_written": self.blocks_written,
            "turns_written": self.turns_written,
            "converted_event_types": dict(self.converted_event_types),
            "skipped_ir": dict(self.skipped_ir),
            "skipped_event_types": dict(self.skipped_event_types),
            "warnings": self.warnings,
            "truncated_after_turn": self.truncated_after_turn,
            "tool_result_order_repairs": self.tool_result_order_repairs,
        }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a legacy Spellbook transcript JSONL file into core transcript IR."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the legacy transcript.jsonl.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to write the imported core transcript.jsonl.",
    )
    parser.add_argument(
        "--session-id",
        help="Override the imported core session id. Defaults to the legacy session id.",
    )
    parser.add_argument(
        "--limit-turns",
        type=int,
        help="Import only records with turn <= N. Useful for smoke tests on huge transcripts.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output transcript if it already exists.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validating the written transcript with the core Rehydrator.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        help="Optionally write the import report as JSON.",
    )
    return parser.parse_args(argv)


def import_legacy_transcript(
    *,
    input_path: Path,
    output_path: Path,
    session_id: str | None = None,
    limit_turns: int | None = None,
    force: bool = False,
    validate: bool = True,
) -> ImportReport:
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    report = ImportReport(
        input_path=input_path,
        output_path=output_path,
        truncated_after_turn=limit_turns,
    )

    if not input_path.exists():
        raise FileNotFoundError(f"Legacy transcript not found: {input_path}")
    if input_path == output_path:
        raise ValueError("Input and output transcript paths must be different.")
    if output_path.exists() and not force:
        raise FileExistsError(f"Output transcript already exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    current_turn_id: str | None = None
    fallback_seq_by_turn: dict[int, int] = {}
    saw_session = False
    records: list[IRRecord] = []

    with input_path.open("r", encoding="utf-8") as src:
        for raw_line in src:
            line = raw_line.strip()
            if not line:
                continue
            report.records_read += 1
            legacy_record = json.loads(line)
            ir = legacy_record.get("ir")

            if _beyond_turn_limit(legacy_record, limit_turns):
                report.skipped_ir[str(ir)] += 1
                continue

            match ir:
                case "session":
                    record = _convert_session_record(legacy_record, session_id)
                    report.legacy_session_id = _legacy_session_id(legacy_record)
                    report.core_session_id = record.session_id
                    records.append(record)
                    report.records_written += 1
                    saw_session = True
                case "turn_start":
                    if not saw_session:
                        raise ValueError(
                            "Legacy transcript has turn_start before session."
                        )
                    record = _convert_turn_start_record(
                        legacy_record, _require_session_id(report)
                    )
                    current_turn_id = record.turn_id
                    records.append(record)
                    report.records_written += 1
                    report.turns_written += 1
                case "turn_end":
                    if not saw_session:
                        raise ValueError(
                            "Legacy transcript has turn_end before session."
                        )
                    record = _convert_turn_end_record(
                        legacy_record, _require_session_id(report)
                    )
                    current_turn_id = None
                    records.append(record)
                    report.records_written += 1
                case "event":
                    if not saw_session:
                        raise ValueError("Legacy transcript has event before session.")
                    converted = _convert_event_record(
                        legacy_record,
                        session_id=_require_session_id(report),
                        current_turn_id=current_turn_id,
                        fallback_seq_by_turn=fallback_seq_by_turn,
                    )
                    if converted is None:
                        event_type = _legacy_event_type(legacy_record)
                        report.skipped_event_types[event_type] += 1
                        continue
                    records.append(converted)
                    report.records_written += 1
                    report.blocks_written += 1
                    report.converted_event_types[_legacy_event_type(legacy_record)] += 1
                case _:
                    report.skipped_ir[str(ir)] += 1

    repaired_records, repair_report = repair_tool_result_order_records(records)
    report.tool_result_order_repairs = repair_report.records_moved
    with output_path.open("w", encoding="utf-8") as dst:
        for record in repaired_records:
            _write_record(dst, record)

    if validate:
        Rehydrator(output_path).run()
    return report


def _convert_session_record(
    legacy_record: dict[str, Any], override_session_id: str | None
) -> IRSessionRecord:
    data = _dict_value(legacy_record.get("data"))
    spellbook_meta = _dict_value(_dict_value(data.get("extra")).get("spellbook"))
    session_id = override_session_id or _legacy_session_id(legacy_record)
    if session_id is None:
        raise ValueError("Legacy session record is missing a session id.")

    config = SpellbookConfig(
        provider=_string_value(spellbook_meta.get("provider"), "anthropic"),
        model=_string_value(data.get("model"), "claude-opus-4-6"),
        effort=_string_value(spellbook_meta.get("effort"), "high"),
        cwd=Path(_string_value(data.get("cwd"), str(Path.cwd()))),
    )
    if is_meta_cwd(config.cwd):
        system_prompt, _ = build_meta_system_prompt(config)
        config = config.model_copy(update={"system_prompt": system_prompt})
    registry = ToolRegistry.build(config.tool_categories, surface=config.session_type)
    return IRSessionRecord(
        session_id=session_id,
        time=_parse_time(data.get("start_time") or legacy_record.get("time")),
        config=config,
        tools=registry.records,
        skill_catalog=discover_skill_catalog(config),
    )


def _convert_turn_start_record(
    legacy_record: dict[str, Any], session_id: str
) -> IRTurnStartRecord:
    return IRTurnStartRecord(
        session_id=session_id,
        time=_parse_time(legacy_record.get("time")),
        turn=_int_value(legacy_record.get("turn")),
        turn_id=_string_value(legacy_record.get("turn_id"), ""),
    )


def _convert_turn_end_record(
    legacy_record: dict[str, Any], session_id: str
) -> IRTurnEndRecord:
    return IRTurnEndRecord(
        session_id=session_id,
        time=_parse_time(legacy_record.get("time")),
        turn=_int_value(legacy_record.get("turn")),
        turn_id=_string_value(legacy_record.get("turn_id"), ""),
        stop_reason="end_turn",
    )


def _convert_event_record(
    legacy_record: dict[str, Any],
    *,
    session_id: str,
    current_turn_id: str | None,
    fallback_seq_by_turn: dict[int, int],
) -> IRBlockRecord | None:
    event = _dict_value(legacy_record.get("event"))
    event_type = _string_value(event.get("type"), "")
    if event_type not in CONVERTED_EVENT_TYPES:
        return None

    block = _convert_event_block(
        event,
        turn_id=_string_value(legacy_record.get("turn_id"), current_turn_id or ""),
    )
    if block is None:
        return None

    turn = _int_value(legacy_record.get("turn"))
    seq = _event_seq(legacy_record, event, fallback_seq_by_turn)
    return IRBlockRecord(session_id=session_id, turn=turn, seq=seq, event=block)


def _convert_event_block(event: dict[str, Any], *, turn_id: str) -> IRBlock | None:
    event_type = _string_value(event.get("type"), "")
    data = _dict_value(event.get("data"))
    time = _parse_time(event.get("time"))

    match event_type:
        case "user_message":
            return IRUserTextBlock(
                time=time,
                turn_id=turn_id,
                origin=_user_origin(data),
                text=_text_value(data.get("text")),
            )
        case "assistant_text":
            return IRAssistantTextBlock(
                time=time,
                turn_id=turn_id,
                text=_text_value(data.get("text")),
            )
        case "thinking" | "assistant_thinking":
            return IRThinkingBlock(
                time=time,
                turn_id=turn_id,
                text=_text_value(data.get("text")),
                signature=_text_value(data.get("signature")),
            )
        case "tool_call":
            tool_input = data.get("input")
            return IRToolCallBlock(
                time=time,
                turn_id=turn_id,
                call_id=_string_value(data.get("call_id"), "legacy_missing_call_id"),
                tool=_string_value(data.get("tool"), "Unknown"),
                input=tool_input
                if isinstance(tool_input, dict)
                else {"value": tool_input},
            )
        case "tool_result":
            output = data.get("output")
            if output is None and data.get("content_ref") is not None:
                output = data.get("content_ref")
            display = data.get("display")
            normalized_display = display if isinstance(display, dict) else {}
            if data.get("summary") is not None:
                normalized_display = {
                    **normalized_display,
                    "summary": _text_value(data.get("summary")),
                }
            if data.get("content_ref") is not None:
                normalized_display = {
                    **normalized_display,
                    "content_ref": data.get("content_ref"),
                }
            return IRToolResultBlock(
                time=time,
                turn_id=turn_id,
                call_id=_string_value(data.get("call_id"), "legacy_missing_call_id"),
                tool=_string_value(data.get("tool"), "Unknown"),
                content=[IRToolTextBlock(text=_text_value(output))],
                display=normalized_display,
                is_error=bool(data.get("is_error", False)),
            )
    return None


def _write_record(dst: TextIO, record: IRRecord) -> None:
    dst.write(record.model_dump_json() + "\n")


def _beyond_turn_limit(legacy_record: dict[str, Any], limit_turns: int | None) -> bool:
    if limit_turns is None:
        return False
    turn = legacy_record.get("turn")
    return isinstance(turn, int) and turn > limit_turns


def _legacy_session_id(legacy_record: dict[str, Any]) -> str | None:
    data = _dict_value(legacy_record.get("data"))
    value = legacy_record.get("session") or data.get("id")
    return value if isinstance(value, str) else None


def _require_session_id(report: ImportReport) -> str:
    if report.core_session_id is None:
        raise ValueError("No core session id has been established yet.")
    return report.core_session_id


def _legacy_event_type(legacy_record: dict[str, Any]) -> str:
    return _string_value(
        _dict_value(legacy_record.get("event")).get("type"), "<missing>"
    )


def _event_seq(
    legacy_record: dict[str, Any],
    event: dict[str, Any],
    fallback_seq_by_turn: dict[int, int],
) -> int:
    for value in (legacy_record.get("seq"), event.get("seq")):
        if isinstance(value, int):
            return value
    turn = _int_value(legacy_record.get("turn"))
    seq = fallback_seq_by_turn.get(turn, 0)
    fallback_seq_by_turn[turn] = seq + 1
    return seq


def _user_origin(data: dict[str, Any]) -> UserTextOrigin:
    legacy_origin = data.get("origin")
    source = data.get("source")
    if legacy_origin == "system" or source == "system":
        return "system"
    if legacy_origin == "conduit":
        return "conduit"
    if source in ("human", "tui", None):
        return "human"
    return "conduit"


def _parse_time(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.now().astimezone()


def _dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string_value(value: Any, default: str) -> str:
    return value if isinstance(value, str) else default


def _int_value(value: Any) -> int:
    return value if isinstance(value, int) else 0


def _text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _print_report(report: ImportReport) -> None:
    data = report.to_dict()
    print(f"Imported legacy transcript: {data['input_path']}")
    print(f"Wrote core transcript: {data['output_path']}")
    print(f"Session: {data['legacy_session_id']} -> {data['core_session_id']}")
    print(
        "Records: "
        f"{data['records_written']} written / {data['records_read']} read; "
        f"{data['blocks_written']} blocks; {data['turns_written']} turns"
    )
    if report.converted_event_types:
        print(f"Converted events: {dict(report.converted_event_types)}")
    if report.skipped_event_types:
        print(f"Skipped events: {dict(report.skipped_event_types)}")
    if report.skipped_ir:
        print(f"Skipped records: {dict(report.skipped_ir)}")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    report = import_legacy_transcript(
        input_path=args.input,
        output_path=args.output,
        session_id=args.session_id,
        limit_turns=args.limit_turns,
        force=args.force,
        validate=not args.no_validate,
    )
    if args.report_json is not None:
        args.report_json.expanduser().resolve().parent.mkdir(
            parents=True, exist_ok=True
        )
        args.report_json.write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    _print_report(report)


if __name__ == "__main__":
    main()

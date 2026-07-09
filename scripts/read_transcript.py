"""Print the last turns of a Spellbook JSONL transcript as Markdown."""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ToolCall:
    name: str
    input: Any


@dataclass
class Turn:
    number: int | str
    user: list[str] = field(default_factory=list)
    thinking: list[str] = field(default_factory=list)
    assistant: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)

    def has_content(self) -> bool:
        return bool(self.user or self.thinking or self.assistant or self.tool_calls)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render the last turns of a Spellbook JSONL transcript."
    )
    parser.add_argument("transcript_path", type=Path)
    parser.add_argument("--last", type=int, required=True, help="number of turns to show")
    args = parser.parse_args()

    if args.last < 1:
        parser.error("--last must be at least 1")

    try:
        turns = read_transcript(args.transcript_path)
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    selected = [turn for turn in turns.values() if turn.has_content()][-args.last :]
    print(render_turns(selected), end="")
    return 0


def read_transcript(path: Path) -> "OrderedDict[int | str, Turn]":
    turns: OrderedDict[int | str, Turn] = OrderedDict()

    with path.open("r", encoding="utf-8") as transcript:
        for line_number, line in enumerate(transcript, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc

            ingest_record(turns, record)

    return turns


def ingest_record(turns: "OrderedDict[int | str, Turn]", record: dict[str, Any]) -> None:
    event = record.get("event") if isinstance(record.get("event"), dict) else None
    block = event or record
    block_type = block.get("type") or block.get("kind")
    if block_type == "message" and block.get("role") in {"user", "assistant"}:
        block_type = block["role"]

    if block_type not in {
        "user_text",
        "user_message",
        "user",
        "thinking",
        "thinking_summary",
        "assistant_text",
        "assistant_message",
        "assistant",
        "tool_call",
    }:
        return

    turn_number = record.get("turn")
    if turn_number is None:
        turn_number = record.get("turn_number") or record.get("turn_id") or "?"

    turn = turns.setdefault(turn_number, Turn(number=turn_number))

    if block_type in {"user_text", "user_message", "user"}:
        if block.get("origin") != "system":
            append_text(turn.user, block)
    elif block_type in {"thinking", "thinking_summary"}:
        append_text(turn.thinking, block)
    elif block_type in {"assistant_text", "assistant_message", "assistant"}:
        append_text(turn.assistant, block)
    elif block_type == "tool_call":
        name = str(block.get("tool") or block.get("name") or block.get("function") or "?")
        turn.tool_calls.append(ToolCall(name=name, input=tool_input(block)))


def append_text(parts: list[str], block: dict[str, Any]) -> None:
    text = block.get("text")
    if text is None:
        text = block.get("content") or block.get("summary")
    if isinstance(text, str) and text.strip():
        parts.append(text.rstrip())


def tool_input(block: dict[str, Any]) -> Any:
    if "input" in block:
        return block["input"]
    if "arguments" in block:
        return block["arguments"]
    function = block.get("function")
    if isinstance(function, dict) and "arguments" in function:
        return function["arguments"]
    return {}


def render_turns(turns: list[Turn]) -> str:
    lines: list[str] = []

    for index, turn in enumerate(turns):
        if index:
            lines.append("")
        lines.append(f"## Turn {turn.number}")
        lines.append("")
        lines.append("### User")
        lines.append("")
        lines.append(join_parts(turn.user))

        if turn.thinking:
            lines.append("")
            lines.append("### Thinking")
            lines.append("")
            lines.append(join_parts(turn.thinking))

        lines.append("")
        lines.append("### Assistant")
        lines.append("")
        lines.append(join_parts(turn.assistant))

        lines.append("")
        lines.append("### Tool Calls")
        lines.append("")
        if not turn.tool_calls:
            lines.append("_None_")
        for call in turn.tool_calls:
            lines.append(f"- `{call.name}`")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(call.input, indent=2, ensure_ascii=False))
            lines.append("```")
            lines.append("")
        while lines and lines[-1] == "":
            lines.pop()

    return "\n".join(lines).rstrip() + "\n"


def join_parts(parts: list[str]) -> str:
    if not parts:
        return "_None_"
    return "\n\n".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())

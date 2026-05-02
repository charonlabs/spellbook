from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from spellbook.config import HomunculusConfig
from spellbook.ir_types import (
    IRBlock,
    IRExecution,
    IRToolResultBlock,
    IRToolResultTTLRecord,
    IRToolTextBlock,
    ToolResultTTLSource,
    ToolResultTTLTrigger,
)
from spellbook.recorder import Recorder

TTL_TRIGGER_END_TURN: ToolResultTTLTrigger = "end_turn"
TTL_TRIGGER_SEQ: ToolResultTTLTrigger = "seq"

AUTO_TTL_SKIP_TOOLS = {
    "Pin",
    "Forget",
    "Skill",
}


@dataclass
class ToolResultTTL:
    call_id: str
    replace_content: str
    remaining: int
    trigger: ToolResultTTLTrigger
    delivered_turn: int
    output_ref: str | None = None

    @classmethod
    def from_record(
        cls, record: IRToolResultTTLRecord, *, last_completed_turn: int
    ) -> ToolResultTTL:
        remaining = record.ttl
        if record.trigger == TTL_TRIGGER_END_TURN:
            elapsed = 0
            if last_completed_turn >= record.delivered_turn:
                elapsed = last_completed_turn - record.delivered_turn + 1
            remaining = max(0, remaining - elapsed)

        return cls(
            call_id=record.call_id,
            replace_content=record.replace_content,
            remaining=remaining,
            trigger=record.trigger,
            delivered_turn=record.delivered_turn,
            output_ref=record.output_ref,
        )


class ToolResultTTLRegistry:
    """Render-time compaction for large historical tool results.

    The transcript keeps the full tool result. This registry only changes the
    provider-facing projection after a registered TTL has expired.
    """

    def __init__(self, *, config: HomunculusConfig, recorder: Recorder) -> None:
        self._config = config
        self._recorder = recorder
        self._ttls: dict[str, ToolResultTTL] = {}

    @property
    def ttls(self) -> dict[str, ToolResultTTL]:
        return self._ttls

    def rehydrate(
        self,
        records: Sequence[IRToolResultTTLRecord],
        *,
        last_completed_turn: int,
    ) -> None:
        self._ttls = {
            record.call_id: ToolResultTTL.from_record(
                record,
                last_completed_turn=last_completed_turn,
            )
            for record in records
        }

    def observe_execution(self, execution: IRExecution) -> None:
        if not self._config.tool_result_ttl_enabled:
            return
        for block in execution.blocks:
            self._maybe_auto_register(block)

    def register(
        self,
        *,
        call_id: str,
        replace_content: str,
        ttl: int | None = None,
        trigger: ToolResultTTLTrigger = TTL_TRIGGER_END_TURN,
        source: ToolResultTTLSource = "auto",
        output_ref: str | None = None,
    ) -> ToolResultTTL:
        ttl_value = self._config.tool_result_ttl_turns if ttl is None else ttl
        record = self._recorder.write_tool_result_ttl(
            call_id=call_id,
            replace_content=replace_content,
            ttl=ttl_value,
            trigger=trigger,
            delivered_turn=self._recorder.current_turn_idx,
            source=source,
            output_ref=output_ref,
        )
        state = ToolResultTTL.from_record(
            record,
            last_completed_turn=max(0, record.delivered_turn - 1),
        )
        self._ttls[call_id] = state
        return state

    def tick(self, trigger: ToolResultTTLTrigger) -> bool:
        """Tick matching TTLs. Returns True if any tool result became collapsed."""
        any_newly_expired = False
        for state in self._ttls.values():
            if state.trigger != trigger or state.remaining <= 0:
                continue
            state.remaining -= 1
            if state.remaining == 0:
                any_newly_expired = True
        return any_newly_expired

    def collapse_blocks(self, blocks: Sequence[IRBlock]) -> list[IRBlock]:
        if not self._ttls:
            return list(blocks)
        return [self._collapse_block(block) for block in blocks]

    def _collapse_block(self, block: IRBlock) -> IRBlock:
        if not isinstance(block, IRToolResultBlock):
            return block
        state = self._ttls.get(block.call_id)
        if state is None or state.remaining > 0:
            return block
        return block.model_copy(
            update={"content": [IRToolTextBlock(text=state.replace_content)]}
        )

    def _maybe_auto_register(self, block: IRToolResultBlock) -> None:
        if block.is_error or block.call_id in self._ttls:
            return
        if block.tool in AUTO_TTL_SKIP_TOOLS:
            return

        output = tool_result_text_content(block)
        if output is None:
            return
        if len(output) < self._config.tool_result_ttl_char_threshold:
            return

        output_ref = self._save_output(block.call_id, output)
        replace_content = build_tool_result_ttl_replacement(
            tool=block.tool,
            output=output,
            display=block.display,
            output_ref=output_ref,
        )
        self.register(
            call_id=block.call_id,
            replace_content=replace_content,
            output_ref=output_ref,
        )

    def _save_output(self, call_id: str, output: str) -> str:
        output_dir = self._recorder.transcript_path.parent / "tool-outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{_safe_filename(call_id)}.txt"
        output_path = output_dir / filename
        output_path.write_text(output, encoding="utf-8")
        return str(output_path.relative_to(self._recorder.transcript_path.parent))


def _line_count(text: str) -> int:
    if text == "":
        return 0
    return len(text.splitlines()) or 1


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _clip(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def tool_result_text_content(block: IRToolResultBlock) -> str | None:
    parts = [
        content.text
        for content in block.content
        if isinstance(content, IRToolTextBlock)
    ]
    if not parts:
        return None
    return "\n".join(parts)


def build_tool_result_ttl_replacement(
    *,
    tool: str,
    output: str,
    display: dict,
    output_ref: str,
) -> str:
    line_count = _line_count(output)
    char_count = len(output)
    kind = display.get("kind")
    match kind:
        case "read":
            path = display.get("path", "(unknown path)")
            start_line = display.get("start_line")
            end_line = display.get("end_line")
            total_lines = display.get("total_lines")
            if isinstance(start_line, int) and isinstance(end_line, int):
                line_part = f"lines {start_line}-{end_line}"
                if isinstance(total_lines, int):
                    line_part += f" of {total_lines}"
            else:
                line_part = f"{line_count} lines"
            return f"[Read: {path} - {line_part}. Full output saved to {output_ref}]"
        case "command":
            command = _clip(str(display.get("command", "")), 160)
            exit_code = display.get("exit_code")
            exit_part = f"exit {exit_code}" if exit_code is not None else "ran"
            return (
                f"[Bash: `{command}` - {exit_part}, {line_count} lines. "
                f"Full output saved to {output_ref}]"
            )
        case "web_search":
            query = _clip(str(display.get("query", "")), 120)
            results = display.get("num_results")
            result_part = (
                f"{results} results" if isinstance(results, int) else "results"
            )
            return (
                f'[WebSearch: "{query}" - {result_part}. '
                f"Full output saved to {output_ref}]"
            )
        case "web_read":
            title = _clip(str(display.get("title") or display.get("url") or ""), 120)
            return (
                f"[WebRead: {title} - {line_count} lines. "
                f"Full output saved to {output_ref}]"
            )
        case "web_answer":
            query = _clip(str(display.get("query", "")), 120)
            citations = display.get("citation_count")
            citation_part = (
                f"{citations} citations"
                if isinstance(citations, int)
                else f"{line_count} lines"
            )
            return (
                f'[WebAnswer: "{query}" - {citation_part}. '
                f"Full output saved to {output_ref}]"
            )
        case "reflect":
            block_idx = display.get("target_block")
            if block_idx is not None:
                return f"[Reflected on block {block_idx}]"
            blocks = display.get("block_count", "?")
            token_summary = display.get("token_summary", "unknown count")
            return f"[Reflected: {blocks} blocks, {token_summary}]"
        case _:
            return (
                f"[{tool}: {line_count} lines, {char_count} chars. "
                f"Full output saved to {output_ref}]"
            )

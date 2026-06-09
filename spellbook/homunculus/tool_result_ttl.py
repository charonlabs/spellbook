from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Literal, Sequence

from spellbook.config import HomunculusConfig
from spellbook.ir_types import (
    IRBlock,
    IRExecution,
    IRRuntimeConfigRecord,
    IRToolResultBlock,
    IRToolResultTTLRecord,
    IRToolTextBlock,
    RuntimeConfigValue,
    ToolResultTTLSource,
    ToolResultTTLTrigger,
)
from spellbook.recorder import Recorder

TTL_TRIGGER_END_TURN: ToolResultTTLTrigger = "end_turn"
TTL_TRIGGER_SEQ: ToolResultTTLTrigger = "seq"

AUTO_TTL_SKIP_TOOLS = {
    "Pin",
    "Forget",
    "ForgetToolResult",
    "Skill",
}

ToolResultTTLStatusKind = Literal[
    "pending",
    "collapsed",
    "large_untracked",
    "small_untracked",
    "ignored",
    "error",
    "non_text",
]


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


@dataclass(frozen=True)
class ToolResultTTLSettings:
    enabled: bool
    ttl_turns: int
    char_threshold: int

    @classmethod
    def from_config(cls, config: HomunculusConfig) -> "ToolResultTTLSettings":
        return cls(
            enabled=config.tool_result_ttl_enabled,
            ttl_turns=config.tool_result_ttl_turns,
            char_threshold=config.tool_result_ttl_char_threshold,
        )

    def as_record_dict(self) -> dict[str, RuntimeConfigValue]:
        return {
            "enabled": self.enabled,
            "ttl_turns": self.ttl_turns,
            "char_threshold": self.char_threshold,
        }


@dataclass(frozen=True)
class ToolResultTTLStatus:
    call_id: str
    tool: str
    label: str | None
    chars: int | None
    lines: int | None
    kind: ToolResultTTLStatusKind
    status: str
    output_ref: str | None = None
    delivered_turn: int | None = None
    age_turns: int | None = None
    remaining: int | None = None
    trigger: ToolResultTTLTrigger | None = None

    @property
    def show_by_default(self) -> bool:
        return self.kind in {"pending", "large_untracked"}


class ToolResultTTLRegistry:
    """Render-time compaction for large historical tool results.

    The transcript keeps the full tool result. This registry only changes the
    provider-facing projection after a registered TTL has expired.
    """

    def __init__(self, *, config: HomunculusConfig, recorder: Recorder) -> None:
        self._settings = ToolResultTTLSettings.from_config(config)
        self._recorder = recorder
        self._ttls: dict[str, ToolResultTTL] = {}

    @property
    def ttls(self) -> dict[str, ToolResultTTL]:
        return self._ttls

    @property
    def settings(self) -> ToolResultTTLSettings:
        return self._settings

    def rehydrate(
        self,
        records: Sequence[IRToolResultTTLRecord],
        *,
        last_completed_turn: int,
        config_records: Sequence[IRRuntimeConfigRecord] = (),
    ) -> None:
        self._ttls = {
            record.call_id: ToolResultTTL.from_record(
                record,
                last_completed_turn=last_completed_turn,
            )
            for record in records
        }
        for record in config_records:
            if record.namespace == "tool_result_ttl":
                self.apply_config(record.effective)

    def configure(
        self,
        *,
        enabled: bool | None = None,
        ttl_turns: int | None = None,
        char_threshold: int | None = None,
    ) -> tuple[
        ToolResultTTLSettings, ToolResultTTLSettings, dict[str, RuntimeConfigValue]
    ]:
        updates: dict[str, RuntimeConfigValue] = {}
        if enabled is not None:
            updates["enabled"] = enabled
        if ttl_turns is not None:
            if ttl_turns < 0:
                raise ValueError("ttl_turns must be >= 0.")
            updates["ttl_turns"] = ttl_turns
        if char_threshold is not None:
            if char_threshold < 0:
                raise ValueError("ttl_char_threshold must be >= 0.")
            updates["char_threshold"] = char_threshold
        old = self._settings
        self.apply_config(updates)
        return old, self._settings, updates

    def apply_config(self, updates: dict[str, RuntimeConfigValue]) -> None:
        allowed = {"enabled", "ttl_turns", "char_threshold"}
        unknown = sorted(set(updates) - allowed)
        if unknown:
            raise ValueError(
                f"Unknown tool_result_ttl config key(s): {', '.join(unknown)}"
            )
        next_settings = self._settings
        if "enabled" in updates:
            value = updates["enabled"]
            if not isinstance(value, bool):
                raise ValueError("enabled must be a boolean.")
            next_settings = replace(next_settings, enabled=value)
        if "ttl_turns" in updates:
            value = updates["ttl_turns"]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError("ttl_turns must be a non-negative integer.")
            next_settings = replace(next_settings, ttl_turns=value)
        if "char_threshold" in updates:
            value = updates["char_threshold"]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError("char_threshold must be a non-negative integer.")
            next_settings = replace(next_settings, char_threshold=value)
        self._settings = next_settings

    def observe_execution(self, execution: IRExecution) -> None:
        if not self._settings.enabled:
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
        ttl_value = self._settings.ttl_turns if ttl is None else ttl
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

    def forget(self, block: IRToolResultBlock) -> ToolResultTTL:
        """Collapse a tool result immediately and persist the manual TTL decision."""
        existing = self._ttls.get(block.call_id)
        if existing is not None and existing.remaining <= 0:
            return existing

        output = tool_result_text_content(block)
        if output is None:
            raise ValueError(
                f"Tool result `{block.call_id}` has no textual output to forget."
            )

        output_ref = existing.output_ref if existing is not None else None
        if output_ref is None:
            output_ref = self._save_output(block.call_id, output)
        replace_content = (
            existing.replace_content
            if existing is not None
            else build_tool_result_ttl_replacement(
                tool=block.tool,
                output=output,
                display=block.display,
                output_ref=output_ref,
            )
        )
        return self.register(
            call_id=block.call_id,
            replace_content=replace_content,
            ttl=0,
            source="manual",
            output_ref=output_ref,
        )

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

    def status_for_block(
        self, block: IRToolResultBlock, *, current_turn: int
    ) -> ToolResultTTLStatus:
        output = tool_result_text_content(block)
        chars = len(output) if output is not None else None
        lines = _line_count(output) if output is not None else None
        label = tool_result_label(block)
        state = self._ttls.get(block.call_id)
        if state is not None:
            age_turns = max(0, current_turn - state.delivered_turn)
            if state.remaining <= 0:
                return ToolResultTTLStatus(
                    call_id=block.call_id,
                    tool=block.tool,
                    label=label,
                    chars=chars,
                    lines=lines,
                    kind="collapsed",
                    status="collapsed",
                    output_ref=state.output_ref,
                    delivered_turn=state.delivered_turn,
                    age_turns=age_turns,
                    remaining=state.remaining,
                    trigger=state.trigger,
                )
            unit = "turn" if state.trigger == TTL_TRIGGER_END_TURN else "round"
            plural = "" if state.remaining == 1 else "s"
            return ToolResultTTLStatus(
                call_id=block.call_id,
                tool=block.tool,
                label=label,
                chars=chars,
                lines=lines,
                kind="pending",
                status=f"pending TTL, {state.remaining} {unit}{plural} remaining",
                output_ref=state.output_ref,
                delivered_turn=state.delivered_turn,
                age_turns=age_turns,
                remaining=state.remaining,
                trigger=state.trigger,
            )

        if block.is_error:
            return ToolResultTTLStatus(
                call_id=block.call_id,
                tool=block.tool,
                label=label,
                chars=chars,
                lines=lines,
                kind="error",
                status="untracked, error result",
            )
        if block.tool in AUTO_TTL_SKIP_TOOLS:
            return ToolResultTTLStatus(
                call_id=block.call_id,
                tool=block.tool,
                label=label,
                chars=chars,
                lines=lines,
                kind="ignored",
                status="ignored, tool is excluded from auto-TTL",
            )
        if output is None:
            return ToolResultTTLStatus(
                call_id=block.call_id,
                tool=block.tool,
                label=label,
                chars=chars,
                lines=lines,
                kind="non_text",
                status="untracked, no textual output",
            )
        if len(output) < self._settings.char_threshold:
            return ToolResultTTLStatus(
                call_id=block.call_id,
                tool=block.tool,
                label=label,
                chars=chars,
                lines=lines,
                kind="small_untracked",
                status="untracked, below TTL threshold",
            )
        return ToolResultTTLStatus(
            call_id=block.call_id,
            tool=block.tool,
            label=label,
            chars=chars,
            lines=lines,
            kind="large_untracked",
            status="untracked, above TTL threshold",
        )

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
        if len(output) < self._settings.char_threshold:
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


def tool_result_label(block: IRToolResultBlock) -> str | None:
    display = block.display
    kind = display.get("kind")
    match kind:
        case "read":
            value = display.get("path")
        case "command":
            value = display.get("command")
        case "web_search" | "web_answer":
            value = display.get("query")
        case "web_read":
            value = display.get("title") or display.get("url")
        case "reflect":
            target = display.get("target_block")
            value = f"block {target}" if target is not None else "context"
        case _:
            value = None
    if value is None:
        return None
    return _clip(str(value), 120)


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

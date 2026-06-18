from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Sequence

from spellbook.config import HomunculusConfig
from spellbook.image_blobs import resolve_blob_path
from spellbook.ir_types import (
    IMAGE_MEDIA_TYPES,
    IRBlock,
    IRExecution,
    IRImageBase64Source,
    IRImageBlobSource,
    IRImageBlock,
    IRImageURLSource,
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
    images: int = 0
    image_bytes: int | None = None
    image_refs: tuple[str, ...] = ()
    output_ref: str | None = None
    delivered_turn: int | None = None
    age_turns: int | None = None
    remaining: int | None = None
    trigger: ToolResultTTLTrigger | None = None

    @property
    def show_by_default(self) -> bool:
        return self.kind in {"pending", "large_untracked"}


@dataclass(frozen=True)
class ToolResultImageRef:
    ref: str
    source_kind: Literal["blob", "url", "base64"]
    media_type: str | None = None
    bytes: int | None = None


@dataclass(frozen=True)
class ToolResultTTLContent:
    text: str | None
    images: tuple[ToolResultImageRef, ...]

    @property
    def chars(self) -> int | None:
        return len(self.text) if self.text is not None else None

    @property
    def lines(self) -> int | None:
        return _line_count(self.text) if self.text is not None else None

    @property
    def image_count(self) -> int:
        return len(self.images)

    @property
    def image_bytes(self) -> int | None:
        sizes = [image.bytes for image in self.images]
        if not sizes or any(size is None for size in sizes):
            return None
        return sum(size for size in sizes if size is not None)

    @property
    def has_ttl_content(self) -> bool:
        return self.text is not None or bool(self.images)

    def should_auto_register(self, char_threshold: int) -> bool:
        if self.images:
            return True
        return self.text is not None and len(self.text) >= char_threshold


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

        content = tool_result_ttl_content(
            block,
            transcript_path=self._recorder.transcript_path,
        )
        if not content.has_ttl_content:
            raise ValueError(
                f"Tool result `{block.call_id}` has no text or image output to forget."
            )

        output_ref = existing.output_ref if existing is not None else None
        if output_ref is None:
            output_ref = self._save_output(
                block.call_id,
                build_tool_result_ttl_saved_output(block, content),
            )
        replace_content = (
            existing.replace_content
            if existing is not None
            else build_tool_result_ttl_replacement(
                tool=block.tool,
                display=block.display,
                output_ref=output_ref,
                content=content,
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
        content = tool_result_ttl_content(
            block,
            transcript_path=self._recorder.transcript_path,
        )
        chars = content.chars
        lines = content.lines
        image_count = content.image_count
        image_bytes = content.image_bytes
        image_refs = tuple(image.ref for image in content.images)
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
                    images=image_count,
                    image_bytes=image_bytes,
                    image_refs=image_refs,
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
                images=image_count,
                image_bytes=image_bytes,
                image_refs=image_refs,
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
                images=image_count,
                image_bytes=image_bytes,
                image_refs=image_refs,
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
                images=image_count,
                image_bytes=image_bytes,
                image_refs=image_refs,
            )
        if not content.has_ttl_content:
            return ToolResultTTLStatus(
                call_id=block.call_id,
                tool=block.tool,
                label=label,
                chars=chars,
                lines=lines,
                kind="non_text",
                status="untracked, no text or image output",
                images=image_count,
                image_bytes=image_bytes,
                image_refs=image_refs,
            )
        if content.images:
            return ToolResultTTLStatus(
                call_id=block.call_id,
                tool=block.tool,
                label=label,
                chars=chars,
                lines=lines,
                kind="large_untracked",
                status="untracked, image result",
                images=image_count,
                image_bytes=image_bytes,
                image_refs=image_refs,
            )
        text = content.text or ""
        if len(text) < self._settings.char_threshold:
            return ToolResultTTLStatus(
                call_id=block.call_id,
                tool=block.tool,
                label=label,
                chars=chars,
                lines=lines,
                kind="small_untracked",
                status="untracked, below TTL threshold",
                images=image_count,
                image_bytes=image_bytes,
                image_refs=image_refs,
            )
        return ToolResultTTLStatus(
            call_id=block.call_id,
            tool=block.tool,
            label=label,
            chars=chars,
            lines=lines,
            kind="large_untracked",
            status="untracked, above TTL threshold",
            images=image_count,
            image_bytes=image_bytes,
            image_refs=image_refs,
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

        content = tool_result_ttl_content(
            block,
            transcript_path=self._recorder.transcript_path,
        )
        if not content.should_auto_register(self._settings.char_threshold):
            return

        output_ref = self._save_output(
            block.call_id,
            build_tool_result_ttl_saved_output(block, content),
        )
        replace_content = build_tool_result_ttl_replacement(
            tool=block.tool,
            display=block.display,
            output_ref=output_ref,
            content=content,
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


def tool_result_ttl_content(
    block: IRToolResultBlock,
    *,
    transcript_path: Path | None = None,
) -> ToolResultTTLContent:
    return ToolResultTTLContent(
        text=tool_result_text_content(block),
        images=tuple(
            _image_ref(content, transcript_path=transcript_path)
            for content in block.content
            if isinstance(content, IRImageBlock)
        ),
    )


def build_tool_result_ttl_manifest(
    block: IRToolResultBlock,
    content: ToolResultTTLContent | None = None,
) -> str:
    content = content or tool_result_ttl_content(block)
    parts = [
        f"Tool result: {block.tool}",
        f"call_id: {block.call_id}",
    ]
    label = tool_result_label(block)
    if label:
        parts.append(f"label: {label}")

    if block.display:
        parts.extend(["", "Display metadata:"])
        for key, value in sorted(block.display.items()):
            parts.append(f"- {key}: {value}")

    if content.text is not None:
        parts.extend(["", "Text output:", content.text])

    if content.images:
        parts.extend(["", "Images:"])
        for idx, image in enumerate(content.images, start=1):
            details = [f"ref={image.ref}", f"source={image.source_kind}"]
            if image.media_type is not None:
                details.append(f"media_type={image.media_type}")
            if image.bytes is not None:
                details.append(f"size={_format_bytes(image.bytes)}")
            parts.append(f"- image {idx}: " + ", ".join(details))

    return "\n".join(parts).rstrip() + "\n"


def build_tool_result_ttl_saved_output(
    block: IRToolResultBlock,
    content: ToolResultTTLContent | None = None,
) -> str:
    content = content or tool_result_ttl_content(block)
    if not content.images and content.text is not None:
        return content.text
    return build_tool_result_ttl_manifest(block, content)


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
            value = display.get("title") or display.get("path") or display.get("body")
    if value is None:
        return None
    return _clip(str(value), 120)


def _image_ref(
    block: IRImageBlock,
    *,
    transcript_path: Path | None,
) -> ToolResultImageRef:
    source = block.source
    media_type: str | None = None
    ref: str
    source_kind: Literal["blob", "url", "base64"]
    size: int | None = None

    if block.blob_path is not None:
        ref = block.blob_path
        source_kind = "blob"
        media_type = _image_media_type(block)
        size = _blob_size(block.blob_path, transcript_path)
    elif isinstance(source, IRImageURLSource):
        ref = source.url
        source_kind = "url"
    elif isinstance(source, IRImageBase64Source):
        ref = f"<base64:{source.media_type}>"
        source_kind = "base64"
        media_type = source.media_type
        size = _approx_base64_bytes(source.data)
    elif isinstance(source, IRImageBlobSource):
        ref = f"<blob:{block.blob_path}>"
        source_kind = "blob"
        media_type = _image_media_type(block)
    else:
        raise TypeError(f"Unsupported image source: {type(source)}")

    return ToolResultImageRef(
        ref=ref,
        source_kind=source_kind,
        media_type=media_type,
        bytes=size,
    )


def _image_media_type(block: IRImageBlock) -> str | None:
    source = block.source
    if isinstance(source, IRImageBase64Source):
        return source.media_type
    if block.blob_path is not None:
        suffix = Path(block.blob_path).suffix.lower()
        return IMAGE_MEDIA_TYPES.get(suffix)
    return None


def _blob_size(blob_path: str, transcript_path: Path | None) -> int | None:
    if transcript_path is None:
        return None
    try:
        return resolve_blob_path(blob_path, transcript_path).stat().st_size
    except OSError:
        return None


def _approx_base64_bytes(data: str) -> int | None:
    if not data:
        return 0
    padding = data.count("=")
    return max(0, (len(data) * 3 // 4) - padding)


def build_tool_result_ttl_replacement(
    *,
    tool: str,
    display: dict,
    output_ref: str,
    output: str | None = None,
    content: ToolResultTTLContent | None = None,
) -> str:
    if content is not None and content.images:
        return _build_image_tool_result_ttl_replacement(
            tool=tool,
            display=display,
            output_ref=output_ref,
            content=content,
        )
    output = output if output is not None else (content.text if content else "")
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


def _build_image_tool_result_ttl_replacement(
    *,
    tool: str,
    display: dict,
    output_ref: str,
    content: ToolResultTTLContent,
) -> str:
    label = _image_tool_result_label(tool, display)
    parts: list[str] = []
    if content.text is not None:
        line_count = content.lines or 0
        text_part = f"{line_count} line{'s' if line_count != 1 else ''} text"
        parts.append(text_part)
    image_part = f"{content.image_count} image{'s' if content.image_count != 1 else ''}"
    if content.image_bytes is not None:
        image_part += f" / {_format_bytes(content.image_bytes)}"
    parts.append(image_part)

    refs = [image.ref for image in content.images]
    if len(refs) == 1:
        image_ref = f"Image stored at {refs[0]}."
    elif refs:
        preview = ", ".join(refs[:3])
        if len(refs) > 3:
            preview += ", ..."
        image_ref = f"Images stored at {preview}."
    else:
        image_ref = ""

    return f"[{label}: {' + '.join(parts)}. {image_ref} Details saved to {output_ref}]"


def _image_tool_result_label(tool: str, display: dict) -> str:
    title = display.get("title")
    if title:
        return _clip(str(title), 80)
    kind = display.get("kind")
    if kind == "read" and display.get("path") is not None:
        return f"Read image {display['path']}"
    if kind and kind != "text":
        return f"{tool} {kind}"
    return f"{tool} image result"


def _format_bytes(size: int) -> str:
    if size < 1024:
        return f"{size}B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f}KB"
    return f"{size / (1024 * 1024):.1f}MB"

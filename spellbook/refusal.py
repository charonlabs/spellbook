"""Canonical refusal parsing, policy persistence, and request projection.

Refusals are transcript truth represented by :class:`IRRefusalBlock`. A
``RefusalRenderPolicy`` controls only what subsequent model requests see.
The production default preserves partial model text and follows it with an
honest user-role system note; provider metadata and the synthetic refusal
envelope never return to the model.

Legacy transcripts flattened refusal output into assistant strings. Those
strings are normalized strictly at projection time, allowing old append-only
history to receive the safer rendering without rewriting block coordinates.
"""

from __future__ import annotations

from collections.abc import Sequence
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict

from spellbook.ir_types import (
    IRAssistantTextBlock,
    IRBlock,
    IRRefusalBlock,
    IRRefusalDetails,
    IRRefusalSegment,
    IRRuntimeConfigRecord,
    IRUserTextBlock,
    RuntimeConfigValue,
)

REFUSAL_OPEN = "<refusal>"
REFUSAL_CLOSE = "</refusal>"
REFUSAL_RUNTIME_CONFIG_NAMESPACE = "refusal_rendering"
SYSTEM_INTERRUPTION_TEXT = (
    "<spellbook>\n"
    "A system interruption occurred. Your previous response was not delivered.\n"
    "</spellbook>"
)

RefusalAssistantMode = Literal["legacy", "partial", "none"]

_WRAPPED_SEGMENT_RE = re.compile(
    r"<(?P<tag>thinking_summary|partial_tool_call_json)>\n"
    r"(?P<text>.*?)\n"
    r"</(?P=tag)>",
    re.DOTALL,
)


class RefusalParseError(ValueError):
    """Raised when a flattened refusal payload cannot be parsed safely."""


class RefusalRenderPolicy(BaseModel, frozen=True):
    """Composable provider projection for canonical refusal events."""

    model_config = ConfigDict(extra="forbid")
    assistant_mode: RefusalAssistantMode = "partial"
    append_system_note: bool = True
    include_thinking_summaries: bool = True


PARTIAL_NOTE_REFUSAL_RENDER_POLICY = RefusalRenderPolicy()
LEGACY_REFUSAL_RENDER_POLICY = RefusalRenderPolicy(
    assistant_mode="legacy",
    append_system_note=False,
)
DEFAULT_REFUSAL_RENDER_POLICY = PARTIAL_NOTE_REFUSAL_RENDER_POLICY


class RefusalRenderer:
    """Render canonical and legacy refusal blocks for model-facing surfaces."""

    def __init__(self, policy: RefusalRenderPolicy | None = None):
        self.policy = policy or DEFAULT_REFUSAL_RENDER_POLICY

    def project_surface(self, blocks: Sequence[IRBlock]) -> list[IRBlock]:
        """Project refusal events into provider-ready ordinary IR blocks."""
        projected: list[IRBlock] = []
        for block in blocks:
            refusal = canonical_refusal(block)
            if refusal is None:
                projected.append(block)
            else:
                projected.extend(self.render_refusal(refusal))
        return projected

    def project_analysis(self, blocks: Sequence[IRBlock]) -> list[IRBlock]:
        """Count-preserving projection for semantic awareness subsystems."""
        projected: list[IRBlock] = []
        for block in blocks:
            refusal = canonical_refusal(block)
            if refusal is None:
                projected.append(block)
                continue
            rendered = self.render_refusal(refusal)
            if len(rendered) == 1:
                projected.append(rendered[0])
            elif not rendered:
                projected.append(
                    IRAssistantTextBlock(
                        time=refusal.time,
                        turn_id=refusal.turn_id,
                        event_id=refusal.event_id,
                        text="",
                    )
                )
            else:
                partial = next(
                    (
                        item.text
                        for item in rendered
                        if isinstance(item, IRAssistantTextBlock)
                    ),
                    "",
                )
                note = next(
                    (
                        item.text
                        for item in rendered
                        if isinstance(item, IRUserTextBlock)
                    ),
                    SYSTEM_INTERRUPTION_TEXT,
                )
                projected.append(
                    IRUserTextBlock(
                        time=refusal.time,
                        turn_id=refusal.turn_id,
                        event_id=refusal.event_id,
                        origin="system",
                        text=(
                            "<assistant-partial>\n"
                            f"{partial}\n"
                            "</assistant-partial>\n\n"
                            f"{note}"
                        ),
                    )
                )
        return projected

    def render_refusal(self, block: IRRefusalBlock) -> list[IRBlock]:
        rendered: list[IRBlock] = []
        assistant_text = self._assistant_text(block)
        if assistant_text:
            rendered.append(
                IRAssistantTextBlock(
                    time=block.time,
                    turn_id=block.turn_id,
                    event_id=block.event_id,
                    text=assistant_text,
                )
            )
        if self.policy.append_system_note:
            rendered.append(
                IRUserTextBlock(
                    time=block.time,
                    turn_id=block.turn_id,
                    origin="system",
                    text=SYSTEM_INTERRUPTION_TEXT,
                )
            )
        return rendered

    def _assistant_text(self, block: IRRefusalBlock) -> str:
        match self.policy.assistant_mode:
            case "legacy":
                return render_legacy_refusal(
                    block,
                    include_thinking_summaries=(self.policy.include_thinking_summaries),
                )
            case "partial":
                return block.partial_text
            case "none":
                return ""


def canonical_refusal(block: IRBlock) -> IRRefusalBlock | None:
    """Return canonical refusal truth for canonical or legacy block shapes."""
    if isinstance(block, IRRefusalBlock):
        return block
    if not isinstance(block, IRAssistantTextBlock) or REFUSAL_OPEN not in block.text:
        return None
    parsed = parse_legacy_refusal_text(block.text)
    return parsed.model_copy(
        update={
            "time": block.time,
            "turn_id": block.turn_id,
            "event_id": block.event_id,
        }
    )


def refusal_policy_runtime_values(
    policy: RefusalRenderPolicy,
) -> dict[str, RuntimeConfigValue]:
    """Serialize a policy into ``IRRuntimeConfigRecord`` values."""
    return {
        "assistant_mode": policy.assistant_mode,
        "append_system_note": policy.append_system_note,
        "include_thinking_summaries": policy.include_thinking_summaries,
    }


def refusal_policy_from_runtime_config_records(
    records: Sequence[IRRuntimeConfigRecord],
) -> RefusalRenderPolicy | None:
    """Return the latest explicit transcript policy, if one exists."""
    for record in reversed(records):
        if record.namespace == REFUSAL_RUNTIME_CONFIG_NAMESPACE:
            return RefusalRenderPolicy.model_validate(record.effective)
    return None


def parse_legacy_refusal_text(
    text: str,
    *,
    provider: str = "anthropic",
) -> IRRefusalBlock:
    """Parse an old flattened assistant refusal into canonical IR."""
    marker = f"{REFUSAL_OPEN}\n"
    refusal_start = text.rfind(marker)
    if refusal_start < 0:
        raise RefusalParseError("Assistant text has no terminal <refusal> block.")
    if refusal_start > 0 and not text[:refusal_start].endswith("\n\n"):
        raise RefusalParseError("Refusal block is not separated from prior output.")

    prefix = text[:refusal_start]
    if prefix.endswith("\n\n"):
        prefix = prefix[:-2]
    refusal_text = text[refusal_start:]
    details = _parse_refusal_details(refusal_text, provider=provider)
    segments = _parse_legacy_segments(prefix)
    return IRRefusalBlock(segments=segments, details=details)


def render_legacy_refusal(
    block: IRRefusalBlock,
    *,
    include_thinking_summaries: bool = True,
) -> str:
    """Render the legacy flattened assistant payload deterministically."""
    sections: list[str] = []
    for segment in block.segments:
        match segment.kind:
            case "text":
                if segment.text:
                    sections.append(segment.text)
            case "thinking_summary":
                if include_thinking_summaries and segment.text:
                    sections.append(
                        f"<thinking_summary>\n{segment.text}\n</thinking_summary>"
                    )
            case "partial_tool_call_json":
                if segment.text:
                    sections.append(
                        "<partial_tool_call_json>\n"
                        f"{segment.text}\n"
                        "</partial_tool_call_json>"
                    )
    sections.append(_render_refusal_details(block.details))
    return "\n\n".join(sections)


def _parse_legacy_segments(prefix: str) -> list[IRRefusalSegment]:
    if not prefix:
        return []
    segments: list[IRRefusalSegment] = []
    cursor = 0
    for match in _WRAPPED_SEGMENT_RE.finditer(prefix):
        before = prefix[cursor : match.start()]
        if before.endswith("\n\n"):
            before = before[:-2]
        if before:
            segments.append(IRRefusalSegment(kind="text", text=before))
        tag = match.group("tag")
        kind = (
            "thinking_summary"
            if tag == "thinking_summary"
            else "partial_tool_call_json"
        )
        segments.append(IRRefusalSegment(kind=kind, text=match.group("text")))
        cursor = match.end()
        if prefix[cursor:].startswith("\n\n"):
            cursor += 2
    tail = prefix[cursor:]
    if tail:
        segments.append(IRRefusalSegment(kind="text", text=tail))
    return segments


def _parse_refusal_details(text: str, *, provider: str) -> IRRefusalDetails:
    if not text.endswith(REFUSAL_CLOSE):
        raise RefusalParseError("Refusal block is missing its closing tag.")
    lines = text.splitlines()
    if len(lines) < 3 or lines[0] != REFUSAL_OPEN or lines[-1] != REFUSAL_CLOSE:
        raise RefusalParseError("Refusal block has an invalid envelope.")
    if lines[1] != "stop_reason: refusal":
        raise RefusalParseError("Refusal block has no refusal stop reason.")

    detail_type: str | None = None
    category: str | None = None
    explanation: str | None = None
    idx = 2
    while idx < len(lines) - 1:
        line = lines[idx]
        if line == "details: unavailable":
            idx += 1
            continue
        if line.startswith("type: "):
            detail_type = line.removeprefix("type: ")
        elif line.startswith("category: "):
            category = line.removeprefix("category: ")
        elif line == "explanation:":
            explanation = "\n".join(lines[idx + 1 : -1])
            idx = len(lines) - 1
            continue
        else:
            raise RefusalParseError(f"Unknown refusal detail line: {line!r}.")
        idx += 1
    return IRRefusalDetails(
        provider=provider,
        detail_type=detail_type,
        category=category,
        explanation=explanation,
    )


def _render_refusal_details(details: IRRefusalDetails | None) -> str:
    lines = [REFUSAL_OPEN, "stop_reason: refusal"]
    if details is not None:
        if details.detail_type is not None:
            lines.append(f"type: {details.detail_type}")
        if details.category is not None:
            lines.append(f"category: {details.category}")
        if details.explanation is not None:
            lines.extend(["explanation:", details.explanation])
        if (
            details.detail_type is None
            and details.category is None
            and details.explanation is None
        ):
            lines.append("details: unavailable")
    else:
        lines.append("details: unavailable")
    lines.append(REFUSAL_CLOSE)
    return "\n".join(lines)

"""Conduit routing helpers for the core app server."""

from __future__ import annotations

import hashlib
import json
from typing import Any
from xml.sax.saxutils import escape as xml_escape

from spellbook.ir_types import IRInboundMessage, IRUserTextBlock

HUMAN_SURFACES: dict[str, str] = {
    "tui": "terminal TUI",
    "human": "terminal TUI",
    "telegram": "Telegram",
    "web": "web UI",
}

RESERVED_CONDUIT_METADATA = frozenset({"priority", "title", "key"})


def surface_for_inbound(message: IRInboundMessage) -> str | None:
    """Return the human-readable surface label for human-origin inbound."""
    metadata = message.source_metadata
    if metadata.get("conduit_type") in {"context", "notification"}:
        return None

    source = str(metadata.get("source") or "").strip()
    if not source and any(
        isinstance(block, IRUserTextBlock) and block.origin == "human"
        for block in message.blocks
    ):
        source = "human"

    return surface_for_source(source)


def surface_for_source(source: str) -> str | None:
    normalized = source.strip()
    if normalized in HUMAN_SURFACES:
        return HUMAN_SURFACES[normalized]
    prefix = normalized.split(".")[0] if "." in normalized else None
    if prefix and prefix in HUMAN_SURFACES:
        return HUMAN_SURFACES[prefix]
    return None


def frame_conduit(
    source: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Wrap non-human conduit content for delivery as a user-role message."""
    source_attr = xml_escape(source, {'"': "&quot;"})
    lines = [f'<chorus-conduit source="{source_attr}">']
    title = conduit_title(metadata)
    if title:
        lines.append(escape_conduit_body(title))
        lines.append("")
    lines.append(escape_conduit_body(content))
    details = format_conduit_details(metadata)
    if details:
        lines.append("")
        lines.extend(details)
    lines.append("</chorus-conduit>")
    return "\n".join(lines)


def conduit_priority(metadata: dict[str, Any] | None) -> int:
    if not metadata:
        return 50
    raw = metadata.get("priority", 50)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 50


def conduit_key(
    conduit_type: str,
    source: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    if metadata:
        explicit_key = " ".join(str(metadata.get("key", "")).split()).strip()
        if explicit_key:
            return f"conduit:{explicit_key}"

    payload = {
        "type": conduit_type,
        "source": source,
        "title": conduit_title(metadata),
        "content": content,
        "metadata": normalized_conduit_metadata(metadata),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()[:16]
    return f"conduit:{digest}"


def format_conduit_footer(
    source: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    prefix = f"[{xml_escape(source)}] " if source else ""
    title = conduit_title(metadata)
    body = (
        f"{prefix}{escape_conduit_body(title)}"
        if title
        else f"{prefix}{escape_conduit_body(content)}"
    )
    details = format_conduit_details(metadata)
    if title and content:
        details = [escape_conduit_body(content), *details]
    if details:
        return "\n".join([body, *details])
    return body


def conduit_title(metadata: dict[str, Any] | None) -> str | None:
    if not metadata:
        return None
    raw = metadata.get("title")
    if raw is None:
        return None
    title = " ".join(str(raw).split()).strip()
    return title or None


def normalized_conduit_metadata(
    metadata: dict[str, Any] | None,
) -> dict[str, str]:
    if not metadata:
        return {}
    normalized: dict[str, str] = {}
    for raw_key, value in sorted(metadata.items(), key=lambda item: str(item[0])):
        key = " ".join(str(raw_key).split()).strip()
        if not key or key in RESERVED_CONDUIT_METADATA or value is None:
            continue
        normalized[key] = render_conduit_value(value)
    return normalized


def render_conduit_value(value: object) -> str:
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    return " ".join(str(value).split())


def escape_conduit_body(text: str) -> str:
    return "\n".join(xml_escape(line) for line in text.splitlines()) if text else ""


def format_conduit_details(metadata: dict[str, Any] | None) -> list[str]:
    details: list[str] = []
    for key, value in normalized_conduit_metadata(metadata).items():
        details.append(f"{xml_escape(key)}: {xml_escape(value)}")
    return details

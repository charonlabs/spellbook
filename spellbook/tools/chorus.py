"""Tools that cross the Spellbook -> MiniChorus boundary."""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel, Field

from spellbook.ir_types import IRToolTextBlock
from spellbook.tools.common import (
    Tool,
    ToolError,
    ToolExecutionResult,
    ToolMetadata,
)

REACH_TIMEOUT_SECONDS = 10


class ReachInput(BaseModel):
    """Send a short outbound bridge message through MiniChorus."""

    message: str = Field(
        min_length=1,
        description="The message to send to Ryan through the bridge.",
    )
    surface: str = Field(
        default="imessage",
        min_length=1,
        description="The external messaging surface to use. Defaults to iMessage.",
    )


class _BridgeMessageResponse(BaseModel):
    accepted: bool
    surface: str | None = None
    detail: str | None = None


async def exec_reach(meta: ToolMetadata, input: ReachInput) -> ToolExecutionResult:
    """Send a message through MiniChorus' outbound bridge endpoint."""
    chorus_url = _normalize_chorus_url(meta.chorus_url)
    endpoint = f"{chorus_url}/bridge/messages"
    payload: dict[str, Any] = {
        "surface": input.surface,
        "content": input.message,
        "source": "spellbook",
        "metadata": {
            "tool": "Reach",
            "transcript_path": str(meta.transcript_path),
        },
    }
    if meta.chorus_entity_name is not None:
        payload["entity_name"] = meta.chorus_entity_name

    try:
        async with httpx.AsyncClient(timeout=REACH_TIMEOUT_SECONDS) as client:
            response = await client.post(endpoint, json=payload)
        if response.status_code >= 400:
            detail = _compact_response_text(response.text)
            raise ToolError(
                f"Reach failed: MiniChorus returned HTTP {response.status_code}: "
                f"{detail}"
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise ToolError("Reach failed: MiniChorus returned invalid JSON.") from exc
    except ToolError:
        raise
    except httpx.HTTPError as exc:
        raise ToolError(f"Reach failed: could not contact MiniChorus: {exc}") from exc

    try:
        result = _BridgeMessageResponse.model_validate(data)
    except Exception as exc:
        raise ToolError(
            "Reach failed: MiniChorus returned an unexpected response."
        ) from exc

    if not result.accepted:
        detail = result.detail or "MiniChorus did not accept the bridge message."
        raise ToolError(f"Reach failed: {detail}")

    surface = result.surface or input.surface
    detail = result.detail or "Message accepted by the bridge."
    return ToolExecutionResult(
        content=[IRToolTextBlock(text=f"Reach accepted via {surface}: {detail}")],
        display={
            "kind": "reach",
            "surface": surface,
            "accepted": result.accepted,
            "detail": detail,
        },
    )


def _normalize_chorus_url(value: str | None) -> str:
    if value is None or not value.strip():
        raise ToolError(
            "Reach is unavailable because this entity has no MiniChorus URL configured."
        )
    return value.strip().rstrip("/")


def _compact_response_text(text: str) -> str:
    compact = " ".join(text.split())
    if not compact:
        return "(empty response body)"
    if len(compact) > 500:
        return f"{compact[:497]}..."
    return compact


REACH_TOOL: Tool[ReachInput] = Tool(
    name="Reach",
    input_model=ReachInput,
    exec=exec_reach,
    category="chorus_tools",
)

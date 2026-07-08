"""Tool access to a configured physical body brainstem."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

import httpx
from pydantic import BaseModel, Field

from spellbook.ir_types import IRImageBlock, IRToolResultContentBlock, IRToolTextBlock
from spellbook.tools.common import (
    Tool,
    ToolError,
    ToolExecutionResult,
    ToolMetadata,
)
from spellbook.tools.filesystem import read_image_file

BODY_TIMEOUT_SECONDS = 25.0
NO_DEVICE_MESSAGE = "Your body is out of reach right now (no device connected)."
CAPTURE_TIMEOUT_MESSAGE = "Your body is out of reach right now (capture timed out)."

BodyAction = Literal["see", "reach", "show", "status", "volume", "clear"]


class BodyInput(BaseModel):
    """Use the configured physical body."""

    action: BodyAction = Field(description="The body action to perform.")
    text: str | None = Field(
        default=None,
        description="Text to show on the body's screen. Required for action='show'.",
    )
    value: int | None = Field(
        default=None,
        ge=0,
        le=255,
        description="Numeric value for action='volume'. Must be 0-255.",
    )


async def exec_body(meta: ToolMetadata, input: BodyInput) -> ToolExecutionResult:
    body_url = _normalize_body_url(meta.body_url)
    async with httpx.AsyncClient(timeout=BODY_TIMEOUT_SECONDS) as client:
        match input.action:
            case "see":
                return await _exec_see(client, body_url=body_url, meta=meta)
            case "reach":
                return await _exec_reach(client, body_url=body_url)
            case "show":
                return await _exec_show(client, body_url=body_url, text=input.text)
            case "status":
                return await _exec_status(client, body_url=body_url)
            case "volume":
                return await _exec_volume(client, body_url=body_url, value=input.value)
            case "clear":
                return await _exec_clear(client, body_url=body_url)


async def _exec_see(
    client: httpx.AsyncClient, *, body_url: str, meta: ToolMetadata
) -> ToolExecutionResult:
    data = await _post_json(client, body_url=body_url, path="/see", json={})
    _require_ok(data)
    raw_path = data.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ToolError(
            "Your body opened its eye, but the brainstem did not return an image path."
        )

    image_path = Path(raw_path)
    if not image_path.is_absolute():
        image_path = meta.cwd / image_path

    image_result = read_image_file(image_path, meta.transcript_path)
    blob_path = _first_blob_path(image_result.content)
    blob_abs_path = (
        meta.transcript_path.parent / blob_path if blob_path is not None else None
    )
    text = (
        f"You opened your eye. Camera file: {image_path}. "
        f"Transcript image blob: {blob_path}"
    )
    if blob_abs_path is not None:
        text += f" ({blob_abs_path})"
    text += ". Read either path to see it again."

    return ToolExecutionResult(
        content=[*image_result.content, IRToolTextBlock(text=text)],
        display={
            "kind": "body",
            "action": "see",
            "path": str(image_path),
            "blob_path": blob_path,
            "body": text,
        },
    )


async def _exec_reach(
    client: httpx.AsyncClient, *, body_url: str
) -> ToolExecutionResult:
    data = await _post_json(client, body_url=body_url, path="/reach", json={})
    _require_ok(data)
    text = (
        "The ember swells and the chime sounds. If a palm answers, it will reach you "
        "as a message."
    )
    return _text_result(text, action="reach")


async def _exec_show(
    client: httpx.AsyncClient, *, body_url: str, text: str | None
) -> ToolExecutionResult:
    screen_text = _required_text(text, field_name="text", action="show")
    data = await _post_json(
        client, body_url=body_url, path="/display", json={"text": screen_text}
    )
    _require_ok(data)
    return _text_result(f'The screen now says: "{screen_text}"', action="show")


async def _exec_status(
    client: httpx.AsyncClient, *, body_url: str
) -> ToolExecutionResult:
    data = await _get_json(client, body_url=body_url, path="/status")
    connected = bool(data.get("device_connected"))
    if not connected:
        return _text_result(
            "your body is out of reach right now (no device connected).",
            action="status",
            connected=False,
        )

    heartbeat = _format_heartbeat_age(data.get("seconds_since_heartbeat"))
    device = data.get("device")
    details = _device_details(device)
    text = f"your body is connected; last heartbeat {heartbeat}."
    if details:
        text += f" {details}."
    return _text_result(text, action="status", connected=True)


async def _exec_volume(
    client: httpx.AsyncClient, *, body_url: str, value: int | None
) -> ToolExecutionResult:
    if value is None:
        raise ToolError("volume requires value.")
    data = await _post_json(
        client, body_url=body_url, path="/volume", json={"value": value}
    )
    _require_ok(data)
    returned_value = data.get("value", value)
    return _text_result(
        f"Your body's chime volume is set to {returned_value}.",
        action="volume",
        value=returned_value,
    )


async def _exec_clear(
    client: httpx.AsyncClient, *, body_url: str
) -> ToolExecutionResult:
    data = await _post_json(client, body_url=body_url, path="/clear", json={})
    _require_ok(data)
    return _text_result("The screen is clear.", action="clear")


async def _post_json(
    client: httpx.AsyncClient, *, body_url: str, path: str, json: dict[str, Any]
) -> dict[str, Any]:
    try:
        response = await client.post(f"{body_url}{path}", json=json)
    except httpx.TimeoutException as exc:
        raise ToolError(_timeout_message(path)) from exc
    except httpx.HTTPError as exc:
        raise ToolError(NO_DEVICE_MESSAGE) from exc
    return _response_json(response)


async def _get_json(
    client: httpx.AsyncClient, *, body_url: str, path: str
) -> dict[str, Any]:
    try:
        response = await client.get(f"{body_url}{path}")
    except httpx.TimeoutException as exc:
        raise ToolError(_timeout_message(path)) from exc
    except httpx.HTTPError as exc:
        raise ToolError(NO_DEVICE_MESSAGE) from exc
    return _response_json(response)


def _response_json(response: httpx.Response) -> dict[str, Any]:
    if response.status_code == 503:
        raise ToolError(NO_DEVICE_MESSAGE)
    if response.status_code == 504:
        raise ToolError(CAPTURE_TIMEOUT_MESSAGE)
    if response.status_code >= 400:
        raise ToolError(
            f"Your body returned HTTP {response.status_code}: "
            f"{_compact_response_text(response.text)}"
        )
    try:
        data = response.json()
    except ValueError as exc:
        raise ToolError("Your body returned invalid JSON.") from exc
    if not isinstance(data, dict):
        raise ToolError("Your body returned an unexpected response.")
    return data


def _require_ok(data: dict[str, Any]) -> None:
    ok = data.get("ok")
    if ok is False:
        raise ToolError(NO_DEVICE_MESSAGE)
    if ok is not True:
        raise ToolError("Your body returned an unexpected response.")


def _normalize_body_url(value: str | None) -> str:
    if value is None or not value.strip():
        raise ToolError(
            "Body is unavailable because this entity has no body URL configured."
        )
    return value.strip().rstrip("/")


def _required_text(value: str | None, *, field_name: str, action: str) -> str:
    if value is None or not value.strip():
        raise ToolError(f"{action} requires {field_name}.")
    return value.strip()


def _timeout_message(path: str) -> str:
    if path == "/see":
        return CAPTURE_TIMEOUT_MESSAGE
    return NO_DEVICE_MESSAGE


def _compact_response_text(text: str) -> str:
    compact = " ".join(text.split())
    if not compact:
        return "(empty response body)"
    if len(compact) > 500:
        return f"{compact[:497]}..."
    return compact


def _first_blob_path(content: list[IRToolResultContentBlock]) -> str | None:
    for block in content:
        if isinstance(block, IRImageBlock):
            return block.blob_path
    return None


def _format_heartbeat_age(value: object) -> str:
    if isinstance(value, int | float):
        return f"{float(value):.1f}s ago"
    return "unknown"


def _device_details(value: object) -> str:
    if not isinstance(value, Mapping):
        return ""
    device = cast(Mapping[str, object], value)
    details: list[str] = []
    device_id = device.get("device_id")
    if isinstance(device_id, str) and device_id:
        details.append(device_id)
    battery = device.get("battery")
    if isinstance(battery, int | float):
        details.append(f"battery {battery:g}%")
    return ", ".join(details)


def _text_result(text: str, *, action: str, **display: Any) -> ToolExecutionResult:
    return ToolExecutionResult(
        content=[IRToolTextBlock(text=text)],
        display={"kind": "body", "action": action, "body": text, **display},
    )


BODY_TOOL: Tool[BodyInput] = Tool(
    name="Body",
    input_model=BodyInput,
    exec=exec_body,
    category="body",
)

"""Runtime surface support for the Golem Minecraft harness."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

from spellbook.ir_types import (
    IRAssistantTextBlock,
    IRGeneration,
    IRInboundMessage,
    IRUserTextBlock,
)
from spellbook.round_lifecycle import RoundContext, RoundLifecycle

logger = logging.getLogger(__name__)

MINECRAFT_STREAM_READ_TIMEOUT = 30.0
MINECRAFT_STREAM_RECONNECT_SECONDS = 2.5
MINECRAFT_CHAT_TIMEOUT_SECONDS = 10.0

MinecraftFocusMode = Literal["explore", "combat", "build", "idle"]

_IMMEDIATE_EVENT_KINDS = frozenset(
    {
        "combat",
        "current",
        "death",
        "drowning",
        "health_critical",
        "hostile",
        "hurt",
        "lava",
        "low_food",
        "threat",
    }
)
_RESOURCE_EVENT_KINDS = frozenset({"gather", "mine", "ore", "pockets", "wear"})
_WORLD_EVENT_KINDS = frozenset({"build", "follow", "job", "scan", "world"})
_AMBIENT_EVENT_KINDS = frozenset({"hear", "nightfall", "pulse"})


SubmitMessage = Callable[[IRInboundMessage], Awaitable[object]]
QueueFooter = Callable[..., Awaitable[None]]
SurfaceProvider = Callable[[], "MinecraftSurface | None"]


@dataclass
class MinecraftSurface:
    """Mutable runtime state for one entity wearing Minecraft."""

    minecraft_url: str
    booted: bool = False
    chat_routing: bool = False
    tool_call_echo: bool = True
    focus_mode: MinecraftFocusMode = "idle"
    chat_cursor: int = 0
    event_cursor: int = 0
    stream_connected: bool = False
    _submit_message: SubmitMessage | None = field(default=None, init=False, repr=False)
    _queue_footer: QueueFooter | None = field(default=None, init=False, repr=False)
    _stream_task: asyncio.Task[None] | None = field(
        default=None, init=False, repr=False
    )
    _pending_event_summaries: list[str] = field(default_factory=list, init=False)
    _closed: bool = field(default=False, init=False)

    @property
    def active(self) -> bool:
        return self.booted and self.chat_routing

    def bind_runtime(
        self,
        *,
        submit_message: SubmitMessage,
        queue_footer: QueueFooter,
    ) -> None:
        self._submit_message = submit_message
        self._queue_footer = queue_footer
        self._closed = False
        self._ensure_stream_task()

    def mark_booted(
        self, *, chat_cursor: object = None, event_cursor: object = None
    ) -> None:
        self.booted = True
        self.chat_routing = True
        self.chat_cursor = _cursor_value(chat_cursor, self.chat_cursor)
        self.event_cursor = _cursor_value(event_cursor, self.event_cursor)
        self._closed = False
        self._ensure_stream_task()

    def configure(
        self,
        *,
        chat_routing: bool | None = None,
        tool_call_echo: bool | None = None,
        focus_mode: MinecraftFocusMode | None = None,
    ) -> None:
        if chat_routing is not None:
            self.chat_routing = chat_routing
        if tool_call_echo is not None:
            self.tool_call_echo = tool_call_echo
        if focus_mode is not None:
            self.focus_mode = focus_mode

    def mark_shutdown(self) -> None:
        self.booted = False
        self.chat_routing = False
        self.stream_connected = False
        self._pending_event_summaries.clear()
        self._cancel_stream_task()

    async def close(self) -> None:
        self._closed = True
        task = self._stream_task
        self._cancel_stream_task()
        if task is not None:
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def route_assistant_text(self, text: str) -> None:
        if not self.active:
            return
        lines = _chat_lines(text)
        if not lines:
            return
        try:
            async with httpx.AsyncClient(
                timeout=MINECRAFT_CHAT_TIMEOUT_SECONDS
            ) as client:
                if len(lines) == 1:
                    await client.get(
                        f"{_normalize_url(self.minecraft_url)}/chat",
                        params={"msg": lines[0], "brief": "1"},
                    )
                else:
                    await client.get(
                        f"{_normalize_url(self.minecraft_url)}/chat",
                        params={"lines": "|".join(lines), "brief": "1"},
                    )
        except httpx.HTTPError as exc:
            logger.warning("minecraft.chat_route_failed error=%s", exc)

    async def flush_pending_events(self) -> None:
        if self._queue_footer is None or not self._pending_event_summaries:
            return
        batch = self._pending_event_summaries[:8]
        del self._pending_event_summaries[:8]
        more = len(self._pending_event_summaries)
        suffix = f"\n... {more} more Minecraft events pending." if more else ""
        await self._queue_footer(
            text="Minecraft events:\n"
            + "\n".join(f"- {line}" for line in batch)
            + suffix,
            key="minecraft:event_batch",
            priority=30,
            wake_on_idle=False,
            metadata={"source": "minecraft", "kind": "event_batch"},
        )

    def _ensure_stream_task(self) -> None:
        if (
            not self.booted
            or self._submit_message is None
            or self._queue_footer is None
        ):
            return
        task = self._stream_task
        if task is not None and not task.done():
            return
        self._stream_task = asyncio.create_task(self._stream_loop())

    def _cancel_stream_task(self) -> None:
        task = self._stream_task
        self._stream_task = None
        if task is not None and not task.done():
            task.cancel()

    async def _stream_loop(self) -> None:
        while self.booted and not self._closed:
            try:
                await self._consume_stream_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.stream_connected = False
                logger.warning("minecraft.stream_disconnected error=%s", exc)
                await asyncio.sleep(MINECRAFT_STREAM_RECONNECT_SECONDS)

    async def _consume_stream_once(self) -> None:
        stream_url = f"{_normalize_url(self.minecraft_url)}/stream"
        timeout = httpx.Timeout(
            MINECRAFT_STREAM_READ_TIMEOUT,
            connect=MINECRAFT_CHAT_TIMEOUT_SECONDS,
        )
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "GET",
                stream_url,
                headers={"Accept": "text/event-stream"},
            ) as response:
                response.raise_for_status()
                self.stream_connected = True
                data_lines: list[str] = []
                async for line in response.aiter_lines():
                    if line == "":
                        if data_lines:
                            await self._handle_stream_data("\n".join(data_lines))
                            data_lines.clear()
                        continue
                    if line.startswith(":"):
                        continue
                    if line == "data":
                        data_lines.append("")
                    elif line.startswith("data:"):
                        data_lines.append(line[5:].lstrip(" "))

    async def _handle_stream_data(self, raw_data: str) -> None:
        try:
            payload = json.loads(raw_data)
        except json.JSONDecodeError:
            logger.warning("minecraft.stream_bad_json data=%r", raw_data[:120])
            return
        if not isinstance(payload, dict):
            return
        channel = _clean(payload.get("ch"))
        if channel == "chat":
            await self._route_game_chat(payload)
        elif channel == "event":
            await self._route_game_event(payload)

    async def _route_game_chat(self, payload: dict[str, Any]) -> None:
        if self._submit_message is None or not self.chat_routing:
            return
        player = _clean(
            payload.get("from") or payload.get("player") or payload.get("username")
        )
        message = _clean(
            payload.get("msg") or payload.get("message") or payload.get("text")
        )
        if not message:
            return
        self.chat_cursor = _cursor_value(payload.get("id"), self.chat_cursor)
        prefix = f"Minecraft chat - {player or 'unknown'}: "
        await self._submit_message(
            IRInboundMessage(
                blocks=[IRUserTextBlock(text=prefix + message, origin="human")],
                source_metadata={
                    "source": "minecraft",
                    "origin": "minecraft",
                    "minecraft": _clean_payload(payload),
                },
                delivery="inject",
            )
        )

    async def _route_game_event(self, payload: dict[str, Any]) -> None:
        self.event_cursor = _cursor_value(payload.get("id"), self.event_cursor)
        kind = _clean(payload.get("kind")) or "event"
        summary = _format_event(payload)
        if not summary:
            return
        if _event_filtered(kind, self.focus_mode):
            return
        if kind in _IMMEDIATE_EVENT_KINDS:
            if self._queue_footer is None:
                return
            await self._queue_footer(
                text=f"Minecraft event - {summary}",
                key=f"minecraft:event:{payload.get('id') or kind}",
                priority=0,
                wake_on_idle=True,
                metadata={"source": "minecraft", "kind": kind},
            )
            return
        self._pending_event_summaries.append(summary)
        if len(self._pending_event_summaries) > 40:
            del self._pending_event_summaries[: len(self._pending_event_summaries) - 40]


class MinecraftRoundLifecycle(RoundLifecycle):
    """Round hook that lets Minecraft behave as a worn surface."""

    def __init__(self, surface_provider: SurfaceProvider):
        self._surface_provider = surface_provider

    async def before_round(self, ctx: RoundContext) -> None:
        surface = self._surface_provider()
        if surface is not None:
            await surface.flush_pending_events()

    async def after_generate(
        self,
        ctx: RoundContext,
        generation: IRGeneration,
    ) -> None:
        surface = self._surface_provider()
        if surface is None or not surface.active:
            return
        for block in generation.blocks:
            if isinstance(block, IRAssistantTextBlock):
                await surface.route_assistant_text(block.text)


def _cursor_value(value: object, default: int) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return default


def _normalize_url(value: str) -> str:
    return value.strip().rstrip("/")


def _clean(value: object) -> str:
    return "" if value is None else str(value).strip()


def _clean_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in payload.items()
        if value is not None and key not in {"ch"}
    }


def _format_event(payload: dict[str, Any]) -> str:
    kind = _clean(payload.get("kind"))
    message = _clean(
        payload.get("msg") or payload.get("summary") or payload.get("message")
    )
    if kind and message:
        summary = f"{kind}: {message}"
    else:
        summary = message or kind
    if not summary:
        summary = json.dumps(
            _clean_payload(payload), sort_keys=True, separators=(",", ":")
        )
    context = {
        key: payload[key]
        for key in ("pos", "hp", "food")
        if payload.get(key) is not None
    }
    if context:
        summary += f" [{json.dumps(context, sort_keys=True, separators=(',', ':'))}]"
    return summary


def _event_filtered(kind: str, focus_mode: MinecraftFocusMode) -> bool:
    if focus_mode in {"explore", "idle"}:
        return False
    if focus_mode == "combat":
        return kind not in _IMMEDIATE_EVENT_KINDS
    if focus_mode == "build":
        return kind not in _IMMEDIATE_EVENT_KINDS and kind not in _WORLD_EVENT_KINDS
    return False


def _chat_lines(text: str) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    if not ascii_text:
        return []
    chunks: list[str] = []
    current = ascii_text
    while current and len(chunks) < 5:
        if len(current) <= 220:
            chunks.append(current)
            break
        split_at = current.rfind(" ", 0, 220)
        if split_at < 80:
            split_at = 220
        chunks.append(current[:split_at].strip())
        current = current[split_at:].strip()
    return [chunk for chunk in chunks if chunk]

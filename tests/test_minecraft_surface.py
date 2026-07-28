from __future__ import annotations

import json
from typing import Any

import pytest

from spellbook.cancel_token import CancelToken
from spellbook.ir_types import (
    IRAssistantTextBlock,
    IRGeneration,
    IRInboundMessage,
    IRUserTextBlock,
)
from spellbook.minecraft_surface import MinecraftRoundLifecycle, MinecraftSurface
from spellbook.round_lifecycle import RoundContext

pytestmark = pytest.mark.asyncio


class _FakeResponse:
    status_code = 200
    text = "{}"

    def json(self) -> dict[str, Any]:
        return {"ok": True}


class _FakeAsyncClient:
    instances: list["_FakeAsyncClient"] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.gets: list[tuple[str, dict[str, Any]]] = []
        self.instances.append(self)

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def get(self, url: str, *, params: dict[str, Any]) -> _FakeResponse:
        self.gets.append((url, params))
        return _FakeResponse()


async def test_round_lifecycle_routes_assistant_text_to_minecraft_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from spellbook import minecraft_surface as surface_module

    _FakeAsyncClient.instances = []
    monkeypatch.setattr(surface_module.httpx, "AsyncClient", _FakeAsyncClient)
    surface = MinecraftSurface(minecraft_url="http://minecraft.local/")
    surface.booted = True
    surface.chat_routing = True
    lifecycle = MinecraftRoundLifecycle(lambda: surface)
    generation = IRGeneration(
        model="test",
        blocks=[IRAssistantTextBlock(text="On my way.")],
        stop_reason="end_turn",
        usage=None,
    )

    await lifecycle.after_generate(
        RoundContext(
            blocks=[],
            round_number=1,
            cancel_token=CancelToken(),
            blocks_this_round=[],
        ),
        generation,
    )

    assert _FakeAsyncClient.instances[0].gets == [
        ("http://minecraft.local/chat", {"msg": "On my way.", "brief": "1"})
    ]


async def test_stream_chat_becomes_injected_human_message() -> None:
    submitted: list[IRInboundMessage] = []
    footers: list[dict[str, Any]] = []
    surface = MinecraftSurface(minecraft_url="http://minecraft.local")
    surface.booted = True
    surface.chat_routing = True
    surface.bind_runtime(
        submit_message=lambda message: _append_message(submitted, message),
        queue_footer=lambda **kwargs: _append_footer(footers, kwargs),
    )
    surface._cancel_stream_task()

    await surface._handle_stream_data(
        json.dumps({"ch": "chat", "id": 3, "from": "Ryan", "msg": "come here"})
    )

    assert len(submitted) == 1
    message = submitted[0]
    assert message.delivery == "inject"
    assert message.source_metadata["source"] == "minecraft"
    block = message.blocks[0]
    assert isinstance(block, IRUserTextBlock)
    assert block.origin == "human"
    assert block.text == "Minecraft chat - Ryan: come here"
    assert footers == []


async def test_stream_events_immediate_or_batched() -> None:
    submitted: list[IRInboundMessage] = []
    footers: list[dict[str, Any]] = []
    surface = MinecraftSurface(minecraft_url="http://minecraft.local")
    surface.booted = True
    surface.chat_routing = True
    surface.bind_runtime(
        submit_message=lambda message: _append_message(submitted, message),
        queue_footer=lambda **kwargs: _append_footer(footers, kwargs),
    )
    surface._cancel_stream_task()

    await surface._handle_stream_data(
        json.dumps({"ch": "event", "id": 4, "kind": "ore", "msg": "iron N ~8"})
    )
    assert footers == []

    await surface.flush_pending_events()
    assert footers[0]["text"] == "Minecraft events:\n- ore: iron N ~8"
    assert footers[0]["wake_on_idle"] is False

    await surface._handle_stream_data(
        json.dumps(
            {
                "ch": "event",
                "id": 5,
                "kind": "threat",
                "msg": "zombie E ~3",
                "hp": 18,
            }
        )
    )
    assert footers[1]["text"] == 'Minecraft event - threat: zombie E ~3 [{"hp":18}]'
    assert footers[1]["wake_on_idle"] is True
    assert submitted == []


async def _append_message(
    target: list[IRInboundMessage], message: IRInboundMessage
) -> object:
    target.append(message)
    return object()


async def _append_footer(target: list[dict[str, Any]], value: dict[str, Any]) -> None:
    target.append(value)

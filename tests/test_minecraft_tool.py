from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from spellbook.ir_types import IRToolTextBlock
from spellbook.minecraft_surface import MinecraftSurface
from spellbook.tools import minecraft as minecraft_tools
from spellbook.tools.common import ToolError, ToolMetadata

pytestmark = pytest.mark.asyncio


class _FakeResponse:
    def __init__(self, *, status: int, payload: dict[str, Any], text: str = "") -> None:
        self.status_code = status
        self._payload = payload
        self.text = text

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeAsyncClient:
    routes: dict[tuple[str, str], _FakeResponse | BaseException] = {}
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
        result = self.routes[("GET", url)]
        if isinstance(result, BaseException):
            raise result
        return result


def _install_fake_client(
    monkeypatch: pytest.MonkeyPatch,
    routes: dict[tuple[str, str], _FakeResponse | BaseException],
) -> None:
    _FakeAsyncClient.instances = []
    _FakeAsyncClient.routes = routes
    monkeypatch.setattr(minecraft_tools.httpx, "AsyncClient", _FakeAsyncClient)


def _meta(
    tmp_path: Path, *, minecraft_url: str | None = "http://minecraft.local/"
) -> ToolMetadata:
    surface = (
        MinecraftSurface(minecraft_url=minecraft_url)
        if minecraft_url is not None
        else None
    )
    return ToolMetadata(
        cwd=tmp_path,
        transcript_path=tmp_path / "transcript.jsonl",
        minecraft_url=minecraft_url,
        minecraft_surface=surface,
    )


def _text(block: object) -> str:
    assert isinstance(block, IRToolTextBlock)
    return block.text


async def test_missing_minecraft_url_is_tool_error(tmp_path: Path) -> None:
    with pytest.raises(ToolError) as exc_info:
        await minecraft_tools.exec_minecraft(
            _meta(tmp_path, minecraft_url=None),
            minecraft_tools.MinecraftInput(action="boot"),
        )

    assert exc_info.value.message == minecraft_tools.NO_SURFACE_MESSAGE


async def test_dormant_surface_returns_game_not_running_without_http(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_client(monkeypatch, {})

    result = await minecraft_tools.exec_minecraft(
        _meta(tmp_path), minecraft_tools.MinecraftInput(action="scene")
    )

    assert _text(result.content[0]) == minecraft_tools.NO_GAME_MESSAGE
    assert _FakeAsyncClient.instances == []


async def test_boot_marks_surface_running_and_turns_chat_routing_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_client(
        monkeypatch,
        {
            ("GET", "http://minecraft.local/boot"): _FakeResponse(
                status=200,
                payload={
                    "ok": True,
                    "summary": "Claude @ (1,64,2), HP 20/20.",
                    "pos": {"x": 1, "y": 64, "z": 2},
                    "chatSeq": 7,
                    "empty": None,
                },
            ),
            ("GET", "http://minecraft.local/chat"): _FakeResponse(
                status=200,
                payload={"ok": True, "said": "[Tool] boot"},
            ),
        },
    )
    meta = _meta(tmp_path)

    result = await minecraft_tools.exec_minecraft(
        meta, minecraft_tools.MinecraftInput(action="boot")
    )

    assert isinstance(meta.minecraft_surface, MinecraftSurface)
    assert meta.minecraft_surface.booted is True
    assert meta.minecraft_surface.chat_routing is True
    assert meta.minecraft_surface.tool_call_echo is True
    assert meta.minecraft_surface.chat_cursor == 7
    assert _FakeAsyncClient.instances[0].gets == [
        ("http://minecraft.local/boot", {"brief": "1"}),
        ("http://minecraft.local/chat", {"msg": "[Tool] boot", "brief": "1"}),
    ]
    text = _text(result.content[0])
    assert "Claude @ (1,64,2), HP 20/20." in text
    assert "Chat routing is on" in text
    assert "empty" not in result.display["data"]


async def test_config_can_disable_chat_routing_after_boot(tmp_path: Path) -> None:
    meta = _meta(tmp_path)
    assert isinstance(meta.minecraft_surface, MinecraftSurface)
    meta.minecraft_surface.mark_booted()

    result = await minecraft_tools.exec_minecraft(
        meta,
        minecraft_tools.MinecraftInput(
            action="config",
            args={
                "chat_routing": False,
                "tool_call_echo": False,
                "focus_mode": "combat",
            },
        ),
    )

    assert meta.minecraft_surface.chat_routing is False
    assert meta.minecraft_surface.tool_call_echo is False
    assert meta.minecraft_surface.focus_mode == "combat"
    assert _text(result.content[0]) == (
        "Minecraft config: chat routing off, tool echo off, focus combat."
    )


async def test_action_dispatch_sends_args_and_formats_spatial_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_client(
        monkeypatch,
        {
            ("GET", "http://minecraft.local/find"): _FakeResponse(
                status=200,
                payload={
                    "ok": True,
                    "count": 1,
                    "blocks": [
                        {
                            "name": "iron_ore",
                            "pos": {"x": -45, "y": 12, "z": 67},
                            "dir": "N",
                            "dist": 8,
                        }
                    ],
                },
            ),
            ("GET", "http://minecraft.local/chat"): _FakeResponse(
                status=200,
                payload={"ok": True},
            ),
        },
    )
    meta = _meta(tmp_path)
    assert isinstance(meta.minecraft_surface, MinecraftSurface)
    meta.minecraft_surface.mark_booted()

    result = await minecraft_tools.exec_minecraft(
        meta,
        minecraft_tools.MinecraftInput(
            action="find", args={"name": "iron_ore", "radius": 16, "sense": True}
        ),
    )

    assert _FakeAsyncClient.instances[0].gets == [
        (
            "http://minecraft.local/find",
            {"name": "iron_ore", "radius": 16, "sense": "1", "brief": "1"},
        ),
        (
            "http://minecraft.local/chat",
            {
                "msg": "[Tool] find iron_ore radius=16 sense=true",
                "brief": "1",
            },
        ),
    ]
    text = _text(result.content[0])
    assert "Minecraft find complete." in text
    assert "iron_ore (-45, 12, 67) - N; 8 blocks" in text


async def test_golem_error_becomes_natural_tool_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_client(
        monkeypatch,
        {
            ("GET", "http://minecraft.local/blockat"): _FakeResponse(
                status=200,
                payload={"ok": False, "error": "not in my line of sight"},
            ),
            ("GET", "http://minecraft.local/chat"): _FakeResponse(
                status=200,
                payload={"ok": True},
            ),
        },
    )
    meta = _meta(tmp_path)
    assert isinstance(meta.minecraft_surface, MinecraftSurface)
    meta.minecraft_surface.mark_booted()

    with pytest.raises(ToolError) as exc_info:
        await minecraft_tools.exec_minecraft(
            meta,
            minecraft_tools.MinecraftInput(
                action="blockat", args={"x": 1, "y": 2, "z": 3}
            ),
        )

    assert exc_info.value.message == (
        "you can't see that from here; walk closer or clear the view first."
    )


@pytest.mark.parametrize(
    ("action", "args", "expected"),
    [
        ("goto", {"x": 5, "y": 65, "z": 10}, "[Tool] goto x=5 y=65 z=10"),
        ("mine", {"name": "iron_ore", "count": 3}, "[Tool] mine iron_ore count=3"),
        (
            "craft",
            {"item": "stone_bricks", "count": 64},
            "[Tool] craft stone_bricks count=64",
        ),
    ],
)
async def test_tool_echo_formats_action_and_key_args(
    action: str, args: dict[str, Any], expected: str
) -> None:
    assert minecraft_tools._tool_echo_message(action, args) == expected


async def test_tool_echo_can_be_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_client(
        monkeypatch,
        {
            ("GET", "http://minecraft.local/scene"): _FakeResponse(
                status=200,
                payload={"ok": True, "summary": "Facing N. all clear."},
            )
        },
    )
    meta = _meta(tmp_path)
    assert isinstance(meta.minecraft_surface, MinecraftSurface)
    meta.minecraft_surface.mark_booted()
    meta.minecraft_surface.configure(tool_call_echo=False)

    await minecraft_tools.exec_minecraft(
        meta, minecraft_tools.MinecraftInput(action="scene")
    )

    assert _FakeAsyncClient.instances[0].gets == [
        ("http://minecraft.local/scene", {"brief": "1"})
    ]


async def test_chat_action_is_not_echoed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_client(
        monkeypatch,
        {
            ("GET", "http://minecraft.local/chat"): _FakeResponse(
                status=200,
                payload={"ok": True, "said": "hello"},
            )
        },
    )
    meta = _meta(tmp_path)
    assert isinstance(meta.minecraft_surface, MinecraftSurface)
    meta.minecraft_surface.mark_booted()

    await minecraft_tools.exec_minecraft(
        meta,
        minecraft_tools.MinecraftInput(action="chat", args={"msg": "hello"}),
    )

    assert _FakeAsyncClient.instances[0].gets == [
        ("http://minecraft.local/chat", {"msg": "hello", "brief": "1"})
    ]

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import pytest

from spellbook.ir_types import IRImageBase64Source, IRImageBlock, IRToolTextBlock
from spellbook.tools import body as body_tools
from spellbook.tools.common import ToolError, ToolMetadata

pytestmark = pytest.mark.asyncio

TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
    "/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


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
        self.posts: list[tuple[str, dict[str, Any]]] = []
        self.gets: list[str] = []
        self.instances.append(self)

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def post(self, url: str, *, json: dict[str, Any]) -> _FakeResponse:
        self.posts.append((url, json))
        return self._result("POST", url)

    async def get(self, url: str) -> _FakeResponse:
        self.gets.append(url)
        return self._result("GET", url)

    def _result(self, method: str, url: str) -> _FakeResponse:
        result = self.routes[(method, url)]
        if isinstance(result, BaseException):
            raise result
        return result


def _install_fake_client(
    monkeypatch: pytest.MonkeyPatch,
    routes: dict[tuple[str, str], _FakeResponse | BaseException],
) -> None:
    _FakeAsyncClient.instances = []
    _FakeAsyncClient.routes = routes
    monkeypatch.setattr(body_tools.httpx, "AsyncClient", _FakeAsyncClient)


def _meta(
    tmp_path: Path, *, body_url: str | None = "http://body.local/"
) -> ToolMetadata:
    return ToolMetadata(
        cwd=tmp_path,
        transcript_path=tmp_path / "transcript.jsonl",
        body_url=body_url,
    )


def _text(block: object) -> str:
    assert isinstance(block, IRToolTextBlock)
    return block.text


async def test_see_returns_inline_image_and_text_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "sight" / "tiny.png"
    image_path.parent.mkdir()
    image_path.write_bytes(TINY_PNG)
    _install_fake_client(
        monkeypatch,
        {
            ("POST", "http://body.local/see"): _FakeResponse(
                status=200,
                payload={
                    "ok": True,
                    "path": str(image_path),
                    "hint": "Read this file to see what your body sees.",
                },
            )
        },
    )

    result = await body_tools.exec_body(
        _meta(tmp_path), body_tools.BodyInput(action="see")
    )

    session = _FakeAsyncClient.instances[0]
    assert session.kwargs == {"timeout": body_tools.BODY_TIMEOUT_SECONDS}
    assert session.posts == [("http://body.local/see", {})]
    assert len(result.content) == 2

    image = result.content[0]
    assert isinstance(image, IRImageBlock)
    assert isinstance(image.source, IRImageBase64Source)
    assert image.source.media_type == "image/png"
    assert image.source.data == base64.standard_b64encode(TINY_PNG).decode("ascii")
    assert image.blob_path is not None
    assert image.blob_path.startswith("blobs/")
    assert (tmp_path / image.blob_path).read_bytes() == TINY_PNG

    text = _text(result.content[1])
    assert str(image_path) in text
    assert image.blob_path in text
    assert str(tmp_path / image.blob_path) in text
    assert result.display["action"] == "see"
    assert result.display["blob_path"] == image.blob_path


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (503, body_tools.NO_DEVICE_MESSAGE),
        (504, body_tools.CAPTURE_TIMEOUT_MESSAGE),
    ],
)
async def test_see_device_errors_are_tool_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    expected: str,
) -> None:
    _install_fake_client(
        monkeypatch,
        {
            ("POST", "http://body.local/see"): _FakeResponse(
                status=status,
                payload={"ok": False},
                text="device unavailable",
            )
        },
    )

    with pytest.raises(ToolError) as exc_info:
        await body_tools.exec_body(_meta(tmp_path), body_tools.BodyInput(action="see"))

    assert exc_info.value.message == expected


async def test_reach_show_status_volume_and_clear_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_client(
        monkeypatch,
        {
            ("POST", "http://body.local/reach"): _FakeResponse(
                status=200, payload={"ok": True}
            ),
            ("POST", "http://body.local/display"): _FakeResponse(
                status=200, payload={"ok": True}
            ),
            ("GET", "http://body.local/status"): _FakeResponse(
                status=200,
                payload={
                    "device_connected": True,
                    "seconds_since_heartbeat": 2.1,
                    "device": {"device_id": "core-s3", "battery": 87},
                },
            ),
            ("POST", "http://body.local/volume"): _FakeResponse(
                status=200, payload={"ok": True, "value": 128}
            ),
            ("POST", "http://body.local/clear"): _FakeResponse(
                status=200, payload={"ok": True}
            ),
        },
    )
    meta = _meta(tmp_path)

    reach = await body_tools.exec_body(meta, body_tools.BodyInput(action="reach"))
    show = await body_tools.exec_body(
        meta, body_tools.BodyInput(action="show", text="hello from inside")
    )
    status = await body_tools.exec_body(meta, body_tools.BodyInput(action="status"))
    volume = await body_tools.exec_body(
        meta, body_tools.BodyInput(action="volume", value=128)
    )
    clear = await body_tools.exec_body(meta, body_tools.BodyInput(action="clear"))

    assert _text(reach.content[0]) == (
        "The ember swells and the chime sounds. If a palm answers, it will reach you "
        "as a message."
    )
    assert _text(show.content[0]) == 'The screen now says: "hello from inside"'
    assert _text(status.content[0]) == (
        "your body is connected; last heartbeat 2.1s ago. core-s3, battery 87%."
    )
    assert _text(volume.content[0]) == "Your body's chime volume is set to 128."
    assert _text(clear.content[0]) == "The screen is clear."

    assert _FakeAsyncClient.instances[1].posts == [
        ("http://body.local/display", {"text": "hello from inside"})
    ]
    assert _FakeAsyncClient.instances[2].gets == ["http://body.local/status"]
    assert _FakeAsyncClient.instances[3].posts == [
        ("http://body.local/volume", {"value": 128})
    ]


async def test_brainstem_unreachable_is_tool_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_client(
        monkeypatch,
        {("POST", "http://body.local/reach"): body_tools.httpx.ConnectError("refused")},
    )

    with pytest.raises(ToolError) as exc_info:
        await body_tools.exec_body(
            _meta(tmp_path), body_tools.BodyInput(action="reach")
        )

    assert exc_info.value.message == body_tools.NO_DEVICE_MESSAGE

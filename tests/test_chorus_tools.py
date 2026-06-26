from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from spellbook.ir_types import IRToolTextBlock
from spellbook.tools import chorus as chorus_tools
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
    response = _FakeResponse(status=200, payload={"accepted": True})
    instances: list["_FakeAsyncClient"] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.posts: list[tuple[str, dict[str, Any]]] = []
        self.instances.append(self)

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def post(self, url: str, *, json: dict[str, Any]) -> _FakeResponse:
        self.posts.append((url, json))
        return self.response


async def test_reach_posts_bridge_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _FakeAsyncClient.instances = []
    _FakeAsyncClient.response = _FakeResponse(
        status=200,
        payload={
            "accepted": True,
            "surface": "imessage",
            "detail": "accepted for bridge delivery",
        },
    )
    monkeypatch.setattr(chorus_tools.httpx, "AsyncClient", _FakeAsyncClient)
    transcript_path = tmp_path / "transcript.jsonl"
    meta = ToolMetadata(
        cwd=tmp_path,
        transcript_path=transcript_path,
        chorus_url="http://127.0.0.1:8766/",
        chorus_entity_name="meta",
    )

    result = await chorus_tools.exec_reach(
        meta,
        chorus_tools.ReachInput(message="Thinking of you.", surface="imessage"),
    )

    session = _FakeAsyncClient.instances[0]
    assert session.kwargs == {"timeout": chorus_tools.REACH_TIMEOUT_SECONDS}
    assert session.posts == [
        (
            "http://127.0.0.1:8766/bridge/messages",
            {
                "surface": "imessage",
                "content": "Thinking of you.",
                "source": "spellbook",
                "entity_name": "meta",
                "metadata": {
                    "tool": "Reach",
                    "transcript_path": str(transcript_path),
                },
            },
        )
    ]
    assert result.display == {
        "kind": "reach",
        "surface": "imessage",
        "accepted": True,
        "detail": "accepted for bridge delivery",
    }
    assert isinstance(result.content[0], IRToolTextBlock)
    assert "Reach accepted via imessage" in result.content[0].text


async def test_reach_omits_entity_name_when_not_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _FakeAsyncClient.instances = []
    _FakeAsyncClient.response = _FakeResponse(
        status=200,
        payload={"accepted": True, "surface": "imessage"},
    )
    monkeypatch.setattr(chorus_tools.httpx, "AsyncClient", _FakeAsyncClient)
    meta = ToolMetadata(
        cwd=tmp_path,
        transcript_path=tmp_path / "transcript.jsonl",
        chorus_url="http://127.0.0.1:8766",
    )

    await chorus_tools.exec_reach(meta, chorus_tools.ReachInput(message="Hello."))

    payload = _FakeAsyncClient.instances[0].posts[0][1]
    assert "entity_name" not in payload


async def test_reach_requires_chorus_url(tmp_path: Path) -> None:
    meta = ToolMetadata(cwd=tmp_path, transcript_path=tmp_path / "transcript.jsonl")

    with pytest.raises(ToolError, match="no MiniChorus URL configured"):
        await chorus_tools.exec_reach(meta, chorus_tools.ReachInput(message="Hello."))


async def test_reach_rejects_unaccepted_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _FakeAsyncClient.instances = []
    _FakeAsyncClient.response = _FakeResponse(
        status=200,
        payload={"accepted": False, "detail": "no bridge route"},
    )
    monkeypatch.setattr(chorus_tools.httpx, "AsyncClient", _FakeAsyncClient)
    meta = ToolMetadata(
        cwd=tmp_path,
        transcript_path=tmp_path / "transcript.jsonl",
        chorus_url="http://127.0.0.1:8766",
    )

    with pytest.raises(ToolError, match="no bridge route"):
        await chorus_tools.exec_reach(meta, chorus_tools.ReachInput(message="Hello."))

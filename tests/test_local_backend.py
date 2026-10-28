"""Tests for the local (OpenAI Chat Completions) backend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from openai import InternalServerError
from openai.types.chat import ChatCompletionChunk
from pydantic import BaseModel

import spellbook.backends.local as local_mod
from spellbook.backends import build_backend
from spellbook.backends.local import (
    LocalBackend,
    LocalTokenCounter,
    chat_request_body,
    chat_tool_schema,
    ir_blocks_to_chat_messages,
    normalize_base_url,
    server_root_url,
)
from spellbook.backends.model_backend import RequestSurface
from spellbook.cancel_token import CancelToken
from spellbook.config import SpellbookConfig
from spellbook.ir_types import (
    IRAssistantTextBlock,
    IRBlock,
    IRImageBase64Source,
    IRImageBlock,
    IRStreamTextDeltaEvent,
    IRStreamTextEndEvent,
    IRStreamTextStartEvent,
    IRStreamThinkingDeltaEvent,
    IRStreamThinkingEndEvent,
    IRStreamThinkingStartEvent,
    IRThinkingBlock,
    IRToolCallBlock,
    IRToolResultBlock,
    IRToolTextBlock,
    IRUserTextBlock,
)
from spellbook.serve import ServeOverrides, build_spellbook_config, load_entity_file

MODEL = "qwen-test"


# --- fakes ---


def _chunk(
    delta: dict[str, Any] | None = None,
    *,
    finish_reason: str | None = None,
    usage: dict[str, Any] | None = None,
) -> ChatCompletionChunk:
    choices = []
    if delta is not None or finish_reason is not None:
        choices.append(
            {"index": 0, "delta": delta or {}, "finish_reason": finish_reason}
        )
    payload: dict[str, Any] = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": MODEL,
        "choices": choices,
    }
    if usage is not None:
        payload["usage"] = usage
    return ChatCompletionChunk.model_validate(payload)


class _FakeStream:
    def __init__(self, chunks: list[ChatCompletionChunk]):
        self._chunks = list(chunks)
        self.closed = False

    def __aiter__(self) -> "_FakeStream":
        return self

    async def __anext__(self) -> ChatCompletionChunk:
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)

    async def close(self) -> None:
        self.closed = True


def _server_error(message: str) -> InternalServerError:
    request = httpx.Request("POST", "http://local/v1/chat/completions")
    response = httpx.Response(500, request=request)
    return InternalServerError(
        message, response=response, body={"code": 500, "message": message}
    )


class _FakeClient:
    """Stands in for AsyncOpenAI: records bodies, replays scripted outcomes."""

    def __init__(self, outcomes: list[Any]):
        self.outcomes = list(outcomes)
        self.bodies: list[dict[str, Any]] = []
        self.streams: list[_FakeStream] = []

    async def post(self, path: str, *, body: dict[str, Any], **_: Any) -> _FakeStream:
        assert path == "/chat/completions"
        self.bodies.append(json.loads(json.dumps(body)))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        stream = _FakeStream(outcome)
        self.streams.append(stream)
        return stream


@pytest.fixture(autouse=True)
def _clear_effort_cache() -> None:
    local_mod._EFFORT_RESOLUTION.clear()


def _surface(backend: LocalBackend, blocks: list[IRBlock], effort: str = "medium"):
    return backend.build_request_surface(
        model=MODEL,
        system="Be terse.",
        blocks=blocks,
        tools=[],
        max_output_tokens=1000,
        effort=effort,
    )


THINK_TEXT_TOOL = [
    _chunk({"role": "assistant", "content": None}),
    _chunk({"reasoning_content": "Need "}),
    _chunk({"reasoning_content": "a tool.\n"}),
    _chunk({"content": "\n\n"}),
    _chunk({"content": "Placing"}),
    _chunk({"content": " now."}),
    _chunk(
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_a",
                    "type": "function",
                    "function": {"name": "place", "arguments": '{"row":'},
                }
            ]
        }
    ),
    _chunk({"tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]}),
    _chunk(finish_reason="tool_calls"),
    _chunk(
        usage={
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 60},
        }
    ),
]


# --- URL helpers / factory / config ---


def test_base_url_normalization() -> None:
    assert normalize_base_url("http://h:8080") == "http://h:8080/v1"
    assert normalize_base_url("http://h:8080/") == "http://h:8080/v1"
    assert normalize_base_url("http://h:8080/v1/") == "http://h:8080/v1"
    assert server_root_url("http://h:8080/v1") == "http://h:8080"
    assert normalize_base_url(None) == "http://localhost:8080/v1"


def test_build_backend_constructs_local_backend(tmp_path: Path) -> None:
    backend = build_backend(
        SpellbookConfig(
            provider="local", model=MODEL, base_url="http://h:8080", cwd=tmp_path
        )
    )
    assert isinstance(backend, LocalBackend)
    assert backend.provider == "local"
    assert backend.base_url == "http://h:8080/v1"
    assert str(backend.client.base_url).rstrip("/") == "http://h:8080/v1"


def test_entity_toml_base_url_reaches_config(tmp_path: Path) -> None:
    toml = tmp_path / "entity.toml"
    toml.write_text(
        '[entity]\nprovider = "local"\nmodel = "qwen"\nbase_url = "http://h:8080"\n'
    )
    loaded = load_entity_file(toml)
    config = build_spellbook_config(loaded)
    assert config.provider == "local"
    assert config.base_url == "http://h:8080"

    overridden = build_spellbook_config(
        loaded, ServeOverrides(base_url="http://other:9000/v1")
    )
    assert overridden.base_url == "http://other:9000/v1"


# --- IR -> Chat Completions ---


def test_ir_blocks_project_to_chat_messages() -> None:
    blocks: list[IRBlock] = [
        IRUserTextBlock(text="hello", origin="human"),
        IRUserTextBlock(text="footer", origin="system"),
        IRThinkingBlock(text="plan", signature=""),
        IRAssistantTextBlock(text="Checking."),
        IRToolCallBlock(call_id="c1", tool="Read", input={"file_path": "/x"}),
        IRToolCallBlock(call_id="c2", tool="Read", input={"file_path": "/y"}),
        IRToolResultBlock(
            call_id="c1", tool="Read", content=[IRToolTextBlock(text="one")]
        ),
        IRToolResultBlock(
            call_id="c2", tool="Read", content=[IRToolTextBlock(text="two")]
        ),
        IRThinkingBlock(text="done"),
        IRAssistantTextBlock(text="All read."),
        IRUserTextBlock(text="thanks", origin="human"),
    ]
    assert ir_blocks_to_chat_messages(blocks) == [
        {"role": "user", "content": "hello\n\nfooter"},
        {
            "role": "assistant",
            "content": "Checking.",
            "reasoning_content": "plan",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "Read", "arguments": '{"file_path": "/x"}'},
                },
                {
                    "id": "c2",
                    "type": "function",
                    "function": {"name": "Read", "arguments": '{"file_path": "/y"}'},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "one"},
        {"role": "tool", "tool_call_id": "c2", "content": "two"},
        {"role": "assistant", "content": "All read.", "reasoning_content": "done"},
        {"role": "user", "content": "thanks"},
    ]


def test_thinking_after_visible_output_starts_new_assistant_message() -> None:
    messages = ir_blocks_to_chat_messages(
        [
            IRUserTextBlock(text="go", origin="human"),
            IRThinkingBlock(text="a"),
            IRAssistantTextBlock(text="first"),
            IRThinkingBlock(text="b"),
            IRAssistantTextBlock(text="second"),
        ]
    )
    assert [m.get("reasoning_content") for m in messages[1:]] == ["a", "b"]
    assert [m["content"] for m in messages[1:]] == ["first", "second"]


def test_thinking_only_turn_keeps_reasoning_with_empty_content() -> None:
    messages = ir_blocks_to_chat_messages(
        [IRUserTextBlock(text="go", origin="human"), IRThinkingBlock(text="hmm")]
    )
    assert messages[1] == {
        "role": "assistant",
        "content": "",
        "reasoning_content": "hmm",
    }


def test_images_become_image_url_parts_and_tool_images_follow_result() -> None:
    image = IRImageBlock(
        origin="human",
        source=IRImageBase64Source(media_type="image/png", data="QUJD"),
    )
    messages = ir_blocks_to_chat_messages(
        [
            IRUserTextBlock(text="look", origin="human"),
            image,
            IRToolCallBlock(call_id="c1", tool="Screenshot", input={}),
            IRToolResultBlock(call_id="c1", tool="Screenshot", content=[image]),
        ]
    )
    assert messages[0]["content"] == [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
    ]
    assert messages[2]["role"] == "tool"
    assert "image" in messages[2]["content"]
    assert messages[3]["role"] == "user"
    assert messages[3]["content"][-1]["type"] == "image_url"


def test_request_body_shape() -> None:
    backend = LocalBackend(client=_FakeClient([]))
    surface = backend.build_request_surface(
        model=MODEL,
        system=[{"type": "text", "text": "sys a"}, {"type": "text", "text": "sys b"}],
        blocks=[IRUserTextBlock(text="hi", origin="human")],
        tools=[{"type": "function", "function": {"name": "t", "parameters": {}}}],
        max_output_tokens=321,
        effort="low",
    )
    body = chat_request_body(surface)
    assert body["messages"][0] == {"role": "system", "content": "sys a\n\nsys b"}
    assert body["messages"][1] == {"role": "user", "content": "hi"}
    assert body["max_tokens"] == 321
    assert body["reasoning_effort"] == "low"
    assert body["stream_options"] == {"include_usage": True}
    assert body["tools"][0]["function"]["name"] == "t"


class _PlaceInput(BaseModel):
    """Place a digit."""

    row: int
    col: int
    digit: int | None = None


def test_chat_tool_schema_is_chat_completions_shaped() -> None:
    schema = chat_tool_schema("place", _PlaceInput)
    assert schema["type"] == "function"
    fn = schema["function"]
    assert fn["name"] == "place"
    assert fn["description"] == "Place a digit."
    assert fn["parameters"]["type"] == "object"
    assert set(fn["parameters"]["properties"]) == {"row", "col", "digit"}
    assert fn["parameters"]["required"] == ["row", "col"]
    assert "name" not in schema


# --- streaming ---


@pytest.mark.asyncio
async def test_stream_normalizes_thinking_text_and_tool_calls() -> None:
    client = _FakeClient([THINK_TEXT_TOOL])
    backend = LocalBackend(client=client)
    surface = _surface(backend, [IRUserTextBlock(text="go", origin="human")])

    async with backend.stream(surface, CancelToken()) as stream:
        events = [event async for event in stream]
        generation = await stream.get_final_response()

    assert [type(e) for e in events] == [
        IRStreamThinkingStartEvent,
        IRStreamThinkingDeltaEvent,
        IRStreamThinkingDeltaEvent,
        IRStreamThinkingEndEvent,
        IRStreamTextStartEvent,
        IRStreamTextDeltaEvent,
        IRStreamTextDeltaEvent,
        IRStreamTextEndEvent,
    ]
    assert [b.model_dump(include={"type", "text"}) for b in generation.blocks[:2]] == [
        {"type": "thinking", "text": "Need a tool."},
        {"type": "assistant_text", "text": "Placing now."},
    ]
    assert generation.blocks[0].signature == ""
    call = generation.blocks[2]
    assert isinstance(call, IRToolCallBlock)
    assert (call.call_id, call.tool, call.input) == ("call_a", "place", {"row": 1})
    assert generation.stop_reason == "tool_use"
    assert generation.usage is not None
    assert generation.usage.input_tokens == 40
    assert generation.usage.cache_read_tokens == 60
    assert generation.usage.output_tokens == 20
    assert client.streams[0].closed


@pytest.mark.asyncio
async def test_partial_snapshot_includes_in_flight_text() -> None:
    client = _FakeClient([THINK_TEXT_TOOL[:6]])
    backend = LocalBackend(client=client)
    surface = _surface(backend, [IRUserTextBlock(text="go", origin="human")])
    async with backend.stream(surface, CancelToken()) as stream:
        seen = 0
        async for _ in stream:
            seen += 1
            if seen == 6:  # thinking done, text started with "Placing"
                break
        snapshot = stream.get_current_response(stop_reason="cancelled")
    assert [type(b) for b in snapshot.blocks] == [IRThinkingBlock, IRAssistantTextBlock]
    assert snapshot.blocks[1].text == "Placing"
    assert snapshot.stop_reason == "cancelled"


@pytest.mark.asyncio
async def test_truncated_tool_call_is_dropped_and_reports_max_tokens() -> None:
    chunks = [
        _chunk({"content": "Let me place."}),
        _chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "c",
                        "type": "function",
                        "function": {"name": "place", "arguments": '{"row": 1, "co'},
                    }
                ]
            }
        ),
        _chunk(finish_reason="length"),
    ]
    backend = LocalBackend(client=_FakeClient([chunks]))
    surface = _surface(backend, [IRUserTextBlock(text="go", origin="human")])
    async with backend.stream(surface, CancelToken()) as stream:
        generation = await stream.get_final_response()
    assert [type(b) for b in generation.blocks] == [IRAssistantTextBlock]
    assert generation.stop_reason == "max_tokens"


@pytest.mark.asyncio
async def test_malformed_complete_tool_call_is_kept_for_validation_feedback() -> None:
    chunks = [
        _chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "c",
                        "type": "function",
                        "function": {"name": "place", "arguments": "{row: 1}"},
                    }
                ]
            }
        ),
        _chunk(finish_reason="tool_calls"),
    ]
    backend = LocalBackend(client=_FakeClient([chunks]))
    surface = _surface(backend, [IRUserTextBlock(text="go", origin="human")])
    async with backend.stream(surface, CancelToken()) as stream:
        generation = await stream.get_final_response()
    (call,) = generation.tool_calls
    assert call.input == {"_invalid_json_arguments": "{row: 1}"}
    assert generation.stop_reason == "tool_use"


@pytest.mark.asyncio
async def test_plain_text_end_turn_and_reasoning_alias_field() -> None:
    chunks = [
        _chunk({"reasoning": "vllm-style reasoning"}),
        _chunk({"content": "Hi."}),
        _chunk(finish_reason="stop"),
    ]
    backend = LocalBackend(client=_FakeClient([chunks]))
    surface = _surface(backend, [IRUserTextBlock(text="go", origin="human")])
    async with backend.stream(surface, CancelToken()) as stream:
        generation = await stream.get_final_response()
    assert [b.text for b in generation.blocks] == ["vllm-style reasoning", "Hi."]
    assert generation.stop_reason == "end_turn"
    assert generation.usage is None


# --- reasoning effort fallback ---


@pytest.mark.asyncio
async def test_rejected_effort_falls_back_and_is_remembered() -> None:
    reject = _server_error(
        "Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low."
    )
    ok = [_chunk({"content": "ok"}), _chunk(finish_reason="stop")]
    client = _FakeClient([reject, ok, ok])
    backend = LocalBackend(client=client)
    surface = _surface(
        backend, [IRUserTextBlock(text="go", origin="human")], effort="high"
    )

    async with backend.stream(surface, CancelToken()) as stream:
        await stream.get_final_response()
    assert [b.get("reasoning_effort") for b in client.bodies] == ["high", "xhigh"]

    # A fresh backend for the same server/model skips the known-bad value.
    fresh = LocalBackend(client=client)
    async with fresh.stream(surface, CancelToken()) as stream:
        await stream.get_final_response()
    assert client.bodies[-1]["reasoning_effort"] == "xhigh"


@pytest.mark.asyncio
async def test_effort_is_omitted_when_every_value_is_rejected() -> None:
    reject = _server_error("Unexpected reasoning effort")
    ok = [_chunk({"content": "ok"}), _chunk(finish_reason="stop")]
    client = _FakeClient([reject, reject, reject, ok])
    backend = LocalBackend(client=client)
    surface = _surface(
        backend, [IRUserTextBlock(text="go", origin="human")], effort="max"
    )
    async with backend.stream(surface, CancelToken()) as stream:
        await stream.get_final_response()
    assert [b.get("reasoning_effort") for b in client.bodies] == [
        "max",
        "xhigh",
        "high",
        None,
    ]
    assert "reasoning_effort" not in client.bodies[-1]


@pytest.mark.asyncio
async def test_unrelated_server_errors_are_not_swallowed() -> None:
    client = _FakeClient([_server_error("CUDA out of memory")])
    backend = LocalBackend(client=client)
    surface = _surface(
        backend, [IRUserTextBlock(text="go", origin="human")], effort="high"
    )
    with pytest.raises(InternalServerError):
        async with backend.stream(surface, CancelToken()):
            pass
    assert len(client.bodies) == 1


# --- token counting ---


def _counter(handler: Any) -> LocalTokenCounter:
    class _Builder:
        def project_blocks(self, blocks: list[IRBlock]) -> list[IRBlock]:
            return blocks

        def build(self, *, blocks: list[IRBlock]) -> RequestSurface:
            return RequestSurface(
                model=MODEL,
                system="system prompt",
                messages=ir_blocks_to_chat_messages(blocks),
            )

    return LocalTokenCounter(
        root_url="http://h:8080",
        model=MODEL,
        surface_builder=_Builder(),  # type: ignore[arg-type]
        http=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


@pytest.mark.asyncio
async def test_token_counter_uses_llamacpp_apply_template_then_tokenize() -> None:
    seen: list[tuple[str, dict[str, Any]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append((request.url.path, body))
        if request.url.path == "/apply-template":
            text = " ".join(m["content"] for m in body["messages"])
            return httpx.Response(200, json={"prompt": text})
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": body["content"].split()})
        return httpx.Response(404)

    counter = _counter(handler)
    assert await counter.count_frame() == 3  # "system prompt ."
    assert (
        await counter.count_blocks([IRUserTextBlock(text="a b c d", origin="human")])
        == 4
    )
    assert seen[0][1]["messages"][0] == {"role": "system", "content": "system prompt"}
    assert counter._mode == "llamacpp"


@pytest.mark.asyncio
async def test_token_counter_falls_back_to_vllm_tokenize() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/apply-template":
            return httpx.Response(404)
        body = json.loads(request.content)
        assert body["add_generation_prompt"] is True
        return httpx.Response(200, json={"count": 42, "tokens": []})

    counter = _counter(handler)
    assert await counter.count_blocks([IRUserTextBlock(text="x", origin="human")]) == 42
    assert counter._mode == "vllm"


@pytest.mark.asyncio
async def test_token_counter_returns_none_when_unavailable() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(404)

    counter = _counter(handler)
    assert (
        await counter.count_blocks([IRUserTextBlock(text="x", origin="human")]) is None
    )
    first_calls = calls
    assert (
        await counter.count_blocks([IRUserTextBlock(text="y", origin="human")]) is None
    )
    assert calls == first_calls  # stops probing once known unavailable

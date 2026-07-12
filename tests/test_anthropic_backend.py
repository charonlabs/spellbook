from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from anthropic import AsyncAnthropic
import pytest

from spellbook.backends.anthropic import AnthropicBackend, AnthropicGenerationStream
from spellbook.backends.model_backend import RequestSurface
from spellbook.cancel_token import CancelToken
from spellbook.ir_types import (
    IRGeneration,
    IRRefusalBlock,
    IRStreamTextDeltaEvent,
    IRStreamTextEndEvent,
    IRStreamTextStartEvent,
    IRStreamThinkingDeltaEvent,
    IRStreamThinkingEndEvent,
    IRStreamThinkingStartEvent,
)


class _FakeAnthropicStreamContext:
    def __init__(self, events: list[object], final_message: object) -> None:
        self._events = events
        self._final_message = final_message

    async def __aenter__(self) -> "_FakeAnthropicStream":
        return _FakeAnthropicStream(self._events, self._final_message)

    async def __aexit__(self, *exc: Any) -> None:
        return None


class _FakeAnthropicStream:
    def __init__(self, events: list[object], final_message: object) -> None:
        self._events = events
        self._final_message = final_message
        self._idx = 0
        self.current_message_snapshot = final_message

    def __aiter__(self) -> "_FakeAnthropicStream":
        return self

    async def __anext__(self) -> object:
        if self._idx >= len(self._events):
            raise StopAsyncIteration
        event = self._events[self._idx]
        self._idx += 1
        return event

    async def get_final_message(self) -> object:
        return self._final_message


def _usage() -> object:
    return SimpleNamespace(
        input_tokens=10,
        output_tokens=5,
        cache_read_input_tokens=2,
        cache_creation_input_tokens=3,
    )


def _refusal_message(*, category: str = "cyber", explanation: str = "Nope.") -> object:
    return SimpleNamespace(
        content=[],
        stop_reason="refusal",
        stop_details=SimpleNamespace(
            type="refusal",
            category=category,
            explanation=explanation,
        ),
        usage=_usage(),
    )


def _block_start(kind: str) -> object:
    return SimpleNamespace(
        type="content_block_start",
        content_block=SimpleNamespace(type=kind),
    )


def _block_stop() -> object:
    return SimpleNamespace(type="content_block_stop")


def _text_delta(text: str) -> object:
    return SimpleNamespace(
        type="content_block_delta",
        delta=SimpleNamespace(type="text_delta", text=text),
    )


def _thinking_delta(text: str) -> object:
    return SimpleNamespace(
        type="content_block_delta",
        delta=SimpleNamespace(type="thinking_delta", thinking=text),
    )


def _tool_json_delta(text: str) -> object:
    return SimpleNamespace(
        type="content_block_delta",
        delta=SimpleNamespace(type="input_json_delta", partial_json=text),
    )


async def _drain_stream(stream: AnthropicGenerationStream) -> IRGeneration:
    async with stream:
        while True:
            try:
                await stream.__anext__()
            except StopAsyncIteration:
                break
        return await stream.get_final_response()


@pytest.mark.asyncio
async def test_anthropic_refusal_after_text_appends_refusal_debug_block() -> None:
    stream = AnthropicGenerationStream(
        cast(
            Any,
            _FakeAnthropicStreamContext(
                [
                    _block_start("text"),
                    _text_delta("I can explain the boundary, "),
                    _text_delta("but not that part."),
                    _block_stop(),
                ],
                _refusal_message(
                    category="reasoning_extraction",
                    explanation="The request asks for hidden reasoning.",
                ),
            ),
        ),
        model="claude-fable-5",
    )

    async with stream:
        assert isinstance(await stream.__anext__(), IRStreamTextStartEvent)
        delta = await stream.__anext__()
        assert isinstance(delta, IRStreamTextDeltaEvent)
        assert delta.text == "I can explain the boundary, "
        delta = await stream.__anext__()
        assert isinstance(delta, IRStreamTextDeltaEvent)
        assert delta.text == "but not that part."
        assert isinstance(await stream.__anext__(), IRStreamTextEndEvent)
        with pytest.raises(StopAsyncIteration):
            await stream.__anext__()
        final = await stream.get_final_response()

    assert final.stop_reason == "refusal"
    assert len(final.blocks) == 1
    block = final.blocks[0]
    assert isinstance(block, IRRefusalBlock)
    assert block.partial_text == "I can explain the boundary, but not that part."
    assert block.details is not None
    assert block.details.category == "reasoning_extraction"
    assert block.details.explanation == "The request asks for hidden reasoning."
    assert final.usage is not None
    assert final.usage.input_tokens == 10
    assert final.usage.cache_read_tokens == 2
    assert final.usage.cache_create_tokens == 3
    assert final.usage.output_tokens == 5


@pytest.mark.asyncio
async def test_anthropic_refusal_after_thinking_collapses_summary_to_text() -> None:
    stream = AnthropicGenerationStream(
        cast(
            Any,
            _FakeAnthropicStreamContext(
                [
                    _block_start("thinking"),
                    _thinking_delta("I considered whether this is allowed."),
                    _block_stop(),
                ],
                _refusal_message(explanation="Safety policy blocked the answer."),
            ),
        ),
        model="claude-fable-5",
    )

    async with stream:
        assert isinstance(await stream.__anext__(), IRStreamThinkingStartEvent)
        delta = await stream.__anext__()
        assert isinstance(delta, IRStreamThinkingDeltaEvent)
        assert delta.text == "I considered whether this is allowed."
        assert isinstance(await stream.__anext__(), IRStreamThinkingEndEvent)
        with pytest.raises(StopAsyncIteration):
            await stream.__anext__()
        final = await stream.get_final_response()

    assert final.stop_reason == "refusal"
    assert len(final.blocks) == 1
    block = final.blocks[0]
    assert isinstance(block, IRRefusalBlock)
    assert [(segment.kind, segment.text) for segment in block.segments] == [
        ("thinking_summary", "I considered whether this is allowed.")
    ]
    assert block.details is not None
    assert block.details.explanation == "Safety policy blocked the answer."


@pytest.mark.asyncio
async def test_anthropic_refusal_during_tool_call_records_partial_json_text() -> None:
    stream = AnthropicGenerationStream(
        cast(
            Any,
            _FakeAnthropicStreamContext(
                [
                    _block_start("tool_use"),
                    _tool_json_delta('{"path": "/tmp/se'),
                    _tool_json_delta('cret.txt", "mode": "'),
                ],
                _refusal_message(category="cyber", explanation="Tool call refused."),
            ),
        ),
        model="claude-fable-5",
    )

    final = await _drain_stream(stream)

    assert final.stop_reason == "refusal"
    assert len(final.blocks) == 1
    block = final.blocks[0]
    assert isinstance(block, IRRefusalBlock)
    assert [(segment.kind, segment.text) for segment in block.segments] == [
        ("partial_tool_call_json", '{"path": "/tmp/secret.txt", "mode": "')
    ]
    assert block.details is not None
    assert block.details.explanation == "Tool call refused."


class _FakeMessages:
    def __init__(self) -> None:
        self.stream_called = False
        self.post_calls: list[dict[str, Any]] = []

    def stream(self, **kwargs: Any) -> object:
        self.stream_called = True
        raise AssertionError("messages.stream should not be called")

    def _post(self, *args: Any, **kwargs: Any) -> object:
        self.post_calls.append({"args": args, "kwargs": kwargs})
        return _FakeAwaitable()


class _FakeAwaitable:
    def __await__(self) -> Any:
        if False:
            yield None
        return object()


class _FakeAnthropicClient:
    def __init__(self) -> None:
        self.messages = _FakeMessages()


def test_anthropic_stream_bypasses_typed_transform_helper() -> None:
    client = _FakeAnthropicClient()
    backend = AnthropicBackend(client=cast(AsyncAnthropic, client))
    surface = RequestSurface(
        model="claude-sonnet-4-6",
        system="system prompt",
        messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        tools=[
            {
                "name": "ReadConstellation",
                "description": "Read a tiny note.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            }
        ],
        thinking={"type": "adaptive", "display": "summarized"},
        output_config={"effort": "high"},
        cache_control={"type": "ephemeral", "ttl": "1h"},
        max_output_tokens=1024,
    )

    backend.stream(surface, CancelToken())

    assert client.messages.stream_called is False
    assert len(client.messages.post_calls) == 1
    call = client.messages.post_calls[0]
    assert call["args"] == ("/v1/messages",)
    body = call["kwargs"]["body"]
    assert body["model"] == "claude-sonnet-4-6"
    assert body["messages"] == surface.messages
    assert body["system"] == "system prompt"
    assert body["tools"] == surface.tools
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert body["output_config"] == {"effort": "high"}
    assert body["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert body["stream"] is True
    assert call["kwargs"]["stream"] is True

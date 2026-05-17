from __future__ import annotations

from typing import Any, cast

from anthropic import AsyncAnthropic

from spellbook.backends.anthropic import AnthropicBackend
from spellbook.backends.model_backend import RequestSurface
from spellbook.cancel_token import CancelToken


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

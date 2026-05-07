"""Tests for OpenAI Responses backend translation and counting."""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import BaseModel, Field

from spellbook.backends.model_backend import RequestSurface
from spellbook.backends.openai import (
    OpenAIBackend,
    OpenAIGenerationStream,
    OpenAITokenCounter,
    _ir_blocks_to_response_input_items,
    _normalize_content_blocks,
)
from spellbook.config import SpellbookConfig
from spellbook.ir_types import (
    IRAssistantTextBlock,
    IRBlock,
    IRImageBase64Source,
    IRImageBlock,
    IRImageURLSource,
    IRStreamTextDeltaEvent,
    IRStreamTextEndEvent,
    IRStreamTextStartEvent,
    IRThinkingBlock,
    IRToolCallBlock,
    IRToolResultBlock,
    IRToolTextBlock,
    IRUserTextBlock,
)
from spellbook.surface_builder import RequestSurfaceBuilder
from spellbook.tools.common import Tool, ToolExecutionResult, ToolMetadata
from spellbook.tools.registry import ToolRegistry


class TestOpenAIBlockTranslation:
    def test_ir_blocks_translate_to_responses_input_items_in_order(self) -> None:
        blocks: list[IRBlock] = [
            IRUserTextBlock(text="hello", origin="human"),
            IRThinkingBlock(text="plan", signature="enc_123"),
            IRAssistantTextBlock(text="Checking.", origin="model"),
            IRToolCallBlock(
                call_id="call_1",
                tool="Read",
                input={"file_path": "/tmp/demo.py"},
            ),
            IRToolResultBlock(
                call_id="call_1",
                tool="Read",
                content=[IRToolTextBlock(text="result")],
            ),
            IRUserTextBlock(text="one more thing", origin="human"),
        ]

        assert _ir_blocks_to_response_input_items(blocks) == [
            {"role": "user", "content": [{"type": "input_text", "text": "hello"}]},
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "plan"}],
                "encrypted_content": "enc_123",
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Checking."}],
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "Read",
                "arguments": '{"file_path":"/tmp/demo.py"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "result",
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "one more thing"}],
            },
        ]

    def test_ir_images_translate_to_openai_image_parts(self) -> None:
        blocks: list[IRBlock] = [
            IRUserTextBlock(text="look", origin="human"),
            IRImageBlock(
                origin="human",
                source=IRImageURLSource(url="https://example.com/cat.png"),
            ),
            IRImageBlock(
                origin="human",
                source=IRImageBase64Source(
                    media_type="image/png",
                    data="abc123",
                ),
            ),
        ]

        assert _ir_blocks_to_response_input_items(blocks) == [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "look"},
                    {
                        "type": "input_image",
                        "image_url": "https://example.com/cat.png",
                    },
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,abc123",
                    },
                ],
            }
        ]

    def test_tool_result_with_images_uses_content_array_output(self) -> None:
        block = IRToolResultBlock(
            call_id="call_1",
            tool="Read",
            content=[
                IRToolTextBlock(text="image:"),
                IRImageBlock(
                    origin="tool",
                    source=IRImageURLSource(url="https://example.com/out.png"),
                ),
            ],
        )

        assert _ir_blocks_to_response_input_items([block]) == [
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": [
                    {"type": "input_text", "text": "image:"},
                    {
                        "type": "input_image",
                        "image_url": "https://example.com/out.png",
                    },
                ],
            }
        ]

    def test_normalize_openai_output_items_to_ir_blocks(self) -> None:
        output = [
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "plan"}],
                "encrypted_content": "enc_123",
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "Read",
                "arguments": '{"file_path":"/tmp/demo.py"}',
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Done"},
                    {"type": "output_text", "text": "."},
                ],
            },
        ]

        blocks, has_tool_call = _normalize_content_blocks(output)

        assert has_tool_call is True
        assert len(blocks) == 3
        assert isinstance(blocks[0], IRThinkingBlock)
        assert blocks[0].text == "plan"
        assert blocks[0].signature == "enc_123"
        assert isinstance(blocks[1], IRToolCallBlock)
        assert blocks[1].call_id == "call_1"
        assert blocks[1].tool == "Read"
        assert blocks[1].input == {"file_path": "/tmp/demo.py"}
        assert isinstance(blocks[2], IRAssistantTextBlock)
        assert blocks[2].text == "Done."


class _NestedToolInput(BaseModel):
    """Nested test tool."""

    path: str = Field(description="A path")
    options: dict[str, str] | None = None


async def _fake_exec(
    meta: ToolMetadata, input: _NestedToolInput
) -> ToolExecutionResult:
    return ToolExecutionResult(content=[])


FAKE_TOOL: Tool[_NestedToolInput] = Tool(
    name="FakeRead",
    input_model=_NestedToolInput,
    exec=_fake_exec,
    category="filesystem",
)


class TestOpenAIToolSchemaAndSurface:
    def test_openai_backend_generates_function_tool_schema(self) -> None:
        backend = OpenAIBackend(client=_FakeOpenAIClient())
        schemas = backend.build_tool_schemas(ToolRegistry(tools=[FAKE_TOOL]))

        assert schemas == [
            {
                "type": "function",
                "name": "FakeRead",
                "description": "Nested test tool.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"description": "A path", "type": "string"},
                        "options": {
                            "additionalProperties": {"type": "string"},
                            "type": "object",
                        },
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
                "strict": False,
            }
        ]

    def test_build_request_surface_translates_blocks_and_config(self) -> None:
        backend = OpenAIBackend(client=_FakeOpenAIClient())

        surface = backend.build_request_surface(
            model="gpt-5.5",
            system=[{"text": "core"}, {"content": "extra"}],
            blocks=[
                IRUserTextBlock(text="hello", origin="human"),
                IRAssistantTextBlock(text="hi", origin="model"),
            ],
            tools=[{"type": "function", "name": "FakeRead"}],
            max_output_tokens=777,
            effort="medium",
        )

        assert surface.model == "gpt-5.5"
        assert surface.system == "core\n\nextra"
        assert surface.messages == [
            {"role": "user", "content": [{"type": "input_text", "text": "hello"}]},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hi"}],
            },
        ]
        assert surface.tools == [{"type": "function", "name": "FakeRead"}]
        assert surface.thinking == {"effort": "medium", "summary": "detailed"}
        assert surface.max_output_tokens == 777


class _FakeInputTokens:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def count(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(input_tokens=321)


class _FakeResponses:
    def __init__(self) -> None:
        self.input_tokens = _FakeInputTokens()
        self.stream_calls: list[dict[str, Any]] = []

    def stream(self, **kwargs: Any) -> "_FakeStreamContext":
        self.stream_calls.append(kwargs)
        return _FakeStreamContext([])


class _FakeOpenAIClient:
    def __init__(self) -> None:
        self.responses = _FakeResponses()


class _CountingBackend(OpenAIBackend):
    def build_tool_schemas(self, registry: ToolRegistry) -> list[dict[str, Any]]:
        return []


class TestOpenAITokenCounter:
    @pytest.mark.asyncio
    async def test_count_blocks_uses_responses_input_token_endpoint(
        self, tmp_path
    ) -> None:
        client = _FakeOpenAIClient()
        backend = _CountingBackend(client=client)
        config = SpellbookConfig(
            provider="openai",
            model="gpt-5.5",
            cwd=tmp_path,
        )
        builder = RequestSurfaceBuilder.from_config(
            backend=backend,
            config=config,
            tool_registry=ToolRegistry(tools=[]),
        )
        counter = OpenAITokenCounter(
            client=client,
            model="gpt-5.5",
            surface_builder=builder,
        )

        count = await counter.count_blocks(
            [IRUserTextBlock(text="hello", origin="human")]
        )

        assert count == 321
        assert client.responses.input_tokens.calls == [
            {
                "model": "gpt-5.5",
                "input": [
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": "hello"}],
                    }
                ],
                "text": {"format": {"type": "text"}},
                "truncation": "disabled",
            }
        ]

    @pytest.mark.asyncio
    async def test_count_surface_includes_instructions_tools_and_reasoning(
        self,
    ) -> None:
        client = _FakeOpenAIClient()
        counter = OpenAITokenCounter(
            client=client,
            model="gpt-5.5",
            surface_builder=cast(RequestSurfaceBuilder, _FakeSurfaceBuilder()),
        )
        surface = RequestSurface(
            model="gpt-5.5",
            system="system",
            tools=[{"type": "function", "name": "Read"}],
            messages=[{"role": "user", "content": "hello"}],
            thinking={"effort": "high"},
        )

        assert await counter.count_surface(surface) == 321

        assert client.responses.input_tokens.calls == [
            {
                "model": "gpt-5.5",
                "input": [{"role": "user", "content": "hello"}],
                "text": {"format": {"type": "text"}},
                "instructions": "system",
                "tools": [{"type": "function", "name": "Read"}],
                "parallel_tool_calls": False,
                "reasoning": {"effort": "high"},
                "truncation": "disabled",
            }
        ]


class _FakeSurfaceBuilder:
    def build(self, blocks: Sequence[IRBlock]) -> RequestSurface:
        return RequestSurface(model="gpt-5.5", messages=[])


class _FakeStream:
    def __init__(self, events: list[Any], final_response: Any | None = None):
        self._events = events
        self._final_response = final_response or SimpleNamespace(
            status="completed",
            output=[],
            usage=None,
        )
        self._idx = 0

    def __aiter__(self) -> "_FakeStream":
        return self

    async def __anext__(self) -> Any:
        if self._idx >= len(self._events):
            raise StopAsyncIteration
        event = self._events[self._idx]
        self._idx += 1
        return event

    async def get_final_response(self) -> Any:
        return self._final_response


class _FakeStreamContext:
    def __init__(self, events: list[Any], final_response: Any | None = None):
        self._stream = _FakeStream(events, final_response)

    async def __aenter__(self) -> _FakeStream:
        return self._stream

    async def __aexit__(self, *exc: Any) -> None:
        return None


class TestOpenAIStream:
    @pytest.mark.asyncio
    async def test_stream_events_and_partial_snapshot_are_normalized(self) -> None:
        final_response = SimpleNamespace(
            status="completed",
            output=[
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hello"}],
                }
            ],
            usage=SimpleNamespace(
                input_tokens=15,
                output_tokens=4,
                input_tokens_details=SimpleNamespace(cached_tokens=5),
            ),
        )
        stream = OpenAIGenerationStream(
            _FakeStreamContext(
                [
                    SimpleNamespace(
                        type="response.output_item.added",
                        item=SimpleNamespace(type="message"),
                    ),
                    SimpleNamespace(type="response.output_text.delta", delta="Hel"),
                    SimpleNamespace(
                        type="response.output_item.done",
                        item=SimpleNamespace(type="message"),
                    ),
                    SimpleNamespace(
                        type="response.completed",
                        response=final_response,
                    ),
                ],
                final_response,
            ),
            model="gpt-5.5",
        )

        async with stream:
            assert isinstance(await stream.__anext__(), IRStreamTextStartEvent)
            delta = await stream.__anext__()
            assert isinstance(delta, IRStreamTextDeltaEvent)
            assert delta.text == "Hel"

            partial = stream.get_current_response(stop_reason="cancelled")
            assert partial.stop_reason == "cancelled"
            assert len(partial.blocks) == 1
            assert isinstance(partial.blocks[0], IRAssistantTextBlock)
            assert partial.blocks[0].text == "Hel"

            assert isinstance(await stream.__anext__(), IRStreamTextEndEvent)
            with pytest.raises(StopAsyncIteration):
                await stream.__anext__()

        final = await stream.get_final_response()

        assert final.stop_reason == "end_turn"
        assert len(final.blocks) == 1
        assert isinstance(final.blocks[0], IRAssistantTextBlock)
        assert final.blocks[0].text == "Hello"
        assert final.usage is not None
        assert final.usage.input_tokens == 10
        assert final.usage.cache_read_tokens == 5
        assert final.usage.output_tokens == 4

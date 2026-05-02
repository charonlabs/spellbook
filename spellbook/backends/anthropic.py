"""Anthropic model backend for Spellbook.

Implements the ``ModelBackend`` protocol for Anthropic's Messages API.
Handles streaming, content block normalization, usage extraction,
and request token counting.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

from anthropic import AsyncAnthropic
from anthropic._types import NotGiven
from anthropic.lib.streaming import AsyncMessageStreamManager
from anthropic.types import (
    Base64ImageSourceParam,
    ImageBlockParam,
    MessageParam,
    ParsedContentBlock,
    ParsedMessage,
    ParsedTextBlock,
    TextBlockParam,
    ThinkingBlock,
    ThinkingBlockParam,
    ToolResultBlockParam,
    ToolUseBlock,
    ToolUseBlockParam,
    URLImageSourceParam,
    Usage,
)
from pydantic import BaseModel

from spellbook.config import SpellbookConfig
from spellbook.surface_builder import RequestSurfaceBuilder

from ..cancel_token import CancelToken
from ..ir_types import (
    IRAssistantTextBlock,
    IRBlock,
    IRGeneration,
    IRImageBase64Source,
    IRImageBlobSource,
    IRImageBlock,
    IRImageURLSource,
    IRStreamEvent,
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
    IRUsage,
    IRUserTextBlock,
    StopReason,
)
from ..tools.common import TOOL_DESCS_DIR
from ..tools.registry import ToolRegistry
from .model_backend import GenerationStream, ModelBackend, RequestSurface, TokenCounter


class AnthropicGenerationStream(GenerationStream):
    """Wraps Anthropic's streaming response into normalized StreamEvents."""

    def __init__(self, stream_ctx: AsyncMessageStreamManager[NotGiven], model: str):
        self._stream_ctx = stream_ctx
        self._model = model
        self._stream = None
        self._response: ParsedMessage[NotGiven] | None = None
        self._exhausted = False
        self._in_thinking = False
        self._in_text = False

    async def __aenter__(self) -> "AnthropicGenerationStream":
        self._stream = await self._stream_ctx.__aenter__()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return await self._stream_ctx.__aexit__(*exc)

    def __aiter__(self) -> "AnthropicGenerationStream":
        return self

    async def __anext__(self) -> IRStreamEvent:
        if self._exhausted or self._stream is None:
            raise StopAsyncIteration

        async for event in self._stream:
            normalized = self._normalize_event(event)
            if normalized is not None:
                return normalized

        # Stream exhausted — grab the final message
        self._response = await self._stream.get_final_message()
        self._exhausted = True
        raise StopAsyncIteration

    async def get_final_response(self) -> IRGeneration:
        """Return the complete response after iteration."""
        if not self._exhausted:
            async for _ in self:
                pass

        if self._response is None:
            raise ValueError("somehow _response is None?")

        blocks, has_tool_use = _normalize_content_blocks(self._response.content)
        usage = _normalize_usage(self._response.usage)
        stop_reason = self._response.stop_reason
        if stop_reason is None:
            if has_tool_use:
                stop_reason = "tool_use"
            else:
                stop_reason = "unspecified"
        return IRGeneration(
            model=self._model,
            blocks=blocks,
            stop_reason=stop_reason,
            usage=usage,
        )

    def get_current_response(self, *, stop_reason: StopReason) -> IRGeneration:
        if self._stream is None:
            return IRGeneration(
                model=self._model,
                blocks=[],
                stop_reason=stop_reason,
                usage=None,
            )

        try:
            snapshot = self._stream.current_message_snapshot
        except AssertionError:
            return IRGeneration(
                model=self._model,
                blocks=[],
                stop_reason=stop_reason,
                usage=None,
            )

        blocks, _ = _normalize_partial_content_blocks(snapshot.content)
        return IRGeneration(
            model=self._model,
            blocks=blocks,
            stop_reason=stop_reason,
            usage=_normalize_usage(snapshot.usage),
        )

    def _normalize_event(self, event: Any) -> IRStreamEvent | None:  # noqa: ANN401
        if event.type == "content_block_start":
            if event.content_block.type == "thinking":
                self._in_thinking = True
                return IRStreamThinkingStartEvent()
            elif event.content_block.type == "text":
                self._in_text = True
                return IRStreamTextStartEvent()
        elif event.type == "content_block_delta":
            if event.delta.type == "thinking_delta":
                return IRStreamThinkingDeltaEvent(text=event.delta.thinking)
            elif event.delta.type == "text_delta":
                return IRStreamTextDeltaEvent(text=event.delta.text)
        elif event.type == "content_block_stop":
            if self._in_thinking:
                self._in_thinking = False
                return IRStreamThinkingEndEvent()
            elif self._in_text:
                self._in_text = False
                return IRStreamTextEndEvent()
        return None


class AnthropicTokenCounter(TokenCounter):
    """TokenCounter implementation for Anthropic models."""

    def __init__(
        self,
        *,
        client: AsyncAnthropic,
        model: str,
        surface_builder: RequestSurfaceBuilder,
    ):
        self._client = client
        self._model = model
        self._builder = surface_builder

    async def count_block_content(self, block: IRBlock) -> int | None:
        """Best-effort token count for one block's content.

        The block is sometimes wrapped in a minimal valid message structure (e.g.
        `.`-padded surrounding messages for blocks that can't stand alone) so the
        count_tokens API accepts it. The returned count therefore includes small
        envelope overhead (~5-10 tokens) that wouldn't exist in a combined request.

        For precise sums, prefer count_blocks on the full sequence. For individual
        block analysis (auto-TTL decisions, retroactive preview), the slight
        overestimate is negligible.
        """
        blocks: list[IRBlock] = []
        match block:
            case IRToolCallBlock():
                blocks.append(IRUserTextBlock(text=".", origin="human"))
                blocks.append(block)
                blocks.append(
                    IRToolResultBlock(
                        call_id=block.call_id,
                        tool=block.tool,
                        content=[IRToolTextBlock(text=".")],
                    )
                )
            case IRToolResultBlock():
                for cb in block.content:
                    match cb:
                        case IRToolTextBlock():
                            blocks.append(IRUserTextBlock(text=cb.text, origin="human"))
                        case IRImageBlock():
                            blocks.append(cb)
            case IRAssistantTextBlock():
                blocks.append(IRUserTextBlock(text=block.text, origin="human"))
            case IRThinkingBlock():
                blocks.append(IRUserTextBlock(text=".", origin="human"))
                blocks.append(block)
                blocks.append(IRAssistantTextBlock(text=".", origin="model"))
            case _:
                blocks.append(block)
        return await self.count_blocks(blocks)

    async def count_blocks(self, blocks: list[IRBlock]) -> int | None:
        msgs = _ir_blocks_to_message_params(blocks)
        try:
            count = await self._client.messages.count_tokens(
                messages=msgs, model=self._model
            )
            return count.input_tokens
        except Exception as e:
            print(f"WARNING: Error when token counting blocks: {e}")
            return None

    async def count_frame(self) -> int | None:
        surface = self._builder.build(
            blocks=[IRUserTextBlock(text=".", origin="human")]
        )
        count = await self.count_surface(surface)
        if count is None:
            print("WARNING: Error when token counting frame, printed above")
        return count

    async def count_surface(self, surface: RequestSurface) -> int | None:
        try:
            count = await self._client.messages.count_tokens(
                system=surface.system,  # type: ignore
                tools=surface.tools,  # type: ignore
                messages=surface.messages,  # type: ignore
                model=self._model,
                thinking=surface.thinking,  # type: ignore
                output_config=surface.output_config,  # type: ignore
            )
            return count.input_tokens
        except Exception as e:
            print(f"WARNING: Error when token counting surface: {e}")
            return None


class AnthropicBackend(ModelBackend):
    """ModelBackend implementation for Anthropic's Messages API."""

    def __init__(self, *, client: AsyncAnthropic | None = None):
        self.client = client or AsyncAnthropic()

    @property
    def provider(self) -> str:
        return "anthropic"

    def build_request_surface(
        self,
        *,
        model: str,
        system: str | list[dict[str, Any]],
        blocks: Sequence[IRBlock],
        tools: list[dict[str, Any]],
        max_output_tokens: int,
        effort: str,
    ) -> RequestSurface:
        if model == "claude-opus-4-7" and effort == "high":
            effort = "xhigh"
        return RequestSurface(
            model=model,
            system=system,
            tools=tools,
            messages=_ir_blocks_to_message_params(blocks),
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": effort},
            cache_control={"type": "ephemeral", "ttl": "1h"},
            max_output_tokens=max_output_tokens,
        )

    def stream(
        self, surface: RequestSurface, cancel_token: CancelToken
    ) -> AnthropicGenerationStream:
        """Start a streaming generation."""
        kwargs: dict[str, Any] = {
            "model": surface.model,
            "messages": surface.messages,
            "max_tokens": surface.max_output_tokens,
        }
        if surface.system is not None:
            kwargs["system"] = surface.system
        if surface.tools:
            kwargs["tools"] = surface.tools
        if surface.thinking is not None:
            kwargs["thinking"] = surface.thinking
        if surface.output_config is not None:
            kwargs["output_config"] = surface.output_config
        if surface.cache_control is not None:
            kwargs["cache_control"] = surface.cache_control

        stream_ctx = self.client.messages.stream(**kwargs)
        return AnthropicGenerationStream(stream_ctx, model=surface.model)

    def build_tool_schemas(
        self,
        registry: ToolRegistry,
    ) -> list[dict[str, Any]]:
        """Generate provider-specific tool schemas from the registry."""
        return [
            _anthropic_tool_schema(tool.name, tool.input_model)
            for tool in registry.tools
        ]

    def build_token_counter(
        self, config: SpellbookConfig, surface_builder: RequestSurfaceBuilder
    ) -> AnthropicTokenCounter:
        return AnthropicTokenCounter(
            client=self.client, model=config.model, surface_builder=surface_builder
        )


# --- Helpers ---


def _normalize_content_blocks(
    content: list[ParsedContentBlock[NotGiven]],
) -> tuple[list[IRBlock], bool]:
    """Normalize Anthropic SDK content blocks to plain dicts.
    Also tells you whether or not there were any tools calls for convenience"""
    if content is None:
        return [], False
    blocks = []
    has_tool_use = False
    for block in content:
        match block:
            case ParsedTextBlock():
                blocks.append(IRAssistantTextBlock(text=block.text))
            case ThinkingBlock():
                blocks.append(
                    IRThinkingBlock(text=block.thinking, signature=block.signature)
                )
            case ToolUseBlock():
                blocks.append(
                    IRToolCallBlock(
                        call_id=block.id, tool=block.name, input=block.input
                    )
                )
                has_tool_use = True

    return blocks, has_tool_use


def _normalize_partial_content_blocks(
    content: list[ParsedContentBlock[NotGiven]],
) -> tuple[list[IRBlock], bool]:
    """Normalize Anthropic SDK content blocks to plain dicts.
    Truncates tool use calls that might be invalid JSON."""
    if content is None:
        return [], False
    blocks = []
    has_tool_use = False
    for block in content:
        match block:
            case ParsedTextBlock():
                blocks.append(IRAssistantTextBlock(text=block.text))
            case ThinkingBlock():
                blocks.append(
                    IRThinkingBlock(text=block.thinking, signature=block.signature)
                )
            case ToolUseBlock():
                pass

    return blocks, has_tool_use


def _parse_image_block(block: IRImageBlock) -> ImageBlockParam:
    match block.source:
        case IRImageBase64Source():
            source = Base64ImageSourceParam(
                type="base64",
                media_type=block.source.media_type,  # type: ignore
                data=block.source.data,
            )
        case IRImageURLSource():
            source = URLImageSourceParam(type="url", url=block.source.url)
        case IRImageBlobSource():
            raise ValueError(
                "Blob image sources must be hydrated before provider rendering."
            )
    return ImageBlockParam(source=source, type="image")


def _ir_blocks_to_message_params(blocks: Sequence[IRBlock]) -> list[MessageParam]:
    messages: list[MessageParam] = []
    buffer: list[IRBlock] = []
    last_role: Literal["assistant", "user"] | None = None

    def _flush(
        buffer: list[IRBlock], last_role: Literal["assistant", "user"] | None
    ) -> None:
        if last_role is None or len(buffer) == 0:
            return
        msg_content = []
        if last_role == "user":
            for b in buffer:
                match b:
                    case IRUserTextBlock():
                        msg_content.append(TextBlockParam(text=b.text, type="text"))
                    case IRImageBlock():
                        msg_content.append(_parse_image_block(b))
                    case IRToolResultBlock():
                        content_blocks = []
                        for cb in b.content:
                            match cb:
                                case IRToolTextBlock():
                                    content_blocks.append(
                                        TextBlockParam(text=cb.text, type="text")
                                    )
                                case IRImageBlock():
                                    content_blocks.append(_parse_image_block(cb))
                        msg_content.append(
                            ToolResultBlockParam(
                                tool_use_id=b.call_id,
                                type="tool_result",
                                content=content_blocks,
                                is_error=b.is_error,
                            )
                        )
                    case _:
                        raise ValueError(
                            f"FOUND BLOCK IN THE WRONG TURN!!! Block of type {type(b)} on {last_role}'s turn!"
                        )
        elif last_role == "assistant":
            for b in buffer:
                match b:
                    case IRAssistantTextBlock():
                        msg_content.append(TextBlockParam(text=b.text, type="text"))
                    case IRThinkingBlock():
                        msg_content.append(
                            ThinkingBlockParam(
                                signature=b.signature, thinking=b.text, type="thinking"
                            )
                        )
                    case IRToolCallBlock():
                        msg_content.append(
                            ToolUseBlockParam(
                                id=b.call_id,
                                input=b.input,
                                name=b.tool,
                                type="tool_use",
                            )
                        )
                    case _:
                        raise ValueError(
                            f"FOUND BLOCK IN THE WRONG TURN!!! Block of type {type(b)} on {last_role}'s turn!"
                        )
        messages.append(MessageParam(role=last_role, content=msg_content))
        buffer.clear()

    for block in blocks:
        match block:
            case IRUserTextBlock():
                if last_role == "assistant":
                    _flush(buffer, last_role)
                buffer.append(block)
                last_role = "user"
            case IRAssistantTextBlock():
                if last_role == "user":
                    _flush(buffer, last_role)
                buffer.append(block)
                last_role = "assistant"
            case IRImageBlock():
                if last_role == "assistant":
                    _flush(buffer, last_role)
                buffer.append(block)
                last_role = "user"
            case IRThinkingBlock():
                if last_role == "user":
                    _flush(buffer, last_role)
                buffer.append(block)
                last_role = "assistant"
            case IRToolCallBlock():
                if last_role == "user":
                    _flush(buffer, last_role)
                buffer.append(block)
                last_role = "assistant"
            case IRToolResultBlock():
                if last_role == "assistant":
                    _flush(buffer, last_role)
                buffer.append(block)
                last_role = "user"

    _flush(buffer, last_role)
    return messages


def _normalize_usage(usage: Usage) -> IRUsage:
    """Normalize an Anthropic usage object to UsageStats."""
    if usage is None:
        return IRUsage()
    return IRUsage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_input_tokens or 0,
        cache_create_tokens=usage.cache_creation_input_tokens or 0,
    )


def _clean_property(prop: dict) -> dict:
    """Strip Pydantic noise from a JSON Schema property to match the CC contract.

    - Removes `title` and `default`
    - Collapses `anyOf: [{type: X}, {type: null}]` → `{type: X}`
    """
    cleaned = {k: v for k, v in prop.items() if k not in ("title", "default")}

    # Pydantic v2 represents Optional[T] as anyOf: [{type: T}, {type: null}].
    # The Anthropic API just wants {type: T} — nullability is implied by the
    # field not being in `required`.
    if "anyOf" in cleaned:
        non_null = [s for s in cleaned["anyOf"] if s != {"type": "null"}]
        if len(non_null) == 1:
            cleaned.pop("anyOf")
            cleaned.update(non_null[0])

    return cleaned


def _anthropic_tool_schema(name: str, model: type[BaseModel]) -> dict[str, Any]:
    """Convert a Pydantic model into an Anthropic API tool definition.

    Loads the tool description from prompts/tools/{name}.md.
    Falls back to the model's docstring if the file doesn't exist.
    """
    desc_path = TOOL_DESCS_DIR / f"{name}.md"
    if desc_path.exists():
        description = desc_path.read_text().strip()
    else:
        description = (model.__doc__ or "").strip()

    schema = model.model_json_schema()

    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": {
                k: _clean_property(v) for k, v in schema.get("properties", {}).items()
            },
            "required": schema.get("required", []),
        },
    }

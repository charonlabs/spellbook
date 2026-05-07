"""OpenAI model backend for Spellbook.

Implements the ``ModelBackend`` protocol for OpenAI's Responses API.
Provider-specific request shapes, streaming normalization, tool schema
serialization, and token counting stay behind this boundary; the rest of
core continues to speak IR.
"""

# ruff: noqa: ANN401

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from openai import AsyncOpenAI
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


class OpenAIGenerationStream(GenerationStream):
    """Wrap OpenAI Responses streaming into normalized IR stream events."""

    def __init__(self, stream_ctx: Any, model: str):
        self._stream_ctx = stream_ctx
        self._model = model
        self._stream: Any | None = None
        self._response: Any | None = None
        self._exhausted = False
        self._text_parts: list[str] | None = None
        self._thinking_parts: list[str] | None = None
        self._partial_blocks: list[IRBlock] = []
        self._latest_usage: IRUsage | None = None

    async def __aenter__(self) -> "OpenAIGenerationStream":
        self._stream = await self._stream_ctx.__aenter__()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return await self._stream_ctx.__aexit__(*exc)

    def __aiter__(self) -> "OpenAIGenerationStream":
        return self

    async def __anext__(self) -> IRStreamEvent:
        if self._exhausted or self._stream is None:
            raise StopAsyncIteration

        async for event in self._stream:
            self._capture_usage(event)
            if _obj_get(event, "type") == "response.completed":
                self._response = _obj_get(event, "response")
                break

            normalized = self._normalize_event(event)
            if normalized is not None:
                return normalized

        if self._response is None and self._stream is not None:
            self._response = await self._stream.get_final_response()

        self._exhausted = True
        raise StopAsyncIteration

    async def get_final_response(self) -> IRGeneration:
        """Return the complete response after iteration."""
        if not self._exhausted:
            async for _ in self:
                pass

        if self._response is None:
            raise ValueError("OpenAI stream exhausted without a final response.")

        blocks, has_tool_call = _normalize_content_blocks(
            _obj_get(self._response, "output")
        )
        usage = _normalize_usage(_obj_get(self._response, "usage"))
        return IRGeneration(
            model=self._model,
            blocks=blocks,
            stop_reason=_normalize_stop_reason(self._response, has_tool_call),
            usage=usage,
        )

    def get_current_response(self, *, stop_reason: StopReason) -> IRGeneration:
        """Return a partial generation snapshot.

        Partial OpenAI tool calls are intentionally omitted, matching the
        Anthropic path's conservative behavior around incomplete tool JSON.
        """
        blocks = list(self._partial_blocks)
        if self._thinking_parts is not None:
            blocks.append(IRThinkingBlock(text="".join(self._thinking_parts)))
        if self._text_parts is not None:
            blocks.append(IRAssistantTextBlock(text="".join(self._text_parts)))
        return IRGeneration(
            model=self._model,
            blocks=blocks,
            stop_reason=stop_reason,
            usage=self._latest_usage,
        )

    def _normalize_event(self, event: Any) -> IRStreamEvent | None:
        event_type = str(_obj_get(event, "type") or "")
        item = _obj_get(event, "item")
        item_type = str(_obj_get(item, "type") or "")

        if event_type == "response.output_item.added":
            if item_type == "reasoning":
                self._thinking_parts = []
                return IRStreamThinkingStartEvent()
            if item_type == "message":
                self._text_parts = []
                return IRStreamTextStartEvent()

        if event_type == "response.reasoning_summary_text.delta":
            delta = str(_obj_get(event, "delta") or "")
            if self._thinking_parts is None:
                self._thinking_parts = []
            self._thinking_parts.append(delta)
            return IRStreamThinkingDeltaEvent(text=delta)

        if event_type == "response.output_text.delta":
            delta = str(_obj_get(event, "delta") or "")
            if self._text_parts is None:
                self._text_parts = []
            self._text_parts.append(delta)
            return IRStreamTextDeltaEvent(text=delta)

        if event_type == "response.output_item.done":
            if item_type == "reasoning":
                self._finish_partial_thinking(item)
                return IRStreamThinkingEndEvent()
            if item_type == "message":
                self._finish_partial_text(item)
                return IRStreamTextEndEvent()

        return None

    def _finish_partial_thinking(self, item: Any) -> None:
        if self._thinking_parts is not None:
            self._partial_blocks.append(
                IRThinkingBlock(
                    text="".join(self._thinking_parts),
                    signature=_reasoning_signature(item),
                )
            )
            self._thinking_parts = None
            return

        block = _translate_openai_reasoning_item_to_ir_block(_to_plain_openai_obj(item))
        if block is not None:
            self._partial_blocks.append(block)

    def _finish_partial_text(self, item: Any) -> None:
        if self._text_parts is not None:
            self._partial_blocks.append(
                IRAssistantTextBlock(text="".join(self._text_parts))
            )
            self._text_parts = None
            return

        block = _translate_openai_message_item_to_ir_block(_to_plain_openai_obj(item))
        if block is not None:
            self._partial_blocks.append(block)

    def _capture_usage(self, event: Any) -> None:
        usage = _obj_get(event, "usage")
        if usage is None:
            response = _obj_get(event, "response")
            usage = _obj_get(response, "usage")
        if usage is not None:
            self._latest_usage = _normalize_usage(usage)


class OpenAITokenCounter(TokenCounter):
    """TokenCounter implementation for OpenAI Responses models."""

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        surface_builder: RequestSurfaceBuilder,
    ):
        self._client = client
        self._model = model
        self._builder = surface_builder

    async def count_block_content(self, block: IRBlock) -> int | None:
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
                for content_block in block.content:
                    match content_block:
                        case IRToolTextBlock():
                            blocks.append(
                                IRUserTextBlock(text=content_block.text, origin="human")
                            )
                        case IRImageBlock():
                            blocks.append(content_block)
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
        surface = RequestSurface(
            model=self._model,
            messages=_ir_blocks_to_response_input_items(blocks),
        )
        return await self.count_surface(surface)

    async def count_frame(self) -> int | None:
        surface = self._builder.build(
            blocks=[IRUserTextBlock(text=".", origin="human")]
        )
        count = await self.count_surface(surface)
        if count is None:
            print("WARNING: Error when token counting OpenAI frame, printed above")
        return count

    async def count_surface(self, surface: RequestSurface) -> int | None:
        try:
            result = await self._client.responses.input_tokens.count(
                **_response_count_kwargs(surface)
            )
            return int(_obj_get(result, "input_tokens") or 0)
        except Exception as exc:
            print(f"WARNING: Error when token counting OpenAI surface: {exc}")
            return None


class OpenAIBackend(ModelBackend):
    """ModelBackend implementation for OpenAI's Responses API."""

    def __init__(self, *, client: Any | None = None):
        self.client = client or AsyncOpenAI()

    @property
    def provider(self) -> str:
        return "openai"

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
        return RequestSurface(
            model=model,
            system=_translate_system_to_instructions(system),
            tools=tools,
            messages=_ir_blocks_to_response_input_items(blocks),
            thinking={
                "effort": effort,
                "summary": "detailed",
            },
            max_output_tokens=max_output_tokens,
        )

    def stream(
        self, surface: RequestSurface, cancel_token: CancelToken
    ) -> OpenAIGenerationStream:
        """Start a streaming generation."""
        stream_ctx = self.client.responses.stream(**_response_stream_kwargs(surface))
        return OpenAIGenerationStream(stream_ctx, model=surface.model)

    def build_tool_schemas(
        self,
        registry: ToolRegistry,
    ) -> list[dict[str, Any]]:
        """Generate provider-specific tool schemas from the registry."""
        return [
            _openai_tool_schema(tool.name, tool.input_model) for tool in registry.tools
        ]

    def build_token_counter(
        self, config: SpellbookConfig, surface_builder: RequestSurfaceBuilder
    ) -> OpenAITokenCounter:
        return OpenAITokenCounter(
            client=self.client,
            model=config.model,
            surface_builder=surface_builder,
        )


# --- Provider request helpers ---


def _response_stream_kwargs(surface: RequestSurface) -> dict[str, Any]:
    kwargs = _response_base_kwargs(surface)
    kwargs["max_output_tokens"] = surface.max_output_tokens
    kwargs["store"] = False
    kwargs["include"] = ["reasoning.encrypted_content"]
    return kwargs


def _response_count_kwargs(surface: RequestSurface) -> dict[str, Any]:
    kwargs = _response_base_kwargs(surface)
    kwargs["truncation"] = "disabled"
    return kwargs


def _response_base_kwargs(surface: RequestSurface) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": surface.model,
        "input": surface.messages,
        "text": {"format": {"type": "text"}},
    }
    if surface.system:
        kwargs["instructions"] = surface.system
    if surface.tools:
        kwargs["tools"] = surface.tools
        kwargs["parallel_tool_calls"] = False
    if surface.thinking is not None:
        kwargs["reasoning"] = surface.thinking
    return kwargs


def _translate_system_to_instructions(
    system: str | list[dict[str, Any]] | None,
) -> str | None:
    if system is None:
        return None
    if isinstance(system, str):
        return system

    parts: list[str] = []
    for block in system:
        text = block.get("text")
        if text is None:
            text = block.get("content")
        if text is None:
            continue
        if not isinstance(text, str):
            raise ValueError("system block text/content must be a string")
        if text:
            parts.append(text)
    return "\n\n".join(parts) if parts else None


def _ir_blocks_to_response_input_items(
    blocks: Sequence[IRBlock],
) -> list[dict[str, Any]]:
    input_items: list[dict[str, Any]] = []
    pending_user_content: list[dict[str, Any]] = []

    def flush_user_content() -> None:
        if not pending_user_content:
            return
        input_items.append({"role": "user", "content": list(pending_user_content)})
        pending_user_content.clear()

    for block in blocks:
        match block:
            case IRUserTextBlock():
                pending_user_content.append({"type": "input_text", "text": block.text})
            case IRImageBlock():
                pending_user_content.append(_image_block_to_openai_part(block))
            case IRToolResultBlock():
                flush_user_content()
                input_items.append(_tool_result_block_to_openai_item(block))
            case IRThinkingBlock():
                flush_user_content()
                input_items.append(_thinking_block_to_openai_item(block))
            case IRAssistantTextBlock():
                flush_user_content()
                input_items.append(_assistant_text_block_to_openai_item(block))
            case IRToolCallBlock():
                flush_user_content()
                input_items.append(_tool_call_block_to_openai_item(block))

    flush_user_content()
    return input_items


def _thinking_block_to_openai_item(block: IRThinkingBlock) -> dict[str, Any]:
    item: dict[str, Any] = {"type": "reasoning", "summary": []}
    if block.text:
        item["summary"] = [{"type": "summary_text", "text": block.text}]
    if block.signature:
        item["encrypted_content"] = block.signature
    return item


def _assistant_text_block_to_openai_item(block: IRAssistantTextBlock) -> dict[str, Any]:
    return {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": block.text}],
    }


def _tool_call_block_to_openai_item(block: IRToolCallBlock) -> dict[str, Any]:
    return {
        "type": "function_call",
        "call_id": block.call_id,
        "name": block.tool,
        "arguments": json.dumps(block.input, separators=(",", ":")),
    }


def _tool_result_block_to_openai_item(block: IRToolResultBlock) -> dict[str, Any]:
    return {
        "type": "function_call_output",
        "call_id": block.call_id,
        "output": _tool_result_output(block),
    }


def _tool_result_output(block: IRToolResultBlock) -> str | list[dict[str, Any]]:
    if not block.content:
        return ""
    if all(
        isinstance(content_block, IRToolTextBlock) for content_block in block.content
    ):
        return "".join(
            content_block.text
            for content_block in block.content
            if isinstance(content_block, IRToolTextBlock)
        )

    output: list[dict[str, Any]] = []
    for content_block in block.content:
        match content_block:
            case IRToolTextBlock():
                output.append({"type": "input_text", "text": content_block.text})
            case IRImageBlock():
                output.append(_image_block_to_openai_part(content_block))
    return output


def _image_block_to_openai_part(block: IRImageBlock) -> dict[str, Any]:
    match block.source:
        case IRImageBase64Source():
            image_url = f"data:{block.source.media_type};base64,{block.source.data}"
        case IRImageURLSource():
            image_url = block.source.url
        case IRImageBlobSource():
            raise ValueError(
                "Blob image sources must be hydrated before provider rendering."
            )
    return {"type": "input_image", "image_url": image_url}


# --- Response normalization helpers ---


def _normalize_content_blocks(output: Any) -> tuple[list[IRBlock], bool]:
    if output is None:
        return [], False
    output_items = _to_plain_openai_obj(output)
    if not isinstance(output_items, list):
        raise ValueError("OpenAI response output must normalize to a list")

    blocks: list[IRBlock] = []
    has_tool_call = False
    for item in output_items:
        if not isinstance(item, dict):
            raise ValueError("OpenAI response output items must normalize to dicts")

        block = _translate_openai_reasoning_item_to_ir_block(item)
        if block is not None:
            blocks.append(block)
            continue

        block = _translate_openai_message_item_to_ir_block(item)
        if block is not None:
            blocks.append(block)
            continue

        block = _translate_openai_function_call_item_to_ir_block(item)
        if block is not None:
            blocks.append(block)
            has_tool_call = True
            continue

        raise ValueError(
            f"Unsupported OpenAI output item type: {str(item.get('type') or '')!r}"
        )

    return blocks, has_tool_call


def _translate_openai_reasoning_item_to_ir_block(item: Any) -> IRThinkingBlock | None:
    if not isinstance(item, dict) or str(item.get("type") or "") != "reasoning":
        return None

    summary_parts = item.get("summary") or []
    if not isinstance(summary_parts, list):
        raise ValueError("OpenAI reasoning item summary must be a list")

    thinking_parts: list[str] = []
    for part in summary_parts:
        if not isinstance(part, dict):
            raise ValueError("OpenAI reasoning summary part must be a dict")
        part_type = str(part.get("type") or "")
        if part_type != "summary_text":
            raise ValueError(
                f"Unsupported OpenAI reasoning summary part type: {part_type!r}"
            )
        text = part.get("text", "")
        if text is None:
            text = ""
        if not isinstance(text, str):
            raise ValueError("OpenAI reasoning summary_text must be a string")
        thinking_parts.append(text)

    encrypted_content = item.get("encrypted_content") or ""
    if not isinstance(encrypted_content, str):
        raise ValueError("OpenAI reasoning encrypted_content must be a string")
    return IRThinkingBlock(text="".join(thinking_parts), signature=encrypted_content)


def _translate_openai_message_item_to_ir_block(
    item: Any,
) -> IRAssistantTextBlock | None:
    if not isinstance(item, dict) or str(item.get("type") or "") != "message":
        return None

    content_parts = item.get("content") or []
    if not isinstance(content_parts, list):
        raise ValueError("OpenAI message item content must be a list")

    text_parts: list[str] = []
    for part in content_parts:
        if not isinstance(part, dict):
            raise ValueError("OpenAI message content parts must be dicts")
        part_type = str(part.get("type") or "")
        if part_type == "output_text":
            text = part.get("text", "")
        elif part_type == "refusal":
            text = part.get("refusal", part.get("text", ""))
        else:
            raise ValueError(
                f"Unsupported OpenAI message content part type: {part_type!r}"
            )
        if text is None:
            text = ""
        if not isinstance(text, str):
            raise ValueError("OpenAI message text content must be a string")
        text_parts.append(text)

    return IRAssistantTextBlock(text="".join(text_parts))


def _translate_openai_function_call_item_to_ir_block(
    item: Any,
) -> IRToolCallBlock | None:
    if not isinstance(item, dict) or str(item.get("type") or "") != "function_call":
        return None

    call_id = str(item.get("call_id") or "")
    if not call_id:
        raise ValueError("OpenAI function_call item is missing call_id")
    name = item.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("OpenAI function_call item is missing string name")

    raw_arguments = item.get("arguments", "")
    if isinstance(raw_arguments, dict):
        tool_input = raw_arguments
    else:
        if not isinstance(raw_arguments, str):
            raise ValueError("OpenAI function_call arguments must be JSON text")
        try:
            tool_input = json.loads(raw_arguments) if raw_arguments else {}
        except json.JSONDecodeError as exc:
            raise ValueError(
                "OpenAI function_call arguments must be valid JSON"
            ) from exc
    if not isinstance(tool_input, dict):
        raise ValueError("OpenAI function_call arguments must decode to a dict")

    return IRToolCallBlock(call_id=call_id, tool=name, input=tool_input)


def _normalize_usage(usage: Any) -> IRUsage:
    if usage is None:
        return IRUsage()

    input_tokens = int(_obj_get(usage, "input_tokens") or 0)
    output_tokens = int(_obj_get(usage, "output_tokens") or 0)
    input_details = _obj_get(usage, "input_tokens_details")
    cached_tokens = int(_obj_get(input_details, "cached_tokens") or 0)

    return IRUsage(
        input_tokens=max(0, input_tokens - cached_tokens),
        output_tokens=output_tokens,
        cache_read_tokens=cached_tokens,
        cache_create_tokens=0,
    )


def _normalize_stop_reason(response: Any, has_tool_call: bool) -> StopReason:
    if has_tool_call:
        return "tool_use"

    if _response_has_refusal(response):
        return "refusal"

    status = str(_obj_get(response, "status") or "")
    incomplete_details = _obj_get(response, "incomplete_details")
    incomplete_reason = str(_obj_get(incomplete_details, "reason") or "")

    if status == "completed":
        return "end_turn"
    if status == "cancelled":
        return "cancelled"
    if status == "failed":
        return "error"
    if status == "incomplete":
        if incomplete_reason == "max_output_tokens":
            return "max_tokens"
        return "unspecified"
    return "unspecified"


def _response_has_refusal(response: Any) -> bool:
    output = _to_plain_openai_obj(_obj_get(response, "output"))
    if not isinstance(output, list):
        return False
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content") or []
        if not isinstance(content, list):
            continue
        if any(
            isinstance(part, dict) and part.get("type") == "refusal" for part in content
        ):
            return True
    return False


# --- Tool schema helpers ---


def _openai_tool_schema(name: str, model: type[BaseModel]) -> dict[str, Any]:
    desc_path = TOOL_DESCS_DIR / f"{name}.md"
    if desc_path.exists():
        description = desc_path.read_text().strip()
    else:
        description = (model.__doc__ or "").strip()

    schema = model.model_json_schema()
    parameters = _ensure_closed_object_schema(
        {
            "type": "object",
            "properties": {
                key: _clean_schema_property(value)
                for key, value in schema.get("properties", {}).items()
            },
            "required": schema.get("required", []),
        }
    )
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": parameters,
        "strict": False,
    }


def _clean_schema_property(prop: Any) -> Any:
    if isinstance(prop, list):
        return [_clean_schema_property(item) for item in prop]
    if not isinstance(prop, dict):
        return prop

    cleaned = {
        str(key): _clean_schema_property(value)
        for key, value in prop.items()
        if key not in {"title", "default"}
    }
    if "anyOf" in cleaned:
        non_null = [schema for schema in cleaned["anyOf"] if schema != {"type": "null"}]
        if len(non_null) == 1:
            cleaned.pop("anyOf")
            cleaned.update(non_null[0])
    return cleaned


def _ensure_closed_object_schema(schema: Any) -> Any:
    if isinstance(schema, list):
        return [_ensure_closed_object_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema

    normalized = {
        str(key): _ensure_closed_object_schema(value) for key, value in schema.items()
    }
    if normalized.get("type") == "object" and "additionalProperties" not in normalized:
        normalized["additionalProperties"] = False
    return normalized


# --- Generic object helpers ---


def _reasoning_signature(item: Any) -> str:
    signature = _obj_get(item, "encrypted_content")
    if signature is None:
        return ""
    if not isinstance(signature, str):
        raise ValueError("OpenAI reasoning encrypted_content must be a string")
    return signature


def _obj_get(value: Any, key: str) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _to_plain_openai_obj(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_to_plain_openai_obj(item) for item in value]
    if isinstance(value, tuple):
        return [_to_plain_openai_obj(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_plain_openai_obj(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        return _to_plain_openai_obj(value.model_dump())
    if hasattr(value, "__dict__"):
        return _to_plain_openai_obj(vars(value))
    return value

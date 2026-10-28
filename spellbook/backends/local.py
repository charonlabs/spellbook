"""Local model backend for Spellbook.

Implements the ``ModelBackend`` protocol over the OpenAI **Chat Completions**
API (``POST {base_url}/chat/completions``), which is what local inference
servers (llama.cpp's ``llama-server``, vLLM, SGLang, ...) actually speak.

Mapping summary (IR <-> Chat Completions):

- ``IRUserTextBlock`` / ``IRImageBlock``  -> ``{"role": "user"}`` messages
- ``IRThinkingBlock``                      <-> assistant ``reasoning_content``
- ``IRAssistantTextBlock``                 <-> assistant ``content``
- ``IRToolCallBlock``                      <-> assistant ``tool_calls``
- ``IRToolResultBlock``                    -> ``{"role": "tool"}`` messages

Consecutive model-side blocks collapse into one assistant message (reasoning,
then text, then tool calls), which is the shape chat templates expect.
Thinking blocks carry no signature: local models have nothing to verify, and
replaying ``reasoning_content`` lets templates (e.g. Qwen's) keep interleaved
reasoning across a tool loop.

Reasoning effort is passed through as ``reasoning_effort``. Servers disagree on
the vocabulary (Qwen's template accepts ``xhigh``/``medium``/``low`` and raises
on ``high``), so a rejected value is retried with aliases and finally omitted;
whatever works is remembered for the rest of the session.
"""

# ruff: noqa: ANN401

from __future__ import annotations

import json
import logging
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import httpx
from openai import APIStatusError, AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk

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
from ..tools.registry import ToolRegistry
from .model_backend import GenerationStream, ModelBackend, RequestSurface, TokenCounter
from .openai import _obj_get, _openai_tool_schema, _translate_system_to_instructions

logger = logging.getLogger(__name__)

DEFAULT_LOCAL_BASE_URL = "http://localhost:8080/v1"
LOCAL_API_KEY = "local"  # local servers ignore auth; the SDK insists on a value

# Fallbacks tried, in order, when a server rejects a reasoning_effort value.
# After the list is exhausted the field is omitted entirely (server default).
_EFFORT_ALIASES: dict[str, tuple[str, ...]] = {
    "max": ("xhigh", "high"),
    "xhigh": ("high",),
    "high": ("xhigh",),
    "medium": (),
    "low": (),
    "minimal": ("low",),
    "none": ("low",),
}


# (base_url, model, requested effort) -> value the server accepted (None = omit).
# Process-wide so every cast/fork against the same server learns it once.
_EFFORT_RESOLUTION: dict[tuple[str, str, str], str | None] = {}


def normalize_base_url(base_url: str | None) -> str:
    """Return an OpenAI-SDK base URL ending in ``/v1``.

    Accepts ``http://host:8080``, ``http://host:8080/`` or ``http://host:8080/v1``.
    """
    url = (base_url or DEFAULT_LOCAL_BASE_URL).strip().rstrip("/")
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return url


def server_root_url(base_url: str | None) -> str:
    """The server root (no ``/v1``), where llama.cpp/vLLM expose ``/tokenize``."""
    url = normalize_base_url(base_url)
    return url[: -len("/v1")]


# --- Streaming ---


@dataclass
class _ToolCallAccumulator:
    index: int
    call_id: str = ""
    name: str = ""
    argument_parts: list[str] = field(default_factory=list)


class LocalGenerationStream(GenerationStream):
    """Normalize a Chat Completions chunk stream into IR stream events.

    Opening the stream (``__aenter__``) is where request-shape fallbacks
    happen: a server that rejects our ``reasoning_effort`` fails at template
    render time, before any token is produced, so retrying is free.
    """

    def __init__(self, backend: "LocalBackend", surface: RequestSurface):
        self._backend = backend
        self._surface = surface
        self._model = surface.model
        self._stream: AsyncStream[ChatCompletionChunk] | None = None
        self._exhausted = False
        self._pending: deque[IRStreamEvent] = deque()

        self._blocks: list[IRBlock] = []
        self._phase: str | None = None  # "thinking" | "text" | "tools" | None
        self._parts: list[str] = []
        self._tool_calls: dict[int, _ToolCallAccumulator] = {}
        self._finish_reason: str | None = None
        self._usage: IRUsage | None = None
        self._dropped_partial_tool_call = False

    async def __aenter__(self) -> "LocalGenerationStream":
        self._stream = await self._backend._open_stream(self._surface)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._stream is not None:
            await self._stream.close()

    def __aiter__(self) -> "LocalGenerationStream":
        return self

    async def __anext__(self) -> IRStreamEvent:
        while True:
            if self._pending:
                return self._pending.popleft()
            if self._exhausted or self._stream is None:
                raise StopAsyncIteration
            try:
                chunk = await self._stream.__anext__()
            except StopAsyncIteration:
                self._close_phase()
                self._exhausted = True
                continue
            self._consume_chunk(chunk)

    async def get_final_response(self) -> IRGeneration:
        if not self._exhausted:
            async for _ in self:
                pass
        return IRGeneration(
            model=self._model,
            blocks=list(self._blocks),
            stop_reason=self._stop_reason(),
            usage=self._usage,
        )

    def get_current_response(self, *, stop_reason: StopReason) -> IRGeneration:
        """Snapshot of what has streamed so far.

        In-flight thinking/text is included; in-flight tool calls are omitted
        (their JSON may be incomplete), matching the other backends.
        """
        blocks = list(self._blocks)
        if self._phase == "thinking":
            block = _thinking_block("".join(self._parts))
            if block is not None:
                blocks.append(block)
        elif self._phase == "text":
            block = _text_block("".join(self._parts))
            if block is not None:
                blocks.append(block)
        return IRGeneration(
            model=self._model,
            blocks=blocks,
            stop_reason=stop_reason,
            usage=self._usage,
        )

    # -- chunk handling --

    def _consume_chunk(self, chunk: Any) -> None:
        usage = _obj_get(chunk, "usage")
        if usage is not None:
            self._usage = _normalize_usage(usage)

        for choice in _obj_get(chunk, "choices") or []:
            if (_obj_get(choice, "index") or 0) != 0:
                continue
            delta = _obj_get(choice, "delta")
            if delta is not None:
                reasoning = _reasoning_text(delta)
                if reasoning:
                    self._on_text_like("thinking", reasoning)
                content = _obj_get(delta, "content")
                if content:
                    self._on_text_like("text", str(content))
                for tool_delta in _obj_get(delta, "tool_calls") or []:
                    self._on_tool_call_delta(tool_delta)
            finish_reason = _obj_get(choice, "finish_reason")
            if finish_reason:
                self._finish_reason = str(finish_reason)

    def _on_text_like(self, phase: str, delta: str) -> None:
        if self._phase != phase:
            # Don't open a block for pure whitespace (e.g. the "\n\n" models
            # emit between </think> and a tool call); hold it until real text.
            if not delta.strip():
                return
            self._close_phase()
            self._phase = phase
            self._parts = []
            self._pending.append(
                IRStreamThinkingStartEvent()
                if phase == "thinking"
                else IRStreamTextStartEvent()
            )
            delta = delta.lstrip()
        self._parts.append(delta)
        self._pending.append(
            IRStreamThinkingDeltaEvent(text=delta)
            if phase == "thinking"
            else IRStreamTextDeltaEvent(text=delta)
        )

    def _on_tool_call_delta(self, tool_delta: Any) -> None:
        if self._phase != "tools":
            self._close_phase()
            self._phase = "tools"
        index = int(_obj_get(tool_delta, "index") or 0)
        acc = self._tool_calls.get(index)
        if acc is None:
            acc = self._tool_calls[index] = _ToolCallAccumulator(index=index)
        call_id = _obj_get(tool_delta, "id")
        if call_id:
            acc.call_id = str(call_id)
        function = _obj_get(tool_delta, "function")
        name = _obj_get(function, "name")
        if name and str(name) != acc.name:
            # Spec: the name arrives whole in the first delta. Some servers
            # fragment it; repeated identical names are just echoes.
            acc.name += str(name)
        arguments = _obj_get(function, "arguments")
        if arguments:
            acc.argument_parts.append(str(arguments))

    def _close_phase(self) -> None:
        if self._phase == "thinking":
            block = _thinking_block("".join(self._parts))
            if block is not None:
                self._blocks.append(block)
            self._pending.append(IRStreamThinkingEndEvent())
        elif self._phase == "text":
            block = _text_block("".join(self._parts))
            if block is not None:
                self._blocks.append(block)
            self._pending.append(IRStreamTextEndEvent())
        elif self._phase == "tools":
            self._flush_tool_calls()
        self._phase = None
        self._parts = []

    def _flush_tool_calls(self) -> None:
        truncated = self._finish_reason == "length"
        for index in sorted(self._tool_calls):
            acc = self._tool_calls[index]
            block = _tool_call_block(acc, truncated=truncated)
            if block is None:
                self._dropped_partial_tool_call = True
                continue
            self._blocks.append(block)
        self._tool_calls = {}

    def _stop_reason(self) -> StopReason:
        if any(isinstance(block, IRToolCallBlock) for block in self._blocks):
            return "tool_use"
        match self._finish_reason:
            case "stop":
                return "end_turn"
            case "length":
                return "max_tokens"
            case "tool_calls":
                # Tool calls were announced but none survived parsing.
                return "max_tokens" if self._dropped_partial_tool_call else "end_turn"
            case "content_filter":
                return "refusal"
            case _:
                return "unspecified"


# --- Token counting ---


class LocalTokenCounter(TokenCounter):
    """Token counting against the local server's own tokenizer.

    llama.cpp: ``POST /apply-template`` (renders the chat template, tools and
    all) then ``POST /tokenize``. vLLM: ``POST /tokenize`` with ``messages``.
    The first strategy that works is remembered. If neither does, counts are
    ``None`` (the protocol's "measurement failed").
    """

    def __init__(
        self,
        *,
        root_url: str,
        model: str,
        surface_builder: RequestSurfaceBuilder,
        http: httpx.AsyncClient | None = None,
    ):
        self._root = root_url.rstrip("/")
        self._model = model
        self._builder = surface_builder
        self._http = http or httpx.AsyncClient(timeout=httpx.Timeout(120.0))
        self._mode: str | None = None  # "llamacpp" | "vllm" | "unavailable"

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
            messages=ir_blocks_to_chat_messages(self._builder.project_blocks(blocks)),
        )
        return await self.count_surface(surface)

    async def count_frame(self) -> int | None:
        surface = self._builder.build(
            blocks=[IRUserTextBlock(text=".", origin="human")]
        )
        return await self.count_surface(surface)

    async def count_surface(self, surface: RequestSurface) -> int | None:
        if self._mode == "unavailable":
            return None
        messages = _surface_messages(surface)
        if not messages:
            return 0
        body: dict[str, Any] = {"model": self._model, "messages": messages}
        if surface.tools:
            body["tools"] = surface.tools

        strategies = [self._mode] if self._mode is not None else ["llamacpp", "vllm"]
        last_error: Exception | None = None
        for mode in strategies:
            try:
                count = (
                    await self._count_llamacpp(body)
                    if mode == "llamacpp"
                    else await self._count_vllm(body)
                )
            except Exception as exc:  # noqa: BLE001 - fall through to next strategy
                last_error = exc
                continue
            self._mode = mode
            return count

        if self._mode is None:
            self._mode = "unavailable"
            logger.warning(
                "local.token_counting_unavailable root=%s error=%s",
                self._root,
                last_error,
            )
        else:
            logger.warning(
                "local.token_count_failed mode=%s error=%s", self._mode, last_error
            )
        return None

    async def _count_llamacpp(self, body: dict[str, Any]) -> int:
        rendered = await self._http.post(f"{self._root}/apply-template", json=body)
        rendered.raise_for_status()
        prompt = rendered.json()["prompt"]
        if not isinstance(prompt, str):
            raise ValueError("apply-template returned a non-string prompt")
        tokenized = await self._http.post(
            f"{self._root}/tokenize",
            json={"content": prompt, "add_special": False},
        )
        tokenized.raise_for_status()
        tokens = tokenized.json()["tokens"]
        if not isinstance(tokens, list) or (prompt and not tokens):
            raise ValueError("tokenize returned no tokens for a non-empty prompt")
        return len(tokens)

    async def _count_vllm(self, body: dict[str, Any]) -> int:
        response = await self._http.post(
            f"{self._root}/tokenize",
            json={**body, "add_generation_prompt": True},
        )
        response.raise_for_status()
        data = response.json()
        if "count" not in data:
            raise ValueError("tokenize response has no 'count' (not vLLM?)")
        return int(data["count"])


# --- Backend ---


class LocalBackend(ModelBackend):
    """ModelBackend for OpenAI-compatible local servers (Chat Completions)."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        client: Any | None = None,
        http: httpx.AsyncClient | None = None,
    ):
        self.base_url = normalize_base_url(base_url)
        self.client = client or AsyncOpenAI(
            base_url=self.base_url,
            api_key=LOCAL_API_KEY,
            # Local prefill of a long context can take minutes before byte one.
            timeout=httpx.Timeout(connect=10.0, read=1800.0, write=120.0, pool=30.0),
            max_retries=0,  # the Generator owns retries
        )
        self._http = http

    @property
    def provider(self) -> str:
        return "local"

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
            messages=ir_blocks_to_chat_messages(blocks),
            thinking={"reasoning_effort": effort} if effort else None,
            max_output_tokens=max_output_tokens,
        )

    def stream(
        self, surface: RequestSurface, cancel_token: CancelToken
    ) -> LocalGenerationStream:
        return LocalGenerationStream(self, surface)

    def build_tool_schemas(self, registry: ToolRegistry) -> list[dict[str, Any]]:
        return [
            chat_tool_schema(tool.name, tool.input_model) for tool in registry.tools
        ]

    def build_token_counter(
        self, config: SpellbookConfig, surface_builder: RequestSurfaceBuilder
    ) -> LocalTokenCounter:
        return LocalTokenCounter(
            root_url=server_root_url(self.base_url),
            model=config.model,
            surface_builder=surface_builder,
            http=self._http,
        )

    # -- request opening with fallbacks --

    async def _open_stream(
        self, surface: RequestSurface
    ) -> AsyncStream[ChatCompletionChunk]:
        body = chat_request_body(surface)
        requested = (surface.thinking or {}).get("reasoning_effort")
        candidates = self._effort_candidates(surface.model, requested)

        last_exc: APIStatusError | None = None
        for effort in candidates:
            attempt = dict(body)
            if effort is None:
                attempt.pop("reasoning_effort", None)
            else:
                attempt["reasoning_effort"] = effort
            try:
                stream = await self._post_stream(attempt)
            except APIStatusError as exc:
                if requested is not None and _is_effort_rejection(exc):
                    logger.warning(
                        "local.reasoning_effort_rejected requested=%s tried=%s status=%s",
                        requested,
                        effort,
                        exc.status_code,
                    )
                    last_exc = exc
                    continue
                raise
            if requested is not None:
                key = (self.base_url, surface.model, requested)
                if key not in _EFFORT_RESOLUTION and effort != requested:
                    logger.warning(
                        "local.reasoning_effort_resolved requested=%s using=%s",
                        requested,
                        effort,
                    )
                _EFFORT_RESOLUTION[key] = effort
            return stream

        assert last_exc is not None
        raise last_exc

    def _effort_candidates(self, model: str, requested: str | None) -> list[str | None]:
        if requested is None:
            return [None]
        ladder: list[str | None] = [
            requested,
            *_EFFORT_ALIASES.get(requested, ()),
            None,
        ]
        key = (self.base_url, model, requested)
        if key in _EFFORT_RESOLUTION:
            # Known-good value first; the rest of the ladder only matters if
            # the server changed underneath us.
            ladder.insert(0, _EFFORT_RESOLUTION[key])
        candidates: list[str | None] = []
        for candidate in ladder:
            if candidate not in candidates:
                candidates.append(candidate)
        return candidates

    async def _post_stream(
        self, body: dict[str, Any]
    ) -> AsyncStream[ChatCompletionChunk]:
        # Post the plain JSON body directly: it carries non-standard fields
        # (reasoning_content on assistant turns, server-specific effort values)
        # that should reach the server untouched.
        return await self.client.post(
            "/chat/completions",
            body={**body, "stream": True},
            cast_to=ChatCompletion,
            stream=True,
            stream_cls=AsyncStream[ChatCompletionChunk],
        )


# --- Request helpers ---


def chat_request_body(surface: RequestSurface) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": surface.model,
        "messages": _surface_messages(surface),
        "max_tokens": surface.max_output_tokens,
        "stream_options": {"include_usage": True},
    }
    if surface.tools:
        body["tools"] = surface.tools
        body["parallel_tool_calls"] = True
    effort = (surface.thinking or {}).get("reasoning_effort")
    if effort:
        body["reasoning_effort"] = effort
    return body


def _surface_messages(surface: RequestSurface) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    system = _translate_system_to_instructions(surface.system)
    if system:
        messages.append({"role": "system", "content": system})
    messages.extend(dict(message) for message in surface.messages)
    return messages


def ir_blocks_to_chat_messages(blocks: Sequence[IRBlock]) -> list[dict[str, Any]]:
    """Project IR blocks onto Chat Completions messages (system excluded)."""
    messages: list[dict[str, Any]] = []
    user_parts: list[dict[str, Any]] = []
    assistant: dict[str, Any] | None = None

    def flush_user() -> None:
        if user_parts:
            messages.append({"role": "user", "content": _user_content(user_parts)})
            user_parts.clear()

    def flush_assistant() -> None:
        nonlocal assistant
        if assistant is not None:
            messages.append(_finish_assistant(assistant))
            assistant = None

    def current_assistant() -> dict[str, Any]:
        nonlocal assistant
        flush_user()
        if assistant is None:
            assistant = {"reasoning": [], "text": [], "tool_calls": []}
        return assistant

    for block in blocks:
        match block:
            case IRUserTextBlock():
                flush_assistant()
                user_parts.append({"type": "text", "text": block.text})
            case IRImageBlock():
                flush_assistant()
                user_parts.append(_image_part(block))
            case IRThinkingBlock():
                current = current_assistant()
                if current["text"] or current["tool_calls"]:
                    # Reasoning after visible output starts a new assistant turn.
                    flush_assistant()
                    current = current_assistant()
                if block.text:
                    current["reasoning"].append(block.text)
            case IRAssistantTextBlock():
                current = current_assistant()
                if current["tool_calls"]:
                    flush_assistant()
                    current = current_assistant()
                current["text"].append(block.text)
            case IRToolCallBlock():
                current_assistant()["tool_calls"].append(
                    {
                        "id": block.call_id,
                        "type": "function",
                        "function": {
                            "name": block.tool,
                            "arguments": json.dumps(block.input, ensure_ascii=False),
                        },
                    }
                )
            case IRToolResultBlock():
                flush_assistant()
                flush_user()
                text, images = _split_tool_result(block)
                messages.append(
                    {"role": "tool", "tool_call_id": block.call_id, "content": text}
                )
                if images:
                    # Chat Completions tool messages are text-only; images ride
                    # along in a user message right after the result.
                    user_parts.append(
                        {
                            "type": "text",
                            "text": f"[Image output from tool call {block.call_id}]",
                        }
                    )
                    user_parts.extend(images)
            case _:
                # Refusal blocks etc. are rendered upstream by the surface builder.
                pass

    flush_assistant()
    flush_user()
    return messages


def _finish_assistant(acc: dict[str, Any]) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": "\n\n".join(acc["text"]),
    }
    if acc["reasoning"]:
        message["reasoning_content"] = "\n\n".join(acc["reasoning"])
    if acc["tool_calls"]:
        message["tool_calls"] = acc["tool_calls"]
    return message


def _user_content(parts: list[dict[str, Any]]) -> str | list[dict[str, Any]]:
    if all(part["type"] == "text" for part in parts):
        return "\n\n".join(part["text"] for part in parts)
    return list(parts)


def _split_tool_result(
    block: IRToolResultBlock,
) -> tuple[str, list[dict[str, Any]]]:
    texts: list[str] = []
    images: list[dict[str, Any]] = []
    for content_block in block.content:
        match content_block:
            case IRToolTextBlock():
                texts.append(content_block.text)
            case IRImageBlock():
                images.append(_image_part(content_block))
    text = "".join(texts)
    if not text and images:
        text = f"[{len(images)} image(s) attached below]"
    return text, images


def _image_part(block: IRImageBlock) -> dict[str, Any]:
    match block.source:
        case IRImageBase64Source():
            url = f"data:{block.source.media_type};base64,{block.source.data}"
        case IRImageURLSource():
            url = block.source.url
        case IRImageBlobSource():
            raise ValueError(
                "Blob image sources must be hydrated before provider rendering."
            )
    return {"type": "image_url", "image_url": {"url": url}}


def chat_tool_schema(name: str, model: Any) -> dict[str, Any]:
    """Chat Completions tool schema (``{"type": "function", "function": {...}}``)."""
    responses_schema = _openai_tool_schema(name, model)
    return {
        "type": "function",
        "function": {
            "name": responses_schema["name"],
            "description": responses_schema["description"],
            "parameters": responses_schema["parameters"],
        },
    }


def _is_effort_rejection(exc: APIStatusError) -> bool:
    if exc.status_code not in (400, 422, 500):
        return False
    text = f"{exc.message} {_error_body_text(exc)}".lower()
    return "effort" in text or "reasoning" in text


def _error_body_text(exc: APIStatusError) -> str:
    body = getattr(exc, "body", None)
    if body is None:
        return ""
    try:
        return json.dumps(body)
    except (TypeError, ValueError):
        return str(body)


# --- Response helpers ---


def _reasoning_text(delta: Any) -> str:
    # llama.cpp / older vLLM: reasoning_content. Newer vLLM: reasoning.
    for key in ("reasoning_content", "reasoning"):
        value = _obj_get(delta, key)
        if value is None and hasattr(delta, "model_extra"):
            value = (delta.model_extra or {}).get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _thinking_block(text: str) -> IRThinkingBlock | None:
    text = text.strip()
    return IRThinkingBlock(text=text) if text else None


def _text_block(text: str) -> IRAssistantTextBlock | None:
    text = text.strip()
    return IRAssistantTextBlock(text=text) if text else None


def _tool_call_block(
    acc: _ToolCallAccumulator, *, truncated: bool
) -> IRToolCallBlock | None:
    if not acc.name:
        return None
    raw = "".join(acc.argument_parts).strip()
    try:
        tool_input = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        if truncated:
            return None  # cut off by max_tokens; don't execute half a call
        # Malformed but complete: keep it so the executor's validation error
        # goes back to the model, which can correct itself.
        logger.warning(
            "local.tool_call_invalid_json tool=%s raw=%r", acc.name, raw[:500]
        )
        tool_input = {"_invalid_json_arguments": raw}
    if not isinstance(tool_input, dict):
        tool_input = {"_non_object_arguments": tool_input}
    return IRToolCallBlock(
        call_id=acc.call_id or f"call_{uuid4().hex[:24]}",
        tool=acc.name,
        input=tool_input,
    )


def _normalize_usage(usage: Any) -> IRUsage:
    prompt_tokens = int(_obj_get(usage, "prompt_tokens") or 0)
    completion_tokens = int(_obj_get(usage, "completion_tokens") or 0)
    details = _obj_get(usage, "prompt_tokens_details")
    cached = int(_obj_get(details, "cached_tokens") or 0)
    cached = min(cached, prompt_tokens)
    return IRUsage(
        input_tokens=prompt_tokens - cached,
        output_tokens=completion_tokens,
        cache_read_tokens=cached,
        cache_create_tokens=0,
    )

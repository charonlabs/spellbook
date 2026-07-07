from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from anthropic import APIStatusError
import httpx
import pytest

from spellbook.backends.model_backend import (
    GenerationStream,
    ModelBackend,
    RequestSurface,
    TokenCounter,
)
from spellbook.cancel_token import CancelToken
from spellbook.config import SpellbookConfig
from spellbook.generator import Generator
from spellbook.ir_types import (
    IRAssistantTextBlock,
    IRBlock,
    IRGeneration,
    IRStreamEvent,
    IRStreamTextDeltaEvent,
    IRStreamTextEndEvent,
    IRStreamTextStartEvent,
    IRUsage,
    IRUserTextBlock,
    StopReason,
)
from spellbook.round_lifecycle import RoundLifecycle
from spellbook.surface_builder import RequestSurfaceBuilder
from spellbook.tools.registry import ToolRegistry

pytestmark = pytest.mark.asyncio


@dataclass(slots=True)
class _Attempt:
    events: list[IRStreamEvent | Exception]
    final: IRGeneration
    current: IRGeneration | None = None


class _FakeGenerationStream(GenerationStream):
    def __init__(self, attempt: _Attempt) -> None:
        self._attempt = attempt
        self._idx = 0

    async def __aenter__(self) -> _FakeGenerationStream:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    def __aiter__(self) -> _FakeGenerationStream:
        return self

    async def __anext__(self) -> IRStreamEvent:
        if self._idx >= len(self._attempt.events):
            raise StopAsyncIteration
        item = self._attempt.events[self._idx]
        self._idx += 1
        if isinstance(item, Exception):
            raise item
        return item

    async def get_final_response(self) -> IRGeneration:
        return self._attempt.final

    def get_current_response(self, *, stop_reason: StopReason) -> IRGeneration:
        if self._attempt.current is not None:
            return self._attempt.current
        return IRGeneration(
            model=self._attempt.final.model,
            blocks=[],
            stop_reason=stop_reason,
            usage=None,
        )


class _FakeBackend:
    provider = "fake"

    def __init__(self, attempts: list[_Attempt]) -> None:
        self._attempts = list(attempts)
        self.stream_calls = 0
        self.surfaces: list[RequestSurface] = []

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
            system=system,
            tools=tools,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{len(blocks)} blocks / {effort}",
                        }
                    ],
                }
            ],
            max_output_tokens=max_output_tokens,
        )

    def stream(
        self,
        surface: RequestSurface,
        cancel_token: CancelToken,
    ) -> _FakeGenerationStream:
        _ = cancel_token
        self.stream_calls += 1
        self.surfaces.append(surface)
        if not self._attempts:
            raise RuntimeError("fake backend ran out of attempts")
        return _FakeGenerationStream(self._attempts.pop(0))

    def build_tool_schemas(self, registry: ToolRegistry) -> list[dict[str, Any]]:
        _ = registry
        return []

    def build_token_counter(
        self,
        config: SpellbookConfig,
        surface_builder: RequestSurfaceBuilder,
    ) -> TokenCounter:
        _ = (config, surface_builder)
        raise NotImplementedError


class _RecordingLifecycle(RoundLifecycle):
    def __init__(self) -> None:
        self.stream_events: list[IRStreamEvent] = []

    async def on_stream_event(self, event: IRStreamEvent) -> None:
        self.stream_events.append(event)


async def _instant_sleep(delay_seconds: float) -> None:
    _ = delay_seconds


def _make_generator(
    backend: _FakeBackend,
    tmp_path: Path,
    *,
    max_retry_attempts: int = 5,
    sleep: Callable[[float], Coroutine[Any, Any, None]] = _instant_sleep,
) -> Generator:
    config = SpellbookConfig(
        cwd=tmp_path,
        model="claude-fable-5",
        system_prompt="system",
        max_output_tokens=1024,
    )
    surface_builder = RequestSurfaceBuilder(
        model=config.model,
        system_provider=lambda: config.system_prompt,
        tool_schemas=[],
        backend=cast(ModelBackend, backend),
        max_output_tokens=config.max_output_tokens,
        effort=config.effort,
    )
    return Generator(
        backend=cast(ModelBackend, backend),
        config=config,
        surface_builder=surface_builder,
        max_retry_attempts=max_retry_attempts,
        retry_jitter_seconds=lambda _delay: 0.0,
        sleep=sleep,
    )


def _generation(text: str, *, stop_reason: StopReason = "end_turn") -> IRGeneration:
    return IRGeneration(
        model="claude-fable-5",
        blocks=[IRAssistantTextBlock(text=text)],
        stop_reason=stop_reason,
        usage=IRUsage(input_tokens=10, output_tokens=5),
    )


def _api_status_error(
    *,
    status_code: int,
    error_type: str,
    message: str,
) -> APIStatusError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(
        status_code=status_code,
        request=request,
        headers={"request-id": "req_test"},
    )
    body = {
        "type": "error",
        "error": {
            "details": None,
            "type": error_type,
            "message": message,
        },
        "request_id": "req_test",
    }
    return APIStatusError(str(body), response=response, body=body)


@pytest.mark.parametrize(
    ("status_code", "error_type", "message"),
    [
        (500, "api_error", "Internal server error"),
        (529, "overloaded_error", "Overloaded"),
    ],
)
async def test_retryable_anthropic_status_error_retries_whole_generation(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    status_code: int,
    error_type: str,
    message: str,
) -> None:
    success = _generation("done")
    backend = _FakeBackend(
        [
            _Attempt(
                events=[
                    IRStreamTextStartEvent(),
                    IRStreamTextDeltaEvent(text="partial"),
                    _api_status_error(
                        status_code=status_code,
                        error_type=error_type,
                        message=message,
                    ),
                ],
                final=_generation("discarded partial"),
            ),
            _Attempt(
                events=[
                    IRStreamTextStartEvent(),
                    IRStreamTextDeltaEvent(text="done"),
                    IRStreamTextEndEvent(),
                ],
                final=success,
            ),
        ]
    )
    delays: list[float] = []

    async def fake_sleep(delay_seconds: float) -> None:
        delays.append(delay_seconds)

    generator = _make_generator(backend, tmp_path, sleep=fake_sleep)

    with caplog.at_level(logging.WARNING, logger="spellbook.generator"):
        result = await generator.run(
            [IRUserTextBlock(text="hello", origin="human")],
            CancelToken(),
            _RecordingLifecycle(),
        )

    assert result == success
    assert len(result.blocks) == 1
    block = result.blocks[0]
    assert isinstance(block, IRAssistantTextBlock)
    assert block.text == "done"
    assert backend.stream_calls == 2
    assert delays == [1.0]
    assert "generator.transient_api_error_retry" in caplog.text
    assert f"error_type={error_type}" in caplog.text


async def test_retry_exhaustion_raises_final_error(tmp_path: Path) -> None:
    first_error = _api_status_error(
        status_code=500,
        error_type="api_error",
        message="Internal server error",
    )
    final_error = _api_status_error(
        status_code=500,
        error_type="api_error",
        message="Internal server error",
    )
    backend = _FakeBackend(
        [
            _Attempt(events=[first_error], final=_generation("unused")),
            _Attempt(events=[final_error], final=_generation("unused")),
        ]
    )
    delays: list[float] = []

    async def fake_sleep(delay_seconds: float) -> None:
        delays.append(delay_seconds)

    generator = _make_generator(
        backend,
        tmp_path,
        max_retry_attempts=2,
        sleep=fake_sleep,
    )

    with pytest.raises(APIStatusError) as exc_info:
        await generator.run(
            [IRUserTextBlock(text="hello", origin="human")],
            CancelToken(),
            _RecordingLifecycle(),
        )

    assert exc_info.value is final_error
    assert backend.stream_calls == 2
    assert delays == [1.0]


async def test_4xx_status_error_is_not_retried(tmp_path: Path) -> None:
    error = _api_status_error(
        status_code=400,
        error_type="invalid_request_error",
        message="Bad request",
    )
    backend = _FakeBackend([_Attempt(events=[error], final=_generation("unused"))])
    delays: list[float] = []

    async def fake_sleep(delay_seconds: float) -> None:
        delays.append(delay_seconds)

    generator = _make_generator(backend, tmp_path, sleep=fake_sleep)

    with pytest.raises(APIStatusError) as exc_info:
        await generator.run(
            [IRUserTextBlock(text="hello", origin="human")],
            CancelToken(),
            _RecordingLifecycle(),
        )

    assert exc_info.value is error
    assert backend.stream_calls == 1
    assert delays == []


async def test_refusal_stream_is_not_retried(tmp_path: Path) -> None:
    refusal = _generation("refused", stop_reason="refusal")
    events: list[IRStreamEvent] = [
        IRStreamTextStartEvent(),
        IRStreamTextDeltaEvent(text="refused"),
        IRStreamTextEndEvent(),
    ]
    backend = _FakeBackend([_Attempt(events=list(events), final=refusal)])
    delays: list[float] = []

    async def fake_sleep(delay_seconds: float) -> None:
        delays.append(delay_seconds)

    lifecycle = _RecordingLifecycle()
    generator = _make_generator(backend, tmp_path, sleep=fake_sleep)

    result = await generator.run(
        [IRUserTextBlock(text="hello", origin="human")],
        CancelToken(),
        lifecycle,
    )

    assert result == refusal
    assert result.stop_reason == "refusal"
    assert backend.stream_calls == 1
    assert delays == []
    assert lifecycle.stream_events == events


async def test_cancel_during_backoff_returns_cancelled_generation(
    tmp_path: Path,
) -> None:
    error = _api_status_error(
        status_code=500,
        error_type="api_error",
        message="Internal server error",
    )
    backend = _FakeBackend([_Attempt(events=[error], final=_generation("unused"))])
    sleep_started = asyncio.Event()
    sleep_cancelled = False
    delays: list[float] = []

    async def fake_sleep(delay_seconds: float) -> None:
        nonlocal sleep_cancelled
        delays.append(delay_seconds)
        sleep_started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            sleep_cancelled = True
            raise

    token = CancelToken()
    generator = _make_generator(backend, tmp_path, sleep=fake_sleep)
    task = asyncio.create_task(
        generator.run(
            [IRUserTextBlock(text="hello", origin="human")],
            token,
            _RecordingLifecycle(),
        )
    )

    await asyncio.wait_for(sleep_started.wait(), timeout=1.0)
    token.cancel()
    result = await asyncio.wait_for(task, timeout=1.0)

    assert result.stop_reason == "cancelled"
    assert result.blocks == []
    assert backend.stream_calls == 1
    assert delays == [1.0]
    assert sleep_cancelled is True


async def test_backoff_timing_uses_exponential_delays_with_fake_clock(
    tmp_path: Path,
) -> None:
    success = _generation("done")
    backend = _FakeBackend(
        [
            _Attempt(
                events=[
                    _api_status_error(
                        status_code=500,
                        error_type="api_error",
                        message="Internal server error",
                    )
                ],
                final=_generation("unused"),
            ),
            _Attempt(
                events=[
                    _api_status_error(
                        status_code=500,
                        error_type="api_error",
                        message="Internal server error",
                    )
                ],
                final=_generation("unused"),
            ),
            _Attempt(
                events=[
                    _api_status_error(
                        status_code=529,
                        error_type="overloaded_error",
                        message="Overloaded",
                    )
                ],
                final=_generation("unused"),
            ),
            _Attempt(events=[], final=success),
        ]
    )
    delays: list[float] = []

    async def fake_sleep(delay_seconds: float) -> None:
        delays.append(delay_seconds)

    generator = _make_generator(backend, tmp_path, sleep=fake_sleep)

    result = await generator.run(
        [IRUserTextBlock(text="hello", origin="human")],
        CancelToken(),
        _RecordingLifecycle(),
    )

    assert result == success
    assert backend.stream_calls == 4
    assert delays == [1.0, 2.0, 4.0]

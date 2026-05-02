from __future__ import annotations

from pathlib import Path
from typing import cast

from fastapi.testclient import TestClient

from spellbook.app.event_bus import AppEventBus
from spellbook.app.protocol import (
    AwarenessResponse,
    AwarenessSnapshot,
    CatchupResponse,
    ConduitResponse,
    HealthResponse,
    MessageQueuedEvent,
    SubmitMessageResponse,
)
from spellbook.app.runtime import CoreAppRuntime
from spellbook.app.server import create_app
from spellbook.config import SpellbookConfig
from spellbook.homunculus.common import (
    AwarenessBudgetSnapshot,
    AwarenessHomunculusSnapshot,
    AwarenessTailSnapshot,
)
from spellbook.ir_types import IRInboundMessage, IRSkillCatalog
from spellbook.nursery import AwarenessNurserySnapshot
from spellbook.rehydrator import RehydrationResult


def _config(tmp_path: Path) -> SpellbookConfig:
    return SpellbookConfig(model="claude-sonnet-4-6", cwd=tmp_path)


def _rehydrated(tmp_path: Path) -> RehydrationResult:
    return RehydrationResult(
        session_id="session_fake",
        records=[],
        blocks=[],
        config=_config(tmp_path),
        tools=[],
        last_completed_turn=0,
        pending_footers={},
        completed_semantic_block_ranges=[],
        buffered_semantic_block_ranges=[],
        semantic_blocks=[],
        plan_proposal=None,
        skill_catalog=IRSkillCatalog(),
    )


class _FakeRuntime:
    def __init__(self, tmp_path: Path) -> None:
        self.bus = AppEventBus()
        self.started = False
        self.shutdown_called = False
        self.interrupt_called = False
        self.submitted: list[IRInboundMessage] = []
        self.conduits: list[dict] = []
        self._tmp_path = tmp_path

    async def startup(self) -> None:
        self.started = True

    async def shutdown(self) -> None:
        self.shutdown_called = True
        self.bus.close()

    def build_health(self) -> HealthResponse:
        return HealthResponse(
            model="claude-sonnet-4-6",
            state="idle",
            turns=0,
            gauge_input_tokens=None,
        )

    def build_catchup(self) -> CatchupResponse:
        return CatchupResponse(rehydrated=_rehydrated(self._tmp_path))

    def build_awareness(self) -> AwarenessResponse:
        return AwarenessResponse(
            snapshot=AwarenessSnapshot(
                homunculus=AwarenessHomunculusSnapshot(
                    budget=AwarenessBudgetSnapshot(
                        max_tokens=1_000_000,
                        reserve_output_tokens=67_000,
                        current_input_tokens=None,
                        current_slack_tokens=None,
                        regime="unknown",
                        warning_threshold=700_000,
                        forced_threshold=850_000,
                        critical_threshold=933_000,
                    ),
                    semantic_blocks=[],
                    proposed_blocks=[],
                    tail=AwarenessTailSnapshot(
                        tail_start=0,
                        tail_end=-1,
                        toks=None,
                    ),
                    plan_proposal=None,
                ),
                nursery=AwarenessNurserySnapshot(jobs=[]),
                surface="Telegram",
                surface_time=None,
            )
        )

    async def submit_message(self, message: IRInboundMessage) -> SubmitMessageResponse:
        self.submitted.append(message)
        self.bus.publish(MessageQueuedEvent(message=message))
        return SubmitMessageResponse(started=False, queued=True)

    async def handle_conduit(self, **kwargs) -> ConduitResponse:
        self.conduits.append(kwargs)
        return ConduitResponse(
            delivered=True,
            action="queued_as_context",
            source=kwargs.get("source", ""),
        )

    async def interrupt(self) -> bool:
        self.interrupt_called = True
        return True


def _make_app(tmp_path: Path) -> tuple[TestClient, _FakeRuntime]:
    runtime = _FakeRuntime(tmp_path)

    def _factory(
        transcript_path: Path,
        config: SpellbookConfig | None,
    ) -> CoreAppRuntime:
        return cast(CoreAppRuntime, runtime)

    app = create_app(
        transcript_path=tmp_path / "transcript.jsonl",
        config=_config(tmp_path),
        runtime_factory=_factory,
    )
    return TestClient(app), runtime


def test_lifespan_starts_and_stops_runtime(tmp_path: Path) -> None:
    client, runtime = _make_app(tmp_path)

    with client:
        assert runtime.started is True
        assert runtime.shutdown_called is False

    assert runtime.shutdown_called is True


def test_health_catchup_message_and_interrupt_routes(tmp_path: Path) -> None:
    client, runtime = _make_app(tmp_path)

    with client:
        health = client.get("/health")
        catchup = client.get("/catchup")
        awareness = client.get("/awareness")
        message = client.post(
            "/message",
            json={
                "text": "hello",
                "metadata": {"source": "web"},
                "inject": True,
            },
        )
        interrupt = client.post("/interrupt")

    assert health.status_code == 200
    assert health.json()["model"] == "claude-sonnet-4-6"
    assert catchup.status_code == 200
    assert catchup.json()["rehydrated"]["session_id"] == "session_fake"
    assert awareness.status_code == 200
    assert awareness.json()["kind"] == "awareness"
    assert awareness.json()["snapshot"]["homunculus"]["budget"]["regime"] == "unknown"
    assert awareness.json()["snapshot"]["surface"] == "Telegram"
    assert message.status_code == 200
    assert message.json()["queued"] is True
    assert interrupt.status_code == 200
    assert interrupt.json()["interrupted"] is True
    assert runtime.interrupt_called is True
    assert len(runtime.submitted) == 1
    assert runtime.submitted[0].delivery == "inject"
    assert runtime.submitted[0].source_metadata == {"source": "web"}


def test_message_rejects_empty_and_unknown_fields(tmp_path: Path) -> None:
    client, runtime = _make_app(tmp_path)

    with client:
        empty = client.post("/message", json={"text": "   "})
        footer = client.post(
            "/message",
            json={"text": "ambient context", "delivery": "footer"},
        )

    assert empty.status_code == 400
    assert footer.status_code == 422
    assert runtime.submitted == []


def test_local_web_origins_are_cors_allowed(tmp_path: Path) -> None:
    client, _runtime = _make_app(tmp_path)

    with client:
        response = client.options(
            "/message",
            headers={
                "origin": "http://localhost:3000",
                "access-control-request-method": "POST",
                "access-control-request-headers": "content-type",
            },
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert "POST" in response.headers["access-control-allow-methods"]


def test_conduit_route_validates_and_normalizes_metadata(tmp_path: Path) -> None:
    client, runtime = _make_app(tmp_path)

    with client:
        invalid_type = client.post(
            "/conduit",
            json={"type": "nope", "source": "telegram", "content": "hello"},
        )
        empty = client.post(
            "/conduit",
            json={"type": "context", "source": "telegram", "content": "   "},
        )
        valid = client.post(
            "/conduit",
            json={
                "type": "context",
                "source": "chorus.notification",
                "content": "Disk pressure warning",
                "metadata": ["not", "a", "dict"],
            },
        )

    assert invalid_type.status_code == 400
    assert invalid_type.json()["detail"] == "invalid conduit type: 'nope'"
    assert empty.status_code == 400
    assert empty.json()["detail"] == "empty content"
    assert valid.status_code == 200
    assert valid.json()["action"] == "queued_as_context"
    assert runtime.conduits == [
        {
            "conduit_type": "context",
            "source": "chorus.notification",
            "content": "Disk pressure warning",
            "metadata": None,
        }
    ]


def test_websocket_sends_catchup_then_live_events(tmp_path: Path) -> None:
    client, runtime = _make_app(tmp_path)

    with client:
        with client.websocket_connect("/ws") as ws:
            catchup = ws.receive_json()
            assert catchup["kind"] == "catchup"
            assert catchup["rehydrated"]["session_id"] == "session_fake"

            response = client.post("/message", json={"text": "hello websocket"})
            assert response.status_code == 200

            event = ws.receive_json()
            assert event["kind"] == "message_queued"
            assert event["message"]["blocks"][0]["text"] == "hello websocket"

    assert runtime.shutdown_called is True
    assert runtime.submitted[-1].delivery == "turn"

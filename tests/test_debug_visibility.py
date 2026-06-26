from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from spellbook.config import SpellbookConfig
from spellbook.debug_visibility import (
    DebugEmitter,
    settings_from_runtime_config_records,
)
from spellbook.homunculus.tool_result_ttl import (
    TTL_TRIGGER_END_TURN,
    ToolResultTTLRegistry,
)
from spellbook.ir_types import (
    IRRuntimeConfigRecord,
    IRSkillCatalog,
    IRSystemResponseRecord,
)
from spellbook.nursery import Nursery
from spellbook.recorder import Recorder
from spellbook.rehydrator import Rehydrator
from spellbook.session_lifecycle import SessionContext, SessionLifecycle
from spellbook.system_response import SystemResponse
from spellbook.tools.registry import DEFAULT_TOOL_REGISTRY


class _Lifecycle(SessionLifecycle):
    def __init__(self) -> None:
        self.responses: list[SystemResponse] = []

    async def on_system_response(
        self, ctx: SessionContext, response: SystemResponse
    ) -> None:
        self.responses.append(response)


def _recorder(tmp_path: Path) -> tuple[Recorder, Path]:
    transcript = tmp_path / "transcript.jsonl"
    recorder = Recorder(
        SpellbookConfig(cwd=tmp_path),
        transcript,
        "session_test",
        DEFAULT_TOOL_REGISTRY,
    )
    recorder.write_session_record(skill_catalog=IRSkillCatalog())
    return recorder, transcript


@pytest.mark.asyncio
async def test_debug_emitter_gates_lifecycle_notices_but_always_emits_alerts(
    tmp_path: Path,
) -> None:
    recorder, transcript = _recorder(tmp_path)
    lifecycle = _Lifecycle()
    emitter = DebugEmitter(recorder=recorder, session_lifecycle=lifecycle)
    emitter.bind_context(SessionContext(session_id="session_test", turn_idx=0))

    emitter.debug(subsystem="ttl", event="tick", title="TTL tick")
    await emitter.flush()

    assert lifecycle.responses == []
    assert Rehydrator(transcript).run().system_responses == []

    emitter.alert(
        subsystem="nursery",
        event="job_failed",
        title="Nursery job failed",
        metadata={"job_id": "job_1"},
    )
    await emitter.flush()

    rehydrated = Rehydrator(transcript).run()
    assert len(lifecycle.responses) == 1
    assert len(rehydrated.system_responses) == 1
    alert = rehydrated.system_responses[0]
    assert alert.command == "/debug"
    assert alert.metadata is not None
    assert alert.metadata["level"] == "error"
    assert alert.metadata["debug_only"] is False

    emitter.configure_enabled(True)
    emitter.debug(subsystem="ttl", event="tick", title="TTL tick")
    await emitter.flush()

    rehydrated = Rehydrator(transcript).run()
    debug_records = [
        record
        for record in rehydrated.records
        if isinstance(record, IRSystemResponseRecord) and record.command == "/debug"
    ]
    assert len(debug_records) == 2
    assert debug_records[-1].metadata is not None
    assert debug_records[-1].metadata["subsystem"] == "ttl"
    assert debug_records[-1].metadata["debug_only"] is True


def test_debug_visibility_rehydrates_from_operator_runtime_config() -> None:
    records = [
        IRRuntimeConfigRecord(
            session_id="session_test",
            namespace="debug_visibility",
            updates={"enabled": True},
            effective={"enabled": True},
            source="operator",
            turn=0,
            turn_id="",
        )
    ]

    settings = settings_from_runtime_config_records(records)

    assert settings.enabled is True


@pytest.mark.asyncio
async def test_nursery_job_failure_emits_always_on_debug_alert(tmp_path: Path) -> None:
    recorder, transcript = _recorder(tmp_path)
    lifecycle = _Lifecycle()
    emitter = DebugEmitter(recorder=recorder, session_lifecycle=lifecycle)
    emitter.bind_context(SessionContext(session_id="session_test", turn_idx=0))
    nursery = Nursery(config=SpellbookConfig(cwd=tmp_path), debug_emitter=emitter)

    async def _fail() -> str:
        raise RuntimeError("background boom")

    nursery.submit(_fail(), kind="bash", source="bash", key="bash:1")
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    await emitter.flush()

    rehydrated = Rehydrator(transcript).run()
    assert len(rehydrated.system_responses) == 1
    response = rehydrated.system_responses[0]
    assert response.command == "/debug"
    assert response.metadata is not None
    assert response.metadata["subsystem"] == "nursery"
    assert response.metadata["event"] == "job_failed"
    assert response.metadata["job_kind"] == "bash"
    assert response.metadata["error_message"] == "background boom"


@pytest.mark.asyncio
async def test_nursery_lifecycle_debug_events_are_debug_gated(
    tmp_path: Path,
) -> None:
    recorder, transcript = _recorder(tmp_path)
    lifecycle = _Lifecycle()
    emitter = DebugEmitter(recorder=recorder, session_lifecycle=lifecycle)
    emitter.bind_context(SessionContext(session_id="session_test", turn_idx=0))
    nursery = Nursery(config=SpellbookConfig(cwd=tmp_path), debug_emitter=emitter)

    async def _ok() -> str:
        return "done"

    nursery.submit(_ok(), kind="bash", source="bash")
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    nursery.collect_ready(source="bash")
    await emitter.flush()

    assert Rehydrator(transcript).run().system_responses == []

    emitter.configure_enabled(True)
    nursery.submit(_ok(), kind="bash", source="bash")
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    nursery.collect_ready(source="bash")
    await emitter.flush()

    responses = Rehydrator(transcript).run().system_responses
    events = [
        response.metadata["event"]
        for response in responses
        if response.metadata is not None
        and response.metadata.get("subsystem") == "nursery"
    ]
    assert events == ["job_started", "job_completed", "job_harvested"]


@pytest.mark.asyncio
async def test_ttl_registration_and_tick_emit_debug_events(tmp_path: Path) -> None:
    recorder, transcript = _recorder(tmp_path)
    recorder.start_turn("turn_1", [])
    lifecycle = _Lifecycle()
    emitter = DebugEmitter(recorder=recorder, session_lifecycle=lifecycle)
    emitter.bind_context(SessionContext(session_id="session_test", turn_idx=0))
    emitter.configure_enabled(True)
    registry = ToolResultTTLRegistry(
        config=SpellbookConfig(cwd=tmp_path).hom_config,
        recorder=recorder,
        debug_emitter=emitter,
    )

    registry.register(
        call_id="toolu_big",
        replace_content="[collapsed]",
        ttl=1,
        trigger=TTL_TRIGGER_END_TURN,
    )
    registry.tick(TTL_TRIGGER_END_TURN)
    await emitter.flush()

    responses = Rehydrator(transcript).run().system_responses
    events = [
        response.metadata["event"]
        for response in responses
        if response.metadata is not None and response.metadata.get("subsystem") == "ttl"
    ]
    assert events == ["registered", "tick"]
    tick = next(
        response
        for response in responses
        if response.metadata is not None and response.metadata.get("event") == "tick"
    )
    assert tick.metadata is not None
    assert tick.metadata["collapsed_call_ids"] == ["toolu_big"]

from __future__ import annotations

from pathlib import Path

import pytest

from spellbook.config import SpellbookConfig
from spellbook.debug_visibility import (
    DebugEmitter,
    settings_from_runtime_config_records,
)
from spellbook.ir_types import (
    IRRuntimeConfigRecord,
    IRSkillCatalog,
    IRSystemResponseRecord,
)
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

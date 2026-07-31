"""Config overrides are append-only, resume-only, and poison-pill resistant."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from spellbook.config import SpellbookConfig
from spellbook.config_override import (
    CONFIG_OVERRIDE_DISCLOSURE_FOOTER_KEY,
    ConfigOverrideValidationError,
    validate_override,
)
from spellbook.ir_types import IRConfigOverrideRecord, IRSkillCatalog
from spellbook.recorder import Recorder
from spellbook.rehydrator import Rehydrator
from spellbook.tools.registry import DEFAULT_TOOL_REGISTRY


def _make_recorder(tmp_path: Path) -> tuple[Recorder, Path, SpellbookConfig]:
    transcript = tmp_path / "transcript.jsonl"
    config = SpellbookConfig(
        model="claude-sonnet-4-6",
        cwd=tmp_path,
        skill_discovery_dirs=[],
    )
    recorder = Recorder(
        config,
        transcript,
        "session_config_override",
        DEFAULT_TOOL_REGISTRY,
    )
    recorder.write_session_record(skill_catalog=IRSkillCatalog())
    return recorder, transcript, config


@pytest.mark.parametrize("field", ["model", "provider", "session_type"])
def test_write_time_validation_denies_frozen_identity_fields(
    tmp_path: Path, field: str
) -> None:
    recorder, transcript, _config = _make_recorder(tmp_path)
    before = transcript.read_text()

    with pytest.raises(
        ConfigOverrideValidationError,
        match=rf"frozen identity field\(s\): {field}",
    ):
        recorder.write_config_override(
            updates={field: "forbidden"},
            source="cli",
            actor="Ryan",
        )

    assert transcript.read_text() == before


def test_write_time_validation_rejects_unknown_fields() -> None:
    with pytest.raises(
        ConfigOverrideValidationError,
        match="unknown SpellbookConfig field.*not_a_real_field",
    ):
        validate_override({"not_a_real_field": True})


def test_resume_merge_is_ordered_last_wins_and_deterministic(tmp_path: Path) -> None:
    recorder, transcript, _config = _make_recorder(tmp_path)
    recorder.write_config_override(
        updates={"hearth_interval_minutes": 40},
        source="configurator",
        actor="Ryan",
        note="first adjustment",
    )
    recorder.write_config_override(
        updates={"hearth_interval_minutes": 35, "hearth_enabled": True},
        source="cli",
        actor="Fable",
    )

    first = Rehydrator(transcript).run()
    second = Rehydrator(transcript).run()

    assert first == second
    assert first.config.hearth_interval_minutes == 35
    assert first.config.hearth_enabled is True
    assert [record.source for record in first.config_override_updates] == [
        "configurator",
        "cli",
    ]
    disclosure = first.pending_footers[CONFIG_OVERRIDE_DISCLOSURE_FOOTER_KEY]
    assert disclosure.priority < 0
    assert (
        "since you last ran, your config changed: hearth_interval_minutes 55->40 "
        "(source: configurator, by Ryan)"
    ) in disclosure.text
    assert (
        "since you last ran, your config changed: hearth_interval_minutes 40->35 "
        "(source: cli, by Fable)"
    ) in disclosure.text
    assert (
        "since you last ran, your config changed: hearth_enabled false->true "
        "(source: cli, by Fable)"
    ) in disclosure.text


def test_merge_refuses_entire_denied_record_and_continues(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _recorder, transcript, original = _make_recorder(tmp_path)
    poison = IRConfigOverrideRecord(
        session_id="session_config_override",
        source="configurator",
        actor="Ryan",
        updates={
            "model": "different-model",
            "hearth_interval_minutes": 40,
        },
        note="must be refused atomically",
    )
    with transcript.open("a") as f:
        f.write(poison.model_dump_json() + "\n")

    with caplog.at_level(logging.CRITICAL, logger="spellbook.rehydrator"):
        result = Rehydrator(transcript).run()

    assert result.config.model == original.model
    assert result.config.hearth_interval_minutes == original.hearth_interval_minutes
    assert "config_override.refused" in caplog.text
    assert "frozen identity field(s): model" in caplog.text
    disclosure = result.pending_footers[CONFIG_OVERRIDE_DISCLOSURE_FOOTER_KEY]
    assert (
        "a config override record was refused: attempted to change model "
        "(source: configurator, by Ryan)"
    ) in disclosure.text


def test_merge_refuses_malformed_record_without_bricking_resume(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _recorder, transcript, original = _make_recorder(tmp_path)
    malformed = {
        "session_id": "session_config_override",
        "ir": "config_override",
        "time": "2026-07-31T12:00:00Z",
        "source": "configurator",
        "updates": ["not", "a", "mapping"],
        "note": None,
    }
    with transcript.open("a") as f:
        f.write(json.dumps(malformed) + "\n")

    with caplog.at_level(logging.CRITICAL, logger="spellbook.rehydrator"):
        result = Rehydrator(transcript).run()

    assert result.config == original
    assert result.config_override_updates == []
    assert "config_override.refused" in caplog.text
    disclosure = result.pending_footers[CONFIG_OVERRIDE_DISCLOSURE_FOOTER_KEY]
    assert (
        "a config override record was refused: malformed record (source: configurator)"
    ) in disclosure.text

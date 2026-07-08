from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from spellbook.config import SpellbookConfig
from spellbook.profiles import (
    BLOCK_DETECTOR,
    BLOCK_SUMMARIZER,
    CUSTOM,
    MAIN,
    QUANTUM,
    SessionProfile,
    SessionType,
)


def test_openai_config_uses_provider_specific_skill_dirs() -> None:
    config = SpellbookConfig(provider="openai", model="gpt-5.5", cwd=Path.cwd())

    assert config.skill_discovery_dirs == [".agents", ".spellbook"]


def test_explicit_skill_dirs_override_provider_defaults() -> None:
    config = SpellbookConfig(
        provider="openai",
        model="gpt-5.5",
        cwd=Path.cwd(),
        skill_discovery_dirs=[".custom"],
    )

    assert config.skill_discovery_dirs == [".custom"]


def test_hearth_config_defaults_to_disabled() -> None:
    config = SpellbookConfig(cwd=Path.cwd())

    assert config.hearth_enabled is False
    assert config.hearth_interval_minutes == 55
    assert config.hearth_quiet_hours == ""


def test_hearth_config_validates_interval_and_quiet_hours() -> None:
    with pytest.raises(ValidationError):
        SpellbookConfig(cwd=Path.cwd(), hearth_interval_minutes=4)

    with pytest.raises(ValidationError):
        SpellbookConfig(cwd=Path.cwd(), hearth_quiet_hours="25:00-07:00")

    config = SpellbookConfig(cwd=Path.cwd(), hearth_quiet_hours="23:00-07:00")
    assert config.hearth_quiet_hours == "23:00-07:00"


def test_body_url_defaults_to_none_and_can_be_set() -> None:
    config = SpellbookConfig(cwd=Path.cwd())

    assert config.body_url is None

    body_config = SpellbookConfig(cwd=Path.cwd(), body_url="http://127.0.0.1:8765")
    assert body_config.body_url == "http://127.0.0.1:8765"


@pytest.mark.parametrize(
    ("session_type", "expected_profile"),
    [
        ("main", MAIN),
        ("custom", CUSTOM),
        ("block_detector", BLOCK_DETECTOR),
        ("block_summarizer", BLOCK_SUMMARIZER),
        ("quantum", QUANTUM),
    ],
)
def test_session_type_resolves_profile_preset(
    session_type: SessionType, expected_profile: SessionProfile
) -> None:
    config = SpellbookConfig(cwd=Path.cwd(), session_type=session_type)

    assert config.profile == expected_profile


def test_model_copy_session_type_update_refreshes_profile() -> None:
    config = SpellbookConfig(cwd=Path.cwd())

    updated = config.model_copy(update={"session_type": "block_detector"})

    assert updated.session_type == "block_detector"
    assert updated.profile == BLOCK_DETECTOR


def test_explicit_profile_overrides_session_type_default() -> None:
    config = SpellbookConfig(
        cwd=Path.cwd(),
        session_type="block_detector",
        profile=MAIN,
    )

    assert config.session_type == "block_detector"
    assert config.profile == MAIN

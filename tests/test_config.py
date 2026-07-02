from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from spellbook.config import SpellbookConfig


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

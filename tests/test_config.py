from __future__ import annotations

from pathlib import Path

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

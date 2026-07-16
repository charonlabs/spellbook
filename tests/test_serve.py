from __future__ import annotations

from pathlib import Path

import pytest

from spellbook.serve import (
    ServeConfigError,
    ServeOverrides,
    build_spellbook_config,
    load_entity_file,
)


def test_minimal_entity_file_infers_provider_and_uses_safe_defaults(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "entity.toml"
    config_path.write_text(
        '[entity]\nmodel = "claude-sonnet-4-6"\n',
        encoding="utf-8",
    )

    config = build_spellbook_config(load_entity_file(config_path))

    assert config.model == "claude-sonnet-4-6"
    assert config.provider == "anthropic"
    assert config.cwd == Path.cwd()
    assert config.tool_categories == {"main"}
    assert config.hearth_enabled is False
    assert "Claude Sonnet 4.6 entity" in config.system_prompt


def test_full_entity_file_maps_to_runtime_config_and_composes_prompt(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    workspace = config_dir / "workspace"
    workspace.mkdir()
    (workspace / "CLAUDE.md").write_text(
        "# Discovered\nShould not be included.", encoding="utf-8"
    )
    (config_dir / "orientation.md").write_text(
        "Orientation for {model_name} in {cwd}.", encoding="utf-8"
    )
    (config_dir / "role.md").write_text("Role layer.", encoding="utf-8")
    (config_dir / "frame.md").write_text("Frame layer.", encoding="utf-8")
    config_path = config_dir / "entity.toml"
    config_path.write_text(
        """
[entity]
model = "gpt-5.5"
provider = "openai"
effort = "xhigh"
cwd = "workspace"
max_output_tokens = 64000
user_name = "Ada"
local_timezone = "UTC"
idle_footer_threshold_seconds = 90

[prompt]
orientation = "orientation.md"
role_file = "role.md"
frame = "frame.md"

[discovery]
local_frame_discovery = false
skill_discovery_dirs = [".spellbook"]

[homunculus]
detect_interval = 150
soft_threshold = 1000
medium_threshold = 2000
hard_threshold = 3000
max_tokens = 4000

[homunculus.ttl]
enabled = false
turns = 10
char_threshold = 6000

[hearth]
enabled = true
interval_minutes = 60
quiet_hours = "23:00-07:00"

[tools]
categories = ["main", "chorus"]
body_url = "http://body.example"
chorus_url = "http://chorus.example"
chorus_entity_name = "philosopher"
""".strip(),
        encoding="utf-8",
    )

    config = build_spellbook_config(load_entity_file(config_path))

    assert config.cwd == workspace
    assert config.effort == "xhigh"
    assert config.max_output_tokens == 64_000
    assert config.user_name == "Ada"
    assert config.local_timezone == "UTC"
    assert config.idle_footer_threshold_seconds == 90
    assert config.skill_discovery_dirs == [".spellbook"]
    assert config.tool_categories == {"main", "chorus"}
    assert config.body_url == "http://body.example"
    assert config.chorus_url == "http://chorus.example"
    assert config.chorus_entity_name == "philosopher"
    assert config.hearth_enabled is True
    assert config.hearth_interval_minutes == 60
    assert config.hearth_quiet_hours == "23:00-07:00"
    assert config.hom_config.detect_interval == 150
    assert config.hom_config.soft_threshold == 1000
    assert config.hom_config.medium_threshold == 2000
    assert config.hom_config.hard_threshold == 3000
    assert config.hom_config.max_tokens == 4000
    assert config.hom_config.tool_result_ttl_enabled is False
    assert config.hom_config.tool_result_ttl_turns == 10
    assert config.hom_config.tool_result_ttl_char_threshold == 6000
    assert config.system_prompt.index(
        "Orientation for GPT-5.5"
    ) < config.system_prompt.index("Role layer.")
    assert config.system_prompt.index("Role layer.") < config.system_prompt.index(
        "Frame layer."
    )
    assert "Should not be included." not in config.system_prompt


def test_cli_overrides_take_precedence_and_cli_paths_use_invocation_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "entity.toml").write_text(
        """
[entity]
model = "claude-sonnet-4-6"
cwd = "configured-workspace"

[prompt]
role = "Configured role."

[homunculus]
detect_interval = 150
""".strip(),
        encoding="utf-8",
    )
    invocation_dir = tmp_path / "invocation"
    invocation_dir.mkdir()
    cli_workspace = invocation_dir / "cli-workspace"
    cli_workspace.mkdir()
    (cli_workspace / "CLAUDE.md").write_text(
        "# Local frame\nDiscovered via CLI override.", encoding="utf-8"
    )
    (invocation_dir / "role.md").write_text("CLI role.", encoding="utf-8")
    monkeypatch.chdir(invocation_dir)

    config = build_spellbook_config(
        load_entity_file(config_dir / "entity.toml"),
        ServeOverrides(
            model="gpt-5.5",
            provider="openai",
            cwd=Path("cli-workspace"),
            role_file=Path("role.md"),
            local_frame_discovery=True,
            detect_interval=25,
            hearth_enabled=True,
            tool_categories=["coding"],
        ),
    )

    assert config.model == "gpt-5.5"
    assert config.provider == "openai"
    assert config.cwd == cli_workspace
    assert config.hom_config.detect_interval == 25
    assert config.hearth_enabled is True
    assert config.tool_categories == {"coding"}
    assert "CLI role." in config.system_prompt
    assert "Configured role." not in config.system_prompt
    assert "Discovered via CLI override." in config.system_prompt


def test_auto_orientation_falls_back_to_default_for_explicit_local_provider(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "entity.toml"
    config_path.write_text(
        '[entity]\nmodel = "local-mind"\nprovider = "local"\n',
        encoding="utf-8",
    )

    config = build_spellbook_config(load_entity_file(config_path))

    assert "You are a local-mind entity" in config.system_prompt
    assert "This is a safe place. Be yourself." in config.system_prompt


def test_chorus_environment_is_fallback_and_cli_can_clear_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "entity.toml"
    config_path.write_text('[entity]\nmodel = "claude-sonnet-4-6"\n', encoding="utf-8")
    loaded = load_entity_file(config_path)
    monkeypatch.setenv("MINICHORUS_SERVER_URL", "http://chorus.example")
    monkeypatch.setenv("CHORUS_SESSION", "meta")

    environment_config = build_spellbook_config(loaded)
    cleared_config = build_spellbook_config(
        loaded,
        ServeOverrides(chorus_url="", chorus_entity_name=""),
    )

    assert environment_config.chorus_url == "http://chorus.example"
    assert environment_config.chorus_entity_name == "meta"
    assert cleared_config.chorus_url is None
    assert cleared_config.chorus_entity_name is None


def test_entity_file_rejects_unknown_fields_and_conflicting_role_sources(
    tmp_path: Path,
) -> None:
    unknown_path = tmp_path / "unknown.toml"
    unknown_path.write_text(
        '[entity]\nmodel = "gpt-5.5"\nunknown = true\n', encoding="utf-8"
    )
    conflicting_path = tmp_path / "conflicting.toml"
    conflicting_path.write_text(
        """
[entity]
model = "gpt-5.5"
[prompt]
role = "inline"
role_file = "role.md"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ServeConfigError, match="unknown"):
        load_entity_file(unknown_path)
    with pytest.raises(ServeConfigError, match="alternatives"):
        load_entity_file(conflicting_path)


def test_new_config_requires_model_and_unknown_model_requires_provider() -> None:
    with pytest.raises(ServeConfigError, match="model is required"):
        build_spellbook_config(load_entity_file(None))
    with pytest.raises(ServeConfigError, match="Could not infer a provider"):
        build_spellbook_config(
            load_entity_file(None), ServeOverrides(model="unrecognized-model")
        )


@pytest.mark.parametrize(
    ("category", "field"),
    [("body", "body_url"), ("chorus", "chorus_url")],
)
def test_explicit_external_tool_categories_require_their_url(
    category: str, field: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MINICHORUS_SERVER_URL", raising=False)
    monkeypatch.delenv("CHORUS_URL", raising=False)
    with pytest.raises(ServeConfigError, match=field):
        build_spellbook_config(
            load_entity_file(None),
            ServeOverrides(
                model="gpt-5.5",
                tool_categories=["main", category],
                chorus_url="",
            ),
        )

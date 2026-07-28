"""Typed entity-file loading and runtime config assembly for ``spellbook serve``."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from spellbook.backends import infer_provider_for_model
from spellbook.config import (
    DEFAULT_DETECT_INTERVAL,
    DEFAULT_EFFORT,
    DEFAULT_HARD_THRESHOLD,
    DEFAULT_HEARTH_ENABLED,
    DEFAULT_HEARTH_INTERVAL_MINUTES,
    DEFAULT_HEARTH_QUIET_HOURS,
    DEFAULT_IDLE_FOOTER_THRESHOLD_SECONDS,
    DEFAULT_LOCAL_TIMEZONE,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MEDIUM_THRESHOLD,
    DEFAULT_SOFT_THRESHOLD,
    DEFAULT_TOOL_RESULT_TTL_CHAR_THRESHOLD,
    DEFAULT_TOOL_RESULT_TTL_TURNS,
    DEFAULT_USER_NAME,
    HomunculusConfig,
    Provider,
    SpellbookConfig,
)
from spellbook.frame_lite import build_system_prompt_with_addenda
from spellbook.orientation import (
    build_core_orientation,
    build_default_orientation,
    build_orientation_from_file,
)
from spellbook.profiles import SessionType

CHORUS_SESSION_ENV = "CHORUS_SESSION"
CHORUS_URL_ENV = "CHORUS_URL"
MINICHORUS_SERVER_URL_ENV = "MINICHORUS_SERVER_URL"


class ServeConfigError(ValueError):
    """A user-facing error in an entity TOML file or serve override."""


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EntitySection(_Section):
    model: str | None = None
    provider: Provider | None = None
    effort: str = DEFAULT_EFFORT
    cwd: Path | None = None
    max_output_tokens: int = Field(default=DEFAULT_MAX_OUTPUT_TOKENS, gt=0)
    user_name: str = DEFAULT_USER_NAME
    local_timezone: str = DEFAULT_LOCAL_TIMEZONE
    idle_footer_threshold_seconds: int = Field(
        default=DEFAULT_IDLE_FOOTER_THRESHOLD_SECONDS, ge=0
    )
    session_type: SessionType = "main"


class PromptSection(_Section):
    orientation: str = "auto"
    role: str | None = None
    role_file: Path | None = None
    frame: str | None = None

    @model_validator(mode="after")
    def _one_role_source(self) -> Self:
        if self.role is not None and self.role_file is not None:
            raise ValueError("prompt.role and prompt.role_file are alternatives")
        return self


class DiscoverySection(_Section):
    local_frame_discovery: bool = False
    skill_discovery_dirs: list[str] | None = None


class TTLSection(_Section):
    enabled: bool = True
    turns: int = Field(default=DEFAULT_TOOL_RESULT_TTL_TURNS, ge=0)
    char_threshold: int = Field(default=DEFAULT_TOOL_RESULT_TTL_CHAR_THRESHOLD, ge=0)


class HomunculusSection(_Section):
    detect_interval: int = Field(default=DEFAULT_DETECT_INTERVAL, gt=0)
    soft_threshold: int = DEFAULT_SOFT_THRESHOLD
    medium_threshold: int = DEFAULT_MEDIUM_THRESHOLD
    hard_threshold: int = DEFAULT_HARD_THRESHOLD
    max_tokens: int = DEFAULT_MAX_TOKENS
    ttl: TTLSection = Field(default_factory=TTLSection)


class HearthSection(_Section):
    enabled: bool = DEFAULT_HEARTH_ENABLED
    interval_minutes: int = Field(default=DEFAULT_HEARTH_INTERVAL_MINUTES, ge=5)
    quiet_hours: str = DEFAULT_HEARTH_QUIET_HOURS


class ToolsSection(_Section):
    categories: list[str] = Field(default_factory=lambda: ["main"])
    body_url: str = ""
    minecraft_url: str = ""
    chorus_url: str = ""
    chorus_entity_name: str = ""


class EntityFile(_Section):
    """The validated top-level shape of an entity TOML file."""

    entity: EntitySection = Field(default_factory=EntitySection)
    prompt: PromptSection = Field(default_factory=PromptSection)
    discovery: DiscoverySection = Field(default_factory=DiscoverySection)
    homunculus: HomunculusSection = Field(default_factory=HomunculusSection)
    hearth: HearthSection = Field(default_factory=HearthSection)
    tools: ToolsSection = Field(default_factory=ToolsSection)


@dataclass(frozen=True)
class LoadedEntityFile:
    config: EntityFile
    path: Path | None

    @property
    def base_dir(self) -> Path:
        return self.path.parent if self.path is not None else Path.cwd()


@dataclass(frozen=True)
class ServeOverrides:
    """CLI values that take precedence over one entity file."""

    model: str | None = None
    provider: Provider | None = None
    effort: str | None = None
    cwd: Path | None = None
    max_output_tokens: int | None = None
    user_name: str | None = None
    local_timezone: str | None = None
    idle_footer_threshold_seconds: int | None = None
    session_type: SessionType | None = None
    orientation: str | None = None
    role: str | None = None
    role_file: Path | None = None
    frame: str | None = None
    local_frame_discovery: bool | None = None
    skill_discovery_dirs: list[str] | None = None
    detect_interval: int | None = None
    soft_threshold: int | None = None
    medium_threshold: int | None = None
    hard_threshold: int | None = None
    max_tokens: int | None = None
    ttl_enabled: bool | None = None
    ttl_turns: int | None = None
    ttl_char_threshold: int | None = None
    hearth_enabled: bool | None = None
    hearth_interval_minutes: int | None = None
    hearth_quiet_hours: str | None = None
    tool_categories: list[str] | None = None
    body_url: str | None = None
    minecraft_url: str | None = None
    chorus_url: str | None = None
    chorus_entity_name: str | None = None


def load_entity_file(path: Path | None) -> LoadedEntityFile:
    """Load and validate an entity TOML file, or return the default shape."""

    if path is None:
        return LoadedEntityFile(config=EntityFile(), path=None)
    resolved = path.expanduser().resolve()
    try:
        with resolved.open("rb") as config_file:
            raw_config = tomllib.load(config_file)
    except FileNotFoundError as exc:
        raise ServeConfigError(f"Entity config does not exist: {resolved}") from exc
    except OSError as exc:
        raise ServeConfigError(
            f"Could not read entity config {resolved}: {exc}"
        ) from exc
    except tomllib.TOMLDecodeError as exc:
        raise ServeConfigError(f"Invalid TOML in {resolved}: {exc}") from exc
    try:
        config = EntityFile.model_validate(raw_config)
    except ValidationError as exc:
        raise ServeConfigError(f"Invalid entity config {resolved}:\n{exc}") from exc
    return LoadedEntityFile(config=config, path=resolved)


def build_spellbook_config(
    loaded: LoadedEntityFile,
    overrides: ServeOverrides = ServeOverrides(),
) -> SpellbookConfig:
    """Merge CLI overrides over TOML and construct the canonical runtime config."""

    source = loaded.config
    entity = source.entity
    model = overrides.model or entity.model
    if model is None:
        raise ServeConfigError(
            "A model is required when initializing a transcript; set "
            "entity.model or pass --model."
        )
    provider = overrides.provider or entity.provider
    if provider is None:
        try:
            provider = infer_provider_for_model(model)
        except ValueError as exc:
            raise ServeConfigError(
                f"Could not infer a provider for model {model!r}; set "
                "entity.provider or pass --provider."
            ) from exc

    cwd = _resolve_path(
        overrides.cwd if overrides.cwd is not None else entity.cwd,
        base_dir=Path.cwd() if overrides.cwd is not None else loaded.base_dir,
        default=Path.cwd(),
    )
    user_name = overrides.user_name or entity.user_name
    system_prompt = compose_system_prompt(
        loaded,
        model=model,
        cwd=cwd,
        user_name=user_name,
        overrides=overrides,
    )
    ttl = source.homunculus.ttl
    hom_config = HomunculusConfig(
        detect_interval=_prefer(
            overrides.detect_interval, source.homunculus.detect_interval
        ),
        soft_threshold=_prefer(
            overrides.soft_threshold, source.homunculus.soft_threshold
        ),
        medium_threshold=_prefer(
            overrides.medium_threshold, source.homunculus.medium_threshold
        ),
        hard_threshold=_prefer(
            overrides.hard_threshold, source.homunculus.hard_threshold
        ),
        max_tokens=_prefer(overrides.max_tokens, source.homunculus.max_tokens),
        tool_result_ttl_enabled=_prefer(overrides.ttl_enabled, ttl.enabled),
        tool_result_ttl_turns=_prefer(overrides.ttl_turns, ttl.turns),
        tool_result_ttl_char_threshold=_prefer(
            overrides.ttl_char_threshold, ttl.char_threshold
        ),
    )
    tools = source.tools
    tool_categories = _prefer(overrides.tool_categories, tools.categories)
    body_url = _override_optional_url(overrides.body_url, tools.body_url)
    minecraft_url = _override_optional_url(overrides.minecraft_url, tools.minecraft_url)
    chorus_url = _chorus_url(overrides, tools)
    chorus_entity_name = _chorus_entity_name(overrides, tools)
    _validate_tool_dependencies(
        categories=tool_categories,
        body_url=body_url,
        minecraft_url=minecraft_url,
        chorus_url=chorus_url,
    )

    config_values: dict[str, object] = {
        "provider": provider,
        "model": model,
        "effort": _prefer(overrides.effort, entity.effort),
        "max_output_tokens": _prefer(
            overrides.max_output_tokens, entity.max_output_tokens
        ),
        "local_timezone": _prefer(overrides.local_timezone, entity.local_timezone),
        "idle_footer_threshold_seconds": _prefer(
            overrides.idle_footer_threshold_seconds,
            entity.idle_footer_threshold_seconds,
        ),
        "hearth_enabled": _prefer(overrides.hearth_enabled, source.hearth.enabled),
        "hearth_interval_minutes": _prefer(
            overrides.hearth_interval_minutes, source.hearth.interval_minutes
        ),
        "hearth_quiet_hours": _prefer(
            overrides.hearth_quiet_hours, source.hearth.quiet_hours
        ),
        "user_name": user_name,
        "tool_categories": set(tool_categories),
        "chorus_url": chorus_url,
        "chorus_entity_name": chorus_entity_name,
        "body_url": body_url,
        "minecraft_url": minecraft_url,
        "session_type": _prefer(overrides.session_type, entity.session_type),
        "cwd": cwd,
        "system_prompt": system_prompt,
        "hom_config": hom_config,
    }
    skill_dirs = _prefer(
        overrides.skill_discovery_dirs, source.discovery.skill_discovery_dirs
    )
    if skill_dirs is not None:
        config_values["skill_discovery_dirs"] = skill_dirs
    try:
        return SpellbookConfig.model_validate(config_values)
    except ValidationError as exc:
        raise ServeConfigError(f"Invalid merged entity configuration:\n{exc}") from exc


def compose_system_prompt(
    loaded: LoadedEntityFile,
    *,
    model: str,
    cwd: Path,
    user_name: str,
    overrides: ServeOverrides = ServeOverrides(),
) -> str:
    """Compose orientation, role, frame, and optional discovered local frame."""

    prompt = loaded.config.prompt
    orientation_value = overrides.orientation or prompt.orientation
    orientation_base = (
        Path.cwd() if overrides.orientation is not None else loaded.base_dir
    )
    orientation = _build_orientation(
        orientation_value,
        model=model,
        cwd=cwd,
        user_name=user_name,
        base_dir=orientation_base,
    )

    if overrides.role is not None:
        role = overrides.role
    elif overrides.role_file is not None:
        role = _read_prompt_file(
            overrides.role_file, base_dir=Path.cwd(), label="role file"
        )
    elif prompt.role is not None:
        role = prompt.role
    elif prompt.role_file is not None:
        role = _read_prompt_file(
            prompt.role_file, base_dir=loaded.base_dir, label="role file"
        )
    else:
        role = None

    frame_value = overrides.frame if overrides.frame is not None else prompt.frame
    frame_base = Path.cwd() if overrides.frame is not None else loaded.base_dir
    frame = None
    if frame_value is not None and frame_value.strip().lower() != "none":
        frame = _read_prompt_file(Path(frame_value), base_dir=frame_base, label="frame")

    discover_local = _prefer(
        overrides.local_frame_discovery,
        loaded.config.discovery.local_frame_discovery,
    )
    return build_system_prompt_with_addenda(
        orientation,
        cwd=cwd,
        addenda=(value for value in (role, frame) if value is not None),
        discover_claude_md=discover_local,
    )


def _build_orientation(
    value: str,
    *,
    model: str,
    cwd: Path,
    user_name: str,
    base_dir: Path,
) -> str:
    normalized = value.strip()
    if normalized == "auto":
        try:
            return build_core_orientation(model, cwd=cwd, user_name=user_name)
        except ValueError:
            return build_default_orientation(model, cwd=cwd, user_name=user_name)
    if normalized == "default":
        return build_default_orientation(model, cwd=cwd, user_name=user_name)
    orientation_path = _resolve_path(Path(normalized), base_dir=base_dir)
    try:
        return build_orientation_from_file(
            orientation_path,
            model=model,
            cwd=cwd,
            user_name=user_name,
        )
    except FileNotFoundError as exc:
        raise ServeConfigError(
            f"Orientation file does not exist: {orientation_path}"
        ) from exc
    except OSError as exc:
        raise ServeConfigError(
            f"Could not read orientation file {orientation_path}: {exc}"
        ) from exc


def _read_prompt_file(path: Path, *, base_dir: Path, label: str) -> str:
    resolved = _resolve_path(path, base_dir=base_dir)
    try:
        return resolved.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ServeConfigError(f"{label.title()} does not exist: {resolved}") from exc
    except OSError as exc:
        raise ServeConfigError(f"Could not read {label} {resolved}: {exc}") from exc


def _resolve_path(
    path: Path | None,
    *,
    base_dir: Path,
    default: Path | None = None,
) -> Path:
    if path is None:
        if default is None:
            raise ValueError("A path or default is required.")
        path = default
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = base_dir / expanded
    return expanded.resolve()


def _prefer[T](override: T | None, configured: T) -> T:
    return configured if override is None else override


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _override_optional_url(override: str | None, configured: str) -> str | None:
    return _clean_optional(configured if override is None else override)


def _validate_tool_dependencies(
    *,
    categories: list[str],
    body_url: str | None,
    minecraft_url: str | None,
    chorus_url: str | None,
) -> None:
    if "body" in categories and body_url is None:
        raise ServeConfigError("The 'body' tool category requires tools.body_url.")
    if "minecraft" in categories and minecraft_url is None:
        raise ServeConfigError(
            "The 'minecraft' tool category requires tools.minecraft_url."
        )
    if {"chorus", "chorus_tools"}.intersection(categories) and chorus_url is None:
        raise ServeConfigError(
            "The 'chorus' tool category requires tools.chorus_url or a Chorus URL "
            "environment variable."
        )


def _chorus_url(overrides: ServeOverrides, tools: ToolsSection) -> str | None:
    if overrides.chorus_url is not None:
        return _clean_optional(overrides.chorus_url)
    return (
        _clean_optional(tools.chorus_url)
        or _clean_optional(os.environ.get(MINICHORUS_SERVER_URL_ENV))
        or _clean_optional(os.environ.get(CHORUS_URL_ENV))
    )


def _chorus_entity_name(overrides: ServeOverrides, tools: ToolsSection) -> str | None:
    if overrides.chorus_entity_name is not None:
        return _clean_optional(overrides.chorus_entity_name)
    return _clean_optional(tools.chorus_entity_name) or _clean_optional(
        os.environ.get(CHORUS_SESSION_ENV)
    )

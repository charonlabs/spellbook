"""Static configuration for a Spellbook entity.

``SpellbookConfig`` is the immutable bundle of parameters that thread
through the core: which provider, which model, how much output headroom,
what cwd, what the system prompt says. It's frozen — callers construct
one at startup and pass it into the Generator, Executor, and surface
builder.

The ``system_prompt`` and ``tool_schemas`` fields are temporary
placeholders that will be replaced by the frame system once that lands
in the rewrite.
"""

from pathlib import Path
import re
from typing import Any, Literal, Mapping, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from .profiles import SessionProfile, SessionType, profile_for_session_type

Provider = Literal["anthropic", "openai", "local"]

DEFAULT_PROVIDER = "anthropic"
DEFAULT_MAX_OUTPUT_TOKENS = 128_000
DEFAULT_EFFORT = "high"
DEFAULT_MODEL_BY_PROVIDER = {
    "anthropic": "claude-opus-4-6",
    "openai": "gpt-5.4",
    "local": "gemma4",
}
DEFAULT_SOFT_THRESHOLD = 700_000
DEFAULT_MEDIUM_THRESHOLD = 850_000
DEFAULT_HARD_THRESHOLD = 933_000
DEFAULT_MAX_TOKENS = 1_000_000
DEFAULT_DETECT_INTERVAL = 500
DEFAULT_TOOL_RESULT_TTL_TURNS = 3
DEFAULT_TOOL_RESULT_TTL_CHAR_THRESHOLD = 4000

DEFAULT_SKILL_DISCOVERY_DIRS = [".claude", ".agents", ".spellbook", ".chorus/.claude"]
DEFAULT_OPENAI_SKILL_DISCOVERY_DIRS = [".agents", ".spellbook"]

DEFAULT_LOCAL_TIMEZONE = "America/New_York"
DEFAULT_IDLE_FOOTER_THRESHOLD_SECONDS = 300
DEFAULT_HEARTH_ENABLED = False
DEFAULT_HEARTH_INTERVAL_MINUTES = 55
DEFAULT_HEARTH_QUIET_HOURS = ""

DEFAULT_USER_NAME = "Ryan"

_QUIET_HOURS_RE = re.compile(
    r"^(?P<start_hour>\d{2}):(?P<start_minute>\d{2})-"
    r"(?P<end_hour>\d{2}):(?P<end_minute>\d{2})$"
)


def default_skill_discovery_dirs(provider: str) -> list[str]:
    if provider == "openai":
        return list(DEFAULT_OPENAI_SKILL_DISCOVERY_DIRS)
    return list(DEFAULT_SKILL_DISCOVERY_DIRS)


class HomunculusConfig(BaseModel, frozen=True):
    """Config object shared by Homunculus subsystems."""

    soft_threshold: int = DEFAULT_SOFT_THRESHOLD
    medium_threshold: int = DEFAULT_MEDIUM_THRESHOLD
    hard_threshold: int = DEFAULT_HARD_THRESHOLD
    detect_interval: int = Field(default=DEFAULT_DETECT_INTERVAL, gt=0)
    max_tokens: int = DEFAULT_MAX_TOKENS
    tool_result_ttl_enabled: bool = True
    tool_result_ttl_turns: int = Field(default=DEFAULT_TOOL_RESULT_TTL_TURNS, ge=0)
    tool_result_ttl_char_threshold: int = Field(
        default=DEFAULT_TOOL_RESULT_TTL_CHAR_THRESHOLD, ge=0
    )


class SpellbookConfig(BaseModel, frozen=True):
    """Main Config object that threads through systems for one Spellbook entity."""

    provider: Provider = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL_BY_PROVIDER[DEFAULT_PROVIDER]
    effort: str = DEFAULT_EFFORT
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    skill_discovery_dirs: list[str] = Field(
        default_factory=lambda: default_skill_discovery_dirs(DEFAULT_PROVIDER)
    )
    local_timezone: str = DEFAULT_LOCAL_TIMEZONE
    idle_footer_threshold_seconds: int = Field(
        default=DEFAULT_IDLE_FOOTER_THRESHOLD_SECONDS, ge=0
    )
    hearth_enabled: bool = DEFAULT_HEARTH_ENABLED
    hearth_interval_minutes: int = Field(default=DEFAULT_HEARTH_INTERVAL_MINUTES, ge=5)
    hearth_quiet_hours: str = DEFAULT_HEARTH_QUIET_HOURS
    user_name: str = DEFAULT_USER_NAME
    tool_categories: set[str] | None = None
    chorus_url: str | None = None
    chorus_entity_name: str | None = None
    session_type: SessionType = "main"
    profile: SessionProfile = Field(
        default_factory=lambda: profile_for_session_type("main")
    )
    cwd: Path

    # TEMPORARY FIELDS to be replaced with frames once they are implemented
    system_prompt: str = ""

    # Nested configs
    hom_config: HomunculusConfig = Field(default_factory=HomunculusConfig)

    @model_validator(mode="before")
    @classmethod
    def _apply_provider_specific_defaults(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        values = dict(data)
        if "skill_discovery_dirs" not in values:
            provider = str(values.get("provider", DEFAULT_PROVIDER))
            values["skill_discovery_dirs"] = default_skill_discovery_dirs(provider)
        if "profile" not in values:
            session_type = str(values.get("session_type", "main"))
            values["profile"] = profile_for_session_type(session_type)
        return values

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> Self:
        resolved_update = update
        if update is not None and "session_type" in update and "profile" not in update:
            resolved_update = dict(update)
            resolved_update["profile"] = profile_for_session_type(
                str(update["session_type"])
            )
        return super().model_copy(update=resolved_update, deep=deep)

    @field_validator("hearth_quiet_hours")
    @classmethod
    def _validate_hearth_quiet_hours(cls, value: str) -> str:
        text = value.strip()
        if not text:
            return ""
        match = _QUIET_HOURS_RE.match(text)
        if match is None:
            raise ValueError("hearth_quiet_hours must be empty or HH:MM-HH:MM.")
        start_hour = int(match.group("start_hour"))
        start_minute = int(match.group("start_minute"))
        end_hour = int(match.group("end_hour"))
        end_minute = int(match.group("end_minute"))
        if start_hour > 23 or end_hour > 23 or start_minute > 59 or end_minute > 59:
            raise ValueError("hearth_quiet_hours must be empty or HH:MM-HH:MM.")
        return f"{start_hour:02d}:{start_minute:02d}-{end_hour:02d}:{end_minute:02d}"

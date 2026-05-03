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
from typing import Literal

from pydantic import BaseModel, Field

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

DEFAULT_LOCAL_TIMEZONE = "America/New_York"
DEFAULT_IDLE_FOOTER_THRESHOLD_SECONDS = 300

DEFAULT_USER_NAME = "Ryan"

SessionType = Literal["main", "block_detector", "block_summarizer"]


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

    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL_BY_PROVIDER[DEFAULT_PROVIDER]
    effort: str = DEFAULT_EFFORT
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    skill_discovery_dirs: list[str] = DEFAULT_SKILL_DISCOVERY_DIRS
    local_timezone: str = DEFAULT_LOCAL_TIMEZONE
    idle_footer_threshold_seconds: int = Field(
        default=DEFAULT_IDLE_FOOTER_THRESHOLD_SECONDS, ge=0
    )
    user_name: str = DEFAULT_USER_NAME
    tool_categories: set[str] | None = None
    session_type: SessionType = "main"
    cwd: Path

    # TEMPORARY FIELDS to be replaced with frames once they are implemented
    system_prompt: str = ""

    # Nested configs
    hom_config: HomunculusConfig = Field(default_factory=HomunculusConfig)

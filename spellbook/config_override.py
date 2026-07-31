"""Append-only, resume-time overrides for :class:`SpellbookConfig`.

The app-server contract in this module is frozen for the Configurator UI:

``POST /config-override``

Request::

    {
      "updates": {"hearth_interval_minutes": 40},
      "source": "configurator",
      "actor": "Ryan",
      "note": "shorter check-in cadence"
    }

``note`` is optional. ``updates`` must contain known ``SpellbookConfig`` fields
and must not contain ``model``, ``provider``, or ``session_type``. A successful
write returns HTTP 200 with::

    {"applies_at": "next_resume"}

Validation failures return HTTP 400 with FastAPI's standard
``{"detail": "..."}`` body. Request-shape failures return HTTP 422. A write
only appends transcript truth; it never mutates the live session config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from spellbook.config import SpellbookConfig

FROZEN_IDENTITY_FIELDS = frozenset({"model", "provider", "session_type"})
CONFIG_OVERRIDE_DISCLOSURE_FOOTER_KEY = "config_override_disclosure"
CONFIG_OVERRIDE_DISCLOSURE_PRIORITY = -100


class ConfigOverrideValidationError(ValueError):
    """An override cannot safely be written or merged."""


class ConfigOverrideBody(BaseModel, frozen=True):
    """Frozen request body for ``POST /config-override``."""

    model_config = ConfigDict(extra="forbid")
    updates: dict[str, Any]
    source: str = Field(min_length=1)
    actor: str = Field(min_length=1)
    note: str | None = None

    @field_validator("source", "actor")
    @classmethod
    def _strip_required_identity(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be blank")
        return normalized


class ConfigOverrideResponse(BaseModel, frozen=True):
    """Frozen success response for ``POST /config-override``."""

    model_config = ConfigDict(extra="forbid")
    applies_at: Literal["next_resume"] = "next_resume"


def validate_override(updates: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one whole override or raise a clear error.

    This is the write-time guard and is intentionally also called at merge
    time. The returned mapping contains only the requested fields, normalized
    through ``SpellbookConfig`` validation.
    """

    if not updates:
        raise ConfigOverrideValidationError(
            "config override updates must contain at least one field"
        )

    attempted_fields = set(updates)
    frozen_fields = sorted(attempted_fields & FROZEN_IDENTITY_FIELDS)
    if frozen_fields:
        raise ConfigOverrideValidationError(
            "config override attempted to change frozen identity field(s): "
            + ", ".join(frozen_fields)
        )

    known_fields = set(SpellbookConfig.model_fields)
    unknown_fields = sorted(attempted_fields - known_fields)
    if unknown_fields:
        raise ConfigOverrideValidationError(
            "config override contains unknown SpellbookConfig field(s): "
            + ", ".join(unknown_fields)
        )

    try:
        validated = SpellbookConfig.model_validate({"cwd": Path("."), **dict(updates)})
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        )
        raise ConfigOverrideValidationError(
            f"config override contains invalid value(s): {details}"
        ) from exc

    return {field: getattr(validated, field) for field in updates}


def apply_override(
    config: SpellbookConfig, updates: Mapping[str, Any]
) -> tuple[SpellbookConfig, dict[str, Any]]:
    """Validate ``updates`` against ``config`` and return a copied config."""

    normalized = validate_override(updates)
    candidate_values = config.model_dump(mode="python")
    candidate_values.update(normalized)
    try:
        validated = SpellbookConfig.model_validate(candidate_values)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        )
        raise ConfigOverrideValidationError(
            f"config override contains invalid value(s): {details}"
        ) from exc

    effective_updates = {field: getattr(validated, field) for field in normalized}
    return config.model_copy(update=effective_updates), effective_updates

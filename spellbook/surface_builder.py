"""Request surface assembly for Spellbook.

A ``RequestSurface`` describes a full API request to the model provider —
model name, system prompt, tools, blocks, and provider-specific fields
like thinking config and cache control. Every token-counting operation
and every actual API call needs one.

This module provides a single class — ``RequestSurfaceBuilder`` — that
owns the "how do I build a surface?" decision. It replaces the scattered
logic that currently lives across three places:

- ``entity._build_request_surface()`` — live path for real entity calls
- ``homunculus._fallback_request_surface()`` — frozen metadata path for
  playgrounds analyzing a transcript without a running entity
- The ``_request_surface_builder`` callback dance — the Homunculus stores
  a function pointer that the entity installs at startup
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence

from .config import SpellbookConfig
from .ir_types import IRBlock

if TYPE_CHECKING:
    from spellbook.fork import ForkConfig

    from .backends.model_backend import ModelBackend, RequestSurface
    from .tools.registry import ToolRegistry


# Default values when a live backend isn't available (session-metadata path).
# These produce a minimal surface suitable for token counting only.
_DEFAULT_MAX_OUTPUT_TOKENS = 128_000
_DEFAULT_EFFORT = "high"


# TODO: type all of the subtypes here that are still looses
class RequestSurfaceBuilder:
    """Produces ``RequestSurface`` objects on demand.

    Stores the "config" half of a request surface (model, system prompt
    source, tool schemas, backend reference, output limits) and combines
    it with a fresh blocks list at call time.

    **NOW REQUIRES A BACKEND**
    """

    def __init__(
        self,
        *,
        model: str,
        system_provider: Callable[[], str | list[dict[str, Any]]],
        tool_schemas: list[dict[str, Any]],
        backend: ModelBackend,
        max_output_tokens: int = _DEFAULT_MAX_OUTPUT_TOKENS,
        effort: str = _DEFAULT_EFFORT,
    ):
        self._model = model
        self._system_provider = system_provider
        self._tool_schemas = list(tool_schemas)
        self._backend = backend
        self._max_output_tokens = max_output_tokens
        self._effort = effort

    def build(self, blocks: Sequence[IRBlock]) -> RequestSurface:
        """Build a ``RequestSurface`` from this builder's config + the given messages."""
        system = self._system_provider()
        return self._backend.build_request_surface(
            model=self._model,
            system=system,
            blocks=blocks,
            tools=self._tool_schemas,
            max_output_tokens=self._max_output_tokens,
            effort=self._effort,
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def has_backend(self) -> bool:
        return self._backend is not None

    # --- Factories ---

    @classmethod
    def from_config(
        cls,
        *,
        backend: ModelBackend,
        config: SpellbookConfig,
        tool_registry: ToolRegistry,
        fork_config: ForkConfig | None = None,
    ) -> "RequestSurfaceBuilder":
        """Build a live surface builder that delegates to a model backend.

        Used by the entity at startup. The returned builder produces
        full provider-specific surfaces (with thinking config, cache
        control, output config) ready for real API calls.

        ``system_provider`` is a callable because the system prompt is
        regenerated on every call (it reflects the current frame snapshot).
        """
        return cls(
            model=config.model,
            system_provider=lambda: config.system_prompt,
            tool_schemas=backend.build_tool_schemas(tool_registry),
            backend=backend,
            max_output_tokens=config.max_output_tokens,
            effort=config.effort,
        )

    # TODO: implement with NullBackend shape to appease new backend req
    # This is all old code atm
    @classmethod
    def from_session_metadata(
        cls,
        session_extra: dict[str, Any] | None,
    ) -> "RequestSurfaceBuilder | None":
        """Reconstruct a builder from frozen transcript session metadata.

        Used by playgrounds and analysis tools that load a transcript
        without a running entity. Returns ``None`` if the metadata
        doesn't contain enough to build a surface (no model field).
        """
        if not isinstance(session_extra, dict):
            return None

        # Session metadata shape: we look for either a top-level "model"
        # field or one nested under the frozen frame snapshot.
        model = session_extra.get("model")
        if not isinstance(model, str):
            return None

        frame = session_extra.get("frame")
        system: str | list[dict[str, Any]] = ""
        tools: list[dict[str, Any]] = []

        if isinstance(frame, dict):
            rendered = _render_system_from_frame(frame)
            if rendered:
                system = rendered  # noqa: F841
            tools = _extract_tools_from_frame(frame)  # noqa: F841

        return None
        # return cls(
        #     model=model,
        #     system_provider=lambda: system,
        #     tool_schemas=tools,
        #     backend=None,
        # )


def _render_system_from_frame(frame: dict[str, Any]) -> str:
    """Render the concatenated system prompt from a frozen frame snapshot.

    Equivalent to the current ``render_frame_prompt`` logic, inlined here
    to avoid cross-module coupling. Concatenates ``system_core`` content
    sources in render_order.
    """
    sources = frame.get("sources") or []
    if not isinstance(sources, list):
        return ""

    # Sort by render_order if present
    def order_key(src: Any) -> int:  # noqa: ANN401
        if not isinstance(src, dict):
            return 0
        val = src.get("render_order", 0)
        return int(val) if isinstance(val, (int, float)) else 0

    ordered = sorted(
        (s for s in sources if isinstance(s, dict)),
        key=order_key,
    )

    parts: list[str] = []
    for src in ordered:
        kind = src.get("kind")
        if kind == "tool_surface":
            continue  # tools are extracted separately
        content = src.get("content")
        if isinstance(content, str) and content.strip():
            parts.append(content)
    return "\n\n".join(parts).strip()


def _extract_tools_from_frame(frame: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull the tool_schemas list out of a frozen frame snapshot."""
    sources = frame.get("sources") or []
    if not isinstance(sources, list):
        return []

    for src in sources:
        if not isinstance(src, dict):
            continue
        if src.get("kind") != "tool_surface":
            continue
        content = src.get("content")
        if isinstance(content, list):
            return [t for t in content if isinstance(t, dict)]
    return []

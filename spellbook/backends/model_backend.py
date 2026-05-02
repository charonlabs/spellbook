"""The model backend abstraction.

``ModelBackend`` is the Protocol every provider (Anthropic, OpenAI, local)
implements. It's the single seam where provider-specific concerns live:
request surface shape, streaming, tool schema serialization. Above this
protocol, everything speaks IR.

A ``RequestSurface`` is the full assembled payload for one API call —
model, system prompt, tools, messages, thinking config, cache control.
The backend's ``build_request_surface`` translates IR blocks into the
provider-shaped messages field. The backend's ``stream`` opens a
``GenerationStream`` that yields normalized ``IRStreamEvent``s and
produces an ``IRGeneration`` on completion.

``GenerationStream`` is an async context manager + async iterator. It
preserves stream position across outer iteration, so callers can
iterate events externally or just await ``get_final_response()`` to
auto-exhaust the stream.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from anthropic.types import MessageParam

from spellbook.config import SpellbookConfig
from spellbook.surface_builder import RequestSurfaceBuilder

from ..cancel_token import CancelToken
from ..ir_types import IRBlock, IRGeneration, IRStreamEvent, StopReason
from ..tools.registry import ToolRegistry


# TODO: make real types for all the fields here, like tool schemas and output_config/cache_control
# this right now is ~Anthropic-shaped. Do we want to change it all?
@dataclass(frozen=True, slots=True)
class RequestSurface:
    model: str
    system: str | list[dict[str, Any]] | None = None
    tools: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] | list[MessageParam] = field(default_factory=list)
    thinking: dict[str, Any] | None = None
    output_config: dict[str, Any] | None = None
    cache_control: dict[str, Any] | None = None
    max_output_tokens: int = 128_000


class TokenCounter(Protocol):
    """Simple class to wrap a backend's token counter. Functions return `None` when the measurement fails."""

    async def count_block_content(self, block: IRBlock) -> int | None: ...

    async def count_blocks(self, blocks: list[IRBlock]) -> int | None: ...

    async def count_frame(  # TODO: make this take a real frame once that exists
        self,
    ) -> int | None: ...

    async def count_surface(self, surface: RequestSurface) -> int | None: ...


class GenerationStream(Protocol):
    """Async context manager + iterator of ``IRStreamEvent``s with a final response accessor."""

    async def __aenter__(self) -> "GenerationStream": ...

    async def __aexit__(self, *exc: Any) -> None: ...

    def __aiter__(self) -> AsyncIterator[IRStreamEvent]: ...

    async def __anext__(self) -> IRStreamEvent: ...

    async def get_final_response(self) -> IRGeneration:
        """Return the complete response after iteration is exhausted."""
        ...

    def get_current_response(self, *, stop_reason: StopReason) -> IRGeneration:
        """Return a snapshot of what the stream has already accumulated.
        Provide a stop reason for stamping the generation."""
        ...


class ModelBackend(Protocol):
    """Abstraction over a model provider's generation and counting APIs.

    The entity loop calls ``build_request_surface`` and ``stream``.
    Everything provider-specific lives behind this interface.
    """

    @property
    def provider(self) -> str:
        """Short provider identifier for session metadata (e.g. 'anthropic', 'openai')."""
        ...

    def build_request_surface(
        self,
        *,
        model: str,
        system: str | list[dict[str, Any]],
        blocks: Sequence[IRBlock],
        tools: list[dict[str, Any]],
        max_output_tokens: int,
        effort: str,
    ) -> RequestSurface:
        """Build a provider-specific request surface."""
        ...

    def stream(
        self,
        surface: RequestSurface,
        cancel_token: CancelToken,
    ) -> GenerationStream:
        """Start a streaming generation from the given request surface."""
        ...

    def build_tool_schemas(
        self,
        registry: ToolRegistry,
    ) -> list[dict[str, Any]]:
        """Generate provider-specific tool schemas from the registry."""
        ...

    def build_token_counter(
        self, config: SpellbookConfig, surface_builder: RequestSurfaceBuilder
    ) -> TokenCounter:
        """Create a token counter for this backend."""
        ...

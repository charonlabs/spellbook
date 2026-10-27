"""Bash through the Chorus Provider.

When ``SpellbookConfig.provider_url`` is set, the Bash tool stops spawning
host subprocesses and instead runs each command in a Provider sandbox through
the ``chorus_provider`` client. ``ProviderBash`` owns one client and at most
one sandbox per session: the sandbox is created lazily on the first command,
reused afterwards, transparently replaced when Provider reports it gone, and
destroyed when the session shuts down.

Safety review lives behind the Provider. A denied command comes back as a
normal exec result with ``denied=True``; it is surfaced to the entity as a
``ToolError`` carrying Provider's explanation.

The ``context`` sent with each exec is the most recent *human* message only
(``IRUserTextBlock.origin == "human"``) — never model output, conduit traffic,
or system footers — so the entity can never author its own review context.
``HumanContextRoundLifecycle`` keeps it current on the tool metadata.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from time import perf_counter
from types import ModuleType
from typing import Protocol

from ..ir_types import IRGeneration, IRToolTextBlock, IRUserTextBlock
from ..round_lifecycle import RoundContext, RoundLifecycle
from .common import ToolError, ToolExecutionResult, ToolMetadata

logger = logging.getLogger(__name__)

PROVIDER_MAX_TIMEOUT_MS = 300_000
PROVIDER_CONTEXT_MAX_BYTES = 16 * 1024
OUTPUT_CHAR_LIMIT = 256_000
SANDBOX_WORKDIR = "/workspace"

_SANDBOX_REPLACED_NOTE = (
    "Note: your previous sandbox was no longer available (sandboxes expire when "
    "idle and reset when a command is interrupted), so this command ran in a "
    f"fresh, empty {SANDBOX_WORKDIR}."
)
_SANDBOX_RESET_NOTE = (
    f"The sandbox was reset; {SANDBOX_WORKDIR} will be empty on your next command."
)


class _Sandbox(Protocol):
    @property
    def id(self) -> str: ...


class _ExecResult(Protocol):
    """Structural view of ``chorus_provider.ExecResult``."""

    @property
    def stdout(self) -> str: ...
    @property
    def stderr(self) -> str: ...
    @property
    def exit_code(self) -> int: ...
    @property
    def timed_out(self) -> bool: ...
    @property
    def output_truncated(self) -> bool: ...
    @property
    def sandbox_destroyed(self) -> bool: ...
    @property
    def denied(self) -> bool: ...
    @property
    def deny_reason(self) -> str | None: ...


class _ProviderClient(Protocol):
    """The subset of ``chorus_provider.ProviderClient`` this adapter uses."""

    async def connect(self) -> object: ...
    async def aclose(self) -> None: ...
    async def create(self) -> _Sandbox: ...
    async def exec(
        self,
        sandbox_id: str,
        command: str,
        *,
        timeout_ms: int = ...,
        context: str | None = ...,
    ) -> _ExecResult: ...
    async def destroy(self, sandbox_id: str) -> object: ...


ClientFactory = Callable[[], _ProviderClient]


def _sdk() -> ModuleType:
    try:
        import chorus_provider
    except ImportError:  # pragma: no cover - depends on install extras
        raise ToolError(
            "Bash is configured to run through the Provider (provider_url is set), "
            "but the chorus-provider client is not installed. Install the "
            "'provider' extra: uv sync --extra provider"
        ) from None
    return chorus_provider


def truncate_utf8(text: str, limit: int = PROVIDER_CONTEXT_MAX_BYTES) -> str:
    """Valid UTF-8 text of at most ``limit`` bytes, cut on a character boundary.

    Keeps the *start* of the message, where intent is usually stated. Lone
    surrogates (unencodable) are replaced so the client never rejects it."""
    encoded = text.encode("utf-8", errors="replace")
    return encoded[:limit].decode("utf-8", errors="ignore")


def latest_human_text(blocks: Sequence[object]) -> str | None:
    """Return the text of the newest non-empty human-authored user block."""
    for block in reversed(blocks):
        if (
            isinstance(block, IRUserTextBlock)
            and block.origin == "human"
            and block.text.strip()
        ):
            return block.text
    return None


class HumanContextRoundLifecycle(RoundLifecycle):
    """Track the latest human message for Provider safety-review context.

    Runs after generation — after any mid-turn injections — so the context a
    Bash call carries reflects everything the human has said up to that call.
    When the human message has been compacted out of the rendered context the
    previous value is kept: it is still the latest thing the human said."""

    def __init__(self, meta: ToolMetadata):
        self._meta = meta

    async def after_generate(self, ctx: RoundContext, generation: IRGeneration) -> None:
        text = latest_human_text(ctx.blocks)
        if text is not None:
            self._meta.last_human_message = text


class ProviderBash:
    """One Provider client and at most one sandbox for one Spellbook session."""

    def __init__(
        self,
        *,
        provider_url: str,
        auth_url: str,
        entity: str,
        admin_token_file: Path | None = None,
        client_factory: ClientFactory | None = None,
    ):
        self.provider_url = provider_url
        self.auth_url = auth_url
        self.entity = entity
        self.admin_token_file = admin_token_file
        self._client_factory = client_factory or self._default_client
        self._client: _ProviderClient | None = None
        self._sandbox_id: str | None = None
        self._lock = asyncio.Lock()
        self._closed = False

    @property
    def sandbox_id(self) -> str | None:
        return self._sandbox_id

    def _default_client(self) -> _ProviderClient:
        sdk = _sdk()
        # Bootstrap claims apply only if the entity is not yet registered; stored
        # claims always win. Spellbook is the trusted host-side orchestrator and
        # manages the sandbox lifecycle itself, so it needs create/destroy.
        claims = sdk.EntityClaims(
            role="builder",
            sandbox=None,
            scopes=("provider",),
            provider_perms=sdk.ProviderPermissions(
                exec=True, put=True, get=True, create=True, destroy=True
            ),
        )
        return sdk.ProviderClient(
            self.provider_url,
            self.auth_url,
            self.entity,
            claims=claims,
            admin_token_file=self.admin_token_file,
        )

    async def _ensure_sandbox(self) -> tuple[_ProviderClient, str]:
        async with self._lock:
            if self._closed:
                raise ToolError("The Provider sandbox session has been closed.")
            if self._client is None:
                # A failed connect closes that client; the next call builds anew.
                client = self._client_factory()
                await client.connect()
                self._client = client
            if self._sandbox_id is None:
                sandbox = await self._client.create()
                self._sandbox_id = sandbox.id
                logger.info("Provider sandbox created: %s", sandbox.id)
            return self._client, self._sandbox_id

    def _forget_sandbox(self, sandbox_id: str) -> None:
        if self._sandbox_id == sandbox_id:
            self._sandbox_id = None

    async def run(
        self, command: str, *, timeout_ms: int, context: str | None
    ) -> ToolExecutionResult:
        sdk = _sdk()
        notes: list[str] = []
        if timeout_ms > PROVIDER_MAX_TIMEOUT_MS:
            notes.append(
                f"Note: timeout capped at {PROVIDER_MAX_TIMEOUT_MS // 1000}s, the sandbox maximum."
            )
            timeout_ms = PROVIDER_MAX_TIMEOUT_MS
        timeout_ms = max(1, timeout_ms)
        review_context = truncate_utf8(context) if context is not None else None

        start = perf_counter()
        try:
            client, sandbox_id = await self._ensure_sandbox()
            try:
                result = await client.exec(
                    sandbox_id, command, timeout_ms=timeout_ms, context=review_context
                )
            except sdk.SandboxNotFoundError:
                # Provider answers 404 before running anything (the sandbox
                # idled out or was reset by an interrupted command), so one
                # retry in a fresh sandbox cannot run the command twice.
                self._forget_sandbox(sandbox_id)
                notes.append(_SANDBOX_REPLACED_NOTE)
                client, sandbox_id = await self._ensure_sandbox()
                result = await client.exec(
                    sandbox_id, command, timeout_ms=timeout_ms, context=review_context
                )
        except ToolError:
            raise
        except sdk.ProviderError as e:
            # SDK messages never include credentials or raw response bodies.
            raise ToolError(f"Provider error: {e}") from None
        duration_ms = int((perf_counter() - start) * 1000)

        if result.denied:
            raise ToolError(
                _join(
                    notes,
                    result.deny_reason
                    or result.stderr
                    or "Command denied by safety review.",
                )
            )
        if result.sandbox_destroyed:
            self._forget_sandbox(sandbox_id)

        output = _combine(result.stdout, result.stderr)
        if result.output_truncated:
            output = _join(
                [output] if output else [],
                "[output truncated: the sandbox captures at most 1 MiB per stream]",
            )
        timeout_s = timeout_ms / 1000
        if result.timed_out:
            output = f"Command timed out after {timeout_s:.0f}s\n{output}".rstrip()
            if result.sandbox_destroyed:
                output = f"{output}\n{_SANDBOX_RESET_NOTE}"
        elif result.exit_code != 0:
            output = (
                f"Exit code {result.exit_code}\n{output}"
                if output
                else f"Exit code {result.exit_code}"
            )
            if result.sandbox_destroyed:
                output = f"{output}\n{_SANDBOX_RESET_NOTE}"
        output = _join(notes, output)
        if len(output) > OUTPUT_CHAR_LIMIT:
            status = (
                f"Command timed out after {timeout_s:.0f}s"
                if result.timed_out
                else "Command executed"
            )
            raise ToolError(
                f"{status}, but output too large. {len(output)} chars. "
                f"The first 1000 chars of the output were: {output[:1000]}"
            )
        if result.timed_out:
            raise ToolError(output)

        return ToolExecutionResult(
            content=[IRToolTextBlock(text=output)],
            display={
                "kind": "command",
                "command": command,
                "description": "",
                "exit_code": result.exit_code,
                "duration_ms": duration_ms,
                "cwd": SANDBOX_WORKDIR,
                "stdout": output,
                "stderr": "",
                "combined_truncated": bool(result.output_truncated),
                "sandbox_id": sandbox_id,
            },
        )

    async def aclose(self) -> None:
        """Destroy the session's sandbox (best effort) and close the client."""
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            client, sandbox_id = self._client, self._sandbox_id
            self._client = self._sandbox_id = None
        if client is None:
            return
        try:
            if sandbox_id is not None:
                await client.destroy(sandbox_id)
                logger.info("Provider sandbox destroyed: %s", sandbox_id)
        except Exception as e:
            # Provider's idle TTL reaps anything we fail to destroy.
            logger.warning("Provider sandbox %s not destroyed: %s", sandbox_id, e)
        finally:
            try:
                await client.aclose()
            except Exception as e:
                logger.warning("Provider client close failed: %s", e)


def _combine(stdout: str, stderr: str) -> str:
    """Provider returns streams separately; the local tool merges them.

    Interleaving is not recoverable, so stdout comes first, then stderr."""
    stdout, stderr = stdout.rstrip(), stderr.rstrip()
    if stdout and stderr:
        return f"{stdout}\n{stderr}"
    return stdout or stderr


def _join(notes: list[str], text: str) -> str:
    return "\n\n".join([*notes, text]) if text else "\n\n".join(notes)

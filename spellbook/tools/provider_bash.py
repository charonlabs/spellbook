"""Bash and file tools through the Chorus Provider.

When ``SpellbookConfig.provider_url`` is set, the Bash, Read, Write and Edit
tools stop touching the host. Every operation becomes a shell command sent to
the session's Provider sandbox through the ``chorus_provider`` client — one
wire, one filesystem, one safety-review path:

- Bash      → the command itself
- Read      → ``base64`` of the file (exact bytes, images included)
- Write     → a quoted heredoc with a random delimiter (literal, reviewable)
- Edit      → Read, replace locally, Write

``ProviderBash`` owns one client and the session's sandbox. How it may treat
that sandbox depends on the entity's role (see ``SANDBOX_MANAGER_ROLES``):

- **familiar** (coordinator): creates its sandbox lazily, replaces it if the
  Provider reaps it, and destroys the ones it created at session shutdown.
- **everyone else**: works inside a sandbox the orchestrator created at summon
  time (``provider_sandbox_id``). It never creates or destroys; if that
  sandbox disappears, tools fail with an explanation instead of recovering.

Safety review lives behind the Provider. A denied operation comes back as an
exec result with ``denied=True`` and is surfaced as a ``ToolError`` carrying
Provider's explanation. The ``context`` sent with each exec is the most recent
*human* message only (``IRUserTextBlock.origin == "human"``) — never model
output, conduit traffic or system footers — so the entity cannot author its
own review context. ``HumanContextRoundLifecycle`` keeps it current.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import re
import shlex
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from time import perf_counter
from types import ModuleType
from typing import Protocol
from uuid import uuid4

from ..ir_types import IRGeneration, IRToolTextBlock, IRUserTextBlock
from ..round_lifecycle import RoundContext, RoundLifecycle
from .common import ToolError, ToolExecutionResult, ToolMetadata

logger = logging.getLogger(__name__)

# Roles that create and destroy their own sandboxes. Everyone else is assigned
# one by the orchestrator and holds only exec/put/get.
SANDBOX_MANAGER_ROLES = frozenset({"familiar"})
PROVIDER_ROLES = ("builder", "reader", "reviewer", "familiar", "meta", "keeper")
SANDBOX_ID_PATTERN = re.compile(r"^sb_[a-f0-9]{32}$")

PROVIDER_MAX_TIMEOUT_MS = 300_000
PROVIDER_CONTEXT_MAX_BYTES = 16 * 1024
# Provider rejects exec bodies above 64 KiB; keep headroom for framing.
EXEC_BODY_BUDGET = 64 * 1024 - 512
# Provider captures at most 1 MiB of stdout; base64 inflates by 4/3.
READ_MAX_BYTES = (1 << 20) * 3 // 4 - 64
WRITE_MAX_BYTES = 8 << 20
FILE_OP_TIMEOUT_MS = 60_000
OUTPUT_CHAR_LIMIT = 256_000
SANDBOX_WORKDIR = PurePosixPath("/workspace")

_SANDBOX_REPLACED_NOTE = (
    "Note: your previous sandbox was no longer available (sandboxes expire when "
    "idle and reset when a command is interrupted), so this ran in a fresh, "
    f"empty {SANDBOX_WORKDIR}."
)
_SANDBOX_RESET_NOTE = (
    f"The sandbox was reset; {SANDBOX_WORKDIR} will be empty on your next command."
)
_SANDBOX_LOST_NOTE = (
    "The Provider removed your sandbox, and your role does not create sandboxes, "
    "so further commands will fail until it is restored. Let your human partner know."
)
_SANDBOX_GONE_ERROR = (
    "Your sandbox is no longer available, and your role does not create "
    "sandboxes. Ask your human partner (or your coordinator) to restore it."
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
            "Tools are configured to run through the Provider (provider_url is set), "
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


def _json_size(value: object) -> int:
    """Exact encoded size, matching httpx's compact UTF-8 JSON encoding."""
    return len(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )


def _char_json_size(char: str) -> int:
    if char in '"\\\n\r\t\b\f':
        return 2
    if ord(char) < 0x20:
        return 6
    return len(char.encode("utf-8"))


def _check_text(text: str, what: str) -> None:
    if "\x00" in text:
        raise ToolError(
            f"{what} contains NUL characters, which the sandbox cannot accept."
        )
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        raise ToolError(
            f"{what} contains invalid Unicode (unpaired surrogates)."
        ) from None


class HumanContextRoundLifecycle(RoundLifecycle):
    """Track the latest human message for Provider safety-review context.

    Runs after generation — after any mid-turn injections — so the context a
    tool call carries reflects everything the human has said up to that call.
    When the human message has been compacted out of the rendered context the
    previous value is kept: it is still the latest thing the human said."""

    def __init__(self, meta: ToolMetadata):
        self._meta = meta

    async def after_generate(self, ctx: RoundContext, generation: IRGeneration) -> None:
        text = latest_human_text(ctx.blocks)
        if text is not None:
            self._meta.last_human_message = text


@dataclass
class ExecOutcome:
    """A completed (non-denied) exec plus notes the entity should see."""

    result: _ExecResult
    sandbox_id: str
    timeout_ms: int
    notes: list[str] = field(default_factory=list)


class ProviderBash:
    """One Provider client and the session's sandbox, for Bash and file tools."""

    def __init__(
        self,
        *,
        provider_url: str,
        auth_url: str,
        entity: str,
        role: str = "builder",
        sandbox_id: str | None = None,
        admin_token_file: Path | None = None,
        client_factory: ClientFactory | None = None,
    ):
        if role not in PROVIDER_ROLES:
            raise ValueError(
                f"Unknown provider role {role!r}; expected one of {PROVIDER_ROLES}."
            )
        if sandbox_id is not None and not SANDBOX_ID_PATTERN.match(sandbox_id):
            raise ValueError(f"Invalid provider sandbox id {sandbox_id!r}.")
        self.manages_sandboxes = role in SANDBOX_MANAGER_ROLES
        if not self.manages_sandboxes and sandbox_id is None:
            raise ValueError(
                f"Role {role!r} works in an orchestrator-assigned sandbox: set "
                "provider_sandbox_id (only familiars create their own)."
            )
        self.provider_url = provider_url
        self.auth_url = auth_url
        self.entity = entity
        self.role = role
        self.admin_token_file = admin_token_file
        self._client_factory = client_factory or self._default_client
        self._client: _ProviderClient | None = None
        self._sandbox_id: str | None = sandbox_id
        self._owned = False  # True only for sandboxes this session created
        self._lost = False  # assigned sandbox vanished (non-managers only)
        self._lock = asyncio.Lock()
        self._closed = False

    @property
    def sandbox_id(self) -> str | None:
        return self._sandbox_id

    def _default_client(self) -> _ProviderClient:
        sdk = _sdk()
        # Bootstrap claims apply only if the entity is not yet registered;
        # stored claims always win. Familiars manage their own sandboxes;
        # everyone else is scoped to the sandbox the orchestrator assigned.
        if self.manages_sandboxes:
            perms = sdk.ProviderPermissions(
                exec=True, put=True, get=True, create=True, destroy=True
            )
            scope = None
        else:
            perms = sdk.ProviderPermissions(exec=True, put=True, get=True)
            scope = self._sandbox_id
        claims = sdk.EntityClaims(
            role=self.role,
            sandbox=scope,
            scopes=("provider",),
            provider_perms=perms,
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
            if self._lost:
                raise ToolError(_SANDBOX_GONE_ERROR)
            if self._client is None:
                # A failed connect closes that client; the next call builds anew.
                client = self._client_factory()
                await client.connect()
                self._client = client
            if self._sandbox_id is None:
                if not self.manages_sandboxes:  # pragma: no cover - guarded above
                    raise ToolError(_SANDBOX_GONE_ERROR)
                sandbox = await self._client.create()
                self._sandbox_id, self._owned = sandbox.id, True
                logger.info("Provider sandbox created: %s", sandbox.id)
            return self._client, self._sandbox_id

    def _sandbox_gone(self, sandbox_id: str) -> None:
        if self._sandbox_id != sandbox_id:
            return
        if self.manages_sandboxes:
            self._sandbox_id, self._owned = None, False
        else:
            self._lost = True

    async def execute(
        self, command: str, *, timeout_ms: int, context: str | None
    ) -> ExecOutcome:
        """Run one command in the session sandbox.

        Raises ``ToolError`` for denials, Provider/transport failures, and an
        unavailable sandbox. Timeouts and nonzero exits are returned."""
        sdk = _sdk()
        _check_text(command, "The command")
        notes: list[str] = []
        if timeout_ms > PROVIDER_MAX_TIMEOUT_MS:
            notes.append(
                f"Note: timeout capped at {PROVIDER_MAX_TIMEOUT_MS // 1000}s, the sandbox maximum."
            )
            timeout_ms = PROVIDER_MAX_TIMEOUT_MS
        timeout_ms = max(1, timeout_ms)
        review_context = truncate_utf8(context) if context is not None else None
        size = _json_size(
            {
                "sandbox_id": "sb_" + "0" * 32,
                "command": command,
                "timeout_ms": timeout_ms,
                **({"context": review_context} if review_context is not None else {}),
            }
        )
        if size > EXEC_BODY_BUDGET:
            raise ToolError(
                f"The command is too long for the sandbox ({size} bytes encoded; the "
                f"limit is about {EXEC_BODY_BUDGET // 1024} KiB including review "
                "context). Write long content to a file in smaller pieces instead."
            )
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
                self._sandbox_gone(sandbox_id)
                if not self.manages_sandboxes:
                    raise ToolError(_join(notes, _SANDBOX_GONE_ERROR)) from None
                notes.append(_SANDBOX_REPLACED_NOTE)
                client, sandbox_id = await self._ensure_sandbox()
                result = await client.exec(
                    sandbox_id, command, timeout_ms=timeout_ms, context=review_context
                )
        except ToolError:
            raise
        except sdk.ProviderError as e:
            # SDK messages never include credentials or raw response bodies.
            raise ToolError(_join(notes, f"Provider error: {e}")) from None
        if result.denied:
            raise ToolError(
                _join(
                    notes,
                    result.deny_reason or result.stderr or "Denied by safety review.",
                )
            )
        if result.sandbox_destroyed:
            self._sandbox_gone(sandbox_id)
        return ExecOutcome(result, sandbox_id, timeout_ms, notes)

    def _reset_note(self) -> str:
        return _SANDBOX_RESET_NOTE if self.manages_sandboxes else _SANDBOX_LOST_NOTE

    # --- Bash -----------------------------------------------------------------

    async def run(
        self, command: str, *, timeout_ms: int, context: str | None
    ) -> ToolExecutionResult:
        start = perf_counter()
        outcome = await self.execute(command, timeout_ms=timeout_ms, context=context)
        duration_ms = int((perf_counter() - start) * 1000)
        result = outcome.result

        output = _combine(result.stdout, result.stderr)
        if result.output_truncated:
            output = _join(
                [output] if output else [],
                "[output truncated: the sandbox captures at most 1 MiB per stream]",
            )
        timeout_s = outcome.timeout_ms / 1000
        if result.timed_out:
            output = f"Command timed out after {timeout_s:.0f}s\n{output}".rstrip()
            if result.sandbox_destroyed:
                output = f"{output}\n{self._reset_note()}"
        elif result.exit_code != 0:
            output = (
                f"Exit code {result.exit_code}\n{output}"
                if output
                else f"Exit code {result.exit_code}"
            )
            if result.sandbox_destroyed:
                output = f"{output}\n{self._reset_note()}"
        output = _join(outcome.notes, output)
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
                "cwd": str(SANDBOX_WORKDIR),
                "stdout": output,
                "stderr": "",
                "combined_truncated": bool(result.output_truncated),
                "sandbox_id": outcome.sandbox_id,
            },
        )

    # --- Files ----------------------------------------------------------------

    async def _file_exec(
        self, command: str, *, context: str | None, path: PurePosixPath
    ) -> ExecOutcome:
        outcome = await self.execute(
            command, timeout_ms=FILE_OP_TIMEOUT_MS, context=context
        )
        if outcome.result.timed_out:
            raise ToolError(
                _join(
                    outcome.notes,
                    f"File operation on {path} timed out. {self._reset_note()}",
                )
            )
        return outcome

    async def read_file(
        self, path: PurePosixPath, *, context: str | None, missing_ok: bool = False
    ) -> tuple[bytes | None, list[str]]:
        """Exact bytes of ``path`` in the sandbox, or None when missing and
        ``missing_ok``. Directories and unreadable files raise ``ToolError``
        with the same wording as the local tools."""
        quoted = shlex.quote(str(path))
        command = (
            f'p={quoted}; if [ -d "$p" ]; then echo DIR; '
            f'elif [ ! -e "$p" ]; then echo MISSING; '
            f'else echo FILE && base64 -w0 -- "$p"; fi'
        )
        outcome = await self._file_exec(command, context=context, path=path)
        result, notes = outcome.result, outcome.notes
        status, _, payload = result.stdout.partition("\n")
        if status == "MISSING":
            if missing_ok:
                return None, notes
            raise ToolError(_join(notes, f"File not found: {path}"))
        if status == "DIR":
            raise ToolError(_join(notes, f"Path is a directory, not a file: {path}"))
        if status != "FILE" or result.exit_code != 0:
            detail = result.stderr.strip()
            if "Permission denied" in detail:
                raise ToolError(_join(notes, f"Permission denied: {path}"))
            raise ToolError(
                _join(notes, f"Could not read {path}: {detail or 'unknown error'}")
            )
        if result.output_truncated:
            raise ToolError(
                _join(
                    notes,
                    f"{path} is too large to read through the sandbox (limit about "
                    f"{READ_MAX_BYTES // 1024} KiB). Use Bash to read a slice, e.g. "
                    f"sed -n '1,200p' {quoted}",
                )
            )
        try:
            return base64.b64decode(payload.strip(), validate=True), notes
        except (binascii.Error, ValueError):
            raise ToolError(
                _join(notes, f"Could not read {path}: corrupt transfer")
            ) from None

    async def write_text(
        self, path: PurePosixPath, text: str, *, context: str | None
    ) -> list[str]:
        """Write ``text`` to ``path`` exactly (creating parent directories).

        Content travels as a quoted heredoc with a random delimiter, so it is
        never shell-expanded and stays human-readable for safety review. A
        heredoc always adds one trailing newline, removed with ``truncate``.
        Content too large for one exec is staged in a sibling temp file and
        then copied over the target in place (preserving its mode and links)."""
        _check_text(text, "The content")
        if len(text.encode("utf-8")) > WRITE_MAX_BYTES:
            raise ToolError(
                f"Content is too large to write through the sandbox "
                f"(limit {WRITE_MAX_BYTES >> 20} MiB)."
            )
        review_context = truncate_utf8(context) if context is not None else None
        target = shlex.quote(str(path))
        parent = shlex.quote(str(path.parent))
        delimiter = _delimiter(text)
        temp = shlex.quote(
            str(path.parent / f".{path.name}.spellbook-{uuid4().hex[:12]}.tmp")
        )

        def command(destination: str, redirect: str, chunk: str) -> str:
            return (
                f"mkdir -p -- {parent} && cat {redirect} {destination} <<'{delimiter}' "
                f"&& truncate -s -1 -- {destination}\n{chunk}\n{delimiter}"
            )

        overhead = _json_size(
            {
                "sandbox_id": "sb_" + "0" * 32,
                "command": command(temp, ">>", ""),
                "timeout_ms": FILE_OP_TIMEOUT_MS,
                **({"context": review_context} if review_context is not None else {}),
            }
        )
        chunks = _split_for_budget(text, EXEC_BODY_BUDGET - overhead)
        notes: list[str] = []
        if len(chunks) == 1:
            outcome = await self._file_exec(
                command(target, ">", chunks[0]), context=context, path=path
            )
            notes += outcome.notes
            _raise_write_failure(outcome, path, notes)
            return notes
        try:
            for index, chunk in enumerate(chunks):
                outcome = await self._file_exec(
                    command(temp, ">" if index == 0 else ">>", chunk),
                    context=context,
                    path=path,
                )
                notes += outcome.notes
                _raise_write_failure(outcome, path, notes)
            outcome = await self._file_exec(
                f"cat -- {temp} > {target} && rm -f -- {temp}",
                context=context,
                path=path,
            )
            notes += outcome.notes
            _raise_write_failure(outcome, path, notes)
        except ToolError:
            await self._remove_quietly(temp, context=context)
            raise
        return notes

    async def _remove_quietly(self, quoted_path: str, *, context: str | None) -> None:
        try:
            await self.execute(
                f"rm -f -- {quoted_path}",
                timeout_ms=FILE_OP_TIMEOUT_MS,
                context=context,
            )
        except ToolError:
            pass

    async def aclose(self) -> None:
        """Destroy a sandbox this session created (best effort); close the client.

        Orchestrator-assigned sandboxes are never destroyed here."""
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            client, sandbox_id, owned = self._client, self._sandbox_id, self._owned
            self._client = None
        if client is None:
            return
        try:
            if sandbox_id is not None and owned and self.manages_sandboxes:
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


def _raise_write_failure(
    outcome: ExecOutcome, path: PurePosixPath, notes: list[str]
) -> None:
    result = outcome.result
    if result.exit_code == 0:
        return
    detail = result.stderr.strip()
    if "Permission denied" in detail:
        raise ToolError(_join(notes, f"Permission denied: {path}"))
    raise ToolError(
        _join(notes, f"Could not write {path}: {detail or f'exit {result.exit_code}'}")
    )


def _delimiter(text: str) -> str:
    lines = set(text.split("\n"))
    while True:
        candidate = f"SPELLBOOK_EOF_{uuid4().hex[:16]}"
        if candidate not in lines:
            return candidate


def _split_for_budget(text: str, budget: int) -> list[str]:
    """Split text into pieces whose JSON-escaped size fits ``budget``."""
    if budget < 1024:
        raise ToolError(
            "The review context leaves no room to write file content through the sandbox."
        )
    chunks: list[str] = []
    start, size = 0, 0
    for index, char in enumerate(text):
        cost = _char_json_size(char)
        if size + cost > budget:
            chunks.append(text[start:index])
            start, size = index, 0
        size += cost
    chunks.append(text[start:])
    return chunks


def _combine(stdout: str, stderr: str) -> str:
    """Provider returns streams separately; the local tool merges them.

    Interleaving is not recoverable, so stdout comes first, then stderr."""
    stdout, stderr = stdout.rstrip(), stderr.rstrip()
    if stdout and stderr:
        return f"{stdout}\n{stderr}"
    return stdout or stderr


def _join(notes: list[str], text: str) -> str:
    return "\n\n".join([*notes, text]) if text else "\n\n".join(notes)

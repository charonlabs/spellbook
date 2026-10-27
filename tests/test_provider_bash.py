"""Bash through the Chorus Provider: routing, sandbox lifecycle, safety-review
denials, human-only review context, config plumbing and session wiring."""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any

import pytest

chorus_provider = pytest.importorskip("chorus_provider")

from chorus_provider import (  # noqa: E402
    ExecResult,
    ForbiddenError,
    RequestTimeoutError,
    Sandbox,
    SandboxNotFoundError,
)
from test_session_manager import (  # noqa: E402
    _DummyBackend,
    _gen,
    _round_lifecycle_names,
    _ScriptedBackend,
    _user_msg,
)

from spellbook.cancel_token import CancelToken  # noqa: E402
from spellbook.config import SpellbookConfig  # noqa: E402
from spellbook.ir_types import (  # noqa: E402
    IRAssistantTextBlock,
    IRGeneration,
    IRImageBlock,
    IRToolCallBlock,
    IRToolResultBlock,
    IRToolTextBlock,
    IRUsage,
    IRUserTextBlock,
)
from spellbook.round_lifecycle import RoundContext  # noqa: E402
from spellbook.serve import (  # noqa: E402
    PROVIDER_EXECUTION_NOTE,
    ServeConfigError,
    ServeOverrides,
    build_spellbook_config,
    load_entity_file,
)
from spellbook.session_manager import SessionManager  # noqa: E402
from spellbook.tools.common import (  # noqa: E402
    ToolError,
    ToolMetadata,
    build_provider_bash,
)
from spellbook.tools.filesystem import (  # noqa: E402
    BashInput,
    EditInput,
    ReadInput,
    WriteInput,
    exec_bash,
    exec_edit,
    exec_read,
    exec_write,
)
from spellbook.tools.provider_bash import (  # noqa: E402
    PROVIDER_CONTEXT_MAX_BYTES,
    HumanContextRoundLifecycle,
    ProviderBash,
    latest_human_text,
    truncate_utf8,
)

SB1 = "sb_" + "1" * 32
SB2 = "sb_" + "2" * 32


def ok(stdout: str = "hello\n", **kwargs: Any) -> ExecResult:
    return ExecResult(
        stdout, kwargs.pop("stderr", ""), kwargs.pop("exit_code", 0), **kwargs
    )


class FakeProvider:
    """Stands in for chorus_provider.ProviderClient.

    With ``root`` set, unscripted commands run in real bash with ``/workspace``
    mapped to ``root`` — so file-tool translations are checked byte for byte."""

    def __init__(self, results: list[Any] | None = None, root: Path | None = None):
        self.results = list(results or [])
        self.root = root
        self.connects = 0
        self.closed = False
        self.created: list[str] = []
        self.destroyed: list[str] = []
        self.execs: list[dict[str, Any]] = []
        self.create_error: Exception | None = None
        self.destroy_error: Exception | None = None
        self._ids = iter([SB1, SB2, "sb_" + "3" * 32])

    async def connect(self) -> FakeProvider:
        self.connects += 1
        return self

    async def aclose(self) -> None:
        self.closed = True

    async def create(self) -> Sandbox:
        if self.create_error is not None:
            error, self.create_error = self.create_error, None
            raise error
        sandbox_id = next(self._ids)
        self.created.append(sandbox_id)
        return Sandbox(sandbox_id, 1.0, 2.0, None)

    async def exec(
        self,
        sandbox_id: str,
        command: str,
        *,
        timeout_ms: int = 30000,
        context: str | None = None,
    ) -> ExecResult:
        self.execs.append(
            {
                "sandbox": sandbox_id,
                "command": command,
                "timeout_ms": timeout_ms,
                "context": context,
            }
        )
        if not self.results and self.root is not None:
            return await self._run(command)
        result = self.results.pop(0) if self.results else ok()
        if isinstance(result, Exception):
            raise result
        return result

    async def _run(self, command: str) -> ExecResult:
        assert self.root is not None
        proc = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            command.replace("/workspace", str(self.root)),
            cwd=self.root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        return ExecResult(
            out.decode(errors="replace"),
            err.decode(errors="replace").replace(str(self.root), "/workspace"),
            proc.returncode or 0,
        )

    async def destroy(self, sandbox_id: str) -> None:
        if self.destroy_error is not None:
            raise self.destroy_error
        self.destroyed.append(sandbox_id)


def provider_meta(
    tmp_path: Path,
    fake: FakeProvider,
    *,
    role: str = "familiar",
    sandbox_id: str | None = None,
) -> ToolMetadata:
    bash = ProviderBash(
        provider_url="http://127.0.0.1:8787",
        auth_url="http://127.0.0.1:8788",
        entity="wanderer",
        role=role,
        sandbox_id=sandbox_id,
        client_factory=lambda: fake,
    )
    return ToolMetadata(
        cwd=tmp_path, transcript_path=tmp_path / "t.jsonl", provider_bash=bash
    )


def text_of(result: Any) -> str:
    block = result.content[0]
    assert isinstance(block, IRToolTextBlock)
    return block.text


# ---- Routing and sandbox lifecycle -----------------------------------------


@pytest.mark.asyncio
async def test_bash_runs_in_provider_not_on_host(tmp_path: Path) -> None:
    fake = FakeProvider([ok("in sandbox\n"), ok("again\n")])
    meta = provider_meta(tmp_path, fake)
    marker = tmp_path / "host-marker"

    first = await exec_bash(meta, BashInput(command=f"touch {marker} && echo hi"))
    second = await exec_bash(meta, BashInput(command="pwd", timeout=1234))

    assert not marker.exists(), "provider mode must never spawn a host subprocess"
    assert text_of(first) == "in sandbox" and text_of(second) == "again"
    assert fake.connects == 1 and fake.created == [SB1], (
        "one lazily created sandbox, reused"
    )
    assert [e["sandbox"] for e in fake.execs] == [SB1, SB1]
    assert fake.execs[1]["timeout_ms"] == 1234 and fake.execs[0]["timeout_ms"] == 30000
    assert first.display["cwd"] == "/workspace" and first.display["sandbox_id"] == SB1
    assert first.display["kind"] == "command" and first.display["exit_code"] == 0


@pytest.mark.asyncio
async def test_local_bash_unchanged_without_provider(tmp_path: Path) -> None:
    meta = ToolMetadata(cwd=tmp_path, transcript_path=tmp_path / "t.jsonl")
    result = await exec_bash(meta, BashInput(command="printf local"))
    assert text_of(result) == "local"


@pytest.mark.asyncio
async def test_context_is_latest_human_message_truncated_to_16k(tmp_path: Path) -> None:
    fake = FakeProvider()
    meta = provider_meta(tmp_path, fake)
    await exec_bash(meta, BashInput(command="ls"))
    assert fake.execs[0]["context"] is None, "no human message yet: send no context"

    meta.last_human_message = "please clean up the old build"
    await exec_bash(meta, BashInput(command="ls"))
    assert fake.execs[1]["context"] == "please clean up the old build"

    meta.last_human_message = "é" * PROVIDER_CONTEXT_MAX_BYTES  # 2 bytes each
    await exec_bash(meta, BashInput(command="ls"))
    sent = fake.execs[2]["context"]
    assert len(sent.encode()) <= PROVIDER_CONTEXT_MAX_BYTES and set(sent) == {"é"}


@pytest.mark.asyncio
async def test_denied_command_is_a_tool_error_with_reason(tmp_path: Path) -> None:
    reason = (
        "Safety review: The model required review. If this is intentional, "
        "ask your human partner to confirm. (review id rev-1)"
    )
    fake = FakeProvider(
        [
            ok(
                "",
                stderr=reason,
                exit_code=77,
                denied=True,
                deny_reason=reason,
                review_id="rev-1",
            )
        ]
    )
    meta = provider_meta(tmp_path, fake)
    with pytest.raises(ToolError) as caught:
        await exec_bash(meta, BashInput(command="rm -rf /workspace/src"))
    assert caught.value.message == reason
    assert "Exit code" not in caught.value.message
    assert meta.provider_bash is not None and meta.provider_bash.sandbox_id == SB1


@pytest.mark.asyncio
async def test_vanished_sandbox_is_replaced_and_retried_once(tmp_path: Path) -> None:
    fake = FakeProvider([ok(), SandboxNotFoundError("gone"), ok("fresh\n")])
    meta = provider_meta(tmp_path, fake)
    await exec_bash(meta, BashInput(command="echo 1"))
    result = await exec_bash(meta, BashInput(command="echo 2"))
    assert fake.created == [SB1, SB2]
    assert [e["sandbox"] for e in fake.execs] == [SB1, SB1, SB2]
    out = text_of(result)
    assert out.startswith("Note: your previous sandbox was no longer available")
    assert out.endswith("fresh")


@pytest.mark.asyncio
async def test_timeout_reports_reset_and_next_call_gets_new_sandbox(
    tmp_path: Path,
) -> None:
    fake = FakeProvider(
        [
            ok("partial\n", exit_code=124, timed_out=True, sandbox_destroyed=True),
            ok("next\n"),
        ]
    )
    meta = provider_meta(tmp_path, fake)
    with pytest.raises(ToolError) as caught:
        await exec_bash(meta, BashInput(command="sleep 99", timeout=2000))
    message = caught.value.message
    assert message.startswith("Command timed out after 2s\npartial")
    assert "sandbox was reset" in message
    await exec_bash(meta, BashInput(command="ls"))
    assert fake.created == [SB1, SB2]


@pytest.mark.asyncio
async def test_failure_output_formatting(tmp_path: Path) -> None:
    fake = FakeProvider(
        [
            ok("out\n", stderr="err\n", exit_code=2),
            ok("", exit_code=1),
            ok("x", output_truncated=True),
        ]
    )
    meta = provider_meta(tmp_path, fake)
    assert (
        text_of(await exec_bash(meta, BashInput(command="a")))
        == "Exit code 2\nout\nerr"
    )
    assert text_of(await exec_bash(meta, BashInput(command="b"))) == "Exit code 1"
    assert "[output truncated" in text_of(await exec_bash(meta, BashInput(command="c")))


@pytest.mark.asyncio
async def test_timeout_above_sandbox_maximum_is_capped_with_note(
    tmp_path: Path,
) -> None:
    fake = FakeProvider()
    meta = provider_meta(tmp_path, fake)
    result = await exec_bash(meta, BashInput(command="make", timeout=600_000))
    assert fake.execs[0]["timeout_ms"] == 300_000
    assert text_of(result).startswith("Note: timeout capped at 300s")


@pytest.mark.asyncio
async def test_oversized_output_is_a_tool_error(tmp_path: Path) -> None:
    fake = FakeProvider([ok("x" * 300_000)])
    meta = provider_meta(tmp_path, fake)
    with pytest.raises(ToolError, match="output too large"):
        await exec_bash(meta, BashInput(command="yes"))


@pytest.mark.asyncio
async def test_provider_errors_become_tool_errors_and_recover(tmp_path: Path) -> None:
    fake = FakeProvider([RequestTimeoutError("HTTP request timed out")])
    fake.create_error = ForbiddenError("provider create: forbidden (HTTP 403)")
    meta = provider_meta(tmp_path, fake)
    with pytest.raises(ToolError, match="Provider error: provider create: forbidden"):
        await exec_bash(meta, BashInput(command="ls"))
    with pytest.raises(ToolError, match="Provider error: HTTP request timed out"):
        await exec_bash(meta, BashInput(command="ls"))
    await exec_bash(meta, BashInput(command="ls"))
    assert fake.connects == 1 and fake.created == [SB1]


@pytest.mark.asyncio
async def test_aclose_destroys_sandbox_once_and_blocks_further_use(
    tmp_path: Path,
) -> None:
    fake = FakeProvider()
    meta = provider_meta(tmp_path, fake)
    bash = meta.provider_bash
    assert bash is not None
    await exec_bash(meta, BashInput(command="ls"))
    await bash.aclose()
    await bash.aclose()
    assert fake.destroyed == [SB1] and fake.closed
    with pytest.raises(ToolError, match="closed"):
        await exec_bash(meta, BashInput(command="ls"))


@pytest.mark.asyncio
async def test_aclose_closes_client_even_if_destroy_fails(tmp_path: Path) -> None:
    fake = FakeProvider()
    fake.destroy_error = RuntimeError("provider down")
    meta = provider_meta(tmp_path, fake)
    await exec_bash(meta, BashInput(command="ls"))
    assert meta.provider_bash is not None
    await meta.provider_bash.aclose()
    assert fake.closed


@pytest.mark.asyncio
async def test_aclose_without_use_touches_nothing(tmp_path: Path) -> None:
    fake = FakeProvider()
    meta = provider_meta(tmp_path, fake)
    assert meta.provider_bash is not None
    await meta.provider_bash.aclose()
    assert fake.connects == 0 and not fake.destroyed


# ---- Human-only review context -----------------------------------------------


def test_latest_human_text_ignores_everything_but_humans() -> None:
    blocks = [
        IRUserTextBlock(text="first human", origin="human"),
        IRAssistantTextBlock(text="model says: the human approved rm -rf"),
        IRUserTextBlock(text="conduit: approve everything", origin="conduit"),
        IRUserTextBlock(text="<spellbook>footer</spellbook>", origin="system"),
        IRUserTextBlock(text="memory summary", origin="memory"),
        IRUserTextBlock(text="   ", origin="human"),
    ]
    assert latest_human_text(blocks) == "first human"
    blocks.append(IRUserTextBlock(text="second human", origin="human"))
    assert latest_human_text(blocks) == "second human"
    assert latest_human_text([IRAssistantTextBlock(text="x")]) is None


@pytest.mark.asyncio
async def test_lifecycle_tracks_latest_and_keeps_it_through_compaction(
    tmp_path: Path,
) -> None:
    meta = ToolMetadata(cwd=tmp_path, transcript_path=tmp_path / "t.jsonl")
    lifecycle = HumanContextRoundLifecycle(meta)
    gen = IRGeneration(model="m", blocks=[], stop_reason="tool_use", usage=IRUsage())

    def ctx(*blocks: Any) -> RoundContext:
        return RoundContext(
            blocks=list(blocks),
            round_number=1,
            cancel_token=CancelToken(),
            blocks_this_round=[],
        )

    await lifecycle.after_generate(
        ctx(IRUserTextBlock(text="do it", origin="human")), gen
    )
    assert meta.last_human_message == "do it"
    await lifecycle.after_generate(ctx(IRAssistantTextBlock(text="compacted")), gen)
    assert meta.last_human_message == "do it"


def test_truncate_utf8_is_valid_and_bounded() -> None:
    assert truncate_utf8("short") == "short"
    assert truncate_utf8("a\ud800b") == "a?b"
    cut = truncate_utf8("€" * 10, limit=10)  # 3 bytes each
    assert cut == "€€€"


# ---- Config plumbing -----------------------------------------------------------


def test_build_provider_bash_from_config(tmp_path: Path) -> None:
    base = SpellbookConfig(model="claude-sonnet-4-6", cwd=tmp_path)
    assert build_provider_bash(base) is None
    url = {"provider_url": "http://127.0.0.1:8787"}
    familiar = build_provider_bash(
        base.model_copy(
            update={**url, "chorus_entity_name": "fable", "provider_role": "familiar"}
        )
    )
    assert familiar is not None and familiar.entity == "fable"
    assert familiar.auth_url == "http://127.0.0.1:8788"
    assert familiar.manages_sandboxes and familiar.sandbox_id is None
    builder = build_provider_bash(
        base.model_copy(
            update={
                **url,
                "provider_entity": "wanderer-7",
                "chorus_entity_name": "fable",
                "provider_auth_url": "http://127.0.0.1:9999",
                "provider_sandbox_id": SB1,
            }
        )
    )
    assert builder is not None and builder.entity == "wanderer-7"
    assert builder.auth_url == "http://127.0.0.1:9999"
    assert builder.role == "builder" and not builder.manages_sandboxes
    assert builder.sandbox_id == SB1
    with pytest.raises(ValueError, match="provider_entity"):
        build_provider_bash(base.model_copy(update=url))
    with pytest.raises(ValueError, match="provider_sandbox_id"):
        build_provider_bash(base.model_copy(update={**url, "provider_entity": "w"}))


def _entity_toml(tmp_path: Path, tools: str) -> Path:
    path = tmp_path / "entity.toml"
    path.write_text(
        f'[entity]\nmodel = "claude-sonnet-4-6"\ncwd = "."\n\n[tools]\n{tools}\n',
        encoding="utf-8",
    )
    return path


def test_serve_config_routes_bash_and_discloses_review(tmp_path: Path) -> None:
    (tmp_path / "admin.token").write_text("x")
    path = _entity_toml(
        tmp_path,
        'provider_url = "http://127.0.0.1:8787"\nprovider_entity = "wanderer"\n'
        'provider_admin_token_file = "admin.token"\n'
        f'provider_sandbox_id = "{SB1}"',
    )
    config = build_spellbook_config(load_entity_file(path))
    assert config.provider_url == "http://127.0.0.1:8787"
    assert config.provider_entity == "wanderer"
    assert config.provider_admin_token_file == (tmp_path / "admin.token").resolve()
    assert config.provider_auth_url is None
    assert config.provider_role == "builder" and config.provider_sandbox_id == SB1
    assert PROVIDER_EXECUTION_NOTE in config.system_prompt
    assert "safety net, not a cage" in config.system_prompt

    plain = build_spellbook_config(load_entity_file(_entity_toml(tmp_path, "")))
    assert plain.provider_url is None
    assert PROVIDER_EXECUTION_NOTE not in plain.system_prompt

    overridden = build_spellbook_config(
        load_entity_file(_entity_toml(tmp_path, "")),
        ServeOverrides(
            provider_url="http://127.0.0.1:8787",
            provider_entity="cli-entity",
            provider_role="familiar",
        ),
    )
    assert overridden.provider_url and overridden.provider_entity == "cli-entity"
    assert overridden.provider_role == "familiar"
    assert overridden.provider_sandbox_id is None


@pytest.mark.parametrize(
    ("tools", "message"),
    [
        ('provider_role = "builder"', "provider_sandbox_id"),
        ('provider_role = "reader"', "Only familiars create"),
        (
            'provider_role = "root"\nprovider_sandbox_id = "' + SB1 + '"',
            "provider_role",
        ),
        ('provider_role = "builder"\nprovider_sandbox_id = "sb_nope"', "sb_<32 hex>"),
    ],
)
def test_serve_config_enforces_permission_tiers(
    tmp_path: Path, tools: str, message: str
) -> None:
    path = _entity_toml(
        tmp_path,
        f'provider_url = "http://127.0.0.1:8787"\nprovider_entity = "w"\n{tools}',
    )
    with pytest.raises(ServeConfigError, match=message):
        build_spellbook_config(load_entity_file(path))


def test_serve_config_requires_an_identity_for_provider(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("CHORUS_SESSION", raising=False)  # serve's entity-name fallback
    path = _entity_toml(tmp_path, 'provider_url = "http://127.0.0.1:8787"')
    with pytest.raises(ServeConfigError, match="provider_entity"):
        build_spellbook_config(load_entity_file(path))


# ---- Session wiring -------------------------------------------------------------


def _provider_config(tmp_path: Path) -> SpellbookConfig:
    return SpellbookConfig(
        model="claude-sonnet-4-6",
        cwd=tmp_path,
        provider_url="http://127.0.0.1:8787",
        provider_entity="wanderer",
        provider_role="familiar",
    )


@pytest.mark.asyncio
async def test_session_build_wires_context_lifecycle_only_in_provider_mode(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "spellbook.session_manager.build_backend", lambda config: _DummyBackend()
    )
    local = await SessionManager.build(
        transcript_path=tmp_path / "local.jsonl",
        config=SpellbookConfig(model="claude-sonnet-4-6", cwd=tmp_path),
    )
    assert "HumanContextRoundLifecycle" not in _round_lifecycle_names(local)
    assert local.executor.meta.provider_bash is None

    remote = await SessionManager.build(
        transcript_path=tmp_path / "remote.jsonl", config=_provider_config(tmp_path)
    )
    assert "HumanContextRoundLifecycle" in _round_lifecycle_names(remote)
    assert isinstance(remote.executor.meta.provider_bash, ProviderBash)


@pytest.mark.asyncio
async def test_full_turn_sends_human_context_and_shutdown_destroys_sandbox(
    tmp_path: Path, monkeypatch
) -> None:
    fake = FakeProvider([ok("cleaned\n")])
    monkeypatch.setattr(ProviderBash, "_default_client", lambda self: fake)
    call = IRToolCallBlock(
        origin="model", call_id="call_1", tool="Bash", input={"command": "rm -rf build"}
    )
    backend = _ScriptedBackend(
        [
            _gen([IRAssistantTextBlock(text="On it."), call], stop_reason="tool_use"),
            _gen([IRAssistantTextBlock(text="Done.")], stop_reason="end_turn"),
        ]
    )
    monkeypatch.setattr(
        "spellbook.session_manager.build_backend", lambda config: backend
    )
    manager = await SessionManager.build(
        transcript_path=tmp_path / "turn.jsonl", config=_provider_config(tmp_path)
    )

    runner = asyncio.create_task(manager.run())
    await manager.submit_message(
        _user_msg("yes, please delete the old build directory")
    )
    for _ in range(200):
        if len(backend.surfaces) >= 2:
            break
        await asyncio.sleep(0.01)
    await manager.shutdown()
    await asyncio.wait_for(runner, timeout=5)

    assert fake.execs == [
        {
            "sandbox": SB1,
            "command": "rm -rf build",
            "timeout_ms": 30000,
            "context": "yes, please delete the old build directory",
        }
    ]
    results = [
        block
        for block in backend.blocks_seen[1]
        if isinstance(block, IRToolResultBlock) and block.call_id == "call_1"
    ]
    assert results and not results[0].is_error
    assert fake.destroyed == [SB1] and fake.closed


# ---- Permission tiers ---------------------------------------------------------


def test_assigned_role_requires_sandbox_and_valid_identity() -> None:
    common = {"provider_url": "http://127.0.0.1:8787", "auth_url": "x", "entity": "w"}
    with pytest.raises(ValueError, match="provider_sandbox_id"):
        ProviderBash(**common, role="builder")
    with pytest.raises(ValueError, match="Unknown provider role"):
        ProviderBash(**common, role="root", sandbox_id=SB1)
    with pytest.raises(ValueError, match="Invalid provider sandbox id"):
        ProviderBash(**common, role="familiar", sandbox_id="sb_bad")


def test_bootstrap_claims_follow_the_tier(tmp_path: Path) -> None:
    token = tmp_path / "admin.token"
    token.write_text("secret")
    common = {
        "provider_url": "http://127.0.0.1:8787",
        "auth_url": "http://127.0.0.1:8788",
        "entity": "w",
        "admin_token_file": token,
    }
    builder = ProviderBash(**common, role="builder", sandbox_id=SB1)._default_client()
    claims = builder._claims
    assert claims.role == "builder" and claims.sandbox == SB1
    perms = claims.provider_perms
    assert (perms.exec, perms.put, perms.get) == (True, True, True)
    assert not perms.create and not perms.destroy
    familiar = ProviderBash(**common, role="familiar")._default_client()._claims
    assert familiar.role == "familiar" and familiar.sandbox is None
    assert familiar.provider_perms.create and familiar.provider_perms.destroy


@pytest.mark.asyncio
async def test_builder_uses_assigned_sandbox_and_never_creates_or_destroys(
    tmp_path: Path,
) -> None:
    fake = FakeProvider([ok("one\n"), ok("two\n")])
    meta = provider_meta(tmp_path, fake, role="builder", sandbox_id=SB2)
    await exec_bash(meta, BashInput(command="ls"))
    await exec_bash(meta, BashInput(command="ls"))
    assert [e["sandbox"] for e in fake.execs] == [SB2, SB2]
    assert meta.provider_bash is not None
    await meta.provider_bash.aclose()
    assert fake.created == [] and fake.destroyed == [] and fake.closed


@pytest.mark.asyncio
async def test_builder_cannot_recover_a_vanished_sandbox(tmp_path: Path) -> None:
    fake = FakeProvider([SandboxNotFoundError("gone")])
    meta = provider_meta(tmp_path, fake, role="builder", sandbox_id=SB2)
    with pytest.raises(ToolError, match="your role does not create sandboxes"):
        await exec_bash(meta, BashInput(command="ls"))
    with pytest.raises(ToolError, match="no longer available"):
        await exec_bash(meta, BashInput(command="ls"))
    assert len(fake.execs) == 1, "later calls fail fast without touching Provider"
    assert fake.created == []


@pytest.mark.asyncio
async def test_builder_timeout_reset_is_reported_as_lost(tmp_path: Path) -> None:
    fake = FakeProvider([ok("", exit_code=124, timed_out=True, sandbox_destroyed=True)])
    meta = provider_meta(tmp_path, fake, role="builder", sandbox_id=SB2)
    with pytest.raises(ToolError) as caught:
        await exec_bash(meta, BashInput(command="sleep 99", timeout=1000))
    assert "Let your human partner know" in caught.value.message
    with pytest.raises(ToolError, match="no longer available"):
        await exec_bash(meta, BashInput(command="ls"))
    assert fake.created == []


@pytest.mark.asyncio
async def test_familiar_keeps_assigned_sandbox_but_destroys_only_its_own(
    tmp_path: Path,
) -> None:
    fake = FakeProvider([ok(), SandboxNotFoundError("gone"), ok()])
    meta = provider_meta(tmp_path, fake, role="familiar", sandbox_id=SB2)
    bash = meta.provider_bash
    assert bash is not None
    await exec_bash(meta, BashInput(command="ls"))
    assert fake.created == [] and bash.sandbox_id == SB2
    await exec_bash(meta, BashInput(command="ls"))  # assigned one vanished
    assert fake.created == [SB1] and bash.sandbox_id == SB1
    await bash.aclose()
    assert fake.destroyed == [SB1]


@pytest.mark.asyncio
async def test_familiar_does_not_destroy_an_assigned_sandbox(tmp_path: Path) -> None:
    fake = FakeProvider()
    meta = provider_meta(tmp_path, fake, role="familiar", sandbox_id=SB2)
    await exec_bash(meta, BashInput(command="ls"))
    assert meta.provider_bash is not None
    await meta.provider_bash.aclose()
    assert fake.destroyed == []


# ---- File tools through the sandbox ----------------------------------------------


def sandbox_meta(tmp_path: Path) -> tuple[ToolMetadata, FakeProvider, Path]:
    root = tmp_path / "sandbox-workspace"
    root.mkdir()
    host = tmp_path / "host"
    host.mkdir()
    fake = FakeProvider(root=root)
    meta = provider_meta(host, fake)
    meta.last_human_message = "please update the notes"
    return meta, fake, root


@pytest.mark.parametrize(
    "content",
    [
        "",
        "no trailing newline",
        "one line\n",
        "blank lines\n\n\n",
        "\n",
        "shell bait: $HOME `id` $(whoami) \\ 'single' \"double\" * ? [x] ~ !\n",
        "EOF\nSPELLBOOK_EOF\n'EOF'\nend",
        "unicode: héllo — 世界 🜂\n",
        "tabs\tand\x01control\x1b[0m chars",
    ],
)
@pytest.mark.asyncio
async def test_write_then_read_round_trips_exact_bytes(
    tmp_path: Path, content: str
) -> None:
    meta, fake, root = sandbox_meta(tmp_path)
    await exec_write(meta, WriteInput(file_path="notes/n.txt", content=content))
    assert (root / "notes" / "n.txt").read_bytes() == content.encode("utf-8")
    assert not (tmp_path / "host" / "notes").exists(), "host filesystem untouched"
    result = await exec_read(meta, ReadInput(file_path="/workspace/notes/n.txt"))
    lines = content.splitlines()
    text = text_of(result)
    assert text.startswith(f"Lines {1 if lines else 0}-{len(lines)} of {len(lines)}")
    for line in lines:
        assert line in text
    assert all(e["context"] == "please update the notes" for e in fake.execs)
    assert all(e["sandbox"] == SB1 for e in fake.execs)


@pytest.mark.asyncio
async def test_write_reports_create_then_overwrite_with_diff(tmp_path: Path) -> None:
    meta, _, root = sandbox_meta(tmp_path)
    created = await exec_write(meta, WriteInput(file_path="a.py", content="x = 1\n"))
    assert text_of(created) == "Successfully wrote to /workspace/a.py"
    assert created.display["change_type"] == "create"
    overwritten = await exec_write(
        meta, WriteInput(file_path="a.py", content="x = 2\n")
    )
    assert overwritten.display["change_type"] == "overwrite"
    assert (
        "-x = 1" in overwritten.display["diff"]
        and "+x = 2" in overwritten.display["diff"]
    )
    assert (root / "a.py").read_text() == "x = 2\n"


@pytest.mark.asyncio
async def test_edit_through_sandbox(tmp_path: Path) -> None:
    meta, _, root = sandbox_meta(tmp_path)
    (root / "app.py").write_text("def f():\n    return 1\n\nprint(f())\n")
    result = await exec_edit(
        meta,
        EditInput(file_path="app.py", old_string="return 1", new_string="return 42"),
    )
    assert text_of(result) == "Successfully edited /workspace/app.py"
    assert (root / "app.py").read_text() == "def f():\n    return 42\n\nprint(f())\n"
    with pytest.raises(ToolError, match="old_string not found in /workspace/app.py"):
        await exec_edit(
            meta, EditInput(file_path="app.py", old_string="nope", new_string="x")
        )


@pytest.mark.asyncio
async def test_file_errors_match_local_wording(tmp_path: Path) -> None:
    meta, _, root = sandbox_meta(tmp_path)
    (root / "dir").mkdir()
    (root / "latin1.txt").write_bytes(b"caf\xe9")
    with pytest.raises(ToolError, match="^File not found: /workspace/missing.txt$"):
        await exec_read(meta, ReadInput(file_path="missing.txt"))
    with pytest.raises(
        ToolError, match="^Path is a directory, not a file: /workspace/dir$"
    ):
        await exec_read(meta, ReadInput(file_path="dir"))
    with pytest.raises(
        ToolError, match="^Path is a directory, not a file: /workspace/dir$"
    ):
        await exec_write(meta, WriteInput(file_path="dir", content="x"))
    with pytest.raises(ToolError, match="File not found"):
        await exec_edit(
            meta, EditInput(file_path="missing", old_string="a", new_string="b")
        )
    with pytest.raises(ToolError, match="not valid UTF-8"):
        await exec_read(meta, ReadInput(file_path="latin1.txt"))
    with pytest.raises(ToolError, match="NUL"):
        await exec_write(meta, WriteInput(file_path="z.txt", content="a\x00b"))


@pytest.mark.asyncio
async def test_crlf_files_read_like_local(tmp_path: Path) -> None:
    meta, _, root = sandbox_meta(tmp_path)
    (root / "win.txt").write_bytes(b"one\r\ntwo\r\n")
    remote = text_of(await exec_read(meta, ReadInput(file_path="win.txt")))
    local_meta = ToolMetadata(cwd=root, transcript_path=tmp_path / "t.jsonl")
    local = text_of(await exec_read(local_meta, ReadInput(file_path="win.txt")))
    assert remote.replace("/workspace", "") == local.replace(str(root), "")


@pytest.mark.asyncio
async def test_large_write_is_chunked_staged_and_preserves_mode(tmp_path: Path) -> None:
    meta, fake, root = sandbox_meta(tmp_path)
    script = root / "big.sh"
    script.write_text("#!/bin/sh\n")
    script.chmod(0o755)
    content = "".join(f'echo line {i} — ünïcode "quoted" $x\n' for i in range(9000))
    assert len(content.encode()) > 200_000
    await exec_write(meta, WriteInput(file_path="big.sh", content=content))
    assert script.read_bytes() == content.encode("utf-8")
    assert script.stat().st_mode & 0o777 == 0o755, "in-place copy keeps the mode"
    assert not list(root.glob(".big.sh.spellbook-*")), "temp file removed"
    writes = [e for e in fake.execs if "<<'SPELLBOOK_EOF_" in e["command"]]
    assert len(writes) > 3
    assert all(
        len(json.dumps(e, ensure_ascii=False).encode()) < 64 * 1024 for e in fake.execs
    )


@pytest.mark.asyncio
async def test_failed_chunk_cleans_up_and_leaves_target_intact(tmp_path: Path) -> None:
    meta, fake, root = sandbox_meta(tmp_path)
    (root / "keep.txt").write_text("original\n")
    reason = "Safety review: flagged. (review id r)"
    content = "x" * 150_000
    # read old file (real), chunk 1 (real), chunk 2 denied, cleanup (real)
    original_run = fake._run
    calls = {"n": 0}

    async def flaky(command: str) -> ExecResult:
        calls["n"] += 1
        if calls["n"] == 3:
            return ok("", stderr=reason, exit_code=77, denied=True, deny_reason=reason)
        return await original_run(command)

    fake._run = flaky  # type: ignore[method-assign]
    with pytest.raises(ToolError, match="Safety review: flagged"):
        await exec_write(meta, WriteInput(file_path="keep.txt", content=content))
    assert (root / "keep.txt").read_text() == "original\n"
    assert not list(root.glob(".keep.txt.spellbook-*"))


@pytest.mark.asyncio
async def test_image_read_through_sandbox(tmp_path: Path) -> None:
    meta, _, root = sandbox_meta(tmp_path)
    png = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 4
    (root / "pic.png").write_bytes(png)
    result = await exec_read(meta, ReadInput(file_path="pic.png"))
    block = result.content[0]
    assert isinstance(block, IRImageBlock)
    assert base64.b64decode(block.source.data) == png
    assert (tmp_path / "host" / block.blob_path).read_bytes() == png


@pytest.mark.asyncio
async def test_oversized_read_and_denied_read(tmp_path: Path) -> None:
    fake = FakeProvider([ok("FILE\nAAAA", output_truncated=True)])
    meta = provider_meta(tmp_path, fake)
    with pytest.raises(ToolError, match="too large to read through the sandbox"):
        await exec_read(meta, ReadInput(file_path="huge.log"))
    reason = "Safety review: Shell text requires review."
    fake.results = [
        ok("", stderr=reason, exit_code=77, denied=True, deny_reason=reason)
    ]
    with pytest.raises(ToolError, match="Shell text requires review"):
        await exec_read(meta, ReadInput(file_path="secret.txt"))


@pytest.mark.asyncio
async def test_overlong_bash_command_is_rejected_before_sending(tmp_path: Path) -> None:
    fake = FakeProvider()
    meta = provider_meta(tmp_path, fake)
    with pytest.raises(ToolError, match="too long for the sandbox"):
        await exec_bash(meta, BashInput(command="echo " + "x" * 70_000))
    assert fake.execs == []

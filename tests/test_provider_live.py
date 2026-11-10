"""Live end-to-end: a real Spellbook session against a running Provider.

Opt-in; skipped unless SPELLBOOK_PROVIDER_LIVE_URL and
SPELLBOOK_PROVIDER_LIVE_TOKEN_FILE are set. Only the model is scripted —
config parsing, SessionManager, the tool registry, Bash/Read/Write/Edit,
ProviderBash, the chorus_provider client and the Provider are all real. The
token needs exec/put/get/create/destroy (the familiar tier).

    SPELLBOOK_PROVIDER_LIVE_URL=http://192.168.1.210:8787 \\
    SPELLBOOK_PROVIDER_LIVE_TOKEN_FILE=/tmp/provider-scout-token.txt \\
    SPELLBOOK_PROVIDER_LIVE_HOST_HEADER=localhost \\
    uv run --extra provider pytest -q tests/test_provider_live.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from urllib.parse import urlsplit

import pytest

chorus_provider = pytest.importorskip("chorus_provider")

from test_session_manager import _gen, _ScriptedBackend, _user_msg  # noqa: E402

from spellbook.ir_types import (  # noqa: E402
    IRAssistantTextBlock,
    IRToolCallBlock,
    IRToolResultBlock,
)
from spellbook.serve import build_spellbook_config, load_entity_file  # noqa: E402
from spellbook.session_manager import SessionManager  # noqa: E402

URL = os.environ.get("SPELLBOOK_PROVIDER_LIVE_URL", "")
TOKEN_FILE = os.environ.get("SPELLBOOK_PROVIDER_LIVE_TOKEN_FILE", "")
HOST_HEADER = os.environ.get("SPELLBOOK_PROVIDER_LIVE_HOST_HEADER", "")

pytestmark = pytest.mark.skipif(
    not (URL and TOKEN_FILE), reason="live Provider not configured"
)

HUMAN = "please try the sandbox: say hello, write a script, run it, then tidy up"
SCRIPT = 'print("hello from a file in the sandbox")\nprint(sum(range(10)))\n'


def _call(call_id: str, tool: str, **input: object) -> IRToolCallBlock:
    return IRToolCallBlock(origin="model", call_id=call_id, tool=tool, input=input)


def _results(blocks: list) -> dict[str, IRToolResultBlock]:
    return {b.call_id: b for b in blocks if isinstance(b, IRToolResultBlock)}


def _text(result: IRToolResultBlock) -> str:
    return "\n".join(getattr(part, "text", "") for part in result.content)


@pytest.mark.asyncio
async def test_live_session_turn_through_provider(tmp_path: Path, monkeypatch) -> None:
    insecure = urlsplit(URL).scheme == "http" and urlsplit(URL).hostname not in (
        "127.0.0.1",
        "localhost",
        "::1",
    )
    tools = [
        f'provider_url = "{URL}"',
        f'provider_token_file = "{TOKEN_FILE}"',
        'provider_role = "familiar"',
        f"provider_allow_insecure_http = {'true' if insecure else 'false'}",
    ]
    if HOST_HEADER:
        tools.append(f'provider_host_header = "{HOST_HEADER}"')
    entity = tmp_path / "entity.toml"
    entity.write_text(
        '[entity]\nmodel = "claude-sonnet-4-6"\ncwd = "."\n\n[tools]\n'
        + "\n".join(tools)
        + "\n",
        encoding="utf-8",
    )
    config = build_spellbook_config(load_entity_file(entity))

    calls = [
        _call(
            "hello",
            "Bash",
            command='echo "hello from the sandbox" && uname -sr && pwd && id -un',
        ),
        _call("write", "Write", file_path="/workspace/notes/hello.py", content=SCRIPT),
        _call("run", "Bash", command="python3 notes/hello.py"),
        _call(
            "edit",
            "Edit",
            file_path="/workspace/notes/hello.py",
            old_string="sum(range(10))",
            new_string="sum(range(100))",
        ),
        _call("read", "Read", file_path="/workspace/notes/hello.py"),
        _call("rerun", "Bash", command="python3 notes/hello.py"),
        _call("fail", "Bash", command="echo oops >&2; exit 3"),
        _call(
            "host",
            "Bash",
            command="test -e /home/rheaton64 && echo HOST || echo ISOLATED",
        ),
    ]
    backend = _ScriptedBackend(
        [_gen([c], stop_reason="tool_use") for c in calls]
        + [_gen([IRAssistantTextBlock(text="Done.")], stop_reason="end_turn")]
    )
    monkeypatch.setattr(
        "spellbook.session_manager.build_backend", lambda config: backend
    )
    manager = await SessionManager.build(
        transcript_path=tmp_path / "live.jsonl", config=config
    )
    provider_bash = manager.executor.meta.provider_bash
    assert provider_bash is not None and provider_bash.token_file is not None

    runner = asyncio.create_task(manager.run())
    await manager.submit_message(_user_msg(HUMAN))
    for _ in range(1200):
        if len(backend.surfaces) > len(calls):
            break
        await asyncio.sleep(0.05)
    sandbox_id = provider_bash.sandbox_id
    await manager.shutdown()
    await asyncio.wait_for(runner, timeout=30)

    results = _results(backend.blocks_seen[-1])
    transcript = "\n".join(
        f"--- {name} ({'error' if r.is_error else 'ok'})\n{_text(r)}"
        for name, r in results.items()
    )
    assert sandbox_id is not None
    text = {name: _text(r) for name, r in results.items()}
    checks = {
        "every call succeeded": not any(r.is_error for r in results.values()),
        "hello": "hello from the sandbox" in text["hello"]
        and "/workspace" in text["hello"]
        and "gvisor" in text["hello"].lower(),
        "run": text["run"].split()[-1] == "45",
        "edit+read": "sum(range(100))" in text["read"],
        "rerun": text["rerun"].split()[-1] == "4950",
        "nonzero exit": "Exit code 3" in text["fail"] and "oops" in text["fail"],
        "host isolation": text["host"].strip() == "ISOLATED",
    }
    failed = [name for name, passed in checks.items() if not passed]
    assert not failed, f"{failed}\n{transcript}"

    # Out-of-band: the session destroyed its sandbox on shutdown, and the
    # Provider recorded every command with the human's words as review context.
    async with chorus_provider.ProviderClient(
        URL,
        token_file=TOKEN_FILE,
        host_header=HOST_HEADER or None,
        allow_insecure_http=insecure,
    ) as client:
        history = await client.sandbox_history(sandbox_id)
    assert history["status"] == "destroyed" and history["destroy_reason"] == "requested"
    commands = history["commands"]
    assert len(commands) >= len(calls)
    assert {c["context"] for c in commands} == {HUMAN}

"""Tests for core filesystem tools."""

from __future__ import annotations

import asyncio
import base64
import os
import signal
from contextlib import suppress
from pathlib import Path
from time import perf_counter
from typing import cast

import pytest

from spellbook.cancel_token import CancelToken
from spellbook.config import SpellbookConfig
from spellbook.executor import Executor
from spellbook.generator import Generator
from spellbook.ir_types import (
    IRAssistantTextBlock,
    IRBlock,
    IRBlockRecord,
    IRGeneration,
    IRImageBase64Source,
    IRImageBlock,
    IRRecord,
    IRSkillCatalog,
    IRToolCallBlock,
    IRToolResultBlock,
    IRToolTextBlock,
    IRUsage,
)
from spellbook.loop import run_loop
from spellbook.recorder import Recorder, RecordingRoundLifecycle
from spellbook.round_lifecycle import RoundLifecycle
from spellbook.tools.common import ToolError, ToolMetadata
from spellbook.tools.filesystem import (
    BASH_TOOL,
    BashInput,
    EditInput,
    ReadInput,
    WriteInput,
    _signal_process_group,
    exec_bash,
    exec_edit,
    exec_read,
    exec_write,
)
from spellbook.tools.registry import ToolRegistry

pytestmark = pytest.mark.asyncio


def _meta(tmp_path: Path) -> ToolMetadata:
    return ToolMetadata(cwd=tmp_path, transcript_path=tmp_path / "transcript.jsonl")


def _result_text(block: object) -> str:
    assert isinstance(block, IRToolTextBlock)
    return block.text


def _bash_executor(tmp_path: Path) -> Executor:
    config = SpellbookConfig(cwd=tmp_path)
    return Executor(
        config,
        tmp_path / "transcript.jsonl",
        ToolRegistry(tools=[BASH_TOOL]),
    )


def _bash_call(command: str, *, timeout: int) -> IRToolCallBlock:
    return IRToolCallBlock(
        origin="model",
        call_id="toolu_bash",
        tool="Bash",
        input={"command": command, "timeout": timeout},
    )


class _QueuedGenerator:
    def __init__(self, responses: list[IRGeneration]) -> None:
        self._responses = list(responses)

    async def run(
        self,
        blocks: list[IRBlock],
        cancel_token: CancelToken,
        lifecycle: RoundLifecycle,
    ) -> IRGeneration:
        return self._responses.pop(0)


async def _wait_for_pid(path: Path, *, timeout: float = 2.0) -> int:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        try:
            value = path.read_text(encoding="utf-8").strip()
            if value:
                return int(value)
        except (FileNotFoundError, ValueError):
            pass
        await asyncio.sleep(0.01)
    raise AssertionError(f"Timed out waiting for process id in {path}")


def _process_is_running(pid: int) -> bool:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
    except FileNotFoundError:
        return False
    return fields[2] not in {"Z", "X"}


async def _wait_for_process_stop(pid: int, *, timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if not _process_is_running(pid):
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"Process {pid} is still running")


async def _kill_process(pid: int) -> None:
    with suppress(ProcessLookupError):
        os.kill(pid, signal.SIGKILL)
    await _wait_for_process_stop(pid)


async def test_read_returns_line_numbered_text(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    result = await exec_read(_meta(tmp_path), ReadInput(file_path="notes.txt"))

    assert _result_text(result.content[0]) == (
        "Lines 1-3 of 3 (3 lines)\n     1\talpha\n     2\tbeta\n     3\tgamma"
    )
    assert result.display == {
        "kind": "read",
        "path": str(path),
        "start_line": 1,
        "end_line": 3,
        "total_lines": 3,
        "line_count": 3,
    }


async def test_read_respects_offset_and_limit(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    result = await exec_read(
        _meta(tmp_path),
        ReadInput(file_path="notes.txt", offset=2, limit=1),
    )

    assert _result_text(result.content[0]) == ("Lines 2-2 of 3 (1 lines)\n     2\tbeta")
    assert result.display["start_line"] == 2
    assert result.display["end_line"] == 2
    assert result.display["line_count"] == 1


async def test_read_missing_file_errors(tmp_path: Path) -> None:
    with pytest.raises(ToolError, match="File not found"):
        await exec_read(_meta(tmp_path), ReadInput(file_path="missing.txt"))


async def test_read_image_returns_image_and_persists_relative_blob(
    tmp_path: Path,
) -> None:
    image_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
        "/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
    )
    (tmp_path / "tiny.png").write_bytes(image_bytes)

    result = await exec_read(_meta(tmp_path), ReadInput(file_path="tiny.png"))

    assert len(result.content) == 1
    image = result.content[0]
    assert isinstance(image, IRImageBlock)
    assert isinstance(image.source, IRImageBase64Source)
    assert image.source.media_type == "image/png"
    assert image.source.data == base64.standard_b64encode(image_bytes).decode("ascii")
    assert image.blob_path is not None
    assert not Path(image.blob_path).is_absolute()
    assert image.blob_path.startswith("blobs/")
    assert (tmp_path / image.blob_path).read_bytes() == image_bytes
    assert result.display["title"] == "Read Image"
    assert "blobs/" in result.display["body"]


async def test_read_directory_errors(tmp_path: Path) -> None:
    directory = tmp_path / "folder"
    directory.mkdir()

    with pytest.raises(ToolError, match="Path is a directory"):
        await exec_read(_meta(tmp_path), ReadInput(file_path="folder"))


async def test_write_creates_parent_dirs_and_returns_diff_display(
    tmp_path: Path,
) -> None:
    result = await exec_write(
        _meta(tmp_path),
        WriteInput(file_path="nested/example.py", content="print('hi')\n"),
    )

    path = tmp_path / "nested" / "example.py"
    assert path.read_text(encoding="utf-8") == "print('hi')\n"
    assert _result_text(result.content[0]) == f"Successfully wrote to {path}"
    assert result.display["kind"] == "diff"
    assert result.display["path"] == str(path)
    assert result.display["change_type"] == "create"
    assert result.display["language"] == "py"
    assert result.display["summary"] == "Created file"
    assert result.display["stats"] == {"added": 1, "removed": 0}
    assert result.display["truncated"] is False
    assert "+print('hi')" in result.display["diff"]


async def test_write_overwrites_existing_text_file(tmp_path: Path) -> None:
    path = tmp_path / "example.txt"
    path.write_text("old\n", encoding="utf-8")

    result = await exec_write(
        _meta(tmp_path),
        WriteInput(file_path="example.txt", content="new\n"),
    )

    assert path.read_text(encoding="utf-8") == "new\n"
    assert result.display["change_type"] == "overwrite"
    assert result.display["summary"] == "Overwrote file"
    assert result.display["stats"] == {"added": 1, "removed": 1}
    assert "-old" in result.display["diff"]
    assert "+new" in result.display["diff"]


async def test_write_directory_errors(tmp_path: Path) -> None:
    (tmp_path / "folder").mkdir()

    with pytest.raises(ToolError, match="Path is a directory"):
        await exec_write(
            _meta(tmp_path),
            WriteInput(file_path="folder", content="nope"),
        )


async def test_edit_replaces_unique_string(tmp_path: Path) -> None:
    path = tmp_path / "example.txt"
    path.write_text("hello world\n", encoding="utf-8")

    result = await exec_edit(
        _meta(tmp_path),
        EditInput(
            file_path="example.txt",
            old_string="world",
            new_string="Ryan",
        ),
    )

    assert path.read_text(encoding="utf-8") == "hello Ryan\n"
    assert _result_text(result.content[0]) == f"Successfully edited {path}"
    assert result.display["change_type"] == "edit"
    assert result.display["summary"] == "Replaced 1 occurrence"
    assert result.display["stats"] == {"added": 1, "removed": 1}


async def test_edit_requires_unique_match_by_default(tmp_path: Path) -> None:
    path = tmp_path / "example.txt"
    path.write_text("same\nsame\n", encoding="utf-8")

    with pytest.raises(ToolError, match="appears 2 times"):
        await exec_edit(
            _meta(tmp_path),
            EditInput(file_path="example.txt", old_string="same", new_string="done"),
        )

    assert path.read_text(encoding="utf-8") == "same\nsame\n"


async def test_edit_replace_all_changes_every_occurrence(tmp_path: Path) -> None:
    path = tmp_path / "example.txt"
    path.write_text("same\nsame\n", encoding="utf-8")

    result = await exec_edit(
        _meta(tmp_path),
        EditInput(
            file_path="example.txt",
            old_string="same",
            new_string="done",
            replace_all=True,
        ),
    )

    assert path.read_text(encoding="utf-8") == "done\ndone\n"
    assert result.display["summary"] == "Replaced 2 occurrences"
    assert result.display["stats"] == {"added": 2, "removed": 2}


async def test_edit_missing_string_errors(tmp_path: Path) -> None:
    path = tmp_path / "example.txt"
    path.write_text("hello world\n", encoding="utf-8")

    with pytest.raises(ToolError, match="old_string not found"):
        await exec_edit(
            _meta(tmp_path),
            EditInput(file_path="example.txt", old_string="missing", new_string="new"),
        )

    assert path.read_text(encoding="utf-8") == "hello world\n"


async def test_edit_empty_old_string_errors(tmp_path: Path) -> None:
    path = tmp_path / "example.txt"
    path.write_text("hello world\n", encoding="utf-8")

    with pytest.raises(ToolError, match="old_string must not be empty"):
        await exec_edit(
            _meta(tmp_path),
            EditInput(file_path="example.txt", old_string="", new_string="new"),
        )


async def test_bash_happy_path_is_unchanged(tmp_path: Path) -> None:
    result = await exec_bash(
        _meta(tmp_path), BashInput(command="printf 'hello from bash'")
    )

    assert _result_text(result.content[0]) == "hello from bash"
    assert result.display["exit_code"] == 0
    assert result.display["stdout"] == "hello from bash"


async def test_bash_exited_shell_with_open_background_pipe_returns_promptly(
    tmp_path: Path,
) -> None:
    pid_path = tmp_path / "detached.pid"
    pid: int | None = None
    start = perf_counter()
    try:
        result = await exec_bash(
            _meta(tmp_path),
            BashInput(
                command=(
                    "nohup sh -c 'sleep 1.2; printf late-output; sleep 60' & "
                    "echo $! > detached.pid"
                ),
                timeout=200,
            ),
        )
        elapsed = perf_counter() - start
        pid = await _wait_for_pid(pid_path)

        output = _result_text(result.content[0])
        assert elapsed < 2.0
        assert result.display["exit_code"] == 0
        assert "a background process is still running" in output
        assert "output may be incomplete" in output
        assert _process_is_running(pid)
        await asyncio.sleep(0.4)
        assert _process_is_running(pid)
    finally:
        if pid is None and pid_path.exists():
            pid = int(pid_path.read_text(encoding="utf-8"))
        if pid is not None:
            await _kill_process(pid)


async def test_bash_timeout_reaps_process_group_and_records_paired_error_result(
    tmp_path: Path,
) -> None:
    call = _bash_call(
        "echo $$ > leader.pid; "
        "sh -c 'trap \"\" TERM; echo $$ > child.pid; exec sleep 60' & "
        "wait",
        timeout=500,
    )
    generator = _QueuedGenerator(
        [
            IRGeneration(
                model="test",
                blocks=[call],
                stop_reason="tool_use",
                usage=IRUsage(),
            ),
            IRGeneration(
                model="test",
                blocks=[IRAssistantTextBlock(text="continued", origin="model")],
                stop_reason="end_turn",
                usage=IRUsage(),
            ),
        ]
    )
    registry = ToolRegistry(tools=[BASH_TOOL])
    config = SpellbookConfig(cwd=tmp_path)
    executor = Executor(config, tmp_path / "transcript.jsonl", registry)
    seen_records: list[IRRecord] = []
    recorder = Recorder(
        config,
        tmp_path / "transcript.jsonl",
        "session_bash_timeout",
        registry,
        record_tap=seen_records.append,
    )
    recorder.write_session_record(IRSkillCatalog())
    recorder.start_turn("turn_bash_timeout", [])

    loop_result = await run_loop(
        generator=cast(Generator, generator),
        executor=executor,
        lifecycle=RecordingRoundLifecycle(recorder),
        initial_blocks=[],
        cancel_token=CancelToken(),
    )
    recorder.end_turn(loop_result.stop_reason)

    leader_pid = await _wait_for_pid(tmp_path / "leader.pid")
    child_pid = await _wait_for_pid(tmp_path / "child.pid")
    await _wait_for_process_stop(leader_pid)
    await _wait_for_process_stop(child_pid)

    assert loop_result.stop_reason == "end_turn"
    block_events = [
        record.event for record in seen_records if isinstance(record, IRBlockRecord)
    ]
    assert isinstance(block_events[0], IRToolCallBlock)
    assert isinstance(block_events[1], IRToolResultBlock)
    assert block_events[0].call_id == block_events[1].call_id == "toolu_bash"
    result = block_events[1]
    assert result.call_id == "toolu_bash"
    assert result.is_error is True
    assert "Command timed out" in _result_text(result.content[0])


async def test_bash_cancellation_reaps_process_group_and_returns_cancelled_result(
    tmp_path: Path,
) -> None:
    executor = _bash_executor(tmp_path)
    cancel_token = CancelToken()
    task = asyncio.create_task(
        executor.run(
            [
                _bash_call(
                    "echo $$ > leader.pid; sleep 60 & echo $! > child.pid; wait",
                    timeout=10_000,
                )
            ],
            cancel_token,
        )
    )
    leader_pid = await _wait_for_pid(tmp_path / "leader.pid")
    child_pid = await _wait_for_pid(tmp_path / "child.pid")

    cancel_token.cancel()
    execution = await asyncio.wait_for(task, timeout=2.0)
    await _wait_for_process_stop(leader_pid)
    await _wait_for_process_stop(child_pid)

    assert execution.cancelled_early is True
    assert len(execution.blocks) == 1
    result = execution.blocks[0]
    assert result.call_id == "toolu_bash"
    assert result.is_error is True
    assert "interrupted" in _result_text(result.content[0])


async def test_bash_signal_treats_missing_process_group_as_already_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_process_lookup(process_group_id: int, sig: signal.Signals) -> None:
        raise ProcessLookupError

    monkeypatch.setattr(os, "killpg", raise_process_lookup)

    _signal_process_group(999_999, signal.SIGKILL)


@pytest.mark.anyio
async def test_bash_nonexistent_cwd_returns_tool_error(tmp_path: Path) -> None:
    """Pre-spawn OSError (e.g. vanished cwd) is a tool result, not a crash."""
    gone = tmp_path / "vanished"
    meta = ToolMetadata(cwd=gone, transcript_path=tmp_path / "t.jsonl")
    with pytest.raises(ToolError) as excinfo:
        await exec_bash(meta, BashInput(command="echo hello"))
    message = str(excinfo.value)
    assert "Command could not start" in message
    assert "vanished" in message

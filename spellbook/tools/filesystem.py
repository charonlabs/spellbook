"""Filesystem tools: Read, Write, Edit, Bash.

Each tool is a pair: a Pydantic ``*Input`` model describing arguments,
and an ``exec_*`` async function implementing behavior. They're
assembled into a ``Tool`` constant at module bottom that the registry
picks up.

``Bash`` runs a shell command under the entity's cwd with a configurable
timeout. Stderr merges into stdout for single-stream output. On
timeout, the subprocess is killed and partial output is returned as a
``ToolError``.
"""

import asyncio
import base64
import difflib
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from pydantic import BaseModel, Field

from spellbook.ir_types import (
    IMAGE_MEDIA_TYPES,
    IRImageBase64Source,
    IRImageBlock,
    IRToolResultContentBlock,
    IRToolTextBlock,
)

from .common import Tool, ToolError, ToolExecutionResult, ToolMetadata

TOOL_DISPLAY_MAX_CHARS = 6000

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


def _resolve_path(meta: ToolMetadata, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = meta.cwd / path
    return path


def _tool_text(text: str) -> list[IRToolResultContentBlock]:
    return [IRToolTextBlock(text=text)]


def _guess_language(path: Path) -> str | None:
    suffix = path.suffix.lstrip(".").lower()
    return suffix or None


def _truncate_display_text(
    text: str, *, max_chars: int = TOOL_DISPLAY_MAX_CHARS
) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    clipped = text[: max_chars - 15].rstrip()
    return f"{clipped}\n... [truncated]", True


def _build_unified_diff(
    old_text: str, new_text: str, *, path: Path, change_type: str
) -> tuple[str, bool]:
    fromfile = "/dev/null" if change_type == "create" else str(path)
    diff_text = "".join(
        difflib.unified_diff(
            old_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=fromfile,
            tofile=str(path),
            lineterm="\n",
        )
    )
    return _truncate_display_text(diff_text)


def _diff_stats(diff_text: str) -> dict[str, int]:
    added = 0
    removed = 0
    for line in diff_text.splitlines():
        if line.startswith(("+++", "---", "@@")):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
    return {"added": added, "removed": removed}


def _read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except PermissionError as e:
        raise ToolError(f"Permission denied: {path}") from e
    except UnicodeDecodeError as e:
        raise ToolError(f"File is not valid UTF-8 text: {path}") from e
    except OSError as e:
        raise ToolError(f"Could not read {path}: {e}") from e


def _read_image_file(path: Path, transcript_path: Path) -> ToolExecutionResult:
    try:
        image_bytes = path.read_bytes()
    except PermissionError as e:
        raise ToolError(f"Permission denied: {path}") from e
    except OSError as e:
        raise ToolError(f"Could not read {path}: {e}") from e

    size_bytes = len(image_bytes)
    media_type = IMAGE_MEDIA_TYPES.get(path.suffix.lower(), "image/png")

    # Store blob
    blob_dir = transcript_path.parent / "blobs"
    blob_name = f"{uuid4().hex[:12]}{path.suffix.lower()}"
    blob_path = blob_dir / blob_name
    try:
        blob_dir.mkdir(parents=True, exist_ok=True)
        blob_path.write_bytes(image_bytes)
    except OSError as e:
        raise ToolError(f"Could not persist image blob for {path}: {e}") from e
    relative_blob_path = Path("blobs") / blob_name

    b64_data = base64.standard_b64encode(image_bytes).decode("ascii")
    content: list[IRToolResultContentBlock] = [
        IRImageBlock(
            origin="tool",
            source=IRImageBase64Source(media_type=media_type, data=b64_data),
            blob_path=str(relative_blob_path),
        )
    ]
    size_label = (
        f"{size_bytes / 1024:.1f}KB" if size_bytes >= 1024 else f"{size_bytes}B"
    )
    output = f"[image: {path} -> {relative_blob_path}, {size_label}]"

    return ToolExecutionResult(
        content=content, display={"kind": "text", "title": "Read Image", "body": output}
    )


def _write_text_file(path: Path, text: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except PermissionError as e:
        raise ToolError(f"Permission denied: {path}") from e
    except OSError as e:
        raise ToolError(f"Could not write {path}: {e}") from e


# --- Read ---


class ReadInput(BaseModel):
    """Read a UTF-8 text file from the filesystem."""

    file_path: str = Field(
        description="The path to the file to read. Relative paths resolve from cwd.",
    )
    offset: int | None = Field(
        default=None,
        ge=1,
        description="The line number to start reading from (1-indexed).",
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        description="The number of lines to read.",
    )


async def exec_read(meta: ToolMetadata, input: ReadInput) -> ToolExecutionResult:
    path = _resolve_path(meta, input.file_path)
    if not path.exists():
        raise ToolError(f"File not found: {path}")
    if path.is_dir():
        raise ToolError(f"Path is a directory, not a file: {path}")

    # Image files: base64-encode, store blob, return as image content block
    if path.suffix.lower() in _IMAGE_EXTENSIONS:
        return _read_image_file(path, meta.transcript_path)

    text = _read_text_file(path)
    lines = text.splitlines()
    total_lines = len(lines)
    start_index = input.offset - 1 if input.offset is not None else 0
    end_index = start_index + input.limit if input.limit is not None else total_lines
    start_index = min(start_index, total_lines)
    end_index = min(end_index, total_lines)
    selected = lines[start_index:end_index]
    line_count = len(selected)

    if line_count:
        start_line = start_index + 1
        end_line = start_index + line_count
    else:
        start_line = start_index + 1 if total_lines else 0
        end_line = start_line - 1 if total_lines else 0

    numbered = [
        f"{line_number:>6}\t{line}"
        for line_number, line in enumerate(selected, start=start_index + 1)
    ]
    indicator = f"Lines {start_line}-{end_line} of {total_lines} ({line_count} lines)"
    output = indicator
    if numbered:
        output = output + "\n" + "\n".join(numbered)

    return ToolExecutionResult(
        content=_tool_text(output),
        display={
            "kind": "read",
            "path": str(path),
            "start_line": start_line,
            "end_line": end_line,
            "total_lines": total_lines,
            "line_count": line_count,
        },
    )


# --- Write ---


class WriteInput(BaseModel):
    """Write a UTF-8 text file. Overwrites existing files."""

    file_path: str = Field(
        description="The path to the file to write. Relative paths resolve from cwd.",
    )
    content: str = Field(
        description="The content to write to the file.",
    )


async def exec_write(meta: ToolMetadata, input: WriteInput) -> ToolExecutionResult:
    path = _resolve_path(meta, input.file_path)
    if path.exists() and path.is_dir():
        raise ToolError(f"Path is a directory, not a file: {path}")

    existed = path.exists()
    old_text = _read_text_file(path) if existed else ""
    _write_text_file(path, input.content)

    change_type = "overwrite" if existed else "create"
    diff_text, truncated = _build_unified_diff(
        old_text, input.content, path=path, change_type=change_type
    )
    summary = "Overwrote file" if existed else "Created file"
    output = f"Successfully wrote to {path}"
    return ToolExecutionResult(
        content=_tool_text(output),
        display={
            "kind": "diff",
            "path": str(path),
            "change_type": change_type,
            "language": _guess_language(path),
            "summary": summary,
            "stats": _diff_stats(diff_text),
            "diff": diff_text,
            "truncated": truncated,
        },
    )


# --- Edit ---


class EditInput(BaseModel):
    """Perform exact string replacement in a UTF-8 text file."""

    file_path: str = Field(
        description="The path to the file to modify. Relative paths resolve from cwd.",
    )
    old_string: str = Field(
        description="The exact text to replace.",
    )
    new_string: str = Field(
        description="The text to replace it with.",
    )
    replace_all: bool = Field(
        default=False,
        description="Replace all occurrences of old_string.",
    )


async def exec_edit(meta: ToolMetadata, input: EditInput) -> ToolExecutionResult:
    path = _resolve_path(meta, input.file_path)
    if not path.exists():
        raise ToolError(f"File not found: {path}")
    if path.is_dir():
        raise ToolError(f"Path is a directory, not a file: {path}")
    if input.old_string == "":
        raise ToolError("old_string must not be empty.")

    old_text = _read_text_file(path)
    count = old_text.count(input.old_string)
    if count == 0:
        raise ToolError(
            f"old_string not found in {path}. Make sure it matches exactly, "
            "including whitespace and indentation."
        )
    if count > 1 and not input.replace_all:
        raise ToolError(
            f"old_string appears {count} times in {path}. Use replace_all=true "
            "to replace all occurrences, or provide more surrounding context "
            "to make the match unique."
        )

    if input.replace_all:
        new_text = old_text.replace(input.old_string, input.new_string)
        replaced_count = count
    else:
        new_text = old_text.replace(input.old_string, input.new_string, 1)
        replaced_count = 1

    _write_text_file(path, new_text)
    diff_text, truncated = _build_unified_diff(
        old_text, new_text, path=path, change_type="edit"
    )
    replacement_label = "occurrence" if replaced_count == 1 else "occurrences"
    output = f"Successfully edited {path}"
    return ToolExecutionResult(
        content=_tool_text(output),
        display={
            "kind": "diff",
            "path": str(path),
            "change_type": "edit",
            "language": _guess_language(path),
            "summary": f"Replaced {replaced_count} {replacement_label}",
            "stats": _diff_stats(diff_text),
            "diff": diff_text,
            "truncated": truncated,
        },
    )


# --- Bash ---
BASH_TOOL_DEFAULT_TIMEOUT_MS = 30000
BASH_TOOL_MAX_TIMEOUT_MS = 600000


class BashInput(BaseModel):
    """Execute a shell command and return its output."""

    command: str = Field(
        description="The command to execute",
    )
    timeout: int | None = Field(
        default=None,
        description=f"Optional timeout in milliseconds (default {BASH_TOOL_DEFAULT_TIMEOUT_MS} if omitted, max {BASH_TOOL_MAX_TIMEOUT_MS})",
    )


# TODO: backgrounding, and threading the CancelToken
async def exec_bash(meta: ToolMetadata, input: BashInput) -> ToolExecutionResult:
    timeout_ms = input.timeout or BASH_TOOL_DEFAULT_TIMEOUT_MS
    timeout_ms = min(timeout_ms, BASH_TOOL_MAX_TIMEOUT_MS)
    timeout_s = timeout_ms / 1000

    start = perf_counter()
    proc = await asyncio.create_subprocess_shell(
        input.command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,  # merge for clean backgrounding
        cwd=meta.cwd,
    )

    collected: list[bytes] = []
    try:

        async def _read_all() -> None:
            while True:
                chunk = await proc.stdout.read(8192)  # type: ignore
                if not chunk:
                    break
                collected.append(chunk)
            await proc.wait()

        await asyncio.wait_for(_read_all(), timeout=timeout_s)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        partial_output = b"".join(collected).decode(errors="replace").rstrip()
        raise ToolError(f"Command timed out after {timeout_s:.0f}s\n{partial_output}")
    duration_ms = int((perf_counter() - start) * 1000)
    output = b"".join(collected).decode(errors="replace").rstrip()
    if proc.returncode != 0:
        output = (
            f"Exit code {proc.returncode}\n{output}"
            if output
            else f"Exit code {proc.returncode}"
        )
    return ToolExecutionResult(
        content=_tool_text(output),
        display={
            "kind": "command",
            "command": input.command,
            "description": "",
            "exit_code": proc.returncode or 0,
            "duration_ms": duration_ms,
            "cwd": str(meta.cwd),
            "stdout": output,
            "stderr": "",
            "combined_truncated": False,
        },
    )


READ_TOOL: Tool[ReadInput] = Tool(
    name="Read",
    input_model=ReadInput,
    exec=exec_read,
    category="filesystem",
)

WRITE_TOOL: Tool[WriteInput] = Tool(
    name="Write",
    input_model=WriteInput,
    exec=exec_write,
    category="filesystem",
)

EDIT_TOOL: Tool[EditInput] = Tool(
    name="Edit",
    input_model=EditInput,
    exec=exec_edit,
    category="filesystem",
)

BASH_TOOL: Tool[BashInput] = Tool(
    name="Bash",
    input_model=BashInput,
    exec=exec_bash,
    category="filesystem",
)

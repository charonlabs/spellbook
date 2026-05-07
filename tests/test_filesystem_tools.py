"""Tests for core filesystem tools."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from spellbook.ir_types import IRImageBase64Source, IRImageBlock, IRToolTextBlock
from spellbook.tools.common import ToolError, ToolMetadata
from spellbook.tools.filesystem import (
    EditInput,
    ReadInput,
    WriteInput,
    exec_edit,
    exec_read,
    exec_write,
)

pytestmark = pytest.mark.asyncio


def _meta(tmp_path: Path) -> ToolMetadata:
    return ToolMetadata(cwd=tmp_path, transcript_path=tmp_path / "transcript.jsonl")


def _result_text(block: object) -> str:
    assert isinstance(block, IRToolTextBlock)
    return block.text


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

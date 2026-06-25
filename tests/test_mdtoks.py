from __future__ import annotations

from pathlib import Path

import pytest
from scripts.dreaming.mdtoks import DEFAULT_MODEL, count_markdown_tokens
from spellbook.backends.model_backend import RequestSurface
from spellbook.ir_types import IRBlock, IRUserTextBlock

pytestmark = pytest.mark.asyncio


class _RecordingTokenCounter:
    def __init__(self, tokens: int) -> None:
        self.tokens = tokens
        self.blocks: list[IRBlock] | None = None

    async def count_block_content(self, block: IRBlock) -> int | None:
        return None

    async def count_blocks(self, blocks: list[IRBlock]) -> int | None:
        self.blocks = blocks
        return self.tokens

    async def count_frame(self) -> int | None:
        return None

    async def count_surface(self, surface: RequestSurface) -> int | None:
        return None


async def test_count_markdown_tokens_counts_file_as_single_user_message(
    tmp_path: Path,
) -> None:
    markdown = tmp_path / "chapter.md"
    markdown.write_text("# Chapter\n\nHello **world**.\n", encoding="utf-8")
    counter = _RecordingTokenCounter(tokens=42)

    report = await count_markdown_tokens(markdown, token_counter=counter)

    assert report.path == markdown.resolve()
    assert report.model == DEFAULT_MODEL
    assert report.tokens == 42
    assert report.characters == len("# Chapter\n\nHello **world**.\n")
    assert counter.blocks is not None
    assert len(counter.blocks) == 1
    block = counter.blocks[0]
    assert isinstance(block, IRUserTextBlock)
    assert block.origin == "human"
    assert block.text == "# Chapter\n\nHello **world**.\n"

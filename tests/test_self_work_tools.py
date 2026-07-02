from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from spellbook.homunculus.block_manager import ForgetBlockResult
from spellbook.ir_types import IRToolTextBlock
from spellbook.tools.common import ToolMetadata
from spellbook.tools.self_work import ForgetInput, exec_forget

pytestmark = pytest.mark.asyncio


class _FakeHomunculus:
    def __init__(self) -> None:
        self.calls: list[tuple[int, bool]] = []

    async def forget(self, block_idx: int, confirm: bool = False) -> ForgetBlockResult:
        self.calls.append((block_idx, confirm))
        return ForgetBlockResult(
            status="summary_queued",
            message=(
                f"Block {block_idx}'s summary is still baking; I queued it just now. "
                "Try Forget again in a moment."
            ),
        )


async def test_forget_returns_kind_result_when_summary_is_queued(
    tmp_path: Path,
) -> None:
    homunculus = _FakeHomunculus()
    meta = ToolMetadata(
        cwd=tmp_path,
        transcript_path=tmp_path / "transcript.jsonl",
        homunculus=cast(Any, homunculus),
    )

    result = await exec_forget(meta, ForgetInput(block_idx=2))

    assert homunculus.calls == [(2, False)]
    assert result.display == {
        "kind": "forget_block",
        "block_idx": 2,
        "status": "summary_queued",
    }
    assert result.content == [
        IRToolTextBlock(
            text=(
                "Block 2's summary is still baking; I queued it just now. "
                "Try Forget again in a moment."
            )
        )
    ]

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.repair_replay_metrics import repair_replay_transcript
from spellbook.backends.model_backend import RequestSurface
from spellbook.config import SpellbookConfig
from spellbook.ir_types import (
    IRBlock,
    IRBlockDetectionRecord,
    IRBlockRecord,
    IRRecord,
    IRSemanticBlockMetricsRecord,
    IRSemanticBlockRange,
    IRSemanticBlockRecord,
    IRSessionRecord,
    IRSkillCatalog,
    IRToolCallBlock,
    IRToolResultBlock,
    IRToolTextBlock,
    IRTurnEndRecord,
    IRTurnStartRecord,
    IRUserTextBlock,
)
from spellbook.rehydrator import Rehydrator
from spellbook.tools.registry import ToolRegistry

pytestmark = pytest.mark.asyncio


class _FakeTokenCounter:
    async def count_block_content(self, block: IRBlock) -> int | None:
        return 10

    async def count_blocks(self, blocks: list[IRBlock]) -> int | None:
        return len(blocks) * 10

    async def count_frame(self) -> int | None:
        return 1000

    async def count_surface(self, surface: RequestSurface) -> int | None:
        return 1000


def _write_records(path: Path, records: list[IRRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(record.model_dump_json() for record in records) + "\n",
        encoding="utf-8",
    )


def _transcript(path: Path) -> None:
    config = SpellbookConfig(model="claude-sonnet-4-6", cwd=path.parent)
    registry = ToolRegistry.build(config.tool_categories, surface=config.session_type)
    semantic_range = IRSemanticBlockRange(
        id="range_1",
        title="Imported quirk",
        start_block=0,
        end_block=3,
        completed=True,
    )
    _write_records(
        path,
        [
            IRSessionRecord(
                session_id="session_test",
                config=config,
                tools=registry.records,
                skill_catalog=IRSkillCatalog(),
            ),
            IRTurnStartRecord(
                session_id="session_test",
                turn=1,
                turn_id="turn_1",
            ),
            IRBlockRecord(
                session_id="session_test",
                turn=1,
                seq=0,
                event=IRUserTextBlock(text="run it", origin="human", turn_id="turn_1"),
            ),
            IRBlockRecord(
                session_id="session_test",
                turn=1,
                seq=1,
                event=IRToolCallBlock(
                    call_id="toolu_1",
                    tool="Bash",
                    input={"command": "pwd"},
                    turn_id="turn_1",
                ),
            ),
            IRTurnEndRecord(
                session_id="session_test",
                turn=1,
                turn_id="turn_1",
                stop_reason="tool_use",
            ),
            IRTurnStartRecord(
                session_id="session_test",
                turn=2,
                turn_id="turn_2",
            ),
            IRBlockRecord(
                session_id="session_test",
                turn=2,
                seq=0,
                event=IRUserTextBlock(
                    text="cwd is already right",
                    origin="human",
                    turn_id="turn_2",
                ),
            ),
            IRBlockRecord(
                session_id="session_test",
                turn=2,
                seq=1,
                event=IRToolResultBlock(
                    call_id="toolu_1",
                    tool="Bash",
                    content=[IRToolTextBlock(text="/tmp")],
                    turn_id="turn_2",
                ),
            ),
            IRTurnEndRecord(
                session_id="session_test",
                turn=2,
                turn_id="turn_2",
                stop_reason="end_turn",
            ),
            IRBlockDetectionRecord(
                session_id="session_test",
                completed=[semantic_range],
                still_buffered=[],
                turn=2,
                turn_id="turn_2",
            ),
            IRSemanticBlockRecord(
                session_id="session_test",
                id="block_1",
                idx=0,
                range_id=semantic_range.id,
                toks=None,
                full_toks=None,
                turn=2,
                turn_id="turn_2",
            ),
        ],
    )


async def test_repair_replay_transcript_reorders_and_backfills_metrics(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "transcript.jsonl"
    report_json = tmp_path / "repair-report.json"
    _transcript(transcript)

    report = await repair_replay_transcript(
        transcript_path=transcript,
        report_json=report_json,
        token_counter=_FakeTokenCounter(),
    )

    rehydrated = Rehydrator(transcript).run()
    block_records = [
        record for record in rehydrated.records if isinstance(record, IRBlockRecord)
    ]
    metric_records = [
        record
        for record in rehydrated.records
        if isinstance(record, IRSemanticBlockMetricsRecord)
    ]

    assert report.order_repair.records_moved == 1
    assert report.metrics_written == 1
    assert report.backup_path is not None
    assert report.backup_path.exists()
    assert report_json.exists()
    assert [record.seq for record in block_records] == [0, 1, 0, 1]
    assert isinstance(block_records[2].event, IRToolResultBlock)
    assert isinstance(block_records[3].event, IRUserTextBlock)
    assert len(metric_records) == 1
    assert metric_records[0].toks.tokens == 40
    assert rehydrated.semantic_blocks[0].full_toks is not None
    assert rehydrated.semantic_blocks[0].full_toks.tokens == 40

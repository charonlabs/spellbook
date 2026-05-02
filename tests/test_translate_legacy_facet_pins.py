from __future__ import annotations

import json
from pathlib import Path

from scripts.translate_legacy_facet_pins import (
    analyze_legacy_facet_pins,
    map_turn_range_to_core_blocks,
    rank_core_facets,
)
from spellbook.config import SpellbookConfig
from spellbook.ir_types import (
    IRBlockDetectionRecord,
    IRBlockRecord,
    IRRecord,
    IRSemanticBlockArtifactRecord,
    IRSemanticBlockFacet,
    IRSemanticBlockRange,
    IRSemanticBlockRecord,
    IRSemanticBlockSummary,
    IRSessionRecord,
    IRSkillCatalog,
    IRTurnEndRecord,
    IRTurnStartRecord,
    IRUserTextBlock,
)
from spellbook.tools.registry import ToolRegistry


def _write_records(path: Path, records: list[IRRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(record.model_dump_json() for record in records) + "\n",
        encoding="utf-8",
    )


def _write_legacy_transcript(path: Path) -> None:
    records = [
        _legacy_system_event(
            {
                "kind": "block_detected",
                "block_id": "legacy_block_1",
                "title": "Legacy Work",
                "turn_range": [1, 4],
            }
        ),
        _legacy_system_event(
            {
                "kind": "block_artifact_created",
                "block_id": "legacy_block_1",
                "artifact": {
                    "kind": "facet_index",
                    "text": "2 facets prepared",
                    "payload": {
                        "facets": [
                            {
                                "id": "pinned_facet",
                                "headline": "Pinned legacy facet",
                                "summary": "The important old facet.",
                                "turn_range": [2, 3],
                            },
                            {
                                "id": "unpinned_facet",
                                "headline": "Not pinned",
                                "summary": "A different facet.",
                                "turn_range": [4, 4],
                            },
                        ]
                    },
                },
            }
        ),
        _legacy_system_event(
            {
                "kind": "facet_pinned",
                "block_id": "legacy_block_1",
                "facet_id": "pinned_facet",
                "source": "bootstrap",
            }
        ),
    ]
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def _legacy_system_event(data: dict) -> dict:
    return {
        "ir": "event",
        "turn": 0,
        "event": {
            "type": "system",
            "layer": "activity",
            "seq": 0,
            "time": "2026-04-29T00:00:00Z",
            "data": data,
        },
    }


def _write_core_transcript(path: Path) -> None:
    config = SpellbookConfig(model="claude-sonnet-4-6", cwd=path.parent)
    registry = ToolRegistry.build(config.tool_categories, surface=config.session_type)
    semantic_range = IRSemanticBlockRange(
        id="range_1",
        title="Core Work",
        start_block=0,
        end_block=4,
        completed=True,
    )
    records: list[IRRecord] = [
        IRSessionRecord(
            session_id="session_test",
            config=config,
            tools=registry.records,
            skill_catalog=IRSkillCatalog(),
        )
    ]
    context_seq = 0
    for turn, texts in {
        1: ["turn one"],
        2: ["turn two A", "turn two B"],
        3: ["turn three"],
        4: ["turn four"],
    }.items():
        turn_id = f"turn_{turn}"
        records.append(
            IRTurnStartRecord(
                session_id="session_test",
                turn=turn,
                turn_id=turn_id,
            )
        )
        for seq, text in enumerate(texts):
            records.append(
                IRBlockRecord(
                    session_id="session_test",
                    turn=turn,
                    seq=seq,
                    event=IRUserTextBlock(
                        text=text,
                        origin="human",
                        turn_id=turn_id,
                    ),
                )
            )
            context_seq += 1
        records.append(
            IRTurnEndRecord(
                session_id="session_test",
                turn=turn,
                turn_id=turn_id,
                stop_reason="end_turn",
            )
        )

    assert context_seq == 5
    records.extend(
        [
            IRBlockDetectionRecord(
                session_id="session_test",
                completed=[semantic_range],
                still_buffered=[],
                turn=4,
                turn_id="turn_4",
            ),
            IRSemanticBlockRecord(
                session_id="session_test",
                id="core_block_1",
                idx=0,
                range_id=semantic_range.id,
                toks=None,
                full_toks=None,
                turn=4,
                turn_id="turn_4",
            ),
            IRSemanticBlockArtifactRecord(
                session_id="session_test",
                block_id="core_block_1",
                artifact=IRSemanticBlockSummary(
                    headline="Core summary",
                    text="Summary text.",
                    facets=[
                        IRSemanticBlockFacet(
                            id="core_exact",
                            title="Exact core facet",
                            description="Matches the old turn range.",
                            start_block=1,
                            end_block=3,
                            resources=[],
                        ),
                        IRSemanticBlockFacet(
                            id="core_partial",
                            title="Partial core facet",
                            description="Only clips the old range.",
                            start_block=0,
                            end_block=1,
                            resources=[],
                        ),
                    ],
                    open_thread=None,
                    toks=None,
                ),
                turn=4,
                turn_id="turn_4",
            ),
        ]
    )
    _write_records(path, records)


def test_map_turn_range_to_core_blocks_uses_all_event_blocks_in_turns() -> None:
    index = {1: [0], 2: [1, 2], 3: [3], 4: [4]}

    assert map_turn_range_to_core_blocks(index, turn_start=2, turn_end=3) == (1, 3)
    assert map_turn_range_to_core_blocks(index, turn_start=9, turn_end=10) is None


def test_rank_core_facets_prefers_highest_overlap() -> None:
    from scripts.translate_legacy_facet_pins import CoreFacet

    exact = CoreFacet(
        block_id="core_block_1",
        block_idx=0,
        block_title="Core Work",
        facet_id="exact",
        facet_title="Exact",
        facet_description="",
        start_block=1,
        end_block=3,
    )
    partial = CoreFacet(
        block_id="core_block_1",
        block_idx=0,
        block_title="Core Work",
        facet_id="partial",
        facet_title="Partial",
        facet_description="",
        start_block=0,
        end_block=1,
    )

    ranked = rank_core_facets(
        mapped_start=1,
        mapped_end=3,
        core_facets=[partial, exact],
    )

    assert [candidate.facet.facet_id for candidate in ranked[:2]] == [
        "exact",
        "partial",
    ]
    assert ranked[0].jaccard == 1.0
    assert ranked[0].legacy_coverage == 1.0
    assert ranked[0].core_coverage == 1.0


def test_analyze_legacy_facet_pins_reports_core_overlap_candidates(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.jsonl"
    core = tmp_path / "core.jsonl"
    _write_legacy_transcript(legacy)
    _write_core_transcript(core)

    report = analyze_legacy_facet_pins(
        legacy_path=legacy,
        core_path=core,
        top=2,
    )

    assert len(report.matches) == 1
    match = report.matches[0]
    assert match.legacy_pin.facet_id == "pinned_facet"
    assert match.legacy_pin.facet_title == "Pinned legacy facet"
    assert (match.mapped_start_block, match.mapped_end_block) == (1, 3)
    assert [candidate.facet.facet_id for candidate in match.candidates] == [
        "core_exact",
        "core_partial",
    ]
    assert match.candidates[0].jaccard == 1.0

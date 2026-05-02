from __future__ import annotations

import json
from pathlib import Path

from pydantic import TypeAdapter

from scripts.repair_tool_result_ttls import repair_tool_result_ttls
from spellbook.config import HomunculusConfig, SpellbookConfig
from spellbook.ir_types import (
    IRRecord,
    IRSkillCatalog,
    IRToolCallBlock,
    IRToolResultBlock,
    IRToolResultTTLRecord,
    IRToolTextBlock,
    IRUserTextBlock,
)
from spellbook.recorder import Recorder
from spellbook.rehydrator import Rehydrator
from spellbook.tools.registry import DEFAULT_TOOL_REGISTRY


def _recorder(tmp_path: Path) -> tuple[Recorder, Path]:
    transcript = tmp_path / "transcript.jsonl"
    config = SpellbookConfig(
        cwd=tmp_path,
        hom_config=HomunculusConfig(
            tool_result_ttl_turns=3,
            tool_result_ttl_char_threshold=20,
        ),
    )
    recorder = Recorder(config, transcript, "session_test", DEFAULT_TOOL_REGISTRY)
    recorder.write_session_record(skill_catalog=IRSkillCatalog())
    return recorder, transcript


def _tool_result(
    call_id: str,
    text: str,
    *,
    tool: str = "Read",
    display: dict | None = None,
) -> IRToolResultBlock:
    return IRToolResultBlock(
        call_id=call_id,
        tool=tool,
        content=[IRToolTextBlock(text=text)],
        display=display or {},
    )


def _read_records(path: Path) -> list[IRRecord]:
    adapter = TypeAdapter(IRRecord)
    return [
        adapter.validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_repair_tool_result_ttls_dry_run_reports_without_mutating(tmp_path: Path) -> None:
    recorder, transcript = _recorder(tmp_path)
    output = "large output\n" * 3
    recorder.start_turn("turn_1", [IRUserTextBlock(text="read it", origin="human")])
    recorder.write_block(IRToolCallBlock(call_id="toolu_big", tool="Read", input={}))
    recorder.write_block(
        _tool_result(
            "toolu_big",
            output,
            display={
                "kind": "read",
                "path": "/tmp/big.txt",
                "start_line": 1,
                "end_line": 3,
                "total_lines": 3,
            },
        )
    )
    recorder.end_turn()
    before = transcript.read_text(encoding="utf-8")

    report = repair_tool_result_ttls(
        transcript,
        apply=False,
        write_report=False,
    )

    assert transcript.read_text(encoding="utf-8") == before
    assert len(report.candidates) == 1
    candidate = report.candidates[0]
    assert candidate.call_id == "toolu_big"
    assert candidate.output_ref == "tool-outputs/toolu_big.txt"
    assert "[Read: /tmp/big.txt - lines 1-3 of 3." in candidate.replace_content
    assert not (tmp_path / "tool-outputs" / "toolu_big.txt").exists()
    assert not any(
        isinstance(record, IRToolResultTTLRecord)
        for record in Rehydrator(transcript).run().records
    )


def test_repair_tool_result_ttls_appends_records_and_saves_outputs(tmp_path: Path) -> None:
    recorder, transcript = _recorder(tmp_path)
    output = "large output\n" * 3
    recorder.start_turn("turn_1", [IRUserTextBlock(text="read it", origin="human")])
    recorder.write_block(IRToolCallBlock(call_id="toolu_big", tool="Read", input={}))
    recorder.write_block(
        _tool_result(
            "toolu_big",
            output,
            display={
                "kind": "read",
                "path": "/tmp/big.txt",
                "start_line": 1,
                "end_line": 3,
                "total_lines": 3,
            },
        )
    )
    recorder.end_turn()

    report = repair_tool_result_ttls(
        transcript,
        apply=True,
        backup=False,
        ttl_turns=1,
        write_report=False,
    )

    assert report.records_appended == 1
    assert (tmp_path / "tool-outputs" / "toolu_big.txt").read_text() == output
    records = _read_records(transcript)
    ttl_records = [
        record for record in records if isinstance(record, IRToolResultTTLRecord)
    ]
    assert len(ttl_records) == 1
    ttl_record = ttl_records[0]
    assert ttl_record.call_id == "toolu_big"
    assert ttl_record.source == "repair"
    assert ttl_record.ttl == 1
    assert ttl_record.delivered_turn == 1
    assert ttl_record.turn == 1
    assert ttl_record.turn_id == "turn_1"
    assert ttl_record.output_ref == "tool-outputs/toolu_big.txt"
    assert isinstance(records[-1], IRToolResultTTLRecord)


def test_repair_tool_result_ttls_skips_existing_small_errors_and_skill_tools(
    tmp_path: Path,
) -> None:
    recorder, transcript = _recorder(tmp_path)
    recorder.start_turn("turn_1", [IRUserTextBlock(text="tools", origin="human")])
    recorder.write_block(_tool_result("toolu_existing", "large\n" * 10))
    recorder.write_tool_result_ttl(
        call_id="toolu_existing",
        replace_content="[already collapsed]",
        ttl=3,
        trigger="end_turn",
    )
    recorder.write_block(_tool_result("toolu_small", "tiny"))
    recorder.write_block(
        _tool_result("toolu_error", "large\n" * 10).model_copy(
            update={"is_error": True}
        )
    )
    recorder.write_block(_tool_result("toolu_skill", "large\n" * 10, tool="Skill"))
    recorder.end_turn()

    report = repair_tool_result_ttls(
        transcript,
        apply=True,
        backup=False,
        write_report=False,
    )

    assert report.candidates == []
    assert report.skipped["already_has_ttl"] == 1
    assert report.skipped["below_threshold"] == 1
    assert report.skipped["error_result"] == 1
    assert report.skipped["memory_or_skill_tool"] == 1
    records = _read_records(transcript)
    assert sum(isinstance(record, IRToolResultTTLRecord) for record in records) == 1


def test_repair_tool_result_ttls_writes_report_json(tmp_path: Path) -> None:
    recorder, transcript = _recorder(tmp_path)
    report_path = tmp_path / "report.json"
    recorder.start_turn("turn_1", [IRUserTextBlock(text="read it", origin="human")])
    recorder.write_block(_tool_result("toolu_big", "large\n" * 10))
    recorder.end_turn()

    repair_tool_result_ttls(
        transcript,
        apply=False,
        report_json=report_path,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["dry_run"] is True
    assert report["candidate_count"] == 1

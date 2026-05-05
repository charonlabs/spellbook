from __future__ import annotations

from pathlib import Path
from typing import Sequence, cast

import pytest

from scripts import replay_context
from spellbook.config import HomunculusConfig, SpellbookConfig
from spellbook.fork import BlockDetectorResult
from spellbook.ir_types import (
    IRBlock,
    IRSemanticBlock,
    IRSemanticBlockRange,
    IRSemanticBlockSummary,
    IRSkillCatalog,
    IRTurnStartRecord,
    IRUserTextBlock,
)
from spellbook.recorder import Recorder
from spellbook.rehydrator import RehydrationResult, Rehydrator
from spellbook.tools.registry import DEFAULT_TOOL_REGISTRY


def _user(text: str) -> IRUserTextBlock:
    return IRUserTextBlock(text=text, origin="human")


def _write_source_transcript(
    tmp_path: Path,
    blocks: list[IRBlock],
    *,
    session_id: str = "source_session",
    hom_config: HomunculusConfig | None = None,
) -> Path:
    return _write_source_turns(
        tmp_path,
        [("source_turn_1", blocks)],
        session_id=session_id,
        hom_config=hom_config,
    )


def _write_source_turns(
    tmp_path: Path,
    turns: list[tuple[str, list[IRBlock]]],
    *,
    session_id: str = "source_session",
    hom_config: HomunculusConfig | None = None,
) -> Path:
    transcript = tmp_path / "source.jsonl"
    config = SpellbookConfig(
        model="claude-sonnet-4-6",
        cwd=tmp_path,
        hom_config=hom_config or HomunculusConfig(),
    )
    recorder = Recorder(config, transcript, session_id, DEFAULT_TOOL_REGISTRY)
    recorder.write_session_record(skill_catalog=IRSkillCatalog())
    for turn_id, blocks in turns:
        recorder.start_turn(turn_id, blocks)
        recorder.end_turn()
    return transcript


class _FakeBlockManager:
    def __init__(self) -> None:
        self.appended: list[list[IRBlock]] = []
        self.append_results: list[
            tuple[list[IRSemanticBlockRange], list[IRSemanticBlock]]
        ] = []
        self.proposed_semantic_blocks: list[IRSemanticBlockRange] = []
        self.semantic_blocks: list[IRSemanticBlock] = []
        self.context_blocks: list[IRBlock] = []
        self.next_block_id = 0
        self.check_nursery_wait_flags: list[bool] = []
        self.force_detect_results: list[bool] = []
        self.force_detect_finalize_flags: list[bool] = []

    def rehydrate(self, rehydrated: RehydrationResult) -> None:
        self.context_blocks = list(rehydrated.blocks)
        self.semantic_blocks = list(rehydrated.semantic_blocks)
        self.next_block_id = len(self.context_blocks)

    async def append_context_blocks(self, blocks: Sequence[IRBlock]) -> int:
        start = self.next_block_id
        self.appended.append(list(blocks))
        self.context_blocks.extend(blocks)
        self.next_block_id += len(blocks)
        if self.append_results:
            proposed, completed = self.append_results.pop(0)
            self.proposed_semantic_blocks = proposed
            self.semantic_blocks = completed
        return start

    async def generate_next_summary(self) -> None:
        return None

    async def force_detect(self, *, finalize: bool = False) -> bool:
        self.force_detect_finalize_flags.append(finalize)
        if not self.force_detect_results:
            return False
        return self.force_detect_results.pop(0)

    async def check_nursery(
        self,
        *,
        wait_for_all: bool = False,
    ) -> None:
        self.check_nursery_wait_flags.append(wait_for_all)
        return None


@pytest.mark.asyncio
async def test_replay_preserves_source_turns_and_replayed_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_source_turns(
        tmp_path,
        [
            ("source_turn_alpha", [_user("one"), _user("two")]),
            ("source_turn_beta", [_user("three")]),
        ],
    )
    output = tmp_path / "replay" / "transcript.jsonl"
    fake_manager = _FakeBlockManager()
    monkeypatch.setattr(
        replay_context,
        "_build_block_manager",
        lambda **kwargs: fake_manager,
    )

    report = await replay_context.replay_transcript(
        transcript_path=source,
        output_path=output,
        interval=2,
    )
    result = Rehydrator(output).run()

    assert report.source_blocks == 3
    assert report.replayed_blocks == 3
    assert report.replay_session_id == "source_session_replay"
    assert [(tick.start_block, tick.end_block) for tick in report.ticks] == [
        (0, 1),
        (2, 2),
    ]
    assert [len(batch) for batch in fake_manager.appended] == [2, 1]
    assert fake_manager.check_nursery_wait_flags == [True, True, True]
    assert result.last_completed_turn == 2
    assert [block.turn_id for block in result.blocks] == [
        "source_turn_alpha",
        "source_turn_alpha",
        "source_turn_beta",
    ]
    assert [
        block.text for block in result.blocks if isinstance(block, IRUserTextBlock)
    ] == [
        "one",
        "two",
        "three",
    ]


@pytest.mark.asyncio
async def test_replay_max_blocks_limits_source_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_source_turns(
        tmp_path,
        [
            ("source_turn_alpha", [_user("one")]),
            ("source_turn_beta", [_user("two")]),
            ("source_turn_gamma", [_user("three")]),
        ],
    )
    output = tmp_path / "replay.jsonl"
    fake_manager = _FakeBlockManager()
    monkeypatch.setattr(
        replay_context,
        "_build_block_manager",
        lambda **kwargs: fake_manager,
    )

    report = await replay_context.replay_transcript(
        transcript_path=source,
        output_path=output,
        interval=10,
        max_blocks=2,
    )
    result = Rehydrator(output).run()

    assert report.source_blocks == 3
    assert report.replayed_blocks == 2
    assert len(result.blocks) == 2
    assert result.last_completed_turn == 2
    assert [
        block.text for block in result.blocks if isinstance(block, IRUserTextBlock)
    ] == [
        "one",
        "two",
    ]


@pytest.mark.asyncio
async def test_replay_uses_source_detect_interval_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_source_transcript(
        tmp_path,
        [_user("one"), _user("two"), _user("three")],
        hom_config=HomunculusConfig(detect_interval=2),
    )
    output = tmp_path / "replay.jsonl"
    fake_manager = _FakeBlockManager()
    monkeypatch.setattr(
        replay_context,
        "_build_block_manager",
        lambda **kwargs: fake_manager,
    )

    report = await replay_context.replay_transcript(
        transcript_path=source,
        output_path=output,
    )

    assert report.interval == 2
    assert [len(batch) for batch in fake_manager.appended] == [2, 1]


@pytest.mark.asyncio
async def test_replay_interval_overrides_detector_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_source_transcript(
        tmp_path,
        [_user("one"), _user("two"), _user("three")],
        hom_config=HomunculusConfig(detect_interval=5),
    )
    output = tmp_path / "replay.jsonl"
    fake_manager = _FakeBlockManager()
    seen_config: SpellbookConfig | None = None

    def _fake_build_block_manager(**kwargs: object) -> _FakeBlockManager:
        nonlocal seen_config
        seen_config = cast(SpellbookConfig, kwargs["config"])
        return fake_manager

    monkeypatch.setattr(
        replay_context,
        "_build_block_manager",
        _fake_build_block_manager,
    )

    report = await replay_context.replay_transcript(
        transcript_path=source,
        output_path=output,
        interval=2,
    )
    result = Rehydrator(output).run()

    assert report.interval == 2
    assert seen_config is not None
    assert seen_config.hom_config.detect_interval == 2
    assert result.config.hom_config.detect_interval == 2
    assert [len(batch) for batch in fake_manager.appended] == [2, 1]


@pytest.mark.asyncio
async def test_replay_force_overwrites_output_and_clears_forks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_source_transcript(tmp_path, [])
    output = tmp_path / "replay" / "transcript.jsonl"
    output.parent.mkdir()
    output.write_text("stale", encoding="utf-8")
    forks_dir = output.parent / "forks"
    forks_dir.mkdir()
    (forks_dir / "detector_stale.jsonl").write_text("stale", encoding="utf-8")
    monkeypatch.setattr(
        replay_context,
        "_build_block_manager",
        lambda **kwargs: _FakeBlockManager(),
    )

    await replay_context.replay_transcript(
        transcript_path=source,
        output_path=output,
        force=True,
    )

    assert output.exists()
    assert not forks_dir.exists()
    result = Rehydrator(output).run()
    assert result.session_id == "source_session_replay"
    assert result.last_completed_turn == 1


def test_replay_refuses_existing_output_without_force(tmp_path: Path) -> None:
    _write_source_transcript(tmp_path, [])
    output = tmp_path / "replay.jsonl"
    output.write_text("stale", encoding="utf-8")

    with pytest.raises(FileExistsError):
        replay_context._prepare_output(output, force=False)  # noqa: SLF001


def test_replay_resume_requires_existing_output(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        replay_context._prepare_output(  # noqa: SLF001
            tmp_path / "missing.jsonl",
            force=False,
            resume=True,
        )


def test_replay_resume_rejects_force(tmp_path: Path) -> None:
    output = tmp_path / "replay.jsonl"
    output.write_text("stale", encoding="utf-8")

    with pytest.raises(ValueError, match="resume and --force"):
        replay_context._prepare_output(output, force=True, resume=True)  # noqa: SLF001


@pytest.mark.asyncio
async def test_replay_rejects_non_positive_interval(tmp_path: Path) -> None:
    source = _write_source_transcript(tmp_path, [])

    with pytest.raises(ValueError, match="interval"):
        await replay_context.replay_transcript(
            transcript_path=source,
            output_path=tmp_path / "replay.jsonl",
            interval=0,
        )


@pytest.mark.asyncio
async def test_replay_report_counts_semantic_blocks_and_summaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_source_transcript(tmp_path, [_user("one")])
    output = tmp_path / "replay.jsonl"
    fake_manager = _FakeBlockManager()
    semantic_range = IRSemanticBlockRange(
        title="Block",
        start_block=0,
        end_block=0,
        completed=True,
    )
    fake_manager.semantic_blocks = [
        IRSemanticBlock(
            idx=0,
            title="Block",
            range=semantic_range,
            toks=None,
            full_toks=None,
        )
    ]
    monkeypatch.setattr(
        replay_context,
        "_build_block_manager",
        lambda **kwargs: fake_manager,
    )

    report = await replay_context.replay_transcript(
        transcript_path=source,
        output_path=output,
    )

    assert report.semantic_blocks == 1
    assert report.summaries == 0


@pytest.mark.asyncio
async def test_replay_finalize_runs_one_eof_detector_flush(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_source_transcript(tmp_path, [_user("one")])
    output = tmp_path / "replay.jsonl"
    fake_manager = _FakeBlockManager()
    fake_manager.force_detect_results = [True, True]
    monkeypatch.setattr(
        replay_context,
        "_build_block_manager",
        lambda **kwargs: fake_manager,
    )

    report = await replay_context.replay_transcript(
        transcript_path=source,
        output_path=output,
        finalize=True,
    )

    assert report.finalization_passes == 1
    assert fake_manager.force_detect_finalize_flags == [True]
    assert fake_manager.check_nursery_wait_flags == [True, True, True]


@pytest.mark.asyncio
async def test_replay_prints_proposals_completions_and_summaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _write_source_transcript(tmp_path, [_user("one")])
    output = tmp_path / "replay.jsonl"
    proposed_range = IRSemanticBlockRange(
        title="Draft block",
        start_block=0,
        end_block=0,
    )
    completed_range = IRSemanticBlockRange(
        title="Completed block",
        start_block=0,
        end_block=0,
        completed=True,
    )
    summary = IRSemanticBlockSummary(
        headline="Summary headline",
        text="Full summary body.",
        facets=[],
        open_thread="Keep inspecting replay artifacts.",
        toks=None,
    )
    completed_block = IRSemanticBlock(
        idx=0,
        title="Completed block",
        range=completed_range,
        toks=None,
        full_toks=None,
        available_modes=["full", "summary"],
        artifacts=[summary],
    )

    class _RecordingFakeBlockManager(_FakeBlockManager):
        def __init__(self, recorder: Recorder) -> None:
            super().__init__()
            self.recorder = recorder

        async def append_context_blocks(self, blocks: Sequence[IRBlock]) -> int:
            start = await super().append_context_blocks(blocks)
            self.recorder.detect_blocks(
                BlockDetectorResult(
                    completed=[completed_range],
                    still_buffered=[proposed_range],
                )
            )
            self.recorder.write_semantic_block(completed_block)
            self.recorder.write_block_artifact(summary, completed_block.id)
            self.semantic_blocks = [completed_block]
            return start

    def _fake_build_block_manager(**kwargs: object) -> _RecordingFakeBlockManager:
        return _RecordingFakeBlockManager(cast(Recorder, kwargs["recorder"]))

    monkeypatch.setattr(
        replay_context,
        "_build_block_manager",
        _fake_build_block_manager,
    )

    await replay_context.replay_transcript(
        transcript_path=source,
        output_path=output,
    )

    out = capsys.readouterr().out
    assert "Proposed Blocks" in out
    assert "Draft block" in out
    assert "Completed Blocks" in out
    assert "Completed block" in out
    assert "Summary headline" in out
    assert "Full summary body." in out


@pytest.mark.asyncio
async def test_replay_resume_continues_existing_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_source_turns(
        tmp_path,
        [
            ("source_turn_alpha", [_user("one")]),
            ("source_turn_beta", [_user("two")]),
            ("source_turn_gamma", [_user("three")]),
        ],
    )
    output = tmp_path / "replay.jsonl"
    first_manager = _FakeBlockManager()
    second_manager = _FakeBlockManager()
    managers = iter([first_manager, second_manager])
    monkeypatch.setattr(
        replay_context,
        "_build_block_manager",
        lambda **kwargs: next(managers),
    )

    await replay_context.replay_transcript(
        transcript_path=source,
        output_path=output,
        interval=2,
        max_blocks=2,
    )
    prefix_result = Rehydrator(output).run()
    report = await replay_context.replay_transcript(
        transcript_path=source,
        output_path=output,
        resume=True,
    )
    result = Rehydrator(output).run()
    backup_path = report.resume_backup_path
    assert backup_path == output.with_suffix(output.suffix + ".resume.bak")
    assert backup_path is not None
    assert backup_path.exists()
    backup_result = Rehydrator(backup_path).run()

    assert report.replayed_blocks == 3
    assert report.interval == 2
    assert [(tick.start_block, tick.end_block) for tick in report.ticks] == [(2, 2)]
    assert [len(batch) for batch in second_manager.appended] == [1]
    assert len(backup_result.blocks) == 2
    assert backup_result.blocks == prefix_result.blocks
    assert result.last_completed_turn == 3
    assert [
        block.text for block in result.blocks if isinstance(block, IRUserTextBlock)
    ] == ["one", "two", "three"]


@pytest.mark.asyncio
async def test_replay_resume_preserves_empty_source_turn_after_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_source_turns(
        tmp_path,
        [
            ("source_turn_alpha", [_user("one")]),
            ("source_turn_empty", []),
            ("source_turn_beta", [_user("two")]),
        ],
    )
    output = tmp_path / "replay.jsonl"
    managers = iter([_FakeBlockManager(), _FakeBlockManager()])
    monkeypatch.setattr(
        replay_context,
        "_build_block_manager",
        lambda **kwargs: next(managers),
    )

    await replay_context.replay_transcript(
        transcript_path=source,
        output_path=output,
        interval=10,
        max_blocks=1,
    )
    await replay_context.replay_transcript(
        transcript_path=source,
        output_path=output,
        resume=True,
    )
    result = Rehydrator(output).run()

    assert [
        record.turn_id
        for record in result.records
        if isinstance(record, IRTurnStartRecord)
    ] == ["source_turn_alpha", "source_turn_empty", "source_turn_beta"]
    assert result.last_completed_turn == 3
    assert [
        block.text for block in result.blocks if isinstance(block, IRUserTextBlock)
    ] == ["one", "two"]


@pytest.mark.asyncio
async def test_replay_resume_rejects_non_prefix_output(tmp_path: Path) -> None:
    source = _write_source_transcript(tmp_path, [_user("one")])
    output = tmp_path / "replay.jsonl"
    source_result = Rehydrator(source).run()
    config = source_result.config.model_copy(
        update={
            "session_type": "main",
            "tool_categories": None,
            "hom_config": source_result.config.hom_config.model_copy(
                update={"detect_interval": 2}
            ),
        }
    )
    recorder = Recorder(config, output, "source_session_replay", DEFAULT_TOOL_REGISTRY)
    recorder.write_session_record(skill_catalog=IRSkillCatalog())
    recorder.start_turn("source_turn_1", [_user("wrong")])
    recorder.end_turn()

    with pytest.raises(ValueError, match="not a prefix"):
        await replay_context.replay_transcript(
            transcript_path=source,
            output_path=output,
            resume=True,
        )


@pytest.mark.asyncio
async def test_replay_resume_summarizes_unfinished_existing_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _write_source_transcript(tmp_path, [_user("one")])
    output = tmp_path / "replay.jsonl"
    source_result = Rehydrator(source).run()
    config = source_result.config.model_copy(
        update={
            "session_type": "main",
            "tool_categories": None,
            "hom_config": source_result.config.hom_config.model_copy(
                update={"detect_interval": 2}
            ),
        }
    )
    recorder = Recorder(config, output, "source_session_replay", DEFAULT_TOOL_REGISTRY)
    recorder.write_session_record(skill_catalog=IRSkillCatalog())
    recorder.start_turn("source_turn_1", [])
    recorder.write_block(source_result.blocks[0])
    completed_range = IRSemanticBlockRange(
        title="Completed block",
        start_block=0,
        end_block=0,
        completed=True,
    )
    completed_block = IRSemanticBlock(
        idx=0,
        title="Completed block",
        range=completed_range,
        toks=None,
        full_toks=None,
    )
    recorder.detect_blocks(
        BlockDetectorResult(completed=[completed_range], still_buffered=[])
    )
    recorder.write_semantic_block(completed_block)
    summary = IRSemanticBlockSummary(
        headline="Recovered summary",
        text="Summary generated while resuming.",
        facets=[],
        open_thread=None,
        toks=None,
    )

    class _SummarizingFakeBlockManager(_FakeBlockManager):
        def __init__(self, recorder: Recorder) -> None:
            super().__init__()
            self.recorder = recorder

        async def generate_next_summary(self) -> None:
            updated_blocks: list[IRSemanticBlock] = []
            for block in self.semantic_blocks:
                if "summary" in block.available_modes:
                    updated_blocks.append(block)
                    continue
                self.recorder.write_block_artifact(summary, block.id)
                updated_blocks.append(
                    block.model_copy(
                        update={
                            "artifacts": [*block.artifacts, summary],
                            "available_modes": [*block.available_modes, "summary"],
                        }
                    )
                )
            self.semantic_blocks = updated_blocks

    def _fake_build_block_manager(**kwargs: object) -> _SummarizingFakeBlockManager:
        return _SummarizingFakeBlockManager(cast(Recorder, kwargs["recorder"]))

    monkeypatch.setattr(
        replay_context,
        "_build_block_manager",
        _fake_build_block_manager,
    )

    report = await replay_context.replay_transcript(
        transcript_path=source,
        output_path=output,
        resume=True,
    )
    result = Rehydrator(output).run()
    out = capsys.readouterr().out

    assert report.replayed_blocks == 1
    assert report.summaries == 1
    assert result.is_unfinished_turn is False
    assert result.last_completed_turn == 1
    assert "Summary:" in out
    assert "Completed block" in out
    assert "Recovered summary" in out
    assert "Summary generated while resuming." in out
    assert result.semantic_blocks[0].available_modes == ["full", "summary"]

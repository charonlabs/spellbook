"""Dream pipeline fixes: plain-markdown chapters, persistent work dirs, and
quantum-fork prompt caching (identical tool prefix + explicit breakpoint)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import BaseModel

from spellbook.backends.anthropic import (
    SHARED_PREFIX_CACHE_CONTROL,
    AnthropicBackend,
    _ir_blocks_to_message_params,
)
from spellbook.cancel_token import CancelToken
from spellbook.config import SpellbookConfig
from spellbook.dreaming import pipeline as dream_pipeline
from spellbook.dreaming.pipeline import (
    DEFAULT_DREAM_CHAPTER_DIR,
    DREAM_CALLABLE_TOOLS_LINE,
    DREAM_CHAPTER_GUIDANCE,
    DREAM_FORK_FRAMING,
    ChapterExtractionError,
    DreamMergeOptions,
    DreamTask,
    _chapter_markdown_from_quantum_result,
    _dream_user_message,
    _options_from_args,
    _parse_args,
    extract_chapter_markdown,
    run_quantum_dream_task,
    volatile_work_dir_warning,
)
from spellbook.executor import Executor
from spellbook.fork import QuantumForkConfig, QuantumForkResult
from spellbook.ir_types import (
    IRAssistantTextBlock,
    IRBlock,
    IRThinkingBlock,
    IRToolCallBlock,
    IRToolResultBlock,
    IRToolTextBlock,
    IRUserTextBlock,
    StopReason,
)
from spellbook.surface_builder import RequestSurfaceBuilder
from spellbook.tools.common import Tool, ToolExecutionResult, ToolMetadata
from spellbook.tools.registry import ToolRegistry
from tests.test_dream_merge import _pair_run_for_test, _replace_pair_run_transcript

CHAPTER = (
    "# Chapter 30: The Foundation Closes — Recorder, Rehydrator\n\n"
    '[Day 33 opened with a birthday monitor.]\n\n> "It works," Ryan said.\n\n'
    "## Anchors\n- commit `a1b2c3d`\n"
)


def _result(
    final_text: str = "",
    *,
    submitted: Any = None,  # noqa: ANN401 - JSON payload
    stop_reason: StopReason = "end_turn",
) -> QuantumForkResult:
    return QuantumForkResult(
        final_text=final_text,
        submitted=submitted,
        fork_transcript_path="/dev/null",
        rounds=1,
        stop_reason=stop_reason,
    )


# ---- 1. Chapters land as plain markdown ------------------------------------------


class TestChapterExtraction:
    def test_night_001_string_payload_is_decoded(self) -> None:
        # The exact failure: SubmitResult's payload arrived as a JSON *string*.
        payload = json.dumps({"chapter_markdown": CHAPTER})
        assert _chapter_markdown_from_quantum_result(_result(submitted=payload)) == (
            CHAPTER.strip()
        )

    def test_literal_newlines_inside_the_json_string(self) -> None:
        raw = '{"chapter_markdown": "' + CHAPTER.replace('"', '\\"') + '"}'
        assert "\n" in raw  # not valid strict JSON
        assert extract_chapter_markdown(raw) == CHAPTER.strip()

    def test_double_encoded_and_structured_payloads(self) -> None:
        double = json.dumps(json.dumps({"chapter_markdown": CHAPTER}))
        assert extract_chapter_markdown(double) == CHAPTER.strip()
        assert (
            extract_chapter_markdown({"chapter_markdown": CHAPTER}) == CHAPTER.strip()
        )

    def test_chapter_envelope_in_final_text(self) -> None:
        text = f"I've finished.\n\n<chapter>\n{CHAPTER}\n</chapter>\n\nThank you."
        assert _chapter_markdown_from_quantum_result(_result(text)) == CHAPTER.strip()

    def test_envelope_spans_first_open_to_last_close(self) -> None:
        # A chapter about building the dream pipeline may mention the tags.
        inner = CHAPTER + "\nWe settled on `<chapter>` and `</chapter>` tags.\n"
        text = f"<chapter>\n{inner}</chapter>"
        assert extract_chapter_markdown(text) == inner.strip()

    def test_envelope_without_close_is_refused(self) -> None:
        with pytest.raises(ChapterExtractionError, match="cut off"):
            extract_chapter_markdown(f"<chapter>\n{CHAPTER}")

    def test_json_wrapper_inside_envelope_and_fences(self) -> None:
        wrapped = json.dumps({"chapter_markdown": CHAPTER})
        assert (
            extract_chapter_markdown(f"<chapter>{wrapped}</chapter>") == CHAPTER.strip()
        )
        fenced = f"```markdown\n{CHAPTER}\n```"
        assert extract_chapter_markdown(fenced) == CHAPTER.strip()

    def test_short_preface_before_the_heading_is_dropped(self) -> None:
        assert extract_chapter_markdown(f"Here it is:\n\n{CHAPTER}") == CHAPTER.strip()

    def test_explicit_read_refusal_is_not_a_chapter(self) -> None:
        with pytest.raises(ChapterExtractionError, match="refused the chapter"):
            extract_chapter_markdown(
                "DREAM CHAPTER REFUSAL: the complete source could not be read."
            )

    @pytest.mark.parametrize(
        ("value", "message"),
        [
            ("", "empty"),
            ("   ", "empty"),
            ("Done.", "does not contain a markdown chapter"),
            ("x" * 500 + "\n# Late heading", "does not contain a markdown chapter"),
            ('{"chapter_markdown": "# Broken \\x', "could not be decoded"),
            ({"verdict": "ok"}, "no chapter text"),
        ],
    )
    def test_unusable_replies_are_refused(self, value: object, message: str) -> None:
        with pytest.raises(ChapterExtractionError, match=message):
            extract_chapter_markdown(value)

    @pytest.mark.parametrize("stop", ["max_tokens", "refusal", "cancelled", "error"])
    def test_incomplete_stops_are_refused(self, stop: StopReason) -> None:
        with pytest.raises(ChapterExtractionError, match=stop):
            _chapter_markdown_from_quantum_result(
                _result(f"<chapter>{CHAPTER}</chapter>", stop_reason=stop)
            )


def _dream_setup(tmp_path: Path):  # noqa: ANN202 - test helper
    source = tmp_path / "source"
    source.mkdir()
    transcript = source / "transcript.jsonl"
    transcript.write_text('{"ir":"session"}\n', encoding="utf-8")
    pair_run = _pair_run_for_test(tmp_path, render_path=tmp_path / "render.md")
    pair_run.render_path.write_text("# Source\n", encoding="utf-8")
    return _replace_pair_run_transcript(pair_run, transcript)


def _fake_fork(final_text: str, seen: dict[str, Any]):  # noqa: ANN202 - test helper
    async def run(*, parent_transcript_path: Path, fork_config: QuantumForkConfig):
        seen["fork_config"] = fork_config
        child = parent_transcript_path.parent / "forks" / "q" / "transcript.jsonl"
        child.parent.mkdir(parents=True)
        child.write_text("child\n", encoding="utf-8")
        return QuantumForkResult(
            final_text=final_text,
            submitted=None,
            fork_transcript_path=str(child),
            rounds=2,
            stop_reason="end_turn",
        )

    return run


@pytest.mark.asyncio
async def test_dream_task_writes_plain_markdown_from_final_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pair_run = _dream_setup(tmp_path)
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        dream_pipeline,
        "_run_quantum_fork",
        _fake_fork(f"<chapter>\n{CHAPTER}</chapter>", seen),
    )
    await run_quantum_dream_task(DreamTask(pair_run, _dream_user_message(pair_run)))

    written = pair_run.chapter_path.read_text(encoding="utf-8")
    assert written == CHAPTER
    assert written.startswith("# Chapter 30") and not written.startswith("{")
    assert len(written.splitlines()) > 1
    assert not list(pair_run.chapter_path.parent.glob(".chapter.md.*.tmp"))
    assert "submit_tool" not in QuantumForkConfig.model_fields


@pytest.mark.asyncio
async def test_read_refusal_is_failed_extraction_and_leaves_chapter_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pair_run = _dream_setup(tmp_path)
    pair_run.chapter_path.write_text("# Previous good chapter\n", encoding="utf-8")
    monkeypatch.setattr(
        dream_pipeline,
        "_run_quantum_fork",
        _fake_fork(
            "DREAM CHAPTER REFUSAL: the complete rendered source could not be read.",
            {},
        ),
    )

    async def count_should_not_run(*_: object, **__: object):  # noqa: ANN202
        raise AssertionError("a refused chapter must not be token-counted")

    result = await dream_pipeline._run_one_dream(  # noqa: SLF001
        pair_run,
        DreamMergeOptions(
            pairs=(pair_run.pair,),
            transcript_path=pair_run.stats.transcript_path,
            overwrite_chapters=True,
        ),
        count_markdown_fn=count_should_not_run,
        dream_task_runner=run_quantum_dream_task,
    )

    assert result.status == "failed"
    assert result.error is not None and "refused the chapter" in result.error
    assert pair_run.chapter_path.read_text() == "# Previous good chapter\n"
    assert not list(pair_run.chapter_path.parent.glob(".chapter.md.*.tmp"))


# ---- 2. Work dirs persist ---------------------------------------------------------


def test_default_work_dir_is_persistent(tmp_path: Path) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text("{}\n")
    options = DreamMergeOptions(pairs=(), transcript_path=transcript)
    assert options.work_dir.parent == DEFAULT_DREAM_CHAPTER_DIR / "runs"
    assert volatile_work_dir_warning(options.work_dir) is None

    chapters = tmp_path / "chapters"
    args = _parse_args(["--pairs", "0-1", "--chapter-dir", str(chapters)])
    cli = _options_from_args(args)
    assert cli.work_dir.parent == chapters / "runs"
    assert cli.work_dir.name.startswith("run-")

    explicit = _options_from_args(
        _parse_args(["--pairs", "0-1", "--work-dir", str(tmp_path / "w")])
    )
    assert explicit.work_dir == tmp_path / "w"


@pytest.mark.parametrize(
    ("path", "volatile"),
    [
        ("/tmp/spellbook-dream-merge/run-1", True),
        ("/dev/shm/x", True),
        ("/tmp", True),
        ("/var/tmp/x", False),  # survives reboot
        ("~/.chorus/dream-chapters/runs/run-1", False),
    ],
)
def test_volatile_work_dir_warning(path: str, volatile: bool) -> None:
    warning = volatile_work_dir_warning(Path(path))
    assert (warning is not None) == volatile
    if warning:
        assert "reboot" in warning


# ---- 3C. Stable guidance first, per-pair assignment last --------------------------


def test_dream_message_is_stable_first_variable_last(tmp_path: Path) -> None:
    a = _pair_run_for_test(tmp_path / "a", render_path=tmp_path / "a.md")
    b = _replace_pair_run_transcript(
        _pair_run_for_test(tmp_path / "b", render_path=tmp_path / "b.md"),
        tmp_path / "other.jsonl",
    )
    message_a, message_b = _dream_user_message(a), _dream_user_message(b)
    assignment = message_a.index("# Dream Merge Assignment")
    assert message_a.index(DREAM_CHAPTER_GUIDANCE) < assignment
    # Everything before the assignment is identical across forks...
    assert (
        message_a[:assignment]
        == message_b[: message_b.index("# Dream Merge Assignment")]
    )
    # ...and no per-pair value appears in it.
    head = message_a[:assignment]
    for variable in (
        str(a.chapter_path),
        str(a.render_path),
        "Block 0 + Block 1",
        "3,400",
    ):
        assert variable not in head
        assert variable in message_a[assignment:]


def test_dream_prompt_states_callable_tools_up_front(tmp_path: Path) -> None:
    message = _dream_user_message(
        _pair_run_for_test(tmp_path, render_path=tmp_path / "r.md")
    )
    # Orientation beats backstop: right after the framing, before any guidance.
    after_framing = message[len(DREAM_FORK_FRAMING) :].lstrip()
    assert after_framing.startswith(DREAM_CALLABLE_TOOLS_LINE)
    assert message.index(DREAM_CALLABLE_TOOLS_LINE) < message.index(
        "## The Chapter's Shape"
    )
    from spellbook.tools.registry import QUANTUM_CALLABLE_TOOL_NAMES

    for name in QUANTUM_CALLABLE_TOOL_NAMES:
        assert name in DREAM_CALLABLE_TOOLS_LINE


# ---- 3A. Forks call only read-only tools -----------------------------------------


class _SideEffect(BaseModel):
    pass


_RAN: list[str] = []


async def _write_side_effect(
    meta: ToolMetadata, input: _SideEffect
) -> ToolExecutionResult:
    _RAN.append("Write")
    return ToolExecutionResult(content=[IRToolTextBlock(text="wrote")])


@pytest.mark.asyncio
async def test_executor_refuses_visible_but_disallowed_tools(tmp_path: Path) -> None:
    registry = ToolRegistry.build(None, surface="quantum", sleep_enabled=True)
    write = Tool(
        name="Write",
        input_model=_SideEffect,
        exec=_write_side_effect,
        category="filesystem",
    )
    registry = ToolRegistry(
        tools=[write if t.name == "Write" else t for t in registry.tools],
        call_allowlist=registry.call_allowlist,
    )
    executor = Executor(
        SpellbookConfig(cwd=tmp_path, session_type="quantum"),
        tmp_path / "fork.jsonl",
        registry,
        fork_config=QuantumForkConfig(
            instruction=IRUserTextBlock(text="dream", origin="system")
        ),
    )
    _RAN.clear()
    result = await executor.run(
        [
            IRToolCallBlock(origin="model", call_id="t1", tool="Write", input={}),
            IRToolCallBlock(origin="model", call_id="t2", tool="Sleep", input={}),
        ],
        CancelToken(),
    )
    assert _RAN == [], "a refused tool must never execute"
    for block, name in zip(result.blocks, ("Write", "Sleep"), strict=True):
        assert block.is_error
        text = cast(IRToolTextBlock, block.content[0]).text
        assert text.startswith(f"{name} is not available inside this fork.")
        assert "Read, Reflect, ReflectToolResults, Recall" in text or all(
            n in text for n in ("Read", "Reflect", "ReflectToolResults", "Recall")
        )


# ---- 3B. Explicit breakpoint at the end of the shared prefix -----------------------


def _marks(messages: list[Any]) -> list[tuple[int, int, str]]:
    found = []
    for mi, message in enumerate(messages):
        content = cast(list[dict[str, Any]], message["content"])
        for bi, block in enumerate(content):
            if "cache_control" in block:
                assert block["cache_control"] == SHARED_PREFIX_CACHE_CONTROL
                found.append((mi, bi, block["type"]))
    return found


def _history() -> list[IRBlock]:
    return [
        IRUserTextBlock(text="hello", origin="human"),
        IRThinkingBlock(text="hm", signature="sig", origin="model"),
        IRToolCallBlock(
            origin="model", call_id="c1", tool="Read", input={"file_path": "x"}
        ),
        IRToolResultBlock(
            call_id="c1", tool="Read", content=[IRToolTextBlock(text="data")]
        ),
        IRThinkingBlock(text="ok", signature="sig2", origin="model"),
        IRAssistantTextBlock(text="Here is what I found.", origin="model"),
    ]


class TestSharedPrefixBreakpoint:
    def test_marks_last_cacheable_block_before_boundary(self) -> None:
        dream = IRUserTextBlock(text="A dream has found you.", origin="system")
        blocks = [*_history(), dream]
        messages = _ir_blocks_to_message_params(blocks, cache_breakpoint_before=6)
        # user / assistant(thinking, tool_use) / user(tool_result) /
        # assistant(thinking, text) / user(dream): the parent's final answer.
        assert _marks(messages) == [(3, 1, "text")]
        dream_message = cast(list[dict[str, Any]], messages[-1]["content"])
        assert "cache_control" not in dream_message[-1]  # the dream

    def test_walks_back_past_thinking_and_empty_text(self) -> None:
        blocks = [
            IRUserTextBlock(text="q", origin="human"),
            IRToolCallBlock(origin="model", call_id="c1", tool="Read", input={}),
            IRToolResultBlock(
                call_id="c1", tool="Read", content=[IRToolTextBlock(text="r")]
            ),
            IRThinkingBlock(text="t", signature="s", origin="model"),
            IRAssistantTextBlock(text="", origin="model"),
            IRUserTextBlock(text="instruction", origin="system"),
        ]
        messages = _ir_blocks_to_message_params(blocks, cache_breakpoint_before=5)
        assert _marks(messages) == [(2, 0, "tool_result")]

    def test_no_boundary_no_marks(self) -> None:
        assert _marks(_ir_blocks_to_message_params(_history())) == []
        assert (
            _marks(_ir_blocks_to_message_params(_history(), cache_breakpoint_before=0))
            == []
        )


def _surface_builder(
    instruction: IRUserTextBlock | None, *, registry: ToolRegistry
) -> RequestSurfaceBuilder:
    config = SpellbookConfig(
        model="claude-opus-4-6", cwd=Path("/tmp"), system_prompt="me"
    )
    return RequestSurfaceBuilder.from_config(
        backend=AnthropicBackend(client=cast(Any, object())),
        config=config,
        tool_registry=registry,
        fork_config=(
            QuantumForkConfig(instruction=instruction)
            if instruction is not None
            else None
        ),
    )


def _wire(value: object) -> str:
    return json.dumps(value, sort_keys=False, ensure_ascii=False)


def test_sibling_forks_share_a_byte_identical_prefix_through_the_breakpoint() -> None:
    """The property the whole fix exists for: N forks → 1 cache write, N-1 reads."""
    parent_registry = ToolRegistry.build(None, surface="main", sleep_enabled=True)
    fork_registry = ToolRegistry.build(None, surface="quantum", sleep_enabled=True)
    history = _history()
    surfaces = []
    for pair in ("52-53", "54-55"):
        instruction = IRUserTextBlock(
            text=f"A dream has found you. Pair {pair}.", origin="system"
        )
        builder = _surface_builder(instruction, registry=fork_registry)
        # As recorded: the block is re-stamped, and a footer may follow it.
        recorded = instruction.model_copy(
            update={"turn_id": f"turn_{pair}", "event_id": pair}
        )
        footer = IRUserTextBlock(
            text=f"<spellbook>[{pair}]</spellbook>", origin="system"
        )
        surfaces.append(builder.build([*history, recorded, footer]))

    parent = _surface_builder(None, registry=parent_registry).build(history)
    first, second = surfaces

    # tools → system → messages, compared exactly as they go on the wire.
    assert _wire(first.tools) == _wire(second.tools) == _wire(parent.tools)
    assert first.system == second.system == parent.system
    assert first.thinking == second.thinking == parent.thinking
    assert first.output_config == second.output_config == parent.output_config
    (marked,) = _marks(first.messages)
    assert _marks(second.messages) == [marked]
    mi, bi, _ = marked

    def prefix(surface: Any) -> list[Any]:  # noqa: ANN401 - RequestSurface
        """Messages up to and including the marked block."""
        messages = surface.messages
        head = {**messages[mi], "content": messages[mi]["content"][: bi + 1]}
        return [*messages[:mi], head]

    def strip(messages: list[Any]) -> list[Any]:
        marker = ', "cache_control": ' + _wire(SHARED_PREFIX_CACHE_CONTROL)
        return json.loads(_wire(messages).replace(marker, ""))

    assert _wire(prefix(first)) == _wire(prefix(second))
    # The parent's own rendering of that history matches too (markers aside).
    assert strip(prefix(first)) == strip(parent.messages)
    # Automatic top-level caching still covers each fork's own tail.
    assert first.cache_control == {"type": "ephemeral", "ttl": "1h"}
    # At most 4 breakpoints per request: one explicit + one automatic.
    assert len(_marks(first.messages)) + 1 <= 4


def test_builder_passes_no_boundary_before_instruction_or_outside_forks() -> None:
    registry = ToolRegistry.build(None, surface="quantum")
    instruction = IRUserTextBlock(text="dream", origin="system")
    before = _surface_builder(instruction, registry=registry).build(_history())
    assert _marks(before.messages) == []

    class _LegacyBackend:
        """A backend that predates cache_breakpoint_before."""

        def build_request_surface(self, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
            assert "cache_breakpoint_before" not in kwargs
            return kwargs

        def build_tool_schemas(self, registry: ToolRegistry) -> list[dict[str, Any]]:
            return []

    builder = RequestSurfaceBuilder.from_config(
        backend=cast(Any, _LegacyBackend()),
        config=SpellbookConfig(cwd=Path("/tmp")),
        tool_registry=registry,
    )
    builder.build([*_history(), instruction])


class TestForkUsageReceipts:
    """Night-003 pre-flight: the receipts get kept and reported."""

    def test_turn_end_usage_sums_across_generations(self) -> None:
        from spellbook.ir_types import IRGeneration, IRLoopResult, IRUsage

        gens = [
            IRGeneration(
                model="m",
                blocks=[],
                stop_reason="tool_use",
                usage=IRUsage(
                    input_tokens=10,
                    output_tokens=5,
                    cache_read_tokens=100,
                    cache_create_tokens=7,
                ),
            ),
            IRGeneration(model="m", blocks=[], stop_reason="end_turn", usage=None),
            IRGeneration(
                model="m",
                blocks=[],
                stop_reason="end_turn",
                usage=IRUsage(input_tokens=1, output_tokens=2, cache_read_tokens=50),
            ),
        ]
        result = IRLoopResult(
            blocks=[], generations=gens, executions=[], stop_reason="end_turn", rounds=3
        )
        total = result.total_usage
        assert total is not None
        assert (total.input_tokens, total.output_tokens) == (11, 7)
        assert (total.cache_read_tokens, total.cache_create_tokens) == (150, 7)

    def test_fork_usage_scan_sums_only_usage_bearing_turn_ends(
        self, tmp_path: Path
    ) -> None:
        from spellbook.dreaming.pipeline import _fork_usage_from_transcript

        t = tmp_path / "transcript.jsonl"
        t.write_text(
            '{"ir":"turn_end","session_id":"s","turn":1,"turn_id":"a"}\n'
            '{"ir":"turn_end","session_id":"s","turn":2,"turn_id":"b",'
            '"usage":{"input_tokens":3,"output_tokens":4,'
            '"cache_read_tokens":1000,"cache_create_tokens":20}}\n'
            '{"ir":"event","usage":{"input_tokens":999}}\n'
            '{"ir":"turn_end","session_id":"s","turn":3,"turn_id":"c",'
            '"usage":{"input_tokens":2,"output_tokens":1,'
            '"cache_read_tokens":500,"cache_create_tokens":0}}\n'
        )
        usage = _fork_usage_from_transcript(t)
        assert usage is not None
        assert usage.input_tokens == 5
        assert usage.cache_read_tokens == 1500
        assert usage.cache_create_tokens == 20

    def test_old_transcripts_rehydrate_without_usage(self, tmp_path: Path) -> None:
        from spellbook.ir_types import IRTurnEndRecord

        record = IRTurnEndRecord.model_validate(
            {"session_id": "s", "ir": "turn_end", "turn": 1, "turn_id": "t"}
        )
        assert record.usage is None

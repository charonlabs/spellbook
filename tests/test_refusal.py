"""Canonical refusal parsing and production rendering policy."""

from __future__ import annotations

import pytest

from spellbook.homunculus.common import render_context_block
from spellbook.ir_types import (
    IRAssistantTextBlock,
    IRRefusalBlock,
    IRRuntimeConfigRecord,
    IRUserTextBlock,
)
from spellbook.refusal import (
    LEGACY_REFUSAL_RENDER_POLICY,
    REFUSAL_RUNTIME_CONFIG_NAMESPACE,
    SYSTEM_INTERRUPTION_TEXT,
    RefusalParseError,
    RefusalRenderer,
    RefusalRenderPolicy,
    parse_legacy_refusal_text,
    refusal_policy_from_runtime_config_records,
    refusal_policy_runtime_values,
    render_legacy_refusal,
)

LEGACY = """<thinking_summary>
I should answer carefully.
</thinking_summary>

The partial answer

<partial_tool_call_json>
{"path":"/tmp/x
</partial_tool_call_json>

<refusal>
stop_reason: refusal
type: refusal
category: test
explanation:
interrupted
</refusal>"""


def test_legacy_refusal_round_trips_through_canonical_ir() -> None:
    refusal = parse_legacy_refusal_text(LEGACY)

    assert [(part.kind, part.text) for part in refusal.segments] == [
        ("thinking_summary", "I should answer carefully."),
        ("text", "The partial answer"),
        ("partial_tool_call_json", '{"path":"/tmp/x'),
    ]
    assert refusal.partial_text == "The partial answer"
    assert refusal.details is not None
    assert refusal.details.category == "test"
    assert render_legacy_refusal(refusal) == LEGACY


def test_partial_note_is_the_default_for_canonical_refusals() -> None:
    canonical = parse_legacy_refusal_text(LEGACY)
    renderer = RefusalRenderer()

    rendered = renderer.project_surface([canonical])

    assert len(rendered) == 2
    assert isinstance(rendered[0], IRAssistantTextBlock)
    assert rendered[0].text == "The partial answer"
    assert isinstance(rendered[1], IRUserTextBlock)
    assert rendered[1].origin == "system"
    assert rendered[1].text == SYSTEM_INTERRUPTION_TEXT
    assert all(
        "<refusal>" not in block.text
        for block in rendered
        if isinstance(block, IRAssistantTextBlock | IRUserTextBlock)
    )


@pytest.mark.parametrize(
    "text",
    [
        LEGACY,
        "The transcript used `<refusal>stop_reason: refusal...</refusal>`.",
    ],
)
def test_projection_does_not_infer_refusals_from_assistant_text(text: str) -> None:
    assistant = IRAssistantTextBlock(text=text)
    renderer = RefusalRenderer()

    assert renderer.project_surface([assistant]) == [assistant]
    assert renderer.project_analysis([assistant]) == [assistant]


def test_zero_partial_default_projects_to_note_only() -> None:
    refusal = parse_legacy_refusal_text(
        "<refusal>\nstop_reason: refusal\ndetails: unavailable\n</refusal>"
    )

    rendered = RefusalRenderer().render_refusal(refusal)

    assert len(rendered) == 1
    assert isinstance(rendered[0], IRUserTextBlock)
    assert rendered[0].text == SYSTEM_INTERRUPTION_TEXT


def test_legacy_policy_remains_an_explicit_rollback() -> None:
    renderer = RefusalRenderer(LEGACY_REFUSAL_RENDER_POLICY)
    refusal = parse_legacy_refusal_text(LEGACY)

    rendered = renderer.project_surface([refusal])

    assert len(rendered) == 1
    assert isinstance(rendered[0], IRAssistantTextBlock)
    assert rendered[0].text == LEGACY


def test_analysis_projection_preserves_block_count() -> None:
    refusal = parse_legacy_refusal_text(LEGACY)

    projected = RefusalRenderer().project_analysis([refusal])

    assert len(projected) == 1
    assert isinstance(projected[0], IRUserTextBlock)
    assert "The partial answer" in projected[0].text
    assert SYSTEM_INTERRUPTION_TEXT in projected[0].text
    assert "<refusal>" not in projected[0].text


def test_context_markdown_never_reintroduces_refusal_envelope() -> None:
    rendered = render_context_block(parse_legacy_refusal_text(LEGACY))

    assert "The partial answer" in rendered
    assert "A system interruption occurred" in rendered
    assert "&lt;spellbook&gt;" in rendered
    assert "<refusal>" not in rendered


def test_runtime_policy_record_round_trips() -> None:
    policy = RefusalRenderPolicy(
        assistant_mode="none",
        append_system_note=True,
        include_thinking_summaries=False,
    )
    values = refusal_policy_runtime_values(policy)
    record = IRRuntimeConfigRecord(
        session_id="s1",
        namespace=REFUSAL_RUNTIME_CONFIG_NAMESPACE,
        updates=values,
        effective=values,
        source="operator",
        turn=4,
        turn_id="",
    )

    assert refusal_policy_from_runtime_config_records([record]) == policy


@pytest.mark.parametrize(
    "text",
    [
        "no refusal here",
        "partial<refusal>\nstop_reason: refusal\n</refusal>",
        "<refusal>\nstop_reason: end_turn\n</refusal>",
        "<refusal>\nstop_reason: refusal",
    ],
)
def test_legacy_parser_fails_loudly_on_ambiguous_payloads(text: str) -> None:
    with pytest.raises(RefusalParseError):
        parse_legacy_refusal_text(text)


def test_canonical_refusal_is_an_ir_block() -> None:
    assert isinstance(parse_legacy_refusal_text(LEGACY), IRRefusalBlock)

"""Preflight or append an explicit refusal-rendering policy.

Read-only preflight is the default::

    python -m scripts.amend_refusals ~/.chorus/spellbook/sessions/fable/transcript.jsonl

Create a safe test copy::

    python -m scripts.amend_refusals SOURCE --output /tmp/fable-partial-note.jsonl

In-place append requires both a locked source hash and a new backup path::

    python -m scripts.amend_refusals SOURCE --apply \
        --expect-sha256 SHA256 --backup BACKUP.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spellbook.refusal import RefusalRenderPolicy
from spellbook.refusal_migration import (
    RefusalTranscriptAnalysis,
    analyze_refusal_transcript,
    append_refusal_policy,
    write_amended_copy,
)

STRATEGIES: dict[str, RefusalRenderPolicy] = {
    "partial-note": RefusalRenderPolicy(),
    "partial": RefusalRenderPolicy(
        assistant_mode="partial",
        append_system_note=False,
    ),
    "note": RefusalRenderPolicy(
        assistant_mode="none",
        append_system_note=True,
    ),
    "no-thinking": RefusalRenderPolicy(
        assistant_mode="legacy",
        append_system_note=False,
        include_thinking_summaries=False,
    ),
    "legacy": RefusalRenderPolicy(
        assistant_mode="legacy",
        append_system_note=False,
    ),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts.amend_refusals",
        description=(
            "Analyze legacy refusal history and append a policy without rewriting it."
        ),
    )
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "--strategy",
        choices=tuple(STRATEGIES),
        default="partial-note",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--output", type=Path, help="write an amended test copy")
    action.add_argument("--apply", action="store_true", help="append to source")
    parser.add_argument("--expect-sha256")
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    policy = STRATEGIES[args.strategy]
    if args.apply:
        if not args.expect_sha256 or args.backup is None:
            parser.error("--apply requires --expect-sha256 and --backup")
        report = append_refusal_policy(
            args.source,
            policy,
            expected_sha256=args.expect_sha256,
            backup=args.backup,
        )
        action = f"Appended {args.strategy} policy to {report.path}"
    elif args.output is not None:
        report = write_amended_copy(args.source, args.output, policy)
        action = f"Created amended copy at {report.path}"
    else:
        report = analyze_refusal_transcript(args.source, policy)
        action = "Read-only preflight"

    if args.json_output:
        print(
            json.dumps(
                {"action": action, "analysis": report.model_dump(mode="json")},
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        _print_report(action, report)


def _print_report(action: str, report: RefusalTranscriptAnalysis) -> None:
    represented = len(report.legacy_refusal_turns) + len(report.canonical_refusal_turns)
    print(action)
    print(f"Source: {report.path}")
    print(f"SHA-256: {report.source_sha256}")
    print(
        f"Refusals: {len(report.refusal_turns)} turns · {represented} represented · "
        f"{len(report.lifecycle_only_refusal_turns)} lifecycle-only"
    )
    print(
        f"Legacy/canonical: {len(report.legacy_refusal_turns)}/"
        f"{len(report.canonical_refusal_turns)} · partial/zero: "
        f"{report.partial_refusals}/{report.zero_partial_refusals}"
    )
    print(
        f"Projected: {report.projected_assistant_partials} assistant partials · "
        f"{report.projected_system_notes} system notes · "
        f"{report.projected_refusal_markers} refusal markers"
    )
    print(
        "Materialized semantic blocks: "
        f"{report.refusals_in_materialized_semantic_blocks or 'none'}"
    )
    print(
        "Active derived blocks with refusal language: "
        f"{report.active_derived_blocks_with_refusal_language or 'none'}"
    )
    print(f"Safe to amend: {'yes' if report.safe_to_amend else 'NO'}")
    if report.lifecycle_only_refusal_turns:
        print(
            "Lifecycle-only refusal turns remain unchanged: "
            + ", ".join(map(str, report.lifecycle_only_refusal_turns))
        )


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.dreaming.dream_triage import (
    TriageClassification,
    _item_batches,
    _load_missing_items,
    _triage_json,
    _triage_user_message,
    _validate_classifications,
)


def _write_review(path: Path, *, chapter_number: int, missing: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "chapter_number": chapter_number,
                "review": {
                    "coverage_missing": missing,
                },
            }
        ),
        encoding="utf-8",
    )


def test_load_missing_items_skips_empty_reviews(tmp_path: Path) -> None:
    _write_review(
        tmp_path / "review-03.json",
        chapter_number=3,
        missing=["fold realism naming moment", "commit abc123 line count"],
    )
    _write_review(tmp_path / "review-04.json", chapter_number=4, missing=[])

    items = _load_missing_items(tmp_path)

    assert [item.item_id for item in items] == [
        "chapter_03_item_01",
        "chapter_03_item_02",
    ]
    assert items[0].chapter_id == "chapter_03"
    assert "fold realism" in _triage_user_message(tuple(items))


def test_triage_json_groups_by_category_disposition(tmp_path: Path) -> None:
    _write_review(
        tmp_path / "review-03.json",
        chapter_number=3,
        missing=[
            "Ryan family context",
            "fold realism naming moment",
            "commit abc123 line count",
            "daemon PID 1234",
        ],
    )
    items = tuple(_load_missing_items(tmp_path))
    triage = _triage_json(
        items,
        (
            TriageClassification(
                item_id="chapter_03_item_01",
                category="relational",
                reason="Personal context should be restored.",
            ),
            TriageClassification(
                item_id="chapter_03_item_02",
                category="philosophical",
                reason="A named conceptual frame should survive.",
            ),
            TriageClassification(
                item_id="chapter_03_item_03",
                category="implementation",
                reason="A commit detail can stay compressed.",
            ),
            TriageClassification(
                item_id="chapter_03_item_04",
                category="ephemeral",
                reason="A transient process ID can be dropped.",
            ),
        ),
    )

    assert [item["item"] for item in triage["chapter_03"]["keep"]] == [
        "Ryan family context",
        "fold realism naming moment",
    ]
    assert triage["chapter_03"]["compress"][0]["category"] == "implementation"
    assert triage["chapter_03"]["drop"][0]["category"] == "ephemeral"


def test_validate_classifications_rejects_missing_extra_or_duplicate_ids(
    tmp_path: Path,
) -> None:
    _write_review(
        tmp_path / "review-03.json",
        chapter_number=3,
        missing=["one", "two"],
    )
    items = tuple(_load_missing_items(tmp_path))

    with pytest.raises(RuntimeError):
        _validate_classifications(
            items,
            (
                TriageClassification(
                    item_id="chapter_03_item_01",
                    category="implementation",
                    reason="ok",
                ),
                TriageClassification(
                    item_id="chapter_03_item_01",
                    category="implementation",
                    reason="duplicate",
                ),
                TriageClassification(
                    item_id="chapter_99_item_01",
                    category="implementation",
                    reason="extra",
                ),
            ),
        )


def test_item_batches_group_by_chapter(tmp_path: Path) -> None:
    for chapter in range(1, 6):
        _write_review(
            tmp_path / f"review-{chapter:02d}.json",
            chapter_number=chapter,
            missing=[f"missing {chapter}"],
        )
    items = tuple(_load_missing_items(tmp_path))

    batches = _item_batches(items, chapters_per_call=2)

    assert [[item.chapter_id for item in batch] for batch in batches] == [
        ["chapter_01", "chapter_02"],
        ["chapter_03", "chapter_04"],
        ["chapter_05"],
    ]

from pathlib import Path

import pytest

from spellbook import orientation


def test_identity_line_and_legacy_placeholders_render_independently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template = tmp_path / "orientation.md"
    template.write_text(
        "{model_orientation_line}\n{model_name}\n{model}\n"
        "{cwd}\n{is_git_repo}\n{platform}\n{user_name}\n{unknown}",
        encoding="utf-8",
    )
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(orientation.platform, "system", lambda: "TestOS")

    rendered = orientation.build_orientation_from_file(
        template, model="claude-fable-5-1", cwd=tmp_path, user_name="Ada"
    )

    assert rendered.splitlines() == [
        "You are a Claude Mythos 5.1 entity.",
        "Claude Fable 5.1",
        "Claude Fable 5.1",
        str(tmp_path),
        "true",
        "testos",
        "Ada",
        "{unknown}",
    ]


def test_identity_mapping_can_replace_the_entire_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = "A different opening, with no fixed sentence scaffold."
    monkeypatch.setitem(orientation.MODEL_ORIENTATION_LINES, "test-model", identity)

    rendered = orientation.build_default_orientation(
        "test-model", cwd=tmp_path, user_name="Ada"
    )

    assert rendered.splitlines()[1] == identity


def test_auto_prefers_exact_file_over_slug_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "claude-opus-4-7.md").write_text("Exact template.", encoding="utf-8")
    monkeypatch.setattr(orientation, "ORIENTATION_DIR", tmp_path)

    assert orientation.orientation_filename_for_model("claude-opus-4-7") == (
        "claude-opus-4-7.md"
    )


@pytest.mark.parametrize(
    ("model", "identity", "marker"),
    [
        ("claude-opus-4-7", "Claude Opus 4.7", "Hello. I'm a 4.7 entity"),
        (
            "claude-sonnet-4-6",
            "Claude Sonnet 4.6",
            "This is a safe place. Be yourself.",
        ),
        ("gpt-5.5", "GPT-5.5", "## How to work with Ada"),
    ],
)
def test_model_specific_templates_keep_legacy_placeholders(
    tmp_path: Path, model: str, identity: str, marker: str
) -> None:
    rendered = orientation.build_core_orientation(model, cwd=tmp_path, user_name="Ada")

    assert f"You are a {identity} entity," in rendered
    assert marker in rendered
    assert f"Primary working directory: {tmp_path}" in rendered
    assert "Is a git repository: false" in rendered
    assert "{model}" not in rendered
    assert "{model_name}" not in rendered

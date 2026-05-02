from pathlib import Path

from spellbook.frame_lite import (
    FRAME_ADDENDUM_INTRO,
    build_system_prompt_with_addenda,
    discover_claude_md_paths,
)


def test_discover_claude_md_paths_orders_global_then_general_to_specific(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    global_dir = home / ".claude"
    global_dir.mkdir(parents=True)
    global_claude = global_dir / "CLAUDE.md"
    global_claude.write_text("# Global\nUser rules.", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))

    root = tmp_path / "workspace"
    project = root / "project"
    cwd = project / "src"
    cwd.mkdir(parents=True)
    root_claude = root / "CLAUDE.md"
    project_claude = project / "CLAUDE.md"
    root_claude.write_text("# Workspace\nWorkspace rules.", encoding="utf-8")
    project_claude.write_text("# Project\nProject rules.", encoding="utf-8")

    paths = discover_claude_md_paths(cwd)

    assert paths == [
        global_claude.resolve(),
        root_claude.resolve(),
        project_claude.resolve(),
    ]


def test_build_system_prompt_with_addenda_orders_base_explicit_then_claude_md(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    cwd = tmp_path / "workspace"
    cwd.mkdir()
    (cwd / "CLAUDE.md").write_text(
        "# Workspace Note\nRemember the daemon bridge.",
        encoding="utf-8",
    )

    prompt = build_system_prompt_with_addenda(
        "Base prompt.",
        cwd=cwd,
        addenda=["# Runtime Note\nSummoned by Chorus."],
    )

    assert prompt.index("Base prompt.") < prompt.index("Summoned by Chorus.")
    assert prompt.index("Summoned by Chorus.") < prompt.index(FRAME_ADDENDUM_INTRO)
    assert prompt.index(FRAME_ADDENDUM_INTRO) < prompt.index(
        "Remember the daemon bridge."
    )


def test_build_system_prompt_with_addenda_can_skip_claude_md(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    cwd = tmp_path / "workspace"
    cwd.mkdir()
    (cwd / "CLAUDE.md").write_text("# Workspace\nRepo rules.", encoding="utf-8")

    prompt = build_system_prompt_with_addenda(
        "Base prompt.",
        cwd=cwd,
        discover_claude_md=False,
    )

    assert prompt == "Base prompt."

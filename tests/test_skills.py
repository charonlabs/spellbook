from __future__ import annotations

from pathlib import Path

from spellbook.config import SpellbookConfig
from spellbook.skills.manager import SkillManager


def _config(tmp_path: Path) -> SpellbookConfig:
    return SpellbookConfig(
        model="claude-sonnet-4-6",
        cwd=tmp_path,
        skill_discovery_dirs=[".test-skills"],
    )


def _write_skill(
    root: Path,
    *,
    skill_dir: str,
    name: str,
    description: str,
    body: str = "Follow the specialized workflow.",
) -> Path:
    skill_path = root / ".test-skills" / "skills" / skill_dir / "SKILL.md"
    skill_path.parent.mkdir(parents=True, exist_ok=True)
    skill_path.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n"
    )
    return skill_path


def _manager(tmp_path: Path, monkeypatch) -> SkillManager:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    return SkillManager(config=_config(tmp_path))


class TestSkillManagerRefresh:
    def test_refresh_returns_none_when_catalog_is_unchanged(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        _write_skill(
            tmp_path,
            skill_dir="compose",
            name="compose",
            description="Draft careful prose.",
        )
        manager = _manager(tmp_path, monkeypatch)
        manager.discover_skills()

        assert manager.refresh() is None

    def test_refresh_reports_added_skill_without_readding_unchanged_skill(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        _write_skill(
            tmp_path,
            skill_dir="compose",
            name="compose",
            description="Draft careful prose.",
        )
        manager = _manager(tmp_path, monkeypatch)
        manager.discover_skills()

        _write_skill(
            tmp_path,
            skill_dir="review",
            name="review",
            description="Review code changes.",
        )
        delta = manager.refresh()

        assert delta is not None
        assert set(delta.added) == {"review"}
        assert delta.updated == {}
        assert delta.removed == []
        assert manager.catalog is not None
        assert set(manager.catalog.skills) == {"compose", "review"}

    def test_refresh_reports_only_changed_skill_as_updated(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        _write_skill(
            tmp_path,
            skill_dir="compose",
            name="compose",
            description="Draft careful prose.",
        )
        _write_skill(
            tmp_path,
            skill_dir="review",
            name="review",
            description="Review code changes.",
        )
        manager = _manager(tmp_path, monkeypatch)
        manager.discover_skills()

        _write_skill(
            tmp_path,
            skill_dir="compose",
            name="compose",
            description="Draft extra careful prose.",
        )
        delta = manager.refresh()

        assert delta is not None
        assert delta.added == {}
        assert set(delta.updated) == {"compose"}
        assert delta.updated["compose"].description == "Draft extra careful prose."
        assert delta.removed == []

    def test_refresh_reports_removed_skill(self, tmp_path: Path, monkeypatch) -> None:
        removed_path = _write_skill(
            tmp_path,
            skill_dir="compose",
            name="compose",
            description="Draft careful prose.",
        )
        _write_skill(
            tmp_path,
            skill_dir="review",
            name="review",
            description="Review code changes.",
        )
        manager = _manager(tmp_path, monkeypatch)
        manager.discover_skills()

        removed_path.unlink()
        delta = manager.refresh()

        assert delta is not None
        assert delta.added == {}
        assert delta.updated == {}
        assert delta.removed == ["compose"]
        assert manager.catalog is not None
        assert set(manager.catalog.skills) == {"review"}

from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from spellbook.config import SpellbookConfig
from spellbook.ir_types import (
    IRSkill,
    IRSkillCatalog,
    IRSkillCatalogDelta,
    SkillScope,
)
from spellbook.round_lifecycle import RoundContext, RoundLifecycle
from spellbook.skills.parsing import parse_skill, strip_frontmatter

if TYPE_CHECKING:
    from spellbook.footer import FooterController
    from spellbook.recorder import Recorder
    from spellbook.rehydrator import RehydrationResult


class SkillManager:
    RESOURCES_MAX_FILES_DISPLAY = 50

    def __init__(self, config: SpellbookConfig):
        self._config = config
        self.catalog: IRSkillCatalog | None = None

    @property
    def num_skills(self) -> int:
        if self.catalog is None:
            return 0
        return len(self.catalog.skills)

    def rehydrate(self, rehydrated: "RehydrationResult") -> None:
        self.catalog = rehydrated.skill_catalog

    def invoke(self, name: str, *, args: str | None = None) -> str:
        """Invoke a skill from the registry. Raises if the skill does not exist."""
        if self.catalog is None:
            raise ValueError(
                (
                    "Skills were not initialized for this session. "
                    "Skill support is not currently functional."
                )
            )
        assert self.catalog is not None
        skill = self.catalog.skills.get(name)
        if skill is None:
            available = self.catalog.skills.keys()
            if available:
                raise ValueError(
                    f"Unknown skill: {name}. Available: {', '.join(available)}"
                )
            raise ValueError(f"Unknown skill: {name}. No skills are available.")

        try:
            body = strip_frontmatter(skill.location.read_text())
        except (OSError, PermissionError) as e:
            raise ValueError(f"Failed to read skill: {e}") from e

        resources = []
        for path in sorted(skill.directory.rglob("*")):
            if path.is_file() and path.name != "SKILL.md":
                try:
                    rel = str(path.relative_to(skill.directory))
                    resources.append(rel)
                except ValueError:
                    continue
                if len(resources) >= self.RESOURCES_MAX_FILES_DISPLAY:
                    break

        parts = [f'<skill-content name="{skill.name}">']
        parts.append(body)
        parts.append("")
        parts.append(f"Skill directory: {skill.directory}")
        parts.append(
            "Relative paths in this skill are relative to the skill directory."
        )

        if resources:
            parts.append("\n<skill-resources>")
            for r in resources:
                parts.append(f"  <file>{r}</file>")
            parts.append("</skill-resources>")

        parts.append("</skill-content>")
        return "\n".join(parts)

    def discover_skills(self) -> IRSkillCatalog:
        self.catalog = self._scan()
        return self.catalog

    def refresh(self) -> IRSkillCatalogDelta | None:
        if self.catalog is None:
            raise ValueError(
                "Call `discover_skills` first. `refresh` is for runtime-scanning."
            )
        new_catalog = self._scan()
        old_by_name = self.catalog.skills
        new_by_name = new_catalog.skills
        if old_by_name == new_by_name:
            return
        old_names = set(old_by_name)
        new_names = set(new_by_name)
        added = {
            name: new_by_name[name] for name in sorted(new_names.difference(old_names))
        }
        updated = {
            name: new_by_name[name]
            for name in sorted(old_names.intersection(new_names))
            if old_by_name[name] != new_by_name[name]
        }
        removed = sorted(old_names.difference(new_names))
        self.catalog = new_catalog
        return IRSkillCatalogDelta(added=added, updated=updated, removed=removed)

    def render_prompt_addendum(self) -> str:
        if self.catalog is None:
            raise ValueError("Must discover skills before you can render the addendum.")
        lines = ["<available-skills>"]
        for skill in self.skills_list:
            lines.append(self.render_skill(skill))
        lines.append("</available-skills>")
        return "\n".join(lines)

    @property
    def skills_list(self) -> list[IRSkill]:
        if self.catalog is None:
            return []
        return self._catalog_to_sorted_list(self.catalog)

    def _catalog_to_sorted_list(self, catalog: IRSkillCatalog) -> list[IRSkill]:
        return [
            skill for skill in sorted(catalog.skills.values(), key=lambda s: s.name)
        ]

    def render_skill(self, skill: IRSkill) -> str:
        lines = ["<skill>"]
        lines.append(f"  <name>{skill.name}</name>")
        lines.append(f"  <description>{skill.description}</description>")
        lines.append("</skill>")
        return "\n".join(lines)

    def _scan(self) -> IRSkillCatalog:
        home = Path.home()
        cwd = self._config.cwd
        skills: dict[str, IRSkill] = {}

        # Scan directories in order: project first (higher precedence), then user
        scan_dirs: list[tuple[Path, SkillScope]] = [
            ((cwd / p / "skills"), "project") for p in self._config.skill_discovery_dirs
        ] + [((home / p / "skills"), "user") for p in self._config.skill_discovery_dirs]

        for base, scope in scan_dirs:
            if not base.is_dir():
                continue

            for skill_dir in sorted(base.iterdir()):
                if not skill_dir.is_dir() or skill_dir.name.startswith("."):
                    continue

                skill_md = skill_dir / "SKILL.md"
                if not skill_md.is_file():
                    continue

                skill = parse_skill(skill_md, scope=scope)
                if skill is None:
                    continue

                if skill.name not in skills:
                    skills[skill.name] = skill
        catalog = IRSkillCatalog(skills=skills)
        return catalog


class SkillManagerRoundLifecycle(RoundLifecycle):
    def __init__(
        self, footer_c: "FooterController", manager: SkillManager, recorder: "Recorder"
    ):
        self._footer_c = footer_c
        self._manager = manager
        self._recorder = recorder

    async def before_round(self, ctx: RoundContext) -> None:
        maybe_delta = self._manager.refresh()
        if maybe_delta is None:
            return
        self._recorder.update_skill_catalog(maybe_delta)
        lines = ["Skill catalog updated:"]
        for new_skill in maybe_delta.added.values():
            lines.append(f"- New skill:\n{self._manager.render_skill(new_skill)}")
        for updated_skill in maybe_delta.updated.values():
            lines.append(
                f"- Updated skill:\n{self._manager.render_skill(updated_skill)}"
            )
        for removed_name in maybe_delta.removed:
            lines.append(f'- Removed: "{removed_name}"')
        self._footer_c.queue_footer(
            text="\n".join(lines),
            footer_type="skill_catalog_update",
            source="skill_manager",
            key=f"skill_delta_{uuid4().hex}",
        )

"""Lightweight runtime frame helpers for the core rewrite.

This module intentionally stops short of the full structured frame system. It
only handles the runtime behavior core needs today: discover CLAUDE.md files
and append their contents to a base system prompt in a predictable order.
"""

from collections.abc import Iterable
from pathlib import Path

FRAME_ADDENDUM_INTRO = (
    "The following additional frame sources are part of your always-on context. "
    "Treat them as active session instructions and working memory."
)


def discover_claude_md_paths(cwd: Path) -> list[Path]:
    """Discover CLAUDE.md files from global user scope and cwd ancestry.

    Order is most general to most specific:
    `~/.claude/CLAUDE.md`, then any `CLAUDE.md` files found while walking from
    filesystem root down toward `cwd`.
    """

    discovered: list[Path] = []
    seen: set[Path] = set()

    current = cwd.expanduser().resolve()
    if current.is_file():
        current = current.parent

    ancestors: list[Path] = []
    while True:
        candidate = current / "CLAUDE.md"
        if candidate.is_file():
            resolved = candidate.resolve()
            if resolved not in seen:
                ancestors.append(resolved)
                seen.add(resolved)
        parent = current.parent
        if parent == current:
            break
        current = parent

    global_claude = (Path.home() / ".claude" / "CLAUDE.md").expanduser().resolve()
    if global_claude.is_file() and global_claude not in seen:
        discovered.append(global_claude)
        seen.add(global_claude)

    discovered.extend(reversed(ancestors))
    return discovered


def build_system_prompt_with_addenda(
    base_prompt: str,
    *,
    cwd: Path,
    addenda: Iterable[str] = (),
    discover_claude_md: bool = True,
) -> str:
    """Append explicit addenda and discovered CLAUDE.md content to a prompt."""

    parts = [base_prompt.rstrip()]
    parts.extend(addendum.strip() for addendum in addenda if addendum.strip())

    if discover_claude_md:
        claude_addendum = render_claude_md_addendum(discover_claude_md_paths(cwd))
        if claude_addendum:
            parts.append(claude_addendum)

    return "\n\n".join(part for part in parts if part).strip()


def render_claude_md_addendum(paths: Iterable[Path]) -> str:
    """Render discovered CLAUDE.md files as supplemental prompt sections."""

    sections: list[str] = []
    for path in paths:
        content = path.expanduser().read_text(encoding="utf-8").strip()
        if not content:
            continue
        title = _first_heading(content) or path.name
        sections.append(f"## {title}\n\n{content}")

    if not sections:
        return ""
    return "\n\n".join([FRAME_ADDENDUM_INTRO, *sections])


def _first_heading(markdown: str) -> str | None:
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        title = stripped.lstrip("#").strip()
        if title:
            return title
    return None

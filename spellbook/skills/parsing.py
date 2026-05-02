import re
from pathlib import Path
from typing import Any

from spellbook.ir_types import IRSkill, SkillScope


def parse_skill(
    path: Path,
    *,
    scope: SkillScope,
) -> IRSkill | None:
    """Parse a SKILL.md file and extract frontmatter metadata."""
    try:
        raw = path.read_text()
    except (OSError, PermissionError):
        return None

    frontmatter = _parse_frontmatter(raw)
    if frontmatter is None:
        return None

    name = frontmatter.get("name")
    description = frontmatter.get("description")

    if not isinstance(name, str) or not name.strip():
        return None
    if not isinstance(description, str) or not description.strip():
        return None

    return IRSkill(
        name=name.strip(),
        description=description.strip(),
        location=path.resolve(),
        directory=path.parent.resolve(),
        scope=scope,
    )


def strip_frontmatter(raw: str) -> str:
    """Remove YAML frontmatter from a SKILL.md file, returning the body."""
    raw = raw.strip()
    if not raw.startswith("---"):
        return raw

    end = raw.find("\n---", 3)
    if end < 0:
        return raw

    return raw[end + 4 :].strip()


def _parse_frontmatter(raw: str) -> dict[str, Any] | None:
    """Extract YAML frontmatter from a SKILL.md file.

    Uses a simple parser rather than a YAML library to avoid
    the dependency and handle the common malformed cases
    (unquoted colons, etc.) gracefully.
    """
    raw = raw.strip()
    if not raw.startswith("---"):
        return None

    # Find closing ---
    end = raw.find("\n---", 3)
    if end < 0:
        return None

    yaml_block = raw[3:end].strip()
    if not yaml_block:
        return None

    result: dict[str, Any] = {}
    current_key: str | None = None
    current_value_lines: list[str] = []

    def flush() -> None:
        nonlocal current_key, current_value_lines
        if current_key is not None:
            value = " ".join(line.strip() for line in current_value_lines).strip()
            result[current_key] = value
        current_key = None
        current_value_lines = []

    for line in yaml_block.split("\n"):
        # Check if this is a new key: value pair
        match = re.match(r"^(\w[\w-]*)\s*:\s*(.*)", line)
        if match:
            flush()
            current_key = match.group(1)
            value = match.group(2).strip()
            # Handle YAML block scalar indicators
            if value in (">", "|", ">-", "|-"):
                current_value_lines = []
            else:
                current_value_lines = [value] if value else []
        elif current_key is not None:
            # Continuation line (indented)
            current_value_lines.append(line)

    flush()
    return result if result else None

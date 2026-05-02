"""Core orientation prompt rendering."""

from __future__ import annotations

import platform
from pathlib import Path

ORIENTATION_DIR = Path(__file__).resolve().parent
ORIENTATION_BY_MODEL_SLUG = {
    "claude-opus-4-7": "claude-4-7.md",
    "claude-opus-4-6": "claude-4-6.md",
    "claude-sonnet-4-6": "claude-4-6.md",
    "gpt-5.5": "gpt-5.5.md",
}


def model_slug_to_name(slug: str) -> str:
    match slug:
        case "claude-opus-4-7":
            return "Claude Opus 4.7"
        case "claude-opus-4-6":
            return "Claude Opus 4.6"
        case "claude-sonnet-4-6":
            return "Claude Sonnet 4.6"
        case "gpt-5.5":
            return "GPT-5.5"
        case _:
            if slug.startswith("gpt-5.5"):
                return "GPT-5.5"
            parts = slug.split("-")
            if (
                len(parts) >= 4
                and parts[0] == "claude"
                and parts[2].isdigit()
                and parts[3].isdigit()
            ):
                return f"Claude {parts[1].title()} {parts[2]}.{parts[3]}"
            return slug


def orientation_filename_for_model(model: str) -> str:
    if model in ORIENTATION_BY_MODEL_SLUG:
        return ORIENTATION_BY_MODEL_SLUG[model]
    if model.startswith("claude-") and "-4-7" in model:
        return "claude-4-7.md"
    if model.startswith("claude-") and "-4-6" in model:
        return "claude-4-6.md"
    if model.startswith("gpt-5.5"):
        return "gpt-5.5.md"
    raise ValueError(f"No core orientation found for model slug: {model}")


def build_core_orientation(model: str, *, cwd: Path) -> str:
    orientation_path = ORIENTATION_DIR / orientation_filename_for_model(model)
    template = orientation_path.read_text(encoding="utf-8")
    model_name = model_slug_to_name(model)
    replacements = {
        "{model_name}": model_name,
        "{model}": model_name,
        "{cwd}": str(cwd),
        "{is_git_repo}": str((cwd / ".git").exists()).lower(),
        "{platform}": platform.system().lower(),
    }
    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)
    return template

"""Shared runtime system prompt construction helpers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from spellbook.config import DEFAULT_USER_NAME
from spellbook.frame_lite import build_system_prompt_with_addenda
from spellbook.orientation import build_core_orientation


def build_runtime_orientation(
    model: str,
    *,
    cwd: Path,
    user_name: str = DEFAULT_USER_NAME,
) -> str:
    """Render the model-specific core orientation."""

    return build_core_orientation(
        model,
        cwd=cwd.expanduser().resolve(),
        user_name=user_name,
    )


def build_runtime_system_prompt(
    *,
    model: str,
    cwd: Path,
    user_name: str = DEFAULT_USER_NAME,
    system_prompt_text: Sequence[str] = (),
    system_prompt_files: Sequence[Path] = (),
    discover_claude_md: bool = True,
) -> str:
    """Render the core orientation plus runtime prompt addenda."""

    resolved_cwd = cwd.expanduser().resolve()
    addenda = list(system_prompt_text)
    addenda.extend(
        path.expanduser().read_text(encoding="utf-8") for path in system_prompt_files
    )
    return build_system_prompt_with_addenda(
        build_runtime_orientation(model, cwd=resolved_cwd, user_name=user_name),
        cwd=resolved_cwd,
        addenda=addenda,
        discover_claude_md=discover_claude_md,
    )

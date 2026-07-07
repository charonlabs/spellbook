"""CLI compatibility wrapper for dream-chapter pipeline orchestration."""

from __future__ import annotations

from spellbook.dreaming.pipeline import *  # noqa: F403
from spellbook.dreaming.pipeline import main


if __name__ == "__main__":
    main()

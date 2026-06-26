from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SystemResponse:
    command: str
    content: str
    plaintext: str
    metadata: dict[str, Any] | None = None

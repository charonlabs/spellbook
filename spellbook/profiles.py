"""Session capability profiles.

``session_type`` remains the persisted, user-facing alias. ``SessionProfile``
is the runtime contract: it splits the old master switch into orthogonal flags
that composition code can consume without adding more enum branches.
"""

from collections.abc import Set as AbstractSet
from typing import Literal

from pydantic import BaseModel, ConfigDict

SessionType = Literal[
    "main",
    "custom",
    "block_detector",
    "block_summarizer",
    "quantum",
]

SessionProfileName = SessionType

ToolSurface = Literal[
    "main",
    "custom",
    "block_detector",
    "block_summarizer",
    "quantum",
]

ToolMetadataKind = Literal["standard", "block_detector", "block_summarizer"]


class SessionProfile(BaseModel, frozen=True):
    """Orthogonal runtime capabilities for one session."""

    model_config = ConfigDict(extra="forbid")

    name: SessionProfileName
    session_id_prefix: str
    tool_surface: ToolSurface
    tool_metadata: ToolMetadataKind = "standard"
    custom_surface: bool = False
    homunculus_lifecycle: bool = False
    block_detection: bool = False
    skill_discovery: bool = False
    inbound_injection: bool = False
    ambient_time: bool = False
    hearth: bool = False
    conduit_surfaces: bool = True

    def should_discover_skills(
        self, custom_surface_categories: AbstractSet[str] | None = None
    ) -> bool:
        if not self.skill_discovery:
            return False
        if not self.custom_surface:
            return True
        return (
            custom_surface_categories is not None
            and "skills" in custom_surface_categories
        )


MAIN = SessionProfile(
    name="main",
    session_id_prefix="session",
    tool_surface="main",
    homunculus_lifecycle=True,
    block_detection=True,
    skill_discovery=True,
    inbound_injection=True,
    ambient_time=True,
    hearth=True,
    conduit_surfaces=True,
)

CUSTOM = SessionProfile(
    name="custom",
    session_id_prefix="custom_session",
    tool_surface="custom",
    custom_surface=True,
    homunculus_lifecycle=True,
    block_detection=True,
    skill_discovery=True,
    ambient_time=False,
    hearth=False,
    conduit_surfaces=True,
)

BLOCK_DETECTOR = SessionProfile(
    name="block_detector",
    session_id_prefix="bd_session",
    tool_surface="block_detector",
    tool_metadata="block_detector",
    custom_surface=False,
    homunculus_lifecycle=False,
    block_detection=False,
    skill_discovery=False,
    ambient_time=False,
    hearth=False,
    conduit_surfaces=False,
)

BLOCK_SUMMARIZER = SessionProfile(
    name="block_summarizer",
    session_id_prefix="bs_session",
    tool_surface="block_summarizer",
    tool_metadata="block_summarizer",
    custom_surface=False,
    homunculus_lifecycle=False,
    block_detection=False,
    skill_discovery=False,
    ambient_time=False,
    hearth=False,
    conduit_surfaces=False,
)

QUANTUM = SessionProfile(
    name="quantum",
    session_id_prefix="quantum_session",
    tool_surface="quantum",
    custom_surface=False,
    homunculus_lifecycle=True,
    block_detection=False,
    skill_discovery=False,
    ambient_time=False,
    hearth=False,
    conduit_surfaces=False,
)

PROFILE_BY_SESSION_TYPE: dict[str, SessionProfile] = {
    "main": MAIN,
    "custom": CUSTOM,
    "block_detector": BLOCK_DETECTOR,
    "block_summarizer": BLOCK_SUMMARIZER,
    "quantum": QUANTUM,
}


def profile_for_session_type(session_type: str) -> SessionProfile:
    try:
        return PROFILE_BY_SESSION_TYPE[session_type]
    except KeyError as e:
        raise ValueError(f"Unknown session_type: {session_type!r}.") from e

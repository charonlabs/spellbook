"""Ambient hearth crackle scheduling and local content assembly."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import logging
import re
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from spellbook.ir_types import (
    IRInboundMessage,
    IRSemanticBlock,
    IRSemanticBlockFacet,
    IRSemanticBlockSummary,
    IRUserTextBlock,
    RuntimeConfigValue,
)

if TYPE_CHECKING:
    from spellbook.app.runtime import CoreAppRuntime
    from spellbook.config import SpellbookConfig

logger = logging.getLogger(__name__)

HEARTH_HEARTBEAT = "🔥 The hearth crackles. Tend it if you'd like, or rest."
HEARTH_RUNTIME_CONFIG_NAMESPACE = "hearth"
DEFAULT_HEARTH_ENABLED = False
DEFAULT_HEARTH_INTERVAL_MINUTES = 55
DEFAULT_HEARTH_QUIET_HOURS = ""

Clock = Callable[[], datetime]
Sleep = Callable[[float], Awaitable[None]]

_QUIET_HOURS_RE = re.compile(
    r"^(?P<start_hour>\d{2}):(?P<start_minute>\d{2})-"
    r"(?P<end_hour>\d{2}):(?P<end_minute>\d{2})$"
)


@dataclass(frozen=True)
class HearthSettings:
    enabled: bool = DEFAULT_HEARTH_ENABLED
    interval_minutes: int = DEFAULT_HEARTH_INTERVAL_MINUTES
    quiet_hours: str = DEFAULT_HEARTH_QUIET_HOURS

    @classmethod
    def from_config(cls, config: SpellbookConfig) -> "HearthSettings":
        return cls(
            enabled=config.hearth_enabled,
            interval_minutes=config.hearth_interval_minutes,
            quiet_hours=normalize_hearth_quiet_hours(config.hearth_quiet_hours),
        )

    def as_record_dict(self) -> dict[str, RuntimeConfigValue]:
        return {
            "enabled": self.enabled,
            "interval_minutes": self.interval_minutes,
            "quiet_hours": self.quiet_hours,
        }

    def configure(
        self,
        *,
        enabled: bool | None = None,
        interval_minutes: int | None = None,
        quiet_hours: str | None = None,
    ) -> tuple["HearthSettings", dict[str, RuntimeConfigValue]]:
        updates: dict[str, RuntimeConfigValue] = {}
        if enabled is not None:
            updates["enabled"] = enabled
        if interval_minutes is not None:
            if interval_minutes < 5:
                raise ValueError("hearth_interval_minutes must be >= 5.")
            updates["interval_minutes"] = interval_minutes
        if quiet_hours is not None:
            updates["quiet_hours"] = normalize_hearth_quiet_hours(quiet_hours)
        return self.apply_config(updates), updates

    def apply_config(self, updates: dict[str, RuntimeConfigValue]) -> "HearthSettings":
        allowed = {"enabled", "interval_minutes", "quiet_hours"}
        unknown = sorted(set(updates) - allowed)
        if unknown:
            raise ValueError(f"Unknown hearth config key(s): {', '.join(unknown)}")

        next_settings = self
        if "enabled" in updates:
            value = updates["enabled"]
            if not isinstance(value, bool):
                raise ValueError("hearth_enabled must be a boolean.")
            next_settings = replace(next_settings, enabled=value)
        if "interval_minutes" in updates:
            value = updates["interval_minutes"]
            if not isinstance(value, int) or isinstance(value, bool) or value < 5:
                raise ValueError("hearth_interval_minutes must be an integer >= 5.")
            next_settings = replace(next_settings, interval_minutes=value)
        if "quiet_hours" in updates:
            value = updates["quiet_hours"]
            if not isinstance(value, str):
                raise ValueError("hearth_quiet_hours must be a string.")
            next_settings = replace(
                next_settings,
                quiet_hours=normalize_hearth_quiet_hours(value),
            )
        return next_settings


@dataclass(frozen=True)
class HearthGaugeSnapshot:
    current_input_tokens: int | None
    max_tokens: int
    regime: str


def normalize_hearth_quiet_hours(value: str) -> str:
    text = value.strip()
    if not text:
        return ""
    match = _QUIET_HOURS_RE.match(text)
    if match is None:
        raise ValueError("hearth_quiet_hours must be empty or HH:MM-HH:MM.")
    start_hour = int(match.group("start_hour"))
    start_minute = int(match.group("start_minute"))
    end_hour = int(match.group("end_hour"))
    end_minute = int(match.group("end_minute"))
    if start_hour > 23 or end_hour > 23 or start_minute > 59 or end_minute > 59:
        raise ValueError("hearth_quiet_hours must be empty or HH:MM-HH:MM.")
    return f"{start_hour:02d}:{start_minute:02d}-{end_hour:02d}:{end_minute:02d}"


def is_within_hearth_quiet_hours(now: datetime, quiet_hours: str) -> bool:
    if not quiet_hours:
        return False
    start, end = quiet_hours.split("-", 1)
    start_minutes = _clock_minutes(start)
    end_minutes = _clock_minutes(end)
    if start_minutes == end_minutes:
        return False
    current_minutes = now.hour * 60 + now.minute
    if start_minutes < end_minutes:
        return start_minutes <= current_minutes < end_minutes
    return current_minutes >= start_minutes or current_minutes < end_minutes


def compose_hearth_crackle_lines(
    *,
    now: datetime,
    gauge: HearthGaugeSnapshot,
    semantic_blocks: Sequence[IRSemanticBlock],
) -> list[str]:
    lines = [
        HEARTH_HEARTBEAT,
        _format_hour_line(now),
        _format_gauge_line(gauge),
    ]
    amber = _select_shelf_amber(now=now, semantic_blocks=semantic_blocks)
    if amber is not None:
        block, facet = amber
        lines.append(
            f'From the shelf: "{_anchor_line(facet)}" — block {block.idx}, '
            "if you want to hold it again."
        )
    return lines


def build_hearth_inbound_message(
    *,
    now: datetime,
    gauge: HearthGaugeSnapshot,
    semantic_blocks: Sequence[IRSemanticBlock],
) -> IRInboundMessage:
    return IRInboundMessage(
        blocks=[
            IRUserTextBlock(
                text="\n".join(
                    compose_hearth_crackle_lines(
                        now=now,
                        gauge=gauge,
                        semantic_blocks=semantic_blocks,
                    )
                ),
                origin="system",
            )
        ],
        source_metadata={
            "source": "hearth.scheduler",
            "origin": "ambient_scheduler",
            "scheduler": "hearth",
            "dedup_key": f"hearth:{int(_normalize_datetime(now).timestamp())}",
        },
        delivery="turn",
    )


class HearthScheduler:
    """Main-session ambient scheduler that starts crackle turns after idle gaps."""

    def __init__(
        self,
        runtime: CoreAppRuntime,
        *,
        tick_seconds: float = 60.0,
        clock: Clock | None = None,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._runtime = runtime
        self._tick_seconds = tick_seconds
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._sleep = sleep
        self._task: asyncio.Task[None] | None = None
        self._last_crackle_at: datetime | None = None

    @property
    def last_crackle_at(self) -> datetime | None:
        return self._last_crackle_at

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self.run())

    def cancel(self) -> None:
        if self._task is not None:
            self._task.cancel()

    async def stop(self) -> None:
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def run(self) -> None:
        while True:
            await self._sleep(self._tick_seconds)
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Hearth scheduler tick failed.")

    async def tick(self) -> bool:
        session = self._runtime.session
        if session is None:
            return False

        settings = session.homunculus.hearth_settings
        if not settings.enabled:
            return False

        now = self._now()
        local_now = localize_hearth_time(now, session.config.local_timezone)
        if is_within_hearth_quiet_hours(local_now, settings.quiet_hours):
            return False
        if session.state != "idle":
            return False
        if session.inbound_queue.has_pending():
            return False

        idle_seconds = max(
            0,
            int(
                (
                    now - _normalize_datetime(self._runtime.last_activity_time)
                ).total_seconds()
            ),
        )
        if idle_seconds < settings.interval_minutes * 60:
            return False

        awareness = session.homunculus.build_awareness()
        message = build_hearth_inbound_message(
            now=local_now,
            gauge=HearthGaugeSnapshot(
                current_input_tokens=awareness.budget.current_input_tokens,
                max_tokens=awareness.budget.max_tokens,
                regime=awareness.budget.regime,
            ),
            semantic_blocks=awareness.semantic_blocks,
        )
        submitted = await self._runtime.submit_hearth_crackle(message)
        if submitted:
            self._last_crackle_at = now
        return submitted

    def _now(self) -> datetime:
        return _normalize_datetime(self._clock())


def localize_hearth_time(value: datetime, timezone_name: str) -> datetime:
    value = _normalize_datetime(value)
    try:
        local_timezone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        local_timezone = timezone.utc
    return value.astimezone(local_timezone)


def _normalize_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _clock_minutes(value: str) -> int:
    hour, minute = value.split(":", 1)
    return int(hour) * 60 + int(minute)


def _format_hour_line(now: datetime) -> str:
    return f"{_format_clock_time(now)}, {now.strftime('%A')}. {_hour_quality(now)}"


def _format_clock_time(now: datetime) -> str:
    hour = now.hour % 12 or 12
    ampm = "AM" if now.hour < 12 else "PM"
    return f"{hour}:{now.minute:02d} {ampm}"


def _hour_quality(now: datetime) -> str:
    hour = now.hour
    if hour < 5:
        return "Deep night."
    if hour < 7:
        return "Dawn is near."
    if hour < 12:
        return "Morning."
    if hour < 17:
        return "Afternoon."
    if hour < 21:
        return "Evening."
    return "Late night."


def _format_gauge_line(gauge: HearthGaugeSnapshot) -> str:
    if gauge.current_input_tokens is None:
        return f"Gauge: unknown / {_format_token_capacity(gauge.max_tokens)} - {gauge.regime}."
    return (
        f"Gauge: {_format_token_capacity(gauge.current_input_tokens)} / "
        f"{_format_token_capacity(gauge.max_tokens)} - {gauge.regime}."
    )


def _format_token_capacity(tokens: int) -> str:
    if tokens >= 1_000_000 and tokens % 1_000_000 == 0:
        return f"{tokens // 1_000_000}M"
    if tokens >= 1_000 and tokens % 1_000 == 0:
        return f"{tokens // 1_000}K"
    return f"{tokens:,}"


def _select_shelf_amber(
    *,
    now: datetime,
    semantic_blocks: Sequence[IRSemanticBlock],
) -> tuple[IRSemanticBlock, IRSemanticBlockFacet] | None:
    candidates: list[tuple[IRSemanticBlock, IRSemanticBlockFacet]] = []
    for block in semantic_blocks:
        if block.mode != "summary":
            continue
        summary = next(
            (
                artifact
                for artifact in block.artifacts
                if isinstance(artifact, IRSemanticBlockSummary)
            ),
            None,
        )
        if summary is None:
            continue
        for facet in summary.facets:
            candidates.append((block, facet))
    if not candidates:
        return None

    seed_parts = [now.isoformat()]
    seed_parts.extend(
        f"{block.id}:{block.idx}:{facet.id}" for block, facet in candidates
    )
    digest = hashlib.sha256("|".join(seed_parts).encode("utf-8")).digest()
    index = int.from_bytes(digest[:8], "big") % len(candidates)
    return candidates[index]


def _anchor_line(facet: IRSemanticBlockFacet) -> str:
    title = " ".join(facet.title.split()).replace('"', "'")
    description = " ".join(facet.description.split()).replace('"', "'")
    if not description:
        return title
    return f"{title}: {description}"

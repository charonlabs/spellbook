"""Ambient time/idle orientation emitted through footer records.

`Timekeeper` owns wall-clock orientation policy for the session. It is a
mechanism service, not Homunculus self-state: it watches idle gaps and clock
rollovers, then emits replayable footer queue records through `FooterController`.
"""

from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from spellbook.config import SpellbookConfig
from spellbook.footer import FooterController
from spellbook.ir_types import IRBlockRecord, IRRecord
from spellbook.round_lifecycle import RoundContext, RoundLifecycle
from spellbook.session_lifecycle import SessionContext, SessionLifecycle

Clock = Callable[[], datetime]


class Timekeeper:
    def __init__(
        self,
        config: SpellbookConfig,
        footer_c: FooterController,
        *,
        clock: Clock | None = None,
    ):
        self._footer_c = footer_c
        self._threshold_seconds = config.idle_footer_threshold_seconds
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._previous = self._now()
        self._idle_since: datetime | None = None
        self._local_timezone_name = config.local_timezone
        try:
            self._local_timezone = ZoneInfo(config.local_timezone)
        except ZoneInfoNotFoundError:
            self._local_timezone = timezone.utc
            self._local_timezone_name = "UTC"

    def enter_idle(self) -> None:
        now = self._now()
        self._idle_since = now
        self._previous = now

    def start_turn(self, turn_idx: int) -> None:
        now = self._now()
        previous = self._idle_since or self._previous
        idle_lines = self._idle_lines(previous, now)
        rollover_lines = self._rollover_lines(
            previous, now, force_current_time=bool(idle_lines)
        )
        self._idle_since = None
        self._previous = now
        if idle_lines:
            self._queue(idle_lines, key=f"time:turn:{turn_idx}")
        if rollover_lines:
            self._queue(rollover_lines, key=self._rollover_key(now))

    def observe_round(self, round_number: int) -> None:
        now = self._now()
        lines = self._rollover_lines(self._previous, now)
        self._previous = now
        if lines:
            self._queue(lines, key=f"{self._rollover_key(now)}:{round_number}")

    def note_resume(
        self,
        *,
        previous_activity_time: datetime | None,
        turn_idx: int,
    ) -> None:
        now = self._now()
        lines = self._resume_orientation_lines(previous_activity_time, now)
        self._previous = now
        if lines:
            self._queue(lines, key=f"time:resume:{turn_idx}")

    @staticmethod
    def latest_activity_time(records: Sequence[IRRecord]) -> datetime | None:
        for record in reversed(records):
            if isinstance(record, IRBlockRecord):
                return record.event.time
            record_time = getattr(record, "time", None)
            if isinstance(record_time, datetime):
                return record_time
        return None

    def _now(self) -> datetime:
        return self._normalize(self._clock())

    @staticmethod
    def _normalize(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    @staticmethod
    def _local_hour_key(value: datetime) -> tuple[int, int, int, int]:
        return (value.year, value.month, value.day, value.hour)

    def _idle_lines(self, previous: datetime, current: datetime) -> list[str]:
        previous = self._normalize(previous)
        current = self._normalize(current)
        gap_seconds = max(0, int((current - previous).total_seconds()))
        lines: list[str] = []

        if gap_seconds >= self._threshold_seconds:
            lines.append(
                f"Idle for {self._format_idle_gap(gap_seconds)} before this turn."
            )
        return lines

    def _resume_orientation_lines(
        self, previous: datetime | None, current: datetime
    ) -> list[str]:
        current = self._normalize(current)
        current_local = current.astimezone(self._local_timezone)
        if previous is None:
            return [
                f"Resumed at {self._format_local_time(current_local, include_minutes=True)}."
            ]

        previous = self._normalize(previous)
        gap_seconds = max(0, int((current - previous).total_seconds()))
        lines: list[str] = []
        if gap_seconds >= self._threshold_seconds:
            lines.append(
                f"Resumed at {self._format_local_time(current_local, include_minutes=True)} "
                f"after {self._format_idle_gap(gap_seconds)} idle."
            )
        lines.extend(self._rollover_lines(previous, current))
        return lines

    def _rollover_lines(
        self,
        previous: datetime,
        current: datetime,
        *,
        force_current_time: bool = False,
    ) -> list[str]:
        previous = self._normalize(previous)
        current = self._normalize(current)
        previous_local = previous.astimezone(self._local_timezone)
        current_local = current.astimezone(self._local_timezone)
        lines: list[str] = []
        if force_current_time or self._local_hour_key(
            previous_local
        ) != self._local_hour_key(current_local):
            day_name = current_local.strftime("%A")
            lines.append(
                f"It is now {self._format_local_time(current_local)}, {day_name}."
            )
        if previous_local.date() != current_local.date():
            lines.append(f"Date changed: {self._format_local_date(current_local)}.")
        return lines

    def _rollover_key(self, current: datetime) -> str:
        current_local = self._normalize(current).astimezone(self._local_timezone)
        return f"time:rollover:{current_local:%Y%m%dT%H}"

    def _queue(self, lines: Sequence[str], *, key: str) -> None:
        text = "\n".join(line for line in lines if line)
        if not text:
            return
        self._footer_c.queue_footer(
            text=text,
            footer_type="time",
            source="idle",
            key=key,
            priority=20,
        )

    @staticmethod
    def _format_idle_gap(seconds: int) -> str:
        minutes, remainder = divmod(max(0, seconds), 60)
        hours, minutes = divmod(minutes, 60)
        if not hours and not minutes and remainder:
            return f"{remainder}s"
        parts: list[str] = []
        if hours:
            parts.append(f"{hours}h")
        if minutes or not parts:
            parts.append(f"{minutes}m")
        if not hours and remainder and minutes < 5:
            parts.append(f"{remainder}s")
        return "".join(parts)

    def _format_local_time(
        self, value: datetime, *, include_minutes: bool = False
    ) -> str:
        hour = value.hour % 12 or 12
        ampm = "AM" if value.hour < 12 else "PM"
        zone = value.tzname() or self._local_timezone_name
        if include_minutes:
            return f"{hour}:{value.minute:02d} {ampm} {zone}"
        return f"{hour} {ampm} {zone}"

    @staticmethod
    def _format_local_date(value: datetime) -> str:
        return f"{value.strftime('%A')}, {value.strftime('%B')} {value.day}"


class TimekeeperSessionLifecycle(SessionLifecycle):
    def __init__(self, timekeeper: Timekeeper):
        self._timekeeper = timekeeper

    async def on_enter_idle(self, ctx: SessionContext) -> None:
        self._timekeeper.enter_idle()

    async def on_turn_started(self, ctx: SessionContext, turn_id: str) -> None:
        del turn_id
        self._timekeeper.start_turn(ctx.turn_idx)


class TimekeeperRoundLifecycle(RoundLifecycle):
    def __init__(self, timekeeper: Timekeeper):
        self._timekeeper = timekeeper

    async def before_round(self, ctx: RoundContext) -> None:
        self._timekeeper.observe_round(ctx.round_number)

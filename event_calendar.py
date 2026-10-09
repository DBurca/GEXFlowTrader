"""
NYSE session calendar and macro-release blackouts.

CPI and NFP dates are the BLS 2026 schedule (08:30 ET). FOMC dates are the
Federal Reserve's 2026 meeting calendar; the blackout is anchored to the
statement at 14:00 ET on the second day. Full-day holidays and early closes
follow the NYSE Group calendar for 2026 and 2027.

A release that prints before the cash open (CPI, NFP) also blacks out
09:30–09:45 ET. The literal 15-minute window around 08:30 ends before the
open, and the first cash prints are when that headline still dominates.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")
RTH_OPEN = time(9, 30)
CASH_REACTION_END = time(9, 45)

# Full-day closures. Source: NYSE Group holiday calendar, 2026–2027.
NYSE_HOLIDAYS: frozenset[date] = frozenset(
    {
        date(2026, 1, 1),
        date(2026, 1, 19),
        date(2026, 2, 16),
        date(2026, 4, 3),
        date(2026, 5, 25),
        date(2026, 6, 19),
        date(2026, 7, 3),
        date(2026, 9, 7),
        date(2026, 11, 26),
        date(2026, 12, 25),
        date(2027, 1, 1),
        date(2027, 1, 18),
        date(2027, 2, 15),
        date(2027, 3, 26),
        date(2027, 5, 31),
        date(2027, 6, 18),
        date(2027, 7, 5),
        date(2027, 9, 6),
        date(2027, 11, 25),
        date(2027, 12, 24),
    }
)

# Early closes, 13:00 ET. Source: NYSE Group.
EARLY_CLOSES: dict[date, time] = {
    date(2026, 11, 27): time(13, 0),
    date(2026, 12, 24): time(13, 0),
    date(2027, 11, 26): time(13, 0),
}

# (release timestamp ET, name). CPI/NFP from BLS; FOMC from the Fed calendar.
MACRO_RELEASES: tuple[tuple[datetime, str], ...] = (
    (datetime(2026, 1, 9, 8, 30, tzinfo=EASTERN), "NFP"),
    (datetime(2026, 1, 13, 8, 30, tzinfo=EASTERN), "CPI"),
    (datetime(2026, 1, 28, 14, 0, tzinfo=EASTERN), "FOMC"),
    (datetime(2026, 2, 11, 8, 30, tzinfo=EASTERN), "NFP"),
    (datetime(2026, 2, 13, 8, 30, tzinfo=EASTERN), "CPI"),
    (datetime(2026, 3, 6, 8, 30, tzinfo=EASTERN), "NFP"),
    (datetime(2026, 3, 11, 8, 30, tzinfo=EASTERN), "CPI"),
    (datetime(2026, 3, 18, 14, 0, tzinfo=EASTERN), "FOMC"),
    (datetime(2026, 4, 3, 8, 30, tzinfo=EASTERN), "NFP"),
    (datetime(2026, 4, 10, 8, 30, tzinfo=EASTERN), "CPI"),
    (datetime(2026, 4, 29, 14, 0, tzinfo=EASTERN), "FOMC"),
    (datetime(2026, 5, 8, 8, 30, tzinfo=EASTERN), "NFP"),
    (datetime(2026, 5, 12, 8, 30, tzinfo=EASTERN), "CPI"),
    (datetime(2026, 6, 5, 8, 30, tzinfo=EASTERN), "NFP"),
    (datetime(2026, 6, 10, 8, 30, tzinfo=EASTERN), "CPI"),
    (datetime(2026, 6, 17, 14, 0, tzinfo=EASTERN), "FOMC"),
    (datetime(2026, 7, 2, 8, 30, tzinfo=EASTERN), "NFP"),
    (datetime(2026, 7, 14, 8, 30, tzinfo=EASTERN), "CPI"),
    (datetime(2026, 7, 29, 14, 0, tzinfo=EASTERN), "FOMC"),
    (datetime(2026, 8, 7, 8, 30, tzinfo=EASTERN), "NFP"),
    (datetime(2026, 8, 12, 8, 30, tzinfo=EASTERN), "CPI"),
    (datetime(2026, 9, 4, 8, 30, tzinfo=EASTERN), "NFP"),
    (datetime(2026, 9, 11, 8, 30, tzinfo=EASTERN), "CPI"),
    (datetime(2026, 9, 16, 14, 0, tzinfo=EASTERN), "FOMC"),
    (datetime(2026, 10, 2, 8, 30, tzinfo=EASTERN), "NFP"),
    (datetime(2026, 10, 14, 8, 30, tzinfo=EASTERN), "CPI"),
    (datetime(2026, 10, 28, 14, 0, tzinfo=EASTERN), "FOMC"),
    (datetime(2026, 11, 6, 8, 30, tzinfo=EASTERN), "NFP"),
    (datetime(2026, 11, 10, 8, 30, tzinfo=EASTERN), "CPI"),
    (datetime(2026, 12, 4, 8, 30, tzinfo=EASTERN), "NFP"),
    (datetime(2026, 12, 9, 14, 0, tzinfo=EASTERN), "FOMC"),
    (datetime(2026, 12, 10, 8, 30, tzinfo=EASTERN), "CPI"),
)


def now_eastern(moment: datetime | None = None) -> datetime:
    current = moment or datetime.now(EASTERN)
    if current.tzinfo is None:
        return current.replace(tzinfo=EASTERN)
    return current.astimezone(EASTERN)


def is_trading_day(session_day: date) -> bool:
    return session_day.weekday() < 5 and session_day not in NYSE_HOLIDAYS


def session_close_time(session_day: date) -> time:
    return EARLY_CLOSES.get(session_day, time(16, 0))


def in_regular_session(moment: datetime | None = None) -> bool:
    current = now_eastern(moment)
    if not is_trading_day(current.date()):
        return False
    close_at = session_close_time(current.date())
    return RTH_OPEN <= current.time() < close_at


def next_session_open(moment: datetime | None = None) -> datetime:
    current = now_eastern(moment)
    candidate = current.replace(hour=9, minute=30, second=0, microsecond=0)
    if current >= candidate or not is_trading_day(candidate.date()):
        candidate = (candidate + timedelta(days=1)).replace(
            hour=9, minute=30, second=0, microsecond=0
        )
    while not is_trading_day(candidate.date()):
        candidate += timedelta(days=1)
    return candidate


def macro_blackout(
    moment: datetime | None = None,
    window_minutes: int = 15,
) -> str | None:
    """
    Name of the release that is blocking new trades, or None.

    The block is the 15 minutes before and after the official timestamp.
    Premarket releases also block the first 15 minutes of the cash session.
    """
    current = now_eastern(moment)
    window = timedelta(minutes=window_minutes)
    for release, name in MACRO_RELEASES:
        if abs(current - release) <= window:
            return name
        if (
            release.date() == current.date()
            and release.time() < RTH_OPEN
            and RTH_OPEN <= current.time() < CASH_REACTION_END
        ):
            return name
    return None

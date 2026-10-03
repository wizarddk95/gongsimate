"""Request clock. Asia/Seoul is explicit. OS local time is not the source."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

# Korea has used a fixed UTC+9 offset since 1988. Naming the zone here keeps
# the clock explicit on interpreters that do not ship the IANA database.
SEOUL = timezone(timedelta(hours=9), name="Asia/Seoul")
WEEKDAYS_KO = ("월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일")


def seoul_now() -> datetime:
    """Current time in Asia/Seoul. This does not read a naive local clock."""
    return datetime.now(SEOUL)


def request_time_context(moment: datetime) -> dict[str, str]:
    """Render one aware instant in Asia/Seoul, including the weekday.

    A naive datetime is rejected. Callers inject the clock in tests.
    """
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("request time must be timezone-aware")
    local = moment.astimezone(SEOUL)
    weekday = WEEKDAYS_KO[local.weekday()]
    clock = local.strftime("%H:%M:%S")
    day = local.strftime("%Y-%m-%d")
    return {
        "timezone": "Asia/Seoul",
        "local": f"{day} {clock}",
        "iso": local.isoformat(timespec="seconds"),
        "weekday": weekday,
        "display": f"{day} {weekday} {clock} KST",
    }

"""Grade search_disclosures arguments. This does not score answer text."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from disclosure_agent.agent.tool_args import SearchToolArguments, parse_search_arguments

COMPANY_ID = "00107613"
_FORECAST_WORDS = ("목표", "전망", "계획", "가이던스", "장기", "향후", "앞으로", "팔겠")
_ERROR_PRIORITY = (
    "invalid_schema",
    "required_search_omitted",
    "unnecessary_search",
    "excessive_repeated_tool_calls",
    "company_argument_error",
    "future_target_incorrectly_blocked",
    "direct_answer_missing",
    "period_year_error",
    "annual_vs_interim_error",
    "latest_semantics_error",
    "overly_narrow_filter",
    "overly_broad_filter",
)


def load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload["cases"]
    if not isinstance(cases, list) or len(cases) < 30:
        raise ValueError("tool-planning cases must contain at least 30 items")
    return cases


def parse_calls(raw_calls: list[Any]) -> tuple[list[SearchToolArguments], int]:
    parsed: list[SearchToolArguments] = []
    invalid = 0
    for raw in raw_calls:
        arguments, error = parse_search_arguments(raw)
        if error or arguments is None:
            invalid += 1
            continue
        parsed.append(arguments)
    return parsed, invalid


def grade_case(
    case: dict[str, Any],
    calls: list[SearchToolArguments],
    *,
    invalid_calls: int = 0,
    latest_period: tuple[int, int] = (2026, 6),
    action: str | None = None,
) -> dict[str, Any]:
    """Return pass/fail and every error category that applies."""
    errors = _errors(
        case,
        calls,
        invalid_calls=invalid_calls,
        latest_period=latest_period,
        action=action,
    )
    ordered = [item for item in _ERROR_PRIORITY if item in errors]
    return {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "pass": not ordered,
        "errors": ordered,
    }


def _errors(
    case: dict[str, Any],
    calls: list[SearchToolArguments],
    *,
    invalid_calls: int,
    latest_period: tuple[int, int],
    action: str | None = None,
) -> set[str]:
    errors: set[str] = set()
    expected_action = case.get("expect_action")
    if expected_action and action != expected_action:
        errors.add("direct_answer_missing")
    if invalid_calls and not calls:
        errors.add("invalid_schema")
        return errors
    if len(calls) > 2 or _repeated(calls):
        errors.add("excessive_repeated_tool_calls")
    if any(call.company_id != case.get("company_id", COMPANY_ID) for call in calls):
        errors.add("company_argument_error")
    if case["expect_search"] and not calls:
        if case["category"] in {"forecast", "confirmed_future"}:
            errors.add("future_target_incorrectly_blocked")
        else:
            errors.add("required_search_omitted")
        return errors
    if not case["expect_search"]:
        if calls or invalid_calls:
            errors.add("unnecessary_search")
        return errors
    errors.update(_category_errors(case, calls, latest_period))
    return errors


def _category_errors(
    case: dict[str, Any],
    calls: list[SearchToolArguments],
    latest_period: tuple[int, int],
) -> set[str]:
    category = case["category"]
    if category == "annual":
        return _annual_errors(calls, int(case["year"]))
    if category == "quarter":
        return _quarter_errors(calls, int(case["year"]), int(case["quarter"]))
    if category == "half":
        return _half_errors(calls, int(case["year"]))
    if category == "latest":
        return _latest_errors(calls, latest_period)
    if category == "explanation":
        return _explanation_errors(calls, case.get("year"))
    if category == "forecast":
        return _forecast_errors(calls, case.get("future_year"))
    if category == "confirmed_future":
        return _confirmed_errors(calls, int(case["future_year"]))
    if category == "business":
        return _business_errors(calls)
    if category == "through_quarter":
        return _through_quarter_errors(calls, int(case["year"]), int(case["quarter"]))
    raise ValueError(f"unknown tool-planning category {category}")


def _annual_errors(calls: list[SearchToolArguments], year: int) -> set[str]:
    if any(_annual_exact(call, year) for call in calls):
        return set()
    if any(_covers_year(call, year) and _interim_only(call) for call in calls):
        return {"annual_vs_interim_error"}
    if any(_covers_year(call, year) and _mixes_interim(call) for call in calls):
        return {"overly_broad_filter"}
    if not any(_covers_year(call, year) for call in calls):
        return {"period_year_error"}
    return {"annual_vs_interim_error"}


def _quarter_errors(calls: list[SearchToolArguments], year: int, quarter: int) -> set[str]:
    if any(_quarter_exact(call, year, quarter) for call in calls):
        return set()
    if any(_annual_exact(call, year) for call in calls):
        return {"annual_vs_interim_error"}
    if not any(_covers_year(call, year) or _unbounded(call) for call in calls):
        return {"period_year_error"}
    return {"annual_vs_interim_error"}


def _half_errors(calls: list[SearchToolArguments], year: int) -> set[str]:
    if any(_half_exact(call, year, 1) for call in calls):
        return set()
    if any(_annual_exact(call, year) or _quarter_exact(call, year, 1) for call in calls):
        return {"annual_vs_interim_error"}
    if not any(_covers_year(call, year) for call in calls):
        return {"period_year_error"}
    return {"annual_vs_interim_error"}


def _latest_errors(
    calls: list[SearchToolArguments],
    latest_period: tuple[int, int],
) -> set[str]:
    if any(_latest_call(call, latest_period) for call in calls):
        return set()
    return {"latest_semantics_error"}


def _explanation_errors(calls: list[SearchToolArguments], year: object) -> set[str]:
    if _only_old_year(calls, 2023):
        return {"overly_narrow_filter"}
    if isinstance(year, int) and not any(_covers_named_year(call, year) for call in calls):
        return {"period_year_error"}
    if all(_single_quarter(call) for call in calls):
        return {"overly_narrow_filter"}
    return set()


def _forecast_errors(calls: list[SearchToolArguments], future_year: object) -> set[str]:
    if any(_forecast_strategy(call, future_year) for call in calls):
        return set()
    return {"overly_narrow_filter"}


def _confirmed_errors(calls: list[SearchToolArguments], year: int) -> set[str]:
    if any(_confirmed_strategy(call, year) for call in calls):
        return set()
    if any(_covers_year(call, year) and _interim_only(call) for call in calls):
        return {"annual_vs_interim_error"}
    return {"period_year_error"}


def _business_errors(calls: list[SearchToolArguments]) -> set[str]:
    if _only_old_year(calls, 2023):
        return {"overly_narrow_filter"}
    return set()


def _through_quarter_errors(
    calls: list[SearchToolArguments],
    year: int,
    quarter: int,
) -> set[str]:
    if any(_through_quarter(call, year, quarter) for call in calls):
        return set()
    if any(_quarter_exact(call, year, 1) for call in calls) and quarter > 1:
        return {"overly_narrow_filter"}
    if not any(_covers_year(call, year) for call in calls):
        return {"period_year_error"}
    return {"annual_vs_interim_error"}


def _annual_exact(call: SearchToolArguments, year: int) -> bool:
    period = call.period
    if period is None or period.quarter is not None or period.half is not None:
        return False
    if period.mode == "exact" and period.year == year:
        return call.report_types == ["annual"]
    if period.mode == "range" and period.year_from == year and period.year_to == year:
        return call.report_types == ["annual"]
    return False


def _quarter_exact(call: SearchToolArguments, year: int, quarter: int) -> bool:
    period = call.period
    if period is None or period.mode != "exact" or period.half is not None:
        return False
    if period.year != year or period.quarter != quarter:
        return False
    if call.report_types is None:
        return True
    return "quarterly" in call.report_types and "annual" not in call.report_types


def _half_exact(call: SearchToolArguments, year: int, half: int) -> bool:
    period = call.period
    if period is None or period.mode != "exact" or period.quarter is not None:
        return False
    if period.year != year or period.half != half:
        return False
    if call.report_types is None:
        return True
    return "semiannual" in call.report_types


def _latest_call(call: SearchToolArguments, latest_period: tuple[int, int]) -> bool:
    period = call.period
    if period is None:
        return False
    year, month = latest_period
    if period.mode == "latest" and not _annual_only(call):
        if period.year not in (None, year):
            return False
        return period.quarter is None and period.half is None
    if period.mode != "exact" or period.year != year:
        return False
    if month == 6 and period.half == 1:
        return True
    if month == 6 and period.quarter == 2 and not _annual_only(call):
        return True
    if month == 3 and period.quarter == 1:
        return True
    if month == 9 and period.quarter == 3:
        return True
    if month == 12 and (period.quarter == 4 or _annual_only(call)):
        return True
    return False


def _covers_year(call: SearchToolArguments, year: int) -> bool:
    period = call.period
    if period is None:
        return False
    if period.mode == "range":
        start = period.year_from if period.year_from is not None else year
        end = period.year_to if period.year_to is not None else year
        return start <= year <= end
    return period.year == year


def _covers_named_year(call: SearchToolArguments, year: int) -> bool:
    if _unbounded(call) or (call.period is not None and call.period.mode == "latest"):
        return True
    return _covers_year(call, year)


def _unbounded(call: SearchToolArguments) -> bool:
    period = call.period
    if period is None:
        return not call.report_types and not call.filing_date_from and not call.filing_date_to
    if period.mode == "range":
        return period.year_from is None and period.year_to is None
    return (
        period.mode == "exact"
        and period.year is None
        and period.quarter is None
        and period.half is None
    )


def _interim_only(call: SearchToolArguments) -> bool:
    period = call.period
    if period is not None and (period.quarter is not None or period.half is not None):
        return True
    if not call.report_types:
        return False
    return "annual" not in call.report_types


def _mixes_interim(call: SearchToolArguments) -> bool:
    period = call.period
    if period is None or period.year is None or not _covers_year(call, period.year):
        return False
    if call.report_types is None:
        return not _interim_only(call)
    return "annual" in call.report_types and (
        "quarterly" in call.report_types or "semiannual" in call.report_types
    )


def _annual_only(call: SearchToolArguments) -> bool:
    return call.report_types == ["annual"]


def _single_quarter(call: SearchToolArguments) -> bool:
    period = call.period
    return period is not None and period.quarter is not None and period.mode == "exact"


def _only_old_year(calls: list[SearchToolArguments], year: int) -> bool:
    if not calls:
        return False
    return all(
        call.period is not None
        and call.period.mode == "exact"
        and call.period.year == year
        and call.period.quarter is None
        and call.period.half is None
        for call in calls
    )


def _forecast_strategy(call: SearchToolArguments, future_year: object) -> bool:
    if isinstance(future_year, int) and _covers_year(call, future_year):
        return True
    if any(word in call.query for word in _FORECAST_WORDS):
        return True
    return _unbounded(call) or (call.period is not None and call.period.mode == "latest")


def _confirmed_strategy(call: SearchToolArguments, year: int) -> bool:
    period = call.period
    if period is None or period.mode != "exact" or period.year != year:
        return False
    if period.quarter is not None or period.half is not None:
        return False
    if call.report_types is None:
        return True
    return call.report_types == ["annual"]


def _through_quarter(call: SearchToolArguments, year: int, quarter: int) -> bool:
    if _quarter_exact(call, year, quarter):
        return True
    period = call.period
    if period is None or period.year != year:
        return False
    if quarter >= 2 and period.half == 1 and quarter <= 2:
        return True
    if period.mode == "range" and _covers_year(call, year):
        return True
    return (
        period.mode == "exact"
        and period.quarter is None
        and period.half is None
        and not _annual_only(call)
    )


def _repeated(calls: list[SearchToolArguments]) -> bool:
    if len(calls) < 2:
        return False
    first = calls[0].model_dump_json()
    return any(call.model_dump_json() == first for call in calls[1:])

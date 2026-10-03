"""search_disclosures arguments.

The model chooses query text and metadata filters. This module checks that
those values are well formed and maps them onto stored filing listings.
It does not read the user question.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from disclosure_agent.agent.clock import SEOUL
from disclosure_agent.agent.filing_relation import FilingRelation, suppressed_originals
from disclosure_agent.agent.temporal import FilingPeriod, period_label, receipt_on_or_before

ReportType = Literal["annual", "semiannual", "quarterly"]
PeriodMode = Literal["exact", "latest", "range"]
TOOL_NAME = "search_disclosures"
MIN_YEAR = 1990
MAX_YEAR = 2100
MAX_QUERY_CHARS = 400
QUARTER_END_MONTH = {1: 3, 2: 6, 3: 9, 4: 12}
HALF_END_MONTH = {1: 6, 2: 12}
REPORT_TYPE_NAME = {
    "annual": "사업보고서",
    "semiannual": "반기보고서",
    "quarterly": "분기보고서",
}

SEARCH_TOOL_DESCRIPTION = """색인된 DART 정기공시를 검색합니다.
사업보고서(연간), 반기보고서, 분기보고서가 있습니다.

query는 찾을 내용입니다.
period, report_types, filing_date는 메타데이터 필터입니다.
질문의 모든 뉘앙스를 hard filter로 바꿀 필요는 없습니다.

- "YYYY년 매출"처럼 연간 실적을 물으면 year와 report_types ["annual"]을 지정합니다.
- 분기 질문은 quarter(1=3월, 2=6월, 3=9월, 4=12월)를 지정합니다.
- 반기는 half(1=상반기, 2=하반기)를 지정합니다.
- "최근", "요즘", "최신", "마지막으로 발표된", "현재 확인 가능한"은
  request_time과 corpus_as_of를 보고 period.mode를 "latest"로 둡니다.
  latest는 색인 목록에서 보고기간이 가장 늦은 공시입니다.
- 증가 이유, 배경, 원인은 그 연도 공시 밖에 후속 공시가 설명할 수 있습니다.
  먼저 좁게 검색하고, 결과가 부족하면 범위를 넓혀 이 도구를 다시 호출합니다.
- 미래 연도의 목표나 전망은 미래라는 이유만으로 건너뛰지 않습니다.
  목표·전망을 query에 담아 검색합니다.
- request_time보다 미래의 확정 실적은 이 도구를 호출하지 않습니다.
  그 경우는 answer_directly입니다. 목표·계획·전망은 여기에 해당하지 않습니다.
- as_of는 검색 상한 시각입니다. 비워 두면 corpus_as_of까지 검색합니다.
  request_time이 corpus_as_of보다 늦으면 as_of에 request_time을 넣지 않습니다.
- 확정 실적은 그 연도·보고서로 exact 필터를 겁니다.
  맞는 공시가 없으면 없는 것입니다.
- company_id는 입력의 canonical id만 사용합니다.
  top_k는 1 이상 서버 상한 이하입니다.
- 기업 사실은 검색 없이 만들지 않습니다.
  검색이 필요 없으면 도구를 호출하지 않습니다.
- 이 도구는 한 질문에서 최대 2번 호출할 수 있습니다.
"""


class PeriodArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    year: int | None = None
    quarter: int | None = None
    half: int | None = None
    mode: PeriodMode = "exact"
    year_from: int | None = None
    year_to: int | None = None


class SearchToolArguments(BaseModel):
    """Metadata filters a retrieval index can apply. Not a semantic label."""

    model_config = ConfigDict(extra="forbid")

    query: str
    company_id: str
    period: PeriodArgs | None = None
    filing_date_from: str | None = None
    filing_date_to: str | None = None
    report_types: list[ReportType] | None = None
    as_of: str | None = None
    top_k: int = Field(default=20)


@dataclass(frozen=True)
class ResolvedSearch:
    """Arguments that passed schema and policy checks."""

    arguments: SearchToolArguments
    filing_date_from: date | None
    filing_date_to: date | None
    as_of: datetime | None
    top_k: int


@dataclass(frozen=True)
class ToolResolution:
    status: str
    message: str
    resolved: ResolvedSearch | None = None
    filings: tuple[FilingPeriod, ...] | None = None


def search_tool_parameters() -> dict[str, Any]:
    """Strict JSON schema sent to the Responses API."""
    period = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "year": {"type": ["integer", "null"]},
            "quarter": {"type": ["integer", "null"]},
            "half": {"type": ["integer", "null"]},
            "mode": {"type": "string", "enum": ["exact", "latest", "range"]},
            "year_from": {"type": ["integer", "null"]},
            "year_to": {"type": ["integer", "null"]},
        },
        "required": ["year", "quarter", "half", "mode", "year_from", "year_to"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "query": {"type": "string"},
            "company_id": {"type": "string"},
            "period": {"anyOf": [{"type": "null"}, period]},
            "filing_date_from": {"type": ["string", "null"]},
            "filing_date_to": {"type": ["string", "null"]},
            "report_types": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["annual", "semiannual", "quarterly"],
                        },
                    },
                ]
            },
            "as_of": {"type": ["string", "null"]},
            "top_k": {"type": "integer"},
        },
        "required": [
            "query",
            "company_id",
            "period",
            "filing_date_from",
            "filing_date_to",
            "report_types",
            "as_of",
            "top_k",
        ],
    }


def search_tool_definition() -> dict[str, Any]:
    return {
        "type": "function",
        "name": TOOL_NAME,
        "description": SEARCH_TOOL_DESCRIPTION,
        "parameters": search_tool_parameters(),
        "strict": True,
    }


def parse_search_arguments(raw: Any) -> tuple[SearchToolArguments | None, str | None]:
    """Schema check only. Company and corpus policy are separate."""
    if not isinstance(raw, dict):
        return None, "tool arguments must be an object"
    try:
        parsed = SearchToolArguments.model_validate(raw)
    except ValidationError:
        return None, "tool arguments did not match the schema"
    if not parsed.query.strip():
        return None, "query is empty"
    if len(parsed.query) > MAX_QUERY_CHARS:
        return None, "query is too long"
    if not parsed.company_id.strip():
        return None, "company_id is empty"
    return parsed, None


def resolve_search_call(
    raw: Any,
    *,
    company_id: str,
    filings: tuple[FilingPeriod, ...],
    corpus_as_of: datetime | None,
    max_top_k: int,
    relations: tuple[FilingRelation, ...] = (),
) -> ToolResolution:
    """Validate one tool call and select listings. Does not rank text."""
    parsed, schema_error = parse_search_arguments(raw)
    if parsed is None:
        return ToolResolution("invalid_arguments", schema_error or "invalid arguments")
    policy = _policy_failure(
        parsed,
        company_id=company_id,
        corpus_as_of=corpus_as_of,
        max_top_k=max_top_k,
    )
    if policy is not None:
        status, message = policy
        return ToolResolution(status, message)
    resolved = ResolvedSearch(
        arguments=parsed,
        filing_date_from=_parse_day(parsed.filing_date_from),
        filing_date_to=_parse_day(parsed.filing_date_to),
        as_of=None if parsed.as_of is None else _parse_moment(parsed.as_of),
        top_k=parsed.top_k,
    )
    if not filings and _is_restrictive(parsed):
        return ToolResolution(
            "no_matching_filing",
            "조건에 맞는 공시 목록이 없습니다.",
            resolved=resolved,
            filings=(),
        )
    if not filings:
        return ToolResolution(
            "ok",
            "filing catalog is empty; retrieval is not receipt-filtered",
            resolved=resolved,
            filings=None,
        )
    matched = select_filings(
        filings,
        resolved,
        corpus_as_of=corpus_as_of,
        relations=relations,
    )
    if not matched:
        return ToolResolution(
            "no_matching_filing",
            "조건에 맞는 공시가 없습니다.",
            resolved=resolved,
            filings=(),
        )
    return ToolResolution("ok", "filings matched", resolved=resolved, filings=matched)


def select_filings(
    filings: tuple[FilingPeriod, ...],
    resolved: ResolvedSearch,
    *,
    corpus_as_of: datetime | None,
    relations: tuple[FilingRelation, ...] = (),
) -> tuple[FilingPeriod, ...]:
    """Apply clock and metadata filters. Receipt date is not a fiscal year."""
    args = resolved.arguments
    hidden = suppressed_originals(filings, relations, as_of=resolved.as_of)
    pool = [item for item in filings if item.receipt_no not in hidden]
    pool = [item for item in pool if receipt_on_or_before(item, corpus_as_of)]
    if resolved.as_of is not None:
        pool = [item for item in pool if receipt_on_or_before(item, resolved.as_of)]
    pool = [
        item
        for item in pool
        if _receipt_in_range(item, resolved.filing_date_from, resolved.filing_date_to)
    ]
    if args.report_types:
        names = {REPORT_TYPE_NAME[item] for item in args.report_types}
        pool = [item for item in pool if item.report_type in names]
    period = args.period
    if period is None:
        return _sort_filings(pool)
    if period.mode == "range":
        pool = [item for item in pool if _year_in_range(item, period.year_from, period.year_to)]
        return _sort_filings(pool)
    if period.year is not None:
        pool = [item for item in pool if item.period_end_year == period.year]
    if period.quarter is not None:
        month = QUARTER_END_MONTH[period.quarter]
        pool = [item for item in pool if item.period_end_month == month]
    if period.half is not None:
        month = HALF_END_MONTH[period.half]
        pool = [item for item in pool if item.period_end_month == month]
    if period.mode == "latest":
        pool = list(_at_max_period(pool))
    return _sort_filings(pool)


def arguments_record(resolved: ResolvedSearch) -> dict[str, Any]:
    args = resolved.arguments
    period = None
    if args.period is not None:
        period = args.period.model_dump()
    return {
        "query": args.query,
        "company_id": args.company_id,
        "period": period,
        "filing_date_from": args.filing_date_from,
        "filing_date_to": args.filing_date_to,
        "report_types": None if args.report_types is None else list(args.report_types),
        "as_of": args.as_of,
        "top_k": resolved.top_k,
    }


def _policy_failure(
    parsed: SearchToolArguments,
    *,
    company_id: str,
    corpus_as_of: datetime | None,
    max_top_k: int,
) -> tuple[str, str] | None:
    if parsed.company_id.strip() != company_id:
        return "unsupported_company", "company_id is not the resolved company"
    if parsed.top_k < 1 or parsed.top_k > max_top_k:
        return "invalid_arguments", "top_k is outside the server limit"
    if parsed.report_types is not None and len(parsed.report_types) == 0:
        return "invalid_arguments", "report_types is empty"
    period = parsed.period
    if period is not None:
        failure = _period_failure(period)
        if failure is not None:
            return "invalid_arguments", failure
    try:
        start = _parse_day(parsed.filing_date_from)
        end = _parse_day(parsed.filing_date_to)
    except ValueError:
        return "invalid_arguments", "filing date is not YYYY-MM-DD"
    if start is not None and end is not None and start > end:
        return "invalid_arguments", "filing date range is reversed"
    if parsed.as_of is not None:
        try:
            as_of = _parse_moment(parsed.as_of)
        except ValueError:
            return "invalid_arguments", "as_of is not a timestamp"
        if corpus_as_of is not None and as_of > corpus_as_of:
            return (
                "corpus_not_fresh_enough",
                "as_of is later than corpus_as_of",
            )
    return None


def _period_failure(period: PeriodArgs) -> str | None:
    if period.quarter is not None and period.half is not None:
        return "quarter and half are both set"
    if period.quarter is not None and period.quarter not in QUARTER_END_MONTH:
        return "quarter must be 1, 2, 3, or 4"
    if period.half is not None and period.half not in HALF_END_MONTH:
        return "half must be 1 or 2"
    for label, value in (
        ("year", period.year),
        ("year_from", period.year_from),
        ("year_to", period.year_to),
    ):
        if value is not None and not MIN_YEAR <= value <= MAX_YEAR:
            return f"{label} is outside {MIN_YEAR}-{MAX_YEAR}"
    if period.mode == "range":
        if period.year_from is None and period.year_to is None:
            return "range mode needs year_from or year_to"
        if (
            period.year_from is not None
            and period.year_to is not None
            and period.year_from > period.year_to
        ):
            return "year range is reversed"
    return None


def _is_restrictive(parsed: SearchToolArguments) -> bool:
    if parsed.report_types or parsed.filing_date_from or parsed.filing_date_to or parsed.as_of:
        return True
    period = parsed.period
    if period is None:
        return False
    if period.mode == "latest":
        return True
    if period.mode == "range":
        return True
    return period.year is not None or period.quarter is not None or period.half is not None


def _parse_day(value: str | None) -> date | None:
    if value is None:
        return None
    text = value.strip()
    if re.fullmatch(r"\d{8}", text):
        return date(int(text[0:4]), int(text[4:6]), int(text[6:8]))
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return date.fromisoformat(text)
    raise ValueError(text)


def _parse_moment(value: str) -> datetime:
    text = value.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return datetime(int(text[0:4]), int(text[5:7]), int(text[8:10]), tzinfo=SEOUL)
    if re.fullmatch(r"\d{8}", text):
        return datetime(int(text[0:4]), int(text[4:6]), int(text[6:8]), tzinfo=SEOUL)
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=SEOUL)
    return moment.astimezone(SEOUL)


def _receipt_day(period: FilingPeriod) -> date | None:
    raw = period.receipt_date
    if len(raw) != 8 or not raw.isdigit():
        return None
    return date(int(raw[0:4]), int(raw[4:6]), int(raw[6:8]))


def _receipt_in_range(
    period: FilingPeriod,
    start: date | None,
    end: date | None,
) -> bool:
    if start is None and end is None:
        return True
    filed = _receipt_day(period)
    if filed is None:
        return True
    if start is not None and filed < start:
        return False
    if end is not None and filed > end:
        return False
    return True


def _year_in_range(period: FilingPeriod, start: int | None, end: int | None) -> bool:
    year = period.period_end_year
    if year is None:
        return False
    if start is not None and year < start:
        return False
    if end is not None and year > end:
        return False
    return True


def _at_max_period(filings: list[FilingPeriod]) -> tuple[FilingPeriod, ...]:
    keyed = [
        (key, item)
        for item in filings
        if (key := _period_key(item)) is not None
    ]
    if not keyed:
        return ()
    best = max(key for key, _item in keyed)
    return tuple(item for key, item in keyed if key == best)


def _period_key(period: FilingPeriod) -> tuple[int, int] | None:
    if period.period_end_year is None or period.period_end_month is None:
        return None
    return (period.period_end_year, period.period_end_month)


def _sort_filings(filings: list[FilingPeriod]) -> tuple[FilingPeriod, ...]:
    return tuple(
        sorted(
            filings,
            key=lambda item: (
                -(item.period_end_year or 0),
                -(item.period_end_month or 0),
                item.receipt_no,
            ),
        )
    )


PUBLIC_TOOL_REASONS: dict[str, str] = {
    "no_matching_filing": "현재 색인에서 요청한 조건의 공시를 찾지 못했습니다.",
    "no_results": "검색은 실행됐지만 색인된 본문에서 해당하는 구간을 찾지 못했습니다.",
    "invalid_arguments": "검색 조건을 확인하지 못했습니다.",
    "unsupported_company": "확인된 기업과 다른 검색 대상입니다.",
    "corpus_not_fresh_enough": "요청한 시점은 색인 기준 시각보다 뒤입니다.",
    "tool_failure": "검색을 완료하지 못했습니다.",
    "search_not_requested": "공시 검색을 수행하지 않았습니다.",
}


def public_tool_observation(
    *,
    status: str,
    company_name: str,
    corpus_as_of: str | None,
    requested_period: dict[str, Any] | None,
    report_types: list[str] | None,
    matched_count: int,
    latest_available_period: str | None,
) -> dict[str, Any]:
    """Facts the model may phrase. This is the indexed corpus, not the world."""
    return {
        "status": status,
        "scope": "indexed_corpus",
        "company": company_name,
        "requested_period": requested_period,
        "report_types": report_types,
        "matched_count": matched_count,
        "corpus_as_of": corpus_as_of,
        "latest_available_period": latest_available_period,
        "public_reason": PUBLIC_TOOL_REASONS.get(
            status, "현재 색인 상태에서 답변을 확정하지 못했습니다."
        ),
    }


def filing_public_row(period: FilingPeriod) -> dict[str, Any]:
    row = {
        "receipt_no": period.receipt_no,
        "report_name": period.report_name,
        "report_type": period.report_type,
        "filing_period": period_label(period),
        "receipt_date": period.receipt_date,
    }
    if period.source_sha256:
        row["source_sha256"] = period.source_sha256
    return row

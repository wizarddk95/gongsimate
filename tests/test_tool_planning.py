"""Tool argument checks. These tests do not call OpenAI."""

from __future__ import annotations

import json
from datetime import datetime

from test_agent_product import events_of, registry, sample_hit
from test_grounded_answer import _output

from disclosure_agent.agent.catalog import repo_root
from disclosure_agent.agent.clock import SEOUL
from disclosure_agent.agent.evidence import EvidenceBuilder
from disclosure_agent.agent.llm import OpenAIAnswerGenerator
from disclosure_agent.agent.planner import PlannerContext, RawQueryPlanner, ScriptedPlanner
from disclosure_agent.agent.runtime import AgentRuntime
from disclosure_agent.agent.search import SearchDisclosuresRequest, SearchDisclosuresResult
from disclosure_agent.agent.temporal import filing_period_from_listing
from disclosure_agent.agent.tool_args import resolve_search_call, search_tool_parameters
from disclosure_agent.agent.tool_planning_eval import grade_case, load_cases, parse_calls
from disclosure_agent.retrieval_unit import D2A_VERSION


def _period(year: int, **extra: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "year": year,
        "quarter": None,
        "half": None,
        "mode": "exact",
        "year_from": None,
        "year_to": None,
    }
    payload.update(extra)
    return payload


def _call(query: str, **extra: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "query": query,
        "company_id": "00107613",
        "period": None,
        "filing_date_from": None,
        "filing_date_to": None,
        "report_types": None,
        "as_of": None,
        "top_k": 20,
    }
    payload.update(extra)
    return payload


def _filing(receipt: str, name: str, received: str):
    return filing_period_from_listing(
        {"report_nm": name, "rcept_no": receipt, "rcept_dt": received, "rm": ""}
    )


def _filings():
    return (
        _filing("20250320001435", "사업보고서 (2024.12)", "20250320"),
        _filing("20250514001051", "분기보고서 (2025.03)", "20250514"),
        _filing("20260323000847", "사업보고서 (2025.12)", "20260323"),
        _filing("20260814002283", "반기보고서 (2026.06)", "20260814"),
    )


def _runtime(search, planner, filings) -> AgentRuntime:
    return AgentRuntime(
        registry=registry(),
        search=search,
        evidence_builder=EvidenceBuilder(),
        answer_generator=OpenAIAnswerGenerator(_Answer()),
        filings=filings,
        corpus_as_of=datetime(2026, 9, 25, 10, 26, tzinfo=SEOUL),
        clock=lambda: datetime(2026, 10, 2, 10, 52, tzinfo=SEOUL),
        planner=planner,
    )


class _Answer:
    model = "gpt-test"
    sdk_version = "test"

    def __init__(self) -> None:
        self.inputs: list[str] = []

    def complete(self, user_input: str):
        self.inputs.append(user_input)
        payload = json.loads(user_input)
        rows = payload["evidence"]
        if not rows:
            return _output(answer="근거 없음", abstain=True)
        return _output(rows[0]["evidence_id"], answer="공시에 적힌 내용입니다.")


def _result(text: str, receipt: str) -> SearchDisclosuresResult:
    hit = sample_hit(text)
    hit.receipt_no = receipt
    hit.retrieval_unit_id = f"unit-{receipt}-{text[:8]}"
    return SearchDisclosuresResult(
        hits=[hit],
        retriever_type="dense",
        representation_version=D2A_VERSION,
        embedding_model="BAAI/bge-m3",
        company_id="00107613",
        corp_code="00107613",
    )


def test_tool_schema_rejects_semantic_policy_fields_and_bad_values() -> None:
    schema = search_tool_parameters()
    assert "historical_explanation" not in json.dumps(schema)
    assert schema["additionalProperties"] is False
    filings = _filings()
    as_of = datetime(2026, 9, 25, tzinfo=SEOUL)
    bad_company = resolve_search_call(
        _call("매출", company_id="99999999"),
        company_id="00107613",
        filings=filings,
        corpus_as_of=as_of,
        max_top_k=20,
    )
    assert bad_company.status == "unsupported_company"
    bad_top = resolve_search_call(
        _call("매출", top_k=100),
        company_id="00107613",
        filings=filings,
        corpus_as_of=as_of,
        max_top_k=20,
    )
    assert bad_top.status == "invalid_arguments"
    both = resolve_search_call(
        _call("매출", period=_period(2025, quarter=1, half=1)),
        company_id="00107613",
        filings=filings,
        corpus_as_of=as_of,
        max_top_k=20,
    )
    assert both.status == "invalid_arguments"
    fresh = resolve_search_call(
        _call("매출", as_of="2027-01-01"),
        company_id="00107613",
        filings=filings,
        corpus_as_of=as_of,
        max_top_k=20,
    )
    assert fresh.status == "corpus_not_fresh_enough"
    extra = resolve_search_call(
        {**_call("매출"), "evidence_id": "ev-forged"},
        company_id="00107613",
        filings=filings,
        corpus_as_of=as_of,
        max_top_k=20,
    )
    assert extra.status == "invalid_arguments"
    future = resolve_search_call(
        _call(
            "2035년 확정 매출",
            period=_period(2035),
            report_types=["annual"],
        ),
        company_id="00107613",
        filings=filings,
        corpus_as_of=as_of,
        max_top_k=20,
    )
    assert future.status == "no_matching_filing"


def test_runtime_applies_annual_quarter_half_and_latest_arguments() -> None:
    filings = _filings()
    seen: list[SearchDisclosuresRequest] = []

    def search(request: SearchDisclosuresRequest) -> SearchDisclosuresResult:
        seen.append(request)
        receipt = (request.eligible_receipt_nos or ("none",))[0]
        return _result("본문", receipt)

    annual = _runtime(
        search,
        ScriptedPlanner(
            [_call("2025년 매출", period=_period(2025), report_types=["annual"])]
        ),
        filings,
    )
    events_of(annual, "이수페타시스의 2025년 매출은?")
    assert seen[-1].eligible_receipt_nos == ("20260323000847",)
    assert seen[-1].query == "2025년 매출"

    quarter = _runtime(
        search,
        ScriptedPlanner(
            [
                _call(
                    "2025년 1분기 매출",
                    period=_period(2025, quarter=1),
                    report_types=["quarterly"],
                )
            ]
        ),
        filings,
    )
    events_of(quarter, "이수페타시스의 2025년 1분기 매출은?")
    assert seen[-1].eligible_receipt_nos == ("20250514001051",)

    half = _runtime(
        search,
        ScriptedPlanner(
            [
                _call(
                    "2026년 상반기 매출",
                    period=_period(2026, half=1),
                    report_types=["semiannual"],
                )
            ]
        ),
        filings,
    )
    events_of(half, "이수페타시스 2026년 상반기 매출")
    assert seen[-1].eligible_receipt_nos == ("20260814002283",)

    latest = _runtime(
        search,
        ScriptedPlanner([_call("최근 실적", period={"mode": "latest"})]),
        filings,
    )
    events_of(latest, "이수페타시스 최근 실적은?")
    assert seen[-1].eligible_receipt_nos == ("20260814002283",)


def test_future_target_is_searched_and_confirmed_year_can_miss() -> None:
    filings = _filings()
    seen: list[SearchDisclosuresRequest] = []

    def search(request: SearchDisclosuresRequest) -> SearchDisclosuresResult:
        seen.append(request)
        return _result("중장기 목표는 기재되지 않았습니다.", "20260323000847")

    runtime = _runtime(
        search,
        ScriptedPlanner([_call("2035년 매출 목표", period=None)]),
        filings,
    )
    events = events_of(runtime, "이수페타시스 2035년 매출 목표가 있어?")
    assert seen
    assert events[-1].data["status"] == "completed"

    target_only = _runtime(
        search,
        ScriptedPlanner([_call("2035년 목표", period=None)], uses_llm=True),
        filings,
    )
    target_events = events_of(target_only, "이수페타시스 2035년 목표가 있어?")
    assert len(seen) == 2
    assert target_events[-1].data["status"] == "completed"

    missed = _runtime(
        search,
        ScriptedPlanner(
            [_call("2035년 확정 매출", period=_period(2035), report_types=["annual"])]
        ),
        filings,
    )
    blocked = events_of(missed, "이수페타시스 2035년 확정 매출은?")
    assert len(seen) == 2
    assert blocked[-1].data["status"] == "no_matching_filing"
    assert blocked[-1].data["answer"]["citations"] == []


def test_unindexed_company_is_phrased_from_a_public_observation() -> None:
    seen: list[SearchDisclosuresRequest] = []

    def search(request: SearchDisclosuresRequest) -> SearchDisclosuresResult:
        seen.append(request)
        return _result("본문", "20260323000847")

    planner = ScriptedPlanner(
        [
            _call("2025년 매출", company_id="00126380"),
            {
                "action": "answer_directly",
                "text": "삼성전자는 확인되지만, 현재 색인에는 그 회사의 공시가 없습니다.",
            },
        ],
        uses_llm=True,
    )
    runtime = _runtime(search, planner, _filings())
    events = events_of(runtime, "삼성전자 2025년 매출")
    assert seen == []
    done = events[-1].data
    assert done["status"] == "unsupported_company"
    assert done["abstain_stage"] == "pre_generation"
    assert done["abstain_reason"] == "unsupported_company"
    assert done["resolved_company"]["corp_code"] == "00126380"
    observation = planner.contexts[-1].prior_results[0]
    assert observation["scope"] == "indexed_corpus"
    assert observation["status"] == "unsupported_company"
    assert "DART" not in json.dumps(observation, ensure_ascii=False)
    assert "색인" in done["answer"]["text"]


def test_invalid_arguments_do_not_reach_retrieval() -> None:
    seen: list[SearchDisclosuresRequest] = []

    def search(request: SearchDisclosuresRequest) -> SearchDisclosuresResult:
        seen.append(request)
        return _result("본문", "20260323000847")

    runtime = _runtime(
        search,
        ScriptedPlanner([_call("매출", company_id="00000000")]),
        _filings(),
    )
    events = events_of(runtime, "이수페타시스 매출")
    assert seen == []
    assert events[-1].data["status"] == "unsupported_company"


def test_second_search_can_widen_after_the_first_result() -> None:
    filings = _filings()
    seen: list[SearchDisclosuresRequest] = []

    def search(request: SearchDisclosuresRequest) -> SearchDisclosuresResult:
        seen.append(request)
        receipt = (request.eligible_receipt_nos or ("x",))[0]
        return _result(f"{receipt} 설명", receipt)

    planner = ScriptedPlanner(
        [
            _call("2025년 매출 증가 이유", period=_period(2025), report_types=["annual"]),
            _call("2025년 매출 증가 이유 후속 공시", period={"mode": "latest"}),
        ]
    )
    runtime = _runtime(search, planner, filings)
    events = events_of(runtime, "이수페타시스의 2025년 매출 증가 이유는?")
    assert [list(item.eligible_receipt_nos or ()) for item in seen] == [
        ["20260323000847"],
        ["20260814002283"],
    ]
    assert planner.contexts[1].prior_results[0]["status"] == "success"
    assert planner.contexts[1].question == "이수페타시스의 2025년 매출 증가 이유는?"
    assert "2026-10-02" in planner.contexts[0].request_time["display"]
    assert planner.contexts[0].corpus_as_of == "2026-09-25T10:26:00+09:00"
    assert events[-1].data["status"] == "completed"
    assert len(events[-1].data["answer"]["citations"]) == 1


def test_search_cap_stops_at_two_calls() -> None:
    seen: list[SearchDisclosuresRequest] = []

    def search(request: SearchDisclosuresRequest) -> SearchDisclosuresResult:
        seen.append(request)
        return _result("본문", "20260323000847")

    class _Always:
        uses_llm = False

        def next_action(self, context: PlannerContext):
            del context
            return ScriptedPlanner([_call("실적")]).next_action(
                PlannerContext(
                    question="실적",
                    request_time={
                        "timezone": "Asia/Seoul",
                        "display": "2026-10-02 금요일 10:52:00 KST",
                        "iso": "2026-10-02T10:52:00+09:00",
                        "weekday": "금요일",
                    },
                    corpus_as_of=None,
                    company_id="00107613",
                    company_name="이수페타시스",
                    filings=(),
                    prior_results=(),
                    max_top_k=20,
                )
            )

    runtime = _runtime(search, _Always(), _filings())
    events = events_of(runtime, "이수페타시스 실적")
    assert len(seen) == 2
    assert events[-1].data["status"] == "completed"


def test_raw_query_planner_does_not_invent_a_period() -> None:
    planner = RawQueryPlanner()
    context = PlannerContext(
        question="이수페타시스 최근 실적은?",
        request_time={
            "timezone": "Asia/Seoul",
            "display": "2026-10-02 금요일 10:52:00 KST",
            "iso": "2026-10-02T10:52:00+09:00",
            "weekday": "금요일",
        },
        corpus_as_of="2026-09-25T10:26:24+09:00",
        company_id="00107613",
        company_name="이수페타시스",
        filings=_filings(),
        prior_results=(),
        max_top_k=20,
    )
    action = planner.next_action(context)
    assert action.arguments is not None
    assert action.arguments.query == "이수페타시스 최근 실적은?"
    assert action.arguments.period is None
    assert planner.next_action(context).requests_tool is False


def test_development_cases_and_grader_cover_required_and_forbidden_strategies() -> None:
    cases = load_cases(repo_root() / "evaluation" / "tool_planning" / "dev_cases.json")
    assert len(cases) >= 40
    counts: dict[str, int] = {}
    for case in cases:
        counts[case["category"]] = counts.get(case["category"], 0) + 1
    assert counts["latest"] >= 6
    assert counts["annual"] >= 4
    assert counts["forecast"] >= 4
    annual = next(item for item in cases if item["id"] == "annual-2025-sales")
    good, invalid = parse_calls(
        [_call("2025년 매출", period=_period(2025), report_types=["annual"])]
    )
    assert grade_case(annual, good, invalid_calls=invalid)["pass"] is True
    interim, _invalid = parse_calls(
        [_call("2025년 매출", period=_period(2025, quarter=1), report_types=["quarterly"])]
    )
    assert "annual_vs_interim_error" in grade_case(annual, interim)["errors"]
    quarter = next(item for item in cases if item["id"] == "q1-2025")
    q1, _invalid = parse_calls(
        [_call("1분기", period=_period(2025, quarter=1), report_types=["quarterly"])]
    )
    assert grade_case(quarter, q1)["pass"] is True
    latest = next(item for item in cases if item["id"] == "latest-recent")
    latest_calls, _invalid = parse_calls([_call("최근 실적", period={"mode": "latest"})])
    assert grade_case(latest, latest_calls, latest_period=(2026, 6))["pass"] is True
    old, _invalid = parse_calls([_call("최근 실적", period=_period(2023), report_types=["annual"])])
    assert "latest_semantics_error" in grade_case(latest, old, latest_period=(2026, 6))["errors"]
    why = next(item for item in cases if item["id"] == "why-2025")
    only_2023, _invalid = parse_calls(
        [_call("이유", period=_period(2023), report_types=["annual"])]
    )
    assert "overly_narrow_filter" in grade_case(why, only_2023)["errors"]
    centered, _invalid = parse_calls(
        [_call("이유", period=_period(2025), report_types=["annual"])]
    )
    assert grade_case(why, centered)["pass"] is True
    target = next(item for item in cases if item["id"] == "forecast-2035-target")
    assert "future_target_incorrectly_blocked" in grade_case(target, [])["errors"]
    confirmed = next(item for item in cases if item["id"] == "confirmed-2035")
    assert confirmed["expect_search"] is False
    assert grade_case(confirmed, [], action="answer_directly")["pass"] is True
    confirmed_calls, _invalid = parse_calls(
        [_call("2035년 확정 매출", period=_period(2035), report_types=["annual"])]
    )
    blocked = grade_case(confirmed, confirmed_calls, action="search_disclosures")
    assert "unnecessary_search" in blocked["errors"]
    assert blocked["pass"] is False
    hello = next(item for item in cases if item["id"] == "none-hello")
    searched, _invalid = parse_calls([_call("안녕")])
    assert "unnecessary_search" in grade_case(hello, searched)["errors"]
    ambiguous = next(item for item in cases if item["id"] == "why-that-year")
    assert ambiguous["expect_search"] is False
    assert ambiguous["expect_action"] == "ask_clarification"
    assert grade_case(ambiguous, [], action="ask_clarification")["pass"] is True
    searched_year, _invalid = parse_calls([_call("그해 매출 증가 원인", period={"mode": "latest"})])
    missed = grade_case(ambiguous, searched_year, action="search_disclosures")
    assert missed["pass"] is False
    assert "clarification_missing" in missed["errors"]
    assert "unnecessary_search" in missed["errors"]
    direct = grade_case(ambiguous, [], action="answer_directly")
    assert direct["pass"] is False
    assert "clarification_missing" in direct["errors"]
    assert "direct_answer_missing" not in direct["errors"]


def test_future_realized_fact_can_be_answered_without_search() -> None:
    seen: list[SearchDisclosuresRequest] = []

    def search(request: SearchDisclosuresRequest) -> SearchDisclosuresResult:
        seen.append(request)
        return _result("없음", "20260323000847")

    direct = {
        "action": "answer_directly",
        "text": "2026년 10월 2일 기준으로 2035년 확정 실적은 아직 발생하지 않았습니다.",
        "company_mention": "이수페타시스",
    }
    runtime = _runtime(search, ScriptedPlanner([direct]), _filings())
    events = events_of(runtime, "이수페타시스 2035년 확정 매출이 어떻게 돼?")
    assert seen == []
    assert events[-1].data["action"] == "answer_directly"
    assert events[-1].data["status"] == "completed"
    assert events[-1].data["abstain_stage"] is None
    assert "2035" in events[-1].data["answer"]["text"]
    assert events[-1].data["answer"]["citations"] == []

    target = _runtime(
        search,
        ScriptedPlanner([_call("2035년 매출 목표", period=None)]),
        _filings(),
    )
    target_events = events_of(target, "이수페타시스 2035년 매출 목표가 있어?")
    assert seen
    assert target_events[-1].data["status"] == "completed"
    assert target_events[-1].data["action"] == "search_disclosures"


def test_exhausted_search_budget_still_asks_the_planner_to_phrase() -> None:
    seen: list[SearchDisclosuresRequest] = []

    def search(request: SearchDisclosuresRequest) -> SearchDisclosuresResult:
        seen.append(request)
        return _result("본문", "20260323000847")

    planner = ScriptedPlanner(
        [
            _call("2010년 확정 매출", period=_period(2010), report_types=["annual"]),
            _call("2010년 매출", period=_period(2010), report_types=["annual"]),
            {
                "action": "answer_directly",
                "text": "현재 색인에서는 2010년 공시를 찾지 못했습니다.",
            },
        ],
        uses_llm=True,
    )
    events = events_of(
        _runtime(search, planner, _filings()),
        "이수페타시스 2010년 확정 매출은?",
    )
    assert seen == []
    done = events[-1].data
    assert done["action"] == "answer_directly"
    assert done["status"] == "no_matching_filing"
    assert done["abstain_stage"] == "pre_generation"
    assert done["abstain_reason"] == "no_matching_filing"
    assert "색인" in done["answer"]["text"]
    observation = planner.contexts[-1].prior_results[-1]
    assert observation["scope"] == "indexed_corpus"
    assert observation["status"] == "no_matching_filing"
    assert "DART" not in json.dumps(observation, ensure_ascii=False)


def test_tool_failure_is_a_public_observation_for_the_planner() -> None:
    seen: list[SearchDisclosuresRequest] = []

    def search(request: SearchDisclosuresRequest) -> SearchDisclosuresResult:
        seen.append(request)
        return _result("본문", "20260323000847")

    missed = ScriptedPlanner(
        [
            _call("2035년 확정 매출", period=_period(2035), report_types=["annual"]),
            {
                "action": "answer_directly",
                "text": "현재 색인에서 확인 가능한 공시에는 2035년 확정 매출이 없습니다.",
            },
        ]
    )
    events = events_of(
        _runtime(search, missed, _filings()),
        "이수페타시스 2035년 확정 매출은?",
    )
    assert seen == []
    assert missed.contexts[1].prior_results[0]["status"] == "no_matching_filing"
    assert missed.contexts[1].prior_results[0]["scope"] == "indexed_corpus"
    assert "DART" not in json.dumps(missed.contexts[1].prior_results[0], ensure_ascii=False)
    assert events[-1].data["status"] == "no_matching_filing"
    assert events[-1].data["action"] == "answer_directly"
    assert events[-1].data["abstain_stage"] == "pre_generation"
    assert events[-1].data["abstain_reason"] == "no_matching_filing"
    assert "색인" in events[-1].data["answer"]["text"]

    invalid = ScriptedPlanner(
        [
            _call("매출", company_id="00000000"),
            {
                "action": "answer_directly",
                "text": "검색 조건을 확인하지 못했습니다. 기간을 다시 알려주세요.",
            },
        ]
    )
    invalid_events = events_of(
        _runtime(search, invalid, _filings()),
        "이수페타시스 매출",
    )
    blob = json.dumps(invalid.contexts[1].prior_results[0], ensure_ascii=False)
    assert "Traceback" not in blob
    assert "company_id is not" not in blob
    assert invalid_events[-1].data["status"] == "unsupported_company"
    assert "company_id is not" not in invalid_events[-1].data["answer"]["text"]
    assert seen == []

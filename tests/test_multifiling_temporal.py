"""Multi-filing eligibility, request time, and the three failure modes."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pytest
from test_agent_product import events_of, registry, sample_hit
from test_grounded_answer import _output

from disclosure_agent.agent.catalog import DenseD2aBinding, repo_root
from disclosure_agent.agent.clock import SEOUL, request_time_context
from disclosure_agent.agent.corpus import plan_corpus, product_index_dir
from disclosure_agent.agent.evidence import EvidenceBuilder
from disclosure_agent.agent.llm import OpenAIAnswerGenerator
from disclosure_agent.agent.record import RunLog
from disclosure_agent.agent.runtime import AgentRuntime
from disclosure_agent.agent.search import (
    DenseD2aIndex,
    SearchDisclosuresRequest,
    SearchDisclosuresResult,
    search_disclosures,
)
from disclosure_agent.agent.temporal import FilingPeriod
from disclosure_agent.agent.tool_args import (
    ResolvedSearch,
    SearchToolArguments,
    select_filings,
)
from disclosure_agent.retrieval_unit import D2A_VERSION


def _filing(
    receipt: str,
    name: str,
    received: str,
) -> FilingPeriod:
    from disclosure_agent.agent.temporal import filing_period_from_listing

    return filing_period_from_listing(
        {"report_nm": name, "rcept_no": receipt, "rcept_dt": received, "rm": ""}
    )


def _corpus() -> tuple[FilingPeriod, ...]:
    return (
        _filing("20241114002025", "분기보고서 (2024.09)", "20241114"),
        _filing("20250320001435", "사업보고서 (2024.12)", "20250320"),
        _filing("20250514001051", "분기보고서 (2025.03)", "20250514"),
        _filing("20260323000847", "사업보고서 (2025.12)", "20260323"),
        _filing("20260515002006", "분기보고서 (2026.03)", "20260515"),
        _filing("20260814002283", "반기보고서 (2026.06)", "20260814"),
    )


def _resolved(period: dict[str, object] | None) -> ResolvedSearch:
    arguments = SearchToolArguments.model_validate(
        {
            "query": "실적",
            "company_id": "00107613",
            "period": period,
            "top_k": 20,
        }
    )
    return ResolvedSearch(
        arguments=arguments,
        filing_date_from=None,
        filing_date_to=None,
        as_of=None,
        top_k=20,
    )


def test_latest_filter_uses_max_period_inside_the_corpus_clock() -> None:
    filings = _corpus()
    as_of = datetime(2026, 6, 1, tzinfo=SEOUL)
    recent = select_filings(filings, _resolved({"mode": "latest"}), corpus_as_of=as_of)
    assert [item.receipt_no for item in recent] == ["20260515002006"]
    business = select_filings(
        filings,
        _resolved({"mode": "latest", "year_from": None}),
        corpus_as_of=as_of,
    )
    assert [item.receipt_no for item in business] == ["20260515002006"]
    full = select_filings(
        filings,
        _resolved({"mode": "latest"}),
        corpus_as_of=datetime(2026, 9, 25, tzinfo=SEOUL),
    )
    assert [item.receipt_no for item in full] == ["20260814002283"]


def test_metadata_filters_follow_tool_arguments_not_question_wording() -> None:
    filings = _corpus()
    annual = select_filings(
        filings,
        _resolved({"year": 2025, "mode": "exact"}),
        corpus_as_of=None,
    )
    assert [item.receipt_no for item in annual] == ["20260323000847", "20250514001051"]
    only_annual = select_filings(
        filings,
        ResolvedSearch(
            arguments=SearchToolArguments.model_validate(
                {
                    "query": "2025년 매출",
                    "company_id": "00107613",
                    "period": {"year": 2025, "mode": "exact"},
                    "report_types": ["annual"],
                    "top_k": 20,
                }
            ),
            filing_date_from=None,
            filing_date_to=None,
            as_of=None,
            top_k=20,
        ),
        corpus_as_of=None,
    )
    assert [item.receipt_no for item in only_annual] == ["20260323000847"]
    future = select_filings(
        filings,
        _resolved({"year": 2035, "mode": "exact"}),
        corpus_as_of=None,
    )
    assert future == ()
    widened = select_filings(filings, _resolved(None), corpus_as_of=None)
    assert len(widened) == len(filings)


def test_request_time_is_seoul_and_does_not_rewrite_the_question(tmp_path: object) -> None:
    from pathlib import Path

    assert isinstance(tmp_path, Path)
    raw = "이수페타시스 최근 실적 어때?"
    moment = datetime(2026, 10, 1, 11, 56, tzinfo=UTC)
    rendered = request_time_context(moment)
    assert rendered["timezone"] == "Asia/Seoul"
    assert rendered["weekday"] == "목요일"
    assert rendered["display"] == "2026-10-01 목요일 20:56:00 KST"
    seen: list[SearchDisclosuresRequest] = []

    def search(request: SearchDisclosuresRequest) -> SearchDisclosuresResult:
        seen.append(request)
        return SearchDisclosuresResult(
            hits=[sample_hit("최근 반기 매출 설명입니다.")],
            retriever_type="dense",
            representation_version=D2A_VERSION,
            embedding_model="BAAI/bge-m3",
            company_id="00107613",
            corp_code="00107613",
        )

    evidence = EvidenceBuilder().build(
        [sample_hit("최근 반기 매출 설명입니다.")],
        company_name="이수페타시스",
    )

    class _Client:
        model = "gpt-custom"
        sdk_version = "test-sdk"
        inputs: list[str] = []

        def complete(self, user_input: str) -> object:
            self.inputs.append(user_input)
            return _output(evidence[0].evidence_id, answer="최근 반기 기준으로 답합니다.")

    client = _Client()
    log = RunLog(tmp_path / "clock.jsonl")
    runtime = AgentRuntime(
        registry=registry(),
        search=search,
        evidence_builder=EvidenceBuilder(),
        answer_generator=OpenAIAnswerGenerator(client),  # type: ignore[arg-type]
        run_log=log,
        filings=_corpus(),
        corpus_as_of=datetime(2026, 9, 25, tzinfo=SEOUL),
        clock=lambda: moment,
    )
    events_of(runtime, raw)
    record = json.loads(log.path.read_text(encoding="utf-8"))
    sent = json.loads(client.inputs[0])
    assert record["raw_user_query"] == raw
    assert sent["question"] == raw
    assert sent["request_time"]["display"] == "2026-10-01 목요일 20:56:00 KST"
    assert sent["corpus_as_of"] == "2026-09-25T00:00:00+09:00"
    assert seen[0].query == raw
    assert set(seen[0].eligible_receipt_nos or ()) == {
        item.receipt_no for item in _corpus()
    }


def test_regression_separates_temporal_retrieval_and_context(tmp_path: object) -> None:
    from pathlib import Path

    assert isinstance(tmp_path, Path)
    filings = _corpus()

    def run(question: str, hits: list[object] | None, *, budget: int = 10**9):
        calls: list[SearchDisclosuresRequest] = []
        client_inputs: list[str] = []

        def search(request: SearchDisclosuresRequest) -> SearchDisclosuresResult:
            calls.append(request)
            if hits is None:
                chosen = [
                    _hit(receipt, index)
                    for index, receipt in enumerate(request.eligible_receipt_nos or (), start=1)
                ]
            else:
                chosen = list(hits)
            return SearchDisclosuresResult(
                hits=chosen,  # type: ignore[arg-type]
                retriever_type="dense",
                representation_version=D2A_VERSION,
                embedding_model="BAAI/bge-m3",
                company_id="00107613",
                corp_code="00107613",
            )

        class _Client:
            model = "gpt-custom"
            sdk_version = "test-sdk"

            def complete(self, user_input: str) -> object:
                client_inputs.append(user_input)
                payload = json.loads(user_input)
                rows = payload["evidence"]
                if not rows:
                    return _output(answer="빈 근거", abstain=True)
                return _output(rows[0]["evidence_id"], answer="공시에 적힌 내용입니다.")

        slug = str(len(list(tmp_path.iterdir())))
        log = RunLog(tmp_path / f"run-{slug}.jsonl")
        runtime = AgentRuntime(
            registry=registry(),
            search=search,
            evidence_builder=EvidenceBuilder(),
            answer_generator=OpenAIAnswerGenerator(_Client()),  # type: ignore[arg-type]
            run_log=log,
            filings=filings,
            context_token_budget=budget,
            clock=lambda: datetime(2026, 10, 1, 11, 56, tzinfo=UTC),
        )
        events = events_of(runtime, question)
        record = json.loads(log.path.read_text(encoding="utf-8"))
        return events, record, calls, client_inputs

    events, record, calls, inputs = run("이수페타시스의 2025년 매출이 얼마야?", None)
    assert record["temporal_intent"] is None
    assert record["planner"] == "raw_query"
    assert record["search_called"] is True
    assert calls[0].query == "이수페타시스의 2025년 매출이 얼마야?"
    assert calls[0].fiscal_year is None
    assert set(record["eligible_receipt_nos"]) == {item.receipt_no for item in filings}
    assert inputs
    assert events[-1].data["status"] == "completed"

    _events, history, history_calls, history_inputs = run(
        "이수페타시스의 2025년 매출 증가 이유가 뭐야?",
        None,
    )
    assert history["temporal_intent"] is None
    assert set(history["eligible_receipt_nos"]) == {item.receipt_no for item in filings}
    assert history_calls[0].fiscal_year is None
    assert history_calls[0].query == "이수페타시스의 2025년 매출 증가 이유가 뭐야?"
    assert history_inputs

    _events, recent, recent_calls, recent_inputs = run("이수페타시스의 최근 실적은?", None)
    assert recent["temporal_intent"] is None
    assert recent_calls[0].query == "이수페타시스의 최근 실적은?"
    assert set(recent_calls[0].eligible_receipt_nos or ()) == {item.receipt_no for item in filings}
    assert recent_inputs

    _events, business, business_calls, _inputs = run("이수페타시스의 주요 사업이 뭐야?", None)
    assert business["temporal_intent"] is None
    assert set(business_calls[0].eligible_receipt_nos or ()) == {
        item.receipt_no for item in filings
    }

    future_events, future, future_calls, future_inputs = run(
        "2035년 이수페타시스 확정 매출 알려줘",
        None,
    )
    assert future["search_called"] is True
    assert future_calls[0].query == "2035년 이수페타시스 확정 매출 알려줘"
    assert future["generator_type"] != "temporal_scope"
    assert future_inputs
    assert future_events[-1].data["status"] == "completed"

    _events, forecast, forecast_calls, forecast_inputs = run(
        "이수페타시스가 2035년 매출 목표를 제시했어?",
        None,
    )
    assert forecast["search_called"] is True
    assert len(forecast_calls[0].eligible_receipt_nos or ()) == len(filings)
    assert forecast_inputs
    assert forecast["generator_type"] == "openai"

    _events, charter, _calls, charter_inputs = run(
        "이수페타시스가 2024년에 정관상 배당 기준일을 바꾼 이유는 뭐야?",
        None,
    )
    assert charter["temporal_intent"] is None
    assert set(charter["eligible_receipt_nos"]) == {item.receipt_no for item in filings}
    assert charter_inputs

    empty_events, empty, empty_calls, empty_inputs = run(
        "이수페타시스의 최근 실적은?",
        [],
    )
    assert empty["search_called"] is True
    assert empty["temporal_intent"] is None
    assert empty_calls
    assert empty_inputs == []
    assert empty["generator_type"] == "answerability_guard"
    assert empty_events[-1].data["status"] == "retrieval_no_result"

    first = _hit("20260814002283", 1)
    second = _hit("20260323000847", 2)
    context_events, context, _calls, context_inputs = run(
        "이수페타시스의 주요 사업이 뭐야?",
        [first, second],
        budget=1,
    )
    assert context["retrieved_filing_counts"] == {
        "20260814002283": 1,
        "20260323000847": 1,
    }
    assert context["context_filing_counts"] == {}
    assert context_inputs == []
    assert context["generator_type"] == "context_budget"
    assert context_events[-1].data["status"] == "abstained"


def test_dense_search_filters_receipts_before_ranking() -> None:
    older = _unit("20250320001435", "2024년 사업 설명")
    newer = _unit("20260814002283", "2026년 반기 실적")
    matrix = np.eye(2, dtype=np.float32)
    binding = DenseD2aBinding(
        corp_code="00107613",
        company_name="이수페타시스",
        units_path=repo_root() / "missing-units.jsonl",
        embeddings_path=repo_root() / "missing-embeddings.f32",
        elements_path=None,
        receipt_no="20260323000847",
        document_name="사업보고서",
        representation_version=D2A_VERSION,
        embedding_model="BAAI/bge-m3",
    )
    index = DenseD2aIndex(units=[older, newer], matrix=matrix, binding=binding)

    class _Encoder:
        def encode(self, text: str) -> object:
            del text
            return matrix[1]

    result = search_disclosures(
        SearchDisclosuresRequest(
            query="최근 실적",
            company_id="00107613",
            corp_code="00107613",
            eligible_receipt_nos=("20250320001435",),
            top_k=5,
        ),
        index,
        _Encoder(),
        binding,
    )
    assert [hit.receipt_no for hit in result.hits] == ["20250320001435"]
    assert result.hits[0].report_name == "사업보고서 (2024.12)"
    assert result.hits[0].filing_period == "2024.12"
    assert result.hits[0].source_sha256 == "abc"


def test_stored_isu_filings_keep_the_partial_correction_out() -> None:
    root = repo_root()
    raw = root / "data" / "raw" / "dart"
    if not raw.is_dir():
        pytest.skip("local raw filings are not in this checkout")
    plan = plan_corpus(root)
    on_disk = {path.name for path in raw.iterdir() if path.is_dir() and path.name.isdigit()}
    assert {row.receipt_no for row in plan.rows} == on_disk
    by_receipt = {row.receipt_no: row for row in plan.rows}
    correction = by_receipt["20250321001007"]
    assert correction.downloaded is True
    assert correction.complete is True
    assert correction.parse_status == "success"
    assert correction.index_eligible is True
    original = by_receipt["20250320001435"]
    assert original.report_name == "사업보고서 (2024.12)"
    assert original.index_eligible is True
    assert original.remark == "정연"
    original_zip = raw / "20250320001435" / "versions"
    assert original_zip.is_dir()
    assert any(original_zip.rglob("document.zip"))
    annual_2025 = by_receipt["20260323000847"]
    assert annual_2025.index_eligible is True
    assert annual_2025.excluded_members
    half_2026 = by_receipt["20260814002283"]
    assert half_2026.report_name == "반기보고서 (2026.06)"
    assert half_2026.index_eligible is True
    eligible = {row.receipt_no for row in plan.rows if row.index_eligible}
    assert "20250321001007" in eligible
    assert plan.corpus_as_of is not None
    assert plan.corpus_as_of.tzinfo is not None
    manifest = product_index_dir(root) / "corpus.json"
    if not manifest.is_file():
        return
    built = json.loads(manifest.read_text(encoding="utf-8"))
    indexed = {member["receipt_no"] for member in built["members"]}
    isu_eligible = {
        row.receipt_no
        for row in plan.rows
        if row.index_eligible and row.corp_code == built["corp_code"]
    }
    assert indexed <= isu_eligible
    assert "20250321001007" not in indexed
    assert built["retrieval_unit_count"] == sum(
        member["retrieval_unit_count"] for member in built["members"]
    )


def _hit(receipt: str, index: int):
    from dataclasses import replace

    return replace(
        sample_hit(f"{receipt} 근거 문장 {index}"),
        retrieval_unit_id=f"unit-{receipt}",
        canonical_chunk_id=f"chunk-{receipt}",
        rank=index,
        receipt_no=receipt,
        report_name="정기공시",
        filing_period="2026.06",
        receipt_date="20260814",
    )


def _unit(receipt: str, text: str) -> dict[str, object]:
    return {
        "retrieval_unit_id": f"id-{receipt}",
        "canonical_chunk_id": f"chunk-{receipt}",
        "canonical_ordinal": 1,
        "unit_index": 0,
        "representation_version": D2A_VERSION,
        "text": text,
        "source_refs": [],
        "source_element_ids": [],
        "section_path": ["II. 사업의 내용"],
        "receipt_no": receipt,
        "document_name": "사업보고서",
        "report_name": "사업보고서 (2024.12)",
        "filing_period": "2024.12",
        "receipt_date": "20250320",
        "source_sha256": "abc",
        "corp_code": "00107613",
    }

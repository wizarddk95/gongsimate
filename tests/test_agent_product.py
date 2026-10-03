"""Product path: company resolution, Dense-D2a search, citations, and SSE."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from disclosure_agent.agent.api import create_app
from disclosure_agent.agent.catalog import load_dense_d2a_binding
from disclosure_agent.agent.citation import Citation, CitationValidator
from disclosure_agent.agent.company import CompanyRecord, CompanyRegistry
from disclosure_agent.agent.errors import CitationValidationFailure, RetrievalFailure
from disclosure_agent.agent.evidence import EvidenceBuilder
from disclosure_agent.agent.runtime import AgentRuntime
from disclosure_agent.agent.search import (
    DenseD2aIndex,
    SearchDisclosuresRequest,
    SearchDisclosuresResult,
    SearchHit,
    search_disclosures,
)
from disclosure_agent.agent.settings import AgentLimits
from disclosure_agent.agent.sse import StreamEvent
from disclosure_agent.dense import EMBEDDING_DIMENSION, exact_normalized_search
from disclosure_agent.retrieval_unit import D2A_VERSION

ISU = CompanyRecord("00107613", "이수페타시스", "007660", "ISU Petasys")
SEC = CompanyRecord("00126380", "삼성전자", "005930", "")
SDI = CompanyRecord("00126362", "삼성SDI", "006400", "")


def registry() -> CompanyRegistry:
    return CompanyRegistry.from_records(
        [ISU, SEC, SDI],
        indexed_corp_codes={ISU.corp_code},
    )


def sample_hit(text: str = "당사는 인쇄회로기판(PCB)을 전문으로 생산하고 있습니다.") -> SearchHit:
    return SearchHit(
        retrieval_unit_id="unit-1",
        score=0.8,
        rank=1,
        text=text,
        canonical_chunk_id="chunk-1",
        unit_index=0,
        source_element_ids=["el-1"],
        source_refs=[
            {
                "document_id": "m0:20260323000847.xml",
                "element_id": "el-1",
                "char_start": 0,
                "char_end": 12,
                "kind": "paragraph",
                "role": "body",
            }
        ],
        receipt_no="20260323000847",
        document_name="사업보고서",
        section_path=["II. 사업의 내용", "1. 사업의 개요"],
        source_locator=[
            {
                "document_id": "m0:20260323000847.xml",
                "element_id": "el-1",
                "char_start": 0,
                "char_end": 12,
            }
        ],
    )


def sample_result(text: str | None = None) -> SearchDisclosuresResult:
    hit = sample_hit() if text is None else sample_hit(text)
    return SearchDisclosuresResult(
        hits=[hit],
        retriever_type="dense",
        representation_version=D2A_VERSION,
        embedding_model="BAAI/bge-m3",
        company_id=ISU.corp_code,
        corp_code=ISU.corp_code,
    )


def make_runtime(
    search_calls: list[SearchDisclosuresRequest],
    *,
    limits: AgentLimits | None = None,
    search_result: SearchDisclosuresResult | None = None,
    search_error: Exception | None = None,
    generator: object | None = None,
) -> AgentRuntime:
    def search(request: SearchDisclosuresRequest) -> SearchDisclosuresResult:
        search_calls.append(request)
        if search_error is not None:
            raise search_error
        return search_result or sample_result()

    kwargs: dict[str, object] = {
        "registry": registry(),
        "search": search,
        "evidence_builder": EvidenceBuilder(),
        "limits": limits or AgentLimits(),
    }
    if generator is not None:
        kwargs["answer_generator"] = generator
    return AgentRuntime(**kwargs)  # type: ignore[arg-type]


def events_of(runtime: AgentRuntime, message: str, **kwargs: object) -> list[StreamEvent]:
    return list(runtime.stream(message, **kwargs))  # type: ignore[arg-type]


def parse_sse(raw: str) -> list[tuple[str, dict]]:
    parsed = []
    for block in raw.split("\n\n"):
        if not block.strip():
            continue
        event_name = None
        data = None
        for line in block.split("\n"):
            if line.startswith("event:"):
                event_name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = json.loads(line.split(":", 1)[1].strip())
        parsed.append((event_name, data))
    return parsed


def test_exact_company_resolves() -> None:
    found = registry().resolve_mention("이수페타시스")
    assert found.status == "resolved"
    assert found.company_id == "00107613"
    assert found.corp_code == "00107613"
    assert found.company_name == "이수페타시스"
    parenthetical = registry().resolve_mentions(
        registry().extract_mentions("(주)이수페타시스의 매출")
    )
    assert parenthetical.status == "resolved"
    assert parenthetical.corp_code == "00107613"


def test_unlisted_exact_name_does_not_hide_longer_companies() -> None:
    books = CompanyRegistry.from_records(
        [
            CompanyRecord("00893765", "삼성", "", ""),
            SEC,
            SDI,
        ],
        indexed_corp_codes=set(),
    )
    found = books.resolve_mention("삼성")
    assert found.status == "ambiguous"
    assert found.company_id is None
    names = {company.corp_name for company in found.candidates}
    assert {"삼성", "삼성전자", "삼성SDI"} <= names
    found = registry().resolve_mentions(registry().extract_mentions("삼성 실적 알려줘"))
    assert found.status == "ambiguous"
    assert found.company_id is None
    names = {company.corp_name for company in found.candidates}
    assert names == {"삼성전자", "삼성SDI"}
    assert found.candidate_total == 2


def test_unindexed_name_does_not_veto_indexed_company() -> None:
    nowadays = CompanyRecord("01719318", "요즘", "", "")
    books = CompanyRegistry.from_records(
        [ISU, nowadays, SEC],
        indexed_corp_codes={ISU.corp_code},
    )
    found = books.resolve_mentions(books.extract_mentions("이수페타시스 요즘 실적 어때?"))
    assert found.status == "resolved"
    assert found.corp_code == ISU.corp_code
    assert found.company_name == "이수페타시스"
    alone = books.resolve_mentions(books.extract_mentions("요즘 실적"))
    assert alone.status == "unsupported"
    assert alone.corp_code == nowadays.corp_code
    both = CompanyRegistry.from_records(
        [ISU, SEC],
        indexed_corp_codes={ISU.corp_code, SEC.corp_code},
    )
    mixed = both.resolve_mentions(both.extract_mentions("이수페타시스 삼성전자 매출"))
    assert mixed.status == "ambiguous"
    assert {company.corp_code for company in mixed.candidates} == {
        ISU.corp_code,
        SEC.corp_code,
    }


def test_spaced_names_resolve_without_a_blacklist() -> None:
    books = CompanyRegistry.from_records(
        [
            ISU,
            SEC,
            CompanyRecord("00126955", "삼양식품", "003230", "Samyang Foods"),
            CompanyRecord("01719318", "요즘", "", ""),
        ],
        indexed_corp_codes={ISU.corp_code, SEC.corp_code, "00126955"},
    )
    samsung = books.resolve_mentions(books.extract_mentions("삼성 전자 2025년 매출"))
    assert samsung.status == "resolved"
    assert samsung.company_name == "삼성전자"
    assert samsung.corp_code == "00126380"
    foods = books.resolve_mentions(books.extract_mentions("삼양 식품 최근 실적"))
    assert foods.company_name == "삼양식품"
    assert foods.corp_code == "00126955"
    mixed = books.resolve_mentions(books.extract_mentions("이수페타시스 요즘 실적 어때?"))
    assert mixed.company_name == "이수페타시스"
    alone = books.resolve_mentions(books.extract_mentions("요즘 실적 어때?"))
    assert alone.status == "unsupported"
    assert alone.company_name == "요즘"
    english = books.resolve_mentions(books.extract_mentions("ISU Petasys 매출"))
    assert english.company_name == "이수페타시스"


def test_unknown_company_and_unsupported_company() -> None:
    missing = registry().resolve_mention("없는기업에이비씨")
    assert missing.status == "not_found"
    unsupported = registry().resolve_mention("삼성전자")
    assert unsupported.status == "unsupported"
    assert unsupported.corp_code == "00126380"
    assert unsupported.company_id == "00126380"


def test_ambiguous_question_does_not_search() -> None:
    calls: list[SearchDisclosuresRequest] = []
    found = events_of(make_runtime(calls), "삼성 실적 알려줘")
    assert calls == []
    completed = found[-1]
    assert completed.event == "run.completed"
    assert completed.data["action"] == "ask_clarification"
    assert completed.data["status"] == "company_ambiguous"
    assert "searching_disclosures" not in _statuses(found)
    for candidate in completed.data["clarification"]["candidates"]:
        assert "corp_code" not in candidate
        assert "company_id" not in candidate
        assert "company_name" in candidate


def test_general_question_does_not_search() -> None:
    calls: list[SearchDisclosuresRequest] = []
    runtime = make_runtime(calls)
    greeting = events_of(runtime, "안녕")
    assert calls == []
    assert greeting[-1].data["action"] == "answer_directly"
    calls.clear()
    definition = events_of(runtime, "RAG가 뭐야?")
    assert calls == []
    assert definition[-1].data["action"] == "answer_directly"
    assert definition[-1].data["answer"]["citations"] == []


def test_company_disclosure_question_searches() -> None:
    calls: list[SearchDisclosuresRequest] = []
    found = events_of(make_runtime(calls), "이수페타시스의 주요 사업이 뭐야?")
    assert len(calls) == 1
    assert calls[0].company_id == "00107613"
    assert calls[0].corp_code == "00107613"
    assert calls[0].query == "이수페타시스의 주요 사업이 뭐야?"
    assert found[-1].data["action"] == "search_disclosures"
    assert found[-1].data["status"] == "completed"
    citations = found[-1].data["answer"]["citations"]
    assert len(citations) == 1
    assert citations[0]["evidence_id"].startswith("ev-")
    assert found[-1].data["evidence"][0]["receipt_no"] == "20260323000847"
    assert found[-1].data["evidence"][0]["document_name"] == "사업보고서"


def test_search_call_limit() -> None:
    calls: list[SearchDisclosuresRequest] = []
    found = events_of(
        make_runtime(calls, limits=AgentLimits(max_search_calls=0, max_llm_calls=4)),
        "이수페타시스의 주요 사업이 뭐야?",
    )
    assert calls == []
    assert found[-1].event == "error"
    assert found[-1].data["code"] == "search_limit_exceeded"
    assert not any(event.event == "run.completed" for event in found)
    assert not any(event.event == "answer.delta" for event in found)


def test_llm_call_limit() -> None:
    calls: list[SearchDisclosuresRequest] = []
    found = events_of(
        make_runtime(calls, limits=AgentLimits(max_search_calls=2, max_llm_calls=1)),
        "이수페타시스의 주요 사업이 뭐야?",
    )
    assert len(calls) == 1
    assert found[-1].event == "error"
    assert found[-1].data["code"] == "llm_limit_exceeded"
    assert not any(event.event == "answer.delta" for event in found)
    assert not any(event.event == "run.completed" for event in found)


def test_search_disclosures_uses_dense_d2a(monkeypatch: pytest.MonkeyPatch) -> None:
    binding = load_dense_d2a_binding()
    assert binding.representation_version == D2A_VERSION
    assert binding.embedding_model == "BAAI/bge-m3"
    assert binding.units_path.parent.name == "isu_petasys_v0_3_dense_d2a"
    lines = binding.units_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2869
    if binding.embeddings_path.is_file():
        assert binding.embeddings_path.stat().st_size == 2869 * EMBEDDING_DIMENSION * 4
    units = [json.loads(line) for line in lines[:3]]
    matrix = np.eye(3, dtype=np.float32)
    index = DenseD2aIndex(units=units, matrix=matrix, binding=binding)
    seen: dict[str, bool] = {}

    def spy(doc_matrix: object, query_vector: object, chunks: list[dict]) -> list[dict]:
        seen["called"] = True
        return exact_normalized_search(doc_matrix, query_vector, chunks)

    monkeypatch.setattr("disclosure_agent.agent.search.exact_normalized_search", spy)

    class _Encoder:
        def encode(self, text: str) -> object:
            return matrix[0]

    result = search_disclosures(
        SearchDisclosuresRequest(
            query="주요 사업",
            company_id=binding.corp_code,
            corp_code=binding.corp_code,
            fiscal_year=2035,
            top_k=2,
        ),
        index,
        _Encoder(),
        binding,
    )
    assert seen["called"] is True
    assert result.retriever_type == "dense"
    assert result.representation_version == D2A_VERSION
    assert result.embedding_model == "BAAI/bge-m3"
    assert result.fiscal_year == 2035
    assert result.hits[0].retrieval_unit_id == units[0]["retrieval_unit_id"]
    assert len(result.hits) == 2


def test_search_output_omits_gold_metadata() -> None:
    binding = load_dense_d2a_binding()
    unit = json.loads(binding.units_path.read_text(encoding="utf-8").splitlines()[0])
    unit["gold_spans"] = ["leak"]
    unit["evidence_class"] = "positive"
    unit["source_refs"][0]["gold"] = "leak-ref"
    matrix = np.eye(1, dtype=np.float32)
    index = DenseD2aIndex(units=[unit], matrix=matrix, binding=binding)

    class _Encoder:
        def encode(self, text: str) -> object:
            return matrix[0]

    result = search_disclosures(
        SearchDisclosuresRequest(
            query="사업",
            company_id=binding.corp_code,
            corp_code=binding.corp_code,
            top_k=1,
        ),
        index,
        _Encoder(),
        binding,
    )
    blob = json.dumps(result.to_dict(), ensure_ascii=False)
    assert "gold" not in blob
    assert "evidence_class" not in blob
    assert "leak" not in blob
    assert "question_id" not in blob


def test_citation_accepts_current_evidence() -> None:
    validator = CitationValidator()
    validator.validate([Citation("1", "ev-current")], {"ev-current"})


def test_citation_rejects_unknown_evidence() -> None:
    validator = CitationValidator()
    with pytest.raises(CitationValidationFailure):
        validator.validate([Citation("1", "ev-missing")], {"ev-current"})


def test_citation_rejects_other_run_evidence() -> None:
    validator = CitationValidator()
    with pytest.raises(CitationValidationFailure):
        validator.validate([Citation("1", "ev-run-a")], {"ev-run-b"})


def test_ask_content_type_is_event_stream() -> None:
    calls: list[SearchDisclosuresRequest] = []
    client = TestClient(create_app(make_runtime(calls)))
    response = client.post("/ask", json={"message": "안녕", "company_id": "00107613"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    parsed = parse_sse(response.text)
    assert parsed[0][0] == "run.started"
    assert parsed[-1][0] == "run.completed"
    assert calls == []


def test_sse_event_order() -> None:
    calls: list[SearchDisclosuresRequest] = []
    client = TestClient(create_app(make_runtime(calls)))
    response = client.post("/ask", json={"message": "이수페타시스의 주요 사업이 뭐야?"})
    parsed = parse_sse(response.text)
    names = [name for name, _data in parsed]
    assert names[0] == "run.started"
    assert names[-1] == "run.completed"
    statuses = [data["status"] for name, data in parsed if name == "agent.status"]
    assert statuses == [
        "resolving_company",
        "searching_disclosures",
        "generating_answer",
    ]
    assert names.index("answer.delta") > names.index("agent.status")
    search_at = next(
        index
        for index, (name, data) in enumerate(parsed)
        if name == "agent.status" and data["status"] == "searching_disclosures"
    )
    delta_at = names.index("answer.delta")
    citation_at = names.index("citation")
    assert search_at < delta_at < citation_at < len(names) - 1
    assert "error" not in names


def test_answer_delta_starts_after_retrieval() -> None:
    order: list[str] = []

    def search(request: SearchDisclosuresRequest) -> SearchDisclosuresResult:
        order.append("search")
        return sample_result()

    runtime = AgentRuntime(
        registry=registry(),
        search=search,
        evidence_builder=EvidenceBuilder(),
    )
    generator = runtime.stream("이수페타시스의 주요 사업이 뭐야?")
    seen_delta = False
    for event in generator:
        if event.event == "answer.delta":
            assert order == ["search"]
            seen_delta = True
            break
    assert seen_delta is True


def test_stream_ends_with_one_terminal_event() -> None:
    calls: list[SearchDisclosuresRequest] = []
    success = events_of(make_runtime(calls), "안녕")
    assert success[-1].event == "run.completed"
    assert sum(event.event in {"run.completed", "error"} for event in success) == 1

    failed = events_of(
        make_runtime([], search_error=RetrievalFailure("index unreadable")),
        "이수페타시스의 주요 사업이 뭐야?",
    )
    assert failed[-1].event == "error"
    assert failed[-1].data["code"] == "retrieval_failure"
    assert sum(event.event in {"run.completed", "error"} for event in failed) == 1
    assert not any(event.event == "answer.delta" for event in failed)


def test_client_supplied_company_id_does_not_select_a_company() -> None:
    calls: list[SearchDisclosuresRequest] = []
    client = TestClient(create_app(make_runtime(calls)))
    response = client.post(
        "/ask",
        json={"message": "매출이 왜 증가했어?", "company_id": "00107613"},
    )
    parsed = parse_sse(response.text)
    assert calls == []
    assert parsed[-1][1]["action"] == "ask_clarification"
    assert parsed[-1][1]["status"] == "company_not_found"


def _statuses(events: list[StreamEvent]) -> list[str]:
    return [event.data["status"] for event in events if event.event == "agent.status"]


def test_runtime_record_links_tool_evidence_and_citation(tmp_path: Path) -> None:
    from disclosure_agent.agent.record import RunLog

    calls: list[SearchDisclosuresRequest] = []
    log = RunLog(tmp_path / "runs.jsonl")
    runtime = AgentRuntime(
        registry=registry(),
        search=lambda request: (calls.append(request) or sample_result()),
        evidence_builder=EvidenceBuilder(),
        run_log=log,
    )
    events_of(runtime, "이수페타시스의 주요 사업이 뭐야?")
    record = json.loads(log.path.read_text(encoding="utf-8").splitlines()[0])
    assert record["agent_action"] == "search_disclosures"
    tool = record["tool_calls"][0]
    assert tool["name"] == "search_disclosures"
    assert tool["retrieval_unit_ids"] == ["unit-1"]
    assert record["evidence_ids"]
    assert record["citation_ids"] == ["1"]
    assert record["evidence_ids"][0] == events_evidence(record)
    assert record["error"] is None


def events_evidence(record: dict) -> str:
    return record["tool_calls"][0]["evidence_ids"][0]

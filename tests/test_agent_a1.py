"""A1 runtime reliability, H10 placement, and the abstention contract.

These tests do not call an answer API. The six A0 budget-exhaustion questions
are the charge pattern, not a new test set.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from test_agent_product import events_of, make_runtime, registry, sample_result
from test_grounded_answer import FakeAnswerClient, _output

from disclosure_agent.agent.answer import EvidenceBoundAnswerGenerator
from disclosure_agent.agent.catalog import DenseD2aBinding
from disclosure_agent.agent.claim_class import classify_product_claim
from disclosure_agent.agent.clock import SEOUL
from disclosure_agent.agent.context import ContextBuilder, evidence_token_cost
from disclosure_agent.agent.corpus import ProductCorpus
from disclosure_agent.agent.errors import StructuredOutputInvalid
from disclosure_agent.agent.evidence import Evidence, EvidenceBuilder
from disclosure_agent.agent.h10_search import H10Search
from disclosure_agent.agent.llm import OpenAIAnswerGenerator
from disclosure_agent.agent.planner import ScriptedPlanner
from disclosure_agent.agent.record import RunLog
from disclosure_agent.agent.runtime import AgentRuntime
from disclosure_agent.agent.search import (
    DenseD2aIndex,
    DenseD2aSearch,
    SearchDisclosuresRequest,
    SearchDisclosuresResult,
)
from disclosure_agent.agent.search_catalog import SearchCatalog
from disclosure_agent.agent.settings import (
    PRODUCT_DISCLOSURE_MIN_TOP_K,
    AgentLimits,
    product_search_top_k,
)
from disclosure_agent.agent_a0.abstention_contract import CORRECTED_RUN_DIR, write_correction
from disclosure_agent.bm25 import Bm25Index
from disclosure_agent.dense import EMBEDDING_DIMENSION
from disclosure_agent.evalset_v2.spec import repo_root
from disclosure_agent.evalset_v2_1.asymmetric_fusion import MARGINAL_SLOT, bm25_primary_dense_head
from disclosure_agent.evalset_v2_1.complementarity import bm25_chunks
from disclosure_agent.retrieval_unit import D2A_VERSION, scoring_chunks

BUDGET_CASES = (
    "V2-00107613-finance_table-01",
    "V2-00126380-finance_table-03",
    "V2-00126955-overview_paragraph-01",
    "V2-00107613-multi_element-01",
    "V2-00126380-multi_element-02",
    "V2-00121941-multi_element-01",
)
AEROSPACE = (
    "이수페타시스의 우주항공용 회로기판은 왜 극심한 온도 변화와 진동·충격을 "
    "견디고 오랫동안 믿고 쓸 수 있어야 하나요?"
)
AEROSPACE_DRAFT = (
    "우주항공용 회로기판은 발사 과정에서 온도 변화와 진동을 겪습니다. "
    "그래서 그 회사 기판은 오래 견디도록 만들어집니다."
)


def _search_step(query: str) -> dict[str, object]:
    return {
        "query": query,
        "company_id": "00107613",
        "period": None,
        "filing_date_from": None,
        "filing_date_to": None,
        "report_types": None,
        "as_of": None,
        "top_k": 5,
    }


def _evidence_id(unit_id: str = "unit-1") -> str:
    digest = hashlib.sha256(unit_id.encode("utf-8")).hexdigest()[:16]
    return f"ev-{digest}"


def test_company_resolution_does_not_spend_llm_budget(tmp_path: Path) -> None:
    """The six A0 exhaustions spent one slot before three planner calls."""
    calls: list[SearchDisclosuresRequest] = []
    client = FakeAnswerClient([_output(_evidence_id(), answer="투자활동현금흐름을 확인했습니다.")])
    planner = ScriptedPlanner(
        [
            {
                "action": "ask_clarification",
                "text": "이수페타시스의 공시를 볼까요?",
                "company_mention": "이수페타시스",
            },
            _search_step("2024년 상반기 연결 현금흐름표 투자활동현금흐름"),
            _search_step("제53기 반기 연결 현금흐름표 투자활동현금흐름"),
        ],
        uses_llm=True,
    )
    log = RunLog(tmp_path / "budget.jsonl")
    runtime = AgentRuntime(
        registry=registry(),
        search=lambda request: calls.append(request) or sample_result(),
        evidence_builder=EvidenceBuilder(),
        answer_generator=OpenAIAnswerGenerator(client),
        planner=planner,
        limits=AgentLimits(max_search_calls=2, max_llm_calls=4),
        run_log=log,
    )
    events = events_of(
        runtime,
        "2024년 상반기 연결 현금흐름표에서 투자활동현금흐름은 얼마였나요?",
    )
    assert len(BUDGET_CASES) == 6
    assert len(calls) == 2
    assert events[-1].event == "run.completed"
    assert events[-1].data["status"] == "completed"
    record = json.loads(log.path.read_text(encoding="utf-8").splitlines()[-1])
    assert record["llm_call_count"] == 4
    assert record["search_call_count"] == 2
    assert "company_resolution" in record["deterministic_operations"]
    assert record["status"] != "llm_limit_exceeded"
    assert record["error"] is None


def test_two_searches_finish_and_a_third_search_is_not_run(tmp_path: Path) -> None:
    calls: list[SearchDisclosuresRequest] = []
    client = FakeAnswerClient([_output(_evidence_id(), answer="발췌에서 확인했습니다.")])
    planner = ScriptedPlanner(
        [_search_step("첫번째"), _search_step("두번째"), _search_step("세번째")],
        uses_llm=True,
    )
    runtime = AgentRuntime(
        registry=registry(),
        search=lambda request: calls.append(request) or sample_result(),
        evidence_builder=EvidenceBuilder(),
        answer_generator=OpenAIAnswerGenerator(client),
        planner=planner,
        limits=AgentLimits(),
        run_log=RunLog(tmp_path / "two.jsonl"),
    )
    events = events_of(runtime, "이수페타시스의 주요 사업이 뭐야?")
    assert len(calls) == 2
    assert [request.query for request in calls] == ["첫번째", "두번째"]
    assert events[-1].data["status"] == "completed"
    record = json.loads((tmp_path / "two.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert record.get("search_suppressed") is not True


def test_search_after_the_limit_is_suppressed() -> None:
    calls: list[SearchDisclosuresRequest] = []

    def search(request: SearchDisclosuresRequest):
        calls.append(request)
        return SearchDisclosuresResult(
            hits=[],
            retriever_type="dense",
            representation_version=D2A_VERSION,
            embedding_model="BAAI/bge-m3",
            company_id="00107613",
            corp_code="00107613",
        )

    planner = ScriptedPlanner(
        [_search_step("하나"), _search_step("둘"), _search_step("셋")],
        uses_llm=True,
    )
    runtime = AgentRuntime(
        registry=registry(),
        search=search,
        evidence_builder=EvidenceBuilder(),
        planner=planner,
        limits=AgentLimits(),
    )
    events = events_of(runtime, "이수페타시스의 주요 사업이 뭐야?")
    assert len(calls) == 2
    assert events[-1].event == "run.completed"
    assert events[-1].data["status"] != "llm_limit_exceeded"


def test_general_question_still_answers_without_search() -> None:
    calls: list[SearchDisclosuresRequest] = []
    planner = ScriptedPlanner(
        [
            {
                "action": "answer_directly",
                "text": "이 서비스는 공시 원문을 찾아 답합니다.",
                "company_mention": None,
            }
        ],
        uses_llm=True,
    )
    runtime = AgentRuntime(
        registry=registry(),
        search=lambda request: calls.append(request) or sample_result(),
        evidence_builder=EvidenceBuilder(),
        planner=planner,
    )
    events = events_of(runtime, "이 서비스는 어떤 기능을 제공하나요?")
    assert calls == []
    assert events[-1].data["action"] == "answer_directly"
    assert events[-1].data["answer"]["citations"] == []
    assert "공시 원문" in events[-1].data["answer"]["text"]


def test_company_fact_without_evidence_is_not_asserted() -> None:
    calls: list[SearchDisclosuresRequest] = []
    planner = ScriptedPlanner(
        [
            {
                "action": "answer_directly",
                "text": AEROSPACE_DRAFT,
                "company_mention": None,
            }
        ],
        uses_llm=True,
    )
    runtime = AgentRuntime(
        registry=registry(),
        search=lambda request: calls.append(request) or sample_result(),
        evidence_builder=EvidenceBuilder(),
        planner=planner,
    )
    events = events_of(runtime, AEROSPACE)
    assert calls == []
    answer = events[-1].data["answer"]["text"]
    assert AEROSPACE_DRAFT not in answer
    assert events[-1].data["answer"]["citations"] == []
    assert events[-1].data["abstain_reason"] == "evidence_required"
    assert "단정하지 않습니다" in answer


def test_planner_may_search_after_an_ungrounded_draft_is_rejected() -> None:
    calls: list[SearchDisclosuresRequest] = []
    client = FakeAnswerClient([_output(_evidence_id(), answer="공시 발췌입니다.")])
    planner = ScriptedPlanner(
        [
            {"action": "answer_directly", "text": AEROSPACE_DRAFT, "company_mention": None},
            _search_step("우주항공 기판"),
        ],
        uses_llm=True,
    )
    runtime = AgentRuntime(
        registry=registry(),
        search=lambda request: calls.append(request) or sample_result(),
        evidence_builder=EvidenceBuilder(),
        answer_generator=OpenAIAnswerGenerator(client),
        planner=planner,
        limits=AgentLimits(max_search_calls=1, max_llm_calls=4),
    )
    events = events_of(runtime, AEROSPACE)
    assert len(calls) == 1
    assert events[-1].data["status"] == "completed"
    assert events[-1].data["answer"]["citations"]


def test_timeout_is_not_budget_exhaustion(tmp_path: Path) -> None:
    log = RunLog(tmp_path / "timeout.jsonl")
    runtime = make_runtime([])
    runtime.timeout_seconds = -1
    runtime.run_log = log
    events = events_of(runtime, "안녕")
    assert events[-1].data["code"] == "request_timeout"
    record = json.loads(log.path.read_text(encoding="utf-8").splitlines()[-1])
    assert record["status"] == "request_timeout"
    assert record["error"]["code"] == "request_timeout"
    assert record["error"]["code"] != "llm_limit_exceeded"


def test_llm_limit_failure_is_stored(tmp_path: Path) -> None:
    client = FakeAnswerClient(
        [
            StructuredOutputInvalid("first"),
            _output("ev-unused"),
        ]
    )
    log = RunLog(tmp_path / "limit.jsonl")
    runtime = AgentRuntime(
        registry=registry(),
        search=lambda request: sample_result(),
        evidence_builder=EvidenceBuilder(),
        answer_generator=OpenAIAnswerGenerator(client),
        limits=AgentLimits(max_search_calls=2, max_llm_calls=1),
        run_log=log,
    )
    events = events_of(runtime, "이수페타시스의 주요 사업이 뭐야?")
    assert events[-1].data["code"] == "llm_limit_exceeded"
    record = json.loads(log.path.read_text(encoding="utf-8").splitlines()[-1])
    assert record["status"] == "llm_limit_exceeded"
    assert record["error"]["code"] == "llm_limit_exceeded"
    assert record["deterministic_operations"]


def test_h10_prefix_and_marginal_slot_match_the_offline_rule() -> None:
    dense = [
        {
            "rank": index,
            "retrieval_unit_id": f"d{index}",
            "source_element_ids": [f"de{index}"],
        }
        for index in range(1, 21)
    ]
    bm25 = [
        {
            "rank": index,
            "retrieval_unit_id": f"b{index}",
            "source_element_ids": [f"be{index}"],
        }
        for index in range(1, 21)
    ]
    inserted = bm25_primary_dense_head(dense, bm25)
    assert [hit["retrieval_unit_id"] for hit in inserted["hits"][:9]] == [
        f"b{index}" for index in range(1, 10)
    ]
    assert inserted["hits"][9]["retrieval_unit_id"] == "d1"
    assert inserted["edit"] == "insertion"
    dense[0] = {
        "rank": 1,
        "retrieval_unit_id": "b1",
        "source_element_ids": ["be1"],
    }
    unchanged = bm25_primary_dense_head(dense, bm25)
    assert unchanged["edit"] == "none"
    assert [hit["retrieval_unit_id"] for hit in unchanged["hits"][:10]] == [
        f"b{index}" for index in range(1, 11)
    ]


def test_product_h10_preserves_bm25_prefix_and_unit_identity(tmp_path: Path) -> None:
    units = [_unit(index) for index in range(20)]
    result = _h10(tmp_path, units, head=19, top_k=20, query="우선어 공통")
    bm25_ids = [
        hit["retrieval_unit_id"]
        for hit in Bm25Index(bm25_chunks(units, scoring_chunks(units))).search("우선어 공통", 20)
    ]
    assert result.retrieval_policy == "bm25_primary_dense_head"
    assert "unit-19" not in bm25_ids[:10]
    expected_edit = "promotion" if "unit-19" in bm25_ids else "insertion"
    assert result.retrieval_edit == expected_edit
    assert [hit.retrieval_unit_id for hit in result.hits[:9]] == bm25_ids[:9]
    assert result.hits[9].retrieval_unit_id == "unit-19"
    for hit in result.hits:
        assert hit.retrieval_unit_id not in hit.source_element_ids
        assert hit.source_element_ids
    built = EvidenceBuilder().build(result.hits[:1], company_name="이수페타시스")
    assert built[0].retrieval_unit_id != built[0].element_ids[0]
    assert built[0].evidence_id != built[0].element_ids[0]

    same = _h10(tmp_path / "same", units, head=0, top_k=10, query="우선어 공통")
    assert same.retrieval_edit == "none"
    assert [hit.retrieval_unit_id for hit in same.hits[:9]] == bm25_ids[:9]

    window = _h10(tmp_path / "window", units, head=19, top_k=5, query="우선어 공통")
    assert [hit.retrieval_unit_id for hit in window.hits] == bm25_ids[:5]


def test_short_h10_window_does_not_silently_use_dense(tmp_path: Path) -> None:
    units = [_unit(index) for index in range(5)]
    result = _h10(tmp_path, units, head=0, top_k=5, query="우선어 공통")
    assert result.hits == []
    assert result.retrieval_policy == "h10_unavailable"
    assert result.policy_reason == "candidate_n_not_met"
    assert result.retriever_type != "dense"


def test_product_search_top_k_keeps_explicit_windows_at_or_above_the_dense_head() -> None:
    assert PRODUCT_DISCLOSURE_MIN_TOP_K == MARGINAL_SLOT == 10
    assert product_search_top_k(5, 20) == 10
    assert product_search_top_k(10, 20) == 10
    assert product_search_top_k(15, 20) == 15
    assert product_search_top_k(20, 20) == 20
    assert product_search_top_k(5, 8) == 8


def test_product_path_raises_planner_top_k_below_the_dense_head_slot(tmp_path: Path) -> None:
    calls: list[SearchDisclosuresRequest] = []
    log = RunLog(tmp_path / "topk.jsonl")
    runtime = AgentRuntime(
        registry=registry(),
        search=lambda request: calls.append(request) or sample_result(),
        evidence_builder=EvidenceBuilder(),
        planner=ScriptedPlanner([_search_step("주요 사업")]),
        run_log=log,
    )
    events_of(runtime, "이수페타시스의 주요 사업이 뭐야?")
    assert calls[0].top_k == 10
    record = json.loads(log.path.read_text(encoding="utf-8").splitlines()[-1])
    tool = record["tool_calls"][0]
    assert tool["arguments"]["top_k"] == 5
    assert tool["top_k_requested"] == 5
    assert tool["top_k_applied"] == 10
    wider = dict(_search_step("주요 사업"))
    wider["top_k"] = 15
    calls.clear()
    runtime.planner = ScriptedPlanner([wider])
    events_of(runtime, "이수페타시스의 주요 사업이 뭐야?")
    assert calls[0].top_k == 15


def test_context_budget_keeps_ten_short_hits_and_drops_a_tenth_after_a_full_prefix() -> None:
    builder = ContextBuilder(token_budget=4096, max_evidence=20)
    short = [_evidence(index, f"문장 {index}") for index in range(10)]
    kept = builder.build(short)
    assert [item.evidence_id for item in kept.evidence] == [f"ev-{index}" for index in range(10)]
    assert kept.token_count <= 4096
    sample = _evidence(0, "가" * 800)
    assert evidence_token_cost(sample) > 400
    rows = [_evidence(index, f"{index}-" + ("가" * 800)) for index in range(10)]
    selected = builder.build(rows)
    assert selected.evidence
    assert len(selected.evidence) < 10
    assert selected.token_count <= 4096
    assert rows[-1].evidence_id not in {item.evidence_id for item in selected.evidence}


def test_h10_unavailable_is_not_shown_as_a_missing_filing() -> None:
    def search(request: SearchDisclosuresRequest) -> SearchDisclosuresResult:
        del request
        return SearchDisclosuresResult(
            hits=[],
            retriever_type="h10_unavailable",
            representation_version=D2A_VERSION,
            embedding_model="BAAI/bge-m3",
            company_id="00107613",
            corp_code="00107613",
            retrieval_policy="h10_unavailable",
            policy_reason="candidate_n_not_met",
        )

    runtime = AgentRuntime(
        registry=registry(),
        search=search,
        evidence_builder=EvidenceBuilder(),
        planner=ScriptedPlanner([_search_step("주요 사업")]),
    )
    done = events_of(runtime, "이수페타시스의 주요 사업이 뭐야?")[-1].data
    assert done["status"] == "retrieval_unavailable"
    assert done["abstain_reason"] == "h10_unavailable"
    text = done["answer"]["text"]
    assert "없다고 확인된 것은 아닙니다" in text
    assert "h10_unavailable" not in text
    assert "찾지 못했습니다" not in text
    assert "dense" not in text.casefold()
    assert done["answer"]["citations"] == []


def test_budget_fallback_hides_retrieval_headers() -> None:
    quoted = (
        "[section] 목 차 > II. 사업의 내용\n"
        "인쇄회로기판을 생산합니다.\n"
        "[source] element_id=el-1 role=body kind=paragraph"
    )
    done = _budget_answer(quoted)
    answer = done["answer"]["text"]
    limitations = " ".join(done["answer"]["limitations"])
    assert done["answer_stop"] == "generation_budget_fallback"
    assert done["status"] == "completed"
    assert "[section]" not in answer
    assert "[source]" not in answer
    assert "목 차" not in answer
    assert "traceback" not in answer.lower()
    assert "retrieval_unit" not in answer
    assert "인쇄회로기판을 생산합니다." in answer
    assert "답변 모델 호출 한도" in limitations
    assert done["answer"]["citations"]

    header_only = _budget_answer("[section] 목 차 > II. 사업의 내용")
    bare = header_only["answer"]["text"]
    assert "[section]" not in bare
    assert "목 차" not in bare
    assert header_only["answer"]["citations"]
    assert "답변 모델 호출 한도" in " ".join(header_only["answer"]["limitations"])


def _budget_answer(text: str) -> dict:
    runtime = AgentRuntime(
        registry=registry(),
        search=lambda request: sample_result(text),
        evidence_builder=EvidenceBuilder(),
        answer_generator=OpenAIAnswerGenerator(FakeAnswerClient([])),
        planner=ScriptedPlanner([_search_step("주요 사업")]),
        limits=AgentLimits(max_search_calls=1, max_llm_calls=0),
    )
    return events_of(runtime, "이수페타시스의 주요 사업이 뭐야?")[-1].data


def _evidence(index: int, text: str) -> Evidence:
    return Evidence(
        evidence_id=f"ev-{index}",
        retrieval_unit_id=f"unit-{index}",
        receipt_no="20260323000847",
        document_name="사업보고서",
        company_name="이수페타시스",
        section_path=["사업의 내용"],
        element_ids=[f"el-{index}"],
        excerpt=text,
        source_locator=[],
        text=text,
    )


def test_abstention_sentences_are_not_factual_claims() -> None:
    assert classify_product_claim("검색된 공시에서 요청한 내용을 확인하지 못했습니다.")[
        "claim_type"
    ] == "abstention"
    assert classify_product_claim("사업보고서에는 해당 정보가 없습니다.")["a0_act"] == (
        "document_absence"
    )
    future = classify_product_claim(
        "2030년 확정 매출은 아직 발생하지 않은 미래 기간이라 알 수 없습니다."
    )
    assert future["a0_act"] == "future_confirmed_fact"
    assert classify_product_claim("검색을 완료하지 못했습니다.")["a0_act"] == "tool_failure"
    outside = classify_product_claim("이 기업은 지원하는 공시가 없습니다.")
    assert outside["a0_act"] == "outside_scope"
    assert classify_product_claim("매출은 1,234원입니다.")["claim_type"] == "factual_claim"
    assert EvidenceBoundAnswerGenerator.uses_llm is False


def test_a0_abstention_correction_keeps_the_baseline(tmp_path: Path) -> None:
    root = repo_root()
    source = root / "evaluation" / "runs" / "gongsi_mate_a0_agent_baseline" / "summary.json"
    before = source.read_text(encoding="utf-8")
    summary = write_correction(root, tmp_path / "corrected")
    assert source.read_text(encoding="utf-8") == before
    assert json.loads(before)["task"]["success"] == 11
    assert summary["judge_recalled"] == 0
    by_id = {
        row["query_id"]: row
        for row in (
            json.loads(line)
            for line in (tmp_path / "corrected" / "evaluation_results.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    }
    assert by_id["A0-abstain-01"]["task_outcome"] == "appropriate_abstention"
    assert by_id["A0-abstain-02"]["task_outcome"] == "appropriate_abstention"
    assert by_id["A0-abstain-01"]["grounding"]["faithfulness_denominator"] == 0
    assert by_id["V2-00107613-business_paragraph-02"]["primary_cause"] == "tool_selection_failure"


def test_write_correction_into_the_run_directory() -> None:
    root = repo_root()
    source = root / "evaluation" / "runs" / "gongsi_mate_a0_agent_baseline" / "summary.json"
    before = source.read_bytes()
    summary = write_correction(root)
    assert source.read_bytes() == before
    destination = root / "evaluation" / "runs" / CORRECTED_RUN_DIR
    assert (destination / "evaluation_contract.md").is_file()
    assert (destination / "manifest.json").is_file()
    assert summary["query_count"] == 30


def _unit(index: int) -> dict[str, object]:
    if index == 19:
        text = "공통 만 있는 문서"
    else:
        text = f"우선어 공통 문서{index}"
    return {
        "retrieval_unit_id": f"unit-{index:02d}",
        "representation_version": D2A_VERSION,
        "text": text,
        "canonical_ordinal": index,
        "unit_index": index,
        "canonical_chunk_id": f"chunk-{index:02d}",
        "source_element_ids": [f"el-{index:02d}"],
        "source_refs": [
            {
                "document_id": "m0:doc.xml",
                "element_id": f"el-{index:02d}",
                "char_start": 0,
                "char_end": 4,
                "kind": "paragraph",
                "role": "body",
            }
        ],
        "receipt_no": "20240814003325",
        "document_name": "사업보고서",
        "section_path": ["사업의 내용"],
        "corp_code": "00107613",
        "report_name": "사업보고서",
        "filing_period": "2024.06",
        "receipt_date": "20240814",
        "source_sha256": "abc",
    }


class _HeadEncoder:
    def __init__(self, head: int) -> None:
        self.head = head

    def encode(self, text: str) -> np.ndarray:
        del text
        vector = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
        vector[self.head] = 1.0
        return vector


def _h10(tmp_path: Path, units: list[dict], *, head: int, top_k: int, query: str):
    binding = DenseD2aBinding(
        corp_code="00107613",
        company_name="이수페타시스",
        units_path=tmp_path / "units.jsonl",
        embeddings_path=tmp_path / "embeddings.f32",
        elements_path=None,
        receipt_no="20240814003325",
        document_name="사업보고서",
        representation_version=D2A_VERSION,
        embedding_model="BAAI/bge-m3",
    )
    matrix = np.eye(len(units), EMBEDDING_DIMENSION, dtype=np.float32)
    index = DenseD2aIndex(units=units, matrix=matrix, binding=binding)
    corpus = ProductCorpus(
        binding=binding,
        filings=(),
        corpus_as_of=datetime(2026, 10, 2, tzinfo=SEOUL),
        elements_paths=(),
        manifest={"corpus_id": "synthetic-h10", "representation_version": D2A_VERSION},
        index_dir=tmp_path,
    )
    catalog = SearchCatalog(root=tmp_path, corpora=(corpus,))
    search = H10Search(catalog)
    search.dense._search["00107613"] = DenseD2aSearch(binding, _HeadEncoder(head), index=index)
    return search(
        SearchDisclosuresRequest(
            query=query,
            company_id="00107613",
            corp_code="00107613",
            top_k=top_k,
        )
    )

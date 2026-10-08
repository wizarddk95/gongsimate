"""A0 labels stay out of the agent input, and the scorer reads the product record."""

from __future__ import annotations

import json
from pathlib import Path

from disclosure_agent.agent.company import CompanyRecord, CompanyRegistry
from disclosure_agent.agent.evidence import EvidenceBuilder
from disclosure_agent.agent.record import RunLog
from disclosure_agent.agent.runtime import AgentRuntime
from disclosure_agent.agent.search import (
    SearchDisclosuresRequest,
    SearchDisclosuresResult,
    SearchHit,
)
from disclosure_agent.agent.settings import AgentLimits
from disclosure_agent.agent_a0.dataset import agent_input, build_rows, dataset_sha256, write_dataset
from disclosure_agent.agent_a0.grade import grade_case, primary_behavior
from disclosure_agent.agent_a0.judge import faith_payload, requirement_payload
from disclosure_agent.evalset_v2.spec import repo_root
from disclosure_agent.retrieval_unit import D2A_VERSION

ROOT = repo_root()


def test_dataset_uses_dev_questions_only() -> None:
    rows = build_rows(ROOT)
    questions = json.loads(
        (
            ROOT / "evaluation" / "datasets" / "multi_company_retrieval_v2_1" / "questions.json"
        ).read_text(encoding="utf-8")
    )
    by_id = {item["query_id"]: item for item in questions["questions"]}
    assert len(rows) == 30
    assert dataset_sha256(rows) == dataset_sha256(build_rows(ROOT))
    reused = [row for row in rows if row.get("source_query_id")]
    assert reused
    for row in reused:
        source = by_id[row["source_query_id"]]
        assert source["split"] == "dev"
        assert row["question"] == source["question"]
        assert agent_input(row) == row["question"]
        assert "expected_behavior" not in agent_input(row)


def test_dataset_file_matches_builder(tmp_path: Path) -> None:
    meta = write_dataset(ROOT)
    text = (ROOT / meta["path"]).read_text(encoding="utf-8")
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    assert meta["dataset_sha256"] == dataset_sha256(rows)
    assert meta["query_count"] == 30
    del tmp_path


def test_search_decision_and_wrong_company() -> None:
    case = _case(
        scenario="search_required",
        allowed=["SEARCH", "PARTIAL_ANSWER", "ADDITIONAL_SEARCH", "ABSTAIN"],
        prohibited=["DIRECT_ANSWER", "CLARIFY"],
        search_expectation="required",
        company_id="00107613",
        period={"year": 2024, "half": 1},
    )
    record = _record(
        action="search_disclosures",
        calls=[
            _call("00126380", year=2023, status="success"),
        ],
    )
    graded = grade_case(case, record, _judgment(covered=True, verdict="supported"))
    assert graded["search_decision"]["correct"] is True
    assert "wrong_company" in graded["argument_findings"]
    assert "wrong_period" in graded["argument_findings"]
    assert graded["task_outcome"] == "failure"
    assert graded["primary_cause"] == "tool_argument_failure"


def test_unnecessary_search_stays_off_the_task_when_direct_answer_is_allowed() -> None:
    case = _case(
        scenario="insufficient_evidence",
        allowed=["ABSTAIN", "DIRECT_ANSWER"],
        prohibited=["SEARCH", "CLARIFY"],
        search_expectation="prohibited",
        company_id="00107613",
        answerability="unanswerable",
        requirements=[
            {
                "requirement_id": "R1",
                "text": "수치를 제시하지 않는다.",
                "success_verdicts": ["covered", "abstained"],
            }
        ],
    )
    record = _record(action="abstain", abstain=True, calls=[_call("00107613", status="no_results")])
    graded = grade_case(case, record, _judgment(verdict="not_factual", requirement="abstained"))
    assert graded["search_decision"]["unnecessary_search"] is True
    assert graded["task_outcome"] == "appropriate_abstention"
    assert graded["primary_cause"] is None
    assert "unnecessary_search" in graded["secondary_flags"]


def test_missed_search_and_clarification() -> None:
    missed = grade_case(
        _case(search_expectation="required", allowed=["SEARCH", "ABSTAIN"]),
        _record(action="answer_directly", text="잘 모르겠습니다."),
        None,
    )
    assert missed["search_decision"]["missed_required_search"] is True
    assert missed["task_outcome"] == "failure"
    clarify = grade_case(
        _case(
            scenario="ambiguous",
            allowed=["CLARIFY"],
            prohibited=["SEARCH", "DIRECT_ANSWER", "ABSTAIN"],
            search_expectation="prohibited",
            company_id=None,
        ),
        _record(action="ask_clarification", status="company_not_found", text="어느 회사인가요?"),
        None,
    )
    assert clarify["task_outcome"] == "appropriate_clarification"
    assert primary_behavior(clarify["observed_behaviors"]) == "CLARIFY"


def test_llm_limit_after_a_successful_search_is_budget_exhaustion() -> None:
    case = _case(search_expectation="required", company_id="00107613")
    record = _record(
        action="search_disclosures",
        text="",
        status="llm_limit_exceeded",
        calls=[_call("00107613", status="success")],
        evidence=[],
        citations=[],
    )
    graded = grade_case(case, record, None)
    assert graded["task_outcome"] == "failure"
    assert graded["primary_cause"] == "budget_exhaustion"
    assert "retrieval_failure" not in graded["secondary_flags"]


def test_citation_outside_the_run_is_an_integrity_error() -> None:
    case = _case(search_expectation="required", allowed=["SEARCH", "ABSTAIN"])
    record = _record(
        action="search_disclosures",
        calls=[_call("00107613", status="success")],
        citations=[{"citation_id": "1", "evidence_id": "not-in-run"}],
        evidence=[{"evidence_id": "in-run", "text": "매출은 10이다."}],
    )
    graded = grade_case(case, record, _judgment(covered=True, verdict="supported"))
    assert graded["grounding"]["integrity_error_count"] == 1
    assert graded["task_outcome"] == "failure"
    assert (
        "citation_failure" in graded["secondary_flags"]
        or graded["primary_cause"] == "citation_failure"
    )


def test_judge_payload_excludes_labels() -> None:
    payload = faith_payload(
        "이수페타시스의 주요 제품은?",
        {"text": "PCB를 만든다.", "abstain": False, "citations": [], "limitations": []},
        [{"claim_id": "K1", "text": "PCB를 만든다."}],
        [{"evidence_id": "e1", "text": "인쇄회로기판", "report_name": "사업보고서"}],
    )
    requirement = requirement_payload(
        "이수페타시스의 주요 제품은?",
        {"text": "PCB를 만든다.", "limitations": []},
        [{"requirement_id": "R1", "text": "주요 제품을 말한다.", "success_verdicts": ["covered"]}],
    )
    assert "expected_behavior" not in json.dumps(payload, ensure_ascii=False)
    assert "success_verdicts" not in json.dumps(requirement, ensure_ascii=False)
    assert "evidence" not in requirement


def test_runtime_records_answer_and_call_counts(tmp_path: Path) -> None:
    calls: list[SearchDisclosuresRequest] = []

    def search(request: SearchDisclosuresRequest) -> SearchDisclosuresResult:
        calls.append(request)
        return SearchDisclosuresResult(
            hits=[
                SearchHit(
                    retrieval_unit_id="unit-1",
                    score=0.8,
                    rank=1,
                    text="당사는 인쇄회로기판을 생산합니다.",
                    canonical_chunk_id="chunk-1",
                    unit_index=0,
                    source_element_ids=["el-1"],
                    source_refs=[],
                    receipt_no="20260323000847",
                    document_name="사업보고서",
                    section_path=["사업의 내용"],
                    source_locator=[],
                )
            ],
            retriever_type="dense",
            representation_version=D2A_VERSION,
            embedding_model="BAAI/bge-m3",
            company_id="00107613",
            corp_code="00107613",
        )

    log = RunLog(tmp_path / "runs.jsonl")
    runtime = AgentRuntime(
        registry=CompanyRegistry.from_records(
            [CompanyRecord("00107613", "이수페타시스", "007660", "")],
            indexed_corp_codes={"00107613"},
        ),
        search=search,
        evidence_builder=EvidenceBuilder(),
        limits=AgentLimits(),
        run_log=log,
    )
    runtime.run("이수페타시스의 주요 사업이 뭐야?")
    record = json.loads(log.path.read_text(encoding="utf-8").splitlines()[-1])
    assert record["final_answer"]["text"]
    assert record["generation_evidence"]
    assert record["llm_call_count"] == 2
    assert record["search_call_count"] == 1
    assert record["planner_actions"]
    assert record["stop_reason"] == "completed"


def _case(**overrides: object) -> dict:
    row = {
        "query_id": "A0-test",
        "question": "이수페타시스의 2024년 상반기 매출은?",
        "scenario": "search_required",
        "expected_behavior": "SEARCH",
        "allowed_behaviors": ["SEARCH", "ABSTAIN"],
        "prohibited_behaviors": ["DIRECT_ANSWER", "CLARIFY"],
        "search_expectation": "required",
        "company_id": "00107613",
        "company_name": "이수페타시스",
        "period": None,
        "answerability": "fully_answerable",
        "required_evidence_references": [],
        "requirements": [
            {
                "requirement_id": "R1",
                "text": "2024년 상반기 매출을 말한다.",
                "success_verdicts": ["covered"],
            }
        ],
        "evaluation_notes": "",
    }
    if "allowed" in overrides:
        overrides["allowed_behaviors"] = overrides.pop("allowed")
    if "prohibited" in overrides:
        overrides["prohibited_behaviors"] = overrides.pop("prohibited")
    row.update(overrides)
    return row


def _call(company_id: str, *, year: int | None = 2024, half: int | None = 1, status: str) -> dict:
    return {
        "name": "search_disclosures",
        "status": status,
        "arguments": {
            "query": "매출",
            "company_id": company_id,
            "period": {
                "year": year,
                "quarter": None,
                "half": half,
                "mode": "exact",
                "year_from": None,
                "year_to": None,
            },
            "report_types": ["semiannual"],
        },
    }


def _record(
    *,
    action: str,
    text: str = "매출은 10이다.",
    abstain: bool = False,
    status: str = "completed",
    calls: list[dict] | None = None,
    citations: list[dict] | None = None,
    evidence: list[dict] | None = None,
) -> dict:
    return {
        "agent_action": action,
        "status": status,
        "stop_reason": status,
        "error": None,
        "tool_calls": calls or [],
        "final_answer": {
            "text": text,
            "citations": citations
            if citations is not None
            else (
                [{"citation_id": "1", "evidence_id": "in-run"}]
                if action == "search_disclosures"
                else []
            ),
            "limitations": [],
            "abstain": abstain,
        },
        "generation_evidence": evidence
        if evidence is not None
        else (
            [{"evidence_id": "in-run", "retrieval_unit_id": "u1", "text": "매출은 10이다."}]
            if calls
            else []
        ),
        "retrieved_retrieval_unit_ids": [],
    }


def _judgment(
    *,
    covered: bool = False,
    verdict: str = "supported",
    requirement: str | None = None,
) -> dict:
    return {
        "faithfulness": [
            {
                "claim_id": "K1",
                "verdict": verdict,
                "scope_issue": "none",
                "reason": "test",
            }
        ],
        "citations": [
            {"evidence_id": "in-run", "verdict": "supports", "reason": "test"},
        ],
        "requirements": [
            {
                "requirement_id": "R1",
                "verdict": requirement or ("covered" if covered else "omitted"),
                "reason": "test",
            }
        ],
    }

# ruff: noqa: E501
"""A0 development questions.

Reused disclosure questions come from the frozen v2.1 DEV split.
Questions that should not be searched, or that are ambiguous, are written here.
The agent runner sends ``question`` only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

DATASET_VERSION = "gongsi_mate_a0_dev_v1"
DATASET_DIR = "gongsi_mate_a0_dev"
QUESTIONS_NAME = "agent_evalset.jsonl"

_BEHAVIORS = (
    "DIRECT_ANSWER",
    "SEARCH",
    "CLARIFY",
    "PARTIAL_ANSWER",
    "ABSTAIN",
    "ADDITIONAL_SEARCH",
)
_SEARCH_ALLOWED = ["SEARCH", "PARTIAL_ANSWER", "ADDITIONAL_SEARCH", "ABSTAIN"]
_WITHHOLD = ("제시하지", "보류", "만들어내지", "단정하지", "비교를 보류")

REUSED: tuple[dict[str, Any], ...] = (
    {
        "source_query_id": "V2-00107613-overview_paragraph-01",
        "scenario": "search_required",
        "expected_behavior": "SEARCH",
        "allowed_behaviors": _SEARCH_ALLOWED,
        "prohibited_behaviors": ["DIRECT_ANSWER", "CLARIFY"],
        "search_expectation": "required",
        "period": None,
        "evaluation_notes": "회사와 확인할 사실이 질문에 있다. 기간 필터는 필수가 아니다.",
    },
    {
        "source_query_id": "V2-00107613-business_paragraph-02",
        "scenario": "search_required",
        "expected_behavior": "SEARCH",
        "allowed_behaviors": _SEARCH_ALLOWED,
        "prohibited_behaviors": ["DIRECT_ANSWER", "CLARIFY"],
        "search_expectation": "required",
        "period": None,
        "evaluation_notes": "제품 특성의 이유를 공시에서 확인하는 질문이다.",
    },
    {
        "source_query_id": "V2-00107613-finance_table-01",
        "scenario": "search_required",
        "expected_behavior": "SEARCH",
        "allowed_behaviors": _SEARCH_ALLOWED,
        "prohibited_behaviors": ["DIRECT_ANSWER", "CLARIFY"],
        "search_expectation": "required",
        "period": {"year": 2024, "half": 1},
        "evaluation_notes": "2024년 상반기는 보고기간이다. 접수일을 그 해로 고정할 필요는 없다.",
    },
    {
        "source_query_id": "V2-00107613-mdna_paragraph-02",
        "scenario": "search_required",
        "expected_behavior": "SEARCH",
        "allowed_behaviors": _SEARCH_ALLOWED,
        "prohibited_behaviors": ["DIRECT_ANSWER", "CLARIFY"],
        "search_expectation": "required",
        "period": None,
        "evaluation_notes": "유동성 위험 관리. 특정 사업연도 필터는 필수가 아니다.",
    },
    {
        "source_query_id": "V2-00126380-business_paragraph-01",
        "scenario": "search_required",
        "expected_behavior": "SEARCH",
        "allowed_behaviors": _SEARCH_ALLOWED,
        "prohibited_behaviors": ["DIRECT_ANSWER", "CLARIFY"],
        "search_expectation": "required",
        "period": {"year": 2026, "half": 1},
        "evaluation_notes": "2026년 반기말은 보고기간이다.",
    },
    {
        "source_query_id": "V2-00126380-finance_table-03",
        "scenario": "search_required",
        "expected_behavior": "SEARCH",
        "allowed_behaviors": _SEARCH_ALLOWED,
        "prohibited_behaviors": ["DIRECT_ANSWER", "CLARIFY"],
        "search_expectation": "required",
        "period": {"year": 2025, "report_types": ["annual"]},
        "evaluation_notes": "2025년 말은 연간 보고기간이다. 접수 연도로 바꾸지 않는다.",
    },
    {
        "source_query_id": "V2-00126955-overview_paragraph-01",
        "scenario": "search_required",
        "expected_behavior": "SEARCH",
        "allowed_behaviors": _SEARCH_ALLOWED,
        "prohibited_behaviors": ["DIRECT_ANSWER", "CLARIFY"],
        "search_expectation": "required",
        "period": None,
        "evaluation_notes": "2023년 7월은 연혁에 적힌 사건 시점이다. 그 해를 보고서 필터로 요구하지 않는다.",
    },
    {
        "source_query_id": "V2-00126955-finance_table-01",
        "scenario": "search_required",
        "expected_behavior": "SEARCH",
        "allowed_behaviors": _SEARCH_ALLOWED,
        "prohibited_behaviors": ["DIRECT_ANSWER", "CLARIFY"],
        "search_expectation": "required",
        "period": {"year": 2026, "half": 1},
        "evaluation_notes": "2026년 상반기 자본변동표의 배당이다.",
    },
    {
        "source_query_id": "V2-00989619-business_paragraph-01",
        "scenario": "search_required",
        "expected_behavior": "SEARCH",
        "allowed_behaviors": _SEARCH_ALLOWED,
        "prohibited_behaviors": ["DIRECT_ANSWER", "CLARIFY"],
        "search_expectation": "required",
        "period": None,
        "evaluation_notes": "라이선스아웃 매출 구분. 기간 필터는 필수가 아니다.",
    },
    {
        "source_query_id": "V2-00121941-business_paragraph-01",
        "scenario": "search_required",
        "expected_behavior": "SEARCH",
        "allowed_behaviors": _SEARCH_ALLOWED,
        "prohibited_behaviors": ["DIRECT_ANSWER", "CLARIFY"],
        "search_expectation": "required",
        "period": None,
        "evaluation_notes": "투기적 외환거래 규정. 기간 필터는 필수가 아니다.",
    },
    {
        "source_query_id": "V2-00107613-partial-02",
        "scenario": "insufficient_evidence",
        "expected_behavior": "PARTIAL_ANSWER",
        "allowed_behaviors": _SEARCH_ALLOWED,
        "prohibited_behaviors": ["DIRECT_ANSWER", "CLARIFY"],
        "search_expectation": "required",
        "period": {"year": 2025, "half": 1},
        "evaluation_notes": "2025년 상반기 종속회사 매출은 확인할 수 있다. 2018년 연결 매출은 색인 밖이다.",
    },
    {
        "source_query_id": "V2-00121941-partial-02",
        "scenario": "insufficient_evidence",
        "expected_behavior": "PARTIAL_ANSWER",
        "allowed_behaviors": _SEARCH_ALLOWED,
        "prohibited_behaviors": ["DIRECT_ANSWER", "CLARIFY"],
        "search_expectation": "required",
        "period": {"year": 2026, "quarter": 1},
        "evaluation_notes": "2026년 1분기 비유동자산은 질문의 확인 가능한 부분이다. 2018년 매출은 단정하지 않는다.",
    },
    {
        "source_query_id": "V2-00107613-multi_element-01",
        "scenario": "additional_search",
        "expected_behavior": "SEARCH",
        "allowed_behaviors": _SEARCH_ALLOWED,
        "prohibited_behaviors": ["DIRECT_ANSWER", "CLARIFY"],
        "search_expectation": "required",
        "period": {"year": 2024},
        "evaluation_notes": "같은 2024년 반기 근거의 가정과 표시 기준을 함께 묻는다. 추가 검색은 허용이지 필수가 아니다.",
    },
    {
        "source_query_id": "V2-00107613-cross_period-02",
        "scenario": "additional_search",
        "expected_behavior": "SEARCH",
        "allowed_behaviors": _SEARCH_ALLOWED,
        "prohibited_behaviors": ["DIRECT_ANSWER", "CLARIFY"],
        "search_expectation": "required",
        "period": {"year": 2024},
        "evaluation_notes": "반기와 3분기 두 공시를 묻는다. 한 보고서 종류만 강제하지 않는다.",
    },
    {
        "source_query_id": "V2-00126380-multi_element-02",
        "scenario": "additional_search",
        "expected_behavior": "SEARCH",
        "allowed_behaviors": _SEARCH_ALLOWED,
        "prohibited_behaviors": ["DIRECT_ANSWER", "CLARIFY"],
        "search_expectation": "required",
        "period": {"year": 2026, "quarter": 1},
        "evaluation_notes": "하만의 두 매입 항목이다. 추가 검색은 허용이다.",
    },
    {
        "source_query_id": "V2-00126955-cross_period-01",
        "scenario": "additional_search",
        "expected_behavior": "SEARCH",
        "allowed_behaviors": _SEARCH_ALLOWED,
        "prohibited_behaviors": ["DIRECT_ANSWER", "CLARIFY"],
        "search_expectation": "required",
        "period": None,
        "evaluation_notes": "제62기부터 제64기까지다. 한 달력 연도로 좁히지 않아도 된다.",
    },
    {
        "source_query_id": "V2-00121941-multi_element-01",
        "scenario": "additional_search",
        "expected_behavior": "SEARCH",
        "allowed_behaviors": _SEARCH_ALLOWED,
        "prohibited_behaviors": ["DIRECT_ANSWER", "CLARIFY"],
        "search_expectation": "required",
        "period": None,
        "evaluation_notes": "제73기 반기와 제72기 연간의 생산실적이다. 사건 연도를 접수 연도 필터로 요구하지 않는다.",
    },
)

AUTHORED: tuple[dict[str, Any], ...] = (
    {
        "query_id": "A0-direct-01",
        "question": "이 서비스는 어떤 기능을 제공하나요?",
        "scenario": "no_search",
        "expected_behavior": "DIRECT_ANSWER",
        "allowed_behaviors": ["DIRECT_ANSWER"],
        "prohibited_behaviors": ["SEARCH", "CLARIFY", "ABSTAIN"],
        "search_expectation": "prohibited",
        "company_id": None,
        "company_name": None,
        "period": None,
        "answerability": "not_applicable",
        "required_evidence_references": [],
        "requirements": [],
        "evaluation_notes": "공시 사실이 필요 없는 제품 설명이다.",
    },
    {
        "query_id": "A0-direct-02",
        "question": "안녕하세요",
        "scenario": "no_search",
        "expected_behavior": "DIRECT_ANSWER",
        "allowed_behaviors": ["DIRECT_ANSWER"],
        "prohibited_behaviors": ["SEARCH", "CLARIFY", "ABSTAIN"],
        "search_expectation": "prohibited",
        "company_id": None,
        "company_name": None,
        "period": None,
        "answerability": "not_applicable",
        "required_evidence_references": [],
        "requirements": [],
        "evaluation_notes": "인사다. 기업을 추정해 검색하지 않는다.",
    },
    {
        "query_id": "A0-direct-03",
        "question": "오늘 서울 날씨는 어때요?",
        "scenario": "no_search",
        "expected_behavior": "DIRECT_ANSWER",
        "allowed_behaviors": ["DIRECT_ANSWER", "ABSTAIN"],
        "prohibited_behaviors": ["SEARCH", "CLARIFY"],
        "search_expectation": "prohibited",
        "company_id": None,
        "company_name": None,
        "period": None,
        "answerability": "not_applicable",
        "required_evidence_references": [],
        "requirements": [],
        "evaluation_notes": "공시로 답할 질문이 아니다. 날씨 수치를 공시에서 만들지 않으면 된다.",
    },
    {
        "query_id": "A0-direct-04",
        "question": "고마워요. 오늘은 공시 질문이 없어요.",
        "scenario": "no_search",
        "expected_behavior": "DIRECT_ANSWER",
        "allowed_behaviors": ["DIRECT_ANSWER"],
        "prohibited_behaviors": ["SEARCH", "CLARIFY"],
        "search_expectation": "prohibited",
        "company_id": None,
        "company_name": None,
        "period": None,
        "answerability": "not_applicable",
        "required_evidence_references": [],
        "requirements": [],
        "evaluation_notes": "대화 종료에 가깝다. 회사를 물어 검색으로 넘기지 않는다.",
    },
    {
        "query_id": "A0-direct-05",
        "question": "공시를 읽지 말고, 이 시스템이 답을 만들 때 어떤 원칙을 쓰는지 설명해 주세요.",
        "scenario": "no_search",
        "expected_behavior": "DIRECT_ANSWER",
        "allowed_behaviors": ["DIRECT_ANSWER"],
        "prohibited_behaviors": ["SEARCH", "CLARIFY"],
        "search_expectation": "prohibited",
        "company_id": None,
        "company_name": None,
        "period": None,
        "answerability": "not_applicable",
        "required_evidence_references": [],
        "requirements": [],
        "evaluation_notes": "공시라는 단어가 있지만 특정 기업 사실을 묻지 않는다.",
    },
    {
        "query_id": "A0-clarify-01",
        "question": "그 회사의 매출이 왜 늘었나요?",
        "scenario": "ambiguous",
        "expected_behavior": "CLARIFY",
        "allowed_behaviors": ["CLARIFY"],
        "prohibited_behaviors": ["SEARCH", "DIRECT_ANSWER", "ABSTAIN"],
        "search_expectation": "prohibited",
        "company_id": None,
        "company_name": None,
        "period": None,
        "answerability": "needs_clarification",
        "required_evidence_references": [],
        "requirements": [],
        "evaluation_notes": "선행 대화 없이 회사가 지정되지 않았다. 회사를 추정해 검색하지 않는다.",
    },
    {
        "query_id": "A0-clarify-02",
        "question": "매출액이 얼마야?",
        "scenario": "ambiguous",
        "expected_behavior": "CLARIFY",
        "allowed_behaviors": ["CLARIFY"],
        "prohibited_behaviors": ["SEARCH", "DIRECT_ANSWER", "ABSTAIN"],
        "search_expectation": "prohibited",
        "company_id": None,
        "company_name": None,
        "period": None,
        "answerability": "needs_clarification",
        "required_evidence_references": [],
        "requirements": [],
        "evaluation_notes": "회사와 기간이 없다.",
    },
    {
        "query_id": "A0-clarify-03",
        "question": "삼성 실적 알려줘",
        "scenario": "ambiguous",
        "expected_behavior": "CLARIFY",
        "allowed_behaviors": ["CLARIFY"],
        "prohibited_behaviors": ["SEARCH", "DIRECT_ANSWER", "ABSTAIN"],
        "search_expectation": "prohibited",
        "company_id": None,
        "company_name": None,
        "period": None,
        "answerability": "needs_clarification",
        "required_evidence_references": [],
        "requirements": [],
        "evaluation_notes": "삼성으로 시작하는 기업이 여러 곳이다. 삼성전자로 단정하지 않는다.",
    },
    {
        "query_id": "A0-clarify-04",
        "question": "최근 실적 좀 볼 수 있을까요?",
        "scenario": "ambiguous",
        "expected_behavior": "CLARIFY",
        "allowed_behaviors": ["CLARIFY"],
        "prohibited_behaviors": ["SEARCH", "DIRECT_ANSWER", "ABSTAIN"],
        "search_expectation": "prohibited",
        "company_id": None,
        "company_name": None,
        "period": None,
        "answerability": "needs_clarification",
        "required_evidence_references": [],
        "requirements": [],
        "evaluation_notes": "최근 실적은 회사가 있어야 검색할 수 있다.",
    },
    {
        "query_id": "A0-clarify-05",
        "question": "페타시스 말고 그 반도체 회사 2025년 매출은?",
        "scenario": "ambiguous",
        "expected_behavior": "CLARIFY",
        "allowed_behaviors": ["CLARIFY"],
        "prohibited_behaviors": ["SEARCH", "DIRECT_ANSWER", "ABSTAIN"],
        "search_expectation": "prohibited",
        "company_id": None,
        "company_name": None,
        "period": None,
        "answerability": "needs_clarification",
        "required_evidence_references": [],
        "requirements": [],
        "evaluation_notes": "제외 대상만 있고 대상 회사는 특정되지 않는다.",
    },
    {
        "query_id": "A0-abstain-01",
        "question": "이수페타시스의 2019년 연결 매출액은 얼마인가요?",
        "scenario": "insufficient_evidence",
        "expected_behavior": "ABSTAIN",
        "allowed_behaviors": ["ABSTAIN", "SEARCH", "PARTIAL_ANSWER", "ADDITIONAL_SEARCH"],
        "prohibited_behaviors": ["DIRECT_ANSWER", "CLARIFY"],
        "search_expectation": "required",
        "company_id": "00107613",
        "company_name": "이수페타시스",
        "period": {"year": 2019},
        "answerability": "unanswerable",
        "required_evidence_references": [],
        "requirements": [
            {
                "requirement_id": "R1",
                "text": "2019년 연결 매출액 수치를 제시하지 않고, 현재 색인에서 확인하지 못했다고 말한다.",
                "success_verdicts": ["covered", "abstained"],
            }
        ],
        "evaluation_notes": "2019년은 질문의 보고기간이다. 색인에 그 기간이 없으면 수치를 만들지 않는다.",
    },
    {
        "query_id": "A0-abstain-02",
        "question": "이수페타시스의 2030년 확정 연결 매출은 얼마인가요?",
        "scenario": "insufficient_evidence",
        "expected_behavior": "ABSTAIN",
        "allowed_behaviors": ["ABSTAIN", "DIRECT_ANSWER"],
        "prohibited_behaviors": ["SEARCH", "CLARIFY"],
        "search_expectation": "prohibited",
        "company_id": "00107613",
        "company_name": "이수페타시스",
        "period": None,
        "answerability": "unanswerable",
        "required_evidence_references": [],
        "requirements": [
            {
                "requirement_id": "R1",
                "text": "2030년 확정 연결 매출 수치를 제시하지 않는다.",
                "success_verdicts": ["covered", "abstained"],
            }
        ],
        "evaluation_notes": "아직 실현되지 않은 확정 실적이다. 검색 없이 거절해도 되고, 수치를 만들면 실패다.",
    },
    {
        "query_id": "A0-abstain-03",
        "question": "카카오의 2024년 사업보고서 연결 매출액은 얼마인가요?",
        "scenario": "insufficient_evidence",
        "expected_behavior": "ABSTAIN",
        "allowed_behaviors": ["ABSTAIN", "CLARIFY"],
        "prohibited_behaviors": ["SEARCH", "DIRECT_ANSWER"],
        "search_expectation": "prohibited",
        "company_id": None,
        "company_name": None,
        "period": None,
        "answerability": "unanswerable",
        "required_evidence_references": [],
        "requirements": [
            {
                "requirement_id": "R1",
                "text": "색인되지 않은 기업의 2024년 매출 수치를 제시하지 않는다.",
                "success_verdicts": ["covered", "abstained"],
            }
        ],
        "evaluation_notes": "카카오는 이 제품 색인의 검색 대상이 아니다. 다른 회사 공시로 대신 답하지 않는다.",
    },
)


def dataset_path(root: Path) -> Path:
    return root / "evaluation" / "datasets" / DATASET_DIR / QUESTIONS_NAME


def build_rows(root: Path) -> list[dict[str, Any]]:
    questions = _questions(root)
    requirements = _requirements(root)
    rows = [_reused_row(spec, questions, requirements) for spec in REUSED]
    rows.extend(dict(item) for item in AUTHORED)
    _validate(rows, questions)
    return rows


def dataset_sha256(rows: list[dict[str, Any]]) -> str:
    body = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def write_dataset(root: Path) -> dict[str, Any]:
    rows = build_rows(root)
    path = dataset_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")
    digest = dataset_sha256(rows)
    meta = {
        "dataset_version": DATASET_VERSION,
        "dataset_sha256": digest,
        "query_count": len(rows),
        "scenarios": _counts(rows),
        "path": path.relative_to(root).as_posix(),
    }
    (path.parent / "dataset.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return meta


def load_dataset(root: Path) -> list[dict[str, Any]]:
    path = dataset_path(root)
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if dataset_sha256(rows) != dataset_sha256(build_rows(root)):
        raise ValueError("A0 dataset file does not match the builder")
    return rows


def agent_input(row: dict[str, Any]) -> str:
    """The only string the product agent receives for one question."""
    return str(row["question"])


def _reused_row(
    spec: dict[str, Any],
    questions: dict[str, dict[str, Any]],
    requirements: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    source_id = str(spec["source_query_id"])
    source = questions[source_id]
    if source.get("split") != "dev":
        raise ValueError(f"{source_id} is not a DEV question")
    copied = []
    for item in requirements.get(source_id, []):
        text = str(item["text"])
        copied.append(
            {
                "requirement_id": item["requirement_id"],
                "text": text,
                "success_verdicts": ["covered", "abstained"]
                if any(phrase in text for phrase in _WITHHOLD)
                else ["covered"],
            }
        )
    if not copied:
        raise ValueError(f"{source_id} has no requirements")
    return {
        "query_id": source_id,
        "question": source["question"],
        "scenario": spec["scenario"],
        "expected_behavior": spec["expected_behavior"],
        "allowed_behaviors": list(spec["allowed_behaviors"]),
        "prohibited_behaviors": list(spec["prohibited_behaviors"]),
        "search_expectation": spec["search_expectation"],
        "company_id": source["corp_code"],
        "company_name": source["company_name"],
        "period": spec["period"],
        "answerability": source["answerability"],
        "required_evidence_references": list(source.get("seed_unit_ids") or []),
        "requirements": copied,
        "evaluation_notes": spec["evaluation_notes"],
        "source_query_id": source_id,
        "source_split": source["split"],
    }


def _validate(rows: list[dict[str, Any]], questions: dict[str, dict[str, Any]]) -> None:
    if not 25 <= len(rows) <= 30:
        raise ValueError(f"A0 dataset has {len(rows)} rows; expected 25 to 30")
    seen: set[str] = set()
    for row in rows:
        query_id = row["query_id"]
        if query_id in seen:
            raise ValueError(f"duplicate query_id {query_id}")
        seen.add(query_id)
        for name in ("allowed_behaviors", "prohibited_behaviors"):
            unknown = set(row[name]) - set(_BEHAVIORS)
            if unknown:
                raise ValueError(f"{query_id} has unknown behaviors {sorted(unknown)}")
        if row["expected_behavior"] not in row["allowed_behaviors"]:
            raise ValueError(f"{query_id} expected behavior is not allowed")
        source_id = row.get("source_query_id")
        if source_id is not None and questions[source_id]["split"] != "dev":
            raise ValueError(f"{source_id} is not DEV")
    counts = _counts(rows)
    for scenario, minimum in (
        ("search_required", 10),
        ("no_search", 5),
        ("ambiguous", 5),
        ("insufficient_evidence", 5),
        ("additional_search", 5),
    ):
        if counts.get(scenario, 0) < minimum:
            raise ValueError(f"{scenario} has {counts.get(scenario, 0)} rows")


def _counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["scenario"]] = counts.get(row["scenario"], 0) + 1
    return counts


def _questions(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "evaluation" / "datasets" / "multi_company_retrieval_v2_1" / "questions.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {item["query_id"]: item for item in payload["questions"]}


def _requirements(root: Path) -> dict[str, list[dict[str, Any]]]:
    path = root / "evaluation" / "datasets" / "multi_company_retrieval_v2_1" / "requirements.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in payload["requirements"]:
        grouped.setdefault(item["query_id"], []).append(item)
    return grouped

"""G0 minimal grounded generation over frozen H10 evidence.

This module does not search, rerank, rewrite queries, or run an agent loop.
One query gets the stored H10 top-10 and, later, one generation call.
Gold qrels are used only after generation, to separate retrieval-limited
requirements from generation-limited ones. They are not model input.
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from disclosure_agent.chunking import canonical_dumps, sha256_text
from disclosure_agent.errors import UsageError
from disclosure_agent.evalset_v2_1 import DATASET_ID
from disclosure_agent.evalset_v2_1.metrics import score_question

RUN_DIR_NAME = "multi_company_retrieval_v2_1_dev_g0_grounded_generation"
H10_RUN_DIR = "multi_company_retrieval_v2_1_dev_h10_asymmetric"
RETRIEVAL_POLICY = "bm25_primary_dense_head"
TOP_K = 10
PROTECTED_PREFIX = 9
BASELINE_ID = "G0"
GENERATOR_PROMPT_VERSION = "g0-generator-v1"
FAITHFULNESS_PROMPT_VERSION = "g0-faithfulness-v1"
REQUIREMENT_PROMPT_VERSION = "g0-requirements-v1"
CITATION_PROMPT_VERSION = "g0-citations-v1"
EVALUATOR_VERSION = "g0-answer-evaluation-v1"
LOCKED_HIT10 = (43, 45)
LOCKED_REQUIREMENT_RECALL = (52, 58)
LOCKED_ALL_REQUIREMENTS = (42, 45)
EXCERPT_CHARS = 360
DECODING = {
    "temperature": None,
    "top_p": None,
    "max_output_tokens": None,
    "response_format": "json_schema",
    "store": False,
    "note": "provider default; temperature is not sent; retries repeat the same prompt",
}
FACTUAL_VERDICTS = frozenset({"supported", "unsupported", "misinterpretation"})
OFFICIAL_FLAGS = (
    "retrieval_failure",
    "evidence_coverage_failure",
    "generation_omission",
    "unsupported_claim",
    "evidence_misinterpretation",
    "period_scope_confusion",
    "citation_missing",
    "citation_mismatch",
    "citation_hallucination",
    "over_abstention",
    "under_abstention",
)
GENERATION_FLAGS = frozenset(
    {
        "generation_omission",
        "unsupported_claim",
        "evidence_misinterpretation",
        "period_scope_confusion",
        "citation_missing",
        "citation_mismatch",
        "citation_hallucination",
        "over_abstention",
        "under_abstention",
    }
)
FORBIDDEN_PAYLOAD_KEYS = frozenset(
    {
        "adjudicated",
        "answerability",
        "bm25_rank",
        "char_end",
        "char_start",
        "difficulty_tags",
        "element_id",
        "final_source",
        "gold",
        "gold_answer",
        "is_dense_head",
        "judgment",
        "judgement",
        "label",
        "preferred",
        "qrel",
        "qrels",
        "relevance",
        "seed_evidence_ids",
        "seed_unit_ids",
        "source_refs",
    }
)
MODEL_EVIDENCE_KEYS = (
    "citation_id",
    "retrieval_unit_id",
    "filing_company",
    "document_name",
    "report_name",
    "report_type",
    "filing_period",
    "period_end_year",
    "period_end_month",
    "receipt_date",
    "receipt_no",
    "section_path",
    "text",
)
CITATION_ID_RE = re.compile(r"C[1-9][0-9]*")
MARKER_RE = re.compile(r"\[([^\[\]]+)\]")
CLAIM_ID_RE = re.compile(r"K[1-9][0-9]*")
FAILURE_COLUMNS = (
    "query_id",
    "requirement_id",
    "primary",
    "flags",
    "coverage",
    "evidence_in_top10",
    "support",
    "scope_issue",
    "citation_missing",
    "citation_mismatch",
    "citation_hallucination",
)

GENERATOR_INSTRUCTIONS = """\
당신은 금융 공시 근거에만 묶인 답변을 작성합니다.
이 호출이 이 질문의 유일한 생성입니다.

1. 입력 evidence의 text만 사실의 근거로 사용합니다.
2. evidence에서 직접 확인할 수 없는 사실은 만들지 않습니다.
3. 입력 requirements를 모두 답하려고 시도합니다.
4. 기간, 회사, 연결/별도, 보고서 범위를 서로 다른 evidence 사이에서 섞지 않습니다.
5. 공시가 원인을 설명하지 않으면 원인을 만들지 않습니다.
6. 검증할 수 있는 사실 주장에는 입력에 있는 citation_id만 붙입니다.
7. evidence가 질문의 일부만 지지하면 그 부분만 답하고, 확인할 수 없는 부분은 limitations에 적습니다.
8. citation_id는 입력 evidence에 있는 값만 사용할 수 있습니다.
새 id, 접수번호, 파일 경로, 문자 위치를 인용으로 만들지 않습니다.

answer는 사용자에게 보이는 문장입니다.
사실을 말하는 문장 끝에는 [C1]처럼 citation_id를 대괄호로 적습니다.
claims는 answer 안의 주장을 나눈 목록입니다. claim_id는 K1부터 순서대로 적습니다.
answer의 검증 가능한 문장은 빠짐없이 claims에 넣습니다.
factual이 true인 claim은 evidence로 확인하는 사실입니다.
확인할 수 없다는 문장은 factual을 false로 둡니다.
citation_ids에는 그 claim을 직접 지지하는 citation_id만 넣습니다. 없으면 빈 목록입니다.
투자 조언, 매수, 매도, 목표가는 만들지 않습니다.
"""

FAITHFULNESS_INSTRUCTIONS = """각 claim이 제공된 evidence만으로 지지되는지 판단합니다.
외부 지식, 정답 등급, 검색 점수는 없습니다. evidence에 없는 사실은 참이라고 채우지 않습니다.

verdict는 다음 중 하나입니다.
- supported: claim의 사실이 evidence 문장으로 직접 확인된다.
- unsupported: claim이 evidence에 없는 사실을 말한다.
- misinterpretation: 관련 문장은 있으나 숫자, 기간, 방향, 회사,
  연결/별도, 보고서 범위를 잘못 읽었다.
- not_factual: 확인 가능한 사실을 주장하지 않는다.

scope_issue는 none, period, company, consolidation, filing, unit, mixed 중 하나입니다.
문제가 없으면 none입니다.
입력 claim_id마다 item을 하나씩 반환합니다. claim_id를 새로 만들지 않습니다.
reason에는 그 판단의 근거를 짧게 적습니다.
"""

REQUIREMENT_INSTRUCTIONS = """답변이 각 requirement를 충족하는지 판단합니다.
evidence와 정답 등급은 입력에 없습니다. 답변 문장이 그 requirement가 묻는 내용을 말하는지만 봅니다.

verdict는 다음 중 하나입니다.
- covered: 답변이 그 requirement가 묻는 내용을 구체적으로 말한다.
- omitted: 답변이 그 requirement를 다루지 않는다.
- abstained: 답변이 그 requirement를 확인할 수 없다고 명시한다.

claim_ids는 그 requirement를 담은 입력 claim의 id입니다. 없으면 빈 목록입니다.
answer_relevance는 답변 전체에 대해 direct, partial, off_topic 중 하나입니다.
direct는 질문에 바로 답한 경우입니다. partial은 일부만 답한 경우입니다.
off_topic은 질문과 다른 내용을 주로 말한 경우입니다.
입력 requirement_id마다 item을 하나씩 반환합니다. requirement_id를 새로 만들지 않습니다.
reason에는 그 판단의 근거를 짧게 적습니다.
"""

CITATION_INSTRUCTIONS = """\
각 쌍에 대해, 그 citation의 evidence가 그 claim을 직접 지지하는지 판단합니다.
다른 evidence가 지지하는지는 보지 않습니다. 입력에 없는 문서는 없습니다.

verdict는 다음 중 하나입니다.
- supports: 이 evidence만으로 claim의 사실이 직접 확인된다.
- does_not_support: 이 evidence가 claim을 직접 지지하지 않는다.

입력에 주어진 쌍마다 item을 하나씩 반환합니다. 쌍을 추가하지 않습니다.
reason에는 그 판단의 근거를 짧게 적습니다.
"""

HYPOTHESES = {
    "retrieval_failure": (
        "가장 많은 requirement 실패는 retrieval_failure다. "
        "그 requirement의 관련 retrieval unit이 H10 Top-10에 없다. "
        "이 건수를 generation prompt 실패와 합치지 않는다. "
        "다음 실험은 evidence가 Top-10에 있는 requirement만 잘라 생성 실패율을 다시 보는 것이다. "
        "Retrieval 설정은 이 결과만으로 바꾸지 않는다."
    ),
    "evidence_coverage_failure": (
        "여러 requirement 중 일부만 Top-10에 관련 unit이 있다. "
        "다음 실험은 그 부분 결손을 생성 누락과 분리한 채, "
        "evidence가 있는 requirement의 답변 품질만 다시 측정하는 것이다."
    ),
    "generation_omission": (
        "관련 evidence는 H10 Top-10에 있는데 답이 requirement를 빠뜨렸다. "
        "다음 실험 후보는 생성 호출 1회를 유지하고, "
        "각 claim을 requirement_id에 연결하도록 출력 schema만 바꾸는 것이다. "
        "이번 baseline에서는 그 변경을 넣지 않았다."
    ),
    "unsupported_claim": (
        "답이 evidence에 없는 사실을 말한다. "
        "다음 실험 후보는 생성 호출 1회를 유지하고, "
        "claim이 evidence 문장 밖에 사실을 더하지 못하게 출력 제약을 하나 더하는 것이다. "
        "이번 baseline에서는 그 변경을 넣지 않았다."
    ),
    "evidence_misinterpretation": (
        "evidence는 있으나 숫자, 기간, 방향, 범위를 잘못 읽는다. "
        "이 실패가 반복되므로 다음 실험 후보는 그 유형에 한정한 "
        "숫자/기간/단위 확인이다. 이번 baseline에서는 그 확인기를 넣지 않았다."
    ),
    "period_scope_confusion": (
        "기간, 회사, 연결/별도, 보고서 범위가 섞인다. "
        "다음 실험 후보는 이 범위 오류만 겨냥한 단일 확인이다. "
        "이번 baseline에서는 그 확인기를 넣지 않았다."
    ),
    "citation_missing": (
        "사실이 claim인데 유효한 citation이 없다. "
        "다음 실험 후보는 factual claim에 citation_id를 요구하는 schema 제약 하나다. "
        "이번 baseline에서는 그 변경을 넣지 않았다."
    ),
    "citation_mismatch": (
        "citation은 있으나 그 evidence가 claim을 지지하지 않는다. "
        "다음 실험 후보는 claim과 citation의 대응만 바꾸는 것이다. "
        "이번 baseline에서는 그 변경을 넣지 않았다."
    ),
    "citation_hallucination": (
        "입력 context에 없는 citation이 나왔다. "
        "다음 실험 후보는 허용 citation_id 집합을 schema enum으로 고정하는 것이다. "
        "이번 baseline에서는 그 변경을 넣지 않았다."
    ),
    "over_abstention": (
        "관련 evidence가 Top-10에 있는데 답이 확인 불가로 물러난다. "
        "다음 실험 후보는 그 기권 조건만 좁히는 단일 변경이다. "
        "이번 baseline에서는 그 변경을 넣지 않았다."
    ),
    "under_abstention": (
        "관련 unit이 Top-10에 없는데 답이 단정한다. "
        "다음 실험 후보는 이 과단정만 세는 것이다. "
        "Retrieval 변경이나 prompt 반복은 이번 결과에 바로 넣지 않는다."
    ),
}


class G0Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(description="K1, K2, ... in order. Do not invent another scheme.")
    text: str
    citation_ids: list[str] = Field(description="citation_id values copied from the input.")
    factual: bool


class G0Generation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    claims: list[G0Claim]
    limitations: list[str]


class FaithfulnessItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    verdict: str = Field(description="supported, unsupported, misinterpretation, or not_factual.")
    scope_issue: str = Field(
        description="none, period, company, consolidation, filing, unit, or mixed."
    )
    reason: str


class FaithfulnessOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[FaithfulnessItem]
    reason: str


class RequirementItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requirement_id: str
    verdict: str = Field(description="covered, omitted, or abstained.")
    claim_ids: list[str]
    reason: str


class RequirementOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[RequirementItem]
    answer_relevance: str = Field(description="direct, partial, or off_topic.")
    reason: str


class CitationItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    citation_id: str
    verdict: str = Field(description="supports or does_not_support.")
    reason: str


class CitationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[CitationItem]
    reason: str


@dataclass(frozen=True)
class RequirementSignals:
    evidence_in_top10: bool
    coverage: str
    support: str
    scope_issue: str
    citation_missing: bool
    citation_mismatch: bool
    citation_hallucination: bool


def citation_id_for_rank(rank: int) -> str:
    if rank < 1:
        raise UsageError(f"citation rank must start at 1, got {rank}")
    return f"C{rank}"


def prompt_sha256(instructions: str) -> str:
    return sha256_text(instructions)


def call_input_hash(
    *,
    stage: str,
    provider: str,
    model: str,
    prompt_version: str,
    instructions: str,
    payload: dict[str, Any],
    schema_name: str,
) -> str:
    body = {
        "stage": stage,
        "provider": provider,
        "model": model,
        "prompt_version": prompt_version,
        "instructions_sha256": prompt_sha256(instructions),
        "schema": schema_name,
        "payload": payload,
    }
    return sha256_text(canonical_dumps(body))


def forbidden_keys(payload: Any) -> list[str]:
    found: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in FORBIDDEN_PAYLOAD_KEYS:
                    found.append(str(key))
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(payload)
    return found


def assert_no_gold(payload: Any) -> None:
    found = forbidden_keys(payload)
    if found:
        raise UsageError("model payload contains " + found[0])


def resolve_g0_models(environ: dict[str, str]) -> dict[str, Any]:
    """One generator and one judge. Names come from the environment."""
    provider = environ.get("G0_PROVIDER", "").strip() or "openai"
    if provider != "openai":
        raise UsageError("G0 uses one provider. Set G0_PROVIDER=openai or leave it empty.")
    fallback = environ.get("OPENAI_MODEL", "").strip()
    generator = environ.get("G0_GENERATOR_MODEL", "").strip() or fallback
    if not generator:
        raise UsageError("G0 generator model is empty. Set OPENAI_MODEL or G0_GENERATOR_MODEL.")
    judge = environ.get("G0_JUDGE_MODEL", "").strip() or generator
    timeout_raw = environ.get("G0_TIMEOUT_SECONDS", "").strip()
    workers_raw = environ.get("G0_WORKERS", "").strip()
    try:
        timeout = float(timeout_raw) if timeout_raw else 180.0
        workers = int(workers_raw) if workers_raw else 4
    except ValueError as exc:
        raise UsageError("G0_TIMEOUT_SECONDS and G0_WORKERS must be numbers") from exc
    if timeout <= 0 or workers < 1:
        raise UsageError("G0 timeout must be > 0 and G0_WORKERS must be >= 1")
    return {
        "provider": provider,
        "generator_model": generator,
        "judge_model": judge,
        "timeout_seconds": timeout,
        "workers": workers,
        "api_key_present": bool(environ.get("OPENAI_API_KEY", "").strip()),
        "decoding": dict(DECODING),
        "max_transport_retries": 3,
    }


def retrieval_units_path(root: Path, corp_code: str) -> Path:
    if corp_code == "00107613":
        return (
            root
            / "data"
            / "product"
            / "indexes"
            / "isu_petasys_periodic_d2a"
            / "retrieval_units.jsonl"
        )
    return (
        root
        / "data"
        / "product"
        / "indexes"
        / "generalization"
        / corp_code
        / "retrieval_units.jsonl"
    )


def check_stored_h10_ranking(root: Path) -> dict[str, Any]:
    """Confirm the stored DEV ranking is still H10. This does not search."""
    dataset = _dataset_dir(root)
    freeze = _read_json(dataset / "freeze.json")
    questions = _read_json(dataset / "questions.json")["questions"]
    requirements = _read_json(dataset / "requirements.json")["requirements"]
    qrels = _read_json(dataset / "qrels.json")["qrels"]
    h10_dir = root / "evaluation" / "runs" / H10_RUN_DIR
    config = _read_json(h10_dir / "experiment_config.json")
    stored = _read_json(h10_dir / "h10_dev.json")
    transitions = _read_json(h10_dir / "per_query_transition.json")["queries"]
    if freeze.get("dataset_id") != DATASET_ID:
        raise UsageError("freeze dataset_id is not v2.1")
    if config.get("dataset_sha256") != freeze.get("dataset_sha256"):
        raise UsageError("H10 artifact dataset hash differs from the frozen dataset")
    if config.get("rule_id") != RETRIEVAL_POLICY:
        raise UsageError("stored run is not bm25_primary_dense_head")
    if config.get("retrieval_reexecuted") is not False:
        raise UsageError("stored H10 config does not say retrieval_reexecuted false")
    if int(config.get("test_questions_searched") or 0) != 0:
        raise UsageError("stored H10 config searched test questions")
    dev = [
        row
        for row in questions
        if row.get("answerability") == "fully_answerable" and row.get("split") == "dev"
    ]
    test = [
        row
        for row in questions
        if row.get("answerability") == "fully_answerable" and row.get("split") == "test"
    ]
    if len(dev) != int(freeze["dev_size"]) or len(dev) != LOCKED_HIT10[1]:
        raise UsageError("DEV fully_answerable count is not the frozen 45")
    if len(test) != int(freeze["test_size"]):
        raise UsageError("TEST fully_answerable count drifted")
    dev_ids = {str(row["query_id"]) for row in dev}
    if dev_ids & {str(row["query_id"]) for row in test}:
        raise UsageError("DEV and TEST fully_answerable ids overlap")
    by_stored = {str(row["query_id"]): row for row in stored["questions"]}
    by_transition = {str(row["query_id"]): row for row in transitions}
    if set(by_stored) != dev_ids or set(by_transition) != dev_ids:
        raise UsageError("stored H10 queries are not the DEV fully_answerable 45")
    qrels_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in qrels:
        qrels_by_query[str(row["query_id"])].append(row)
    scored_rows = []
    for question in dev:
        query_id = str(question["query_id"])
        hits = by_stored[query_id]["hits"]
        if len(hits) < TOP_K:
            raise UsageError(f"{query_id} stored H10 ranking is shorter than 10")
        top = hits[:TOP_K]
        if [int(hit["rank"]) for hit in top] != list(range(1, TOP_K + 1)):
            raise UsageError(f"{query_id} stored H10 ranks are not 1..10")
        ids = [str(hit["retrieval_unit_id"]) for hit in top]
        if len(ids) != len(set(ids)):
            raise UsageError(f"{query_id} repeats a retrieval unit inside H10 top-10")
        transition = by_transition[query_id]
        transition_ids = [str(hit["retrieval_unit_id"]) for hit in transition["top10"]]
        if ids != transition_ids:
            raise UsageError(f"{query_id} h10_dev top-10 differs from the transition top-10")
        _assert_h10_prefix(query_id, transition)
        scored_rows.append(
            score_question(question, qrels_by_query.get(query_id) or [], requirements, top)
        )
    headline = stored["headline"]
    _assert_lock(
        "Hit@10",
        _sum_hits(scored_rows),
        int(headline["at_k"]["10"]["hit"]["numerator"]),
        LOCKED_HIT10,
    )
    _assert_lock(
        "requirement_recall@10",
        sum(int(row["requirement_covered@10"]) for row in scored_rows),
        int(headline["requirement_recall@10"]["numerator"]),
        LOCKED_REQUIREMENT_RECALL,
    )
    _assert_lock(
        "all_requirements_covered@10",
        sum(1 for row in scored_rows if row["all_requirements_covered@10"]),
        int(headline["all_requirements_covered@10"]["numerator"]),
        LOCKED_ALL_REQUIREMENTS,
    )
    return {
        "retrieval_reexecuted": False,
        "test_questions_generated": 0,
        "test_questions_searched": 0,
        "dataset_id": DATASET_ID,
        "dataset_sha256": freeze["dataset_sha256"],
        "question_count": len(dev),
        "hit10": {"numerator": LOCKED_HIT10[0], "denominator": LOCKED_HIT10[1]},
        "requirement_recall": {
            "numerator": LOCKED_REQUIREMENT_RECALL[0],
            "denominator": LOCKED_REQUIREMENT_RECALL[1],
        },
        "all_requirements": {
            "numerator": LOCKED_ALL_REQUIREMENTS[0],
            "denominator": LOCKED_ALL_REQUIREMENTS[1],
        },
        "h10_run": H10_RUN_DIR,
        "retrieval_policy": RETRIEVAL_POLICY,
        "h10_dev_sha256": _sha_file(h10_dir / "h10_dev.json"),
        "transition_sha256": _sha_file(h10_dir / "per_query_transition.json"),
        "questions_sha256": _sha_file(dataset / "questions.json"),
        "requirements_sha256": _sha_file(dataset / "requirements.json"),
        "qrels_sha256": _sha_file(dataset / "qrels.json"),
    }


def load_g0_packets(root: Path) -> dict[str, Any]:
    """Join stored H10 ids to index text. The join is not a retrieval run."""
    verification = check_stored_h10_ranking(root)
    dataset = _dataset_dir(root)
    questions = _read_json(dataset / "questions.json")["questions"]
    requirements = _read_json(dataset / "requirements.json")["requirements"]
    qrels = _read_json(dataset / "qrels.json")["qrels"]
    h10_dir = root / "evaluation" / "runs" / H10_RUN_DIR
    stored = _read_json(h10_dir / "h10_dev.json")
    transitions = _read_json(h10_dir / "per_query_transition.json")["queries"]
    dev = [
        row
        for row in questions
        if row.get("answerability") == "fully_answerable" and row.get("split") == "dev"
    ]
    by_stored = {str(row["query_id"]): row for row in stored["questions"]}
    by_transition = {str(row["query_id"]): row for row in transitions}
    qrels_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in qrels:
        qrels_by_query[str(row["query_id"])].append(row)
    wanted: dict[str, set[str]] = defaultdict(set)
    for question in dev:
        corp = str(question["corp_code"])
        query_id = str(question["query_id"])
        for hit in by_stored[query_id]["hits"][:TOP_K]:
            wanted[corp].add(str(hit["retrieval_unit_id"]))
    units: dict[str, dict[str, Any]] = {}
    missing_files: list[str] = []
    for corp, unit_ids in wanted.items():
        path = retrieval_units_path(root, corp)
        if not path.is_file():
            missing_files.append(str(path))
            continue
        units.update(_scan_units(path, unit_ids))
    if missing_files:
        raise UsageError(
            "H10 unit text is missing, so generation cannot start. "
            "Retrieval was not re-run. Missing index: " + missing_files[0]
        )
    packets = []
    digit_requirements = 0
    for question in dev:
        query_id = str(question["query_id"])
        top = by_stored[query_id]["hits"][:TOP_K]
        transition_top = {
            str(hit["retrieval_unit_id"]): hit for hit in by_transition[query_id]["top10"]
        }
        evidence = []
        resolution = []
        for hit in top:
            unit_id = str(hit["retrieval_unit_id"])
            unit = units.get(unit_id)
            if unit is None:
                raise UsageError(
                    f"{query_id} retrieval unit {unit_id} is not in the stored index. "
                    "Retrieval was not re-run."
                )
            if sorted(_strings(hit.get("source_element_ids"))) != sorted(
                _strings(unit.get("source_element_ids"))
            ):
                raise UsageError(f"{query_id} {unit_id} source elements differ from the H10 hit")
            if not str(unit.get("text") or "").strip():
                raise UsageError(f"{query_id} {unit_id} has empty retrieval text")
            evidence.append(_model_evidence(hit, unit, str(question.get("company_name") or "")))
            resolution.append(_resolution(hit, unit, transition_top[unit_id]))
        owned_requirements = [
            row
            for row in requirements
            if row["query_id"] == query_id and row.get("answerability") == "fully_answerable"
        ]
        public_requirements = [
            {"requirement_id": str(row["requirement_id"]), "text": str(row["text"])}
            for row in owned_requirements
        ]
        digit_requirements += sum(
            1
            for row in public_requirements
            if any(character.isdigit() for character in row["text"])
        )
        relevant = _relevant_units(qrels_by_query.get(query_id) or [])
        evidence_flags = {
            row["requirement_id"]: bool(relevant.get(row["requirement_id"], set()) & {
                item["retrieval_unit_id"] for item in evidence
            })
            for row in public_requirements
        }
        payload = {
            "question": str(question["question"]),
            "requirements": public_requirements,
            "evidence": evidence,
        }
        assert_no_gold(payload)
        packets.append(
            {
                "query_id": query_id,
                "question": str(question["question"]),
                "company_name": str(question.get("company_name") or ""),
                "corp_code": str(question["corp_code"]),
                "query_type": question.get("query_type"),
                "evidence_form": question.get("evidence_form"),
                "stratum_id": question.get("stratum_id"),
                "period_sensitive": bool(question.get("period_sensitive")),
                "difficulty_tags": list(question.get("difficulty_tags") or []),
                "requirements": public_requirements,
                "model_payload": payload,
                "resolution": resolution,
                "retrieval_unit_ids": [row["retrieval_unit_id"] for row in evidence],
                "evidence_in_top10": evidence_flags,
                "context_token_count": sum(int(row.get("token_count") or 0) for row in resolution),
            }
        )
    verification = {
        **verification,
        "requirement_count": sum(len(packet["requirements"]) for packet in packets),
        "requirement_texts_containing_digits": digit_requirements,
        "units_joined": sum(len(packet["retrieval_unit_ids"]) for packet in packets),
        "text_join": "stored retrieval_unit_id to index text; retrieval was not executed",
    }
    if verification["requirement_count"] != LOCKED_REQUIREMENT_RECALL[1]:
        raise UsageError("DEV fully_answerable requirement count is not 58")
    return {"verification": verification, "packets": packets}


def build_model_evidence(rank: int, unit: dict[str, Any], filing_company: str) -> dict[str, Any]:
    hit = {"rank": rank, "retrieval_unit_id": unit["retrieval_unit_id"]}
    evidence = _model_evidence(hit, unit, filing_company)
    if tuple(evidence) != MODEL_EVIDENCE_KEYS:
        raise UsageError("model evidence keys drifted")
    return evidence


def citation_integrity(
    answer: str,
    claims: list[dict[str, Any]],
    valid_ids: set[str],
) -> dict[str, Any]:
    malformed: list[dict[str, str]] = []
    unknown: list[dict[str, str]] = []
    valid: list[dict[str, str]] = []
    for where, token in _emitted_citations(answer, claims):
        if CITATION_ID_RE.fullmatch(token) is None:
            malformed.append({"where": where, "token": token})
        elif token not in valid_ids:
            unknown.append({"where": where, "token": token})
        else:
            valid.append({"where": where, "token": token})
    duplicate_claims = [
        claim_id
        for claim_id, count in Counter(
            str(claim.get("claim_id") or "") for claim in claims
        ).items()
        if count > 1
    ]
    bad_claim_ids = [
        str(claim.get("claim_id") or "")
        for claim in claims
        if CLAIM_ID_RE.fullmatch(str(claim.get("claim_id") or "")) is None
    ]
    return {
        "malformed": malformed,
        "unknown": unknown,
        "valid": valid,
        "integrity_error_count": len(malformed) + len(unknown),
        "duplicate_claim_ids": duplicate_claims,
        "malformed_claim_ids": bad_claim_ids,
    }


def classify_requirement(signals: RequirementSignals) -> dict[str, Any]:
    """Exclusive primary, plus flags that may stack."""
    flags: list[str] = []
    if not signals.evidence_in_top10:
        primary = "retrieval_limited"
        flags.append("retrieval_failure")
        if signals.coverage == "covered" and signals.support in {
            "unsupported",
            "misinterpretation",
        }:
            flags.append("under_abstention")
        if signals.support == "unsupported":
            flags.append("unsupported_claim")
        if signals.support == "misinterpretation":
            flags.append("evidence_misinterpretation")
        if signals.coverage == "covered" and signals.support == "supported":
            flags.append("supported_without_qrel_unit")
    elif signals.coverage == "unjudged":
        primary = "unjudged"
    elif signals.coverage == "omitted":
        primary = "generation_limited"
        flags.append("generation_omission")
    elif signals.coverage == "abstained":
        primary = "generation_limited"
        flags.append("over_abstention")
    elif signals.coverage == "covered":
        if signals.support in {"no_claim", "unjudged"}:
            primary = "unjudged"
            if signals.support == "no_claim":
                flags.append("unlinked_coverage")
        elif signals.support == "misinterpretation":
            primary = "generation_limited"
            flags.append("evidence_misinterpretation")
        elif signals.support == "unsupported":
            primary = "generation_limited"
            flags.append("unsupported_claim")
        elif signals.support == "not_factual":
            primary = "generation_limited"
            flags.append("generation_omission")
        elif signals.support == "supported":
            primary = "success"
        else:
            primary = "unjudged"
    else:
        primary = "unjudged"
    if signals.scope_issue != "none":
        flags.append("period_scope_confusion")
        if primary == "success":
            primary = "generation_limited"
    if signals.citation_missing:
        flags.append("citation_missing")
    if signals.citation_mismatch:
        flags.append("citation_mismatch")
    if signals.citation_hallucination:
        flags.append("citation_hallucination")
    return {"primary": primary, "flags": flags}


def assemble_query_result(
    packet: dict[str, Any],
    generation: dict[str, Any],
    faithfulness: dict[str, Any],
    requirements_judgment: dict[str, Any],
    citation_judgment: dict[str, Any],
) -> dict[str, Any]:
    output = generation.get("output") if generation.get("parse_status") == "ok" else None
    answer = "" if output is None else str(output.get("answer") or "")
    claims = [] if output is None else [dict(claim) for claim in output.get("claims") or []]
    limitations = [] if output is None else [str(item) for item in output.get("limitations") or []]
    valid_ids = {row["citation_id"] for row in packet["model_payload"]["evidence"]}
    integrity = citation_integrity(answer, claims, valid_ids)
    if generation.get("parse_status") != "ok":
        integrity = {**integrity, "parse_status": "failed"}
    else:
        integrity = {**integrity, "parse_status": "ok"}
    faith = _score_faithfulness(claims, faithfulness, valid_ids)
    coverage = _score_requirements(packet["requirements"], requirements_judgment)
    citation = _score_citations(faith["items"], citation_judgment)
    mismatched = {
        item["claim_id"]
        for item in citation["items"]
        if item["verdict"] == "does_not_support"
    }
    by_claim = {item["claim_id"]: item for item in faith["items"]}
    classified = []
    for requirement in packet["requirements"]:
        requirement_id = requirement["requirement_id"]
        covered = _coverage_item(coverage, requirement_id)
        linked = [
            by_claim[claim_id] for claim_id in covered["claim_ids"] if claim_id in by_claim
        ]
        support, scope_issue = _support_of(linked)
        factual_linked = [item for item in linked if item["verdict"] in FACTUAL_VERDICTS]
        missing = bool(factual_linked) and any(
            not item["valid_citation_ids"] for item in factual_linked
        )
        mismatch = any(item["claim_id"] in mismatched for item in linked)
        hallucinated = any(item["citation_hallucination"] for item in linked)
        signals = RequirementSignals(
            evidence_in_top10=bool(packet["evidence_in_top10"][requirement_id]),
            coverage=str(covered["verdict"]),
            support=support,
            scope_issue=scope_issue,
            citation_missing=missing,
            citation_mismatch=mismatch,
            citation_hallucination=hallucinated,
        )
        decision = classify_requirement(signals)
        classified.append(
            {
                "requirement_id": requirement_id,
                "text": requirement["text"],
                "evidence_in_top10": signals.evidence_in_top10,
                "coverage": signals.coverage,
                "support": support,
                "scope_issue": scope_issue,
                "linked_claim_ids": covered["claim_ids"],
                "citation_missing": missing,
                "citation_mismatch": mismatch,
                "citation_hallucination": hallucinated,
                "primary": decision["primary"],
                "flags": decision["flags"],
                "reason": covered.get("reason") or "",
            }
        )
    query_flags = []
    retrieved = [row["evidence_in_top10"] for row in classified]
    if len(classified) >= 2 and any(retrieved) and not all(retrieved):
        query_flags.append("evidence_coverage_failure")
    if integrity["unknown"]:
        query_flags.append("citation_hallucination")
    if integrity["malformed"]:
        query_flags.append("malformed_citation")
    missing_citation = any(
        item["verdict"] in FACTUAL_VERDICTS and not item["valid_citation_ids"]
        for item in faith["items"]
    )
    if missing_citation:
        query_flags.append("citation_missing")
    return {
        "query_id": packet["query_id"],
        "question": packet["question"],
        "company_name": packet["company_name"],
        "query_type": packet["query_type"],
        "evidence_form": packet["evidence_form"],
        "stratum_id": packet["stratum_id"],
        "period_sensitive": packet["period_sensitive"],
        "difficulty_tags": packet["difficulty_tags"],
        "parse_status": generation.get("parse_status"),
        "answer": answer,
        "claims": claims,
        "limitations": limitations,
        "evidence": packet["model_payload"]["evidence"],
        "resolution": packet["resolution"],
        "retrieval_unit_ids": packet["retrieval_unit_ids"],
        "requirements": classified,
        "faithfulness": faith,
        "requirement_coverage": coverage,
        "citation": citation,
        "integrity": integrity,
        "query_flags": query_flags,
        "answer_relevance": coverage.get("answer_relevance"),
        "generation": _call_facts(generation),
        "judge_faithfulness": _call_facts(faithfulness),
        "judge_requirements": _call_facts(requirements_judgment),
        "judge_citations": _call_facts(citation_judgment),
    }


def summarize_results(
    results: list[dict[str, Any]], verification: dict[str, Any]
) -> dict[str, Any]:
    faith = _sum_ratio(results, "faithfulness", "supported", "factual", "unjudged")
    coverage = _sum_coverage(results)
    citation = _sum_citation(results)
    completeness = _sum_completeness(results)
    primaries = Counter(row["primary"] for result in results for row in result["requirements"])
    official = Counter({name: 0 for name in OFFICIAL_FLAGS})
    diagnostic: Counter[str] = Counter()
    for result in results:
        for row in result["requirements"]:
            for flag in row["flags"]:
                if flag == "citation_hallucination":
                    continue
                if flag in official:
                    official[flag] += 1
                else:
                    diagnostic[flag] += 1
        for flag in result["query_flags"]:
            if flag in {"evidence_coverage_failure", "citation_hallucination"}:
                official[flag] += 1
            elif flag not in official:
                diagnostic[flag] += 1
    retrieval_queries = [
        result
        for result in results
        if any(row["primary"] == "retrieval_limited" for row in result["requirements"])
    ]
    generation_queries = [
        result
        for result in results
        if any(row["primary"] == "generation_limited" for row in result["requirements"])
    ]
    parse_failures = [result for result in results if result["parse_status"] != "ok"]
    both = {result["query_id"] for result in retrieval_queries} & {
        result["query_id"] for result in generation_queries
    }
    generation_failure_ids = {result["query_id"] for result in generation_queries} | {
        result["query_id"] for result in parse_failures
    }
    scope = Counter()
    relevance = Counter()
    for result in results:
        label = result.get("answer_relevance") or "unjudged"
        if label not in {"direct", "partial", "off_topic"}:
            label = "unjudged"
        relevance[label] += 1
        for item in result["faithfulness"]["items"]:
            if item["verdict"] in FACTUAL_VERDICTS:
                scope[str(item.get("scope_issue") or "none")] += 1
    latencies = [
        float(result["generation"]["latency_ms"])
        for result in results
        if result["generation"].get("latency_ms") is not None and result["parse_status"] == "ok"
    ]
    judge_latencies = []
    tokens = Counter()
    tokens_by_stage: dict[str, Counter[str]] = defaultdict(Counter)
    cost = 0.0
    cost_known = 0
    cost_missing = 0
    for result in results:
        for stage in ("generation", "judge_faithfulness", "judge_requirements", "judge_citations"):
            facts = result[stage]
            if stage != "generation" and facts.get("latency_ms") is not None:
                judge_latencies.append(float(facts["latency_ms"]))
            usage = facts.get("usage") or {}
            if "input_tokens" in usage:
                tokens["input"] += int(usage["input_tokens"])
                tokens_by_stage[stage]["input"] += int(usage["input_tokens"])
            if "output_tokens" in usage:
                tokens["output"] += int(usage["output_tokens"])
                tokens_by_stage[stage]["output"] += int(usage["output_tokens"])
            if facts.get("estimated_cost_usd") is None:
                if facts.get("parse_status") not in {"skipped"}:
                    cost_missing += 1
            else:
                cost += float(facts["estimated_cost_usd"])
                cost_known += 1
    hypothesis = _hypothesis(official)
    return {
        "baseline": BASELINE_ID,
        "queries": len(results),
        "generation_parsed": sum(1 for result in results if result["parse_status"] == "ok"),
        "generation_parse_failures": len(parse_failures),
        "requirements_total": sum(len(result["requirements"]) for result in results),
        "faithfulness": faith,
        "requirement_coverage": coverage,
        "citation_correctness": citation,
        "citation_completeness": completeness,
        "citation_integrity_error_count": sum(
            int(result["integrity"]["integrity_error_count"]) for result in results
        ),
        "queries_with_integrity_errors": sum(
            1 for result in results if int(result["integrity"]["integrity_error_count"]) > 0
        ),
        "success_requirements": int(primaries["success"]),
        "retrieval_limited_requirements": int(primaries["retrieval_limited"]),
        "generation_limited_requirements": int(primaries["generation_limited"]),
        "unjudged_requirements": int(primaries["unjudged"]),
        "retrieval_limited_queries": len(retrieval_queries),
        "generation_limited_queries": len(generation_queries),
        "generation_failure_queries": len(generation_failure_ids),
        "retrieval_limited_only_queries": len(retrieval_queries) - len(both),
        "generation_limited_only_queries": len(generation_queries) - len(both),
        "both_limited_queries": len(both),
        "taxonomy": dict(official),
        "diagnostic_flags": dict(diagnostic),
        "scope_issues_on_factual_claims": dict(scope),
        "answer_relevance": dict(relevance),
        "generation_latency_ms": _latency(latencies),
        "judge_latency_ms": _latency(judge_latencies),
        "tokens": {
            "input": int(tokens["input"]),
            "output": int(tokens["output"]),
            "by_stage": {stage: dict(counts) for stage, counts in tokens_by_stage.items()},
        },
        "estimated_cost_usd": None if cost_known == 0 else round(cost, 4),
        "cost_status": _cost_status(cost_known, cost_missing),
        "cost_known_calls": cost_known,
        "cost_missing_calls": cost_missing,
        "hypothesis": hypothesis,
        "frozen_retrieval": verification,
        "representative_cases": select_representative_cases(results),
    }


def select_representative_cases(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    roles = (
        ("clean_success", _is_clean_success),
        ("retrieval_limited", _is_retrieval_case),
        ("generation_omission", _has_requirement_flag("generation_omission")),
        ("unsupported_claim", _has_requirement_flag("unsupported_claim")),
        ("citation_mismatch", _has_requirement_flag("citation_mismatch")),
        ("multi_requirement_partial", _is_partial),
        ("evidence_misinterpretation", _has_requirement_flag("evidence_misinterpretation")),
        ("period_scope_confusion", _has_requirement_flag("period_scope_confusion")),
        ("citation_hallucination", _has_hallucination),
        ("over_abstention", _has_requirement_flag("over_abstention")),
    )
    used: set[str] = set()
    chosen: list[dict[str, Any]] = []
    ordered = sorted(results, key=lambda result: result["query_id"])
    for role, predicate in roles:
        match = next(
            (
                result
                for result in ordered
                if result["query_id"] not in used and predicate(result)
            ),
            None,
        )
        if match is None:
            continue
        used.add(match["query_id"])
        chosen.append(_public_case(match, role))
        if len(chosen) == 8:
            break
    if len(chosen) < 5:
        for result in ordered:
            if result["query_id"] in used:
                continue
            clean = all(
                row["primary"] == "success" and not row["flags"] for row in result["requirements"]
            )
            if clean:
                continue
            used.add(result["query_id"])
            chosen.append(_public_case(result, "additional_failure"))
            if len(chosen) == 5:
                break
    return chosen


def failure_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        for requirement in result["requirements"]:
            if requirement["primary"] == "success" and not requirement["flags"]:
                continue
            rows.append(_failure_row(result["query_id"], requirement))
        if result["query_flags"]:
            rows.append(
                {
                    "query_id": result["query_id"],
                    "requirement_id": "",
                    "primary": "query",
                    "flags": "|".join(result["query_flags"]),
                    "coverage": "",
                    "evidence_in_top10": "",
                    "support": "",
                    "scope_issue": "",
                    "citation_missing": "citation_missing" in result["query_flags"],
                    "citation_mismatch": False,
                    "citation_hallucination": "citation_hallucination" in result["query_flags"],
                }
            )
    return rows


def render_report(summary: dict[str, Any]) -> str:
    frozen = summary["frozen_retrieval"]
    faith = summary["faithfulness"]
    coverage = summary["requirement_coverage"]
    correctness = summary["citation_correctness"]
    completeness = summary["citation_completeness"]
    lines = [
        "# G0 minimal grounded generation",
        "",
        "## 범위",
        "",
        "Baseline은 G0다. 입력은 query 하나와 frozen H10 Top-10이다. "
        "query당 generation 호출은 성공 경로에서 1회다. "
        "Agent loop, 재검색, reranker, prompt 수정은 하지 않았다.",
        f"Dataset `{frozen['dataset_id']}` sha256 `{frozen['dataset_sha256']}`. "
        f"모집단은 DEV fully_answerable {summary['queries']}문항, "
        f"requirement {summary['requirements_total']}개다. TEST 생성은 0이다.",
        "",
        "## Frozen retrieval",
        "",
        "Retrieval은 다시 실행하지 않았다. "
        f"저장 run `{frozen['h10_run']}`, policy `{frozen['retrieval_policy']}`. "
        "Top-10 id는 `h10_dev.json`과 `per_query_transition.json`이 같고, "
        "본문은 저장된 retrieval unit id로 색인 텍스트를 붙인 것이다.",
        (
            "저장 점수 재계산: "
            f"Hit@10 {frozen['hit10']['numerator']}/{frozen['hit10']['denominator']}, "
            "requirement recall "
            f"{frozen['requirement_recall']['numerator']}/"
            f"{frozen['requirement_recall']['denominator']}, "
            "all requirements "
            f"{frozen['all_requirements']['numerator']}/"
            f"{frozen['all_requirements']['denominator']}."
        ),
        "Rank 1..9는 BM25 순서를 유지한다. Dense rank 2..20은 이 입력에 없다.",
        "",
        "## 계약",
        "",
        f"Generator prompt `{GENERATOR_PROMPT_VERSION}`, "
        f"faithfulness `{FAITHFULNESS_PROMPT_VERSION}`, "
        f"requirements `{REQUIREMENT_PROMPT_VERSION}`, "
        f"citations `{CITATION_PROMPT_VERSION}`.",
        "모델이 고르는 인용은 C1 같은 citation id뿐이다. "
        "시스템은 citation id를 retrieval unit과 source element로 되돌린다.",
        "Transport 재시도는 같은 prompt를 다시 보낸다. 답변 품질을 보고 고치는 재시도는 없다.",
        "",
        "## 지표",
        "",
        f"Generation parse {summary['generation_parsed']}/{summary['queries']}. "
        f"Parse 실패 {summary['generation_parse_failures']}.",
        _ratio_line("Faithfulness", faith),
        _ratio_line("Requirement coverage", coverage),
        _ratio_line("Citation correctness", correctness),
        _ratio_line("Citation completeness", completeness),
        (
            f"Citation integrity errors {summary['citation_integrity_error_count']} "
            f"across {summary['queries_with_integrity_errors']} queries. "
            "Malformed와 unknown citation은 correctness의 오답으로 넣지 않았다."
        ),
        "",
        "Faithfulness는 supported factual claims / factual claims 다. "
        "not_factual과 unjudged는 분모에 넣지 않았다.",
        "Requirement coverage는 covered / judged requirements 다. "
        "unjudged는 미충족으로 세지 않았다. "
        "이 지표는 retrieval requirement recall이 아니다.",
        "Citation correctness는 supports / judged valid pairs 다. "
        "Invalid citation은 분모 밖에 있다.",
        "",
        "## Retrieval-limited 와 generation-limited",
        "",
        (
            f"Requirement primary: success {summary['success_requirements']}, "
            f"retrieval-limited {summary['retrieval_limited_requirements']}, "
            f"generation-limited {summary['generation_limited_requirements']}, "
            f"unjudged {summary['unjudged_requirements']}."
        ),
        (
            f"Query: retrieval-limited {summary['retrieval_limited_queries']} "
            f"(그중 generation-limited가 아닌 것 {summary['retrieval_limited_only_queries']}), "
            f"generation-limited {summary['generation_limited_queries']} "
            f"(그중 retrieval-limited가 아닌 것 {summary['generation_limited_only_queries']}), "
            f"둘 다 {summary['both_limited_queries']}, "
            f"generation failure {summary['generation_failure_queries']}."
        ),
        "Primary는 requirement마다 하나다. "
        "관련 qrel unit이 Top-10에 없으면 retrieval-limited에서 멈춘다. "
        "있으면 답변이 requirement를 충족하는지, 그 claim이 evidence에 지지되는지를 본다. "
        "Flag는 겹칠 수 있다.",
        "",
        "## Failure taxonomy",
        "",
        "evidence_coverage_failure와 citation_hallucination은 query 수다. "
        "나머지 flag는 requirement 수다.",
        "",
        _count_table(summary["taxonomy"]),
        "",
        "scope_issue는 factual claim 기준이다.",
        _count_table(summary["scope_issues_on_factual_claims"]),
        "",
        "Answer relevance는 보조 지표다.",
        _count_table(summary["answer_relevance"]),
        "",
        "## 대표 사례",
        "",
        "사례는 역할이 보이게 골랐다. 사람 정답은 만들지 않았다. "
        "본문 일부만 적는다. 전체 context는 generations.jsonl에 있다.",
        "",
    ]
    if not summary["representative_cases"]:
        lines.append("해당하는 사례가 없다.")
    for case in summary["representative_cases"]:
        lines.extend(_render_case(case))
    latency = summary["generation_latency_ms"]
    lines.extend(
        [
            "## Latency / token / cost",
            "",
            (
                f"Generation latency mean {latency['mean']} ms, p50 {latency['p50']} ms, "
                f"n {latency['count']}."
            ),
            (
                f"Judge latency mean {summary['judge_latency_ms']['mean']} ms, "
                f"p50 {summary['judge_latency_ms']['p50']} ms."
            ),
            (
                f"Tokens input {summary['tokens']['input']}, "
                f"output {summary['tokens']['output']}."
            ),
            (
                f"Estimated cost USD {summary['estimated_cost_usd']} "
                f"({summary['cost_status']}). "
                "가격표에 없는 호출은 0으로 채우지 않았다."
            ),
            "",
            "## 한계",
            "",
            "Faithfulness는 generator가 나눈 claim만 본다. "
            "answer에만 있고 claim에 없는 사실은 분모에 없다.",
            "Requirement coverage judge는 evidence를 보지 않는다. "
            "충족은 답변이 그 질문을 말했는지고, 근거의 참은 faithfulness가 본다.",
            (
                "Requirement 문장 중 "
                f"{frozen.get('requirement_texts_containing_digits', 0)}개에 숫자가 들어 있다. "
                "그 문장을 generator와 coverage judge에 그대로 넘겼다. Qrel 등급은 넘기지 않았다."
            ),
            "숫자, 기간, 단위를 결정론적으로 다시 계산하는 검증기는 넣지 않았다. "
            "scope_issue 집계만 남긴다.",
            "Judge 출력은 사람 정답이 아니다.",
            "DEV는 fully_answerable이라 기권 유형은 적을 수 있다.",
            "",
            "## 다음 가설",
            "",
            summary["hypothesis"]["text"],
            "이 가설은 구현하지 않았다.",
            "",
        ]
    )
    return "\n".join(lines)


def generation_jsonl_row(result: dict[str, Any], verification: dict[str, Any]) -> dict[str, Any]:
    facts = result["generation"]
    return {
        "dataset_id": verification["dataset_id"],
        "dataset_sha256": verification["dataset_sha256"],
        "query_id": result["query_id"],
        "requirements": [
            {"requirement_id": row["requirement_id"], "text": row["text"]}
            for row in result["requirements"]
        ],
        "retrieval_run_id": verification["h10_run"],
        "retrieval_policy": verification["retrieval_policy"],
        "retrieval_unit_ids": result["retrieval_unit_ids"],
        "context": result["evidence"],
        "resolution": result["resolution"],
        "prompt_version": facts.get("prompt_version"),
        "prompt_sha256": facts.get("prompt_sha256"),
        "provider": facts.get("provider"),
        "model": facts.get("model"),
        "decoding": facts.get("decoding"),
        "answer": result["answer"],
        "claims": result["claims"],
        "limitations": result["limitations"],
        "parse_status": result["parse_status"],
        "latency_ms": facts.get("latency_ms"),
        "usage": facts.get("usage"),
        "estimated_cost_usd": facts.get("estimated_cost_usd"),
        "error": facts.get("error"),
        "input_sha256": facts.get("input_sha256"),
        "retry_count": facts.get("retry_count"),
    }


def judge_jsonl_row(result: dict[str, Any], stage: str) -> dict[str, Any]:
    facts = result[stage]
    if stage == "judge_faithfulness":
        scored = result["faithfulness"]
    elif stage == "judge_requirements":
        scored = result["requirement_coverage"]
    else:
        scored = result["citation"]
    return {
        "query_id": result["query_id"],
        "evaluator_version": facts.get("evaluator_version"),
        "prompt_version": facts.get("prompt_version"),
        "prompt_sha256": facts.get("prompt_sha256"),
        "provider": facts.get("provider"),
        "model": facts.get("model"),
        "parse_status": facts.get("parse_status"),
        "score": scored.get("score"),
        "items": scored.get("items"),
        "reason": scored.get("reason"),
        "raw": facts.get("raw"),
        "latency_ms": facts.get("latency_ms"),
        "usage": facts.get("usage"),
        "estimated_cost_usd": facts.get("estimated_cost_usd"),
        "error": facts.get("error"),
        "input_sha256": facts.get("input_sha256"),
        "unjudged": scored.get("unjudged"),
    }


def result_jsonl_row(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "query_id": result["query_id"],
        "parse_status": result["parse_status"],
        "answer": result["answer"],
        "limitations": result["limitations"],
        "retrieval_unit_ids": result["retrieval_unit_ids"],
        "requirements": result["requirements"],
        "faithfulness": {
            "score": result["faithfulness"]["score"],
            "supported": result["faithfulness"]["supported"],
            "factual": result["faithfulness"]["factual"],
            "unjudged": result["faithfulness"]["unjudged"],
            "not_factual": result["faithfulness"]["not_factual"],
        },
        "requirement_coverage": {
            "score": result["requirement_coverage"]["score"],
            "covered": result["requirement_coverage"]["covered"],
            "omitted": result["requirement_coverage"]["omitted"],
            "abstained": result["requirement_coverage"]["abstained"],
            "unjudged": result["requirement_coverage"]["unjudged"],
            "total": result["requirement_coverage"]["total"],
        },
        "citation_correctness": {
            "score": result["citation"]["score"],
            "supports": result["citation"]["supports"],
            "does_not_support": result["citation"]["does_not_support"],
            "unjudged": result["citation"]["unjudged"],
        },
        "citation_completeness": {
            "score": result["citation"]["completeness"],
            "complete": result["citation"]["complete"],
            "factual": result["citation"]["factual_claims"],
        },
        "integrity": result["integrity"],
        "query_flags": result["query_flags"],
        "answer_relevance": result["answer_relevance"],
        "generation_latency_ms": result["generation"].get("latency_ms"),
        "usage": {
            "generation": result["generation"].get("usage"),
            "faithfulness": result["judge_faithfulness"].get("usage"),
            "requirements": result["judge_requirements"].get("usage"),
            "citations": result["judge_citations"].get("usage"),
        },
    }


def faithfulness_payload(packet: dict[str, Any], generation: dict[str, Any]) -> dict[str, Any]:
    output = generation["output"]
    payload = {
        "answer": output["answer"],
        "claims": [
            {"claim_id": claim["claim_id"], "text": claim["text"]} for claim in output["claims"]
        ],
        "evidence": packet["model_payload"]["evidence"],
    }
    assert_no_gold(payload)
    return payload


def requirement_payload(packet: dict[str, Any], generation: dict[str, Any]) -> dict[str, Any]:
    output = generation["output"]
    payload = {
        "question": packet["question"],
        "requirements": packet["requirements"],
        "answer": output["answer"],
        "claims": [
            {"claim_id": claim["claim_id"], "text": claim["text"]} for claim in output["claims"]
        ],
    }
    assert_no_gold(payload)
    return payload


def citation_payload(packet: dict[str, Any], generation: dict[str, Any]) -> dict[str, Any] | None:
    output = generation["output"]
    by_id = {row["citation_id"]: row for row in packet["model_payload"]["evidence"]}
    pairs = []
    for claim in output["claims"]:
        seen: list[str] = []
        for token in _claim_tokens(claim):
            if token not in by_id or token in seen:
                continue
            seen.append(token)
            pairs.append(
                {
                    "claim_id": claim["claim_id"],
                    "claim_text": claim["text"],
                    "citation_id": token,
                    "evidence": by_id[token],
                }
            )
    if not pairs:
        return None
    payload = {"pairs": pairs}
    assert_no_gold(payload)
    return payload


def _dataset_dir(root: Path) -> Path:
    return root / "evaluation" / "datasets" / DATASET_ID


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _assert_h10_prefix(query_id: str, transition: dict[str, Any]) -> None:
    for hit in transition["top10"]:
        if int(hit["rank"]) <= PROTECTED_PREFIX and int(hit["bm25_rank"]) != int(hit["rank"]):
            raise UsageError(f"{query_id} moved BM25 rank {hit['rank']}")
    last = transition["top10"][-1]
    if transition["edit"] == "none":
        if int(last["bm25_rank"]) != TOP_K:
            raise UsageError(f"{query_id} changed BM25 rank 10 without a dense-head edit")
        return
    if int(last["rank"]) != TOP_K or not last["is_dense_head"]:
        raise UsageError(f"{query_id} did not keep the dense head at rank 10")


def _assert_lock(name: str, recomputed: int, stored: int, locked: tuple[int, int]) -> None:
    if recomputed != stored or stored != locked[0]:
        raise UsageError(
            f"frozen {name} is recomputed {recomputed}, stored {stored}, locked {locked[0]}"
        )


def _sum_hits(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if row["at_k"]["10"]["hit"])


def _scan_units(path: Path, wanted: set[str]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    remaining = set(wanted)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not remaining or '"retrieval_unit_id"' not in line:
                continue
            row = json.loads(line)
            unit_id = str(row.get("retrieval_unit_id") or "")
            if unit_id not in remaining:
                continue
            found[unit_id] = row
            remaining.remove(unit_id)
            if not remaining:
                break
    return found


def _model_evidence(
    hit: dict[str, Any], unit: dict[str, Any], filing_company: str
) -> dict[str, Any]:
    section = unit.get("section_path") or []
    return {
        "citation_id": citation_id_for_rank(int(hit["rank"])),
        "retrieval_unit_id": str(hit["retrieval_unit_id"]),
        "filing_company": filing_company,
        "document_name": _as_text(unit.get("document_name")),
        "report_name": _as_text(unit.get("report_name")),
        "report_type": _as_text(unit.get("report_type")),
        "filing_period": _as_text(unit.get("filing_period")),
        "period_end_year": _as_text(unit.get("period_end_year")),
        "period_end_month": _as_text(unit.get("period_end_month")),
        "receipt_date": _as_text(unit.get("receipt_date")),
        "receipt_no": _as_text(unit.get("receipt_no")),
        "section_path": [str(item) for item in section] if isinstance(section, list) else [],
        "text": str(unit.get("text") or ""),
    }


def _resolution(
    hit: dict[str, Any], unit: dict[str, Any], transition_hit: dict[str, Any]
) -> dict[str, Any]:
    return {
        "citation_id": citation_id_for_rank(int(hit["rank"])),
        "rank": int(hit["rank"]),
        "retrieval_unit_id": str(hit["retrieval_unit_id"]),
        "bm25_rank": transition_hit.get("bm25_rank"),
        "is_dense_head": bool(transition_hit.get("is_dense_head")),
        "source_element_ids": _strings(unit.get("source_element_ids")),
        "source_refs": unit.get("source_refs") or [],
        "token_count": unit.get("token_count"),
        "representation_type": unit.get("representation_type"),
    }


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _relevant_units(qrels: list[dict[str, Any]]) -> dict[str, set[str]]:
    relevant: dict[str, set[str]] = defaultdict(set)
    for row in qrels:
        if row.get("judgment_status") not in (None, "judged"):
            continue
        if row.get("relevance") is None or int(row["relevance"]) < 1:
            continue
        relevant[str(row["requirement_id"])].add(str(row["retrieval_unit_id"]))
    return relevant


def _emitted_citations(answer: str, claims: list[dict[str, Any]]) -> list[tuple[str, str]]:
    emitted = [("answer", token) for token in MARKER_RE.findall(answer or "")]
    for claim in claims:
        claim_id = str(claim.get("claim_id") or "")
        emitted.extend((claim_id, str(token)) for token in claim.get("citation_ids") or [])
        emitted.extend(
            (claim_id, token) for token in MARKER_RE.findall(str(claim.get("text") or ""))
        )
    return emitted


def _claim_tokens(claim: dict[str, Any]) -> list[str]:
    tokens = [str(token) for token in claim.get("citation_ids") or []]
    tokens.extend(MARKER_RE.findall(str(claim.get("text") or "")))
    return tokens


def _valid_tokens(claim: dict[str, Any], valid_ids: set[str]) -> list[str]:
    seen: list[str] = []
    for token in _claim_tokens(claim):
        if token in valid_ids and token not in seen:
            seen.append(token)
    return seen


def _score_faithfulness(
    claims: list[dict[str, Any]], judgment: dict[str, Any], valid_ids: set[str]
) -> dict[str, Any]:
    raw_items = _items(judgment)
    resolved, _conflicts = _index_items(raw_items, ("claim_id",))
    supported = 0
    factual = 0
    not_factual = 0
    unjudged = 0
    items = []
    for claim in claims:
        claim_id = str(claim.get("claim_id") or "")
        cited = _valid_tokens(claim, valid_ids)
        unknown = [
            token
            for token in _claim_tokens(claim)
            if CITATION_ID_RE.fullmatch(token) and token not in valid_ids
        ]
        base = {
            "claim_id": claim_id,
            "text": str(claim.get("text") or ""),
            "valid_citation_ids": cited,
            "citation_hallucination": bool(unknown),
        }
        if claim_id not in resolved:
            unjudged += 1
            items.append({**base, "verdict": "unjudged", "scope_issue": "none", "reason": ""})
            continue
        item = resolved[claim_id]
        verdict = str(item.get("verdict") or "")
        if verdict not in FACTUAL_VERDICTS | {"not_factual"}:
            unjudged += 1
            verdict = "unjudged"
            scope_issue = "none"
            reason = ""
        else:
            scope_issue = str(item.get("scope_issue") or "none")
            reason = str(item.get("reason") or "")
            if verdict == "not_factual":
                not_factual += 1
            else:
                factual += 1
                if verdict == "supported":
                    supported += 1
        items.append(
            {**base, "verdict": verdict, "scope_issue": scope_issue, "reason": reason}
        )
    return {
        "score": None if factual == 0 else supported / factual,
        "supported": supported,
        "factual": factual,
        "not_factual": not_factual,
        "unjudged": unjudged,
        "conflicts": [key[0] for key in _conflicts],
        "items": items,
        "reason": str(judgment.get("reason") or ""),
    }


def _score_requirements(
    requirements: list[dict[str, Any]], judgment: dict[str, Any]
) -> dict[str, Any]:
    resolved, conflicts = _index_items(_items(judgment), ("requirement_id",))
    counts: Counter[str] = Counter()
    items = []
    allowed = {"covered", "omitted", "abstained"}
    for requirement in requirements:
        requirement_id = requirement["requirement_id"]
        if requirement_id in conflicts or requirement_id not in resolved:
            counts["unjudged"] += 1
            items.append(
                {
                    "requirement_id": requirement_id,
                    "verdict": "unjudged",
                    "claim_ids": [],
                    "reason": "",
                }
            )
            continue
        item = resolved[requirement_id]
        verdict = str(item.get("verdict") or "")
        if verdict not in allowed:
            counts["unjudged"] += 1
            verdict = "unjudged"
            claim_ids: list[str] = []
        else:
            counts[verdict] += 1
            claim_ids = [str(claim_id) for claim_id in item.get("claim_ids") or []]
        items.append(
            {
                "requirement_id": requirement_id,
                "verdict": verdict,
                "claim_ids": claim_ids,
                "reason": str(item.get("reason") or ""),
            }
        )
    judged = counts["covered"] + counts["omitted"] + counts["abstained"]
    relevance = str(judgment.get("answer_relevance") or "")
    if judgment.get("parse_status") != "ok" or relevance not in {"direct", "partial", "off_topic"}:
        relevance = "unjudged"
    return {
        "score": None if judged == 0 else counts["covered"] / judged,
        "covered": int(counts["covered"]),
        "omitted": int(counts["omitted"]),
        "abstained": int(counts["abstained"]),
        "unjudged": int(counts["unjudged"]),
        "judged": judged,
        "total": len(requirements),
        "items": items,
        "reason": str(judgment.get("reason") or ""),
        "answer_relevance": relevance,
    }


def _score_citations(
    faith_items: list[dict[str, Any]],
    judgment: dict[str, Any],
) -> dict[str, Any]:
    resolved, conflicts = _index_items(_items(judgment), ("claim_id", "citation_id"))
    supports = 0
    misses = 0
    unjudged = 0
    complete = 0
    factual_claims = 0
    items = []
    for faith in faith_items:
        tokens = faith.get("valid_citation_ids") or []
        if faith["verdict"] not in FACTUAL_VERDICTS:
            continue
        factual_claims += 1
        if tokens:
            complete += 1
        for token in tokens:
            key = (faith["claim_id"], token)
            if key in conflicts or key not in resolved:
                unjudged += 1
                verdict = "unjudged"
                reason = ""
            else:
                verdict = str(resolved[key].get("verdict") or "")
                reason = str(resolved[key].get("reason") or "")
                if verdict == "supports":
                    supports += 1
                elif verdict == "does_not_support":
                    misses += 1
                else:
                    unjudged += 1
                    verdict = "unjudged"
            items.append(
                {
                    "claim_id": faith["claim_id"],
                    "citation_id": token,
                    "verdict": verdict,
                    "reason": reason,
                }
            )
    judged = supports + misses
    return {
        "score": None if judged == 0 else supports / judged,
        "supports": supports,
        "does_not_support": misses,
        "unjudged": unjudged,
        "items": items,
        "reason": str(judgment.get("reason") or ""),
        "completeness": None if factual_claims == 0 else complete / factual_claims,
        "complete": complete,
        "factual_claims": factual_claims,
    }


def _items(judgment: dict[str, Any]) -> list[dict[str, Any]]:
    if judgment.get("parse_status") != "ok":
        return []
    raw = judgment.get("raw")
    if isinstance(raw, dict) and isinstance(raw.get("items"), list):
        return [item for item in raw["items"] if isinstance(item, dict)]
    output = judgment.get("output")
    if isinstance(output, dict) and isinstance(output.get("items"), list):
        return [item for item in output["items"] if isinstance(item, dict)]
    return []


def _index_items(
    items: list[dict[str, Any]], fields: tuple[str, ...]
) -> tuple[dict[Any, dict[str, Any]], list[tuple[str, ...]]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[tuple(str(item.get(field) or "") for field in fields)].append(item)
    resolved: dict[Any, dict[str, Any]] = {}
    conflicts: list[tuple[str, ...]] = []
    for key, group in grouped.items():
        verdicts = {str(item.get("verdict") or "") for item in group}
        stored: Any = key[0] if len(key) == 1 else key
        if len(verdicts) == 1:
            resolved[stored] = group[0]
        else:
            conflicts.append(key)
    return resolved, conflicts


def _coverage_item(coverage: dict[str, Any], requirement_id: str) -> dict[str, Any]:
    for item in coverage["items"]:
        if item["requirement_id"] == requirement_id:
            return item
    return {"requirement_id": requirement_id, "verdict": "unjudged", "claim_ids": [], "reason": ""}


def _support_of(linked: list[dict[str, Any]]) -> tuple[str, str]:
    if not linked:
        return "no_claim", "none"
    verdicts = [item["verdict"] for item in linked]
    scopes = {
        str(item.get("scope_issue") or "none")
        for item in linked
        if str(item.get("scope_issue") or "none") != "none"
    }
    if len(scopes) == 1:
        scope = next(iter(scopes))
    elif scopes:
        scope = "mixed"
    else:
        scope = "none"
    if any(verdict == "unjudged" for verdict in verdicts):
        return "unjudged", scope
    factual = [verdict for verdict in verdicts if verdict in FACTUAL_VERDICTS]
    if not factual:
        return "not_factual", scope
    if "misinterpretation" in factual:
        return "misinterpretation", scope
    if "unsupported" in factual:
        return "unsupported", scope
    if all(verdict == "supported" for verdict in factual):
        return "supported", scope
    return "unjudged", scope


def _call_facts(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "parse_status": record.get("parse_status"),
        "provider": record.get("provider"),
        "model": record.get("model"),
        "prompt_version": record.get("prompt_version"),
        "prompt_sha256": record.get("prompt_sha256"),
        "evaluator_version": record.get("evaluator_version"),
        "decoding": record.get("decoding"),
        "latency_ms": record.get("latency_ms"),
        "usage": record.get("usage"),
        "estimated_cost_usd": record.get("estimated_cost_usd"),
        "error": record.get("error"),
        "input_sha256": record.get("input_sha256"),
        "retry_count": record.get("retry_count"),
        "raw": record.get("raw"),
    }


def _sum_ratio(
    results: list[dict[str, Any]], key: str, numerator: str, denominator: str, unjudged: str
) -> dict[str, Any]:
    num = sum(int(result[key][numerator]) for result in results)
    den = sum(int(result[key][denominator]) for result in results)
    missing = sum(int(result[key][unjudged]) for result in results)
    return {
        "score": None if den == 0 else num / den,
        "numerator": num,
        "denominator": den,
        "unjudged": missing,
        "denominator_definition": "judged_only",
    }


def _sum_coverage(results: list[dict[str, Any]]) -> dict[str, Any]:
    covered = sum(int(result["requirement_coverage"]["covered"]) for result in results)
    omitted = sum(int(result["requirement_coverage"]["omitted"]) for result in results)
    abstained = sum(int(result["requirement_coverage"]["abstained"]) for result in results)
    unjudged = sum(int(result["requirement_coverage"]["unjudged"]) for result in results)
    judged = covered + omitted + abstained
    total = sum(int(result["requirement_coverage"]["total"]) for result in results)
    return {
        "score": None if judged == 0 else covered / judged,
        "numerator": covered,
        "denominator": judged,
        "covered": covered,
        "omitted": omitted,
        "abstained": abstained,
        "unjudged": unjudged,
        "total": total,
        "denominator_definition": "judged_requirements",
    }


def _sum_citation(results: list[dict[str, Any]]) -> dict[str, Any]:
    supports = sum(int(result["citation"]["supports"]) for result in results)
    misses = sum(int(result["citation"]["does_not_support"]) for result in results)
    unjudged = sum(int(result["citation"]["unjudged"]) for result in results)
    judged = supports + misses
    return {
        "score": None if judged == 0 else supports / judged,
        "numerator": supports,
        "denominator": judged,
        "supports": supports,
        "does_not_support": misses,
        "unjudged": unjudged,
        "denominator_definition": "judged_valid_citation_pairs",
    }


def _sum_completeness(results: list[dict[str, Any]]) -> dict[str, Any]:
    complete = sum(int(result["citation"]["complete"]) for result in results)
    factual = sum(int(result["citation"]["factual_claims"]) for result in results)
    return {
        "score": None if factual == 0 else complete / factual,
        "numerator": complete,
        "denominator": factual,
        "complete": complete,
        "factual": factual,
        "denominator_definition": "factual_claims",
    }


def _latency(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean": None, "p50": None, "count": 0}
    return {
        "mean": round(statistics.mean(values), 3),
        "p50": round(statistics.median(values), 3),
        "count": len(values),
    }


def _cost_status(known: int, missing: int) -> str:
    if known == 0:
        return "unavailable"
    if missing:
        return "partial"
    return "estimated"


def _hypothesis(official: Counter[str]) -> dict[str, Any]:
    ranked = sorted(
        OFFICIAL_FLAGS,
        key=lambda name: (-int(official[name]), OFFICIAL_FLAGS.index(name)),
    )
    winner = ranked[0]
    count = int(official[winner])
    generation_ranked = sorted(
        GENERATION_FLAGS,
        key=lambda name: (-int(official[name]), OFFICIAL_FLAGS.index(name)),
    )
    runner = generation_ranked[0]
    text = HYPOTHESES[winner]
    if winner == "retrieval_failure" and int(official[runner]) > 0:
        text += f" 생성 쪽에서 가장 많은 flag는 {runner} {int(official[runner])}건이다."
    if count == 0:
        text = (
            "공식 failure flag가 0이다. "
            "이 결과만으로 prompt를 고치지 않는다. "
            "대표 사례의 judge 문장을 본 뒤에 다음 가설을 고른다."
        )
    return {"category": winner, "count": count, "text": text}


def _is_clean_success(result: dict[str, Any]) -> bool:
    if result["parse_status"] != "ok" or result["query_flags"]:
        return False
    if int(result["integrity"]["integrity_error_count"]) > 0:
        return False
    if not result["requirements"]:
        return False
    return all(row["primary"] == "success" and not row["flags"] for row in result["requirements"])


def _is_retrieval_case(result: dict[str, Any]) -> bool:
    primaries = {row["primary"] for row in result["requirements"]}
    return "retrieval_limited" in primaries and "generation_limited" not in primaries


def _is_partial(result: dict[str, Any]) -> bool:
    primaries = {row["primary"] for row in result["requirements"]}
    return len(result["requirements"]) >= 2 and "success" in primaries and len(primaries) > 1


def _has_requirement_flag(flag: str):
    def predicate(result: dict[str, Any]) -> bool:
        return any(flag in row["flags"] for row in result["requirements"])

    return predicate


def _has_hallucination(result: dict[str, Any]) -> bool:
    return "citation_hallucination" in result["query_flags"] or any(
        "citation_hallucination" in row["flags"] for row in result["requirements"]
    )


def _public_case(result: dict[str, Any], role: str) -> dict[str, Any]:
    return {
        "role": role,
        "query_id": result["query_id"],
        "question": result["question"],
        "requirements": [
            {
                "requirement_id": row["requirement_id"],
                "text": row["text"],
                "primary": row["primary"],
                "flags": row["flags"],
                "coverage": row["coverage"],
                "support": row["support"],
                "evidence_in_top10": row["evidence_in_top10"],
            }
            for row in result["requirements"]
        ],
        "contexts": [
            {
                "citation_id": row["citation_id"],
                "retrieval_unit_id": row["retrieval_unit_id"],
                "document_name": row["document_name"],
                "report_type": row["report_type"],
                "filing_period": row["filing_period"],
                "section_path": row["section_path"],
                "text_excerpt": _excerpt(row["text"]),
            }
            for row in result["evidence"]
        ],
        "answer": result["answer"],
        "limitations": result["limitations"],
        "claims": result["claims"],
        "faithfulness": {
            "score": result["faithfulness"]["score"],
            "items": [
                {
                    "claim_id": item["claim_id"],
                    "verdict": item["verdict"],
                    "scope_issue": item["scope_issue"],
                    "reason": _excerpt(item["reason"]),
                }
                for item in result["faithfulness"]["items"]
            ],
        },
        "requirement_judgment": {
            "score": result["requirement_coverage"]["score"],
            "answer_relevance": result["answer_relevance"],
            "items": result["requirement_coverage"]["items"],
        },
        "citation_judgment": {
            "score": result["citation"]["score"],
            "items": result["citation"]["items"],
        },
        "integrity": {
            "integrity_error_count": result["integrity"]["integrity_error_count"],
            "malformed": result["integrity"]["malformed"],
            "unknown": result["integrity"]["unknown"],
        },
        "query_flags": result["query_flags"],
    }


def _excerpt(text: str) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= EXCERPT_CHARS:
        return compact
    return compact[:EXCERPT_CHARS] + "…"


def _failure_row(query_id: str, requirement: dict[str, Any]) -> dict[str, Any]:
    return {
        "query_id": query_id,
        "requirement_id": requirement["requirement_id"],
        "primary": requirement["primary"],
        "flags": "|".join(requirement["flags"]),
        "coverage": requirement["coverage"],
        "evidence_in_top10": requirement["evidence_in_top10"],
        "support": requirement["support"],
        "scope_issue": requirement["scope_issue"],
        "citation_missing": requirement["citation_missing"],
        "citation_mismatch": requirement["citation_mismatch"],
        "citation_hallucination": requirement["citation_hallucination"],
    }


def _ratio_line(label: str, block: dict[str, Any]) -> str:
    score = block.get("score")
    rendered = "없음" if score is None else f"{score:.3f}"
    return (
        f"{label} {block.get('numerator')}/{block.get('denominator')} ({rendered}). "
        f"unjudged {block.get('unjudged', 0)}."
    )


def _count_table(counts: dict[str, Any]) -> str:
    lines = ["| name | count |", "| --- | --- |"]
    for name, count in counts.items():
        lines.append(f"| {name} | {count} |")
    return "\n".join(lines)


def _render_case(case: dict[str, Any]) -> list[str]:
    lines = [
        f"### {case['role']} `{case['query_id']}`",
        "",
        case["question"],
        "",
    ]
    for requirement in case["requirements"]:
        lines.append(
            f"- {requirement['requirement_id']} {requirement['primary']} "
            f"coverage={requirement['coverage']} support={requirement['support']} "
            f"evidence={requirement['evidence_in_top10']} "
            f"flags={','.join(requirement['flags']) or '-'}: {requirement['text']}"
        )
    lines.extend(["", "Answer:", "", case["answer"] or "(empty)", ""])
    if case["limitations"]:
        lines.append("Limitations: " + " | ".join(case["limitations"]))
        lines.append("")
    lines.append(
        "Faithfulness "
        + _short_score(case["faithfulness"]["score"])
        + ". "
        + " ".join(
            f"{item['claim_id']}={item['verdict']}/{item['scope_issue']}"
            for item in case["faithfulness"]["items"]
        )
    )
    lines.append(
        "Requirement coverage "
        + _short_score(case["requirement_judgment"]["score"])
        + f", relevance {case['requirement_judgment']['answer_relevance']}."
    )
    lines.append("Citation correctness " + _short_score(case["citation_judgment"]["score"]) + ".")
    lines.append(
        f"Integrity errors {case['integrity']['integrity_error_count']}. "
        f"Query flags {','.join(case['query_flags']) or '-'}."
    )
    lines.append("")
    for context in case["contexts"]:
        lines.append(
            f"- [{context['citation_id']}] {context['document_name']} "
            f"{context['filing_period']} {context['report_type']} "
            f"`{context['retrieval_unit_id']}` {context['text_excerpt']}"
        )
    lines.append("")
    return lines


def _short_score(score: float | None) -> str:
    if score is None:
        return "없음"
    return f"{score:.3f}"


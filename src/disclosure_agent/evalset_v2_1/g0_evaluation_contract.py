"""Correct the frozen G0 evaluation contract without a new generation.

The raw G0 run stays where it is. This module rereads its generations and
judge files, classifies claims, and writes a separate corrected evaluation.
It does not search, does not call a model, and does not edit qrels.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from disclosure_agent.errors import UsageError
from disclosure_agent.evalset_v2_1.grounded_generation import (
    FACTUAL_VERDICTS,
    GENERATOR_INSTRUCTIONS,
    OFFICIAL_FLAGS,
    RequirementSignals,
    _hypothesis,
    classify_requirement,
    prompt_sha256,
)
from disclosure_agent.evaluation import git_revision

SOURCE_RUN_DIR = "multi_company_retrieval_v2_1_dev_g0_grounded_generation"
CORRECTED_RUN_DIR = "multi_company_retrieval_v2_1_dev_g0_evaluator_corrected"
CONTRACT_VERSION = "g0-evaluation-contract-v1"
RAW_EVALUATOR_VERSION = "g0-answer-evaluation-v1"
CLAIM_TYPES = ("factual_claim", "abstention", "limitation", "meta", "unjudged")

_EVIDENCE_SCOPE = re.compile(
    r"제공된\s*(?:공시\s*)?(?:분기보고서\s*)?근거|입력된\s*근거|입력\s*evidence"
)
_NONCONFIRM = re.compile(r"확인할\s*수\s*없|확인되지\s*않")
_SCOPE_LIMIT = re.compile(r"로\s*볼\s*수(?:는|도)?\s*없")
_DOCUMENT_ABSENCE = re.compile(
    r"(?:사업보고서|반기보고서|분기보고서|공시|문서)에는[^.]{0,80}?"
    r"(?:없|기재되어\s*있지|포함되어\s*있지|제시되어\s*있지|공시되어\s*있지)"
)
_META = re.compile(r"^(?:아래와 같이 정리|다음과 같이 정리|이 답변은 참고)")

_ALTEOGEN_QUERY = "V2-00989619-finance_table-02"
_SAMSUNG_CAPITAL_QUERY = "V2-00126380-finance_table-03"
_PREFERRED_QUARTERLY_UNIT = "56179b6584b73ea27d42851a174e6dc934fdb80767b575e0453486389fc03cde"
_ANNUAL_SERVICE_UNIT = "5777aa1c30dc503922bfccf33e575e2581573f63ea382211560b71d6c8cbe6de"
_SAMSUNG_CAPITAL_UNIT = "777475fc47a37a5862baa33be44b0f46cf17df7a721c06f211fb464c9aa87d79"


def classify_claim(text: str) -> dict[str, str]:
    """Classify one claim from its wording. The generator factual flag is not an input."""
    sentence = " ".join(text.split())
    evidence_scoped = _EVIDENCE_SCOPE.search(sentence) is not None
    nonconfirm = _NONCONFIRM.search(sentence) is not None
    document_absence = _DOCUMENT_ABSENCE.search(sentence) is not None
    scope_limit = _SCOPE_LIMIT.search(sentence) is not None
    if evidence_scoped and nonconfirm and (document_absence or scope_limit):
        return {
            "claim_type": "unjudged",
            "rule": "mixed_abstention_and_other_act",
        }
    if evidence_scoped and nonconfirm:
        return {
            "claim_type": "abstention",
            "rule": "evidence_scoped_nonconfirmation",
        }
    if scope_limit:
        return {"claim_type": "limitation", "rule": "scope_limitation"}
    if _META.search(sentence):
        return {"claim_type": "meta", "rule": "answer_guidance"}
    if document_absence:
        return {"claim_type": "factual_claim", "rule": "document_absence_is_factual"}
    return {"claim_type": "factual_claim", "rule": "verifiable_assertion"}


def citation_requirement(claim_type: str) -> dict[str, Any]:
    if claim_type == "factual_claim":
        return {
            "citation_required": True,
            "basis": "verifiable_filing_fact",
        }
    if claim_type == "unjudged":
        return {"citation_required": None, "basis": "unjudged_claim_type"}
    return {
        "citation_required": False,
        "basis": f"{claim_type}_does_not_require_citation",
    }


def abstention_appropriateness(claim_type: str, raw_verdict: str) -> str | None:
    if claim_type != "abstention":
        return None
    if raw_verdict == "supported":
        return "appropriate"
    return "unjudged"


def build_correction(root: Path) -> dict[str, Any]:
    source = _source_dir(root)
    generations = _read_jsonl(source / "generations.jsonl")
    faith_rows = {row["query_id"]: row for row in _read_jsonl(source / "judge_faithfulness.jsonl")}
    citation_rows = {row["query_id"]: row for row in _read_jsonl(source / "judge_citations.jsonl")}
    results = {row["query_id"]: row for row in _read_jsonl(source / "results.jsonl")}
    raw_summary = _read_json(source / "summary.json")
    raw_manifest = _read_json(source / "manifest.json")
    _assert_raw_contract(generations, results, raw_summary, raw_manifest)
    questions = _question_text(root)
    classifications: list[dict[str, Any]] = []
    corrected_results: list[dict[str, Any]] = []
    for generation in generations:
        query_id = generation["query_id"]
        classified = _classify_query(
            generation,
            faith_rows[query_id],
            citation_rows[query_id],
        )
        classifications.extend(classified)
        corrected_results.append(
            _correct_query(
                generation,
                results[query_id],
                classified,
                questions[query_id],
            )
        )
    summary = _summarize(corrected_results, classifications, raw_summary, raw_manifest, source)
    deltas = _deltas(summary, classifications, corrected_results)
    return {
        "manifest": _manifest(root, source, raw_manifest, summary),
        "contract": CONTRACT_MARKDOWN,
        "classifications": classifications,
        "results": corrected_results,
        "summary": summary,
        "deltas": deltas,
        "report": render_report(summary, deltas),
    }


def write_correction(root: Path, output_dir: Path | None = None) -> dict[str, Any]:
    payload = build_correction(root)
    source = _source_dir(root)
    destination = output_dir or (root / "evaluation" / "runs" / CORRECTED_RUN_DIR)
    if destination.resolve() == source.resolve():
        raise UsageError("corrected evaluation cannot overwrite the raw G0 run")
    destination.mkdir(parents=True, exist_ok=True)
    _dump(destination / "manifest.json", payload["manifest"])
    (destination / "evaluation_contract.md").write_text(payload["contract"], encoding="utf-8")
    _jsonl(destination / "claim_classifications.jsonl", payload["classifications"])
    _jsonl(destination / "results.jsonl", payload["results"])
    _dump(destination / "summary.json", payload["summary"])
    _dump(destination / "metric_deltas.json", payload["deltas"])
    (destination / "report.md").write_text(payload["report"], encoding="utf-8")
    return payload


def render_report(summary: dict[str, Any], deltas: dict[str, Any]) -> str:
    faith = summary["faithfulness"]
    coverage = summary["requirement_coverage"]
    correctness = summary["citation_correctness"]
    completeness = summary["citation_completeness"]
    lines = [
        "# G0 evaluator-corrected",
        "",
        "## 범위",
        "",
        "기존 G0 generation 45건과 기존 judge 출력을 재사용했다. "
        "Generator prompt, retrieval, qrel, TEST는 그대로 두었다. "
        "모델은 다시 호출하지 않았다.",
        f"원본 run `{SOURCE_RUN_DIR}`. 수정본 `{CORRECTED_RUN_DIR}`.",
        "",
        "## Raw 와 corrected",
        "",
        "| metric | raw | corrected |",
        "| --- | ---: | ---: |",
        _pair("Faithfulness", deltas["metrics"]["faithfulness"]),
        _pair("Requirement coverage", deltas["metrics"]["requirement_coverage"]),
        _pair("Citation correctness", deltas["metrics"]["citation_correctness"]),
        _pair("Citation completeness", deltas["metrics"]["citation_completeness"]),
        _pair("Requirement success", deltas["metrics"]["requirement_success"]),
        "",
        f"Corrected faithfulness {faith['numerator']}/{faith['denominator']}. "
        f"Coverage {coverage['numerator']}/{coverage['denominator']}. "
        f"Citation correctness {correctness['numerator']}/{correctness['denominator']}. "
        f"Citation completeness {completeness['numerator']}/{completeness['denominator']}.",
        f"Integrity errors {summary['citation_integrity_error_count']}.",
        "",
        "Faithfulness 분모는 검증 가능한 factual claim이다. "
        f"Abstention {faith['excluded_abstention']}건, "
        f"limitation {faith['excluded_limitation']}건, "
        f"meta {faith['excluded_meta']}건을 그 분모 밖에 두었다. "
        f"Claim type unjudged {faith['unjudged_claim_types']}건.",
        "Citation completeness 분모는 citation이 필요한 factual claim이다.",
        "Citation correctness의 기존 pair verdict는 유지했다. "
        "Factual claim이 아닌 pair만 분모에서 뺐다.",
        "",
        "## Retrieval-limited 와 generation-limited",
        "",
        "Requirement primary: "
        f"success {summary['success_requirements']}, "
        f"retrieval-limited {summary['retrieval_limited_requirements']}, "
        f"generation-limited {summary['generation_limited_requirements']}, "
        f"unjudged {summary['unjudged_requirements']}.",
        "관련 qrel unit이 Top-10에 없으면 retrieval-limited다. "
        "있으면 답변의 충족과 factual claim의 지지로 generation을 본다.",
        "",
        "## 판정이 바뀐 claim",
        "",
        _claim_lines(deltas["changed_claims"]),
        "",
        "## 판정이 바뀐 requirement",
        "",
        _requirement_lines(deltas["changed_requirements"]),
        "",
        "## Coverage 재검토",
        "",
    ]
    for review in summary["coverage_reviews"]:
        lines.extend(
            [
                f"### `{review['query_id']}` {review['requirement_id']}",
                "",
                review["reason"],
                "",
                (
                    f"Requirement {'충족' if review['requirement_satisfied'] else '미충족'}. "
                    f"{'Judge 오류' if review['judge_error'] else 'Judge 판정 유지'}. "
                    f"Coverage {review['raw_verdict']} -> {review['corrected_verdict']}."
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## 남아 있는 generation failure",
            "",
            _failure_lines(summary["remaining_generation_failures"]),
            "",
            "Citation mismatch로 남은 factual pair는 아래다. "
            "기존 primary 규칙은 이 pair만으로 success를 generation-limited로 바꾸지 않는다.",
            "",
            _mismatch_lines(summary["remaining_citation_mismatches"]),
            "",
            "## Abstention 적절성",
            "",
            _count_table(summary["abstention_appropriateness"]),
            "",
            summary["abstention_note"],
            "",
            "## 평가 한계",
            "",
            summary["limits"],
            "",
            "## Failure taxonomy",
            "",
            _count_table(summary["taxonomy"]),
            "",
        ]
    )
    return "\n".join(lines)


CONTRACT_MARKDOWN = """\
# G0 evaluation contract v1

이 계약은 저장된 G0 generation과 저장된 judge 출력을 다시 집계한다.
Generator prompt, generation model, retrieval, qrel, TEST는 이 계약의 입력이 아니다.

## 발견한 오류

Raw faithfulness와 citation completeness는 judge verdict가
`supported`, `unsupported`, `misinterpretation`이면 그 claim을 factual claim으로 세었다.
Generator가 `factual: false`로 둔 확인 불가 문장도 judge가 `supported`로 두면 분모에 들어갔다.
Citation이 없는 그 문장이 citation completeness의 미충족으로 집계됐다.

Requirement coverage는 요구한 수치가 답변에 있어도, 기준을 한정하는 문장이 함께 있으면
`abstained`가 될 수 있었다.

## Claim type

한 claim은 다음 중 하나다.

- `factual_claim`: evidence로 확인할 수 있는 사실 주장. 문서 전체에 대한 부재 단정도 여기 둔다.
- `abstention`: 제공된 evidence 범위에서 확인할 수 없다는 진술.
- `limitation`: 다른 사실의 해석 범위를 제한하는 문장.
- `meta`: 답변 절차를 안내하는 문장.
- `unjudged`: 위 유형이 한 문장에서 겹치거나 분류가 닫히지 않은 문장.

Generator의 `factual` 값은 기록만 하고 유형을 정하지 않는다.

분류 규칙:

1. `제공된 근거` 또는 `입력된 근거`가 확인 불가와 함께 있고,
   문서 전체 부재나 범위 제한이 없으면 `abstention`.
2. `A를 B로 볼 수 없다`가 주된 행위이면 `limitation`.
3. `분기보고서에는 해당 정보가 없다`처럼 문서 자체의 부재를 말하면 `factual_claim`.
4. 1과 2, 또는 1과 3이 한 문장에 있으면 `unjudged`.
5. 나머지 검증 가능한 주장은 `factual_claim`.

`unjudged`는 faithfulness와 citation completeness의 점수 분모에 넣지 않고, 건수를 별도로 남긴다.

## Faithfulness

분모는 `factual_claim`이면서 raw verdict가
`supported`, `unsupported`, `misinterpretation`인 claim이다.
Abstention, limitation, meta는 이 분모 밖에 둔다.

Abstention의 적절성은 별도다. Raw verdict가 `supported`이면 `appropriate`다.
그 외 verdict는 `unjudged`로 남긴다. 문서 부재로 읽힌 판정을 적절한 기권으로 바꾸지 않고,
faithfulness 오답으로 다시 넣지도 않는다.

## Citation completeness

분모는 citation이 필요한 `factual_claim`이다.
이 데이터에서 그 claim은 공시에서 확인하는 사실이라 citation이 필요하다.
Abstention, limitation, meta에는 citation을 요구하지 않는다.
`unjudged` claim의 citation 필요는 `unjudged`로 남긴다.

## Citation correctness

저장된 pair verdict는 유지한다.
`factual_claim`이 아닌 claim의 pair는 수정 분모에서 빼고, verdict와 제외 이유를 남긴다.

## Requirement coverage

Coverage는 답변이 요구한 내용을 말하는지를 본다.
수치의 참과 evidence 충실성은 faithfulness와 citation이 본다.
요구한 내용이 있고 한정이 함께 있다는 이유만으로 `abstained`로 두지 않는다.

적용한 재검토:

- `V2-00126380-finance_table-03` R1은 보통주 약 87.9%, 우선주 약 12.1%를 말했고 C5가 있다.
  Raw judge는 납입자본금이 종류별로 없다는 한정만 보고 `abstained`로 두었다.
  요구한 비중은 답변에 있다. Coverage를 `covered`로 고친다. 이 requirement는 충족되었다.
- `V2-00989619-finance_table-02` R1은 분기보고서 기간을 말하지 않았다.
  사업보고서 기간을 제시한 뒤 그 기간을 분기보고서 기간으로 보지 않았다.
  선호 분기보고서 unit은 Top-10에 없고, 관련 사업보고서 unit은 있다.
  `abstained`는 답변의 기권과 같다. Coverage는 유지한다. 이 requirement는 충족되지 않았다.

## 유지한 규칙

Retrieval-limited와 generation-limited의 분리는 유지한다.
관련 qrel unit이 Top-10에 없으면 retrieval-limited다.
Citation mismatch가 success requirement의 primary를 바꾸던 규칙은 유지한다.
Qrel, H10 policy, generator prompt는 유지한다.
"""


def _classify_query(
    generation: dict[str, Any],
    faith: dict[str, Any],
    citations: dict[str, Any],
) -> list[dict[str, Any]]:
    by_faith = {item["claim_id"]: item for item in faith["items"]}
    pairs: dict[str, list[dict[str, Any]]] = {}
    for item in citations.get("items") or []:
        pairs.setdefault(item["claim_id"], []).append(item)
    rows = []
    for claim in generation["claims"]:
        claim_id = claim["claim_id"]
        judged = by_faith[claim_id]
        kind = classify_claim(claim["text"])
        required = citation_requirement(kind["claim_type"])
        raw_verdict = str(judged["verdict"])
        factual = kind["claim_type"] == "factual_claim" and raw_verdict in FACTUAL_VERDICTS
        needs_citation = required["citation_required"] is True
        pair_rows = []
        for pair in pairs.get(claim_id, []):
            pair_rows.append(
                {
                    "citation_id": pair["citation_id"],
                    "verdict": pair["verdict"],
                    "reason": pair.get("reason") or "",
                    "in_corrected_denominator": factual,
                }
            )
        rows.append(
            {
                "query_id": generation["query_id"],
                "claim_id": claim_id,
                "text": claim["text"],
                "generator_factual": claim.get("factual"),
                "claim_type": kind["claim_type"],
                "classification_rule": kind["rule"],
                "raw_verdict": raw_verdict,
                "raw_scope_issue": judged.get("scope_issue") or "none",
                "raw_reason": judged.get("reason") or "",
                "valid_citation_ids": list(judged.get("valid_citation_ids") or []),
                "citation_hallucination": bool(judged.get("citation_hallucination")),
                "in_raw_faithfulness_denominator": raw_verdict in FACTUAL_VERDICTS,
                "in_faithfulness_denominator": factual,
                "faithfulness_supported": factual and raw_verdict == "supported",
                "citation_required": required["citation_required"],
                "citation_requirement_basis": required["basis"],
                "in_completeness_denominator": needs_citation and factual,
                "completeness_hit": needs_citation
                and factual
                and bool(judged.get("valid_citation_ids")),
                "abstention_appropriateness": abstention_appropriateness(
                    kind["claim_type"], raw_verdict
                ),
                "citation_pairs": pair_rows,
            }
        )
    return rows


def _correct_query(
    generation: dict[str, Any],
    raw: dict[str, Any],
    claims: list[dict[str, Any]],
    question: str,
) -> dict[str, Any]:
    by_claim = {row["claim_id"]: row for row in claims}
    _check_context(generation)
    reviews = {
        (review["query_id"], review["requirement_id"]): review for review in _coverage_reviews()
    }
    requirements = []
    relevance = raw["answer_relevance"]
    for requirement in raw["requirements"]:
        key = (generation["query_id"], requirement["requirement_id"])
        review = reviews.get(key)
        corrected = _correct_requirement(requirement, by_claim, review)
        requirements.append(corrected)
        if review and review.get("corrected_answer_relevance"):
            relevance = review["corrected_answer_relevance"]
    flags = _query_flags(requirements, raw["integrity"], claims)
    return {
        "query_id": generation["query_id"],
        "question": question,
        "parse_status": raw["parse_status"],
        "answer": raw["answer"],
        "limitations": raw["limitations"],
        "reused_generation": True,
        "requirements": requirements,
        "query_flags": flags,
        "raw_query_flags": raw["query_flags"],
        "answer_relevance": relevance,
        "raw_answer_relevance": raw["answer_relevance"],
        "integrity": raw["integrity"],
        "faithfulness": _faith_block(claims),
        "citation_correctness": _correctness_block(claims),
        "citation_completeness": _completeness_block(claims),
    }


def _correct_requirement(
    requirement: dict[str, Any],
    by_claim: dict[str, dict[str, Any]],
    review: dict[str, Any] | None,
) -> dict[str, Any]:
    coverage = requirement["coverage"]
    linked_ids = list(requirement["linked_claim_ids"])
    reason = requirement["reason"]
    if review:
        if requirement["coverage"] != review["expected_raw_verdict"]:
            raise UsageError(
                f"{review['query_id']} {review['requirement_id']} raw coverage changed"
            )
        coverage = review["corrected_verdict"]
        if review["corrected_claim_ids"] is not None:
            linked_ids = list(review["corrected_claim_ids"])
        reason = review["reason"]
    linked = [_require_claim(by_claim, claim_id) for claim_id in linked_ids]
    support, scope = _support_of(linked)
    factual = [row for row in linked if row["claim_type"] == "factual_claim"]
    missing = any(row["citation_required"] and not row["valid_citation_ids"] for row in factual)
    mismatch = any(
        pair["verdict"] == "does_not_support"
        for row in factual
        for pair in row["citation_pairs"]
    )
    hallucinated = any(row["citation_hallucination"] for row in factual)
    decision = classify_requirement(
        RequirementSignals(
            evidence_in_top10=bool(requirement["evidence_in_top10"]),
            coverage=coverage,
            support=support,
            scope_issue=scope,
            citation_missing=missing,
            citation_mismatch=mismatch,
            citation_hallucination=hallucinated,
        )
    )
    changed = (
        coverage != requirement["coverage"]
        or support != requirement["support"]
        or decision["primary"] != requirement["primary"]
        or decision["flags"] != requirement["flags"]
    )
    return {
        "requirement_id": requirement["requirement_id"],
        "text": requirement["text"],
        "evidence_in_top10": requirement["evidence_in_top10"],
        "raw_coverage": requirement["coverage"],
        "coverage": coverage,
        "raw_support": requirement["support"],
        "support": support,
        "raw_scope_issue": requirement["scope_issue"],
        "scope_issue": scope,
        "raw_linked_claim_ids": requirement["linked_claim_ids"],
        "linked_claim_ids": linked_ids,
        "raw_primary": requirement["primary"],
        "primary": decision["primary"],
        "raw_flags": requirement["flags"],
        "flags": decision["flags"],
        "raw_reason": requirement["reason"],
        "reason": reason,
        "review_applied": review is not None,
        "changed": changed,
    }


def _support_of(linked: list[dict[str, Any]]) -> tuple[str, str]:
    factual = [row for row in linked if row["claim_type"] == "factual_claim"]
    scopes = {row["raw_scope_issue"] for row in factual if row["raw_scope_issue"] != "none"}
    if len(scopes) == 1:
        scope = next(iter(scopes))
    elif scopes:
        scope = "mixed"
    else:
        scope = "none"
    if not factual:
        return "not_factual", "none"
    verdicts = [row["raw_verdict"] for row in factual]
    if any(verdict == "unjudged" for verdict in verdicts):
        return "unjudged", scope
    if "misinterpretation" in verdicts:
        return "misinterpretation", scope
    if "unsupported" in verdicts:
        return "unsupported", scope
    if all(verdict == "supported" for verdict in verdicts):
        return "supported", scope
    return "unjudged", scope


def _coverage_reviews() -> list[dict[str, Any]]:
    return [
        {
            "query_id": _SAMSUNG_CAPITAL_QUERY,
            "requirement_id": "R1",
            "expected_raw_verdict": "abstained",
            "corrected_verdict": "covered",
            "corrected_claim_ids": ["K1", "K2", "K3"],
            "corrected_answer_relevance": "direct",
            "judge_error": True,
            "requirement_satisfied": True,
            "reason": (
                "답변은 2025년 말 발행주식 액면총액 기준 보통주 약 87.9%, 우선주 약 12.1%를 말하고 "
                "C5를 붙였다. K1과 K2는 supported이고 citation도 supports다. "
                "K3는 납입자본금 897,514백만원이 종류별로 나뉘어 있지 않아 "
                "비중 기준이 액면총액이라는 한정이다. "
                "Raw judge는 K3만 연결하고, 한정이 있다는 이유로 abstained로 두었다. "
                "요구한 비중은 답변에 있다. 값의 참과 근거의 지지는 이미 supported다. "
                "Coverage를 covered로 고친다."
            ),
        },
        {
            "query_id": _ALTEOGEN_QUERY,
            "requirement_id": "R1",
            "expected_raw_verdict": "abstained",
            "corrected_verdict": "abstained",
            "corrected_claim_ids": None,
            "corrected_answer_relevance": None,
            "judge_error": False,
            "requirement_satisfied": False,
            "reason": (
                "요구는 분기보고서에 나온 기술지원·위탁생산 용역기간이다. "
                "답변은 제공된 분기보고서 근거에서 그 기간이 확인되지 않는다고 하고, "
                "2025년 사업보고서의 기간을 적은 뒤 그 기간을 분기보고서 기간으로 보지 않는다. "
                "선호 분기보고서 unit은 저장된 Top-10에 없다. "
                "관련 사업보고서 unit은 C5로 Top-10에 있다. "
                "요구한 분기보고서 기간은 답으로 제시되지 않았다. "
                "abstained는 그 기권과 같다. Coverage는 유지한다."
            ),
        },
    ]


def _check_context(generation: dict[str, Any]) -> None:
    units = {row["retrieval_unit_id"] for row in generation["context"]}
    query_id = generation["query_id"]
    if query_id == _SAMSUNG_CAPITAL_QUERY:
        if _SAMSUNG_CAPITAL_UNIT not in units:
            raise UsageError("samsung capital citation unit is missing from the stored context")
        answer = generation["answer"]
        if "87.9" not in answer or "12.1" not in answer:
            raise UsageError("samsung capital answer does not state the stored shares")
    if query_id == _ALTEOGEN_QUERY:
        if _PREFERRED_QUARTERLY_UNIT in units:
            raise UsageError("alteogen preferred quarterly unit is inside the stored top-10")
        if _ANNUAL_SERVICE_UNIT not in units:
            raise UsageError("alteogen annual service unit is missing from the stored context")


def _query_flags(
    requirements: list[dict[str, Any]],
    integrity: dict[str, Any],
    claims: list[dict[str, Any]],
) -> list[str]:
    flags: list[str] = []
    retrieved = [bool(row["evidence_in_top10"]) for row in requirements]
    if len(requirements) >= 2 and any(retrieved) and not all(retrieved):
        flags.append("evidence_coverage_failure")
    if integrity.get("unknown"):
        flags.append("citation_hallucination")
    if integrity.get("malformed"):
        flags.append("malformed_citation")
    missing = any(
        row["in_faithfulness_denominator"] and not row["valid_citation_ids"] for row in claims
    )
    if missing:
        flags.append("citation_missing")
    return flags


def _faith_block(claims: list[dict[str, Any]]) -> dict[str, Any]:
    denominator = [row for row in claims if row["in_faithfulness_denominator"]]
    numerator = sum(1 for row in denominator if row["faithfulness_supported"])
    return {
        "numerator": numerator,
        "denominator": len(denominator),
        "score": None if not denominator else numerator / len(denominator),
    }


def _correctness_block(claims: list[dict[str, Any]]) -> dict[str, Any]:
    supports = 0
    misses = 0
    excluded = 0
    for row in claims:
        for pair in row["citation_pairs"]:
            if not pair["in_corrected_denominator"]:
                excluded += 1
                continue
            if pair["verdict"] == "supports":
                supports += 1
            elif pair["verdict"] == "does_not_support":
                misses += 1
    judged = supports + misses
    return {
        "numerator": supports,
        "denominator": judged,
        "supports": supports,
        "does_not_support": misses,
        "excluded_pairs": excluded,
        "score": None if judged == 0 else supports / judged,
    }


def _completeness_block(claims: list[dict[str, Any]]) -> dict[str, Any]:
    denominator = [row for row in claims if row["in_completeness_denominator"]]
    numerator = sum(1 for row in denominator if row["completeness_hit"])
    return {
        "numerator": numerator,
        "denominator": len(denominator),
        "score": None if not denominator else numerator / len(denominator),
    }


def _summarize(
    results: list[dict[str, Any]],
    claims: list[dict[str, Any]],
    raw_summary: dict[str, Any],
    raw_manifest: dict[str, Any],
    source: Path,
) -> dict[str, Any]:
    covered = sum(
        1 for result in results for row in result["requirements"] if row["coverage"] == "covered"
    )
    omitted = sum(
        1 for result in results for row in result["requirements"] if row["coverage"] == "omitted"
    )
    abstained = sum(
        1 for result in results for row in result["requirements"] if row["coverage"] == "abstained"
    )
    unjudged = sum(
        1 for result in results for row in result["requirements"] if row["coverage"] == "unjudged"
    )
    judged = covered + omitted + abstained
    faith_num = sum(result["faithfulness"]["numerator"] for result in results)
    faith_den = sum(result["faithfulness"]["denominator"] for result in results)
    correct_num = sum(result["citation_correctness"]["numerator"] for result in results)
    correct_den = sum(result["citation_correctness"]["denominator"] for result in results)
    complete_num = sum(result["citation_completeness"]["numerator"] for result in results)
    complete_den = sum(result["citation_completeness"]["denominator"] for result in results)
    primaries = Counter(row["primary"] for result in results for row in result["requirements"])
    official: Counter[str] = Counter({name: 0 for name in OFFICIAL_FLAGS})
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
    types = Counter(row["claim_type"] for row in claims)
    appropriateness = Counter(
        row["abstention_appropriateness"]
        for row in claims
        if row["abstention_appropriateness"] is not None
    )
    relevance = Counter(result["answer_relevance"] for result in results)
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
    failures = [
        {
            "query_id": result["query_id"],
            "requirement_id": row["requirement_id"],
            "primary": row["primary"],
            "coverage": row["coverage"],
            "support": row["support"],
            "flags": row["flags"],
            "text": row["text"],
        }
        for result in results
        for row in result["requirements"]
        if row["primary"] == "generation_limited"
    ]
    citation_failures = [
        {
            "query_id": row["query_id"],
            "claim_id": row["claim_id"],
            "citation_id": pair["citation_id"],
            "verdict": pair["verdict"],
            "reason": pair["reason"],
        }
        for row in claims
        for pair in row["citation_pairs"]
        if pair["in_corrected_denominator"] and pair["verdict"] == "does_not_support"
    ]
    return {
        "baseline": "G0",
        "contract_version": CONTRACT_VERSION,
        "source_run": SOURCE_RUN_DIR,
        "queries": len(results),
        "generation_reused": len(results),
        "new_model_calls": 0,
        "judge_recalled": False,
        "requirements_total": sum(len(result["requirements"]) for result in results),
        "claim_types": {name: int(types[name]) for name in CLAIM_TYPES},
        "faithfulness": {
            "score": None if faith_den == 0 else faith_num / faith_den,
            "numerator": faith_num,
            "denominator": faith_den,
            "unjudged_claim_types": int(types["unjudged"]),
            "excluded_abstention": int(types["abstention"]),
            "excluded_limitation": int(types["limitation"]),
            "excluded_meta": int(types["meta"]),
            "denominator_definition": "judged_factual_claims",
        },
        "requirement_coverage": {
            "score": None if judged == 0 else covered / judged,
            "numerator": covered,
            "denominator": judged,
            "covered": covered,
            "omitted": omitted,
            "abstained": abstained,
            "unjudged": unjudged,
            "total": covered + omitted + abstained + unjudged,
            "denominator_definition": "judged_requirements",
        },
        "citation_correctness": {
            "score": None if correct_den == 0 else correct_num / correct_den,
            "numerator": correct_num,
            "denominator": correct_den,
            "supports": correct_num,
            "does_not_support": correct_den - correct_num,
            "excluded_pairs": sum(
                result["citation_correctness"]["excluded_pairs"] for result in results
            ),
            "denominator_definition": "judged_factual_citation_pairs",
        },
        "citation_completeness": {
            "score": None if complete_den == 0 else complete_num / complete_den,
            "numerator": complete_num,
            "denominator": complete_den,
            "denominator_definition": "factual_claims_that_require_citation",
        },
        "citation_integrity_error_count": int(raw_summary["citation_integrity_error_count"]),
        "success_requirements": int(primaries["success"]),
        "retrieval_limited_requirements": int(primaries["retrieval_limited"]),
        "generation_limited_requirements": int(primaries["generation_limited"]),
        "unjudged_requirements": int(primaries["unjudged"]),
        "retrieval_limited_queries": len(retrieval_queries),
        "generation_limited_queries": len(generation_queries),
        "taxonomy": dict(official),
        "diagnostic_flags": dict(diagnostic),
        "answer_relevance": dict(relevance),
        "abstention_appropriateness": {
            "appropriate": int(appropriateness["appropriate"]),
            "inappropriate": int(appropriateness["inappropriate"]),
            "unjudged": int(appropriateness["unjudged"]),
            "total": int(sum(appropriateness.values())),
        },
        "abstention_note": (
            "V2-00989619-finance_table-02 K1은 제공된 분기보고서 근거에 대한 기권이다. "
            "Raw judge는 그 문장을 문서 부재 주장으로 읽고 unsupported로 두었다. "
            "그 판정을 적절한 기권으로 바꾸지 않았고, faithfulness 분모에도 다시 넣지 않았다. "
            "적절성은 unjudged로 남아 있다."
        ),
        "coverage_reviews": [
            {
                "query_id": review["query_id"],
                "requirement_id": review["requirement_id"],
                "raw_verdict": review["expected_raw_verdict"],
                "corrected_verdict": review["corrected_verdict"],
                "judge_error": review["judge_error"],
                "requirement_satisfied": review["requirement_satisfied"],
                "reason": review["reason"],
            }
            for review in _coverage_reviews()
        ],
        "remaining_generation_failures": failures,
        "remaining_citation_mismatches": citation_failures,
        "hypothesis": _hypothesis(official),
        "raw_metrics": _raw_metrics(raw_summary),
        "source_manifest_git_commit": raw_manifest.get("git_commit"),
        "source_generations_sha256": _sha_file(source / "generations.jsonl"),
        "limits": (
            "유형은 문장 규칙으로 정했다. 새 judge는 호출하지 않았다. "
            "K1 알테오젠 기권의 적절성은 unjudged다. "
            "Citation correctness의 pair verdict는 다시 판단하지 않았다. "
            "Coverage 수정은 두 사례의 기록된 재검토다. "
            "Qrel과 retrieval policy는 바꾸지 않았다."
        ),
    }


def _deltas(
    summary: dict[str, Any],
    claims: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    raw = summary["raw_metrics"]
    return {
        "source_run": SOURCE_RUN_DIR,
        "corrected_run": CORRECTED_RUN_DIR,
        "metrics": {
            "faithfulness": _delta_ratio(raw["faithfulness"], summary["faithfulness"]),
            "requirement_coverage": _delta_ratio(
                raw["requirement_coverage"], summary["requirement_coverage"]
            ),
            "citation_correctness": _delta_ratio(
                raw["citation_correctness"], summary["citation_correctness"]
            ),
            "citation_completeness": _delta_ratio(
                raw["citation_completeness"], summary["citation_completeness"]
            ),
            "requirement_success": _delta_ratio(
                raw["requirement_success"],
                {
                    "numerator": summary["success_requirements"],
                    "denominator": summary["requirements_total"],
                },
            ),
        },
        "requirement_primary": {
            "raw": raw["requirement_primary"],
            "corrected": {
                "success": summary["success_requirements"],
                "retrieval_limited": summary["retrieval_limited_requirements"],
                "generation_limited": summary["generation_limited_requirements"],
                "unjudged": summary["unjudged_requirements"],
            },
        },
        "taxonomy": {"raw": raw["taxonomy"], "corrected": summary["taxonomy"]},
        "changed_claims": [
            {
                "query_id": row["query_id"],
                "claim_id": row["claim_id"],
                "claim_type": row["claim_type"],
                "rule": row["classification_rule"],
                "raw_verdict": row["raw_verdict"],
                "generator_factual": row["generator_factual"],
                "abstention_appropriateness": row["abstention_appropriateness"],
                "left_faithfulness_denominator": row["in_raw_faithfulness_denominator"]
                and not row["in_faithfulness_denominator"],
                "left_completeness_denominator": row["in_raw_faithfulness_denominator"]
                and not row["in_completeness_denominator"],
                "text": row["text"],
            }
            for row in claims
            if row["in_raw_faithfulness_denominator"] and not row["in_faithfulness_denominator"]
        ],
        "excluded_citation_pairs": [
            {
                "query_id": row["query_id"],
                "claim_id": row["claim_id"],
                "claim_type": row["claim_type"],
                "citation_id": pair["citation_id"],
                "verdict": pair["verdict"],
                "reason": "pair verdict kept; claim is outside the factual correctness denominator",
            }
            for row in claims
            for pair in row["citation_pairs"]
            if not pair["in_corrected_denominator"]
        ],
        "changed_requirements": [
            {
                "query_id": result["query_id"],
                "requirement_id": row["requirement_id"],
                "raw_coverage": row["raw_coverage"],
                "coverage": row["coverage"],
                "raw_primary": row["raw_primary"],
                "primary": row["primary"],
                "raw_support": row["raw_support"],
                "support": row["support"],
                "raw_flags": row["raw_flags"],
                "flags": row["flags"],
            }
            for result in results
            for row in result["requirements"]
            if row["changed"]
        ],
    }


def _raw_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "faithfulness": {
            "numerator": summary["faithfulness"]["numerator"],
            "denominator": summary["faithfulness"]["denominator"],
        },
        "requirement_coverage": {
            "numerator": summary["requirement_coverage"]["numerator"],
            "denominator": summary["requirement_coverage"]["denominator"],
        },
        "citation_correctness": {
            "numerator": summary["citation_correctness"]["numerator"],
            "denominator": summary["citation_correctness"]["denominator"],
        },
        "citation_completeness": {
            "numerator": summary["citation_completeness"]["numerator"],
            "denominator": summary["citation_completeness"]["denominator"],
        },
        "requirement_success": {
            "numerator": summary["success_requirements"],
            "denominator": summary["requirements_total"],
        },
        "requirement_primary": {
            "success": summary["success_requirements"],
            "retrieval_limited": summary["retrieval_limited_requirements"],
            "generation_limited": summary["generation_limited_requirements"],
            "unjudged": summary["unjudged_requirements"],
        },
        "taxonomy": summary["taxonomy"],
    }


def _manifest(
    root: Path,
    source: Path,
    raw_manifest: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    git = git_revision(root)
    generator_sha = prompt_sha256(GENERATOR_INSTRUCTIONS)
    if generator_sha != raw_manifest["prompt_sha256"]["generator"]:
        raise UsageError("generator prompt hash differs from the frozen G0 manifest")
    return {
        "baseline": "G0",
        "contract_version": CONTRACT_VERSION,
        "source_run": SOURCE_RUN_DIR,
        "source_evaluator_version": raw_manifest["evaluator_version"],
        "corrected_evaluator_version": CONTRACT_VERSION,
        "dataset_id": raw_manifest["dataset_id"],
        "dataset_sha256": raw_manifest["dataset_sha256"],
        "population": raw_manifest["population"],
        "question_count": summary["queries"],
        "requirement_count": summary["requirements_total"],
        "generation_reused": summary["generation_reused"],
        "source_generations_sha256": summary["source_generations_sha256"],
        "source_generations_path": str(source / "generations.jsonl"),
        "new_model_calls": 0,
        "judge_recalled": False,
        "retrieval_reexecuted": False,
        "test_questions_generated": 0,
        "test_questions_searched": 0,
        "qrels_changed": False,
        "retrieval_policy": raw_manifest["retrieval_policy"],
        "retrieval_run_id": raw_manifest["retrieval_run_id"],
        "generator_model": raw_manifest["generator_model"],
        "judge_model": raw_manifest["judge_model"],
        "generator_prompt_version": raw_manifest["prompt_versions"]["generator"],
        "generator_prompt_sha256": generator_sha,
        "generator_prompt_changed": False,
        "raw_prompt_sha256": raw_manifest["prompt_sha256"],
        "git_commit": git.get("git_commit"),
        "git_dirty": git.get("git_dirty"),
        "source_manifest_git_commit": raw_manifest.get("git_commit"),
    }


def _assert_raw_contract(
    generations: list[dict[str, Any]],
    results: dict[str, dict[str, Any]],
    summary: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    if len(generations) != 45 or summary["queries"] != 45:
        raise UsageError("raw G0 generation count is not 45")
    if summary["requirements_total"] != 58:
        raise UsageError("raw G0 requirement count is not 58")
    if manifest["evaluator_version"] != RAW_EVALUATOR_VERSION:
        raise UsageError("raw G0 evaluator version changed")
    expected = {
        "faithfulness": (73, 75),
        "requirement_coverage": (50, 58),
        "citation_correctness": (71, 73),
        "citation_completeness": (68, 75),
    }
    for name, (numerator, denominator) in expected.items():
        block = summary[name]
        if int(block["numerator"]) != numerator or int(block["denominator"]) != denominator:
            raise UsageError(f"raw G0 {name} is not the frozen score")
    if {row["query_id"] for row in generations} != set(results):
        raise UsageError("raw G0 generations and results do not match")


def _question_text(root: Path) -> dict[str, str]:
    path = root / "evaluation" / "datasets" / "multi_company_retrieval_v2_1" / "questions.json"
    payload = _read_json(path)
    return {row["query_id"]: row["question"] for row in payload["questions"]}


def _source_dir(root: Path) -> Path:
    return root / "evaluation" / "runs" / SOURCE_RUN_DIR


def _require_claim(by_claim: dict[str, dict[str, Any]], claim_id: str) -> dict[str, Any]:
    if claim_id not in by_claim:
        raise UsageError(f"linked claim {claim_id} is missing")
    return by_claim[claim_id]


def _delta_ratio(raw: dict[str, Any], corrected: dict[str, Any]) -> dict[str, Any]:
    return {
        "raw": {
            "numerator": int(raw["numerator"]),
            "denominator": int(raw["denominator"]),
        },
        "corrected": {
            "numerator": int(corrected["numerator"]),
            "denominator": int(corrected["denominator"]),
        },
    }


def _pair(label: str, block: dict[str, Any]) -> str:
    raw = block["raw"]
    corrected = block["corrected"]
    return (
        f"| {label} | {raw['numerator']}/{raw['denominator']} | "
        f"{corrected['numerator']}/{corrected['denominator']} |"
    )


def _claim_lines(rows: list[dict[str, Any]]) -> str:
    lines = []
    for row in rows:
        appropriateness = row["abstention_appropriateness"] or "-"
        lines.append(
            f"- `{row['query_id']}` {row['claim_id']} {row['claim_type']} "
            f"raw={row['raw_verdict']} appropriateness={appropriateness}: "
            f"{row['text']}"
        )
    return "\n".join(lines)


def _requirement_lines(rows: list[dict[str, Any]]) -> str:
    lines = []
    for row in rows:
        lines.append(
            f"- `{row['query_id']}` {row['requirement_id']} "
            f"coverage {row['raw_coverage']} -> {row['coverage']}, "
            f"support {row['raw_support']} -> {row['support']}, "
            f"primary {row['raw_primary']} -> {row['primary']}, "
            f"flags {','.join(row['raw_flags']) or '-'} -> {','.join(row['flags']) or '-'}"
        )
    return "\n".join(lines)


def _failure_lines(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "Generation-limited requirement가 없다."
    lines = []
    for row in rows:
        lines.append(
            f"- `{row['query_id']}` {row['requirement_id']} {row['primary']} "
            f"coverage={row['coverage']} support={row['support']} "
            f"flags={','.join(row['flags']) or '-'}: {row['text']}"
        )
    return "\n".join(lines)


def _mismatch_lines(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "남은 does_not_support pair가 없다."
    lines = []
    for row in rows:
        lines.append(
            f"- `{row['query_id']}` {row['claim_id']} {row['citation_id']} "
            f"{row['verdict']}: {row['reason']}"
        )
    return "\n".join(lines)


def _count_table(counts: dict[str, Any]) -> str:
    lines = ["| name | count |", "| --- | ---: |"]
    for name, count in counts.items():
        lines.append(f"| {name} | {count} |")
    return "\n".join(lines)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

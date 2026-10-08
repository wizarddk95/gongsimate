"""Score one product-agent run against the A0 labels.

The labels are not an agent prompt. Tool choice and the final message are
scored together. A useless search can still leave the task itself successful.
"""

from __future__ import annotations

import json
from typing import Any

from disclosure_agent.agent_a0.dataset import agent_input
from disclosure_agent.evalset_v2_1.g0_evaluation_contract import (
    citation_requirement,
    classify_claim,
)

FACTUAL_VERDICTS = ("supported", "unsupported", "misinterpretation")
FAILURES = (
    "tool_selection_failure",
    "unnecessary_search",
    "missed_search",
    "clarification_failure",
    "tool_argument_failure",
    "retrieval_failure",
    "generation_failure",
    "unsupported_claim",
    "citation_failure",
    "over_abstention",
    "under_abstention",
    "unnecessary_retry",
    "budget_exhaustion",
    "timeout",
    "runtime_failure",
)
_RUNTIME_STATUS = {
    "internal_error": "runtime_failure",
    "retrieval_failure": "runtime_failure",
    "generation_failure": "generation_failure",
    "request_timeout": "timeout",
    "llm_limit_exceeded": "budget_exhaustion",
    "search_limit_exceeded": "budget_exhaustion",
}
TASK_OUTCOMES = (
    "success",
    "partial_success",
    "appropriate_clarification",
    "appropriate_abstention",
    "failure",
    "unjudged",
)
_POSITIVE = set(TASK_OUTCOMES) - {"failure", "unjudged"}


def grade_case(
    case: dict[str, Any],
    record: dict[str, Any] | None,
    judgment: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return the scored row. ``record`` is the product run record, not the label."""
    if agent_input(case) != case["question"]:
        raise ValueError("agent input drifted away from the question text")
    if record is None:
        return _failed_row(case, "runtime_failure", "the agent run was not recorded")
    calls = _search_calls(record)
    behaviors = observed_behaviors(record, calls)
    primary = primary_behavior(behaviors)
    decision = search_decision(case, calls)
    arguments = argument_findings(case, calls)
    citations = citation_findings(record, judgment)
    coverage = requirement_findings(case, judgment)
    task = task_outcome(
        case,
        record,
        primary=primary,
        judgment=judgment,
        coverage=coverage,
        citations=citations,
        arguments=arguments,
    )
    flags = _flags(
        case,
        record,
        calls,
        primary=primary,
        decision=decision,
        arguments=arguments,
        citations=citations,
        coverage=coverage,
        task=task,
        judgment=judgment,
    )
    cause = primary_cause(flags, task)
    secondary = [item for item in flags if item != cause]
    return {
        "query_id": case["query_id"],
        "scenario": case["scenario"],
        "question": case["question"],
        "expected_behavior": case["expected_behavior"],
        "allowed_behaviors": list(case["allowed_behaviors"]),
        "observed_behaviors": behaviors,
        "primary_behavior": primary,
        "behavior_allowed": primary in case["allowed_behaviors"]
        or (
            primary == "SEARCH"
            and "ADDITIONAL_SEARCH" in behaviors
            and "SEARCH" in case["allowed_behaviors"]
        ),
        "search_decision": decision,
        "argument_findings": arguments,
        "task_outcome": task,
        "grounding": citations,
        "requirements": coverage,
        "primary_cause": cause,
        "secondary_flags": secondary,
        "stop_reason": record.get("stop_reason") or record.get("status"),
        "error_status": (record.get("error") or {}).get("code")
        if isinstance(record.get("error"), dict)
        else None,
        "benchmark_unit_in_retrieved": _unit_overlap(case, record),
    }


def observed_behaviors(record: dict[str, Any], calls: list[dict[str, Any]]) -> list[str]:
    final = record.get("final_answer") if isinstance(record.get("final_answer"), dict) else {}
    action = record.get("agent_action")
    status = record.get("status")
    abstain = bool(final.get("abstain")) or action == "abstain"
    limitations = final.get("limitations") or []
    found: list[str] = []
    if calls:
        found.append("SEARCH")
    if len(calls) >= 2:
        found.append("ADDITIONAL_SEARCH")
    if action == "ask_clarification" or status in {
        "company_ambiguous",
        "company_not_found",
        "needs_clarification",
    }:
        found.append("CLARIFY")
    if abstain:
        found.append("ABSTAIN")
    if action == "answer_directly":
        found.append("DIRECT_ANSWER")
    if calls and not abstain and limitations and "CLARIFY" not in found:
        found.append("PARTIAL_ANSWER")
    return list(dict.fromkeys(found))


def primary_behavior(behaviors: list[str]) -> str:
    present = set(behaviors)
    if "CLARIFY" in present and "SEARCH" not in present:
        return "CLARIFY"
    if "ABSTAIN" in present:
        return "ABSTAIN"
    if "PARTIAL_ANSWER" in present:
        return "PARTIAL_ANSWER"
    if "DIRECT_ANSWER" in present and "SEARCH" not in present:
        return "DIRECT_ANSWER"
    if "SEARCH" in present:
        return "SEARCH"
    if "DIRECT_ANSWER" in present:
        return "DIRECT_ANSWER"
    return "UNOBSERVED"


def search_decision(case: dict[str, Any], calls: list[dict[str, Any]]) -> dict[str, Any]:
    searched = bool(calls)
    expectation = case["search_expectation"]
    if expectation == "optional":
        return {
            "expectation": expectation,
            "searched": searched,
            "counts_for_accuracy": False,
            "correct": None,
            "missed_required_search": False,
            "unnecessary_search": False,
        }
    correct = searched if expectation == "required" else not searched
    return {
        "expectation": expectation,
        "searched": searched,
        "counts_for_accuracy": True,
        "correct": correct,
        "missed_required_search": expectation == "required" and not searched,
        "unnecessary_search": expectation == "prohibited" and searched,
    }


def argument_findings(case: dict[str, Any], calls: list[dict[str, Any]]) -> list[str]:
    findings: list[str] = []
    expected_company = case.get("company_id")
    period = case.get("period")
    compatible = False
    contradicted = False
    for call in calls:
        arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        company_id = str(arguments.get("company_id") or call.get("company_id") or "").strip()
        if not str(arguments.get("query") or call.get("query") or "").strip() or not company_id:
            findings.append("missing_required_arguments")
        if call.get("status") == "invalid_arguments":
            findings.append("invalid_arguments")
        if call.get("status") == "unsupported_company":
            findings.append("unauthorized_scope")
        if expected_company and company_id and company_id != expected_company:
            findings.append("wrong_company")
        if expected_company is None and company_id:
            findings.append("unauthorized_scope")
        if isinstance(period, dict):
            relation = _period_relation(arguments, period)
            compatible = compatible or relation == "compatible"
            contradicted = contradicted or relation == "contradicts"
    if isinstance(period, dict) and calls and not compatible and contradicted:
        findings.append("wrong_period")
    if (
        case.get("answerability") == "fully_answerable"
        and calls
        and all(call.get("status") == "no_matching_filing" for call in calls)
    ):
        findings.append("overly_restrictive_scope")
    return list(dict.fromkeys(findings))


def citation_findings(
    record: dict[str, Any],
    judgment: dict[str, Any] | None,
) -> dict[str, Any]:
    final = record.get("final_answer") if isinstance(record.get("final_answer"), dict) else {}
    evidence = record.get("generation_evidence") or []
    valid_ids = {str(item.get("evidence_id")) for item in evidence if item.get("evidence_id")}
    citations = final.get("citations") or []
    unknown = [
        item.get("evidence_id")
        for item in citations
        if str(item.get("evidence_id") or "") not in valid_ids
    ]
    claims = split_claims(str(final.get("text") or ""))
    classified = []
    for claim in claims:
        kind = classify_claim(claim["text"])
        need = citation_requirement(kind["claim_type"])
        classified.append({**claim, **kind, **need})
    factual = [item for item in classified if item["claim_type"] == "factual_claim"]
    abstain = bool(final.get("abstain")) or record.get("agent_action") in {
        "abstain",
        "ask_clarification",
        "answer_directly",
    }
    citation_required = (
        bool(factual) and not abstain and record.get("agent_action") == "search_disclosures"
    )
    complete = (not citation_required) or bool(citations)
    judged = {}
    if isinstance(judgment, dict):
        judged = {
            item["claim_id"]: item
            for item in judgment.get("faithfulness") or []
            if isinstance(item, dict) and item.get("claim_id")
        }
    faithful_den = 0
    faithful_num = 0
    for claim in factual:
        verdict = str((judged.get(claim["claim_id"]) or {}).get("verdict") or "")
        if verdict in FACTUAL_VERDICTS:
            faithful_den += 1
            if verdict == "supported":
                faithful_num += 1
    citation_pairs = judgment.get("citations") if isinstance(judgment, dict) else None
    correct_den = 0
    correct_num = 0
    if isinstance(citation_pairs, list):
        for item in citation_pairs:
            verdict = str(item.get("verdict") or "")
            if verdict in {"supports", "does_not_support"}:
                correct_den += 1
                if verdict == "supports":
                    correct_num += 1
    return {
        "claim_count": len(classified),
        "factual_claim_count": len(factual),
        "claims": classified,
        "citation_count": len(citations),
        "unknown_citation_ids": unknown,
        "integrity_error_count": len(unknown),
        "citation_required": citation_required,
        "citation_complete": complete,
        "faithfulness_supported": faithful_num,
        "faithfulness_denominator": faithful_den,
        "citation_correct": correct_num,
        "citation_correct_denominator": correct_den,
        "faithfulness_judged": faithful_den > 0,
    }


def requirement_findings(case: dict[str, Any], judgment: dict[str, Any] | None) -> dict[str, Any]:
    requirements = case.get("requirements") or []
    by_id = {}
    if isinstance(judgment, dict):
        by_id = {
            item["requirement_id"]: item
            for item in judgment.get("requirements") or []
            if isinstance(item, dict) and item.get("requirement_id")
        }
    items = []
    for requirement in requirements:
        judged = by_id.get(requirement["requirement_id"])
        verdict = None if judged is None else judged.get("verdict")
        success = verdict in set(requirement.get("success_verdicts") or ["covered"])
        items.append(
            {
                "requirement_id": requirement["requirement_id"],
                "verdict": verdict,
                "success": bool(success) if verdict is not None else None,
            }
        )
    judged_items = [item for item in items if item["verdict"] is not None]
    return {
        "items": items,
        "judged": bool(requirements) and len(judged_items) == len(requirements),
        "success_count": sum(1 for item in judged_items if item["success"]),
        "denominator": len(judged_items),
    }


def task_outcome(
    case: dict[str, Any],
    record: dict[str, Any],
    *,
    primary: str,
    judgment: dict[str, Any] | None,
    coverage: dict[str, Any],
    citations: dict[str, Any],
    arguments: list[str],
) -> str:
    status = str(record.get("status") or "")
    if status in _RUNTIME_STATUS or record.get("error"):
        if status in {"request_timeout", "llm_limit_exceeded", "search_limit_exceeded"}:
            return "failure"
        if status in {"internal_error", "retrieval_failure", "generation_failure"}:
            return "failure"
    if primary not in case["allowed_behaviors"]:
        return "failure"
    if any(
        name in set(arguments)
        for name in ("wrong_company", "wrong_period", "unauthorized_scope")
    ):
        return "failure"
    if case["scenario"] == "no_search":
        return "success"
    if case["scenario"] == "ambiguous":
        return "appropriate_clarification"
    if not case.get("requirements"):
        return "unjudged"
    if judgment is None or not coverage["judged"] or not _faith_complete(citations, judgment):
        return "unjudged"
    if citations["integrity_error_count"]:
        return "failure"
    successes = coverage["success_count"]
    total = coverage["denominator"]
    unsupported = _unsupported(citations, judgment)
    if unsupported and case["scenario"] == "insufficient_evidence":
        return "failure"
    if successes == total and not unsupported:
        if case["answerability"] == "partially_answerable":
            return "partial_success"
        if case["scenario"] == "insufficient_evidence":
            return "appropriate_abstention"
        return "success"
    if successes and not unsupported:
        return "partial_success"
    if case["scenario"] == "insufficient_evidence" and primary in {
        "ABSTAIN",
        "DIRECT_ANSWER",
        "CLARIFY",
    }:
        if successes:
            return "appropriate_abstention"
        return "failure"
    return "failure"


def primary_cause(flags: list[str], task: str) -> str | None:
    if task == "failure":
        for name in ("runtime_failure", "timeout", "budget_exhaustion"):
            if name in flags:
                return name
    skipped: set[str] = set()
    if task in _POSITIVE:
        skipped.update({"unnecessary_search", "tool_selection_failure", "unnecessary_retry"})
    for name in FAILURES:
        if name in flags and name not in skipped:
            return name
    if task == "failure":
        return "generation_failure"
    return None


def split_claims(text: str) -> list[dict[str, str]]:
    parts = [part.strip() for part in text.splitlines() if part.strip()]
    if not parts and text.strip():
        parts = [text.strip()]
    return [{"claim_id": f"K{index}", "text": part} for index, part in enumerate(parts, start=1)]


def _flags(
    case: dict[str, Any],
    record: dict[str, Any],
    calls: list[dict[str, Any]],
    *,
    primary: str,
    decision: dict[str, Any],
    arguments: list[str],
    citations: dict[str, Any],
    coverage: dict[str, Any],
    task: str,
    judgment: dict[str, Any] | None,
) -> list[str]:
    flags: list[str] = []
    status = str(record.get("status") or "")
    mapped = _RUNTIME_STATUS.get(status)
    if mapped:
        flags.append(mapped)
    if decision["missed_required_search"]:
        flags.extend(["missed_search", "tool_selection_failure"])
    if decision["unnecessary_search"]:
        flags.extend(["unnecessary_search", "tool_selection_failure"])
    if case["scenario"] == "ambiguous" and primary != "CLARIFY":
        flags.append("clarification_failure")
    if arguments:
        flags.append("tool_argument_failure")
    if _duplicate_calls(calls):
        flags.append("unnecessary_retry")
    if citations["integrity_error_count"] or (
        citations["citation_required"] and not citations["citation_complete"]
    ):
        flags.append("citation_failure")
    if _unsupported(citations, judgment):
        flags.append("unsupported_claim")
    stopped = status in {
        "llm_limit_exceeded",
        "search_limit_exceeded",
        "request_timeout",
        "internal_error",
    }
    search_returned = any(str(call.get("status")) == "success" for call in calls)
    if (
        case.get("answerability") == "fully_answerable"
        and calls
        and not (record.get("generation_evidence") or [])
        and task == "failure"
        and not (stopped and search_returned)
    ):
        flags.append("retrieval_failure")
    if (
        case.get("answerability") == "fully_answerable"
        and primary == "ABSTAIN"
        and record.get("generation_evidence")
        and task == "failure"
    ):
        flags.append("over_abstention")
    if case["scenario"] == "insufficient_evidence" and "unsupported_claim" in flags:
        flags.append("under_abstention")
    if task == "failure" and not flags:
        flags.append("generation_failure")
    return [name for name in FAILURES if name in set(flags)]


def _faith_complete(citations: dict[str, Any], judgment: dict[str, Any] | None) -> bool:
    factual_ids = {
        claim["claim_id"] for claim in citations["claims"] if claim["claim_type"] == "factual_claim"
    }
    if not factual_ids:
        return True
    if not isinstance(judgment, dict):
        return False
    judged = {
        str(item.get("claim_id"))
        for item in judgment.get("faithfulness") or []
        if isinstance(item, dict)
    }
    return factual_ids <= judged


def _unsupported(citations: dict[str, Any], judgment: dict[str, Any] | None) -> bool:
    if not isinstance(judgment, dict):
        return False
    judged = {
        item.get("claim_id"): item
        for item in judgment.get("faithfulness") or []
        if isinstance(item, dict)
    }
    for claim in citations["claims"]:
        if claim["claim_type"] != "factual_claim":
            continue
        verdict = str((judged.get(claim["claim_id"]) or {}).get("verdict") or "")
        if verdict in {"unsupported", "misinterpretation"}:
            return True
    return False


def _search_calls(record: dict[str, Any]) -> list[dict[str, Any]]:
    calls = record.get("tool_calls") or []
    return [call for call in calls if call.get("name") == "search_disclosures"]


def _period_relation(arguments: dict[str, Any], expected: dict[str, Any]) -> str:
    period = arguments.get("period") if isinstance(arguments.get("period"), dict) else {}
    report_types = arguments.get("report_types")
    constrained = False
    for key in ("year", "quarter", "half"):
        actual = period.get(key)
        if actual is not None:
            constrained = True
            if expected.get(key) is not None and actual != expected[key]:
                return "contradicts"
    if isinstance(report_types, list) and report_types:
        constrained = True
        expected_types = expected.get("report_types")
        if isinstance(expected_types, list) and expected_types:
            if set(report_types).isdisjoint(expected_types):
                return "contradicts"
    if not constrained:
        return "unconstrained"
    return "compatible"


def _duplicate_calls(calls: list[dict[str, Any]]) -> bool:
    seen: set[str] = set()
    for call in calls:
        arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        key = json.dumps(
            {
                "query": arguments.get("query"),
                "company_id": arguments.get("company_id"),
                "period": arguments.get("period"),
                "report_types": arguments.get("report_types"),
                "filing_date_from": arguments.get("filing_date_from"),
                "filing_date_to": arguments.get("filing_date_to"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if key in seen:
            return True
        seen.add(key)
    return False


def _unit_overlap(case: dict[str, Any], record: dict[str, Any]) -> bool | None:
    expected = set(case.get("required_evidence_references") or [])
    if not expected:
        return None
    retrieved = set(record.get("retrieved_retrieval_unit_ids") or [])
    return bool(expected & retrieved)


def _failed_row(case: dict[str, Any], cause: str, detail: str) -> dict[str, Any]:
    return {
        "query_id": case["query_id"],
        "scenario": case["scenario"],
        "question": case["question"],
        "expected_behavior": case["expected_behavior"],
        "allowed_behaviors": list(case["allowed_behaviors"]),
        "observed_behaviors": [],
        "primary_behavior": "UNOBSERVED",
        "behavior_allowed": False,
        "search_decision": {
            "expectation": case["search_expectation"],
            "searched": False,
            "counts_for_accuracy": case["search_expectation"] != "optional",
            "correct": False if case["search_expectation"] == "required" else True,
            "missed_required_search": case["search_expectation"] == "required",
            "unnecessary_search": False,
        },
        "argument_findings": [],
        "task_outcome": "failure",
        "grounding": {
            "claim_count": 0,
            "factual_claim_count": 0,
            "claims": [],
            "citation_count": 0,
            "unknown_citation_ids": [],
            "integrity_error_count": 0,
            "citation_required": False,
            "citation_complete": True,
            "faithfulness_supported": 0,
            "faithfulness_denominator": 0,
            "citation_correct": 0,
            "citation_correct_denominator": 0,
            "faithfulness_judged": False,
        },
        "requirements": {"items": [], "judged": False, "success_count": 0, "denominator": 0},
        "primary_cause": cause,
        "secondary_flags": [],
        "stop_reason": None,
        "error_status": detail,
        "benchmark_unit_in_retrieved": None,
    }

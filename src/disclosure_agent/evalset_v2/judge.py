"""Blind two-judge relevance labels, with a third model only on disagreement."""

from __future__ import annotations

from typing import Any

from disclosure_agent.errors import UsageError
from disclosure_agent.evalset_v2.generate import logged_complete
from disclosure_agent.evalset_v2.llm import (
    CallLog,
    StructuredClient,
    coerce_judgment,
    judge_payload,
    judgments_agree,
)
from disclosure_agent.evalset_v2.prompts import JUDGE_INSTRUCTIONS
from disclosure_agent.evalset_v2.schemas import PassageJudgment


def judge_candidate(
    *,
    query_id: str,
    question: str,
    requirements: list[dict[str, Any]],
    candidate: dict[str, Any],
    judge_a: StructuredClient,
    judge_b: StructuredClient,
    adjudicator: StructuredClient,
    log_a: CallLog,
    log_b: CallLog,
    log_c: CallLog,
    max_retries: int,
    reasoning_setting: str,
) -> dict[str, Any]:
    requirement_ids = [item["requirement_id"] for item in requirements]
    payload = judge_payload(question, requirements, _visible_candidate(candidate))
    left = _coerce_call(
        log_a,
        judge_a,
        stage="judge_a",
        payload=payload,
        requirement_ids=requirement_ids,
        max_retries=max_retries,
        reasoning_setting=reasoning_setting,
    )
    right = _coerce_call(
        log_b,
        judge_b,
        stage="judge_b",
        payload=payload,
        requirement_ids=requirement_ids,
        max_retries=max_retries,
        reasoning_setting=reasoning_setting,
    )
    adjudicated = False
    final = None
    source = None
    if left is not None and right is not None and judgments_agree(left, right):
        final = left
        source = "agreement"
    elif left is not None and right is not None:
        adjudicated = True
        final = _coerce_call(
            log_c,
            adjudicator,
            stage="adjudicate",
            payload=payload,
            requirement_ids=requirement_ids,
            max_retries=max_retries,
            reasoning_setting=reasoning_setting,
        )
        source = "adjudicator" if final is not None else "unresolved"
    else:
        source = "unresolved"
    return {
        "query_id": query_id,
        "candidate_id": candidate["candidate_id"],
        "retrieval_unit_id": candidate["retrieval_unit_id"],
        "element_ids": list(candidate["element_ids"]),
        "document_name": candidate.get("document_name"),
        "filing_period": candidate.get("filing_period"),
        "judge_a": left,
        "judge_b": right,
        "adjudicated": adjudicated,
        "final": final,
        "final_source": source,
        "models": {
            "judge_a": judge_a.model,
            "judge_b": judge_b.model,
            "adjudicator": adjudicator.model,
        },
    }


def _visible_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "text": candidate.get("text"),
        "document_name": candidate.get("document_name"),
        "filing_period": candidate.get("filing_period"),
        "section_path": list(candidate.get("section_path") or []),
        "receipt_date": candidate.get("receipt_date"),
    }


def _coerce_call(
    log: CallLog,
    client: StructuredClient,
    *,
    stage: str,
    payload: dict[str, Any],
    requirement_ids: list[str],
    max_retries: int,
    reasoning_setting: str,
) -> dict[str, Any] | None:
    feedback = None
    current = payload
    for _ in range(2):
        if feedback:
            current = {**payload, "feedback": feedback}
        try:
            record = logged_complete(
                log,
                client,
                stage=stage,
                instructions=JUDGE_INSTRUCTIONS,
                payload=current,
                schema=PassageJudgment,
                max_retries=max_retries,
                reasoning_setting=reasoning_setting,
            )
        except UsageError:
            return None
        coerced = coerce_judgment(record["output"], requirement_ids)
        if coerced is not None:
            return coerced
        feedback = "requirement_id는 입력과 같아야 하고 relevance는 0, 1, 2만 가능하다."
    return None


def agreement_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    compared = [
        row
        for row in rows
        if row.get("judge_a") is not None and row.get("judge_b") is not None
    ]
    agreed = sum(
        1 for row in compared if not row.get(
            "adjudicated") and row.get("final_source") == "agreement")
    adjudicated = sum(1 for row in rows if row.get("adjudicated"))
    unresolved = sum(
        1 for row in rows if row.get("final_source") == "unresolved" or row.get("final") is None)
    return {
        "compared": len(compared),
        "agreement_count": agreed,
        "agreement_rate": None if not compared else agreed / len(compared),
        "disagreement_count": len(compared) - agreed,
        "adjudication_count": adjudicated,
        "unresolved_count": unresolved,
        "note": "Agreement is between two LLM judges. It is not human correctness.",
    }

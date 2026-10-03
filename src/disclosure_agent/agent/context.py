"""Pack dense-ranked evidence into the answer-model context budget.

This is not a relevance selector. Dense rank stays as it is. Exact duplicate
texts are dropped. Items are then taken in that order while their prompt cost
fits the token budget. ``max_evidence`` is only a safety cap.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from disclosure_agent.agent.evidence import Evidence
from disclosure_agent.agent.tokens import CONTEXT_TOKEN_COUNTER_ID, count_context_tokens


@dataclass(frozen=True)
class ContextSelection:
    evidence: list[Evidence]
    token_count: int
    token_budget: int
    token_counter: str
    max_evidence: int


class ContextBuilder:
    def __init__(self, *, token_budget: int, max_evidence: int) -> None:
        if token_budget < 1 or max_evidence < 1:
            raise ValueError("context budget and max evidence must be >= 1")
        self.token_budget = token_budget
        self.max_evidence = max_evidence

    def build(self, evidence: list[Evidence]) -> ContextSelection:
        selected: list[Evidence] = []
        seen_text: set[str] = set()
        tokens = 0
        for item in evidence:
            key = " ".join(item.text.split())
            if key in seen_text:
                continue
            if len(selected) >= self.max_evidence:
                break
            cost = evidence_token_cost(item)
            if tokens + cost > self.token_budget:
                break
            selected.append(item)
            seen_text.add(key)
            tokens += cost
        return ContextSelection(
            evidence=selected,
            token_count=tokens,
            token_budget=self.token_budget,
            token_counter=CONTEXT_TOKEN_COUNTER_ID,
            max_evidence=self.max_evidence,
        )


def evidence_prompt_payload(item: Evidence) -> dict[str, object]:
    """The evidence object placed in the answer-model input.

    Filing fields tell the model which disclosure a passage came from.
    They are not a second relevance rank.
    """
    return {
        "evidence_id": item.evidence_id,
        "company": item.company_name,
        "report_name": item.report_name,
        "receipt_no": item.receipt_no,
        "filing_period": item.filing_period,
        "receipt_date": item.receipt_date,
        "document_name": item.document_name,
        "section_path": list(item.section_path),
        "text": item.text,
    }


def evidence_token_cost(item: Evidence) -> int:
    """Cost of the JSON object the model actually receives for this evidence."""
    encoded = json.dumps(evidence_prompt_payload(item), ensure_ascii=False, separators=(",", ":"))
    return count_context_tokens(encoded)

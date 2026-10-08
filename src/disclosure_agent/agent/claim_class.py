"""Product claim types for answers. Generator factual flags are not an input.

G0's wording rules stay the first pass. A0 adds acts that G0 left as
verifiable assertions: evidence-scoped nonconfirmation, support-scope limits,
tool failure, and refusal of an unrealized confirmed fact. A claim that the
document itself lacks the information stays a factual claim.
"""

from __future__ import annotations

import re

from disclosure_agent.evalset_v2_1.g0_evaluation_contract import classify_claim

_AMOUNT = re.compile(r"\d{1,3}(?:,\d{3})+|\d+\s*(?:원|억|조)")
_FUTURE_REFUSAL = re.compile(r"아직|발생하지|실현되지|오지\s*않|알\s*수\s*없")
_FUTURE_TOPIC = re.compile(r"확정|미래")
_TOOL_FAILURE = re.compile(r"검색을\s*완료하지\s*못했|도구\s*실행|실행에\s*실패|오류가\s*발생")
_OUTSIDE_SCOPE = re.compile(
    r"지원하는\s*공시가\s*없|검색\s*대상이\s*아니|색인에\s*없|수집하지\s*않|범위\s*밖|지원\s*범위"
)
_NONCONFIRM = re.compile(r"확인하지\s*못했|확인할\s*수\s*없|확인되지\s*않|찾지\s*못했")
_CLARIFICATION = re.compile(r"\?|까요|나요\?|말씀하")


def classify_product_claim(text: str) -> dict[str, str]:
    """Classify one answer sentence. Does not read a generator factual flag."""
    base = classify_claim(text)
    if base["claim_type"] != "factual_claim":
        return {**base, "a0_act": base["claim_type"]}
    if base["rule"] == "document_absence_is_factual":
        return {**base, "a0_act": "document_absence"}
    sentence = " ".join(text.split())
    if _TOOL_FAILURE.search(sentence):
        return {
            "claim_type": "abstention",
            "rule": "tool_execution_failure",
            "a0_act": "tool_failure",
        }
    if _OUTSIDE_SCOPE.search(sentence):
        return {
            "claim_type": "limitation",
            "rule": "outside_support_scope",
            "a0_act": "outside_scope",
        }
    future_refusal = _FUTURE_REFUSAL.search(sentence) and _FUTURE_TOPIC.search(sentence)
    if future_refusal and not _AMOUNT.search(sentence):
        return {
            "claim_type": "abstention",
            "rule": "future_confirmed_fact_refusal",
            "a0_act": "future_confirmed_fact",
        }
    if _NONCONFIRM.search(sentence):
        if _AMOUNT.search(sentence):
            return {
                "claim_type": "unjudged",
                "rule": "mixed_amount_and_nonconfirmation",
                "a0_act": "unjudged",
            }
        return {
            "claim_type": "abstention",
            "rule": "provided_evidence_nonconfirmation",
            "a0_act": "evidence_nonconfirmation",
        }
    if _CLARIFICATION.search(sentence) and not _AMOUNT.search(sentence):
        return {
            "claim_type": "meta",
            "rule": "clarification_question",
            "a0_act": "clarification",
        }
    return {**base, "a0_act": "factual_claim"}

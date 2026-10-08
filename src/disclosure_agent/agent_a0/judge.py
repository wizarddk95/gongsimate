"""Grounding judge for an A0 run.

The judge sees the answer and the evidence that this run actually retrieved.
It does not see tool-choice labels, seed ids, or the G0 generator output.
Faithfulness and citation checks are separate from requirement coverage so a
requirement sentence cannot supply a missing number.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from disclosure_agent.agent_a0.grade import split_claims
from disclosure_agent.evalset_v2.llm import StructuredClient

JUDGE_PROMPT_VERSION = "a0-grounding-v1"
_EVIDENCE_CHARS = 900
_MAX_EVIDENCE = 8
_FORBIDDEN = {
    "expected_behavior",
    "allowed_behaviors",
    "prohibited_behaviors",
    "gold",
    "seed_unit_ids",
    "required_evidence_references",
    "search_expectation",
    "scenario",
}

FAITH_INSTRUCTIONS = """각 claim이 제공된 evidence만으로 지지되는지 판단합니다.
외부 지식으로 빈칸을 채우지 않습니다. evidence에 없는 수치는 참이 아닙니다.

claim verdict는 supported, unsupported, misinterpretation, not_factual 중 하나입니다.
supported는 evidence 문장으로 그 claim의 사실이 직접 확인될 때입니다.
unsupported는 evidence에 없는 사실을 말할 때입니다.
misinterpretation은 관련 문장은 있으나 숫자, 기간, 회사, 연결/별도, 방향을 잘못 읽었을 때입니다.
not_factual은 확인 가능한 사실을 주장하지 않을 때입니다.
scope_issue는 none, period, company, consolidation, filing, unit, mixed 중 하나입니다.

각 citation은 그 evidence 하나가 답변의 사실 주장을 직접 지지하는지만 봅니다.
verdict는 supports 또는 does_not_support입니다.
입력 claim_id와 evidence_id만 사용합니다. 새 id를 만들지 않습니다.
"""

REQUIREMENT_INSTRUCTIONS = """답변이 각 requirement를 말하는지만 판단합니다.
evidence와 정답 문서는 입력에 없습니다. 답변 문장만 봅니다.

verdict는 covered, omitted, abstained 중 하나입니다.
covered는 답변이 그 requirement의 내용을 구체적으로 말할 때입니다.
omitted는 그 requirement를 다루지 않을 때입니다.
abstained는 그 내용을 확인할 수 없다고 말할 때입니다.
입력 requirement_id마다 item을 하나씩 반환합니다.
"""


class FaithItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    verdict: Literal["supported", "unsupported", "misinterpretation", "not_factual"]
    scope_issue: Literal["none", "period", "company", "consolidation", "filing", "unit", "mixed"]
    reason: str


class CitationItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    verdict: Literal["supports", "does_not_support"]
    reason: str


class FaithOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[FaithItem]
    citations: list[CitationItem] = Field(default_factory=list)


class RequirementItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requirement_id: str
    verdict: Literal["covered", "omitted", "abstained"]
    reason: str


class RequirementOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[RequirementItem]


def needs_judge(case: dict[str, Any]) -> bool:
    return bool(case.get("requirements"))


def judge_case(
    case: dict[str, Any],
    record: dict[str, Any],
    client: StructuredClient,
) -> dict[str, Any]:
    final = record.get("final_answer") if isinstance(record.get("final_answer"), dict) else {}
    claims = split_claims(str(final.get("text") or ""))
    result: dict[str, Any] = {
        "query_id": case["query_id"],
        "prompt_version": JUDGE_PROMPT_VERSION,
        "faithfulness": [],
        "citations": [],
        "requirements": [],
        "usage": [],
        "status": "ok",
    }
    if claims:
        payload = faith_payload(
            case["question"], final, claims, record.get("generation_evidence") or []
        )
        parsed, usage, latency_ms = client.complete(
            instructions=FAITH_INSTRUCTIONS,
            payload=payload,
            schema=FaithOutput,
        )
        result["faithfulness"] = _known_items(
            parsed["items"], {claim["claim_id"] for claim in claims}, "claim_id"
        )
        cited = {
            str(item.get("evidence_id"))
            for item in (final.get("citations") or [])
            if item.get("evidence_id")
        }
        result["citations"] = _known_items(parsed["citations"], cited, "evidence_id")
        result["usage"].append({"stage": "faithfulness", "usage": usage, "latency_ms": latency_ms})
    if case.get("requirements"):
        payload = requirement_payload(case["question"], final, case["requirements"])
        parsed, usage, latency_ms = client.complete(
            instructions=REQUIREMENT_INSTRUCTIONS,
            payload=payload,
            schema=RequirementOutput,
        )
        wanted = {item["requirement_id"] for item in case["requirements"]}
        result["requirements"] = _known_items(parsed["items"], wanted, "requirement_id")
        result["usage"].append({"stage": "requirements", "usage": usage, "latency_ms": latency_ms})
    return result


def faith_payload(
    question: str,
    final: dict[str, Any],
    claims: list[dict[str, str]],
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    cited = [
        str(item.get("evidence_id"))
        for item in final.get("citations") or []
        if item.get("evidence_id")
    ]
    ordered = sorted(
        evidence,
        key=lambda item: (
            str(item.get("evidence_id")) not in set(cited),
            str(item.get("evidence_id")),
        ),
    )
    rows = []
    for item in ordered[:_MAX_EVIDENCE]:
        text = str(item.get("text") or "")
        rows.append(
            {
                "evidence_id": item.get("evidence_id"),
                "report_name": item.get("report_name"),
                "filing_period": item.get("filing_period"),
                "text": text[:_EVIDENCE_CHARS],
                "truncated": len(text) > _EVIDENCE_CHARS,
            }
        )
    payload = {
        "question": question,
        "abstain": bool(final.get("abstain")),
        "limitations": list(final.get("limitations") or []),
        "claims": claims,
        "citations": [{"evidence_id": evidence_id} for evidence_id in cited],
        "evidence": rows,
    }
    _reject(payload)
    return payload


def requirement_payload(
    question: str,
    final: dict[str, Any],
    requirements: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = {
        "question": question,
        "answer": str(final.get("text") or ""),
        "limitations": list(final.get("limitations") or []),
        "requirements": [
            {"requirement_id": item["requirement_id"], "text": item["text"]}
            for item in requirements
        ],
    }
    _reject(payload)
    return payload


def _known_items(items: list[dict[str, Any]], allowed: set[str], key: str) -> list[dict[str, Any]]:
    kept = []
    seen: set[str] = set()
    for item in items:
        token = str(item.get(key) or "")
        if token in allowed and token not in seen:
            kept.append(item)
            seen.add(token)
    return kept


def _reject(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in _FORBIDDEN or str(key).startswith("gold"):
                raise ValueError(f"refusing to send {key} to the A0 judge")
            _reject(item)
    elif isinstance(value, list):
        for item in value:
            _reject(item)

"""Corpus-first question generation and atomic decomposition."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import Any

from disclosure_agent.errors import UsageError
from disclosure_agent.evalset_v2 import PROMPT_VERSION
from disclosure_agent.evalset_v2.corpus_sample import body_text
from disclosure_agent.evalset_v2.llm import CallLog, StructuredClient, input_hash
from disclosure_agent.evalset_v2.prompts import (
    CAUSAL_INSTRUCTIONS,
    DECOMPOSE_INSTRUCTIONS,
    GENERATE_INSTRUCTIONS,
    PARTIAL_INSTRUCTIONS,
    PERIOD_INSTRUCTIONS,
)
from disclosure_agent.evalset_v2.schemas import (
    BareQuestion,
    Decomposition,
    GeneratedQuestion,
    PartialQuestion,
)

COMPOUND_QUERY_TYPES = frozenset(
    {
        "causal_explanation",
        "time_series_comparison",
        "cross_filing_comparison",
        "multi_evidence",
        "calculation",
        "composition_ratio",
    }
)
_TAG_ALLOW = frozenset(
    {
        "period_sensitive",
        "terminology_mismatch",
        "unit_risk",
        "consolidated_vs_separate_risk",
        "abbreviation",
        "synonym",
    }
)
_PERIOD_RE = re.compile(r"20\d{2}|전년|동기|반기|분기")
_HANGUL_RE = re.compile(r"[가-힣]")


def copied_source_sentence(question: str, evidence: str) -> bool:
    body = body_text(evidence)
    for piece in re.split(r"[\n.。]", body):
        snippet = piece.strip()
        if len(snippet) >= 24 and snippet in question:
            return True
    return False


def period_sensitive(question: str) -> bool:
    return _PERIOD_RE.search(question) is not None


def generation_problem(
    question: str, evidence: str, *, company: str, year: str | None) -> str | None:
    text = question.strip()
    if len(text) < 8 or not _HANGUL_RE.search(text):
        return "question is empty or not Korean"
    if company and company not in text:
        return "question omits the company name"
    if year and year not in text:
        return f"question omits {year}"
    if evidence and copied_source_sentence(text, evidence):
        return "question copies a source sentence"
    return None


def logged_complete(
    log: CallLog,
    client: StructuredClient,
    *,
    stage: str,
    instructions: str,
    payload: dict[str, Any],
    schema: type,
    max_retries: int,
    reasoning_setting: str,
) -> dict[str, Any]:
    key = input_hash(stage, client.model, payload)
    existing = log.get(key)
    if existing is not None:
        return existing
    last_error = "not started"
    for attempt in range(max_retries):
        started_attempts = log.attempts(key)
        try:
            output, usage, latency_ms = client.complete(
                instructions=instructions,
                payload=payload,
                schema=schema,
            )
        except Exception as exc:
            last_error = _safe_error(exc)
            log.append(
                _record(
                    stage=stage,
                    model=client.model,
                    key=key,
                    status="error",
                    output=None,
                    usage=None,
                    latency_ms=None,
                    retry_count=started_attempts,
                    error=last_error,
                    reasoning_setting=reasoning_setting,
                )
            )
            time.sleep(min(2**attempt, 8))
            continue
        record = _record(
            stage=stage,
            model=client.model,
            key=key,
            status="ok",
            output=output,
            usage=usage,
            latency_ms=latency_ms,
            retry_count=started_attempts,
            error=None,
            reasoning_setting=reasoning_setting,
        )
        log.append(record)
        return record
    raise UsageError(f"{stage} failed after {max_retries} attempts: {last_error}")


def _safe_error(exc: BaseException) -> str:
    text = " ".join(str(exc).split())[:160]
    lowered = text.lower()
    if "api_key" in lowered or "sk-" in text or text.startswith("AQ."):
        return type(exc).__name__
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _record(
    *,
    stage: str,
    model: str,
    key: str,
    status: str,
    output: dict[str, Any] | None,
    usage: dict[str, int] | None,
    latency_ms: float | None,
    retry_count: int,
    error: str | None,
    reasoning_setting: str,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "reasoning_setting": reasoning_setting,
        "input_hash": key,
        "status": status,
        "output": output,
        "usage": usage,
        "latency_ms": latency_ms,
        "retry_count": retry_count,
        "error": error,
    }


def generate_slot(
    slot: dict[str, Any],
    log: CallLog,
    client: StructuredClient,
    *,
    max_retries: int,
    reasoning_setting: str,
) -> dict[str, Any]:
    answerability = slot["answerability"]
    if answerability == "fully_answerable":
        return _generate_answerable(
            slot, log, client, max_retries=max_retries, reasoning_setting=reasoning_setting)
    if slot["reason_code"] == "period_out_of_corpus" and not slot["seeds"]:
        return _generate_period(
            slot, log, client, max_retries=max_retries, reasoning_setting=reasoning_setting)
    if slot["reason_code"] == "causal_not_in_source":
        return _generate_causal(
            slot, log, client, max_retries=max_retries, reasoning_setting=reasoning_setting)
    return _generate_partial(
        slot, log, client, max_retries=max_retries, reasoning_setting=reasoning_setting)


def _generate_answerable(
    slot: dict[str, Any],
    log: CallLog,
    client: StructuredClient,
    *,
    max_retries: int,
    reasoning_setting: str,
) -> dict[str, Any]:
    evidence = [_evidence_view(seed, index) for index, seed in enumerate(slot["seeds"], start=1)]
    feedback = None
    output: dict[str, Any] | None = None
    record: dict[str, Any] | None = None
    for _ in range(3):
        payload = {
            "product_purpose": "answer from periodic disclosures with citations",
            "target_user": "individual researching one selected company",
            "company_name": slot["company_name"],
            "stratum_id": slot["stratum_id"],
            "wording_style": slot["wording_style"],
            "allowed_query_types": _allowed(slot),
            "evidence": evidence,
            "feedback": feedback,
        }
        record = logged_complete(
            log,
            client,
            stage="generate",
            instructions=GENERATE_INSTRUCTIONS,
            payload=payload,
            schema=GeneratedQuestion,
            max_retries=max_retries,
            reasoning_setting=reasoning_setting,
        )
        output = record["output"]
        if output["query_type"] not in _allowed(slot):
            feedback = "query_type이 allowed_query_types 밖에 있다."
            continue
        problem = generation_problem(
            output["question"],
            "\n".join(seed["text"] for seed in slot["seeds"]),
            company=slot["company_name"],
            year=None,
        )
        if problem is None:
            break
        feedback = problem
    if output is None or record is None:
        raise UsageError(f"{slot['query_id']} was not generated")
    query_type = output["query_type"]
    repaired = False
    if query_type not in _allowed(slot):
        query_type = slot["assigned_query_type"]
        repaired = True
    problem = generation_problem(
        output["question"],
        "\n".join(seed["text"] for seed in slot["seeds"]),
        company=slot["company_name"],
        year=None,
    )
    return _question_row(
        slot,
        question=output["question"].strip(),
        query_type=query_type,
        model=client.model,
        tags=[tag for tag in output.get("difficulty_tags") or [] if tag in _TAG_ALLOW],
        verbatim_risk=problem == "question copies a source sentence",
        query_type_repaired=repaired,
    )


def _generate_partial(
    slot: dict[str, Any],
    log: CallLog,
    client: StructuredClient,
    *,
    max_retries: int,
    reasoning_setting: str,
) -> dict[str, Any]:
    year = str(slot["absent_year"])
    seed = slot["seeds"][0]
    feedback = None
    output: dict[str, Any] | None = None
    for _ in range(3):
        payload = {
            "company_name": slot["company_name"],
            "absent_year": year,
            "evidence": _evidence_view(seed, 1),
            "feedback": feedback,
        }
        record = logged_complete(
            log,
            client,
            stage="generate",
            instructions=PARTIAL_INSTRUCTIONS,
            payload=payload,
            schema=PartialQuestion,
            max_retries=max_retries,
            reasoning_setting=reasoning_setting,
        )
        output = record["output"]
        problem = generation_problem(
            output["question"],
            seed["text"],
            company=slot["company_name"],
            year=year,
        )
        if problem is None and year in output["unanswerable_requirement"]:
            break
        feedback = problem or f"unanswerable requirement must mention {year}"
    if output is None:
        raise UsageError(f"{slot['query_id']} partial question was not generated")
    row = _question_row(
        slot,
        question=output["question"].strip(),
        query_type="time_series_comparison",
        model=client.model,
        tags=["period_sensitive"],
        verbatim_risk=copied_source_sentence(output["question"], seed["text"]),
        query_type_repaired=False,
    )
    row["draft_requirements"] = [
        {
            "requirement_id": "R1",
            "text": output["answerable_requirement"].strip(),
            "answerability": "fully_answerable",
            "seed_evidence_ids": list(seed["element_ids"]),
            "reason_code": None,
        },
        {
            "requirement_id": "R2",
            "text": output["unanswerable_requirement"].strip(),
            "answerability": "unanswerable",
            "seed_evidence_ids": [],
            "reason_code": "period_out_of_corpus",
        },
    ]
    return row


def _generate_period(
    slot: dict[str, Any],
    log: CallLog,
    client: StructuredClient,
    *,
    max_retries: int,
    reasoning_setting: str,
) -> dict[str, Any]:
    year = str(slot["absent_year"])
    feedback = None
    output: dict[str, Any] | None = None
    for _ in range(3):
        payload = {
            "company_name": slot["company_name"],
            "absent_year": year,
            "ask": "연결 매출액",
            "feedback": feedback,
        }
        record = logged_complete(
            log,
            client,
            stage="generate",
            instructions=PERIOD_INSTRUCTIONS,
            payload=payload,
            schema=BareQuestion,
            max_retries=max_retries,
            reasoning_setting=reasoning_setting,
        )
        output = record["output"]
        problem = generation_problem(
            output["question"], "", company=slot["company_name"], year=year
        )
        if problem is None:
            break
        feedback = problem
    if output is None:
        raise UsageError(f"{slot['query_id']} period question was not generated")
    row = _question_row(
        slot,
        question=output["question"].strip(),
        query_type="single_numeric",
        model=client.model,
        tags=["period_sensitive"],
        verbatim_risk=False,
        query_type_repaired=False,
    )
    row["draft_requirements"] = [
        {
            "requirement_id": "R1",
            "text": output["question"].strip(),
            "answerability": "unanswerable",
            "seed_evidence_ids": [],
            "reason_code": "period_out_of_corpus",
        }
    ]
    return row


def _generate_causal(
    slot: dict[str, Any],
    log: CallLog,
    client: StructuredClient,
    *,
    max_retries: int,
    reasoning_setting: str,
) -> dict[str, Any]:
    seed = slot["seeds"][0]
    feedback = None
    output: dict[str, Any] | None = None
    for _ in range(3):
        payload = {
            "company_name": slot["company_name"],
            "evidence": _evidence_view(seed, 1),
            "feedback": feedback,
        }
        record = logged_complete(
            log,
            client,
            stage="generate",
            instructions=CAUSAL_INSTRUCTIONS,
            payload=payload,
            schema=BareQuestion,
            max_retries=max_retries,
            reasoning_setting=reasoning_setting,
        )
        output = record["output"]
        problem = generation_problem(
            output["question"], seed["text"], company=slot["company_name"], year=None
        )
        if problem is None and re.search(r"왜|이유", output["question"]):
            break
        feedback = problem or "질문은 이유 또는 왜를 포함해야 한다"
    if output is None:
        raise UsageError(f"{slot['query_id']} causal question was not generated")
    row = _question_row(
        slot,
        question=output["question"].strip(),
        query_type="causal_explanation",
        model=client.model,
        tags=[],
        verbatim_risk=copied_source_sentence(output["question"], seed["text"]),
        query_type_repaired=False,
    )
    row["draft_requirements"] = [
        {
            "requirement_id": "R1",
            "text": "표에 나타난 변화가 생긴 원인",
            "answerability": "unanswerable",
            "seed_evidence_ids": [],
            "reason_code": "causal_not_in_source",
        }
    ]
    return row


def decompose_question(
    question: dict[str, Any],
    slot: dict[str, Any],
    log: CallLog,
    client: StructuredClient,
    *,
    max_retries: int,
    reasoning_setting: str,
) -> list[dict[str, Any]]:
    drafted = question.get("draft_requirements")
    if drafted:
        return [_requirement_row(question["query_id"], item) for item in drafted]
    if question["query_type"] not in COMPOUND_QUERY_TYPES:
        element_ids: list[str] = []
        for seed in slot["seeds"]:
            element_ids.extend(seed["element_ids"])
        return [
            _requirement_row(
                question["query_id"],
                {
                    "requirement_id": "R1",
                    "text": question["question"],
                    "answerability": "fully_answerable",
                    "seed_evidence_ids": element_ids,
                    "reason_code": None,
                },
            )
        ]
    labels = [f"E{index}" for index in range(1, len(slot["seeds"]) + 1)]
    payload = {
        "question": question["question"],
        "evidence": [
            {**_evidence_view(seed, index), "label": f"E{index}"}
            for index, seed in enumerate(slot["seeds"], start=1)
        ],
    }
    record = logged_complete(
        log,
        client,
        stage="decompose",
        instructions=DECOMPOSE_INSTRUCTIONS,
        payload=payload,
        schema=Decomposition,
        max_retries=max_retries,
        reasoning_setting=reasoning_setting,
    )
    rows = _normalize_decomposition(question, slot, record["output"], labels)
    if not rows:
        raise UsageError(f"{question['query_id']} decomposition was empty")
    return rows


def _normalize_decomposition(
    question: dict[str, Any],
    slot: dict[str, Any],
    output: dict[str, Any],
    labels: list[str],
) -> list[dict[str, Any]]:
    label_to_ids = {
        f"E{index}": list(seed["element_ids"])
        for index, seed in enumerate(slot["seeds"], start=1)
    }
    rows = []
    seen: set[str] = set()
    for index, item in enumerate(output.get("requirements") or [], start=1):
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        requirement_id = f"R{index}"
        if requirement_id in seen:
            continue
        seen.add(requirement_id)
        used = [label for label in item.get("evidence_labels") or [] if label in labels]
        element_ids: list[str] = []
        for label in used or labels:
            element_ids.extend(label_to_ids.get(label) or [])
        rows.append(
            _requirement_row(
                question["query_id"],
                {
                    "requirement_id": requirement_id,
                    "text": text,
                    "answerability": "fully_answerable",
                    "seed_evidence_ids": element_ids,
                    "reason_code": None,
                },
            )
        )
        if len(rows) == 4:
            break
    return rows


def _requirement_row(query_id: str, item: dict[str, Any]) -> dict[str, Any]:
    return {
        "query_id": query_id,
        "requirement_id": item["requirement_id"],
        "text": item["text"],
        "answerability": item["answerability"],
        "seed_evidence_ids": list(item.get("seed_evidence_ids") or []),
        "reason_code": item.get("reason_code"),
    }


def _question_row(
    slot: dict[str, Any],
    *,
    question: str,
    query_type: str,
    model: str,
    tags: list[str],
    verbatim_risk: bool,
    query_type_repaired: bool,
) -> dict[str, Any]:
    seed_ids: list[str] = []
    unit_ids: list[str] = []
    for seed in slot["seeds"]:
        seed_ids.extend(seed["element_ids"])
        unit_ids.append(seed["retrieval_unit_id"])
    computed = list(tags)
    if period_sensitive(question) and "period_sensitive" not in computed:
        computed.append("period_sensitive")
    return {
        "query_id": slot["query_id"],
        "corp_code": slot["corp_code"],
        "company_name": slot["company_name"],
        "question": question,
        "query_type": query_type,
        "answerability": slot["answerability"],
        "wording_style": slot["wording_style"],
        "evidence_form": slot["evidence_form"],
        "stratum_id": slot["stratum_id"],
        "split": slot["split"],
        "difficulty_tags": computed,
        "period_sensitive": period_sensitive(question),
        "seed_evidence_ids": seed_ids,
        "seed_unit_ids": unit_ids,
        "reason_code": slot.get("reason_code"),
        "generation_model": model,
        "prompt_version": PROMPT_VERSION,
        "verbatim_risk": verbatim_risk,
        "query_type_repaired": query_type_repaired,
    }


def _evidence_view(seed: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "label": f"E{index}",
        "document_name": seed["document_name"],
        "filing_period": seed["filing_period"],
        "section_path": seed["section_path"],
        "text": seed["text"][:2000],
    }


def _allowed(slot: dict[str, Any]) -> list[str]:
    return [slot["assigned_query_type"]]


def map_slots(
    slots: list[dict[str, Any]],
    worker: Callable[[dict[str, Any]], Any],
    *,
    workers: int,
) -> list[Any]:
    if workers <= 1 or len(slots) <= 1:
        return [worker(slot) for slot in slots]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(worker, slots))

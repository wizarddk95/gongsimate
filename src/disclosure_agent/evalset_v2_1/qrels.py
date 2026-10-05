"""Rebuild v2 judgments as retrieval-unit qrels.

A judge call saw one retrieval unit's text. That call becomes one relevance
row per atomic requirement. Source elements stay on a provenance row and do
not receive their own relevance.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from disclosure_agent.errors import UsageError

RELEVANCE_GRADES = frozenset({0, 1, 2})


def build_unit_qrels(
    judgments: list[dict[str, Any]],
    questions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return qrel rows and provenance rows.

    Missing or unresolved judgments are omitted. They are not stored as
    relevance 0. A stored row has ``judgment_status`` ``judged``.
    """
    best: dict[tuple[str, str, str], dict[str, Any]] = {}
    elements: dict[tuple[str, str], list[str]] = {}
    for item in judgments:
        final = item.get("final")
        if not isinstance(final, dict):
            continue
        query_id = str(item["query_id"])
        unit_id = str(item["retrieval_unit_id"])
        element_ids = _element_ids(item.get("element_ids"))
        previous_elements = elements.get((query_id, unit_id))
        if previous_elements is None:
            elements[(query_id, unit_id)] = element_ids
        elif previous_elements != element_ids:
            raise UsageError(f"{query_id} {unit_id} has conflicting source elements")
        models = item.get("models") or {}
        for requirement in final["requirement_judgments"]:
            relevance = int(requirement["relevance"])
            if relevance not in RELEVANCE_GRADES:
                raise UsageError("judged relevance must be 0, 1, or 2")
            key = (query_id, str(requirement["requirement_id"]), unit_id)
            row = {
                "query_id": query_id,
                "requirement_id": key[1],
                "retrieval_unit_id": unit_id,
                "relevance": relevance,
                "preferred": False,
                "judgment_status": "judged",
                "document_name": item.get("document_name"),
                "filing_period": item.get("filing_period") or "",
                "final_source": item.get("final_source"),
                "adjudicated": bool(item.get("adjudicated")),
                "judge_a_model": models.get("judge_a"),
                "judge_b_model": models.get("judge_b"),
                "adjudicator_model": models.get("adjudicator"),
            }
            previous = best.get(key)
            if previous is None or relevance > int(previous["relevance"]):
                best[key] = row
    rows = list(best.values())
    _assign_preferred(rows, questions)
    rows.sort(key=_qrel_key)
    provenance = [
        {
            "query_id": query_id,
            "retrieval_unit_id": unit_id,
            "source_element_ids": element_ids,
        }
        for (query_id, unit_id), element_ids in sorted(elements.items())
    ]
    return rows, provenance


def _element_ids(value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        raise UsageError("a judged retrieval unit needs at least one source element")
    return sorted({str(item) for item in value})


def _qrel_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (row["query_id"], row["requirement_id"], row["retrieval_unit_id"])


def _assign_preferred(rows: list[dict[str, Any]], questions: list[dict[str, Any]]) -> None:
    """Preferred is a property of a retrieval unit, not of each source element.

    A seed unit that is still judged relevant stays preferred. Non-preferred
    units can still be relevant. If no seed unit remains relevant, exactly one
    judged-relevant unit is marked: a 사업보고서 when one exists, then the
    latest filing period, then the unit id. Preferred is not a judge output.
    """
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_query[row["query_id"]].append(row)
    questions_by_id = {str(item["query_id"]): item for item in questions}
    for query_id, group in by_query.items():
        question = questions_by_id.get(query_id) or {}
        grades: dict[str, int] = {}
        meta: dict[str, dict[str, Any]] = {}
        for row in group:
            unit_id = row["retrieval_unit_id"]
            grades[unit_id] = max(grades.get(unit_id, 0), int(row["relevance"]))
            meta[unit_id] = row
        positive = {unit_id for unit_id, grade in grades.items() if grade >= 1}
        seeds = {str(unit_id) for unit_id in question.get("seed_unit_ids") or []}
        chosen = positive & seeds
        if not chosen and positive:
            annual = [
                unit_id
                for unit_id in positive
                if meta[unit_id].get("document_name") == "사업보고서"
            ]
            pool = annual or list(positive)
            chosen = {
                max(
                    pool,
                    key=lambda unit_id: (
                        str(meta[unit_id].get("filing_period") or ""),
                        unit_id,
                    ),
                )
            }
        for row in group:
            row["preferred"] = row["retrieval_unit_id"] in chosen and int(row["relevance"]) >= 1

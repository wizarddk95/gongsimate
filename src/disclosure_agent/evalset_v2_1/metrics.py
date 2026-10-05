"""Retrieval-unit metrics for v2.1.

Recall, Hit, MRR, and NDCG count a retrieval unit once. Source elements are
not relevance rows. Unjudged units stay unlabeled and are not relevance 0.
Unanswerable questions stay out of the retrieval headline.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from disclosure_agent.errors import UsageError

HEADLINE_KS = (1, 5, 10)
MRR_CUTOFF = 20
SMALL_N = 5


def qrel_state(relevance: int | None) -> str:
    if relevance is None:
        return "unjudged"
    if relevance == 0:
        return "irrelevant"
    if relevance in {1, 2}:
        return "relevant"
    raise UsageError(f"relevance {relevance} is not 0, 1, 2, or null")


def gain(relevance: int) -> float:
    """Graded gain. relevance 2 -> 3, relevance 1 -> 1, otherwise 0."""
    if relevance <= 0:
        return 0.0
    return float((2**relevance) - 1)


def score_question(
    question: dict[str, Any],
    qrels: list[dict[str, Any]],
    requirements: list[dict[str, Any]],
    hits: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score one raw ranking by retrieval_unit_id.

    A hit is judged only when that unit has its own qrel. Sharing a source
    element with a different judged unit does not copy the grade.
    """
    grades = _unit_grades(qrels)
    preferred = {
        row["retrieval_unit_id"]
        for row in qrels
        if row.get("preferred") and int(row["relevance"]) >= 1
    }
    annotated = [_annotate(hit, grades, preferred) for hit in hits]
    positive = [grade for grade in grades.values() if grade >= 1]
    at_k = {}
    for k in HEADLINE_KS:
        window = annotated[:k]
        judged = [hit for hit in window if hit["judged"]]
        relevant = [
            hit for hit in window if hit["relevance"] is not None and hit["relevance"] >= 1
        ]
        at_k[str(k)] = {
            "hit": bool(relevant),
            "preferred_hit": any(hit["preferred"] for hit in relevant),
            "judged_coverage": _ratio(len(judged), k),
            "unjudged_count": sum(1 for hit in window if not hit["judged"]),
            "ndcg_condensed": _ndcg(annotated, positive, k, condense=True),
            "ndcg_unjudged_as_zero": _ndcg(annotated, positive, k, condense=False),
        }
    found10, total = _recall_counts(annotated[:10], grades)
    found5, _ = _recall_counts(annotated[:5], grades)
    covered, req_total, all_covered = _requirement_coverage(
        question["query_id"], requirements, qrels, annotated[:10]
    )
    return {
        "query_id": question["query_id"],
        "company_name": question.get("company_name"),
        "query_type": question.get("query_type"),
        "evidence_form": question.get("evidence_form"),
        "wording_style": question.get("wording_style"),
        "answerability": question.get("answerability"),
        "split": question.get("split"),
        "period_sensitive": bool(question.get("period_sensitive")),
        "stratum_id": question.get("stratum_id"),
        "positive_qrel_count": len(positive),
        "at_k": at_k,
        "recall@5": _ratio(found5, total),
        "recall@10": _ratio(found10, total),
        "mrr": _mrr(annotated),
        "requirement_recall@10": _ratio(covered, req_total),
        "all_requirements_covered@10": all_covered,
        "requirement_total": req_total,
        "requirement_covered@10": covered,
        "multi_evidence": req_total > 1,
        "judged_coverage@20": _ratio(sum(1 for hit in annotated[:20] if hit["judged"]), 20),
        "unjudged_count@20": sum(1 for hit in annotated[:20] if not hit["judged"]),
        "hits": [
            {
                "rank": hit["rank"],
                "retrieval_unit_id": hit["retrieval_unit_id"],
                "judged": hit["judged"],
                "relevance": hit["relevance"],
                "preferred": hit["preferred"],
                "source_element_ids": hit["source_element_ids"],
            }
            for hit in annotated
        ],
    }


def aggregate_headline(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Fully answerable rows that have at least one judged relevant unit."""
    excluded = [row["query_id"] for row in rows if row["positive_qrel_count"] == 0]
    kept = [row for row in rows if row["positive_qrel_count"] > 0]
    block = _aggregate(kept)
    block["excluded_no_positive_qrel"] = excluded
    block["excluded_count"] = len(excluded)
    return block


def filter_headline(
    questions: list[dict[str, Any]], scored: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    allowed = {
        item["query_id"]
        for item in questions
        if item.get("answerability") == "fully_answerable" and item.get("split") == "test"
    }
    return [row for row in scored if row["query_id"] in allowed]


def sufficiency(scored: list[dict[str, Any]], qrels: list[dict[str, Any]]) -> dict[str, Any]:
    """Unanswerable questions are counted here and not in the retrieval headline."""
    ids = {row["query_id"] for row in scored if row.get("answerability") == "unanswerable"}
    direct = set()
    supporting = set()
    for row in qrels:
        if row["query_id"] not in ids:
            continue
        if int(row["relevance"]) >= 2:
            direct.add(row["query_id"])
        elif int(row["relevance"]) == 1:
            supporting.add(row["query_id"])
    return {
        "question_count": len(ids),
        "direct_evidence_leak": sorted(direct),
        "supporting_only": sorted(supporting - direct),
        "no_supporting_judgment": sorted(ids - direct - supporting),
        "in_retrieval_headline": False,
    }


def breakdown(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(field))].append(row)
    return {key: _aggregate(value) for key, value in sorted(groups.items())}


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean_flag(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"numerator": 0, "denominator": 0, "value": None, "small_n": True}
        return {
            "numerator": None,
            "denominator": len(values),
            "value": sum(values) / len(values),
            "small_n": len(values) < SMALL_N,
        }

    at_k = {}
    for k in HEADLINE_KS:
        key = str(k)
        hits = sum(1 for row in rows if row["at_k"][key]["hit"])
        preferred = sum(1 for row in rows if row["at_k"][key]["preferred_hit"])
        at_k[key] = {
            "hit": _count_block(hits, len(rows)),
            "preferred_hit": _count_block(preferred, len(rows)),
            "judged_coverage": mean_flag(
                [row["at_k"][key]["judged_coverage"]["value"] or 0 for row in rows]
            ),
            "unjudged_count_mean": mean_flag(
                [row["at_k"][key]["unjudged_count"] for row in rows]
            ),
            "ndcg_condensed": mean_flag([row["at_k"][key]["ndcg_condensed"] for row in rows]),
            "ndcg_unjudged_as_zero": mean_flag(
                [row["at_k"][key]["ndcg_unjudged_as_zero"] for row in rows]
            ),
        }
        at_k[key]["preferred_hit"]["role"] = "diagnostic"
        at_k[key]["ndcg_unjudged_as_zero"]["role"] = "sensitivity_not_headline"
    micro_found = sum(row["recall@10"]["numerator"] for row in rows)
    micro_total = sum(row["recall@10"]["denominator"] for row in rows)
    micro_found5 = sum(row["recall@5"]["numerator"] for row in rows)
    req_found = sum(row["requirement_covered@10"] for row in rows)
    req_total = sum(row["requirement_total"] for row in rows)
    all_req = [row for row in rows if row["requirement_total"] > 0]
    all_hit = sum(1 for row in all_req if row["all_requirements_covered@10"])
    return {
        "question_count": len(rows),
        "small_n": len(rows) < SMALL_N,
        "metric_policy": {
            "unit": "retrieval_unit_id",
            "hit": "raw top-k contains a unit whose own judged relevance is >= 1",
            "unjudged": "not a hit and not stored as 0",
            "recall_denominator": "unique retrieval units with judged relevance >= 1",
            "ndcg_headline": "ndcg_condensed",
            "gain": "2^relevance - 1",
            "mrr_cutoff": MRR_CUTOFF,
        },
        "mrr": mean_flag([row["mrr"] for row in rows]),
        "recall@5_macro": mean_flag(
            [row["recall@5"]["value"] for row in rows if row["recall@5"]["value"] is not None]
        ),
        "recall@10_macro": mean_flag(
            [row["recall@10"]["value"] for row in rows if row["recall@10"]["value"] is not None]
        ),
        "recall@5_micro": _ratio(micro_found5, micro_total),
        "recall@10_micro": _ratio(micro_found, micro_total),
        "requirement_recall@10": _ratio(req_found, req_total),
        "all_requirements_covered@10": _count_block(all_hit, len(all_req)),
        "judged_coverage@20": mean_flag(
            [row["judged_coverage@20"]["value"] or 0 for row in rows]
        ),
        "unjudged_count@20_mean": mean_flag([row["unjudged_count@20"] for row in rows]),
        "at_k": at_k,
    }


def _unit_grades(qrels: list[dict[str, Any]]) -> dict[str, int]:
    grades: dict[str, int] = {}
    for row in qrels:
        if row.get("judgment_status") not in (None, "judged"):
            continue
        if row.get("relevance") is None:
            continue
        unit_id = str(row["retrieval_unit_id"])
        grades[unit_id] = max(grades.get(unit_id, 0), int(row["relevance"]))
    return grades


def _annotate(
    hit: dict[str, Any],
    grades: dict[str, int],
    preferred: set[str],
) -> dict[str, Any]:
    if "retrieval_unit_id" not in hit:
        raise UsageError("a retrieval hit needs retrieval_unit_id")
    unit_id = str(hit["retrieval_unit_id"])
    judged = unit_id in grades
    relevance = grades[unit_id] if judged else None
    return {
        "rank": int(hit["rank"]),
        "retrieval_unit_id": unit_id,
        "source_element_ids": [str(item) for item in hit.get("source_element_ids") or []],
        "judged": judged,
        "relevance": relevance,
        "preferred": unit_id in preferred and relevance is not None and relevance >= 1,
    }


def _ndcg(
    hits: list[dict[str, Any]],
    positive: list[int],
    k: int,
    *,
    condense: bool,
) -> float:
    """NDCG@k with gain 2^rel-1.

    Both variants award a retrieval unit's gain once, on its first occurrence.
    A later copy of the same unit keeps the rank and contributes 0.

    Condensed walks past unjudged units without giving them a rank, and stops
    after k judged units. Unjudged-as-zero uses raw ranks 1..k and writes 0
    for an unjudged unit, so later units stay at their raw discount. When
    every raw rank in 1..k is judged, the two lists are the same.
    """
    gains = _gains(hits, k, condense=condense)
    idcg = _dcg(sorted((gain(grade) for grade in positive), reverse=True)[:k])
    if idcg == 0:
        return 0.0
    return _dcg(gains) / idcg


def _gains(hits: list[dict[str, Any]], k: int, *, condense: bool) -> list[float]:
    gains: list[float] = []
    seen: set[str] = set()
    for hit in hits:
        if len(gains) >= k:
            break
        if not hit["judged"]:
            if condense:
                continue
            gains.append(0.0)
            continue
        unit_id = str(hit["retrieval_unit_id"])
        if unit_id in seen:
            gains.append(0.0)
        else:
            seen.add(unit_id)
            relevance = hit["relevance"]
            gains.append(0.0 if relevance is None else gain(int(relevance)))
    return gains


def _dcg(gains: list[float]) -> float:
    total = 0.0
    for index, value in enumerate(gains):
        total += value / math.log2(index + 2)
    return total


def _mrr(hits: list[dict[str, Any]]) -> float:
    for hit in hits[:MRR_CUTOFF]:
        if hit["relevance"] is not None and hit["relevance"] >= 1:
            return 1.0 / int(hit["rank"])
    return 0.0


def _recall_counts(
    hits: list[dict[str, Any]], grades: dict[str, int]
) -> tuple[int, int]:
    positive = {unit_id for unit_id, grade in grades.items() if grade >= 1}
    found: set[str] = set()
    for hit in hits:
        if hit["judged"] and hit["retrieval_unit_id"] in positive:
            found.add(hit["retrieval_unit_id"])
    return len(found), len(positive)


def _requirement_coverage(
    query_id: str,
    requirements: list[dict[str, Any]],
    qrels: list[dict[str, Any]],
    hits: list[dict[str, Any]],
) -> tuple[int, int, bool | None]:
    """One requirement is covered when one of its own relevant units is in the window.

    The denominator is the requirement count. It does not grow with source elements.
    """
    needed = [
        item
        for item in requirements
        if item["query_id"] == query_id and item["answerability"] == "fully_answerable"
    ]
    if not needed:
        return 0, 0, None
    top = {hit["retrieval_unit_id"] for hit in hits}
    covered = 0
    for requirement in needed:
        ok = any(
            row["query_id"] == query_id
            and row["requirement_id"] == requirement["requirement_id"]
            and row["retrieval_unit_id"] in top
            and int(row["relevance"]) >= 1
            for row in qrels
        )
        covered += int(ok)
    return covered, len(needed), covered == len(needed)


def _ratio(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": None if denominator == 0 else numerator / denominator,
    }


def _count_block(numerator: int, denominator: int) -> dict[str, Any]:
    block = _ratio(numerator, denominator)
    block["small_n"] = denominator < SMALL_N
    return block

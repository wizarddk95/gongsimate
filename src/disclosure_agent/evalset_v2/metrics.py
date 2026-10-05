"""Metrics for incomplete, LLM-judged qrels.

Unjudged elements stay unlabeled. They are not stored as relevance 0.
Unanswerable questions are not folded into Hit, Recall, MRR, or NDCG.
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
    if relevance <= 0:
        return 0.0
    return float((2**relevance) - 1)


def build_qrels(
    judgments: list[dict[str, Any]],
    questions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """One row per judged requirement and element. Missing finals stay out of the file."""
    best: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for item in judgments:
        final = item.get("final")
        if not isinstance(final, dict):
            continue
        for element_id in item.get("element_ids") or []:
            for requirement in final["requirement_judgments"]:
                relevance = int(requirement["relevance"])
                if relevance not in {0, 1, 2}:
                    raise UsageError("judged relevance must be 0, 1, or 2")
                key = (
                    str(item["query_id"]),
                    str(requirement["requirement_id"]),
                    str(element_id),
                    str(item["retrieval_unit_id"]),
                )
                row = {
                    "query_id": key[0],
                    "requirement_id": key[1],
                    "element_id": key[2],
                    "retrieval_unit_id": key[3],
                    "relevance": relevance,
                    "preferred": False,
                    "document_name": item.get("document_name"),
                    "filing_period": item.get("filing_period") or "",
                    "final_source": item.get("final_source"),
                    "adjudicated": bool(item.get("adjudicated")),
                }
                previous = best.get(key)
                if previous is None or relevance > int(previous["relevance"]):
                    best[key] = row
    rows = list(best.values())
    _assign_preferred(rows, questions)
    rows.sort(
        key=lambda item: (
            item["query_id"],
            item["requirement_id"],
            item["element_id"],
            item["retrieval_unit_id"],
        )
    )
    return rows


def _assign_preferred(rows: list[dict[str, Any]], questions: list[dict[str, Any]]) -> None:
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_query[row["query_id"]].append(row)
    questions_by_id = {item["query_id"]: item for item in questions}
    for query_id, group in by_query.items():
        question = questions_by_id.get(query_id) or {}
        grades: dict[str, int] = {}
        meta: dict[str, dict[str, Any]] = {}
        for row in group:
            element_id = row["element_id"]
            grades[element_id] = max(grades.get(element_id, 0), int(row["relevance"]))
            meta[element_id] = row
        positive = {element_id for element_id, grade in grades.items() if grade >= 1}
        seeds = set(question.get("seed_evidence_ids") or [])
        chosen = positive & seeds
        if not chosen and positive:
            annual = [
                element_id
                for element_id in positive
                if meta[element_id].get("document_name") == "사업보고서"
            ]
            pool = annual or list(positive)
            chosen = {
                max(
                    pool, key=lambda element_id: (
                        str(meta[element_id].get("filing_period") or ""), element_id))
            }
        for row in group:
            row["preferred"] = row["element_id"] in chosen and grades[row["element_id"]] >= 1


def score_question(
    question: dict[str, Any],
    qrels: list[dict[str, Any]],
    requirements: list[dict[str, Any]],
    hits: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score one raw ranking. Unjudged units are not given relevance 0."""
    element_grade = _element_grades(qrels)
    element_preferred = {
        row["element_id"]
        for row in qrels
        if row.get("preferred") and int(row["relevance"]) >= 1
    }
    annotated = [_annotate(hit, element_grade, element_preferred) for hit in hits]
    unit_positive = _unit_positive_grades(qrels)
    at_k = {}
    for k in HEADLINE_KS:
        window = annotated[:k]
        judged = [hit for hit in window if hit["judged"]]
        relevant = [hit for hit in window if hit["relevance"] is not None and hit["relevance"] >= 1]
        at_k[str(k)] = {
            "hit": bool(relevant),
            "preferred_hit": any(hit["preferred"] for hit in relevant),
            "judged_coverage": _ratio(len(judged), k),
            "unjudged_count": sum(1 for hit in window if not hit["judged"]),
            "ndcg_condensed": _ndcg(_condense(annotated), unit_positive, k),
            "ndcg_unjudged_as_zero": _ndcg_unjudged_as_zero(window, unit_positive, k),
        }
    found, total = _recall_counts(annotated[:10], element_grade)
    found5, _ = _recall_counts(annotated[:5], element_grade)
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
        "positive_qrel_count": len(_positive_grades(element_grade)),
        "at_k": at_k,
        "recall@5": _ratio(found5, total),
        "recall@10": _ratio(found, total),
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
                "judged": hit["judged"],
                "relevance": hit["relevance"],
                "preferred": hit["preferred"],
                "source_element_ids": hit["source_element_ids"],
            }
            for hit in annotated
        ],
    }


def aggregate_headline(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Fully answerable test questions that have at least one judged relevant element."""
    excluded = [row["query_id"] for row in rows if row["positive_qrel_count"] == 0]
    kept = [row for row in rows if row["positive_qrel_count"] > 0]
    block = _aggregate(kept)
    block["excluded_no_positive_qrel"] = excluded
    block["excluded_count"] = len(excluded)
    return block


def filter_headline(
    questions: list[dict[str, Any]], scored: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
                [row["at_k"][key]["judged_coverage"]["value"] or 0 for row in rows]),
            "unjudged_count_mean": mean_flag([row["at_k"][key]["unjudged_count"] for row in rows]),
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
            "hit": "raw top-k contains judged relevance >= 1",
            "unjudged": "not a hit and not stored as 0",
            "ndcg_headline": "ndcg_condensed",
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


def breakdown(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(field))].append(row)
    return {key: _aggregate(value) for key, value in sorted(groups.items())}


def _element_grades(qrels: list[dict[str, Any]]) -> dict[str, int]:
    grades: dict[str, int] = {}
    for row in qrels:
        element_id = row["element_id"]
        grades[element_id] = max(grades.get(element_id, 0), int(row["relevance"]))
    return grades


def _positive_grades(grades: dict[str, int]) -> list[int]:
    return [grade for grade in grades.values() if grade >= 1]


def _unit_positive_grades(qrels: list[dict[str, Any]]) -> list[int]:
    """Ideal gain uses one grade per judged unit, the max of its element labels."""
    grades: dict[str, int] = {}
    for row in qrels:
        unit_id = str(row["retrieval_unit_id"])
        grades[unit_id] = max(grades.get(unit_id, 0), int(row["relevance"]))
    return [grade for grade in grades.values() if grade >= 1]


def _annotate(
    hit: dict[str, Any],
    grades: dict[str, int],
    preferred: set[str],
) -> dict[str, Any]:
    element_ids = [str(item) for item in hit.get("source_element_ids") or []]
    known = [element_id for element_id in element_ids if element_id in grades]
    judged = bool(element_ids) and len(known) == len(element_ids)
    relevance = max((grades[element_id] for element_id in known), default=None) if judged else None
    return {
        "rank": int(hit["rank"]),
        "source_element_ids": element_ids,
        "judged": judged,
        "relevance": relevance,
        "preferred": bool(preferred & set(known)) and relevance is not None and relevance >= 1,
    }


def _condense(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [hit for hit in hits if hit["judged"]]


def _ndcg(hits: list[dict[str, Any]], positive: list[int], k: int) -> float:
    gains = []
    seen: set[str] = set()
    for hit in hits:
        if len(gains) >= k:
            break
        fresh = [element_id for element_id in hit["source_element_ids"] if element_id not in seen]
        seen.update(hit["source_element_ids"])
        relevance = 0 if not fresh else int(hit["relevance"] or 0)
        gains.append(gain(relevance))
    ideal = sorted((gain(grade) for grade in positive), reverse=True)[:k]
    idcg = _dcg(ideal)
    if idcg == 0:
        return 0.0
    return _dcg(gains[:k]) / idcg


def _ndcg_unjudged_as_zero(hits: list[dict[str, Any]], positive: list[int], k: int) -> float:
    """Sensitivity only. The headline does not treat unjudged as zero."""
    gains = []
    for hit in hits[:k]:
        gains.append(0.0 if hit["relevance"] is None else gain(int(hit["relevance"])))
    ideal = sorted((gain(grade) for grade in positive), reverse=True)[:k]
    idcg = _dcg(ideal)
    if idcg == 0:
        return 0.0
    return _dcg(gains) / idcg


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


def _recall_counts(hits: list[dict[str, Any]], grades: dict[str, int]) -> tuple[int, int]:
    positive = {element_id for element_id, grade in grades.items() if grade >= 1}
    found: set[str] = set()
    for hit in hits:
        if not hit["judged"]:
            continue
        found.update(
            element_id for element_id in hit["source_element_ids"] if element_id in positive)
    return len(found), len(positive)


def _requirement_coverage(
    query_id: str,
    requirements: list[dict[str, Any]],
    qrels: list[dict[str, Any]],
    hits: list[dict[str, Any]],
) -> tuple[int, int, bool | None]:
    needed = [
        item
        for item in requirements
        if item["query_id"] == query_id and item["answerability"] == "fully_answerable"
    ]
    if not needed:
        return 0, 0, None
    top: set[str] = set()
    for hit in hits:
        top.update(hit["source_element_ids"])
    covered = 0
    for requirement in needed:
        ok = any(
            row["query_id"] == query_id
            and row["requirement_id"] == requirement["requirement_id"]
            and row["element_id"] in top
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

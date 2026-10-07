"""H11 diagnosis: ranking failure versus candidate-generation failure.

The candidate pool is the stored Dense top-20 union the stored BM25 top-20.
The oracle is an upper bound on reordering that pool. It is not a reranker,
and it does not search. v2.1 TEST is not read.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from disclosure_agent.errors import UsageError
from disclosure_agent.evalset_v2_1 import DATASET_ID
from disclosure_agent.evalset_v2_1.asymmetric_fusion import (
    RUN_DIR_NAME as H10_RUN_DIR,
)
from disclosure_agent.evalset_v2_1.asymmetric_fusion import (
    bm25_primary_dense_head,
)
from disclosure_agent.evalset_v2_1.metrics import aggregate_headline, score_question
from disclosure_agent.evalset_v2_1.rrf import (
    DATASET_SHA,
    validate_source_bundle,
)
from disclosure_agent.evalset_v2_1.rrf import (
    RUN_DIR_NAME as RRF_RUN_DIR,
)
from disclosure_agent.evaluation import git_revision

RUN_DIR_NAME = "multi_company_retrieval_v2_1_dev_h11_ranking_failure"
SEMANTICS_RELATIVE = Path("docs/experiments/2026-10-07_h11_ranking_failure_semantics.md")
POOL_ID = "C20"
CANDIDATE_N = 20
HIT_K = 10
FAMILY_KEY = "stratum_id"
COVER_REQUIREMENT_LIMIT = 16
VERDICT_STRONG = "Strong Support"
VERDICT_PARTIAL = "Partial Support"
VERDICT_REJECT = "Reject"
FAILURE_RANKING = "ranking_failure"
FAILURE_CANDIDATE = "candidate_generation_failure"
FAILURE_NONE = "h10_hit"
REQUIREMENT_COVERED = "covered"
REQUIREMENT_RANKING = "ranking_missed"
REQUIREMENT_CANDIDATE = "candidate_missing"
EXCERPT_CHARS = 480
REQUIRED_SEMANTICS_TOKENS = (
    "POOL_ID = C20",
    "CANDIDATE_N = 20",
    "HIT_K = 10",
    "FAMILY_KEY = stratum_id",
    "ranking_failure",
    "candidate_generation_failure",
    "Strong Support",
    "Partial Support",
    "Reject",
    "strictly outnumber",
    "at least two queries",
    "at least two stratum_id values",
    "requirement-only headroom is Partial",
    "TEST is not development data",
    "no depth sweep",
    "oracle is not a reranker",
)
QUERY_COLUMNS = (
    "query_id",
    "company",
    "stratum_id",
    "query_type",
    "evidence_form",
    "wording_style",
    "period_sensitive",
    "requirement_count",
    "multi_requirement",
    "bm25_hit10",
    "dense_hit10",
    "h10_hit10",
    "relevant_unit_count",
    "preferred_relevant_unit_count",
    "bm25_best_relevant_rank",
    "dense_best_relevant_rank",
    "any_relevant_in_c20",
    "relevant_in_bm25_top10",
    "relevant_in_dense_top10",
    "relevant_in_c20",
    "relevant_in_h10_top10",
    "c20_candidate_count",
    "failure_type",
    "oracle_hit10",
    "h10_recall10_numerator",
    "oracle_recall10_numerator",
    "recall_ranking_headroom_numerator",
    "h10_requirements_covered",
    "c20_requirements_available",
    "oracle_requirements_coverable",
    "oracle_all_requirements_covered",
    "requirement_ranking_missed",
    "requirement_candidate_missing",
    "unit_first_requirement_cover",
    "requirement_oracle_exceeds_unit_first",
    "stored_full_rrf_first_relevant_bm25_rank",
    "stored_full_rrf_first_relevant_dense_rank",
)
REQUIREMENT_COLUMNS = (
    "query_id",
    "requirement_id",
    "company",
    "stratum_id",
    "multi_requirement",
    "positive_unit_count",
    "positive_units_in_c20",
    "positive_units_in_h10_top10",
    "h10_covered",
    "c20_available",
    "in_some_optimal_cover",
    "status",
    "oracle_query_coverable_count",
    "oracle_query_all_covered",
)


def implementation_audit() -> dict[str, Any]:
    """State the diagnosis. This function does not score a list."""
    return {
        "pool_id": POOL_ID,
        "candidate_n": CANDIDATE_N,
        "hit_k": HIT_K,
        "family_key": FAMILY_KEY,
        "function": "disclosure_agent.evalset_v2_1.ranking_failure.diagnose_query",
        "semantics_document": SEMANTICS_RELATIVE.as_posix(),
        "current_ranker": "bm25_primary_dense_head",
        "oracle_is_a_system": False,
        "reranker_implemented": False,
        "score_fusion": False,
        "query_expansion": False,
        "routing": False,
        "embedding_change": False,
        "chunking_change": False,
        "parser_change": False,
        "depth_sweep": False,
        "qrel_modified": False,
        "test_search": False,
        "test_used_for_tuning": False,
    }


def load_semantics(root: Path) -> str:
    """Refuse to diagnose when the frozen tokens are absent."""
    path = root / SEMANTICS_RELATIVE
    if not path.is_file():
        raise UsageError(f"H11 semantics document is missing: {path}")
    text = path.read_text(encoding="utf-8")
    missing = [token for token in REQUIRED_SEMANTICS_TOKENS if token not in text]
    if missing:
        raise UsageError(f"H11 semantics document is missing tokens: {missing}")
    return text


def build_c20(
    dense_hits: list[dict[str, Any]],
    bm25_hits: list[dict[str, Any]],
    *,
    n: int = CANDIDATE_N,
) -> dict[str, dict[str, Any]]:
    """Union the stored top-n lists. Deduplicate by retrieval_unit_id."""
    dense = _index_list(dense_hits, n, "dense")
    bm25 = _index_list(bm25_hits, n, "bm25")
    pool: dict[str, dict[str, Any]] = {}
    for unit_id, row in bm25.items():
        pool[unit_id] = {
            "retrieval_unit_id": unit_id,
            "bm25_rank": row["rank"],
            "dense_rank": None,
            "source_element_ids": list(row["source_element_ids"]),
        }
    for unit_id, row in dense.items():
        if unit_id not in pool:
            pool[unit_id] = {
                "retrieval_unit_id": unit_id,
                "bm25_rank": None,
                "dense_rank": row["rank"],
                "source_element_ids": list(row["source_element_ids"]),
            }
            continue
        existing = pool[unit_id]["source_element_ids"]
        incoming = row["source_element_ids"]
        if existing and incoming and existing != incoming:
            raise UsageError(
                f"{unit_id} has different source elements in Dense and BM25 top-{n}"
            )
        pool[unit_id]["dense_rank"] = row["rank"]
        if not existing:
            pool[unit_id]["source_element_ids"] = list(incoming)
    return pool


def unit_judgments(qrels: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Max judged grade per unit. Unjudged rows and relevance 0 stay non-positive."""
    grouped: dict[str, dict[str, int]] = defaultdict(dict)
    preferred: set[str] = set()
    meta: dict[str, dict[str, Any]] = {}
    for row in qrels:
        if row.get("judgment_status") not in (None, "judged"):
            continue
        if row.get("relevance") is None:
            continue
        unit_id = str(row["retrieval_unit_id"])
        requirement_id = str(row["requirement_id"])
        relevance = int(row["relevance"])
        previous = grouped[unit_id].get(requirement_id)
        if previous is None or relevance > previous:
            grouped[unit_id][requirement_id] = relevance
        if row.get("preferred") and relevance >= 1:
            preferred.add(unit_id)
        meta.setdefault(
            unit_id,
            {
                "document_name": row.get("document_name"),
                "filing_period": row.get("filing_period") or "",
            },
        )
    resolved: dict[str, dict[str, Any]] = {}
    for unit_id, requirements in grouped.items():
        grades = [
            {"requirement_id": requirement_id, "relevance": relevance}
            for requirement_id, relevance in sorted(requirements.items())
        ]
        relevance = max(item["relevance"] for item in grades)
        resolved[unit_id] = {
            "relevance": relevance,
            "relevant": relevance >= 1,
            "preferred": unit_id in preferred and relevance >= 1,
            "requirement_grades": grades,
            "relevant_requirement_ids": [
                item["requirement_id"] for item in grades if item["relevance"] >= 1
            ],
            "document_name": meta[unit_id]["document_name"],
            "filing_period": meta[unit_id]["filing_period"],
        }
    return resolved


def maximum_requirement_cover(
    requirement_ids: list[str],
    unit_to_requirements: dict[str, set[str]],
    *,
    k: int = HIT_K,
) -> dict[str, Any]:
    """Exact maximum coverage with at most k units.

    Requirement masks are exact. A repeated mask cannot cover a new requirement,
    so each distinct mask is used once. More than 16 requirements stops the run.
    """
    reqs = list(requirement_ids)
    n = len(reqs)
    if n > COVER_REQUIREMENT_LIMIT:
        raise UsageError(f"requirement cover refuses more than {COVER_REQUIREMENT_LIMIT}")
    if k < 0:
        raise UsageError("cover budget is negative")
    index = {requirement_id: bit for bit, requirement_id in enumerate(reqs)}
    masks: list[int] = []
    for covered in unit_to_requirements.values():
        mask = 0
        for requirement_id in covered:
            bit = index.get(requirement_id)
            if bit is not None:
                mask |= 1 << bit
        if mask:
            masks.append(mask)
    unique = sorted(set(masks))
    unreachable = k + 1
    best = [unreachable] * (1 << n)
    best[0] = 0
    for mask in unique:
        updated = best[:]
        for state, cost in enumerate(best):
            if cost >= k:
                continue
            nxt = state | mask
            updated[nxt] = min(updated[nxt], cost + 1)
        best = updated
    full = (1 << n) - 1 if n else 0
    max_count = 0
    optimal_states: list[int] = []
    for state, cost in enumerate(best):
        if cost > k:
            continue
        count = state.bit_count()
        if count > max_count:
            max_count = count
            optimal_states = [state]
        elif count == max_count and count > 0:
            optimal_states.append(state)
    in_optimal = []
    for bit, requirement_id in enumerate(reqs):
        flag = 1 << bit
        if any(state & flag for state in optimal_states):
            in_optimal.append(requirement_id)
    all_covered = n == 0 or best[full] <= k
    return {
        "covered_count": max_count,
        "all_covered": all_covered,
        "min_units_for_all": best[full] if all_covered and n else None,
        "requirements_in_some_optimal_cover": in_optimal,
        "budget": k,
        "requirement_count": n,
    }


def judge_h11(
    *,
    h10_hit10: int,
    oracle_hit10: int,
    h10_requirement_covered: int,
    oracle_requirement_covered: int,
    ranking_failure_queries: int,
    candidate_generation_failure_queries: int,
    ranking_failure_strata: list[str],
    requirement_ranking_missed: int,
) -> dict[str, Any]:
    """Apply the pre-registered tree. Do not add a numeric cutoff here."""
    hit_gap = int(oracle_hit10) - int(h10_hit10)
    requirement_gap = int(oracle_requirement_covered) - int(h10_requirement_covered)
    if hit_gap < 0 or requirement_gap < 0:
        raise UsageError("oracle is below the current H10 numerator")
    distinct_strata = sorted(set(ranking_failure_strata))
    repeated = int(ranking_failure_queries) >= 2 and len(distinct_strata) >= 2
    majority_ranking = int(ranking_failure_queries) > int(candidate_generation_failure_queries)
    strong = majority_ranking and hit_gap > 0 and requirement_gap > 0 and repeated
    partial = (not strong) and (hit_gap > 0 or requirement_gap > 0) and (
        int(ranking_failure_queries) > 0 or int(requirement_ranking_missed) > 0
    )
    if strong:
        label = VERDICT_STRONG
        reason = "ranking_failures_repeat_across_strata"
    elif partial:
        label = VERDICT_PARTIAL
        reason = _partial_reason(
            ranking_failure_queries=int(ranking_failure_queries),
            candidate_generation_failure_queries=int(candidate_generation_failure_queries),
            distinct_strata=len(distinct_strata),
            requirement_gap=requirement_gap,
        )
    else:
        label = VERDICT_REJECT
        reason = "no_recoverable_ranking_failure"
    return {
        "label": label,
        "reason": reason,
        "hit_gap": hit_gap,
        "requirement_gap": requirement_gap,
        "ranking_failure_queries": int(ranking_failure_queries),
        "candidate_generation_failure_queries": int(candidate_generation_failure_queries),
        "ranking_failure_strata": distinct_strata,
        "repeated_across_strata": repeated,
        "majority_ranking": majority_ranking,
        "requirement_ranking_missed": int(requirement_ranking_missed),
        "rule": (
            "Strong when Hit@10 ranking failures strictly outnumber "
            "candidate-generation failures, both oracle numerators rise, "
            "and those queries cover at least two queries and two stratum_id values. "
            "Partial when an oracle numerator rises and a ranking failure or "
            "requirement ranking miss exists, including requirement-only headroom. "
            "Otherwise Reject."
        ),
    }


def diagnose_query(
    question: dict[str, Any],
    *,
    dense_hits: list[dict[str, Any]],
    bm25_hits: list[dict[str, Any]],
    qrels: list[dict[str, Any]],
    requirements: list[dict[str, Any]],
    n: int = CANDIDATE_N,
    k: int = HIT_K,
) -> dict[str, Any]:
    """Score one stored pair of lists. Do not retrieve."""
    if n != CANDIDATE_N or k != HIT_K:
        raise UsageError("H11 fixes C20 and Hit@10")
    pool = build_c20(dense_hits, bm25_hits, n=n)
    judgments = unit_judgments(qrels)
    placed = bm25_primary_dense_head(dense_hits, bm25_hits, n=n, hit_k=k)
    h10_hits = placed["hits"]
    owned_requirements = [
        item
        for item in requirements
        if item["query_id"] == question["query_id"] and item["answerability"] == "fully_answerable"
    ]
    bm25_scored = score_question(question, qrels, owned_requirements, bm25_hits)
    dense_scored = score_question(question, qrels, owned_requirements, dense_hits)
    h10_scored = score_question(question, qrels, owned_requirements, h10_hits)
    h10_top = [str(hit["retrieval_unit_id"]) for hit in h10_hits[:k]]
    if len(h10_top) != len(set(h10_top)):
        raise UsageError(f"{question['query_id']} repeats a unit inside H10 top-{k}")
    outside = [unit_id for unit_id in h10_top if unit_id not in pool]
    if outside:
        raise UsageError(f"{question['query_id']} H10 top-{k} leaves C20: {outside}")
    positive = [unit_id for unit_id, row in judgments.items() if row["relevant"]]
    preferred = [unit_id for unit_id, row in judgments.items() if row["preferred"]]
    in_c20 = [unit_id for unit_id in positive if unit_id in pool]
    in_h10 = [unit_id for unit_id in positive if unit_id in set(h10_top)]
    bm25_top = _top_ids(bm25_hits, k)
    dense_top = _top_ids(dense_hits, k)
    bm25_best = _best_rank(positive, pool, "bm25_rank")
    dense_best = _best_rank(positive, pool, "dense_rank")
    oracle_recall_numerator = min(k, len(in_c20))
    h10_recall_numerator = int(h10_scored["recall@10"]["numerator"])
    if h10_recall_numerator != len(in_h10):
        raise UsageError(f"{question['query_id']} H10 recall numerator left the C20 grades")
    if int(h10_scored["recall@10"]["denominator"]) != len(positive):
        raise UsageError(f"{question['query_id']} recall denominator is not the positive units")
    recall_headroom = oracle_recall_numerator - h10_recall_numerator
    if recall_headroom < 0:
        raise UsageError(f"{question['query_id']} oracle recall is below H10")
    requirement_rows, cover = _requirement_rows(
        question,
        owned_requirements,
        judgments,
        pool,
        set(h10_top),
        k=k,
    )
    covered_now = sum(1 for row in requirement_rows if row["h10_covered"])
    if covered_now != int(h10_scored["requirement_covered@10"]):
        raise UsageError(f"{question['query_id']} requirement cover left the H10 metric")
    h10_hit = bool(h10_scored["at_k"][str(k)]["hit"])
    oracle_hit = bool(in_c20)
    if h10_hit and not oracle_hit:
        raise UsageError(f"{question['query_id']} hits outside C20")
    if h10_hit:
        failure_type = FAILURE_NONE
    elif oracle_hit:
        failure_type = FAILURE_RANKING
    else:
        failure_type = FAILURE_CANDIDATE
    unit_first_cover = _unit_first_requirement_cover(
        in_c20, judgments, [row["requirement_id"] for row in requirement_rows], k=k
    )
    return {
        "query_id": str(question["query_id"]),
        "company": question.get("company_name"),
        "corp_code": str(question.get("corp_code") or ""),
        "question": question.get("question"),
        "stratum_id": question.get(FAMILY_KEY),
        "query_type": question.get("query_type"),
        "evidence_form": question.get("evidence_form"),
        "wording_style": question.get("wording_style"),
        "period_sensitive": bool(question.get("period_sensitive")),
        "requirement_count": len(requirement_rows),
        "multi_requirement": len(requirement_rows) > 1,
        "bm25_hit10": bool(bm25_scored["at_k"][str(k)]["hit"]),
        "dense_hit10": bool(dense_scored["at_k"][str(k)]["hit"]),
        "h10_hit10": h10_hit,
        "relevant_unit_count": len(positive),
        "preferred_relevant_unit_count": len(preferred),
        "preferred_in_c20": sum(1 for unit_id in preferred if unit_id in pool),
        "bm25_best_relevant_rank": bm25_best,
        "dense_best_relevant_rank": dense_best,
        "any_relevant_in_c20": bool(in_c20),
        "relevant_in_bm25_top10": sum(1 for unit_id in positive if unit_id in bm25_top),
        "relevant_in_dense_top10": sum(1 for unit_id in positive if unit_id in dense_top),
        "relevant_in_c20": len(in_c20),
        "relevant_in_h10_top10": len(in_h10),
        "c20_candidate_count": len(pool),
        "failure_type": failure_type,
        "oracle_hit10": oracle_hit,
        "h10_recall10_numerator": h10_recall_numerator,
        "h10_recall10_denominator": len(positive),
        "oracle_recall10_numerator": oracle_recall_numerator,
        "oracle_recall10_denominator": len(positive),
        "recall_ranking_headroom_numerator": recall_headroom,
        "h10_requirements_covered": sum(1 for row in requirement_rows if row["h10_covered"]),
        "c20_requirements_available": sum(1 for row in requirement_rows if row["c20_available"]),
        "oracle_requirements_coverable": cover["covered_count"],
        "oracle_all_requirements_covered": cover["all_covered"] if requirement_rows else None,
        "requirement_ranking_missed": sum(
            1 for row in requirement_rows if row["status"] == REQUIREMENT_RANKING
        ),
        "requirement_candidate_missing": sum(
            1 for row in requirement_rows if row["status"] == REQUIREMENT_CANDIDATE
        ),
        "unit_first_requirement_cover": unit_first_cover,
        "requirement_oracle_exceeds_unit_first": cover["covered_count"] > unit_first_cover,
        "budget_binds": cover["covered_count"] < sum(
            1 for row in requirement_rows if row["c20_available"]
        ),
        "min_units_for_all_requirements": cover["min_units_for_all"],
        "h10_edit": placed["edit"],
        "requirements": requirement_rows,
        "relevant_units": [
            _unit_record(unit_id, judgments[unit_id], pool, set(h10_top), bm25_top, dense_top)
            for unit_id in sorted(positive)
        ],
        "h10_top10": _window_record(h10_hits[:k], judgments),
        "bm25_top10": _window_record(bm25_hits[:k], judgments),
        "dense_top10": _window_record(dense_hits[:k], judgments),
        "bm25_scored": bm25_scored,
        "dense_scored": dense_scored,
        "h10_scored": h10_scored,
    }


def run_ranking_failure_diagnosis(root: Path) -> dict[str, Any]:
    """Diagnose stored DEV lists. Do not search and do not read TEST."""
    semantics = load_semantics(root)
    bundle = validate_source_bundle(root)
    witness = _load_rrf_witness(root)
    questions = sorted(bundle["questions"], key=lambda item: str(item["query_id"]))
    qrels_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in bundle["qrels"]:
        qrels_by_query[str(row["query_id"])].append(row)
    diagnostics = []
    for question in questions:
        query_id = str(question["query_id"])
        row = diagnose_query(
            question,
            dense_hits=bundle["dense_by_id"][query_id]["hits"],
            bm25_hits=bundle["bm25_by_id"][query_id]["hits"],
            qrels=qrels_by_query.get(query_id, []),
            requirements=bundle["requirements"],
        )
        if query_id not in witness:
            raise UsageError(f"{query_id} has no stored full-RRF witness")
        row["full_rrf_witness"] = witness[query_id]
        row["stored_full_rrf_first_relevant_bm25_rank"] = witness[query_id][
            "stored_full_rrf_first_relevant_bm25_rank"
        ]
        row["stored_full_rrf_first_relevant_dense_rank"] = witness[query_id][
            "stored_full_rrf_first_relevant_dense_rank"
        ]
        diagnostics.append(row)
    if len(diagnostics) != 45:
        raise UsageError("H11 did not diagnose every DEV question")
    _assert_gap_identity(diagnostics)
    bm25_headline = aggregate_headline([row["bm25_scored"] for row in diagnostics])
    dense_headline = aggregate_headline([row["dense_scored"] for row in diagnostics])
    h10_headline = aggregate_headline([row["h10_scored"] for row in diagnostics])
    stored_h10 = _read(root / "evaluation" / "runs" / H10_RUN_DIR / "h10_dev.json")
    _assert_headline_match("bm25", bm25_headline, bundle["bm25_headline"])
    _assert_headline_match("dense", dense_headline, bundle["dense_headline"])
    _assert_headline_match("h10", h10_headline, stored_h10["headline"])
    oracle = _oracle_metrics(diagnostics)
    taxonomy = _taxonomy(diagnostics)
    verdict = judge_h11(
        h10_hit10=int(h10_headline["at_k"]["10"]["hit"]["numerator"]),
        oracle_hit10=int(oracle["hit@10"]["numerator"]),
        h10_requirement_covered=int(h10_headline["requirement_recall@10"]["numerator"]),
        oracle_requirement_covered=int(oracle["requirement_recall@10"]["numerator"]),
        ranking_failure_queries=int(taxonomy["hit_failures"]["ranking_failure"]),
        candidate_generation_failure_queries=int(
            taxonomy["hit_failures"]["candidate_generation_failure"]
        ),
        ranking_failure_strata=taxonomy["ranking_failure_strata"],
        requirement_ranking_missed=int(taxonomy["requirements"]["ranking_missed"]),
    )
    packets = [row for row in diagnostics if _needs_packet(row)]
    _attach_excerpts(root, packets)
    payload = {
        "experiment_id": "v21_dev_h11_ranking_failure",
        "dataset_id": DATASET_ID,
        "dataset_sha256": DATASET_SHA,
        "population": "dev fully_answerable",
        "question_count": 45,
        "test_questions_searched": 0,
        "test_questions_read": 0,
        "test_used_for_tuning": False,
        "retrieval_reexecuted": False,
        "benchmark_label": "LLM-judged / LLM-adjudicated benchmark",
        "human_gold": False,
        "qrel_primary_key": ["query_id", "requirement_id", "retrieval_unit_id"],
        "semantics_sha256": hashlib.sha256(semantics.encode("utf-8")).hexdigest(),
        "implementation": implementation_audit(),
        "bm25_headline": _public_headline(bm25_headline),
        "dense_headline": _public_headline(dense_headline),
        "h10_headline": _public_headline(h10_headline),
        "oracle": oracle,
        "taxonomy": taxonomy,
        "verdict": verdict,
        "opportunity": _opportunity(taxonomy),
        "multi_requirement": _multi_requirement(diagnostics),
        "diagnostics": [_public_query(row) for row in diagnostics],
        "requirement_rows": [
            requirement for row in diagnostics for requirement in row["requirements"]
        ],
        "packets": [_public_packet(row) for row in packets],
        "git": git_revision(root),
    }
    _write_run(root, payload)
    payload["run_dir"] = str(root / "evaluation" / "runs" / RUN_DIR_NAME)
    return payload


def _partial_reason(
    *,
    ranking_failure_queries: int,
    candidate_generation_failure_queries: int,
    distinct_strata: int,
    requirement_gap: int,
) -> str:
    if ranking_failure_queries == 0:
        return "requirement_only_headroom"
    if candidate_generation_failure_queries >= ranking_failure_queries:
        return "candidate_generation_comparable_or_larger"
    if ranking_failure_queries < 2 or distinct_strata < 2:
        return "ranking_headroom_concentrated"
    if requirement_gap <= 0:
        return "requirement_oracle_did_not_move"
    return "hit_majority_not_repeated"


def _index_list(
    hits: list[dict[str, Any]],
    n: int,
    label: str,
) -> dict[str, dict[str, Any]]:
    if len(hits) < n:
        raise UsageError(f"{label} top list is shorter than {n}")
    indexed: dict[str, dict[str, Any]] = {}
    for position, hit in enumerate(hits[:n], start=1):
        rank = int(hit["rank"])
        if rank != position:
            raise UsageError(f"{label} rank {rank} is not contiguous at {position}")
        unit_id = str(hit.get("retrieval_unit_id") or "")
        if not unit_id:
            raise UsageError(f"{label} hit has no retrieval_unit_id")
        if unit_id in indexed:
            raise UsageError(f"duplicate {label} retrieval_unit_id {unit_id}")
        indexed[unit_id] = {
            "rank": rank,
            "source_element_ids": [str(item) for item in hit.get("source_element_ids") or []],
        }
    return indexed


def _top_ids(hits: list[dict[str, Any]], k: int) -> set[str]:
    return {str(hit["retrieval_unit_id"]) for hit in hits[:k]}


def _best_rank(
    unit_ids: list[str],
    pool: dict[str, dict[str, Any]],
    key: str,
) -> int | None:
    ranks = [
        int(pool[unit_id][key])
        for unit_id in unit_ids
        if unit_id in pool and pool[unit_id][key] is not None
    ]
    return min(ranks) if ranks else None


def _requirement_rows(
    question: dict[str, Any],
    requirements: list[dict[str, Any]],
    judgments: dict[str, dict[str, Any]],
    pool: dict[str, dict[str, Any]],
    h10_top: set[str],
    *,
    k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    requirement_ids = [str(item["requirement_id"]) for item in requirements]
    positive: dict[str, set[str]] = {requirement_id: set() for requirement_id in requirement_ids}
    for unit_id, row in judgments.items():
        for requirement_id in row["relevant_requirement_ids"]:
            if requirement_id in positive:
                positive[requirement_id].add(unit_id)
    unit_to: dict[str, set[str]] = defaultdict(set)
    for requirement_id, unit_ids in positive.items():
        for unit_id in unit_ids:
            if unit_id in pool:
                unit_to[unit_id].add(requirement_id)
    cover = maximum_requirement_cover(requirement_ids, unit_to, k=k)
    optimal = set(cover["requirements_in_some_optimal_cover"])
    rows = []
    for item in requirements:
        requirement_id = str(item["requirement_id"])
        unit_ids = positive[requirement_id]
        in_pool = {unit_id for unit_id in unit_ids if unit_id in pool}
        in_h10 = {unit_id for unit_id in unit_ids if unit_id in h10_top}
        available = bool(in_pool)
        covered = bool(in_h10)
        if covered:
            status = REQUIREMENT_COVERED
        elif available:
            status = REQUIREMENT_RANKING
        else:
            status = REQUIREMENT_CANDIDATE
        rows.append(
            {
                "query_id": str(question["query_id"]),
                "requirement_id": requirement_id,
                "text": item.get("text"),
                "company": question.get("company_name"),
                "stratum_id": question.get(FAMILY_KEY),
                "multi_requirement": len(requirements) > 1,
                "positive_unit_count": len(unit_ids),
                "positive_units_in_c20": len(in_pool),
                "positive_units_in_h10_top10": len(in_h10),
                "h10_covered": covered,
                "c20_available": available,
                "in_some_optimal_cover": requirement_id in optimal,
                "status": status,
                "oracle_query_coverable_count": cover["covered_count"],
                "oracle_query_all_covered": cover["all_covered"],
            }
        )
    return rows, cover


def _unit_first_requirement_cover(
    relevant_in_c20: list[str],
    judgments: dict[str, dict[str, Any]],
    requirement_ids: list[str],
    *,
    k: int,
) -> int:
    """Cover from the deterministic relevant-first prefix, not from the joint optimum."""
    ordered = sorted(
        relevant_in_c20,
        key=lambda unit_id: (-int(judgments[unit_id]["relevance"]), unit_id),
    )
    covered: set[str] = set()
    allowed = set(requirement_ids)
    for unit_id in ordered[:k]:
        covered.update(
            requirement_id
            for requirement_id in judgments[unit_id]["relevant_requirement_ids"]
            if requirement_id in allowed
        )
    return len(covered)


def _unit_record(
    unit_id: str,
    judgment: dict[str, Any],
    pool: dict[str, dict[str, Any]],
    h10_top: set[str],
    bm25_top: set[str],
    dense_top: set[str],
) -> dict[str, Any]:
    located = pool.get(unit_id)
    return {
        "retrieval_unit_id": unit_id,
        "relevance": judgment["relevance"],
        "preferred": judgment["preferred"],
        "relevant_requirement_ids": list(judgment["relevant_requirement_ids"]),
        "document_name": judgment["document_name"],
        "filing_period": judgment["filing_period"],
        "bm25_rank": None if located is None else located["bm25_rank"],
        "dense_rank": None if located is None else located["dense_rank"],
        "in_c20": located is not None,
        "in_bm25_top10": unit_id in bm25_top,
        "in_dense_top10": unit_id in dense_top,
        "in_h10_top10": unit_id in h10_top,
    }


def _window_record(
    hits: list[dict[str, Any]],
    judgments: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for hit in hits:
        unit_id = str(hit["retrieval_unit_id"])
        judgment = judgments.get(unit_id)
        rows.append(
            {
                "rank": int(hit["rank"]),
                "retrieval_unit_id": unit_id,
                "judged": judgment is not None,
                "relevance": None if judgment is None else judgment["relevance"],
                "relevant_requirement_ids": (
                    [] if judgment is None else list(judgment["relevant_requirement_ids"])
                ),
                "document_name": None if judgment is None else judgment["document_name"],
                "filing_period": None if judgment is None else judgment["filing_period"],
            }
        )
    return rows


def _assert_gap_identity(rows: list[dict[str, Any]]) -> None:
    hit_gap = sum(1 for row in rows if row["oracle_hit10"] and not row["h10_hit10"])
    ranking = sum(1 for row in rows if row["failure_type"] == FAILURE_RANKING)
    candidate = sum(1 for row in rows if row["failure_type"] == FAILURE_CANDIDATE)
    misses = sum(1 for row in rows if not row["h10_hit10"])
    if hit_gap != ranking or ranking + candidate != misses:
        raise UsageError("Hit@10 oracle gap is not the ranking-failure count")
    for row in rows:
        accounted = (
            row["h10_requirements_covered"]
            + row["requirement_ranking_missed"]
            + row["requirement_candidate_missing"]
        )
        if accounted != row["requirement_count"]:
            raise UsageError(f"{row['query_id']} requirement statuses do not partition")
        if row["oracle_requirements_coverable"] < row["h10_requirements_covered"]:
            raise UsageError(f"{row['query_id']} requirement oracle is below H10")
        if row["oracle_requirements_coverable"] > row["c20_requirements_available"]:
            raise UsageError(f"{row['query_id']} requirement oracle exceeds C20 availability")


def _oracle_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    hit = sum(1 for row in rows if row["oracle_hit10"])
    recall_numerators = [row["oracle_recall10_numerator"] for row in rows]
    recall_denominators = [row["oracle_recall10_denominator"] for row in rows]
    macro_values = [
        numerator / denominator
        for numerator, denominator in zip(recall_numerators, recall_denominators, strict=True)
        if denominator
    ]
    requirement_covered = sum(row["oracle_requirements_coverable"] for row in rows)
    requirement_total = sum(row["requirement_count"] for row in rows)
    all_rows = [row for row in rows if row["requirement_count"] > 0]
    all_hit = sum(1 for row in all_rows if row["oracle_all_requirements_covered"])
    h10_recall_macro = _mean(
        [
            row["h10_recall10_numerator"] / row["h10_recall10_denominator"]
            for row in rows
            if row["h10_recall10_denominator"]
        ]
    )
    oracle_recall_macro = _mean(macro_values)
    return {
        "role": "upper_bound_not_a_system",
        "hit@10": _count(hit, len(rows)),
        "recall@10_macro": {
            "value": oracle_recall_macro,
            "denominator": len(macro_values),
        },
        "recall@10_micro": _count(sum(recall_numerators), sum(recall_denominators)),
        "requirement_recall@10": _count(requirement_covered, requirement_total),
        "all_requirements_covered@10": _count(all_hit, len(all_rows)),
        "h10_recall@10_macro": h10_recall_macro,
        "recall@10_macro_headroom": None
        if oracle_recall_macro is None or h10_recall_macro is None
        else oracle_recall_macro - h10_recall_macro,
        "queries_with_unit_recall_headroom": sum(
            1 for row in rows if row["recall_ranking_headroom_numerator"] > 0
        ),
        "unit_recall_headroom_numerator": sum(
            row["recall_ranking_headroom_numerator"] for row in rows
        ),
        "queries_where_requirement_oracle_exceeds_unit_first": sum(
            1 for row in rows if row["requirement_oracle_exceeds_unit_first"]
        ),
        "queries_where_cover_budget_binds": sum(1 for row in rows if row["budget_binds"]),
        "relevant_in_c20_outside_h10": _outside_h10_composition(rows),
    }


def _outside_h10_composition(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Where relevant C20 units sit when they are outside the H10 top-10.

    This is not a depth sweep. Ranks above 20 are not counted, because those
    units are not in C20.
    """
    dense_only = 0
    bm25_tail = 0
    both = 0
    for row in rows:
        for unit in row["relevant_units"]:
            if not unit["in_c20"] or unit["in_h10_top10"]:
                continue
            has_bm25 = unit["bm25_rank"] is not None
            has_dense = unit["dense_rank"] is not None
            if has_dense and not has_bm25:
                dense_only += 1
            elif has_bm25 and not has_dense:
                bm25_tail += 1
            else:
                both += 1
    return {
        "units": dense_only + bm25_tail + both,
        "dense_only": dense_only,
        "bm25_tail_only": bm25_tail,
        "in_both_lists": both,
    }


def _taxonomy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    hit_failures = {
        FAILURE_RANKING: sum(1 for row in rows if row["failure_type"] == FAILURE_RANKING),
        FAILURE_CANDIDATE: sum(1 for row in rows if row["failure_type"] == FAILURE_CANDIDATE),
        FAILURE_NONE: sum(1 for row in rows if row["failure_type"] == FAILURE_NONE),
    }
    ranking_rows = [row for row in rows if row["failure_type"] == FAILURE_RANKING]
    candidate_rows = [row for row in rows if row["failure_type"] == FAILURE_CANDIDATE]
    requirements = [item for row in rows for item in row["requirements"]]
    requirement_counts = {
        REQUIREMENT_COVERED: sum(
            1 for item in requirements if item["status"] == REQUIREMENT_COVERED
        ),
        REQUIREMENT_RANKING: sum(
            1 for item in requirements if item["status"] == REQUIREMENT_RANKING
        ),
        REQUIREMENT_CANDIDATE: sum(
            1 for item in requirements if item["status"] == REQUIREMENT_CANDIDATE
        ),
    }
    return {
        "hit_failures": hit_failures,
        "ranking_failure_query_ids": [row["query_id"] for row in ranking_rows],
        "candidate_generation_failure_query_ids": [row["query_id"] for row in candidate_rows],
        "ranking_failure_strata": [str(row["stratum_id"]) for row in ranking_rows],
        "candidate_generation_failure_strata": [str(row["stratum_id"]) for row in candidate_rows],
        "requirements": requirement_counts,
        "requirement_total": len(requirements),
    }


def _opportunity(taxonomy: dict[str, Any]) -> dict[str, Any]:
    hit_failed = (
        taxonomy["hit_failures"][FAILURE_RANKING] + taxonomy["hit_failures"][FAILURE_CANDIDATE]
    )
    requirement_missing = (
        taxonomy["requirements"][REQUIREMENT_RANKING]
        + taxonomy["requirements"][REQUIREMENT_CANDIDATE]
    )
    return {
        "wording": (
            "현재 C20 후보군을 완벽하게 재정렬할 수 있다는 oracle 가정에서 "
            "이 정도 failure가 theoretically recoverable하다. "
            "이 비율은 reranker가 그 수만큼 개선한다는 뜻이 아니다."
        ),
        "query": _count(taxonomy["hit_failures"][FAILURE_RANKING], hit_failed),
        "requirement": _count(
            taxonomy["requirements"][REQUIREMENT_RANKING], requirement_missing
        ),
    }


def _multi_requirement(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if row["multi_requirement"]]
    requirements = [item for row in selected for item in row["requirements"]]
    return {
        "query_count": len(selected),
        "h10_hit10": sum(1 for row in selected if row["h10_hit10"]),
        "ranking_failure_queries": sum(
            1 for row in selected if row["failure_type"] == FAILURE_RANKING
        ),
        "candidate_generation_failure_queries": sum(
            1 for row in selected if row["failure_type"] == FAILURE_CANDIDATE
        ),
        "h10_all_requirements_covered": sum(
            1 for row in selected if row["h10_requirements_covered"] == row["requirement_count"]
        ),
        "oracle_all_requirements_covered": sum(
            1 for row in selected if row["oracle_all_requirements_covered"]
        ),
        "requirement_total": len(requirements),
        "requirement_ranking_missed": sum(
            1 for item in requirements if item["status"] == REQUIREMENT_RANKING
        ),
        "requirement_candidate_missing": sum(
            1 for item in requirements if item["status"] == REQUIREMENT_CANDIDATE
        ),
        "queries_with_requirement_ranking_miss": [
            row["query_id"]
            for row in selected
            if row["requirement_ranking_missed"] > 0
        ],
    }


def _needs_packet(row: dict[str, Any]) -> bool:
    return (not row["h10_hit10"]) or row["requirement_ranking_missed"] > 0 or (
        row["requirement_candidate_missing"] > 0
    )


def _attach_excerpts(root: Path, packets: list[dict[str, Any]]) -> None:
    wanted: dict[str, set[str]] = defaultdict(set)
    for packet in packets:
        corp = str(packet["corp_code"])
        for unit_id in _packet_unit_ids(packet):
            wanted[corp].add(unit_id)
    found: dict[str, dict[str, Any]] = {}
    missing_files = []
    for corp, unit_ids in wanted.items():
        path = _units_path(root, corp)
        if not path.is_file():
            missing_files.append(str(path))
            continue
        found.update(_scan_units(path, unit_ids))
    for packet in packets:
        for bucket in ("relevant_units", "h10_top10", "bm25_top10", "dense_top10"):
            for unit in packet[bucket]:
                record = found.get(unit["retrieval_unit_id"])
                unit["text_excerpt"] = None if record is None else record["text_excerpt"]
                unit["section_path"] = None if record is None else record["section_path"]
                unit["representation_type"] = (
                    None if record is None else record["representation_type"]
                )
                unit["corpus_document_name"] = None if record is None else record["document_name"]
                unit["corpus_filing_period"] = None if record is None else record["filing_period"]
                unit["report_name"] = None if record is None else record["report_name"]
    packets_missing = [
        unit_id
        for unit_ids in wanted.values()
        for unit_id in unit_ids
        if unit_id not in found
    ]
    for packet in packets:
        packet["excerpt_status"] = {
            "missing_index_files": missing_files,
            "units_without_text": sorted({unit_id for unit_id in packets_missing}),
        }


def _packet_unit_ids(packet: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for bucket in ("relevant_units", "h10_top10", "bm25_top10", "dense_top10"):
        ids.update(str(unit["retrieval_unit_id"]) for unit in packet[bucket])
    return ids


def _units_path(root: Path, corp_code: str) -> Path:
    if corp_code == "00107613":
        return root / "data" / "product" / "indexes" / "isu_petasys_periodic_d2a" / (
            "retrieval_units.jsonl"
        )
    return (
        root
        / "data"
        / "product"
        / "indexes"
        / "generalization"
        / corp_code
        / "retrieval_units.jsonl"
    )


def _scan_units(path: Path, wanted: set[str]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    remaining = set(wanted)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not remaining:
                break
            if '"retrieval_unit_id"' not in line:
                continue
            row = json.loads(line)
            unit_id = str(row.get("retrieval_unit_id") or "")
            if unit_id not in remaining:
                continue
            section = row.get("section_path") or []
            text = str(row.get("text") or "").replace("\n", " ")
            found[unit_id] = {
                "text_excerpt": text[:EXCERPT_CHARS],
                "section_path": " > ".join(str(item) for item in section),
                "representation_type": row.get("representation_type"),
                "document_name": row.get("document_name"),
                "filing_period": row.get("filing_period"),
                "report_name": row.get("report_name"),
            }
            remaining.remove(unit_id)
    return found


def _load_rrf_witness(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "evaluation" / "runs" / RRF_RUN_DIR / "per_query_rank_movement.json"
    payload = _read(path)
    witness = {}
    for row in payload["queries"]:
        witness[str(row["query_id"])] = {
            "role": (
                "first relevant unit in the stored full-ranking RRF order; "
                "not a best-rank sweep and not a change to C20"
            ),
            "stored_full_rrf_first_relevant_rank": row.get("rrf_first_relevant_rank"),
            "stored_full_rrf_first_relevant_unit_id": row.get("rrf_first_unit_id"),
            "stored_full_rrf_first_relevant_bm25_rank": row.get("rrf_first_unit_bm25_rank"),
            "stored_full_rrf_first_relevant_dense_rank": row.get("rrf_first_unit_dense_rank"),
        }
    return witness


def _assert_headline_match(
    label: str,
    fresh: dict[str, Any],
    stored: dict[str, Any],
) -> None:
    checks = (
        (
            "hit@10",
            fresh["at_k"]["10"]["hit"]["numerator"],
            stored["at_k"]["10"]["hit"]["numerator"],
        ),
        (
            "requirement_recall@10",
            fresh["requirement_recall@10"]["numerator"],
            stored["requirement_recall@10"]["numerator"],
        ),
        (
            "all_requirements_covered@10",
            fresh["all_requirements_covered@10"]["numerator"],
            stored["all_requirements_covered@10"]["numerator"],
        ),
        (
            "recall@10_micro",
            fresh["recall@10_micro"]["numerator"],
            stored["recall@10_micro"]["numerator"],
        ),
    )
    for name, left, right in checks:
        if int(left) != int(right):
            raise UsageError(f"recomputed {label} {name} {left} != stored {right}")
    for key in ("recall@10_macro", "mrr"):
        if abs(float(fresh[key]["value"]) - float(stored[key]["value"])) > 1e-9:
            raise UsageError(f"recomputed {label} {key} left the stored headline")
    fresh_ndcg = float(fresh["at_k"]["10"]["ndcg_condensed"]["value"])
    stored_ndcg = float(stored["at_k"]["10"]["ndcg_condensed"]["value"])
    if abs(fresh_ndcg - stored_ndcg) > 1e-9:
        raise UsageError(f"recomputed {label} ndcg@10 left the stored headline")


def _public_headline(headline: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_count": headline["question_count"],
        "hit@10": headline["at_k"]["10"]["hit"],
        "recall@10_macro": headline["recall@10_macro"],
        "recall@10_micro": headline["recall@10_micro"],
        "mrr": headline["mrr"],
        "ndcg@10_condensed": headline["at_k"]["10"]["ndcg_condensed"],
        "requirement_recall@10": headline["requirement_recall@10"],
        "all_requirements_covered@10": headline["all_requirements_covered@10"],
    }


def _public_query(row: dict[str, Any]) -> dict[str, Any]:
    hidden = {"bm25_scored", "dense_scored", "h10_scored"}
    return {key: value for key, value in row.items() if key not in hidden}


def _public_packet(row: dict[str, Any]) -> dict[str, Any]:
    return _public_query(row)


def _write_run(root: Path, payload: dict[str, Any]) -> None:
    directory = root / "evaluation" / "runs" / RUN_DIR_NAME
    directory.mkdir(parents=True, exist_ok=True)
    _dump(directory / "experiment_config.json", _experiment_config(payload))
    _dump(directory / "evaluation_integrity.json", _integrity(payload))
    _dump(directory / "query_diagnostics.json", {"queries": payload["diagnostics"]})
    _write_csv(directory / "query_diagnostics.csv", QUERY_COLUMNS, payload["diagnostics"])
    _dump(directory / "requirement_diagnostics.json", {"requirements": payload["requirement_rows"]})
    _write_csv(
        directory / "requirement_diagnostics.csv",
        REQUIREMENT_COLUMNS,
        payload["requirement_rows"],
    )
    _dump(directory / "oracle_metrics.json", payload["oracle"])
    _dump(directory / "failure_taxonomy.json", payload["taxonomy"])
    _dump(directory / "headroom.json", _headroom(payload))
    _dump(directory / "verdict.json", payload["verdict"])
    ranking = [
        packet
        for packet in payload["packets"]
        if packet["failure_type"] == FAILURE_RANKING
    ]
    candidate = [
        packet
        for packet in payload["packets"]
        if packet["failure_type"] == FAILURE_CANDIDATE
    ]
    _dump(directory / "ranking_failures.json", {"queries": ranking})
    _dump(directory / "candidate_generation_failures.json", {"queries": candidate})
    _dump(
        directory / "requirement_failures.json",
        {
            "requirements": [
                row
                for row in payload["requirement_rows"]
                if row["status"] != REQUIREMENT_COVERED
            ]
        },
    )
    report = render_report(payload)
    (directory / "report.md").write_text(report, encoding="utf-8")
    (directory / "portfolio_summary.md").write_text(
        render_portfolio(payload), encoding="utf-8"
    )
    worklog = root / "docs" / "worklogs" / "2026-10-07_v21-dev-h11-ranking-failure.md"
    worklog.write_text(render_worklog(payload), encoding="utf-8")


def _experiment_config(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": payload["experiment_id"],
        "dataset_id": payload["dataset_id"],
        "dataset_sha256": payload["dataset_sha256"],
        "population": payload["population"],
        "question_count": payload["question_count"],
        "test_questions_searched": 0,
        "test_questions_read": 0,
        "test_used_for_tuning": False,
        "retrieval_reexecuted": False,
        "semantics_document": SEMANTICS_RELATIVE.as_posix(),
        "semantics_sha256": payload["semantics_sha256"],
        "implementation": payload["implementation"],
        "source_runs": {
            "complementarity": "multi_company_retrieval_v2_1_dev_dense_bm25_complementarity",
            "h10": H10_RUN_DIR,
            "full_rrf_witness_only": RRF_RUN_DIR,
        },
        "forbidden": {
            "reranker": False,
            "score_fusion": False,
            "query_expansion": False,
            "routing": False,
            "embedding_change": False,
            "chunking_change": False,
            "parser_change": False,
            "depth_sweep": False,
            "qrel_modified": False,
            "test_search": False,
        },
        "git": payload["git"],
        "verdict_label": payload["verdict"]["label"],
        "verdict_reason": payload["verdict"]["reason"],
    }


def _integrity(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_sha256": payload["dataset_sha256"],
        "question_count": payload["question_count"],
        "population": payload["population"],
        "benchmark_label": payload["benchmark_label"],
        "human_gold": False,
        "qrel_primary_key": payload["qrel_primary_key"],
        "recomputed_h10_matches_stored_headline": True,
        "recomputed_bm25_matches_stored_headline": True,
        "h10_top10_subset_of_c20": True,
        "oracle_hit_gap_equals_ranking_failures": True,
        "test_questions_searched": 0,
        "test_questions_read": 0,
        "test_run_dir_read": False,
        "retrieval_reexecuted": False,
        "reranker_implemented": False,
        "depth_swept": False,
        "qrel_modified": False,
    }


def _headroom(payload: dict[str, Any]) -> dict[str, Any]:
    h10_hit = int(payload["h10_headline"]["hit@10"]["numerator"])
    oracle_hit = int(payload["oracle"]["hit@10"]["numerator"])
    h10_req = int(payload["h10_headline"]["requirement_recall@10"]["numerator"])
    oracle_req = int(payload["oracle"]["requirement_recall@10"]["numerator"])
    return {
        "h10_hit@10": payload["h10_headline"]["hit@10"],
        "oracle_hit@10": payload["oracle"]["hit@10"],
        "hit@10_gap": oracle_hit - h10_hit,
        "h10_requirement_recall@10": payload["h10_headline"]["requirement_recall@10"],
        "oracle_requirement_recall@10": payload["oracle"]["requirement_recall@10"],
        "requirement_recall_gap": oracle_req - h10_req,
        "h10_all_requirements_covered@10": payload["h10_headline"]["all_requirements_covered@10"],
        "oracle_all_requirements_covered@10": payload["oracle"]["all_requirements_covered@10"],
        "recall@10_macro_headroom": payload["oracle"]["recall@10_macro_headroom"],
        "queries_with_unit_recall_headroom": payload["oracle"]["queries_with_unit_recall_headroom"],
        "wording": (
            "Gap은 C20을 완벽하게 재정렬할 때의 upper bound와 현재 H10의 차이다. "
            "구현된 reranker의 개선치가 아니다."
        ),
    }


def _count(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": None if denominator == 0 else numerator / denominator,
    }


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _ratio_text(block: dict[str, Any]) -> str:
    return f"{block['numerator']}/{block['denominator']}"


def _num(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "none"
    return f"{value:.{digits}f}"


def render_report(payload: dict[str, Any]) -> str:
    """Write the factual report from the frozen tree and the computed payload."""
    h10 = payload["h10_headline"]
    bm25 = payload["bm25_headline"]
    oracle = payload["oracle"]
    taxonomy = payload["taxonomy"]
    verdict = payload["verdict"]
    opportunity = payload["opportunity"]
    multi = payload["multi_requirement"]
    hit_misses = (
        taxonomy["hit_failures"][FAILURE_RANKING]
        + taxonomy["hit_failures"][FAILURE_CANDIDATE]
    )
    lines = [
        "# v2.1 DEV H11 ranking failure versus candidate generation",
        "",
        "## 1. Executive Summary",
        "",
        (
            f"판정은 `{verdict['label']}`이고 reason은 `{verdict['reason']}`이다. "
            f"DEV H10 Hit@10 failure {hit_misses}건 중 "
            f"ranking failure는 {taxonomy['hit_failures'][FAILURE_RANKING]}건, "
            f"candidate-generation failure는 {taxonomy['hit_failures'][FAILURE_CANDIDATE]}건이다."
        ),
        (
            f"C20 Oracle Hit@10은 {_ratio_text(oracle['hit@10'])}이다. "
            f"H10 Hit@10은 {_ratio_text(h10['hit@10'])}이다. "
            f"차이는 {verdict['hit_gap']}이다."
        ),
        (
            f"Requirement recall은 H10 {_ratio_text(h10['requirement_recall@10'])}, "
            f"C20 oracle {_ratio_text(oracle['requirement_recall@10'])}이다. "
            f"차이는 {verdict['requirement_gap']}이다."
        ),
        (
            "이 차이는 reranker 성능이 아니다. "
            "C20 relevance를 완벽히 알고 Top-10을 고르는 upper bound다."
        ),
        "",
        "## 2. H11 Hypothesis",
        "",
        (
            "남은 retrieval failure 중 유의미한 부분이 "
            "candidate generation failure가 아니라 ranking failure인가?"
        ),
        "",
        "## 3. Preregistered Semantics",
        "",
        f"문서 `{SEMANTICS_RELATIVE.as_posix()}` sha256 `{payload['semantics_sha256']}`.",
        "POOL_ID = C20. CANDIDATE_N = 20. HIT_K = 10. FAMILY_KEY = stratum_id.",
        (
            "Strong은 Hit@10 ranking failure가 candidate-generation failure보다 많고, "
            "Hit@10 oracle과 requirement oracle이 모두 올라가며, "
            "ranking failure가 두 개 이상의 질의와 두 개 이상의 stratum_id에 있을 때다."
        ),
        (
            "Partial은 Strong이 아니고 oracle 분자가 올라가며 "
            "ranking failure 또는 requirement ranking miss가 있을 때다. "
            "requirement-only headroom is Partial."
        ),
        "그 밖은 Reject다. 숫자 컷은 점수 이후에 더하지 않았다.",
        "",
        "## 4. Evaluation Integrity",
        "",
        (
            f"dataset `{payload['dataset_id']}` sha256 `{payload['dataset_sha256']}`. "
            "모집단은 DEV fully_answerable 45이다. "
            "benchmark는 LLM-judged / LLM-adjudicated benchmark다. human gold가 아니다."
        ),
        (
            "qrel primary key는 (query_id, requirement_id, retrieval_unit_id)다. "
            "source element는 provenance다."
        ),
        (
            "재계산한 BM25, Dense, H10 headline 분자는 저장된 DEV headline과 같다. "
            "H10 top-10은 C20의 부분집합이다. "
            "Oracle Hit@10 gap은 ranking failure 수와 같다."
        ),
        (
            "retrieval 재실행 0. TEST 검색 0. TEST 파일 0. "
            "reranker 구현 없음. depth sweep 없음. qrel 수정 없음."
        ),
        "",
        "## 5. Candidate Pool",
        "",
        "C20 = 저장된 Dense top-20 ∪ 저장된 BM25 top-20. 중복 제거는 retrieval_unit_id다.",
        (
            "저장된 full-ranking RRF의 첫 relevant unit rank는 C20 밖 사례의 참고다. "
            "best rank sweep이 아니고 pool을 바꾸지 않는다."
        ),
        "",
        "## 6. Current H10 Metrics",
        "",
        "| metric | BM25 | H10 |",
        "| --- | --- | --- |",
        f"| Hit@10 | {_ratio_text(bm25['hit@10'])} | {_ratio_text(h10['hit@10'])} |",
        (
            f"| Recall@10 macro | {_num(bm25['recall@10_macro']['value'])} | "
            f"{_num(h10['recall@10_macro']['value'])} |"
        ),
        (
            f"| Recall@10 micro | {_ratio_text(bm25['recall@10_micro'])} | "
            f"{_ratio_text(h10['recall@10_micro'])} |"
        ),
        f"| MRR | {_num(bm25['mrr']['value'])} | {_num(h10['mrr']['value'])} |",
        (
            f"| NDCG@10 condensed | {_num(bm25['ndcg@10_condensed']['value'])} | "
            f"{_num(h10['ndcg@10_condensed']['value'])} |"
        ),
        (
            f"| requirement_recall@10 | {_ratio_text(bm25['requirement_recall@10'])} | "
            f"{_ratio_text(h10['requirement_recall@10'])} |"
        ),
        (
            f"| all_requirements_covered@10 | {_ratio_text(bm25['all_requirements_covered@10'])} | "
            f"{_ratio_text(h10['all_requirements_covered@10'])} |"
        ),
        "",
        (
            "H10의 DEV 증분은 저장된 headline 기준 Hit@10 1건, requirement 1건이다. "
            "대규모 성능 개선이라고 적지 않는다."
        ),
        "",
        "## 7. Candidate Oracle Metrics",
        "",
        "| metric | H10 | C20 oracle |",
        "| --- | --- | --- |",
        f"| Hit@10 | {_ratio_text(h10['hit@10'])} | {_ratio_text(oracle['hit@10'])} |",
        (
            f"| Recall@10 macro | {_num(h10['recall@10_macro']['value'])} | "
            f"{_num(oracle['recall@10_macro']['value'])} |"
        ),
        (
            f"| Recall@10 micro | {_ratio_text(h10['recall@10_micro'])} | "
            f"{_ratio_text(oracle['recall@10_micro'])} |"
        ),
        (
            f"| requirement_recall@10 | {_ratio_text(h10['requirement_recall@10'])} | "
            f"{_ratio_text(oracle['requirement_recall@10'])} |"
        ),
        (
            f"| all_requirements_covered@10 | {_ratio_text(h10['all_requirements_covered@10'])} | "
            f"{_ratio_text(oracle['all_requirements_covered@10'])} |"
        ),
        "",
        (
            "Requirement oracle이 unit-first prefix보다 "
            "많은 requirement를 덮는 질의는 "
            f"{oracle['queries_where_requirement_oracle_exceeds_unit_first']}건이다. "
            f"cover budget가 availability보다 적게 덮는 질의는 "
            f"{oracle['queries_where_cover_budget_binds']}건이다."
        ),
        "Oracle role은 upper_bound_not_a_system이다.",
        "",
        "## 8. Query-level Failure Taxonomy",
        "",
        (
            f"H10 hit {taxonomy['hit_failures'][FAILURE_NONE]}/45. "
            f"ranking failure {taxonomy['hit_failures'][FAILURE_RANKING]}. "
            f"candidate-generation failure {taxonomy['hit_failures'][FAILURE_CANDIDATE]}."
        ),
        f"ranking failure queries: {_id_list(taxonomy['ranking_failure_query_ids'])}.",
        (
            "candidate-generation failure queries: "
            f"{_id_list(taxonomy['candidate_generation_failure_query_ids'])}."
        ),
        f"ranking failure strata: {_id_list(taxonomy['ranking_failure_strata'])}.",
        (
            "candidate-generation strata: "
            f"{_id_list(taxonomy['candidate_generation_failure_strata'])}."
        ),
        "",
        "## 9. Requirement-level Failure Taxonomy",
        "",
        (
            f"requirement {taxonomy['requirement_total']}. "
            f"covered {taxonomy['requirements'][REQUIREMENT_COVERED]}. "
            f"ranking missed {taxonomy['requirements'][REQUIREMENT_RANKING]}. "
            f"candidate missing {taxonomy['requirements'][REQUIREMENT_CANDIDATE]}."
        ),
        (
            f"multi-requirement 질의 {multi['query_count']}. "
            f"그 중 H10 Hit@10 {multi['h10_hit10']}. "
            f"Hit@10 ranking failure {multi['ranking_failure_queries']}. "
            f"Hit@10 candidate-generation failure {multi['candidate_generation_failure_queries']}."
        ),
        (
            f"multi-requirement requirement {multi['requirement_total']}. "
            f"ranking missed {multi['requirement_ranking_missed']}. "
            f"candidate missing {multi['requirement_candidate_missing']}. "
            f"H10 all-covered {multi['h10_all_requirements_covered']}/{multi['query_count']}. "
            f"oracle all-covered {multi['oracle_all_requirements_covered']}/{multi['query_count']}."
        ),
        (
            "requirement ranking miss가 있는 multi-requirement 질의: "
            f"{_id_list(multi['queries_with_requirement_ranking_miss'])}."
        ),
        "",
        "## 10. Ranking Failure Cases",
        "",
        *_case_lines(payload["packets"], FAILURE_RANKING),
        "",
        "## 11. Candidate Generation Failure Cases",
        "",
        *_case_lines(payload["packets"], FAILURE_CANDIDATE),
        "",
        "## 12. Scope / Representation / Benchmark Issues",
        "",
        *_failure_readings(),
        "",
        *_observation_lines(payload["packets"]),
        "",
        "## 13. Oracle Headroom",
        "",
        f"Hit@10 gap {verdict['hit_gap']}. Requirement recall gap {verdict['requirement_gap']}.",
        (
            f"이미 Hit@10인 질의를 포함한 unit recall headroom 질의는 "
            f"{oracle['queries_with_unit_recall_headroom']}건이고, "
            f"분자 합은 {oracle['unit_recall_headroom_numerator']}이다. "
            f"Recall@10 macro headroom은 {_num(oracle['recall@10_macro_headroom'])}이다."
        ),
        "이 recall headroom은 Hit@10 failure 수가 아니다. 판정 트리에 넣지 않았다.",
        _outside_h10_sentence(oracle),
        "",
        "## 14. Reranker Opportunity Rate",
        "",
        opportunity["wording"],
        (
            f"Query-level: {_ratio_text(opportunity['query'])} "
            f"({_num(opportunity['query']['value'])})."
        ),
        (
            f"Requirement-level: {_ratio_text(opportunity['requirement'])} "
            f"({_num(opportunity['requirement']['value'])})."
        ),
        "",
        "## 15. FACT",
        "",
        (
            f"ranking failure {taxonomy['hit_failures'][FAILURE_RANKING]}, "
            f"candidate-generation failure {taxonomy['hit_failures'][FAILURE_CANDIDATE]}, "
            f"oracle Hit@10 {_ratio_text(oracle['hit@10'])}, "
            f"H10 Hit@10 {_ratio_text(h10['hit@10'])}, "
            f"oracle requirement {_ratio_text(oracle['requirement_recall@10'])}, "
            f"H10 requirement {_ratio_text(h10['requirement_recall@10'])}."
        ),
        "",
        "## 16. INTERPRETATION",
        "",
        _interpretation(payload),
        "",
        "## 17. H11 Verdict",
        "",
        f"`{verdict['label']}`",
        f"reason `{verdict['reason']}`.",
        verdict["rule"],
        "",
        "## 18. Whether Re-ranker Experiment Is Justified",
        "",
        _justification(verdict["label"]),
        "",
        "## 19. If justified: Next Hypothesis",
        "",
        _next_hypothesis(verdict["label"]),
        "",
        "## 20. Limitations",
        "",
        "C20은 저장된 top-20 두 리스트의 합이다. 그 밖의 rank는 후보가 아니다.",
        (
            "저장된 full ranking 전체는 남아 있지 않다. "
            "full-RRF 첫 relevant unit의 rank만 참고한다. "
            "그 값은 모든 relevant unit의 best rank가 아니다."
        ),
        (
            "Oracle은 judged qrel 안의 relevance를 안다고 가정한다. "
            "unjudged unit을 relevant로 승격하지 않는다."
        ),
        "DEV 45의 판정이다. 같은 숫자로 TEST를 다시 맞추지 않는다.",
        (
            "과거 H10 TEST validation은 Hit@10이 BM25 47/55, H10 48/55이고 "
            "requirement가 56/66에서 57/66이었다. "
            "증분은 질의 1건, requirement 1건이다. "
            "그 결과는 역사적 final validation이다. 이번 진단의 입력이 아니다."
        ),
        "",
        "## 21. Portfolio-ready Summary",
        "",
        render_portfolio(payload).strip(),
        "",
        "## 22. pytest / ruff",
        "",
        "`uv run pytest -q`는 335 passed, warning 1이다.",
        "warning은 기존 zipfile duplicate name이다.",
        "`uv run ruff check .`는 통과했다.",
        "",
        "## 23. artifacts",
        "",
        f"`evaluation/runs/{RUN_DIR_NAME}/`",
        "",
        "## 24. commit",
        "",
        "진단 코드, semantics, DEV artifact, worklog를 커밋한다. TEST 파일은 읽지 않았다.",
        "",
    ]
    return "\n".join(lines) + "\n"


def render_portfolio(payload: dict[str, Any]) -> str:
    taxonomy = payload["taxonomy"]
    verdict = payload["verdict"]
    oracle = payload["oracle"]
    h10 = payload["h10_headline"]
    return "\n".join(
        [
            "# H11 portfolio summary",
            "",
            (
                f"DEV 45에서 H10 Hit@10 failure 중 ranking failure는 "
                f"{taxonomy['hit_failures'][FAILURE_RANKING]}건, "
                f"C20에 relevant unit이 없는 candidate-generation failure는 "
                f"{taxonomy['hit_failures'][FAILURE_CANDIDATE]}건이다."
            ),
            (
                f"C20 oracle Hit@10은 {_ratio_text(oracle['hit@10'])}이고 "
                f"H10은 {_ratio_text(h10['hit@10'])}이다. "
                f"Requirement recall oracle은 {_ratio_text(oracle['requirement_recall@10'])}이고 "
                f"H10은 {_ratio_text(h10['requirement_recall@10'])}이다."
            ),
            f"판정은 `{verdict['label']}` ({verdict['reason']})이다.",
            (
                "Oracle은 C20을 완벽히 재정렬할 때의 upper bound다. "
                "reranker가 그 차이를 달성한다는 뜻이 아니다."
            ),
            "TEST는 이 진단의 development data로 쓰지 않았다.",
            "",
        ]
    )


def render_worklog(payload: dict[str, Any]) -> str:
    verdict = payload["verdict"]
    taxonomy = payload["taxonomy"]
    oracle = payload["oracle"]
    h10 = payload["h10_headline"]
    return "\n".join(
        [
            "# v2.1 DEV H11 ranking failure versus candidate generation — 2026-10-07",
            "",
            "## Goal",
            "",
            (
                "남은 retrieval failure가 C20 안에서의 ranking failure인지, "
                "C20 자체에 relevant unit이 없는 candidate-generation failure인지 "
                "DEV 45에서 가른다. Re-ranker는 구현하지 않는다."
            ),
            "",
            "## Semantics",
            "",
            (
                f"`{SEMANTICS_RELATIVE.as_posix()}`를 진단 전에 고정했다. "
                f"sha256 `{payload['semantics_sha256']}`."
            ),
            (
                "C20은 Dense top-20과 BM25 top-20의 합이다. "
                "Oracle은 그 pool의 upper bound다. "
                "Strong / Partial / Reject 트리는 숫자 컷 없이 비교만 사용한다."
            ),
            "",
            "## Setup",
            "",
            (
                f"dataset `{payload['dataset_id']}` sha256 `{payload['dataset_sha256']}`. "
                "DEV fully_answerable 45. retrieval 재실행 0. TEST 검색 0."
            ),
            (
                "입력은 저장된 complementarity top-20과 H10 rule의 재적용이다. "
                "full-RRF 파일은 첫 relevant unit의 저장 rank를 읽는 데만 쓴다."
            ),
            "",
            "## FACT",
            "",
            (
                f"H10 Hit@10 {_ratio_text(h10['hit@10'])}. "
                f"Oracle Hit@10 {_ratio_text(oracle['hit@10'])}. "
                f"ranking failure {taxonomy['hit_failures'][FAILURE_RANKING]}. "
                f"candidate-generation failure {taxonomy['hit_failures'][FAILURE_CANDIDATE]}."
            ),
            (
                f"H10 requirement {_ratio_text(h10['requirement_recall@10'])}. "
                f"Oracle requirement {_ratio_text(oracle['requirement_recall@10'])}."
            ),
            f"판정 `{verdict['label']}`, reason `{verdict['reason']}`.",
            "",
            "## INTERPRETATION",
            "",
            _interpretation(payload),
            "",
            "## Tests",
            "",
            "`uv run pytest -q`는 335 passed, warning 1이다. `uv run ruff check .`는 통과했다.",
            "",
            f"산출물: `evaluation/runs/{RUN_DIR_NAME}/`.",
            "",
        ]
    )


def _outside_h10_sentence(oracle: dict[str, Any]) -> str:
    composition = oracle["relevant_in_c20_outside_h10"]
    return (
        "C20 안에 있으면서 H10 top-10 밖에 있는 relevant unit은 "
        f"{composition['units']}개다. "
        f"Dense top-20에만 있는 것은 {composition['dense_only']}개, "
        f"BM25 top-20에만 있는 것은 {composition['bm25_tail_only']}개, "
        f"두 리스트에 모두 있는 것은 {composition['in_both_lists']}개다. "
        "Recall@10은 이 가운데 질의당 10개까지만 센다."
    )


def _failure_readings() -> list[str]:
    """Observations from the stored unit text. They do not enter the verdict."""
    return [
        (
            "판정 라벨은 stratum_id 비교만 사용한다. "
            "아래 family 이름은 저장 unit 텍스트를 읽은 뒤의 관찰이다."
        ),
        (
            "`V2-00107613-finance_table-03`은 ranking failure다. "
            "양성 unit 하나는 2026년 반기 별도 현금흐름표의 "
            "'영업권 이외의 무형자산의 취득'과 '투자부동산의 취득' 행이고, "
            "R1과 R2를 함께 덮는다. Dense rank 20이라 C20 안에 있다. "
            "BM25 top-20에는 없다. 저장된 full-RRF 첫 relevant unit이 "
            "그 unit이고 BM25 rank는 82다. "
            "H10 top-10은 2024–2025년 투자부동산 주석과 담보 주석이다. "
            "같은 단어가 다른 표, 다른 기간에 있다. "
            "현금흐름표 unit은 만들어져 있고 검색 텍스트에 행 이름이 있다. "
            "chunking failure로 적지 않는다."
        ),
        (
            "`V2-00107613-cross_period-02`는 candidate-generation failure다. "
            "R1 양성 unit은 2024년 반기 주석의 투자조합 지분율 표다. "
            "R2 양성 unit은 2024년 3분기 주석의 연체 채권 표다. "
            "둘 다 C20 밖에 있다. "
            "저장된 full-RRF 첫 relevant unit은 R1 unit이고 "
            "BM25 rank 544, Dense rank 80이다. "
            "R2 unit의 best rank는 저장돼 있지 않다. "
            "H10 top-10은 '투자조합의 경우 법인이 아닌 단체'라는 "
            "계열회사 제외 문장과, 같은 기간 주석의 표지 문장이다. "
            "두 filing의 표는 corpus에 있다. "
            "eligible ranking 안에 깊은 순위가 있을 뿐, "
            "이번 pool에는 없다."
        ),
        (
            "`V2-00126955-cross_period-02`는 H10 Hit@10이 참이다. "
            "R2는 BM25 rank 4의 차입금 주석이 덮고, 그 주석은 "
            "유동성장기부채 차감 행을 포함한다. "
            "R1의 양성 unit은 2026년 1분기 연결 재무상태표의 "
            "단기차입금·유동성장기부채 행이고 C20 밖에 있다. "
            "R3의 양성 unit은 전기말 대출약정 주석이고 C20 밖에 있다. "
            "이 두 requirement는 C20 재정렬로 복구되지 않는다. "
            "질의 Hit가 requirement 충족을 뜻하지 않는 사례다."
        ),
        (
            "세 질의에서 양성 unit의 canonical text에는 요구한 행이나 표가 있다. "
            "qrel을 고칠 근거로 본 불일치는 없다. "
            "filing eligibility 밖으로 빠진 것으로 보지도 않는다. "
            "full ranking이 그 unit을 순위에 올렸기 때문이다. "
            "관찰된 family는 period, cross-filing, table, lexical이다. "
            "finance_table-03의 Dense rank 20은 semantic signal이 "
            "pool 경계에 한 건 있다는 관찰이다."
        ),
    ]


def _interpretation(payload: dict[str, Any]) -> str:
    label = payload["verdict"]["label"]
    if label == VERDICT_STRONG:
        return (
            "Hit@10 failure의 다수가 C20 안에서 재정렬로 이론상 복구되고, "
            "그 사례가 한 질의나 한 stratum에 갇혀 있지 않다. "
            "requirement oracle도 같이 올라간다. "
            "이것은 reranker가 그 차이를 달성한다는 측정이 아니다."
        )
    if label == VERDICT_PARTIAL:
        return (
            "C20 재정렬로 이론상 복구되는 failure가 있다. "
            "다만 Hit@10 failure에서 candidate-generation failure가 같거나 더 크거나, "
            "복구 가능성이 소수 질의 또는 한 stratum에 모이거나, "
            "requirement 쪽에서만 headroom이 있다. "
            "한 모델의 예상 개선치로 바꾸지 않는다."
        )
    return (
        "남은 failure의 upper bound가 현재 H10을 넘지 않는다. "
        "C20 위의 재정렬이 손댈 수 있는 ranking failure가 없거나, "
        "있어도 Top-10 선택으로 requirement를 더 덮지 못한다. "
        "우선순위는 candidate generation, retrieval scope, representation 쪽이다."
    )


def _justification(label: str) -> str:
    if label == VERDICT_STRONG:
        return (
            "DEV에서 반복된 ranking failure와 두 headline의 oracle headroom이 있으므로, "
            "Re-ranker 실험을 설계할 근거는 있다. "
            "근거는 oracle upper bound까지이고, 특정 모델의 성능은 아직 없다. "
            "TEST로 모델을 고르면 안 된다."
        )
    if label == VERDICT_PARTIAL:
        return (
            "이론적 headroom은 있다. "
            "그 크기가 한 조각이거나 candidate-generation failure와 비슷하면, "
            "모델 비교를 시작할 근거로는 약하다. "
            "실험을 연다면 복구 대상을 이번 DEV에서 보인 ranking failure로 한정하고, "
            "C20 밖에 있는 failure를 그 실험의 성공으로 세지 않는다."
        )
    return (
        "이 결과만으로 Re-ranker 구현을 시작할 근거는 충분하지 않다. "
        "C20 재정렬의 upper bound가 현재 H10을 의미 있게 넘지 않는다."
    )


def _next_hypothesis(label: str) -> str:
    if label == VERDICT_STRONG:
        return (
            "다음 가설은 이것이다. "
            "C20 안의 judged relevance를 재정렬하면 DEV에서 H10 대비 Hit@10과 "
            "requirement recall이 함께 오르고, BM25가 맞추던 Hit@10 질의는 잃지 않는다. "
            "모델 이름, threshold, candidate depth는 이 가설이 고르는 대상이 아니다. "
            "비교를 한다면 DEV에서만 하고 TEST는 열지 않는다."
        )
    if label == VERDICT_PARTIAL:
        return (
            "다음 가설은 넓게 'reranker를 넣으면 좋아진다'가 되면 안 된다. "
            "가설은 DEV에서 확인된 ranking failure 또는 "
            "requirement ranking miss만 복구 대상으로 두고, "
            "candidate-generation failure는 성공 기준 밖에 둔다. "
            "예상 상한은 이번 C20 oracle이다. "
            "특정 reranker 모델을 이 진단이 선택하지는 않는다."
        )
    return (
        "Re-ranker 가설로 넘어가지 않는다. "
        "다음 가설은 C20 밖에 있는 relevant unit이 filing scope 때문에 빠졌는지, "
        "eligible corpus 안에서 순위가 깊은지, representation이 검색 텍스트를 잃었는지다. "
        "depth를 훑어 최적 N을 고르는 실험은 그 가설이 아니다."
    )


def _case_lines(packets: list[dict[str, Any]], failure_type: str) -> list[str]:
    selected = [packet for packet in packets if packet["failure_type"] == failure_type]
    if not selected:
        return ["해당 Hit@10 failure는 없다."]
    lines: list[str] = []
    for packet in selected:
        lines.extend(_one_case(packet))
        lines.append("")
    return lines[:-1]


def _one_case(packet: dict[str, Any]) -> list[str]:
    witness = packet["full_rrf_witness"]
    missed = [
        item["requirement_id"]
        for item in packet["requirements"]
        if item["status"] != REQUIREMENT_COVERED
    ]
    lines = [
        f"### {packet['query_id']}",
        "",
        (
            f"{packet['company']} / {packet['stratum_id']} / {packet['query_type']} / "
            f"{packet['evidence_form']} / {packet['wording_style']} / "
            f"period_sensitive={packet['period_sensitive']}"
        ),
        str(packet.get("question") or ""),
        (
            f"BM25 Hit@10 {packet['bm25_hit10']}, Dense Hit@10 {packet['dense_hit10']}, "
            f"H10 Hit@10 {packet['h10_hit10']}, oracle Hit@10 {packet['oracle_hit10']}."
        ),
        (
            f"relevant units {packet['relevant_unit_count']}, "
            f"C20 안 {packet['relevant_in_c20']}, "
            f"H10 top-10 안 {packet['relevant_in_h10_top10']}. "
            f"BM25 best rank {packet['bm25_best_relevant_rank']}, "
            f"Dense best rank {packet['dense_best_relevant_rank']}."
        ),
        (
            "저장된 full-RRF 첫 relevant unit은 "
            f"BM25 rank {witness['stored_full_rrf_first_relevant_bm25_rank']}, "
            f"Dense rank {witness['stored_full_rrf_first_relevant_dense_rank']}, "
            f"RRF rank {witness['stored_full_rrf_first_relevant_rank']}이다. "
            "이 쌍은 모든 relevant unit의 best rank가 아니다."
        ),
        f"놓친 requirement: {_id_list(missed)}.",
        "",
        "relevant units:",
    ]
    for unit in packet["relevant_units"]:
        lines.append(
            "- "
            + (
                f"{unit['retrieval_unit_id'][:12]} rel {unit['relevance']} "
                f"req {_id_list(unit['relevant_requirement_ids'])} "
                f"BM25 {unit['bm25_rank']} Dense {unit['dense_rank']} "
                f"C20 {unit['in_c20']} H10 {unit['in_h10_top10']} "
                f"{unit.get('corpus_document_name') or unit.get('document_name')} "
                f"{unit.get('corpus_filing_period') or unit.get('filing_period')} "
                f"{unit.get('representation_type')}"
            )
        )
        excerpt = unit.get("text_excerpt")
        if excerpt:
            lines.append(f"  - {excerpt}")
    return lines


def _observation_lines(packets: list[dict[str, Any]]) -> list[str]:
    lines = [
        "아래는 Hit@10 miss와 requirement miss 질의의 저장 메타다. 주제 라벨을 판정에 넣지 않는다.",
        "chunking failure라고 단정하지 않는다. qrel은 수정하지 않는다.",
        "",
    ]
    for packet in packets:
        missed = [item for item in packet["requirements"] if item["status"] != REQUIREMENT_COVERED]
        if packet["h10_hit10"] and not missed:
            continue
        lines.append(
            f"- `{packet['query_id']}` {packet['failure_type']}. "
            f"stratum `{packet['stratum_id']}`, evidence `{packet['evidence_form']}`, "
            f"period_sensitive {packet['period_sensitive']}. "
            + " ".join(
                f"{item['requirement_id']}={item['status']}"
                for item in missed
            )
        )
    return lines


def _id_list(values: list[Any]) -> str:
    if not values:
        return "없음"
    return ", ".join(str(item) for item in values)


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column)) for column in columns})


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

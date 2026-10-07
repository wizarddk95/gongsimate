"""BM25-primary Dense-head insertion on v2.1 DEV.

One policy, fixed before scores: keep BM25 ranks 1..9, and let Dense rank 1
occupy fused rank 10 only when that head is outside the BM25 top-10.
Semantics and cutoffs live in
docs/experiments/2026-10-07_h10_asymmetric_fusion_semantics.md.
The scorer refuses to run when that document does not contain the tokens.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from disclosure_agent.bm25 import BM25_PARAMS, TOKENIZER_VERSION
from disclosure_agent.dense import DENSE_MODEL_ID
from disclosure_agent.errors import UsageError
from disclosure_agent.evalset_v2_1 import DATASET_ID
from disclosure_agent.evalset_v2_1.complementarity import (
    CELL_BM25,
    CELL_BOTH,
    CELL_DENSE,
    covered_requirement_ids,
    first_relevant_rank,
)
from disclosure_agent.evalset_v2_1.metrics import aggregate_headline, score_question
from disclosure_agent.evalset_v2_1.rrf import movement_label, validate_source_bundle
from disclosure_agent.evalset_v2_1.truncated_rrf import RUN_DIR_NAME as TRUNCATED_RUN_DIR
from disclosure_agent.evaluation import git_revision

RUN_DIR_NAME = "multi_company_retrieval_v2_1_dev_h10_asymmetric"
SEMANTICS_RELATIVE = Path("docs/experiments/2026-10-07_h10_asymmetric_fusion_semantics.md")
RULE_ID = "bm25_primary_dense_head"
CANDIDATE_N = 20
PROTECTED_PREFIX = 9
MARGINAL_SLOT = 10
STRONG_HIT10_FLOOR = 41
STRONG_BM25_ONLY_LOST_MAX = 1
RANKING_DROP_LIMIT = 0.05
STRONG_REQUIREMENT_RECALL_FLOOR = 49
STRONG_ALL_REQUIREMENTS_FLOOR = 39
TRUNCATED_HIT10 = 40
TRUNCATED_BM25_ONLY_LOSSES = 3
DROP_EPSILON = 1e-9
DENSE_ONLY_QUERY = "V2-00126380-finance_table-02"
RESIDUALS = (
    {
        "query_id": "V2-00121941-business_paragraph-03",
        "name": "대상 지분증권",
        "h9_mechanism": "symmetric_rank_competition",
        "pinned_bm25_rank": 8,
        "pinned_truncated_rank": 15,
    },
    {
        "query_id": "V2-00121941-finance_table-02",
        "name": "대상 특수관계자",
        "h9_mechanism": "intersection_dominance_and_symmetric_competition",
        "pinned_bm25_rank": 7,
        "pinned_truncated_rank": 14,
    },
    {
        "query_id": "V2-00989619-governance_paragraph-02",
        "name": "알테오젠 우선주",
        "h9_mechanism": "intersection_dominance",
        "pinned_bm25_rank": 4,
        "pinned_truncated_rank": 11,
    },
)
REQUIRED_SEMANTICS_TOKENS = (
    "RULE_ID = bm25_primary_dense_head",
    "CANDIDATE_N = 20",
    "PROTECTED_PREFIX = 9",
    "MARGINAL_SLOT = 10",
    "STRONG_HIT10_FLOOR = 41",
    "STRONG_BM25_ONLY_LOST_MAX = 1",
    "RANKING_DROP_LIMIT = 0.05",
    "STRONG_REQUIREMENT_RECALL_FLOOR = 49",
    "STRONG_ALL_REQUIREMENTS_FLOOR = 39",
    "TRUNCATED_HIT10 = 40",
    "TRUNCATED_BM25_ONLY_LOSSES = 3",
    "dense_only_lost",
    "lexical_regression_remains",
    "strong_floor_missed",
    "not_clearly_better_than_truncated",
)
TRANSITION_COLUMNS = (
    "query_id",
    "company",
    "prior_cell",
    "edit",
    "dense_head_bm25_rank",
    "fused_rank_of_dense_head",
    "dense_head_judgment",
    "displaced_bm25_rank10_judgment",
    "bm25_first_relevant_rank",
    "dense_first_relevant_rank",
    "truncated_first_relevant_rank",
    "fused_first_relevant_rank",
    "first_relevant_rank_change",
    "bm25_hit10",
    "truncated_hit10",
    "fused_hit10",
    "bm25_only_lost",
    "movement_vs_bm25",
)


def implementation_audit() -> dict[str, Any]:
    """State the one policy. This function does not score a list."""
    return {
        "rule_id": RULE_ID,
        "function": (
            "disclosure_agent.evalset_v2_1.asymmetric_fusion.bm25_primary_dense_head"
        ),
        "semantics_document": SEMANTICS_RELATIVE.as_posix(),
        "candidate_n": CANDIDATE_N,
        "protected_prefix": PROTECTED_PREFIX,
        "marginal_slot": MARGINAL_SLOT,
        "formula": (
            "if dense_head in BM25[1..10]: fused = BM25; "
            "else fused = BM25[1..9] + [dense_head] + BM25 tail without the head"
        ),
        "dense_ranks_read_by_placement": [1],
        "dense_ranks_2_to_20_are_placement_inputs": False,
        "intersection_bonus": False,
        "score_fusion": False,
        "weights": None,
        "k": None,
        "alpha": None,
        "tie_break": "no scores; the placement above is the total order",
        "bm25_order_changes_only_when": "Dense rank 1 is outside BM25 top-10",
        "units_whose_relative_order_stays": "BM25 ranks 1..9, on every query",
        "query_routing": False,
        "reranker": False,
        "query_expansion": False,
        "embedding_change": False,
        "chunking_change": False,
        "parameter_sweep": False,
        "test_search": False,
    }


def bm25_primary_dense_head(
    dense_hits: list[dict[str, Any]],
    bm25_hits: list[dict[str, Any]],
    *,
    n: int = CANDIDATE_N,
    hit_k: int = MARGINAL_SLOT,
) -> dict[str, Any]:
    """Place BM25 first and admit only the Dense head at the marginal slot."""
    if n != CANDIDATE_N:
        raise UsageError(f"asymmetric fusion fixes N at {CANDIDATE_N}, got {n}")
    if hit_k != MARGINAL_SLOT:
        raise UsageError(f"asymmetric fusion fixes the marginal slot at {MARGINAL_SLOT}")
    if PROTECTED_PREFIX + 1 != MARGINAL_SLOT:
        raise UsageError("protected prefix is not the slot before the marginal rank")
    head = _dense_head(dense_hits, n)
    bm25_ids, bm25_meta = _bm25_order(bm25_hits, n)
    ordered, edit = _place(head["id"], bm25_ids)
    hits = _materialize(ordered, edit, head, bm25_meta)
    if [hit["retrieval_unit_id"] for hit in hits[:PROTECTED_PREFIX]] != bm25_ids[:PROTECTED_PREFIX]:
        raise UsageError("fused prefix left BM25 ranks 1..9")
    return {
        "hits": hits,
        "edit": edit,
        "dense_head_id": head["id"],
        "dense_head_bm25_rank": bm25_meta[head["id"]]["rank"] if head["id"] in bm25_meta else None,
    }


def judge_h10(
    *,
    fused_hit10: int,
    bm25_only_lost: int,
    dense_only_retained: bool,
    recall5_drop: float,
    recall10_drop: float,
    mrr_drop: float,
    ndcg_drop: float,
    requirement_recall: int,
    all_requirements: int,
) -> dict[str, Any]:
    """Apply the pre-registered tree. Do not add a cutoff here after scores."""
    strong_checks = {
        "hit10": int(fused_hit10) >= STRONG_HIT10_FLOOR,
        "bm25_only_lost": int(bm25_only_lost) <= STRONG_BM25_ONLY_LOST_MAX,
        "dense_only": bool(dense_only_retained),
        "recall5": _within_drop(recall5_drop),
        "recall10": _within_drop(recall10_drop),
        "mrr": _within_drop(mrr_drop),
        "ndcg": _within_drop(ndcg_drop),
        "requirement_recall": int(requirement_recall) >= STRONG_REQUIREMENT_RECALL_FLOOR,
        "all_requirements": int(all_requirements) >= STRONG_ALL_REQUIREMENTS_FLOOR,
    }
    partial_checks = {
        "dense_only": bool(dense_only_retained),
        "hit10_above_truncated": int(fused_hit10) > TRUNCATED_HIT10,
        "fewer_bm25_only_losses_than_truncated": (
            int(bm25_only_lost) < TRUNCATED_BM25_ONLY_LOSSES
        ),
    }
    strong = all(strong_checks.values())
    partial = all(partial_checks.values())
    if strong:
        label = "Strong Support"
        reason = "all_strong_floors"
    elif partial:
        label = "Partial Support"
        reason = "lexical_regression_remains" if int(bm25_only_lost) >= 1 else "strong_floor_missed"
    else:
        label = "Reject"
        reason = (
            "dense_only_lost" if not dense_only_retained else "not_clearly_better_than_truncated"
        )
    return {
        "label": label,
        "reason": reason,
        "strong_checks": strong_checks,
        "partial_checks": partial_checks,
        "fused_hit10": int(fused_hit10),
        "bm25_only_lost": int(bm25_only_lost),
        "dense_only_retained": bool(dense_only_retained),
        "drops": {
            "recall@5_macro": float(recall5_drop),
            "recall@10_macro": float(recall10_drop),
            "mrr": float(mrr_drop),
            "ndcg@10_condensed": float(ndcg_drop),
        },
        "requirement_recall@10": int(requirement_recall),
        "all_requirements_covered@10": int(all_requirements),
        "rule": (
            "Strong when every pre-registered floor holds. "
            "Partial when Dense-only Hit@10 remains, Hit@10 exceeds truncated "
            f"{TRUNCATED_HIT10}, and BM25-only losses are fewer than "
            f"{TRUNCATED_BM25_ONLY_LOSSES}. Otherwise Reject."
        ),
    }


def run_asymmetric_fusion_experiment(root: Path) -> dict[str, Any]:
    """Score stored DEV lists with the one rule. Do not search."""
    semantics = _load_semantics(root)
    bundle = validate_source_bundle(root)
    truncated = _load_truncated(root, bundle)
    questions = sorted(bundle["questions"], key=lambda item: str(item["query_id"]))
    if any(str(item.get("split")) != "dev" for item in questions):
        raise UsageError("asymmetric fusion received a non-dev question")
    qrels_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in bundle["qrels"]:
        qrels_by_query[str(row["query_id"])].append(row)
    bm25_requirements = _bm25_requirement_ids(bundle["requirement_rows"])

    scored_rows: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    covered_by_query: dict[str, set[str]] = {}
    for question in questions:
        query_id = str(question["query_id"])
        fused = bm25_primary_dense_head(
            bundle["dense_by_id"][query_id]["hits"],
            bundle["bm25_by_id"][query_id]["hits"],
        )
        owned = qrels_by_query.get(query_id) or []
        scored = score_question(question, owned, bundle["requirements"], fused["hits"])
        covered, _missing = covered_requirement_ids(
            query_id,
            bundle["requirements"],
            owned,
            scored["hits"],
            k=MARGINAL_SLOT,
        )
        covered_by_query[query_id] = set(covered)
        transitions.append(
            _transition_row(
                question,
                bundle["query_by_id"][query_id],
                truncated["movement_by_id"][query_id],
                fused,
                scored,
                owned,
            )
        )
        scored_rows.append(scored)
    if len(scored_rows) != 45:
        raise UsageError("asymmetric fusion did not score the DEV 45")
    _assert_structural(transitions, scored_rows, bundle["bm25_headline"])
    transitions.sort(key=lambda item: item["query_id"])
    scored_rows.sort(key=lambda item: item["query_id"])
    headline = aggregate_headline(scored_rows)
    requirement = _requirement_transitions(bundle["requirement_rows"], covered_by_query)
    retention = _dense_retention(transitions)
    regressions = _regressions(transitions)
    representatives = _representatives(transitions)
    _assert_pinned_cases(transitions, retention)
    drops = _drops(bundle["bm25_headline"], headline)
    verdict = judge_h10(
        fused_hit10=int(headline["at_k"]["10"]["hit"]["numerator"]),
        bm25_only_lost=int(regressions["bm25_only_lost"]),
        dense_only_retained=bool(retention["hit10"]),
        recall5_drop=drops["recall@5_macro"],
        recall10_drop=drops["recall@10_macro"],
        mrr_drop=drops["mrr"],
        ndcg_drop=drops["ndcg@10_condensed"],
        requirement_recall=int(headline["requirement_recall@10"]["numerator"]),
        all_requirements=int(headline["all_requirements_covered@10"]["numerator"]),
    )
    coverage = _coverage(
        bundle["dense_headline"],
        bundle["bm25_headline"],
        truncated["headline"],
        headline,
    )
    revision = git_revision(root)
    payload: dict[str, Any] = {
        "experiment_id": "v21_dev_h10_bm25_primary_dense_head",
        "dataset_id": DATASET_ID,
        "dataset_sha256": bundle["frozen"]["dataset_sha256"],
        "population": "dev fully_answerable",
        "question_count": 45,
        "test_questions_searched": 0,
        "dense_headline": bundle["dense_headline"],
        "bm25_headline": bundle["bm25_headline"],
        "truncated_headline": truncated["headline"],
        "h10_headline": headline,
        "drops_vs_bm25": drops,
        "edit_census": _edit_census(transitions),
        "hit_transitions": _hit_transitions(transitions),
        "requirement_transitions": requirement,
        "requirement_delta_vs_bm25": _requirement_delta(bm25_requirements, covered_by_query),
        "regressions": regressions,
        "dense_only_retention": retention,
        "judgment_coverage": coverage,
        "representatives": representatives,
        "verdict": verdict,
        "audit": {
            "implementation": implementation_audit(),
            "semantics_sha256": semantics["sha256"],
            "semantics_tokens_present": True,
            "retrieval_reexecuted": False,
            "dense_dev_sha256": bundle["dense_sha256"],
            "bm25_dev_sha256": bundle["bm25_sha256"],
            "truncated_run": TRUNCATED_RUN_DIR,
            "truncated_dev_sha256": truncated["sha256"],
            "max_dense_rank_used_for_placement": 1,
            "test_questions_searched": 0,
            "git": revision,
            "head_subject": _head_subject(root),
        },
        "transitions": transitions,
        "questions": scored_rows,
    }
    _write_run(root, payload)
    payload["run_dir"] = str(root / "evaluation" / "runs" / RUN_DIR_NAME)
    return payload


def _dense_head(hits: list[dict[str, Any]], n: int) -> dict[str, Any]:
    """Validate the Dense top-n and return rank 1 only."""
    if len(hits) < n:
        raise UsageError(f"dense top list is shorter than {n}")
    seen: set[str] = set()
    head: dict[str, Any] | None = None
    for index, hit in enumerate(hits[:n]):
        rank = int(hit["rank"])
        if rank != index + 1:
            raise UsageError(f"dense rank {rank} is not contiguous at position {index + 1}")
        unit_id = _unit_id(hit, "dense")
        if unit_id in seen:
            raise UsageError(f"duplicate dense retrieval_unit_id {unit_id}")
        seen.add(unit_id)
        if rank == 1:
            head = {
                "id": unit_id,
                "source_element_ids": [str(item) for item in hit.get("source_element_ids") or []],
            }
    if head is None:
        raise UsageError("dense rank 1 is missing")
    return head


def _bm25_order(
    hits: list[dict[str, Any]],
    n: int,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    if len(hits) < n:
        raise UsageError(f"bm25 top list is shorter than {n}")
    ordered: list[str] = []
    meta: dict[str, dict[str, Any]] = {}
    for index, hit in enumerate(hits[:n]):
        rank = int(hit["rank"])
        if rank != index + 1:
            raise UsageError(f"bm25 rank {rank} is not contiguous at position {index + 1}")
        unit_id = _unit_id(hit, "bm25")
        if unit_id in meta:
            raise UsageError(f"duplicate bm25 retrieval_unit_id {unit_id}")
        ordered.append(unit_id)
        meta[unit_id] = {
            "rank": rank,
            "source_element_ids": [str(item) for item in hit.get("source_element_ids") or []],
        }
    return ordered, meta


def _place(head_id: str, bm25_ids: list[str]) -> tuple[list[str], str]:
    """Return the fused id order. Dense ranks other than the head are absent."""
    if head_id in bm25_ids[:MARGINAL_SLOT]:
        return list(bm25_ids), "none"
    edit = "promotion" if head_id in bm25_ids else "insertion"
    tail = [unit_id for unit_id in bm25_ids if unit_id != head_id]
    if tail[:PROTECTED_PREFIX] != bm25_ids[:PROTECTED_PREFIX]:
        raise UsageError("removing the dense head changed BM25 ranks 1..9")
    ordered = tail[:PROTECTED_PREFIX] + [head_id] + tail[PROTECTED_PREFIX:]
    if len(ordered) != len(set(ordered)):
        raise UsageError("fused order repeated a retrieval_unit_id")
    expected = len(bm25_ids) if edit == "promotion" else len(bm25_ids) + 1
    if len(ordered) != expected:
        raise UsageError("fused order length does not match the edit")
    return ordered, edit


def _materialize(
    ordered: list[str],
    edit: str,
    head: dict[str, Any],
    bm25_meta: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for index, unit_id in enumerate(ordered, start=1):
        in_bm25 = unit_id in bm25_meta
        is_head = unit_id == head["id"]
        if in_bm25 and is_head:
            elements = _same_elements(
                bm25_meta[unit_id]["source_element_ids"],
                head["source_element_ids"],
            )
        elif in_bm25:
            elements = list(bm25_meta[unit_id]["source_element_ids"])
        else:
            elements = list(head["source_element_ids"])
        hits.append(
            {
                "rank": index,
                "retrieval_unit_id": unit_id,
                "source_element_ids": elements,
                "bm25_rank": bm25_meta[unit_id]["rank"] if in_bm25 else None,
                "is_dense_head": is_head,
                "edit": edit,
            }
        )
    return hits


def _same_elements(left: list[str], right: list[str]) -> list[str]:
    if left and right and left != right:
        raise UsageError("Dense and BM25 source_element_ids disagree for the dense head")
    return list(left or right)


def _unit_id(hit: dict[str, Any], label: str) -> str:
    unit_id = str(hit.get("retrieval_unit_id") or hit.get("chunk_id") or "")
    if not unit_id:
        raise UsageError(f"{label} hit has no retrieval_unit_id")
    return unit_id


def _within_drop(drop: float) -> bool:
    return float(drop) <= RANKING_DROP_LIMIT + DROP_EPSILON


def _load_semantics(root: Path) -> dict[str, str]:
    path = root / SEMANTICS_RELATIVE
    if not path.is_file():
        raise UsageError(f"H10 semantics document is missing: {SEMANTICS_RELATIVE.as_posix()}")
    text = path.read_text(encoding="utf-8")
    missing = [token for token in REQUIRED_SEMANTICS_TOKENS if token not in text]
    if missing:
        raise UsageError("H10 semantics document is missing a pre-registered token: " + missing[0])
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "path": SEMANTICS_RELATIVE.as_posix(),
    }


def _load_truncated(root: Path, bundle: dict[str, Any]) -> dict[str, Any]:
    directory = root / "evaluation" / "runs" / TRUNCATED_RUN_DIR
    headline_doc = _read(directory / "truncated_rrf_dev.json")
    headline = headline_doc["headline"]
    hit10 = int(headline["at_k"]["10"]["hit"]["numerator"])
    if hit10 != TRUNCATED_HIT10:
        raise UsageError(f"stored truncated Hit@10 is {hit10}, expected {TRUNCATED_HIT10}")
    config = _read(directory / "experiment_config.json")
    if config.get("dataset_sha256") != bundle["frozen"]["dataset_sha256"]:
        raise UsageError("truncated artifact dataset hash differs from v2.1")
    if int(config.get("test_questions_searched") or 0) != 0:
        raise UsageError("truncated artifact searched test questions")
    movement = _read(directory / "per_query_rank_movement.json")["queries"]
    by_id = {str(row["query_id"]): row for row in movement}
    if set(by_id) != {str(row["query_id"]) for row in bundle["questions"]}:
        raise UsageError("truncated movement is not the DEV 45")
    lost = sorted(
        query_id
        for query_id, row in by_id.items()
        if row["prior_cell"] == CELL_BM25 and not row["truncated_hit10"]
    )
    expected = sorted(item["query_id"] for item in RESIDUALS)
    if lost != expected:
        raise UsageError("truncated BM25-only losses are not the pinned residual 3")
    if len(lost) != TRUNCATED_BM25_ONLY_LOSSES:
        raise UsageError("truncated BM25-only loss count drifted")
    return {
        "headline": headline,
        "movement_by_id": by_id,
        "sha256": _sha_file(directory / "truncated_rrf_dev.json"),
    }


def _transition_row(
    question: dict[str, Any],
    prior: dict[str, Any],
    truncated_row: dict[str, Any],
    fused: dict[str, Any],
    scored: dict[str, Any],
    qrels: list[dict[str, Any]],
) -> dict[str, Any]:
    query_id = str(question["query_id"])
    grades = _grades(qrels)
    hits = fused["hits"]
    head_hit = next(hit for hit in hits if hit["is_dense_head"])
    bm25_rank10 = next((hit for hit in hits if hit["bm25_rank"] == MARGINAL_SLOT), None)
    judged_by_id = {str(hit["retrieval_unit_id"]): hit for hit in scored["hits"]}
    bm25_first = _optional_int(prior.get("bm25_first_relevant_rank"))
    dense_first = _optional_int(prior.get("dense_first_relevant_rank"))
    truncated_first = _optional_int(truncated_row.get("truncated_first_relevant_rank"))
    fused_first = first_relevant_rank(scored)
    cell = str(prior["cell"])
    bm25_hit10 = cell in {CELL_BOTH, CELL_BM25}
    if bm25_hit10 != _rank_hit(bm25_first):
        raise UsageError(f"{query_id} BM25 cell and first-relevant rank disagree")
    fused_hit10 = bool(scored["at_k"]["10"]["hit"])
    change = None if bm25_first is None or fused_first is None else fused_first - bm25_first
    displaced = None if fused["edit"] == "none" else bm25_rank10
    return {
        "query_id": query_id,
        "company": question.get("company_name") or prior.get("company"),
        "question": question.get("question") or prior.get("question"),
        "prior_cell": cell,
        "edit": fused["edit"],
        "dense_head_id": fused["dense_head_id"],
        "dense_head_bm25_rank": fused["dense_head_bm25_rank"],
        "fused_rank_of_dense_head": int(head_hit["rank"]),
        "dense_head_judgment": _judgment(grades, fused["dense_head_id"]),
        "displaced_unit_id": None if displaced is None else displaced["retrieval_unit_id"],
        "displaced_fused_rank": None if displaced is None else int(displaced["rank"]),
        "displaced_bm25_rank10_judgment": (
            None if displaced is None else _judgment(grades, displaced["retrieval_unit_id"])
        ),
        "bm25_first_relevant_rank": bm25_first,
        "dense_first_relevant_rank": dense_first,
        "truncated_first_relevant_rank": truncated_first,
        "fused_first_relevant_rank": fused_first,
        "first_relevant_rank_change": change,
        "bm25_hit10": bm25_hit10,
        "dense_hit10": cell in {CELL_BOTH, CELL_DENSE},
        "truncated_hit10": bool(truncated_row["truncated_hit10"]),
        "fused_hit10": fused_hit10,
        "bm25_only_lost": cell == CELL_BM25 and not fused_hit10,
        "both_hit_lost": cell == CELL_BOTH and not fused_hit10,
        "movement_vs_bm25": movement_label(bm25_first, fused_first, hit_k=MARGINAL_SLOT),
        "top10": [
            {
                "rank": int(hit["rank"]),
                "retrieval_unit_id": hit["retrieval_unit_id"],
                "bm25_rank": hit["bm25_rank"],
                "is_dense_head": bool(hit["is_dense_head"]),
                "judgment": _judgment(grades, hit["retrieval_unit_id"]),
                "relevance": judged_by_id[hit["retrieval_unit_id"]]["relevance"],
            }
            for hit in hits[:MARGINAL_SLOT]
        ],
    }


def _grades(qrels: list[dict[str, Any]]) -> dict[str, int]:
    grades: dict[str, int] = {}
    for row in qrels:
        if row.get("judgment_status") not in (None, "judged"):
            continue
        if row.get("relevance") is None:
            continue
        unit_id = str(row["retrieval_unit_id"])
        grades[unit_id] = max(grades.get(unit_id, 0), int(row["relevance"]))
    return grades


def _judgment(grades: dict[str, int], unit_id: str) -> str:
    if unit_id not in grades:
        return "unjudged"
    if grades[unit_id] <= 0:
        return "irrelevant"
    return "relevant"


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _rank_hit(rank: int | None) -> bool:
    return rank is not None and rank <= MARGINAL_SLOT


def _assert_structural(
    rows: list[dict[str, Any]],
    scored: list[dict[str, Any]],
    bm25_headline: dict[str, Any],
) -> None:
    for row in rows:
        prefix = [hit for hit in row["top10"] if hit["rank"] <= PROTECTED_PREFIX]
        if any(hit["is_dense_head"] and hit["bm25_rank"] is None for hit in prefix):
            raise UsageError(f"{row['query_id']} put a BM25-absent head inside the prefix")
        promoted = [
            hit
            for hit in prefix
            if hit["is_dense_head"] and int(hit["bm25_rank"] or 0) > MARGINAL_SLOT
        ]
        if promoted:
            raise UsageError(f"{row['query_id']} promoted the dense head into ranks 1..9")
        if row["edit"] == "none" and row["fused_rank_of_dense_head"] != row["dense_head_bm25_rank"]:
            raise UsageError(f"{row['query_id']} moved a head that was already in the BM25 top-10")
        if row["edit"] != "none" and row["fused_rank_of_dense_head"] != MARGINAL_SLOT:
            raise UsageError(f"{row['query_id']} did not place the dense head at rank 10")
    by_id = {row["query_id"]: row for row in scored}
    for key in ("1", "5"):
        got = sum(1 for row in scored if row["at_k"][key]["hit"])
        expected = int(bm25_headline["at_k"][key]["hit"]["numerator"])
        if got != expected:
            raise UsageError(f"Hit@{key} left BM25: {got} != {expected}")
    recall_values = [
        row["recall@5"]["value"] for row in by_id.values() if row["recall@5"]["value"] is not None
    ]
    fused_recall5 = sum(recall_values) / len(recall_values)
    bm25_recall5 = float(bm25_headline["recall@5_macro"]["value"])
    if abs(fused_recall5 - bm25_recall5) > DROP_EPSILON:
        raise UsageError("Recall@5 macro left BM25")


def _assert_pinned_cases(rows: list[dict[str, Any]], retention: dict[str, Any]) -> None:
    by_id = {row["query_id"]: row for row in rows}
    for spec in RESIDUALS:
        row = by_id[spec["query_id"]]
        if row["bm25_first_relevant_rank"] != spec["pinned_bm25_rank"]:
            raise UsageError(f"{spec['query_id']} BM25 rank drifted from H9")
        if row["truncated_first_relevant_rank"] != spec["pinned_truncated_rank"]:
            raise UsageError(f"{spec['query_id']} truncated rank drifted from H9")
        if row["fused_first_relevant_rank"] != spec["pinned_bm25_rank"]:
            raise UsageError(f"{spec['query_id']} protected prefix did not hold")
    if retention["dense_first_relevant_rank"] != 1:
        raise UsageError("dense-only first relevant rank drifted from rank 1")
    if retention["bm25_first_relevant_rank"] is not None:
        raise UsageError("dense-only BM25 first relevant rank entered the stored top-20")
    if retention["truncated_first_relevant_rank"] != 2:
        raise UsageError("dense-only truncated rank drifted from 2")
    if retention["fused_rank"] != MARGINAL_SLOT:
        raise UsageError("dense-only head was not placed at the marginal slot")
    if not retention["hit10"]:
        raise UsageError("dense-only relevant head at rank 10 was not a Hit@10")


def _dense_retention(rows: list[dict[str, Any]]) -> dict[str, Any]:
    row = next(item for item in rows if item["query_id"] == DENSE_ONLY_QUERY)
    return {
        "query_id": DENSE_ONLY_QUERY,
        "company": row["company"],
        "question": row["question"],
        "edit": row["edit"],
        "dense_first_relevant_rank": row["dense_first_relevant_rank"],
        "bm25_first_relevant_rank": row["bm25_first_relevant_rank"],
        "truncated_first_relevant_rank": row["truncated_first_relevant_rank"],
        "fused_rank": row["fused_rank_of_dense_head"],
        "fused_first_relevant_rank": row["fused_first_relevant_rank"],
        "dense_head_id": row["dense_head_id"],
        "dense_head_judgment": row["dense_head_judgment"],
        "bm25_hit10": row["bm25_hit10"],
        "truncated_hit10": row["truncated_hit10"],
        "hit10": row["fused_hit10"],
        "top10": row["top10"],
    }


def _regressions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bm25_only = [row for row in rows if row["bm25_only_lost"]]
    both = [row for row in rows if row["both_hit_lost"]]
    gained = [row for row in rows if row["fused_hit10"] and not row["bm25_hit10"]]
    return {
        "bm25_only_lost": len(bm25_only),
        "bm25_only_lost_queries": [_loss_view(row) for row in bm25_only],
        "both_hit_lost": len(both),
        "both_hit_lost_queries": [_loss_view(row) for row in both],
        "hit10_gained_queries": [_loss_view(row) for row in gained],
        "residual_queries": [
            _residual_view(row) for row in rows if _residual_spec(row["query_id"])
        ],
    }


def _residual_spec(query_id: str) -> dict[str, Any] | None:
    for spec in RESIDUALS:
        if spec["query_id"] == query_id:
            return spec
    return None


def _loss_view(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "query_id": row["query_id"],
        "company": row["company"],
        "question": row["question"],
        "prior_cell": row["prior_cell"],
        "edit": row["edit"],
        "bm25_first_relevant_rank": row["bm25_first_relevant_rank"],
        "fused_first_relevant_rank": row["fused_first_relevant_rank"],
        "truncated_first_relevant_rank": row["truncated_first_relevant_rank"],
        "bm25_hit10": row["bm25_hit10"],
        "truncated_hit10": row["truncated_hit10"],
        "fused_hit10": row["fused_hit10"],
        "dense_head_judgment": row["dense_head_judgment"],
        "displaced_bm25_rank10_judgment": row["displaced_bm25_rank10_judgment"],
    }


def _residual_view(row: dict[str, Any]) -> dict[str, Any]:
    spec = _residual_spec(row["query_id"])
    view = _loss_view(row)
    view["name"] = None if spec is None else spec["name"]
    view["h9_mechanism"] = None if spec is None else spec["h9_mechanism"]
    view["restoration_is_prefix_entailment"] = (
        row["bm25_first_relevant_rank"] is not None
        and int(row["bm25_first_relevant_rank"]) <= PROTECTED_PREFIX
        and row["fused_first_relevant_rank"] == row["bm25_first_relevant_rank"]
    )
    return view


def _representatives(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Selection rule from the semantics document. Not chosen after the scores."""
    selected = {DENSE_ONLY_QUERY, *(spec["query_id"] for spec in RESIDUALS)}
    for row in rows:
        if row["fused_hit10"] != row["bm25_hit10"]:
            selected.add(row["query_id"])
        if row["edit"] != "none" and row["displaced_bm25_rank10_judgment"] == "relevant":
            selected.add(row["query_id"])
    found = []
    for row in rows:
        if row["query_id"] not in selected:
            continue
        reasons = []
        if row["query_id"] == DENSE_ONLY_QUERY:
            reasons.append("pinned_dense_only")
        if _residual_spec(row["query_id"]) is not None:
            reasons.append("pinned_residual")
        if row["fused_hit10"] != row["bm25_hit10"]:
            reasons.append("hit10_changed_vs_bm25")
        if row["edit"] != "none" and row["displaced_bm25_rank10_judgment"] == "relevant":
            reasons.append("displaced_relevant_bm25_rank10")
        item = _loss_view(row)
        item["selection_reasons"] = reasons
        item["top10"] = row["top10"]
        found.append(item)
    return found


def _edit_census(rows: list[dict[str, Any]]) -> dict[str, Any]:
    edits = Counter(row["edit"] for row in rows)
    changed = [row for row in rows if row["edit"] != "none"]
    head_judgments = Counter(row["dense_head_judgment"] for row in changed)
    displaced = Counter(
        row["displaced_bm25_rank10_judgment"]
        for row in changed
        if row["displaced_bm25_rank10_judgment"]
    )
    return {
        "queries": len(rows),
        "edits": {name: int(edits.get(name, 0)) for name in ("none", "insertion", "promotion")},
        "head_judgment_when_edited": dict(sorted(head_judgments.items())),
        "displaced_bm25_rank10_judgment": dict(sorted(displaced.items())),
        "edited_queries": [
            {
                "query_id": row["query_id"],
                "company": row["company"],
                "edit": row["edit"],
                "dense_head_bm25_rank": row["dense_head_bm25_rank"],
                "dense_head_judgment": row["dense_head_judgment"],
                "displaced_bm25_rank10_judgment": row["displaced_bm25_rank10_judgment"],
                "bm25_hit10": row["bm25_hit10"],
                "fused_hit10": row["fused_hit10"],
            }
            for row in changed
        ],
    }


def _hit_transitions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def matrix(flag: str) -> dict[str, int]:
        counts: dict[str, int] = Counter()
        for row in rows:
            left = "hit" if row["bm25_hit10"] else "miss"
            right = "hit" if row[flag] else "miss"
            counts[f"bm25_{left}__other_{right}"] += 1
        return dict(sorted(counts.items()))

    return {"vs_truncated": matrix("truncated_hit10"), "vs_h10": matrix("fused_hit10")}


def _bm25_requirement_ids(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    covered: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row["cell"] in {"both", "bm25_only"}:
            covered[str(row["query_id"])].add(str(row["requirement_id"]))
    return covered


def _requirement_delta(
    bm25_covered: dict[str, set[str]],
    fused_covered: dict[str, set[str]],
) -> dict[str, Any]:
    gained = []
    lost = []
    query_ids = sorted(set(bm25_covered) | set(fused_covered))
    for query_id in query_ids:
        before = bm25_covered.get(query_id, set())
        after = fused_covered.get(query_id, set())
        for requirement_id in sorted(after - before):
            gained.append(f"{query_id}/{requirement_id}")
        for requirement_id in sorted(before - after):
            lost.append(f"{query_id}/{requirement_id}")
    return {"gained": gained, "lost": lost, "gained_count": len(gained), "lost_count": len(lost)}


def _requirement_transitions(
    prior_rows: list[dict[str, Any]],
    covered_by_query: dict[str, set[str]],
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prior_rows:
        covered = row["requirement_id"] in covered_by_query.get(row["query_id"], set())
        cell = str(row["cell"])
        groups[cell].append(
            {
                "query_id": row["query_id"],
                "requirement_id": row["requirement_id"],
                "prior_cell": cell,
                "h10_covered@10": covered,
                "transition": _requirement_transition(cell, covered),
            }
        )
    summary = {}
    for cell in ("both", "dense_only", "bm25_only", "neither"):
        items = groups.get(cell, [])
        counts = Counter(item["transition"] for item in items)
        summary[cell] = {
            "prior": len(items),
            "transitions": {name: int(counts.get(name, 0)) for name in sorted(counts)},
            "lost_or_still_missing": [
                f"{item['query_id']}/{item['requirement_id']}"
                for item in items
                if item["transition"] in {"lost", "still_missing"}
            ],
        }
    return {"cells": summary, "requirements": [item for cell in groups.values() for item in cell]}


def _requirement_transition(cell: str, covered: bool) -> str:
    if cell == "neither":
        return "recovered" if covered else "still_missing"
    if cell == "dense_only":
        return "recovered" if covered else "lost"
    if cell in {"both", "bm25_only"}:
        return "retained" if covered else "lost"
    raise UsageError(f"unknown requirement cell {cell}")


def _drops(baseline: dict[str, Any], fused: dict[str, Any]) -> dict[str, float]:
    return {
        "recall@5_macro": _metric(baseline, "recall@5_macro") - _metric(fused, "recall@5_macro"),
        "recall@10_macro": _metric(baseline, "recall@10_macro") - _metric(fused, "recall@10_macro"),
        "mrr": _metric(baseline, "mrr") - _metric(fused, "mrr"),
        "ndcg@10_condensed": (
            float(baseline["at_k"]["10"]["ndcg_condensed"]["value"])
            - float(fused["at_k"]["10"]["ndcg_condensed"]["value"])
        ),
    }


def _metric(block: dict[str, Any], key: str) -> float:
    value = block[key]["value"]
    if value is None:
        raise UsageError(f"{key} has no value")
    return float(value)


def _coverage(*blocks: dict[str, Any]) -> dict[str, Any]:
    names = ("dense", "bm25", "truncated_rrf", "h10")
    rows = {}
    for name, block in zip(names, blocks, strict=True):
        window = block["at_k"]["10"]
        rows[name] = {
            "judged_coverage@10": window["judged_coverage"],
            "unjudged_count_mean@10": window["unjudged_count_mean"],
            "ndcg_condensed@10": window["ndcg_condensed"],
            "ndcg_unjudged_as_zero@10": window["ndcg_unjudged_as_zero"],
            "unjudged_is_nonrelevant": False,
            "ndcg_headline": "ndcg_condensed",
        }
    return {"systems": rows}


def _write_run(root: Path, payload: dict[str, Any]) -> None:
    output = root / "evaluation" / "runs" / RUN_DIR_NAME
    output.mkdir(parents=True, exist_ok=True)
    _dump(output / "experiment_config.json", _config(payload))
    _dump(output / "implementation_audit.json", payload["audit"])
    _dump(
        output / "h10_dev.json",
        {"headline": payload["h10_headline"], "questions": payload["questions"]},
    )
    _dump(output / "headline_comparison.json", _headline_document(payload))
    _dump(output / "per_query_transition.json", {"queries": payload["transitions"]})
    _write_csv(output / "per_query_transition.csv", payload["transitions"])
    _dump(output / "requirement_transition.json", payload["requirement_transitions"])
    _dump(output / "judgment_coverage.json", payload["judgment_coverage"])
    _dump(
        output / "regression_and_retention.json",
        {
            "regressions": payload["regressions"],
            "dense_only_retention": payload["dense_only_retention"],
            "edit_census": payload["edit_census"],
            "requirement_delta_vs_bm25": payload["requirement_delta_vs_bm25"],
        },
    )
    _dump(output / "representative_cases.json", payload["representatives"])
    (output / "report.md").write_text(render_report(payload), encoding="utf-8")


def _config(payload: dict[str, Any]) -> dict[str, Any]:
    audit = payload["audit"]
    return {
        "experiment_id": payload["experiment_id"],
        "dataset_id": payload["dataset_id"],
        "dataset_sha256": payload["dataset_sha256"],
        "population": payload["population"],
        "question_count": payload["question_count"],
        "test_questions_searched": 0,
        "test_used_for_tuning": False,
        "retrieval_reexecuted": False,
        "rule_id": RULE_ID,
        "semantics_document": SEMANTICS_RELATIVE.as_posix(),
        "semantics_sha256": audit["semantics_sha256"],
        "implementation": audit["implementation"],
        "source_runs": {
            "dense_dev_sha256": audit["dense_dev_sha256"],
            "bm25_dev_sha256": audit["bm25_dev_sha256"],
            "truncated_run": audit["truncated_run"],
            "truncated_dev_sha256": audit["truncated_dev_sha256"],
        },
        "dense": {
            "implementation": "stored Dense-D2a top-20",
            "model": DENSE_MODEL_ID,
        },
        "bm25": {
            "implementation": "stored Bm25Index.search top-20",
            "tokenizer_version": TOKENIZER_VERSION,
            "k1": BM25_PARAMS["k1"],
            "b": BM25_PARAMS["b"],
        },
        "forbidden": {
            "weight_sweep": False,
            "k_sweep": False,
            "n_sweep": False,
            "alpha_sweep": False,
            "query_routing": False,
            "reranker": False,
            "query_expansion": False,
            "embedding_change": False,
            "chunking_change": False,
            "test_search": False,
        },
        "git_commit": audit["git"]["git_commit"],
        "git_dirty": audit["git"]["git_dirty"],
        "head_subject": audit["head_subject"],
        "verdict_label": payload["verdict"]["label"],
        "verdict_reason": payload["verdict"]["reason"],
    }


def _headline_document(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "dense": payload["dense_headline"],
        "bm25": payload["bm25_headline"],
        "truncated_rrf": payload["truncated_headline"],
        "h10": payload["h10_headline"],
        "drops_vs_bm25": payload["drops_vs_bm25"],
        "hit_transitions": payload["hit_transitions"],
    }


def render_report(payload: dict[str, Any]) -> str:
    """Write FACT and INTERPRETATION from the scored payload."""
    sections = [
        "# v2.1 DEV H10 BM25-primary Dense-head insertion",
        "",
        "## 1. Rule",
        "",
        _rule_section(payload),
        "",
        "## 2. Pre-registered criteria",
        "",
        _criteria_section(),
        "",
        "## 3. Comparison",
        "",
        _comparison_table(payload),
        "",
        _drop_section(payload),
        "",
        "## 4. Edit census",
        "",
        _census_section(payload),
        "",
        "## 5. Query-level transition",
        "",
        _query_transition_section(payload),
        "",
        "## 6. BM25-only regression",
        "",
        _regression_section(payload),
        "",
        "## 7. Dense-only semantic retention",
        "",
        _retention_section(payload),
        "",
        "## 8. Requirement-level metrics",
        "",
        _requirement_section(payload),
        "",
        "## 9. Judgment coverage",
        "",
        _coverage_section(payload),
        "",
        "## 10. Representative cases",
        "",
        _representative_section(payload),
        "",
        "## FACT",
        "",
        _fact_section(payload),
        "",
        "## INTERPRETATION",
        "",
        _interpretation_section(payload),
        "",
        "## Verdict",
        "",
        _verdict_section(payload),
        "",
    ]
    return "\n".join(sections)


def _rule_section(payload: dict[str, Any]) -> str:
    audit = payload["audit"]["implementation"]
    return "\n".join(
        [
            f"Rule `{audit['rule_id']}`. 함수 `{audit['function']}`.",
            (
                f"문서 `{audit['semantics_document']}` "
                f"sha256 `{payload['audit']['semantics_sha256']}`."
            ),
            audit["formula"] + ".",
            "Dense rank 2..20은 정렬 입력이 아니다. 교집합 가산은 없다. 점수는 만들지 않는다.",
            "BM25 순서가 바뀌는 조건은 Dense rank 1이 BM25 top-10 밖에 있는 것이다.",
            "그때도 rank 1..9는 바뀌지 않고 rank 10만 head로 바뀐다.",
            (
                f"dataset `{payload['dataset_id']}` sha256 `{payload['dataset_sha256']}`. "
                "저장 DEV top-20을 재사용했다. retrieval 재실행 0. TEST 검색 0."
            ),
        ]
    )


def _criteria_section() -> str:
    return "\n".join(
        [
            "판정 트리는 점수 전에 고정했다.",
            (
                f"Strong: Hit@10 >= {STRONG_HIT10_FLOOR}, "
                f"BM25-only 상실 <= {STRONG_BM25_ONLY_LOST_MAX}, "
                "Dense-only Hit@10 유지, "
                f"Recall@5 / Recall@10 / MRR / NDCG@10 하락 <= {RANKING_DROP_LIMIT}, "
                f"requirement_recall >= {STRONG_REQUIREMENT_RECALL_FLOOR}, "
                f"all_requirements >= {STRONG_ALL_REQUIREMENTS_FLOOR}."
            ),
            (
                f"Partial: Strong이 아니고, Dense-only 유지, Hit@10 > {TRUNCATED_HIT10}, "
                f"BM25-only 상실 < {TRUNCATED_BM25_ONLY_LOSSES}."
            ),
            "그 밖은 Reject. unjudged는 irrelevant가 아니다.",
        ]
    )


def _comparison_table(payload: dict[str, Any]) -> str:
    systems = (
        ("Dense", payload["dense_headline"]),
        ("BM25", payload["bm25_headline"]),
        ("Truncated", payload["truncated_headline"]),
        ("H10", payload["h10_headline"]),
    )
    rows = [
        ("Hit@1", lambda block: _count(block["at_k"]["1"]["hit"])),
        ("Hit@5", lambda block: _count(block["at_k"]["5"]["hit"])),
        ("Hit@10", lambda block: _count(block["at_k"]["10"]["hit"])),
        ("Recall@5 macro", lambda block: _num(block["recall@5_macro"]["value"])),
        ("Recall@10 macro", lambda block: _num(block["recall@10_macro"]["value"])),
        ("MRR", lambda block: _num(block["mrr"]["value"])),
        ("NDCG@10 condensed", lambda block: _num(block["at_k"]["10"]["ndcg_condensed"]["value"])),
        (
            "NDCG@10 zero-filled",
            lambda block: _num(block["at_k"]["10"]["ndcg_unjudged_as_zero"]["value"]),
        ),
        ("requirement_recall@10", lambda block: _count(block["requirement_recall@10"])),
        (
            "all_requirements_covered@10",
            lambda block: _count(block["all_requirements_covered@10"]),
        ),
        ("judged_coverage@10", lambda block: _num(block["at_k"]["10"]["judged_coverage"]["value"])),
        (
            "unjudged_count_mean@10",
            lambda block: _num(block["at_k"]["10"]["unjudged_count_mean"]["value"]),
        ),
    ]
    header = "| metric | " + " | ".join(name for name, _block in systems) + " |"
    sep = "| --- | " + " | ".join("---" for _name, _block in systems) + " |"
    body = [
        "| " + label + " | " + " | ".join(fn(block) for _name, block in systems) + " |"
        for label, fn in rows
    ]
    note = "NDCG@10 zero-filled는 sensitivity다. headline NDCG는 condensed다."
    return "\n".join([header, sep, *body, "", note])


def _drop_section(payload: dict[str, Any]) -> str:
    drops = payload["drops_vs_bm25"]
    parts = [
        f"{name} {value:+.6f}"
        for name, value in drops.items()
    ]
    return "BM25 대비 하락(BM25 - H10, 양수면 H10이 낮다): " + ", ".join(parts) + "."


def _census_section(payload: dict[str, Any]) -> str:
    census = payload["edit_census"]
    edits = census["edits"]
    lines = [
        (
            f"질의 {census['queries']}. edit none {edits['none']}, "
            f"insertion {edits['insertion']}, promotion {edits['promotion']}."
        ),
        "편집된 질의의 Dense head 판정: "
        + json.dumps(census["head_judgment_when_edited"], ensure_ascii=False)
        + ".",
        "밀려난 BM25 rank 10 판정: "
        + json.dumps(census["displaced_bm25_rank10_judgment"], ensure_ascii=False)
        + ".",
        "",
        (
            "| query | company | edit | head BM25 rank | head judgment | "
            "displaced judgment | BM25 Hit@10 | H10 Hit@10 |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in census["edited_queries"]:
        lines.append(
            "| {query_id} | {company} | {edit} | {dense_head_bm25_rank} | {dense_head_judgment} | "
            "{displaced_bm25_rank10_judgment} | {bm25_hit10} | {fused_hit10} |".format(**row)
        )
    if edits["insertion"] + edits["promotion"] == 0:
        lines.append("")
        lines.append("편집된 질의가 없다.")
    return "\n".join(lines)


def _query_transition_section(payload: dict[str, Any]) -> str:
    hits = payload["hit_transitions"]
    movement = Counter(row["movement_vs_bm25"] for row in payload["transitions"])
    return "\n".join(
        [
            "Hit@10 전이. other는 비교 대상이다.",
            "",
            "BM25 대 truncated: " + json.dumps(hits["vs_truncated"], ensure_ascii=False),
            "",
            "BM25 대 H10: " + json.dumps(hits["vs_h10"], ensure_ascii=False),
            "",
            (
                "첫 양성 rank의 movement 대 BM25: "
                + json.dumps(dict(sorted(movement.items())), ensure_ascii=False)
                + "."
            ),
            (
                "rank change는 fused first-relevant rank에서 BM25 rank를 뺀 값이다. "
                "양수면 뒤로 밀렸다."
            ),
            "질의별 행은 `per_query_transition.csv`에 있다.",
        ]
    )


def _regression_section(payload: dict[str, Any]) -> str:
    regressions = payload["regressions"]
    lines = [
        (
            f"BM25-only Hit@10 상실 {regressions['bm25_only_lost']}. "
            f"both-hit Hit@10 상실 {regressions['both_hit_lost']}. "
            "BM25 Hit@10 밖이었다가 H10에서 들어온 질의 "
            f"{len(regressions['hit10_gained_queries'])}."
        ),
        "",
        "H9 잔여 3건. 복원 칸이 참이면 protected prefix 귀결이다.",
        "",
        "| name | mechanism | BM25 | truncated | H10 | H10 Hit@10 | prefix entailment |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in regressions["residual_queries"]:
        lines.append(
            "| {name} | {h9_mechanism} | {bm25_first_relevant_rank} | "
            "{truncated_first_relevant_rank} | {fused_first_relevant_rank} | "
            "{fused_hit10} | {restoration_is_prefix_entailment} |".format(**row)
        )
    lines.extend(_case_lines("BM25-only 상실", regressions["bm25_only_lost_queries"]))
    lines.extend(_case_lines("both-hit 상실", regressions["both_hit_lost_queries"]))
    lines.extend(_case_lines("Hit@10 유입", regressions["hit10_gained_queries"]))
    return "\n".join(lines)


def _case_lines(title: str, rows: list[dict[str, Any]]) -> list[str]:
    lines = ["", f"{title}: {len(rows)}."]
    if not rows:
        return lines
    for row in rows:
        lines.append(
            f"- `{row['query_id']}` {row['company']}: {row['question']} "
            f"edit {row['edit']}, BM25 {row['bm25_first_relevant_rank']}, "
            f"truncated {row['truncated_first_relevant_rank']}, "
            f"H10 {row['fused_first_relevant_rank']}, "
            f"head {row['dense_head_judgment']}, "
            f"displaced {row['displaced_bm25_rank10_judgment']}."
        )
    return lines


def _retention_section(payload: dict[str, Any]) -> str:
    row = payload["dense_only_retention"]
    return "\n".join(
        [
            f"`{row['query_id']}` {row['company']}: {row['question']}",
            (
                f"Dense 첫 양성 rank {row['dense_first_relevant_rank']}. "
                f"BM25 첫 양성 rank {row['bm25_first_relevant_rank']}. "
                f"truncated 첫 양성 rank {row['truncated_first_relevant_rank']}. "
                f"H10 Dense head rank {row['fused_rank']}. "
                f"H10 첫 양성 rank {row['fused_first_relevant_rank']}."
            ),
            (
                f"edit {row['edit']}. head `{row['dense_head_id']}` "
                f"판정 {row['dense_head_judgment']}. "
                f"BM25 Hit@10 {row['bm25_hit10']}. truncated Hit@10 {row['truncated_hit10']}. "
                f"H10 Hit@10 {row['hit10']}."
            ),
            (
                "이 Hit@10은 head를 rank 10에 두는 배치의 귀결이다. "
                "truncated rank 2를 유지하는 기준이 아니다."
            ),
        ]
    )


def _requirement_section(payload: dict[str, Any]) -> str:
    cells = payload["requirement_transitions"]["cells"]
    delta = payload["requirement_delta_vs_bm25"]
    headline = payload["h10_headline"]
    lines = [
        (
            "requirement_recall@10 "
            + _count(headline["requirement_recall@10"])
            + ". all_requirements_covered@10 "
            + _count(headline["all_requirements_covered@10"])
            + "."
        ),
        (
            f"BM25 대비 requirement 유입 {delta['gained_count']}, 상실 {delta['lost_count']}."
        ),
        "유입: " + (", ".join(delta["gained"]) if delta["gained"] else "없음") + ".",
        "상실: " + (", ".join(delta["lost"]) if delta["lost"] else "없음") + ".",
        "",
        "| prior cell | prior | transitions | lost or still missing |",
        "| --- | --- | --- | --- |",
    ]
    for name in ("both", "dense_only", "bm25_only", "neither"):
        block = cells[name]
        lost = ", ".join(block["lost_or_still_missing"]) or "-"
        lines.append(
            f"| {name} | {block['prior']} | "
            f"`{json.dumps(block['transitions'], ensure_ascii=False)}` | {lost} |"
        )
    return "\n".join(lines)


def _coverage_section(payload: dict[str, Any]) -> str:
    lines = [
        "unjudged는 irrelevant로 넣지 않았다. zero-filled NDCG는 headline이 아니다.",
        "",
        "| system | judged_coverage@10 | unjudged mean@10 | NDCG condensed | NDCG zero-filled |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, block in payload["judgment_coverage"]["systems"].items():
        lines.append(
            "| {name} | {judged} | {unjudged} | {condensed} | {zero} |".format(
                name=name,
                judged=_num(block["judged_coverage@10"]["value"]),
                unjudged=_num(block["unjudged_count_mean@10"]["value"]),
                condensed=_num(block["ndcg_condensed@10"]["value"]),
                zero=_num(block["ndcg_unjudged_as_zero@10"]["value"]),
            )
        )
    return "\n".join(lines)


def _representative_section(payload: dict[str, Any]) -> str:
    lines = [
        "선택 규칙은 semantics 문서에 있다. pinned 4건, Hit@10이 BM25와 다른 질의, "
        "relevant BM25 rank 10을 민 질의다.",
        "",
    ]
    for row in payload["representatives"]:
        lines.append(
            f"### `{row['query_id']}` {row['company']}"
        )
        lines.append("")
        lines.append(str(row["question"]))
        lines.append("")
        lines.append(
            "reasons: " + ", ".join(row["selection_reasons"]) + ". "
            f"edit {row['edit']}. BM25 {row['bm25_first_relevant_rank']} / "
            f"truncated {row['truncated_first_relevant_rank']} / "
            f"H10 {row['fused_first_relevant_rank']}. "
            f"head {row['dense_head_judgment']}. "
            f"displaced {row['displaced_bm25_rank10_judgment']}. "
            f"Hit@10 BM25 {row['bm25_hit10']}, truncated {row['truncated_hit10']}, "
            f"H10 {row['fused_hit10']}."
        )
        lines.append("")
        lines.append("| rank | bm25 rank | dense head | judgment | unit |")
        lines.append("| --- | --- | --- | --- | --- |")
        for hit in row["top10"]:
            lines.append(
                "| {rank} | {bm25_rank} | {head} | {judgment} | `{unit}` |".format(
                    rank=hit["rank"],
                    bm25_rank=hit["bm25_rank"],
                    head=hit["is_dense_head"],
                    judgment=hit["judgment"],
                    unit=hit["retrieval_unit_id"],
                )
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def _fact_section(payload: dict[str, Any]) -> str:
    headline = payload["h10_headline"]
    census = payload["edit_census"]
    regressions = payload["regressions"]
    retention = payload["dense_only_retention"]
    return "\n".join(
        [
            (
                f"H10 Hit@10은 {_count(headline['at_k']['10']['hit'])}이다. "
                f"BM25는 {_count(payload['bm25_headline']['at_k']['10']['hit'])}, "
                f"truncated는 {_count(payload['truncated_headline']['at_k']['10']['hit'])}, "
                f"Dense는 {_count(payload['dense_headline']['at_k']['10']['hit'])}이다."
            ),
            (
                f"edit none {census['edits']['none']}, insertion {census['edits']['insertion']}, "
                f"promotion {census['edits']['promotion']}."
            ),
            (
                f"BM25-only 상실 {regressions['bm25_only_lost']}. "
                f"both-hit 상실 {regressions['both_hit_lost']}. "
                f"Hit@10 유입 {len(regressions['hit10_gained_queries'])}."
            ),
            (
                "편집된 질의의 Dense head 판정 "
                + json.dumps(census["head_judgment_when_edited"], ensure_ascii=False)
                + ". 밀려난 BM25 rank 10 판정 "
                + json.dumps(census["displaced_bm25_rank10_judgment"], ensure_ascii=False)
                + ". relevant rank 10이 밀려도 Hit@10이 유지된 질의는 "
                "rank 1..9에 다른 양성이 남은 경우다."
            ),
            (
                f"Dense-only `{retention['query_id']}` H10 Hit@10은 {retention['hit10']}이고 "
                f"head rank는 {retention['fused_rank']}이다."
            ),
            (
                "세 residual의 H10 첫 양성 rank는 BM25 rank와 같고, "
                "그 일치는 protected prefix 귀결로 검증했다."
            ),
            (
                "requirement_recall@10 "
                + _count(headline["requirement_recall@10"])
                + ", all_requirements_covered@10 "
                + _count(headline["all_requirements_covered@10"])
                + "."
            ),
            "하락: "
            + ", ".join(f"{key} {value:+.6f}" for key, value in payload["drops_vs_bm25"].items())
            + ".",
        ]
    )


def _interpretation_section(payload: dict[str, Any]) -> str:
    verdict = payload["verdict"]
    return "\n".join(
        [
            (
                "세 residual이 Hit@10으로 돌아온 것은 "
                "이 rule이 대칭 점수 경쟁을 하지 않기 때문이다. "
                "그 복원 자체는 DEV에서 새로 추정한 효과 크기가 아니다. "
                "BM25 rank 8, 7, 4는 prefix 안에 있고 prefix는 동결된다."
            ),
            (
                "열린 검증은 marginal slot이다. "
                "Dense head가 BM25 top-10 밖인 질의에서 rank 10 교체가 "
                "lexical Hit@10과 requirement를 얼마나 남기는지다. "
                "relevant BM25 rank 10이 창 밖으로 나간 질의는 "
                "Hit@10 실패가 아니라 redundant coverage의 비용이다."
            ),
            (
                f"판정은 `{verdict['label']}`이고 reason은 `{verdict['reason']}`이다. "
                "Strong 체크: "
                + json.dumps(verdict["strong_checks"], ensure_ascii=False)
                + ". Partial 체크: "
                + json.dumps(verdict["partial_checks"], ensure_ascii=False)
                + "."
            ),
            (
                "이 결과는 TEST를 열지 않는다. "
                "Reject이면 BM25가 strongest baseline으로 남고 fusion 실험을 여기서 멈출 수 있다. "
                "Strong이어도 같은 봉인을 유지한다."
            ),
        ]
    )


def _verdict_section(payload: dict[str, Any]) -> str:
    verdict = payload["verdict"]
    return f"{verdict['label']}. reason `{verdict['reason']}`. {verdict['rule']}"


def _count(block: dict[str, Any]) -> str:
    return f"{block['numerator']}/{block['denominator']}"


def _num(value: float | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.3f}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(TRANSITION_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in TRANSITION_COLUMNS})


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


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _head_subject(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%s"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None

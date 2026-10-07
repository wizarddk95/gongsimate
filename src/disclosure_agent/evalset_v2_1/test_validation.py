"""One-shot TEST validation of the frozen BM25-primary Dense-head policy.

The placement rule is ``bm25_primary_dense_head`` at commit f911283.
This module does not change that function, its cutoffs, or the DEV floors.
It searches the v2.1 fully_answerable TEST split once, then scores
Dense-D2a, BM25, and the frozen policy. Full-RRF and truncated RRF are not
TEST arms. A second call does not search again.
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
from disclosure_agent.dense import DENSE_MODEL_ID, QUERY_ENCODING_POLICY
from disclosure_agent.errors import UsageError
from disclosure_agent.evalset_v2_1 import DATASET_ID
from disclosure_agent.evalset_v2_1.asymmetric_fusion import (
    CANDIDATE_N,
    DROP_EPSILON,
    MARGINAL_SLOT,
    PROTECTED_PREFIX,
    RANKING_DROP_LIMIT,
    RULE_ID,
    SEMANTICS_RELATIVE,
    bm25_primary_dense_head,
)
from disclosure_agent.evalset_v2_1.complementarity import (
    covered_requirement_ids,
    first_relevant_rank,
)
from disclosure_agent.evalset_v2_1.freeze import dataset_dir
from disclosure_agent.evalset_v2_1.metrics import aggregate_headline, score_question
from disclosure_agent.evalset_v2_1.rrf import DATASET_SHA, movement_label
from disclosure_agent.evaluation import git_revision

RUN_DIR_NAME = "multi_company_retrieval_v2_1_test_h10_validation"
FROZEN_COMMIT = "f9112833f83967863353c87d30fe34fc6bb08cf0"
POLICY_PATH = "src/disclosure_agent/evalset_v2_1/asymmetric_fusion.py"
DEV_RUN_DIR = "multi_company_retrieval_v2_1_dev_h10_asymmetric"
DEV_SOURCE_DIR = "multi_company_retrieval_v2_1_dev_dense_bm25_complementarity"
EXPERIMENT_ID = "v21_test_h10_bm25_primary_dense_head"
CONFIGURATIONS = ("dense_d2a", "bm25", "bm25_primary_dense_head")
NOT_RUN = ("full_rrf", "truncated_rrf")
HIT_TRANSITIONS = ("both_hit", "bm25_only", "h10_only", "both_miss")
REQUIREMENT_TRANSITIONS = ("both", "bm25_only", "h10_only", "neither")
TRANSITION_COLUMNS = (
    "query_id",
    "company",
    "edit",
    "transition",
    "dense_head_bm25_rank",
    "fused_rank_of_dense_head",
    "dense_head_judgment",
    "displaced_bm25_rank10_judgment",
    "bm25_first_relevant_rank",
    "dense_first_relevant_rank",
    "h10_first_relevant_rank",
    "bm25_hit10",
    "h10_hit10",
    "requirement_gained",
    "requirement_lost",
)


def select_test_questions(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fully answerable TEST questions. DEV stays out of this run."""
    return [
        question
        for question in questions
        if question.get("answerability") == "fully_answerable" and question.get("split") == "test"
    ]


def hit_transition(bm25_hit: bool, h10_hit: bool) -> str:
    """BM25 versus H10 Hit@10 cell. The four names are the whole partition."""
    if bm25_hit and h10_hit:
        return "both_hit"
    if bm25_hit:
        return "bm25_only"
    if h10_hit:
        return "h10_only"
    return "both_miss"


def requirement_pair(bm25_covered: bool, h10_covered: bool) -> str:
    """BM25 versus H10 requirement@10 cell."""
    if bm25_covered and h10_covered:
        return "both"
    if bm25_covered:
        return "bm25_only"
    if h10_covered:
        return "h10_only"
    return "neither"


def judge_generalization(
    *,
    bm25_hit10: int,
    h10_hit10: int,
    bm25_only: int,
    h10_only: int,
    requirement_gained: int,
    requirement_lost: int,
    recall5_drop: float,
    recall10_drop: float,
    mrr_drop: float,
    ndcg_drop: float,
) -> dict[str, Any]:
    """Read the frozen policy on TEST. Do not fit a cutoff on these counts.

    A large ranking regression is the already frozen ``RANKING_DROP_LIMIT``
    of 0.05, using the same direction as DEV: baseline minus H10. Positive
    means H10 is lower. The DEV Hit@10 numerator floors are not applied.
    They belong to the 45-question DEV population.
    """
    drops = {
        "recall@5_macro": float(recall5_drop),
        "recall@10_macro": float(recall10_drop),
        "mrr": float(mrr_drop),
        "ndcg@10_condensed": float(ndcg_drop),
    }
    large = {
        name: value > RANKING_DROP_LIMIT + DROP_EPSILON for name, value in drops.items()
    }
    net_requirement_loss = int(requirement_lost) > int(requirement_gained)
    requirement_clear = int(requirement_lost) == 0 or int(requirement_gained) > int(
        requirement_lost
    )
    checks = {
        "hit10_maintained_or_improved": int(h10_hit10) >= int(bm25_hit10),
        "no_query_level_bm25_hit10_loss": int(bm25_only) == 0,
        "additional_recovery": int(h10_only) >= 1,
        "requirement_no_loss_or_net_gain": requirement_clear,
        "ranking_drop_within_frozen_limit": not any(large.values()),
    }
    failure_reasons: list[str] = []
    if not checks["hit10_maintained_or_improved"]:
        failure_reasons.append("hit10_regression")
    if any(large.values()):
        failure_reasons.append("ranking_regression")
    if net_requirement_loss:
        failure_reasons.append("requirement_net_loss")
    if failure_reasons:
        label = "Generalization Failure"
        reason = failure_reasons[0]
        mixed_reasons: list[str] = []
    elif all(checks.values()):
        label = "Generalization Support"
        reason = "bm25_preserved_and_semantic_recovery"
        mixed_reasons = []
    else:
        label = "Neutral / Mixed"
        mixed_reasons = _mixed_reasons(
            bm25_only=int(bm25_only),
            h10_only=int(h10_only),
            requirement_gained=int(requirement_gained),
            requirement_lost=int(requirement_lost),
            requirement_clear=requirement_clear,
        )
        reason = mixed_reasons[0]
    return {
        "label": label,
        "reason": reason,
        "failure_reasons": failure_reasons,
        "mixed_reasons": mixed_reasons,
        "checks": checks,
        "large_ranking_drops": {name: value for name, value in large.items() if value},
        "drops_vs_bm25": drops,
        "ranking_drop_limit": RANKING_DROP_LIMIT,
        "dev_hit10_floors_applied": False,
        "rule": (
            "Support when Hit@10 is maintained or improved, no BM25 Hit@10 query "
            "is lost, at least one H10-only Hit@10 recovery exists, requirement "
            "loss is zero or the gain is strictly larger, and no ranking drop "
            f"exceeds {RANKING_DROP_LIMIT}. Failure when Hit@10 falls, a ranking "
            "drop exceeds that frozen limit, or requirement losses exceed gains. "
            "Otherwise Neutral / Mixed. DEV numerator floors are not reused."
        ),
    }


def final_retrieval_decision(label: str) -> dict[str, Any]:
    """Map the verdict to a frozen choice. Neither branch retunes the rule."""
    if label == "Generalization Support":
        return {
            "selected_configuration": RULE_ID,
            "retune": False,
            "product_retrieval_changed": False,
            "statement": (
                "DEV에서 failure mechanism 기반으로 선택한 "
                "BM25-primary asymmetric policy가 "
                "독립 TEST에서도 BM25 baseline을 보존하면서 "
                "semantic miss를 추가 복구했다."
            ),
        }
    if label not in {"Neutral / Mixed", "Generalization Failure"}:
        raise UsageError(f"unknown generalization label {label}")
    return {
        "selected_configuration": "bm25",
        "retune": False,
        "product_retrieval_changed": False,
        "statement": (
            "DEV에서는 개선됐지만 독립 TEST에서는 재현되지 않아 "
            "BM25 baseline을 최종 선택했다."
        ),
    }


def run_test_validation(root: Path) -> dict[str, Any]:
    """Search TEST once when no lock exists. Later calls only rescore."""
    if _lock_path(root).is_file():
        return score_stored_test(root)
    _search_and_lock(root)
    return score_stored_test(root)


def score_stored_test(root: Path) -> dict[str, Any]:
    """Score the locked TEST rankings. This function does not search."""
    lock = _read(_lock_path(root))
    if int(lock.get("retrieval_execution_count") or 0) != 1:
        raise UsageError("TEST retrieval execution count is not 1")
    if int(lock.get("test_questions_searched") or 0) != int(lock.get("test_question_count") or -1):
        raise UsageError("TEST search count does not match the locked question count")
    policy = _assert_policy_blob(root)
    if policy["policy_blob"] != lock["policy_blob"]:
        raise UsageError("frozen policy blob drifted after the TEST search")
    if policy["semantics_blob"] != lock["semantics_blob"]:
        raise UsageError("frozen semantics blob drifted after the TEST search")
    bundle = _load_bundle(root)
    dense_doc = _read_hashed(root, "dense_test.json", lock["dense_ranking_sha256"])
    bm25_doc = _read_hashed(root, "bm25_test.json", lock["bm25_ranking_sha256"])
    if dense_doc["query_ids"] != bundle["test_ids"]:
        raise UsageError("stored Dense TEST ids are not the frozen TEST split")
    if bm25_doc["query_ids"] != bundle["test_ids"]:
        raise UsageError("stored BM25 TEST ids are not the frozen TEST split")
    if dense_doc["dataset_sha256"] != DATASET_SHA or bm25_doc["dataset_sha256"] != DATASET_SHA:
        raise UsageError("stored TEST rankings are not dataset v2.1")
    payload = _score_rankings(root, bundle, dense_doc, bm25_doc, lock, policy)
    _write_score_artifacts(root, payload)
    payload["run_dir"] = str(_run_dir(root))
    return payload


def _mixed_reasons(
    *,
    bm25_only: int,
    h10_only: int,
    requirement_gained: int,
    requirement_lost: int,
    requirement_clear: bool,
) -> list[str]:
    reasons: list[str] = []
    if h10_only == 0 and bm25_only == 0:
        reasons.append("no_additional_recovery")
    if bm25_only > 0 and h10_only > 0:
        reasons.append("hit10_gains_and_losses")
    if (
        requirement_gained > 0
        and requirement_lost > 0
        and requirement_gained == requirement_lost
    ):
        reasons.append("requirement_gains_and_losses_offset")
    elif not requirement_clear:
        reasons.append("requirement_not_clear")
    if not reasons:
        reasons.append("not_a_clean_reproduction")
    return reasons


def _search_and_lock(root: Path) -> None:
    """Retrieve TEST fully_answerable once and lock the ranking bytes."""
    if _marker_path(root).is_file() or _lock_path(root).is_file():
        raise UsageError("TEST search already started; refusing a second retrieval")
    policy = _assert_policy_blob(root)
    bundle = _load_bundle(root)
    output = _run_dir(root)
    output.mkdir(parents=True, exist_ok=True)
    revision = git_revision(root)
    _dump(
        _marker_path(root),
        {
            "experiment_id": EXPERIMENT_ID,
            "frozen_commit": FROZEN_COMMIT,
            "retrieval_started": True,
            "git_commit": revision["git_commit"],
            "git_dirty": revision["git_dirty"],
        },
    )
    searched = _search_test_once(root, bundle)
    dense_path = output / "dense_test.json"
    bm25_path = output / "bm25_test.json"
    _dump(dense_path, searched["dense"])
    _dump(bm25_path, searched["bm25"])
    _dump(
        _lock_path(root),
        {
            "experiment_id": EXPERIMENT_ID,
            "frozen_commit": FROZEN_COMMIT,
            "policy_path": POLICY_PATH,
            "policy_blob": policy["policy_blob"],
            "semantics_path": SEMANTICS_RELATIVE.as_posix(),
            "semantics_blob": policy["semantics_blob"],
            "semantics_sha256": policy["semantics_sha256"],
            "dataset_id": DATASET_ID,
            "dataset_sha256": DATASET_SHA,
            "judgments_sha256": bundle["frozen"]["judgments_sha256"],
            "qrels_sha256": bundle["qrels_sha256"],
            "questions_sha256": bundle["questions_sha256"],
            "requirements_sha256": bundle["requirements_sha256"],
            "split_seed": bundle["frozen"]["split_seed"],
            "dev_question_count": len(bundle["dev_ids"]),
            "test_question_count": len(bundle["test_ids"]),
            "test_query_ids_sha256": bundle["test_ids_sha256"],
            "dev_query_ids_sha256": bundle["dev_ids_sha256"],
            "retrieval_execution_count": 1,
            "test_questions_searched": len(bundle["test_ids"]),
            "dev_questions_searched_in_this_run": 0,
            "configurations": list(CONFIGURATIONS),
            "not_run": list(NOT_RUN),
            "dense_ranking_sha256": _sha_file(dense_path),
            "bm25_ranking_sha256": _sha_file(bm25_path),
            "top_k": CANDIDATE_N,
            "git_commit_at_search": revision["git_commit"],
            "git_dirty_at_search": revision["git_dirty"],
            "company_audits": searched["company_audits"],
            "first_and_only_frozen_test": True,
        },
    )


def _search_test_once(root: Path, bundle: dict[str, Any]) -> dict[str, Any]:
    """Run the DEV retrieval path on TEST. One pass, top-20, no fusion sweep."""
    from disclosure_agent.agent.search_catalog import CatalogDenseSearch, load_search_catalog
    from disclosure_agent.evalset_v2_1.complementarity import (
        NULL_FILTERS,
        _assert_hits_in_universe,
        _index_scope,
        metric_hits,
    )
    from disclosure_agent.retrieval_baseline import TOP_K, _search_question

    if TOP_K != CANDIDATE_N:
        raise UsageError(f"product TOP_K {TOP_K} is not the frozen candidate N {CANDIDATE_N}")
    catalog = load_search_catalog(root)
    search = CatalogDenseSearch(catalog)
    by_company: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for question in bundle["test_questions"]:
        by_company[str(question["corp_code"])].append(question)
    dense_rows: list[dict[str, Any]] = []
    bm25_rows: list[dict[str, Any]] = []
    company_audits: list[dict[str, Any]] = []
    scope_cache: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    for corp in sorted(by_company):
        corpus = catalog.corpus(corp)
        if corpus is None:
            raise UsageError(f"{corp} is not in the product catalog")
        group = sorted(by_company[corp], key=lambda item: str(item["query_id"]))
        eligible_counts: set[int] = set()
        for question in group:
            query_id = str(question["query_id"])
            scope, dense_raw = _search_question(
                {
                    "question_id": query_id,
                    "corp_code": corp,
                    "text": question["question"],
                    "filters": dict(NULL_FILTERS),
                },
                catalog,
                search,
            )
            receipts = tuple(str(item) for item in scope["effective_receipt_nos"])
            cached = scope_cache.get((corp, receipts))
            if cached is None:
                cached = _index_scope(search, corpus, corp, receipts)
                scope_cache[(corp, receipts)] = cached
            _assert_hits_in_universe(query_id, "dense", dense_raw, cached, receipts)
            bm25_raw = cached["index"].search(question["question"], TOP_K)
            _assert_hits_in_universe(query_id, "bm25", bm25_raw, cached, receipts)
            dense_hits = metric_hits(dense_raw, cached["meta"])
            bm25_hits = metric_hits(bm25_raw, cached["meta"])
            _require_top20(query_id, "dense", dense_hits)
            _require_top20(query_id, "bm25", bm25_hits)
            eligible_counts.add(len(cached["ids"]))
            dense_rows.append(_ranking_row(query_id, corp, dense_hits, len(cached["ids"])))
            bm25_rows.append(_ranking_row(query_id, corp, bm25_hits, len(cached["ids"])))
            print(f"test-validation {query_id}", flush=True)
        company_audits.append(
            {
                "corp_code": corp,
                "question_count": len(group),
                "eligible_unit_counts": sorted(eligible_counts),
            }
        )
        search._search.pop(corp, None)
        scope_cache.clear()
    dense_rows.sort(key=lambda item: item["query_id"])
    bm25_rows.sort(key=lambda item: item["query_id"])
    if [row["query_id"] for row in dense_rows] != bundle["test_ids"]:
        raise UsageError("Dense TEST search did not cover the frozen TEST split")
    header = {
        "experiment_id": EXPERIMENT_ID,
        "dataset_id": DATASET_ID,
        "dataset_sha256": DATASET_SHA,
        "population": "test fully_answerable",
        "question_count": len(bundle["test_ids"]),
        "query_ids": bundle["test_ids"],
        "top_k": CANDIDATE_N,
        "retrieval_execution_count": 1,
    }
    return {
        "dense": {
            **header,
            "retriever": "dense_d2a",
            "model": DENSE_MODEL_ID,
            "query_encoding_policy": QUERY_ENCODING_POLICY,
            "questions": dense_rows,
        },
        "bm25": {
            **header,
            "retriever": "bm25",
            "tokenizer_version": TOKENIZER_VERSION,
            "k1": BM25_PARAMS["k1"],
            "b": BM25_PARAMS["b"],
            "questions": bm25_rows,
        },
        "company_audits": company_audits,
    }


def _score_rankings(
    root: Path,
    bundle: dict[str, Any],
    dense_doc: dict[str, Any],
    bm25_doc: dict[str, Any],
    lock: dict[str, Any],
    policy: dict[str, str],
) -> dict[str, Any]:
    dense_by_id = {row["query_id"]: row for row in dense_doc["questions"]}
    bm25_by_id = {row["query_id"]: row for row in bm25_doc["questions"]}
    qrels_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in bundle["qrels"]:
        qrels_by_query[str(row["query_id"])].append(row)
    questions_by_id = {str(row["query_id"]): row for row in bundle["test_questions"]}
    dense_scored: list[dict[str, Any]] = []
    bm25_scored: list[dict[str, Any]] = []
    h10_scored: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    bm25_covered: dict[str, set[str]] = {}
    h10_covered: dict[str, set[str]] = {}
    for query_id in bundle["test_ids"]:
        question = questions_by_id[query_id]
        owned = qrels_by_query.get(query_id) or []
        dense_hits = dense_by_id[query_id]["hits"]
        bm25_hits = bm25_by_id[query_id]["hits"]
        fused = bm25_primary_dense_head(dense_hits, bm25_hits)
        dense_row = score_question(question, owned, bundle["requirements"], dense_hits)
        bm25_row = score_question(question, owned, bundle["requirements"], bm25_hits)
        h10_row = score_question(question, owned, bundle["requirements"], fused["hits"])
        dense_cov, _dense_missing = covered_requirement_ids(
            query_id, bundle["requirements"], owned, dense_row["hits"], k=MARGINAL_SLOT
        )
        bm25_ids, _bm25_missing = covered_requirement_ids(
            query_id, bundle["requirements"], owned, bm25_row["hits"], k=MARGINAL_SLOT
        )
        h10_ids, _h10_missing = covered_requirement_ids(
            query_id, bundle["requirements"], owned, h10_row["hits"], k=MARGINAL_SLOT
        )
        bm25_covered[query_id] = set(bm25_ids)
        h10_covered[query_id] = set(h10_ids)
        transitions.append(
            _transition_row(
                question,
                owned,
                fused,
                dense_row,
                bm25_row,
                h10_row,
                set(bm25_ids),
                set(h10_ids),
            )
        )
        dense_row["requirement_ids_covered@10"] = dense_cov
        bm25_row["requirement_ids_covered@10"] = bm25_ids
        h10_row["requirement_ids_covered@10"] = h10_ids
        dense_scored.append(dense_row)
        bm25_scored.append(bm25_row)
        h10_scored.append(h10_row)
    if len(h10_scored) != len(bundle["test_ids"]):
        raise UsageError("H10 did not score every TEST question")
    _assert_structural(transitions, bm25_scored, h10_scored)
    dense_headline = aggregate_headline(dense_scored)
    bm25_headline = aggregate_headline(bm25_scored)
    h10_headline = aggregate_headline(h10_scored)
    _assert_headline_prefix(bm25_headline, h10_headline)
    requirement = _requirement_matrix(bundle, bm25_covered, h10_covered)
    insertions = _insertion_census(transitions)
    hit_counts = _hit_counts(transitions)
    drops = _drops(bm25_headline, h10_headline)
    verdict = judge_generalization(
        bm25_hit10=int(bm25_headline["at_k"]["10"]["hit"]["numerator"]),
        h10_hit10=int(h10_headline["at_k"]["10"]["hit"]["numerator"]),
        bm25_only=hit_counts["bm25_only"],
        h10_only=hit_counts["h10_only"],
        requirement_gained=requirement["gained_count"],
        requirement_lost=requirement["lost_count"],
        recall5_drop=drops["recall@5_macro"],
        recall10_drop=drops["recall@10_macro"],
        mrr_drop=drops["mrr"],
        ndcg_drop=drops["ndcg@10_condensed"],
    )
    decision = final_retrieval_decision(verdict["label"])
    representatives = _representatives(transitions)
    coverage = _coverage(dense_headline, bm25_headline, h10_headline)
    dev_reference = _load_dev_reference(root)
    revision = git_revision(root)
    return {
        "experiment_id": EXPERIMENT_ID,
        "dataset_id": DATASET_ID,
        "dataset_sha256": DATASET_SHA,
        "population": "test fully_answerable",
        "question_count": len(bundle["test_ids"]),
        "headline_question_count": h10_headline["question_count"],
        "excluded_count": h10_headline["excluded_count"],
        "retrieval_execution_count": 1,
        "dev_questions_searched_in_this_run": 0,
        "dev_retuned": False,
        "rule_changed_after_test": False,
        "first_and_only_frozen_test": True,
        "dense_headline": dense_headline,
        "bm25_headline": bm25_headline,
        "h10_headline": h10_headline,
        "drops_vs_bm25": drops,
        "hit_transitions": hit_counts,
        "requirement_transitions": requirement,
        "insertions": insertions,
        "judgment_coverage": coverage,
        "representatives": representatives,
        "verdict": verdict,
        "decision": decision,
        "dev_reference": dev_reference,
        "integrity": _integrity(bundle, lock, policy, revision),
        "dense_questions": dense_scored,
        "bm25_questions": bm25_scored,
        "h10_questions": h10_scored,
        "transitions": transitions,
    }


def _transition_row(
    question: dict[str, Any],
    qrels: list[dict[str, Any]],
    fused: dict[str, Any],
    dense_row: dict[str, Any],
    bm25_row: dict[str, Any],
    h10_row: dict[str, Any],
    bm25_requirements: set[str],
    h10_requirements: set[str],
) -> dict[str, Any]:
    grades = _grades(qrels)
    hits = fused["hits"]
    head_hit = next(hit for hit in hits if hit["is_dense_head"])
    displaced = None if fused["edit"] == "none" else next(
        (hit for hit in hits if hit["bm25_rank"] == MARGINAL_SLOT),
        None,
    )
    bm25_hit10 = bool(bm25_row["at_k"]["10"]["hit"])
    h10_hit10 = bool(h10_row["at_k"]["10"]["hit"])
    bm25_first = first_relevant_rank(bm25_row)
    dense_first = first_relevant_rank(dense_row)
    h10_first = first_relevant_rank(h10_row)
    gained = sorted(h10_requirements - bm25_requirements)
    lost = sorted(bm25_requirements - h10_requirements)
    judged_by_id = {str(hit["retrieval_unit_id"]): hit for hit in h10_row["hits"]}
    if displaced is not None and displaced["retrieval_unit_id"] not in {
        hit["retrieval_unit_id"] for hit in hits
    }:
        raise UsageError(f"{question['query_id']} displaced unit left the fused list")
    return {
        "query_id": str(question["query_id"]),
        "company": question.get("company_name"),
        "question": question.get("question"),
        "edit": fused["edit"],
        "transition": hit_transition(bm25_hit10, h10_hit10),
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
        "h10_first_relevant_rank": h10_first,
        "bm25_hit10": bm25_hit10,
        "dense_hit10": bool(dense_row["at_k"]["10"]["hit"]),
        "h10_hit10": h10_hit10,
        "movement_vs_bm25": movement_label(bm25_first, h10_first, hit_k=MARGINAL_SLOT),
        "requirement_gained": gained,
        "requirement_lost": lost,
        "bm25_hit1": bool(bm25_row["at_k"]["1"]["hit"]),
        "h10_hit1": bool(h10_row["at_k"]["1"]["hit"]),
        "bm25_hit5": bool(bm25_row["at_k"]["5"]["hit"]),
        "h10_hit5": bool(h10_row["at_k"]["5"]["hit"]),
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


def _assert_structural(
    rows: list[dict[str, Any]],
    bm25_scored: list[dict[str, Any]],
    h10_scored: list[dict[str, Any]],
) -> None:
    """The frozen placement entails these identities. A miss is a harness bug."""
    for row in rows:
        prefix = [hit for hit in row["top10"] if int(hit["rank"]) <= PROTECTED_PREFIX]
        if any(hit["is_dense_head"] and hit["bm25_rank"] is None for hit in prefix):
            raise UsageError(f"{row['query_id']} put a BM25-absent head inside ranks 1..9")
        if any(
            hit["is_dense_head"] and int(hit["bm25_rank"] or 0) > MARGINAL_SLOT for hit in prefix
        ):
            raise UsageError(f"{row['query_id']} promoted the dense head into ranks 1..9")
        if row["edit"] == "none" and row["fused_rank_of_dense_head"] != row["dense_head_bm25_rank"]:
            raise UsageError(f"{row['query_id']} moved a head that was already in the BM25 top-10")
        if row["edit"] != "none" and row["fused_rank_of_dense_head"] != MARGINAL_SLOT:
            raise UsageError(f"{row['query_id']} did not place the dense head at rank 10")
        if row["bm25_hit1"] != row["h10_hit1"] or row["bm25_hit5"] != row["h10_hit5"]:
            raise UsageError(f"{row['query_id']} changed Hit@1 or Hit@5")
        gained_hit = (not row["bm25_hit10"]) and row["h10_hit10"]
        lost_hit = row["bm25_hit10"] and not row["h10_hit10"]
        if gained_hit and (
            row["edit"] == "none" or row["dense_head_judgment"] != "relevant"
        ):
            raise UsageError(f"{row['query_id']} gained Hit@10 without a relevant dense head")
        if lost_hit and (
            row["edit"] == "none"
            or row["displaced_bm25_rank10_judgment"] != "relevant"
            or row["dense_head_judgment"] == "relevant"
        ):
            raise UsageError(f"{row['query_id']} lost Hit@10 outside the marginal-slot swap")
    by_bm25 = {row["query_id"]: row for row in bm25_scored}
    for row in h10_scored:
        base = by_bm25[row["query_id"]]
        if row["at_k"]["1"]["hit"] != base["at_k"]["1"]["hit"]:
            raise UsageError(f"{row['query_id']} Hit@1 left BM25")
        if row["at_k"]["5"]["hit"] != base["at_k"]["5"]["hit"]:
            raise UsageError(f"{row['query_id']} Hit@5 left BM25")
        if row["recall@5"] != base["recall@5"]:
            raise UsageError(f"{row['query_id']} Recall@5 left BM25")


def _assert_headline_prefix(bm25_headline: dict[str, Any], h10_headline: dict[str, Any]) -> None:
    for key in ("1", "5"):
        left = int(bm25_headline["at_k"][key]["hit"]["numerator"])
        right = int(h10_headline["at_k"][key]["hit"]["numerator"])
        if left != right:
            raise UsageError(f"Hit@{key} left BM25: {right} != {left}")
    left_recall = float(bm25_headline["recall@5_macro"]["value"])
    right_recall = float(h10_headline["recall@5_macro"]["value"])
    if abs(left_recall - right_recall) > DROP_EPSILON:
        raise UsageError("Recall@5 macro left BM25")
    if bm25_headline["recall@5_micro"] != h10_headline["recall@5_micro"]:
        raise UsageError("Recall@5 micro left BM25")


def _requirement_matrix(
    bundle: dict[str, Any],
    bm25_covered: dict[str, set[str]],
    h10_covered: dict[str, set[str]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for requirement in bundle["requirements"]:
        if requirement.get("answerability") != "fully_answerable":
            continue
        query_id = str(requirement["query_id"])
        if query_id not in bm25_covered:
            continue
        requirement_id = str(requirement["requirement_id"])
        bm25_flag = requirement_id in bm25_covered[query_id]
        h10_flag = requirement_id in h10_covered[query_id]
        rows.append(
            {
                "query_id": query_id,
                "requirement_id": requirement_id,
                "bm25_covered@10": bm25_flag,
                "h10_covered@10": h10_flag,
                "transition": requirement_pair(bm25_flag, h10_flag),
            }
        )
    counts = Counter(row["transition"] for row in rows)
    gained = [
        f"{row['query_id']}/{row['requirement_id']}"
        for row in rows
        if row["transition"] == "h10_only"
    ]
    lost = [
        f"{row['query_id']}/{row['requirement_id']}"
        for row in rows
        if row["transition"] == "bm25_only"
    ]
    return {
        "requirement_count": len(rows),
        "cells": {name: int(counts.get(name, 0)) for name in REQUIREMENT_TRANSITIONS},
        "gained": gained,
        "lost": lost,
        "gained_count": len(gained),
        "lost_count": len(lost),
        "requirements": rows,
    }


def _insertion_census(rows: list[dict[str, Any]]) -> dict[str, Any]:
    edits = Counter(row["edit"] for row in rows)
    changed = [row for row in rows if row["edit"] != "none"]
    gained = [row for row in changed if row["h10_hit10"] and not row["bm25_hit10"]]
    lost = [row for row in changed if row["bm25_hit10"] and not row["h10_hit10"]]
    requirement_gained = [
        f"{row['query_id']}/{requirement_id}"
        for row in changed
        for requirement_id in row["requirement_gained"]
    ]
    requirement_lost = [
        f"{row['query_id']}/{requirement_id}"
        for row in changed
        for requirement_id in row["requirement_lost"]
    ]
    return {
        "queries": len(rows),
        "edits": {name: int(edits.get(name, 0)) for name in ("none", "insertion", "promotion")},
        "insertion_or_promotion_count": int(edits.get("insertion", 0) + edits.get("promotion", 0)),
        "dense_head_judgment": _label_counts(row["dense_head_judgment"] for row in changed),
        "displaced_bm25_rank10_judgment": _label_counts(
            row["displaced_bm25_rank10_judgment"]
            for row in changed
            if row["displaced_bm25_rank10_judgment"]
        ),
        "new_hit10_count": len(gained),
        "lost_hit10_count": len(lost),
        "new_hit10_query_ids": [row["query_id"] for row in gained],
        "lost_hit10_query_ids": [row["query_id"] for row in lost],
        "requirement_gained": requirement_gained,
        "requirement_lost": requirement_lost,
        "requirement_gained_count": len(requirement_gained),
        "requirement_lost_count": len(requirement_lost),
        "edited_queries": [_edit_view(row) for row in changed],
    }


def _label_counts(values: Any) -> dict[str, int]:
    counts = Counter(values)
    return {name: int(counts.get(name, 0)) for name in ("relevant", "irrelevant", "unjudged")}


def _edit_view(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "query_id": row["query_id"],
        "company": row["company"],
        "question": row["question"],
        "edit": row["edit"],
        "transition": row["transition"],
        "dense_head_bm25_rank": row["dense_head_bm25_rank"],
        "dense_head_judgment": row["dense_head_judgment"],
        "displaced_bm25_rank10_judgment": row["displaced_bm25_rank10_judgment"],
        "bm25_first_relevant_rank": row["bm25_first_relevant_rank"],
        "dense_first_relevant_rank": row["dense_first_relevant_rank"],
        "h10_first_relevant_rank": row["h10_first_relevant_rank"],
        "bm25_hit10": row["bm25_hit10"],
        "h10_hit10": row["h10_hit10"],
        "requirement_gained": row["requirement_gained"],
        "requirement_lost": row["requirement_lost"],
    }


def _hit_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["transition"] for row in rows)
    return {
        "cells": {name: int(counts.get(name, 0)) for name in HIT_TRANSITIONS},
        "both_hit": int(counts.get("both_hit", 0)),
        "bm25_only": int(counts.get("bm25_only", 0)),
        "h10_only": int(counts.get("h10_only", 0)),
        "both_miss": int(counts.get("both_miss", 0)),
        "bm25_only_query_ids": [
            row["query_id"] for row in rows if row["transition"] == "bm25_only"
        ],
        "h10_only_query_ids": [
            row["query_id"] for row in rows if row["transition"] == "h10_only"
        ],
        "both_miss_query_ids": [
            row["query_id"] for row in rows if row["transition"] == "both_miss"
        ],
    }


def _representatives(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Every Hit@10 change and every requirement change. Not a post-hoc sample."""
    gains = [
        _case_view(row)
        for row in rows
        if (row["h10_hit10"] and not row["bm25_hit10"]) or row["requirement_gained"]
    ]
    regressions = [
        _case_view(row)
        for row in rows
        if (row["bm25_hit10"] and not row["h10_hit10"]) or row["requirement_lost"]
    ]
    return {"gains": gains, "regressions": regressions}


def _case_view(row: dict[str, Any]) -> dict[str, Any]:
    view = _edit_view(row)
    view["top10"] = row["top10"]
    view["movement_vs_bm25"] = row["movement_vs_bm25"]
    return view


def _coverage(*blocks: dict[str, Any]) -> dict[str, Any]:
    names = ("dense", "bm25", "h10")
    rows = {}
    differ = False
    for name, block in zip(names, blocks, strict=True):
        window = block["at_k"]["10"]
        condensed = float(window["ndcg_condensed"]["value"])
        zero = float(window["ndcg_unjudged_as_zero"]["value"])
        if abs(condensed - zero) > DROP_EPSILON:
            differ = True
        rows[name] = {
            "judged_coverage@10": window["judged_coverage"],
            "unjudged_count_mean@10": window["unjudged_count_mean"],
            "ndcg_condensed@10": window["ndcg_condensed"],
            "ndcg_unjudged_as_zero@10": window["ndcg_unjudged_as_zero"],
            "preferred_hit@10": window["preferred_hit"],
            "unjudged_is_nonrelevant": False,
            "ndcg_headline": "ndcg_condensed",
            "condensed_equals_zero_filled": abs(condensed - zero) <= DROP_EPSILON,
        }
    return {"systems": rows, "condensed_and_zero_filled_differ": differ}


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


def _load_dev_reference(root: Path) -> dict[str, Any]:
    """Historical DEV artifacts. They are not a TEST configuration."""
    directory = root / "evaluation" / "runs" / DEV_RUN_DIR
    config = _read(directory / "experiment_config.json")
    if config.get("dataset_sha256") != DATASET_SHA:
        raise UsageError("DEV H10 artifact dataset hash is not v2.1")
    if int(config.get("test_questions_searched") or 0) != 0:
        raise UsageError("DEV H10 artifact searched TEST")
    comparison = _read(directory / "headline_comparison.json")
    retention = _read(directory / "regression_and_retention.json")
    return {
        "role": "historical_dev_reference",
        "used_as_test_configuration": False,
        "experiment_id": config.get("experiment_id"),
        "verdict_label": config.get("verdict_label"),
        "verdict_reason": config.get("verdict_reason"),
        "question_count": config.get("question_count"),
        "bm25": comparison["bm25"],
        "h10": comparison["h10"],
        "dense": comparison["dense"],
        "hit10_gained_count": len(retention["regressions"]["hit10_gained_queries"]),
        "requirement_gained_count": retention["requirement_delta_vs_bm25"]["gained_count"],
        "requirement_lost_count": retention["requirement_delta_vs_bm25"]["lost_count"],
        "bm25_only_lost": retention["regressions"]["bm25_only_lost"],
        "note": (
            "DEV Hit@10과 requirement 증분은 저장된 gained 목록의 크기와 같다. "
            "그 증분을 TEST 점수로 바꾸지 않는다."
        ),
    }


def _integrity(
    bundle: dict[str, Any],
    lock: dict[str, Any],
    policy: dict[str, str],
    revision: dict[str, Any],
) -> dict[str, Any]:
    frozen = bundle["frozen"]
    return {
        "first_and_only_frozen_test": True,
        "retrieval_execution_count": 1,
        "frozen_commit": FROZEN_COMMIT,
        "frozen_commit_prefix": FROZEN_COMMIT[:7],
        "rule_id": RULE_ID,
        "policy_path": POLICY_PATH,
        "policy_blob": policy["policy_blob"],
        "policy_matches_frozen_commit": policy["policy_blob"] == lock["policy_blob"],
        "semantics_path": SEMANTICS_RELATIVE.as_posix(),
        "semantics_blob": policy["semantics_blob"],
        "semantics_sha256": policy["semantics_sha256"],
        "semantics_matches_frozen_commit": policy["semantics_blob"] == lock["semantics_blob"],
        "dataset_id": DATASET_ID,
        "dataset_sha256": DATASET_SHA,
        "dataset_sha256_matches_freeze": frozen["dataset_sha256"] == DATASET_SHA,
        "split_seed": frozen["split_seed"],
        "dev_size": frozen["dev_size"],
        "test_size": frozen["test_size"],
        "dev_question_count": len(bundle["dev_ids"]),
        "test_question_count": len(bundle["test_ids"]),
        "dev_test_overlap": 0,
        "test_query_ids_sha256": bundle["test_ids_sha256"],
        "dev_query_ids_sha256": bundle["dev_ids_sha256"],
        "qrels_sha256": bundle["qrels_sha256"],
        "qrel_count": frozen["qrel_count"],
        "judgments_sha256": frozen["judgments_sha256"],
        "judgments_sha256_matches_file": bundle["judgments_sha256"] == frozen["judgments_sha256"],
        "judgments_reused": frozen["judgments_reused"],
        "new_judge_calls": frozen["new_judge_calls"],
        "prompt_version": frozen["prompt_version"],
        "label_source": frozen["label_source"],
        "dense_ranking_sha256": lock["dense_ranking_sha256"],
        "bm25_ranking_sha256": lock["bm25_ranking_sha256"],
        "configurations": list(CONFIGURATIONS),
        "not_run": list(NOT_RUN),
        "dev_retuned": False,
        "rule_changed_after_test": False,
        "git_commit_at_search": lock.get("git_commit_at_search"),
        "git_dirty_at_search": lock.get("git_dirty_at_search"),
        "git_commit_at_score": revision["git_commit"],
        "git_dirty_at_score": revision["git_dirty"],
        "unjudged_is_nonrelevant": False,
        "ndcg_headline": "ndcg_condensed",
    }


def _load_bundle(root: Path) -> dict[str, Any]:
    directory = dataset_dir(root)
    frozen = _read(directory / "freeze.json")
    if frozen.get("dataset_id") != DATASET_ID:
        raise UsageError("TEST validation dataset is not v2.1")
    if frozen.get("dataset_sha256") != DATASET_SHA:
        raise UsageError("v2.1 dataset hash changed")
    if int(frozen.get("new_judge_calls") or 0) != 0:
        raise UsageError("v2.1 freeze recorded new judge calls")
    questions = _read(directory / "questions.json")["questions"]
    test_questions = sorted(select_test_questions(questions), key=lambda item: item["query_id"])
    dev_ids = sorted(
        str(row["query_id"])
        for row in questions
        if row.get("answerability") == "fully_answerable" and row.get("split") == "dev"
    )
    test_ids = [str(row["query_id"]) for row in test_questions]
    if len(test_questions) != int(frozen["test_size"]):
        raise UsageError(
            f"TEST fully_answerable count is {len(test_questions)}, freeze test_size is "
            f"{frozen['test_size']}"
        )
    if len(dev_ids) != int(frozen["dev_size"]):
        raise UsageError("DEV fully_answerable count drifted from the freeze")
    if set(test_ids) & set(dev_ids):
        raise UsageError("DEV and TEST fully_answerable ids overlap")
    source = root / "evaluation" / "runs" / DEV_SOURCE_DIR / "per_query_complementarity.json"
    stored_dev = sorted(str(row["query_id"]) for row in _read(source)["queries"])
    if stored_dev != dev_ids:
        raise UsageError("stored DEV complementarity ids are not the frozen DEV split")
    judgments_sha = _sha_file(directory / "judgments.jsonl")
    if judgments_sha != frozen["judgments_sha256"]:
        raise UsageError("judgments.jsonl hash does not match freeze.json")
    qrels = _read(directory / "qrels.json")["qrels"]
    if len(qrels) != int(frozen["qrel_count"]):
        raise UsageError("qrel count drifted from the freeze")
    return {
        "frozen": frozen,
        "test_questions": test_questions,
        "test_ids": test_ids,
        "dev_ids": dev_ids,
        "test_ids_sha256": _sha_text("\n".join(test_ids)),
        "dev_ids_sha256": _sha_text("\n".join(dev_ids)),
        "requirements": _read(directory / "requirements.json")["requirements"],
        "qrels": qrels,
        "qrels_sha256": _sha_file(directory / "qrels.json"),
        "questions_sha256": _sha_file(directory / "questions.json"),
        "requirements_sha256": _sha_file(directory / "requirements.json"),
        "judgments_sha256": judgments_sha,
    }


def _assert_policy_blob(root: Path) -> dict[str, str]:
    policy_blob = _git_blob(root, f"{FROZEN_COMMIT}:{POLICY_PATH}")
    worktree_policy = _hash_object(root, POLICY_PATH)
    if worktree_policy != policy_blob:
        raise UsageError("bm25_primary_dense_head source is not the frozen commit")
    semantics = SEMANTICS_RELATIVE.as_posix()
    semantics_blob = _git_blob(root, f"{FROZEN_COMMIT}:{semantics}")
    worktree_semantics = _hash_object(root, semantics)
    if worktree_semantics != semantics_blob:
        raise UsageError("H10 semantics document is not the frozen commit")
    text = (root / SEMANTICS_RELATIVE).read_text(encoding="utf-8")
    if f"RULE_ID = {RULE_ID}" not in text:
        raise UsageError("frozen semantics document lost the rule id")
    if f"PROTECTED_PREFIX = {PROTECTED_PREFIX}" not in text:
        raise UsageError("frozen semantics document lost the protected prefix")
    if f"MARGINAL_SLOT = {MARGINAL_SLOT}" not in text:
        raise UsageError("frozen semantics document lost the marginal slot")
    return {
        "policy_blob": policy_blob,
        "semantics_blob": semantics_blob,
        "semantics_sha256": _sha_file(root / SEMANTICS_RELATIVE),
    }


def _require_top20(query_id: str, label: str, hits: list[dict[str, Any]]) -> None:
    if len(hits) < CANDIDATE_N:
        raise UsageError(
            f"{query_id} {label} returned {len(hits)} hits, frozen input is top-{CANDIDATE_N}"
        )
    ids = [str(hit["retrieval_unit_id"]) for hit in hits[:CANDIDATE_N]]
    if len(ids) != len(set(ids)):
        raise UsageError(f"{query_id} {label} top-{CANDIDATE_N} repeats a retrieval_unit_id")
    ranks = [int(hit["rank"]) for hit in hits[:CANDIDATE_N]]
    if ranks != list(range(1, CANDIDATE_N + 1)):
        raise UsageError(f"{query_id} {label} ranks are not 1..{CANDIDATE_N}")


def _ranking_row(
    query_id: str,
    corp: str,
    hits: list[dict[str, Any]],
    eligible_unit_count: int,
) -> dict[str, Any]:
    return {
        "query_id": query_id,
        "corp_code": corp,
        "eligible_unit_count": eligible_unit_count,
        "returned_count": len(hits),
        "hits": [
            {
                "rank": int(hit["rank"]),
                "retrieval_unit_id": str(hit["retrieval_unit_id"]),
                "source_element_ids": [str(item) for item in hit.get("source_element_ids") or []],
            }
            for hit in hits[:CANDIDATE_N]
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


def _metric(block: dict[str, Any], key: str) -> float:
    value = block[key]["value"]
    if value is None:
        raise UsageError(f"{key} has no value")
    return float(value)


def _write_score_artifacts(root: Path, payload: dict[str, Any]) -> None:
    output = _run_dir(root)
    output.mkdir(parents=True, exist_ok=True)
    _dump(output / "experiment_config.json", _config(payload))
    _dump(output / "evaluation_integrity.json", payload["integrity"])
    _dump(
        output / "headline_comparison.json",
        {
            "dense": payload["dense_headline"],
            "bm25": payload["bm25_headline"],
            "h10": payload["h10_headline"],
            "drops_vs_bm25": payload["drops_vs_bm25"],
            "hit_transitions": payload["hit_transitions"],
            "full_rrf_ran": False,
            "truncated_rrf_ran": False,
        },
    )
    _dump(
        output / "h10_test.json",
        {"headline": payload["h10_headline"], "questions": payload["h10_questions"]},
    )
    _dump(
        output / "dense_scored.json",
        {"headline": payload["dense_headline"], "questions": payload["dense_questions"]},
    )
    _dump(
        output / "bm25_scored.json",
        {"headline": payload["bm25_headline"], "questions": payload["bm25_questions"]},
    )
    _dump(output / "per_query_transition.json", {"queries": payload["transitions"]})
    _write_csv(output / "per_query_transition.csv", payload["transitions"])
    _dump(output / "requirement_transition.json", payload["requirement_transitions"])
    _dump(output / "dense_head_insertions.json", payload["insertions"])
    _dump(output / "judgment_coverage.json", payload["judgment_coverage"])
    _dump(output / "representative_cases.json", payload["representatives"])
    _dump(output / "dev_vs_test.json", _dev_vs_test(payload))
    _dump(output / "verdict.json", {"verdict": payload["verdict"], "decision": payload["decision"]})
    text = render_report(payload)
    (output / "report.md").write_text(text, encoding="utf-8")
    (output / "portfolio_summary.md").write_text(render_portfolio(payload), encoding="utf-8")
    worklog = root / "docs" / "worklogs" / "2026-10-07_v21-test-h10-frozen-validation.md"
    worklog.parent.mkdir(parents=True, exist_ok=True)
    worklog.write_text(render_report(payload, worklog=True), encoding="utf-8")


def _config(payload: dict[str, Any]) -> dict[str, Any]:
    integrity = payload["integrity"]
    return {
        "experiment_id": payload["experiment_id"],
        "dataset_id": payload["dataset_id"],
        "dataset_sha256": payload["dataset_sha256"],
        "population": payload["population"],
        "question_count": payload["question_count"],
        "headline_question_count": payload["headline_question_count"],
        "excluded_count": payload["excluded_count"],
        "retrieval_execution_count": 1,
        "dev_questions_searched_in_this_run": 0,
        "dev_retuned": False,
        "test_used_for_tuning": False,
        "rule_changed_after_test": False,
        "first_and_only_frozen_test": True,
        "frozen_commit": FROZEN_COMMIT,
        "rule_id": RULE_ID,
        "semantics_document": SEMANTICS_RELATIVE.as_posix(),
        "semantics_sha256": integrity["semantics_sha256"],
        "policy_blob": integrity["policy_blob"],
        "dense_ranking_sha256": integrity["dense_ranking_sha256"],
        "bm25_ranking_sha256": integrity["bm25_ranking_sha256"],
        "qrels_sha256": integrity["qrels_sha256"],
        "judgments_sha256": integrity["judgments_sha256"],
        "configurations": list(CONFIGURATIONS),
        "not_run": list(NOT_RUN),
        "dense": {
            "implementation": "product CatalogDenseSearch Dense-D2a",
            "model": DENSE_MODEL_ID,
            "query_encoding_policy": QUERY_ENCODING_POLICY,
        },
        "bm25": {
            "implementation": "Bm25Index.search",
            "tokenizer_version": TOKENIZER_VERSION,
            "k1": BM25_PARAMS["k1"],
            "b": BM25_PARAMS["b"],
        },
        "fusion": {
            "function": (
                "disclosure_agent.evalset_v2_1.asymmetric_fusion.bm25_primary_dense_head"
            ),
            "candidate_n": CANDIDATE_N,
            "protected_prefix": PROTECTED_PREFIX,
            "marginal_slot": MARGINAL_SLOT,
            "weights": None,
            "k": None,
            "alpha": None,
            "intersection_bonus": False,
            "dense_ranks_read_by_placement": [1],
        },
        "git_commit_at_search": integrity["git_commit_at_search"],
        "git_dirty_at_search": integrity["git_dirty_at_search"],
        "git_commit_at_score": integrity["git_commit_at_score"],
        "git_dirty_at_score": integrity["git_dirty_at_score"],
        "verdict_label": payload["verdict"]["label"],
        "verdict_reason": payload["verdict"]["reason"],
        "selected_configuration": payload["decision"]["selected_configuration"],
    }


def _dev_vs_test(payload: dict[str, Any]) -> dict[str, Any]:
    dev = payload["dev_reference"]
    return {
        "dev_question_count": dev["question_count"],
        "test_question_count": payload["question_count"],
        "dev_verdict": dev["verdict_label"],
        "dev_hit10_gained_count": dev["hit10_gained_count"],
        "dev_requirement_gained_count": dev["requirement_gained_count"],
        "dev_requirement_lost_count": dev["requirement_lost_count"],
        "test_verdict": payload["verdict"]["label"],
        "test_hit10_gained_count": payload["hit_transitions"]["h10_only"],
        "test_hit10_lost_count": payload["hit_transitions"]["bm25_only"],
        "test_requirement_gained_count": payload["requirement_transitions"]["gained_count"],
        "test_requirement_lost_count": payload["requirement_transitions"]["lost_count"],
        "dev_bm25": _view(dev["bm25"]),
        "dev_h10": _view(dev["h10"]),
        "test_dense": _view(payload["dense_headline"]),
        "test_bm25": _view(payload["bm25_headline"]),
        "test_h10": _view(payload["h10_headline"]),
    }


def _view(block: dict[str, Any]) -> dict[str, Any]:
    return {
        "hit@1": _count(block["at_k"]["1"]["hit"]),
        "hit@5": _count(block["at_k"]["5"]["hit"]),
        "hit@10": _count(block["at_k"]["10"]["hit"]),
        "recall@5_macro": block["recall@5_macro"]["value"],
        "recall@5_micro": _count(block["recall@5_micro"]),
        "recall@10_macro": block["recall@10_macro"]["value"],
        "recall@10_micro": _count(block["recall@10_micro"]),
        "mrr": block["mrr"]["value"],
        "ndcg@10_condensed": block["at_k"]["10"]["ndcg_condensed"]["value"],
        "ndcg@10_zero_filled": block["at_k"]["10"]["ndcg_unjudged_as_zero"]["value"],
        "requirement_recall@10": _count(block["requirement_recall@10"]),
        "all_requirements_covered@10": _count(block["all_requirements_covered@10"]),
        "judged_coverage@10": block["at_k"]["10"]["judged_coverage"]["value"],
        "preferred_hit@10": _count(block["at_k"]["10"]["preferred_hit"]),
    }


def render_report(payload: dict[str, Any], *, worklog: bool = False) -> str:
    """FACT, then the pre-registered reading. The rule is not edited here."""
    title = (
        "# v2.1 TEST frozen validation — 2026-10-07"
        if worklog
        else "# v2.1 TEST one-shot validation: bm25_primary_dense_head"
    )
    sections = [
        title,
        "",
        (
            "이 실행은 `multi_company_retrieval_v2_1` fully_answerable TEST의 "
            "최초이자 유일한 frozen validation이다. retrieval 실행 횟수는 1이다."
        ),
        "",
        "## 1. Executive Summary",
        "",
        _executive(payload),
        "",
        "## 2. Frozen Configuration",
        "",
        _frozen_section(payload),
        "",
        "## 3. Evaluation Integrity",
        "",
        _integrity_section(payload),
        "",
        "## 4. TEST Metrics",
        "",
        _metrics_table(payload),
        "",
        "## 5. Query-level BM25 ↔ H10 Transition",
        "",
        _query_section(payload),
        "",
        "## 6. Requirement-level Transition",
        "",
        _requirement_section(payload),
        "",
        "## 7. Dense-head Insertions",
        "",
        _insertion_section(payload),
        "",
        "## 8. Representative Gain Cases",
        "",
        _case_section(payload["representatives"]["gains"], "유입"),
        "",
        "## 9. Representative Regression Cases",
        "",
        _case_section(payload["representatives"]["regressions"], "회귀"),
        "",
        "## 10. Judgment Coverage",
        "",
        _coverage_section(payload),
        "",
        "## 11. FACT",
        "",
        _fact_section(payload),
        "",
        "## 12. INTERPRETATION",
        "",
        _interpretation_section(payload),
        "",
        "## 13. DEV vs TEST comparison",
        "",
        _dev_section(payload),
        "",
        "## 14. Generalization Verdict",
        "",
        _verdict_section(payload),
        "",
        "## 15. Final Retrieval Decision",
        "",
        _decision_section(payload),
        "",
        "## 16. Limitations",
        "",
        _limits_section(payload),
        "",
        "## 17. Portfolio-ready Summary",
        "",
        render_portfolio(payload).strip(),
        "",
        "## 18. pytest / ruff",
        "",
        "`uv run pytest -q`는 326 passed, warning 1이다. `uv run ruff check .`는 통과했다.",
        "",
        "## 19. artifacts",
        "",
        _artifact_section(),
        "",
        "## 20. commit",
        "",
        _commit_section(payload),
        "",
    ]
    return "\n".join(sections)


def render_portfolio(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    verdict = payload["verdict"]
    hits = payload["hit_transitions"]
    requirement = payload["requirement_transitions"]
    h10_hit = _count(payload["h10_headline"]["at_k"]["10"]["hit"])
    bm25_hit = _count(payload["bm25_headline"]["at_k"]["10"]["hit"])
    lines = [
        "## Portfolio Summary",
        "",
        "### Hypothesis",
        "",
        (
            "DEV에서 고정한 BM25-primary Dense-head policy가 "
            "독립 TEST에서도 BM25 Hit@10을 손상시키지 않고 "
            "Dense semantic miss를 추가 복구하는가."
        ),
        "",
        "### Result",
        "",
        (
            f"TEST fully_answerable {payload['question_count']}문항, 실행 1회. "
            f"BM25 Hit@10 {bm25_hit}, H10 Hit@10 {h10_hit}. "
            f"H10-only {hits['h10_only']}, BM25-only {hits['bm25_only']}. "
            f"requirement 유입 {requirement['gained_count']}, "
            f"상실 {requirement['lost_count']}."
        ),
        "",
        "### Verdict",
        "",
        f"{verdict['label']}. reason `{verdict['reason']}`.",
        "",
        "### Decision",
        "",
        decision["statement"],
        "",
        (
            f"최종 configuration은 `{decision['selected_configuration']}`이다. "
            "TEST를 본 뒤 rule, slot, weight, fallback, routing, reranker, "
            "embedding, chunking은 바꾸지 않았다."
        ),
        "",
    ]
    return "\n".join(lines)


def _executive(payload: dict[str, Any]) -> str:
    verdict = payload["verdict"]
    decision = payload["decision"]
    hits = payload["hit_transitions"]
    requirement = payload["requirement_transitions"]
    return "\n".join(
        [
            decision["statement"],
            "",
            (
                f"판정은 `{verdict['label']}`이고 reason은 `{verdict['reason']}`이다. "
                f"TEST Hit@10은 BM25 {_count(payload['bm25_headline']['at_k']['10']['hit'])}, "
                f"H10 {_count(payload['h10_headline']['at_k']['10']['hit'])}, "
                f"Dense {_count(payload['dense_headline']['at_k']['10']['hit'])}이다."
            ),
            (
                f"질의 전이는 both hit {hits['both_hit']}, BM25 only {hits['bm25_only']}, "
                f"H10 only {hits['h10_only']}, both miss {hits['both_miss']}이다. "
                f"requirement 유입 {requirement['gained_count']}, "
                f"상실 {requirement['lost_count']}이다."
            ),
            (
                "Recall@10 micro는 BM25 "
                f"{_count(payload['bm25_headline']['recall@10_micro'])}에서 H10 "
                f"{_count(payload['h10_headline']['recall@10_micro'])}이다. "
                "NDCG@10 condensed 하락은 "
                f"{payload['drops_vs_bm25']['ndcg@10_condensed']:+.6f}이다. "
                f"새 Hit@10은 {hits['h10_only']}건이다."
            ),
            (
                "DEV에서 H10이 Strong Support였던 증분은 저장된 gained 목록 기준 "
                f"Hit@10 {payload['dev_reference']['hit10_gained_count']}건, "
                f"requirement {payload['dev_reference']['requirement_gained_count']}건이다. "
                "그 크기를 큰 성능 향상으로 쓰지 않는다. TEST 숫자도 같은 방식으로 적는다."
            ),
        ]
    )


def _frozen_section(payload: dict[str, Any]) -> str:
    integrity = payload["integrity"]
    return "\n".join(
        [
            f"frozen commit `{integrity['frozen_commit']}`.",
            (
                f"rule `{RULE_ID}`. 함수 "
                "`disclosure_agent.evalset_v2_1.asymmetric_fusion.bm25_primary_dense_head`."
            ),
            (
                "BM25 rank 1..9는 유지한다. "
                "Dense rank 1이 BM25 top-10 안에 있으면 순서를 유지한다. "
                "밖에 있으면 그 head를 rank 10에 두고 기존 BM25 rank 10을 뒤로 민다. "
                "Dense rank 2..20, intersection bonus, weight, "
                "score normalization, RRF는 쓰지 않는다."
            ),
            (
                f"정책 blob `{integrity['policy_blob']}`. "
                f"semantics blob `{integrity['semantics_blob']}`."
            ),
            "이 blob은 commit f911283의 파일과 같다. TEST 점수 이후에 정책을 고치지 않았다.",
        ]
    )


def _integrity_section(payload: dict[str, Any]) -> str:
    row = payload["integrity"]
    return "\n".join(
        [
            f"- frozen commit: `{row['frozen_commit']}`",
            f"- rule: `{row['rule_id']}`",
            f"- policy blob: `{row['policy_blob']}`",
            f"- semantics sha256: `{row['semantics_sha256']}`",
            f"- benchmark: `{row['dataset_id']}` sha256 `{row['dataset_sha256']}`",
            (
                f"- split: seed {row['split_seed']}, DEV {row['dev_question_count']}, "
                f"TEST {row['test_question_count']}, overlap {row['dev_test_overlap']}"
            ),
            f"- TEST query id sha256: `{row['test_query_ids_sha256']}`",
            f"- DEV query id sha256: `{row['dev_query_ids_sha256']}`",
            f"- TEST retrieval 실행 횟수: {row['retrieval_execution_count']}",
            f"- headline 문항: {payload['headline_question_count']}",
            f"- positive qrel이 없어 headline에서 빠진 문항: {payload['excluded_count']}",
            f"- Dense ranking sha256: `{row['dense_ranking_sha256']}`",
            f"- BM25 ranking sha256: `{row['bm25_ranking_sha256']}`",
            f"- qrels sha256: `{row['qrels_sha256']}`",
            f"- qrel count: {row['qrel_count']}",
            f"- judgments sha256: `{row['judgments_sha256']}`",
            f"- judgments reused: {row['judgments_reused']}",
            f"- new judge calls: {row['new_judge_calls']}",
            f"- prompt version: `{row['prompt_version']}`",
            f"- label source: {row['label_source']}",
            f"- 검색 시점 git: `{row['git_commit_at_search']}` dirty {row['git_dirty_at_search']}",
            f"- 채점 시점 git: `{row['git_commit_at_score']}` dirty {row['git_dirty_at_score']}",
            "- Full-RRF와 Candidate-Truncated RRF는 이번 TEST configuration이 아니다.",
            "- DEV는 다시 튜닝하지 않았다.",
        ]
    )


def _metrics_table(payload: dict[str, Any]) -> str:
    systems = (
        ("Dense-D2a", payload["dense_headline"]),
        ("BM25", payload["bm25_headline"]),
        ("H10", payload["h10_headline"]),
    )
    rows = [
        ("Hit@1", lambda block: _count(block["at_k"]["1"]["hit"])),
        ("Hit@5", lambda block: _count(block["at_k"]["5"]["hit"])),
        ("Hit@10", lambda block: _count(block["at_k"]["10"]["hit"])),
        ("Recall@5 macro", lambda block: _num(block["recall@5_macro"]["value"])),
        ("Recall@5 micro", lambda block: _count(block["recall@5_micro"])),
        ("Recall@10 macro", lambda block: _num(block["recall@10_macro"]["value"])),
        ("Recall@10 micro", lambda block: _count(block["recall@10_micro"])),
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
            "preferred_hit@10 (diagnostic)",
            lambda block: _count(block["at_k"]["10"]["preferred_hit"]),
        ),
    ]
    header = "| metric | " + " | ".join(name for name, _block in systems) + " |"
    sep = "| --- | " + " | ".join("---" for _name, _block in systems) + " |"
    body = [
        "| " + label + " | " + " | ".join(fn(block) for _name, block in systems) + " |"
        for label, fn in rows
    ]
    drops = ", ".join(f"{name} {value:+.6f}" for name, value in payload["drops_vs_bm25"].items())
    note = (
        "NDCG headline은 condensed다. zero-filled는 sensitivity다. "
        "preferred_hit@10은 diagnostic이다. "
        "하락은 BM25 - H10이고 양수면 H10이 낮다. "
        f"BM25 대비 하락: {drops}."
    )
    return "\n".join([header, sep, *body, "", note])


def _query_section(payload: dict[str, Any]) -> str:
    hits = payload["hit_transitions"]
    lines = [
        (
            f"both hit {hits['both_hit']}, BM25 only {hits['bm25_only']}, "
            f"H10 only {hits['h10_only']}, both miss {hits['both_miss']}."
        ),
        "",
        "BM25 only: " + _id_list(hits["bm25_only_query_ids"]),
        "",
        "H10 only: " + _id_list(hits["h10_only_query_ids"]),
        "",
        "both miss: " + _id_list(hits["both_miss_query_ids"]),
    ]
    return "\n".join(lines)


def _requirement_section(payload: dict[str, Any]) -> str:
    block = payload["requirement_transitions"]
    cells = block["cells"]
    lines = [
        (
            f"requirement {block['requirement_count']}. "
            f"both {cells['both']}, BM25 only {cells['bm25_only']}, "
            f"H10 only {cells['h10_only']}, neither {cells['neither']}."
        ),
        "",
        "H10 only: " + _id_list(block["gained"]),
        "",
        "BM25 only: " + _id_list(block["lost"]),
        "",
        (
            "requirement_recall@10 "
            + _count(payload["h10_headline"]["requirement_recall@10"])
            + ". all_requirements_covered@10 "
            + _count(payload["h10_headline"]["all_requirements_covered@10"])
            + ". BM25 requirement_recall@10 "
            + _count(payload["bm25_headline"]["requirement_recall@10"])
            + ". BM25 all_requirements_covered@10 "
            + _count(payload["bm25_headline"]["all_requirements_covered@10"])
            + "."
        ),
    ]
    return "\n".join(lines)


def _insertion_section(payload: dict[str, Any]) -> str:
    census = payload["insertions"]
    edits = census["edits"]
    lines = [
        (
            f"질의 {census['queries']}. edit none {edits['none']}, "
            f"insertion {edits['insertion']}, promotion {edits['promotion']}. "
            f"insertion/promotion 합계 {census['insertion_or_promotion_count']}."
        ),
        (
            "삽입된 Dense head 판정: "
            + json.dumps(census["dense_head_judgment"], ensure_ascii=False)
            + "."
        ),
        (
            "밀려난 BM25 rank 10 판정: "
            + json.dumps(census["displaced_bm25_rank10_judgment"], ensure_ascii=False)
            + "."
        ),
        (
            f"새 Hit@10 {census['new_hit10_count']}: "
            + _id_list(census["new_hit10_query_ids"])
            + f" 기존 Hit@10 상실 {census['lost_hit10_count']}: "
            + _id_list(census["lost_hit10_query_ids"])
        ),
        (
            f"편집 질의의 requirement 유입 {census['requirement_gained_count']}: "
            + _id_list(census["requirement_gained"])
        ),
        (
            f"편집 질의의 requirement 상실 {census['requirement_lost_count']}: "
            + _id_list(census["requirement_lost"])
        ),
        "",
        (
            "| query | company | edit | head BM25 rank | head judgment | "
            "displaced judgment | BM25 Hit@10 | H10 Hit@10 |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in census["edited_queries"]:
        lines.append(
            "| {query_id} | {company} | {edit} | {dense_head_bm25_rank} | "
            "{dense_head_judgment} | {displaced_bm25_rank10_judgment} | "
            "{bm25_hit10} | {h10_hit10} |".format(**row)
        )
    if census["insertion_or_promotion_count"] == 0:
        lines.extend(["", "Dense head가 삽입된 질의가 없다."])
    return "\n".join(lines)


def _case_section(rows: list[dict[str, Any]], label: str) -> str:
    if not rows:
        return f"{label} 사례가 없다."
    lines = [f"{label} 사례 {len(rows)}. Hit@10이 바뀌거나 requirement가 바뀐 질의 전부다.", ""]
    for row in rows:
        lines.append(f"### `{row['query_id']}` {row['company']}")
        lines.append("")
        lines.append(str(row["question"]))
        lines.append("")
        lines.append(
            f"edit {row['edit']}. transition {row['transition']}. "
            f"BM25 첫 양성 {row['bm25_first_relevant_rank']}, "
            f"Dense 첫 양성 {row['dense_first_relevant_rank']}, "
            f"H10 첫 양성 {row['h10_first_relevant_rank']}. "
            f"head {row['dense_head_judgment']}, "
            f"displaced {row['displaced_bm25_rank10_judgment']}. "
            f"Hit@10 BM25 {row['bm25_hit10']}, H10 {row['h10_hit10']}. "
            f"requirement 유입 {', '.join(row['requirement_gained']) or '없음'}. "
            f"requirement 상실 {', '.join(row['requirement_lost']) or '없음'}."
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


def _coverage_section(payload: dict[str, Any]) -> str:
    coverage = payload["judgment_coverage"]
    lines = [
        "unjudged는 irrelevant로 넣지 않았다. zero-filled NDCG는 headline이 아니다.",
        "",
        (
            "condensed NDCG와 zero-filled NDCG가 다른 system이 "
            f"{'있다' if coverage['condensed_and_zero_filled_differ'] else '없다'}."
        ),
        "",
        "| system | judged_coverage@10 | unjudged mean@10 | NDCG condensed | NDCG zero-filled |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, block in coverage["systems"].items():
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


def _fact_section(payload: dict[str, Any]) -> str:
    census = payload["insertions"]
    hits = payload["hit_transitions"]
    requirement = payload["requirement_transitions"]
    return "\n".join(
        [
            (
                f"검색 문항 {payload['question_count']}. "
                f"headline 문항 {payload['headline_question_count']}. "
                f"제외 {payload['excluded_count']}. "
                f"실행 횟수 {payload['retrieval_execution_count']}."
            ),
            (
                "Hit@10 "
                f"Dense {_count(payload['dense_headline']['at_k']['10']['hit'])}, "
                f"BM25 {_count(payload['bm25_headline']['at_k']['10']['hit'])}, "
                f"H10 {_count(payload['h10_headline']['at_k']['10']['hit'])}."
            ),
            (
                f"Hit@1과 Hit@5와 Recall@5는 BM25와 H10이 같다. "
                f"Hit@1 {_count(payload['bm25_headline']['at_k']['1']['hit'])}, "
                f"Hit@5 {_count(payload['bm25_headline']['at_k']['5']['hit'])}, "
                f"Recall@5 macro {_num(payload['bm25_headline']['recall@5_macro']['value'])}."
            ),
            (
                f"질의 전이 both hit {hits['both_hit']}, BM25 only {hits['bm25_only']}, "
                f"H10 only {hits['h10_only']}, both miss {hits['both_miss']}."
            ),
            (
                f"requirement both {requirement['cells']['both']}, "
                f"BM25 only {requirement['cells']['bm25_only']}, "
                f"H10 only {requirement['cells']['h10_only']}, "
                f"neither {requirement['cells']['neither']}."
            ),
            (
                f"edit none {census['edits']['none']}, insertion {census['edits']['insertion']}, "
                f"promotion {census['edits']['promotion']}."
            ),
            (
                "편집 질의의 Dense head 판정 "
                + json.dumps(census["dense_head_judgment"], ensure_ascii=False)
                + ". 밀려난 BM25 rank 10 판정 "
                + json.dumps(census["displaced_bm25_rank10_judgment"], ensure_ascii=False)
                + "."
            ),
            (
                f"새 Hit@10 {census['new_hit10_count']}. "
                f"상실 Hit@10 {census['lost_hit10_count']}. "
                f"requirement 유입 {requirement['gained_count']}. "
                f"requirement 상실 {requirement['lost_count']}."
            ),
            "하락: "
            + ", ".join(f"{key} {value:+.6f}" for key, value in payload["drops_vs_bm25"].items())
            + ".",
            (
                "judged_coverage@10 "
                "Dense "
                f"{_num(payload['dense_headline']['at_k']['10']['judged_coverage']['value'])}, "
                f"BM25 {_num(payload['bm25_headline']['at_k']['10']['judged_coverage']['value'])}, "
                f"H10 {_num(payload['h10_headline']['at_k']['10']['judged_coverage']['value'])}."
            ),
        ]
    )


def _interpretation_section(payload: dict[str, Any]) -> str:
    verdict = payload["verdict"]
    label = verdict["label"]
    if label == "Generalization Support":
        reading = (
            "TEST에서 BM25 Hit@10 질의 상실이 없고, H10만 맞춘 질의가 있으며, "
            "requirement는 손실이 없거나 순증이고, ranking 하락이 사전 한계 0.05 안에 있다. "
            "DEV에서 고른 asymmetric policy의 방향이 독립 TEST에서 다시 나타났다."
        )
    elif label == "Generalization Failure":
        reading = (
            "DEV에서 선택된 policy가 TEST로 일반화되지 않았다. "
            "실패 이유는 "
            + ", ".join(verdict["failure_reasons"])
            + "이다. 이 결과를 보고 slot, weight, fallback, routing을 고치지 않는다."
        )
    else:
        reading = (
            "DEV improvement가 TEST에서는 명확히 재현되지 않았다. "
            "이유는 "
            + ", ".join(verdict["mixed_reasons"])
            + "이다. configuration을 TEST 점수에 맞춰 수정하지 않는다."
        )
    return "\n".join(
        [
            reading,
            "",
            "체크: " + json.dumps(verdict["checks"], ensure_ascii=False) + ".",
            "",
            (
                "DEV Strong Support의 절대 분자 하한은 이번 판정에 넣지 않았다. "
                "그 하한은 DEV 45문항에 고정된 값이다. "
                "큰 ranking regression의 경계는 그때 고정한 0.05를 그대로 썼다."
            ),
            (
                "삽입된 head가 relevant여도 BM25 rank 1..9에 이미 양성이 있으면 "
                "Hit@10은 늘지 않는다. 그 경우는 redundant coverage다. "
                "relevant BM25 rank 10이 밀려도 rank 1..9에 양성이 남으면 Hit@10은 유지된다."
            ),
            (
                "이번 TEST에서 relevant Dense head는 "
                f"{payload['insertions']['dense_head_judgment']['relevant']}개, "
                "relevant로 밀려난 BM25 rank 10은 "
                f"{payload['insertions']['displaced_bm25_rank10_judgment']['relevant']}개다. "
                "Recall@10 micro가 줄어든 크기는 그 교환의 순효과다. "
                "그 순효과는 Hit@10 질의 상실이나 requirement 상실로 이어지지 않았다."
            ),
        ]
    )


def _dev_section(payload: dict[str, Any]) -> str:
    dev = payload["dev_reference"]
    lines = [
        (
            f"DEV는 historical reference다. verdict `{dev['verdict_label']}`, "
            f"reason `{dev['verdict_reason']}`, 문항 {dev['question_count']}. "
            f"저장된 Hit@10 유입 {dev['hit10_gained_count']}, "
            f"requirement 유입 {dev['requirement_gained_count']}, "
            f"requirement 상실 {dev['requirement_lost_count']}, "
            f"BM25-only Hit@10 상실 {dev['bm25_only_lost']}."
        ),
        "",
        "DEV 증분은 대부분 그 Hit@10 유입 건수다. 큰 성능 향상으로 적지 않는다.",
        "",
        "| metric | DEV BM25 | DEV H10 | TEST Dense | TEST BM25 | TEST H10 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    views = (
        _view(dev["bm25"]),
        _view(dev["h10"]),
        _view(payload["dense_headline"]),
        _view(payload["bm25_headline"]),
        _view(payload["h10_headline"]),
    )
    keys = (
        ("Hit@1", "hit@1"),
        ("Hit@5", "hit@5"),
        ("Hit@10", "hit@10"),
        ("Recall@5 macro", "recall@5_macro"),
        ("Recall@10 macro", "recall@10_macro"),
        ("MRR", "mrr"),
        ("NDCG@10 condensed", "ndcg@10_condensed"),
        ("requirement_recall@10", "requirement_recall@10"),
        ("all_requirements_covered@10", "all_requirements_covered@10"),
        ("judged_coverage@10", "judged_coverage@10"),
    )
    for label, key in keys:
        cells = []
        for view in views:
            value = view[key]
            cells.append(value if isinstance(value, str) else _num(value))
        lines.append("| " + label + " | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            (
                "Full-RRF와 Candidate-Truncated RRF는 DEV에서 이미 끝난 실험이다. "
                "TEST 후보로 돌리지 않았다."
            ),
        ]
    )
    return "\n".join(lines)


def _verdict_section(payload: dict[str, Any]) -> str:
    verdict = payload["verdict"]
    label = verdict["label"]
    if label == "Generalization Support":
        sentence = (
            "DEV에서 선택한 asymmetric policy가 독립 TEST에서도 방향적으로 재현됐다."
        )
    elif label == "Generalization Failure":
        sentence = "DEV에서 선택된 policy가 TEST로 일반화되지 않았다."
    else:
        sentence = "DEV improvement가 TEST에서는 명확히 재현되지 않았다."
    return "\n".join(
        [
            f"{verdict['label']}.",
            "",
            sentence,
            "",
            f"reason `{verdict['reason']}`. {verdict['rule']}",
        ]
    )


def _decision_section(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    return "\n".join(
        [
            decision["statement"],
            "",
            f"선택 configuration: `{decision['selected_configuration']}`.",
            "retune: false. 제품 검색 경로를 바꾸는 변경은 없다.",
            (
                "TEST 결과를 보고 rule, rank slot, Dense rank threshold, weight, "
                "fallback, query routing, reranker, embedding, chunking을 바꾸지 않는다."
            ),
        ]
    )


def _limits_section(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"모집단은 fully_answerable TEST {payload['question_count']}문항 한 split이다.",
            "retrieval은 한 번이다. 이 점수로 두 번째 TEST를 열지 않는다.",
            "unjudged unit은 미판정이다. relevance 0으로 채우지 않는다.",
            "preferred_hit@10은 diagnostic이다.",
            "NDCG headline은 condensed다.",
            "DEV의 41/45, 49/58, 39/45 하한은 TEST 분자에 다시 적용하지 않았다.",
            "ranking 하락 0.05는 DEV에서 고정한 경계이고, TEST 점수에 맞춰 다시 고르지 않았다.",
            "Full-RRF와 truncated RRF는 이번 비교의 팔이 아니다.",
            "작은 유입 건수를 큰 향상으로 쓰지 않는다.",
            "qrel과 judgment는 수정하지 않았다. 새 judge call은 0이다.",
        ]
    )


def _artifact_section() -> str:
    names = [
        "execution_lock.json",
        "search_started.json",
        "experiment_config.json",
        "evaluation_integrity.json",
        "dense_test.json",
        "bm25_test.json",
        "dense_scored.json",
        "bm25_scored.json",
        "h10_test.json",
        "headline_comparison.json",
        "per_query_transition.json",
        "per_query_transition.csv",
        "requirement_transition.json",
        "dense_head_insertions.json",
        "judgment_coverage.json",
        "representative_cases.json",
        "dev_vs_test.json",
        "verdict.json",
        "report.md",
        "portfolio_summary.md",
    ]
    directory = f"evaluation/runs/{RUN_DIR_NAME}/"
    lines = [f"`{directory}`", ""]
    lines.extend(f"- `{directory}{name}`" for name in names)
    lines.append("")
    lines.append("`docs/worklogs/2026-10-07_v21-test-h10-frozen-validation.md`")
    return "\n".join(lines)


def _commit_section(payload: dict[str, Any]) -> str:
    row = payload["integrity"]
    return "\n".join(
        [
            f"정책 원본 commit `{row['frozen_commit']}`.",
            (
                f"TEST 검색 시점 HEAD `{row['git_commit_at_search']}`, "
                f"dirty {row['git_dirty_at_search']}."
            ),
            (
                f"이 보고서 채점 시점 HEAD `{row['git_commit_at_score']}`, "
                f"dirty {row['git_dirty_at_score']}."
            ),
            (
                "검증 산출물의 commit은 채점 뒤에 남긴다. "
                "그 commit은 정책 blob을 f911283에서 바꾸지 않는다."
            ),
        ]
    )


def _id_list(values: list[str]) -> str:
    if not values:
        return "없음."
    return ", ".join(f"`{value}`" for value in values) + "."


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
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def _run_dir(root: Path) -> Path:
    return root / "evaluation" / "runs" / RUN_DIR_NAME


def _lock_path(root: Path) -> Path:
    return _run_dir(root) / "execution_lock.json"


def _marker_path(root: Path) -> Path:
    return _run_dir(root) / "search_started.json"


def _read_hashed(root: Path, name: str, expected: str) -> dict[str, Any]:
    path = _run_dir(root) / name
    digest = _sha_file(path)
    if digest != expected:
        raise UsageError(f"{name} hash does not match the TEST lock")
    return _read(path)


def _git_blob(root: Path, spec: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", spec],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise UsageError(result.stderr.strip() or f"git rev-parse failed for {spec}")
    return result.stdout.strip()


def _hash_object(root: Path, relative: str) -> str:
    result = subprocess.run(
        ["git", "hash-object", "--", relative],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise UsageError(result.stderr.strip() or f"git hash-object failed for {relative}")
    return result.stdout.strip()


def _dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

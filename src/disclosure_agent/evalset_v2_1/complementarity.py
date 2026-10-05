"""DEV-only comparison of Dense-D2a and the existing BM25 baseline.

The population is v2.1 fully_answerable dev. Test questions are not searched
and are not used to choose a parameter. No embedding, query expansion,
reranker, hybrid, or RRF is added. v2 and v2.1 datasets are read and not written.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from disclosure_agent.agent.search import _eligible_rows
from disclosure_agent.agent.search_catalog import CatalogDenseSearch, load_search_catalog
from disclosure_agent.bm25 import (
    BM25_PARAMS,
    TOKENIZER_RULES,
    TOKENIZER_VERSION,
    Bm25Index,
    index_manifest,
)
from disclosure_agent.dense import DENSE_MODEL_ID, QUERY_ENCODING_POLICY
from disclosure_agent.errors import UsageError
from disclosure_agent.evalset_v2_1 import DATASET_ID
from disclosure_agent.evalset_v2_1.freeze import dataset_dir
from disclosure_agent.evalset_v2_1.metrics import (
    SMALL_N,
    _unit_grades,
    aggregate_headline,
    breakdown,
    score_question,
)
from disclosure_agent.evaluation import git_revision
from disclosure_agent.retrieval_baseline import TOP_K, _search_question
from disclosure_agent.retrieval_unit import D2A_VERSION

RUN_DIR_NAME = "multi_company_retrieval_v2_1_dev_dense_bm25_complementarity"
HIT_K = 10
CELL_BOTH = "dense_hit_bm25_hit"
CELL_DENSE = "dense_hit_bm25_miss"
CELL_BM25 = "dense_miss_bm25_hit"
CELL_MISS = "dense_miss_bm25_miss"
QUERY_CELLS = (CELL_BOTH, CELL_DENSE, CELL_BM25, CELL_MISS)
REQUIREMENT_CELLS = ("both", "dense_only", "bm25_only", "neither")
NULL_FILTERS = {"as_of": None, "period": None, "report_types": None}
EXCERPT_CHARS = 480
COVERAGE_GAP_FLAG = 0.05
BM25_PARAMETER_HISTORY = (
    "k1=1.2 and b=0.75 were introduced as pinned defaults in commit 236c2a3 "
    "and were not edited later. The only later bm25.py change added rank_all. "
    "The repository has no k1/b sweep. This run does not retune them and does "
    "not read v2.1 TEST. They are conventional BM25 defaults, not a dev-tuned "
    "operating point for this benchmark."
)
HYBRID_DECISION_RULE = (
    "Primary label uses DEV Hit@10 and requirement@10 only. "
    "B when BM25 recovers nothing Dense missed: BM25-only query hits are 0, "
    "BM25-only requirement recoveries are 0, and no query has crossed "
    "requirement coverage. "
    "C when that is not true and both-miss queries are at least as many as "
    "the two asymmetric query cells combined and at least one third of DEV. "
    "A otherwise, which requires BM25 to add a query hit, a requirement, or "
    "a crossed multi-requirement case without shared misses dominating. "
    "B takes precedence over C so a nested BM25 does not become a hybrid "
    "proposal. Shared-miss dominance still selects the non-hybrid follow-up."
)
_SURFACE_RE = re.compile(r"\d+(?:\.\d+)?|[가-힣]{2,}|[A-Za-z][A-Za-z0-9]{1,}")
_SUBGROUPS = (
    "company_name",
    "wording_style",
    "evidence_form",
    "query_type",
    "stratum_id",
    "period_sensitive",
    "requirement_band",
)


def dev_fully_answerable(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fully answerable dev questions. Test, partial, and unanswerable stay out."""
    return [
        question
        for question in questions
        if question.get("answerability") == "fully_answerable" and question.get("split") == "dev"
    ]


def assert_dev_population(
    questions: list[dict[str, Any]], *, expected_count: int
) -> list[dict[str, Any]]:
    kept = dev_fully_answerable(questions)
    if len(kept) != expected_count:
        raise UsageError(
            f"dev fully_answerable count is {len(kept)}, freeze dev_size is {expected_count}"
        )
    if any(question.get("split") != "dev" for question in kept):
        raise UsageError("complementarity population contains a non-dev question")
    return kept


def query_cell(dense_hit: bool, bm25_hit: bool) -> str:
    if dense_hit and bm25_hit:
        return CELL_BOTH
    if dense_hit:
        return CELL_DENSE
    if bm25_hit:
        return CELL_BM25
    return CELL_MISS


def decide_hybrid(
    cell_counts: dict[str, int],
    requirement_counts: dict[str, int],
    *,
    crossed_queries: int,
) -> dict[str, Any]:
    """Apply the pre-registered DEV rule. Do not feed this TEST counts."""
    missing = [name for name in QUERY_CELLS if name not in cell_counts]
    if missing:
        raise UsageError(f"query cells missing: {missing}")
    n = sum(int(cell_counts[name]) for name in QUERY_CELLS)
    dense_only = int(cell_counts[CELL_DENSE])
    bm25_only = int(cell_counts[CELL_BM25])
    both_miss = int(cell_counts[CELL_MISS])
    req_bm25_only = int(requirement_counts.get("bm25_only") or 0)
    bm25_adds_nothing = bm25_only == 0 and req_bm25_only == 0 and crossed_queries == 0
    shared_dominant = n > 0 and both_miss >= dense_only + bm25_only and both_miss * 3 >= n
    bidirectional = dense_only > 0 and bm25_only > 0
    if bm25_adds_nothing:
        primary = "B"
    elif shared_dominant:
        primary = "C"
    else:
        primary = "A"
    return {
        "primary": primary,
        "rule": HYBRID_DECISION_RULE,
        "bm25_adds_nothing": bm25_adds_nothing,
        "shared_miss_dominant": shared_dominant,
        "bidirectional_query_complementarity": bidirectional,
        "requirement_cross_coverage_queries": crossed_queries,
        "question_count": n,
        "test_used": False,
    }


def ranking_duplicates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Count repeated retrieval_unit_id values inside returned rankings.

    The metric keeps the duplicate rank and gives the later copy gain 0.
    This record is the defect log. It does not rewrite the ranking.
    """
    questions = []
    slots = 0
    for row in rows:
        ids = [str(hit["retrieval_unit_id"]) for hit in row.get("hits") or []]
        repeated = sorted({unit_id for unit_id in ids if ids.count(unit_id) > 1})
        if not repeated:
            continue
        extra = len(ids) - len(set(ids))
        slots += extra
        questions.append(
            {
                "query_id": row.get("query_id"),
                "duplicate_ids": repeated,
                "extra_slots": extra,
            }
        )
    return {
        "questions_with_duplicates": len(questions),
        "duplicate_slots": slots,
        "questions": questions,
    }


def metric_hits(
    raw_hits: list[dict[str, Any]],
    meta: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Copy retrieval_unit_id through. Do not substitute a source element id."""
    rows = []
    for hit in raw_hits:
        if "retrieval_unit_id" not in hit and "chunk_id" not in hit:
            raise UsageError("retriever hit has no retrieval_unit_id")
        unit_id = str(hit.get("retrieval_unit_id") or hit["chunk_id"])
        unit = meta.get(unit_id) or {}
        element_ids = [str(item) for item in unit.get("source_element_ids") or []]
        rows.append(
            {
                "rank": int(hit["rank"]),
                "retrieval_unit_id": unit_id,
                "source_element_ids": [
                    str(item) for item in hit.get("source_element_ids") or element_ids
                ],
            }
        )
    return rows


def bm25_chunks(
    units: list[dict[str, Any]], scoring: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """One BM25 document per eligible D2a row, using that row's text and id."""
    if len(units) != len(scoring):
        raise UsageError("eligible units and scoring rows diverged")
    chunks = []
    for unit, row in zip(units, scoring, strict=True):
        unit_id = str(unit["retrieval_unit_id"])
        if unit_id != str(row["retrieval_unit_id"]):
            raise UsageError("scoring row is not the eligible unit")
        if str(unit.get("representation_version") or "") != D2A_VERSION:
            raise UsageError(f"{unit_id} is not a D2a retrieval unit")
        if (unit.get("text") or "") != (row.get("text") or ""):
            raise UsageError("BM25 text is not the dense unit text")
        chunks.append(
            {
                "chunk_id": row["chunk_id"],
                "ordinal": row["ordinal"],
                "text": row["text"],
                "retrieval_unit_id": unit_id,
                "unit_index": row["unit_index"],
            }
        )
    return chunks


def covered_requirement_ids(
    query_id: str,
    requirements: list[dict[str, Any]],
    qrels: list[dict[str, Any]],
    hits: list[dict[str, Any]],
    *,
    k: int,
) -> tuple[list[str], list[str]]:
    """Same coverage rule as the v2.1 metric: one relevant unit for that requirement."""
    top = {str(hit["retrieval_unit_id"]) for hit in hits[:k]}
    needed = [
        item
        for item in requirements
        if item["query_id"] == query_id and item["answerability"] == "fully_answerable"
    ]
    covered: list[str] = []
    missing: list[str] = []
    for requirement in needed:
        ok = any(
            row["query_id"] == query_id
            and row["requirement_id"] == requirement["requirement_id"]
            and str(row["retrieval_unit_id"]) in top
            and int(row["relevance"]) >= 1
            for row in qrels
        )
        bucket = covered if ok else missing
        bucket.append(str(requirement["requirement_id"]))
    return covered, missing


def requirement_complementarity(
    query_id: str,
    requirements: list[dict[str, Any]],
    qrels: list[dict[str, Any]],
    dense_hits: list[dict[str, Any]],
    bm25_hits: list[dict[str, Any]],
    *,
    k: int,
) -> list[dict[str, Any]]:
    needed = [
        item
        for item in requirements
        if item["query_id"] == query_id and item["answerability"] == "fully_answerable"
    ]
    dense_top = {str(hit["retrieval_unit_id"]) for hit in dense_hits[:k]}
    bm25_top = {str(hit["retrieval_unit_id"]) for hit in bm25_hits[:k]}
    rows = []
    for requirement in needed:
        requirement_id = str(requirement["requirement_id"])
        positive = {
            str(row["retrieval_unit_id"])
            for row in qrels
            if row["query_id"] == query_id
            and row["requirement_id"] == requirement_id
            and int(row["relevance"]) >= 1
        }
        dense_found = sorted(positive & dense_top)
        bm25_found = sorted(positive & bm25_top)
        if dense_found and bm25_found:
            cell = "both"
        elif dense_found:
            cell = "dense_only"
        elif bm25_found:
            cell = "bm25_only"
        else:
            cell = "neither"
        rows.append(
            {
                "query_id": query_id,
                "requirement_id": requirement_id,
                "requirement_text": requirement.get("text") or "",
                "cell": cell,
                "positive_unit_count": len(positive),
                "dense_unit_ids": dense_found,
                "bm25_unit_ids": bm25_found,
                "same_unit": bool(set(dense_found) & set(bm25_found)),
            }
        )
    return rows


def crossed_query_ids(rows: list[dict[str, Any]]) -> list[str]:
    """Queries where each retriever covers a requirement the other misses."""
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_query[row["query_id"]].append(row)
    found = []
    for query_id, group in sorted(by_query.items()):
        dense_only = any(item["cell"] == "dense_only" for item in group)
        bm25_only = any(item["cell"] == "bm25_only" for item in group)
        if dense_only and bm25_only:
            found.append(query_id)
    return found


def first_relevant_rank(scored: dict[str, Any]) -> int | None:
    for hit in scored["hits"]:
        if hit["relevance"] is not None and int(hit["relevance"]) >= 1:
            return int(hit["rank"])
    return None


def relevant_ids_at(scored: dict[str, Any], k: int) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for hit in scored["hits"][:k]:
        if hit["relevance"] is None or int(hit["relevance"]) < 1:
            continue
        unit_id = str(hit["retrieval_unit_id"])
        if unit_id in seen:
            continue
        seen.add(unit_id)
        found.append(unit_id)
    return found


def failure_note(row: dict[str, Any]) -> str:
    """State ranks and blocked units. Do not assign a causal category."""
    parts: list[str] = []
    outside = int(row["positive_outside_candidate_count"])
    if outside:
        parts.append(f"양성 unit {outside}개는 이번 filing scope 밖에 있다.")
    if row["candidate_filter_blocks_all_positives"]:
        parts.append("양성 unit이 후보 집합에 하나도 없다.")
    dense_rank = row["dense_first_relevant_rank"]
    bm25_rank = row["bm25_first_relevant_rank"]
    cell = row["cell"]
    if cell == CELL_BOTH:
        parts.append(
            f"둘 다 상위 {HIT_K}에서 양성을 찾았다. "
            f"Dense 첫 양성 rank {dense_rank}, BM25 첫 양성 rank {bm25_rank}."
        )
    elif cell == CELL_DENSE:
        parts.append(_one_sided_note("Dense", dense_rank, "BM25", bm25_rank))
    elif cell == CELL_BM25:
        parts.append(_one_sided_note("BM25", bm25_rank, "Dense", dense_rank))
    else:
        parts.append(f"둘 다 상위 {HIT_K}에 양성 unit이 없다.")
        later = []
        if dense_rank is not None:
            later.append(f"Dense rank {dense_rank}")
        if bm25_rank is not None:
            later.append(f"BM25 rank {bm25_rank}")
        if later:
            parts.append("반환된 순위 안에서는 " + ", ".join(later) + "이다.")
    if row["bm25_returned_count"] < HIT_K:
        parts.append(
            f"BM25 양수 점수는 {row['bm25_returned_count']}개뿐이라 "
            f"상위 {HIT_K}이 채워지지 않았다."
        )
    if row["crossed_requirements"]:
        parts.append("같은 질문에서 서로 다른 requirement를 각각 보완한다.")
    elif row["bm25_only_requirements"] or row["dense_only_requirements"]:
        only = []
        if row["dense_only_requirements"]:
            only.append("Dense만 " + ",".join(row["dense_only_requirements"]))
        if row["bm25_only_requirements"]:
            only.append("BM25만 " + ",".join(row["bm25_only_requirements"]))
        parts.append("requirement 차이: " + "; ".join(only) + ".")
    if row["missing_requirements"]:
        parts.append("둘 다 놓친 requirement: " + ",".join(row["missing_requirements"]) + ".")
    return " ".join(parts)


def cell_breakdown(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(field))].append(row)
    payload = {}
    for key, items in sorted(groups.items()):
        counts = Counter(item["cell"] for item in items)
        payload[key] = {
            "question_count": len(items),
            "small_n": len(items) < SMALL_N,
            "cells": {name: int(counts.get(name, 0)) for name in QUERY_CELLS},
        }
    return payload


def run_complementarity(root: Path) -> dict[str, Any]:
    directory = dataset_dir(root)
    frozen = _read(directory / "freeze.json")
    if frozen.get("dataset_id") != DATASET_ID:
        raise UsageError("complementarity dataset is not v2.1")
    questions = assert_dev_population(
        _read(directory / "questions.json")["questions"],
        expected_count=int(frozen["dev_size"]),
    )
    requirements = _read(directory / "requirements.json")["requirements"]
    qrels = _read(directory / "qrels.json")["qrels"]
    qrels_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in qrels:
        qrels_by_query[row["query_id"]].append(row)
    catalog = load_search_catalog(root)
    search = CatalogDenseSearch(catalog)
    by_company: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for question in questions:
        by_company[str(question["corp_code"])].append(question)

    dense_scored: list[dict[str, Any]] = []
    bm25_scored: list[dict[str, Any]] = []
    per_query: list[dict[str, Any]] = []
    requirement_rows: list[dict[str, Any]] = []
    dossiers: list[dict[str, Any]] = []
    company_audits: list[dict[str, Any]] = []
    scope_cache: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}

    for corp in sorted(by_company):
        corpus = catalog.corpus(corp)
        if corpus is None:
            raise UsageError(f"{corp} is not in the product catalog")
        group = sorted(by_company[corp], key=lambda item: str(item["query_id"]))
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
            owned_qrels = qrels_by_query.get(query_id) or []
            dense_row = score_question(question, owned_qrels, requirements, dense_hits)
            bm25_row = score_question(question, owned_qrels, requirements, bm25_hits)
            for row in (dense_row, bm25_row):
                row["requirement_band"] = "multi" if row["multi_evidence"] else "single"
            req_rows = requirement_complementarity(
                query_id, requirements, owned_qrels, dense_hits, bm25_hits, k=HIT_K
            )
            _assert_requirement_counts(query_id, dense_row, bm25_row, req_rows)
            dense_scored.append(dense_row)
            bm25_scored.append(bm25_row)
            requirement_rows.extend(req_rows)
            record = _query_record(
                question,
                dense_row,
                bm25_row,
                req_rows,
                owned_qrels,
                cached["ids"],
                len(bm25_raw),
            )
            per_query.append(record)
            if _needs_dossier(record):
                dossiers.append(
                    _dossier(
                        question,
                        record,
                        req_rows,
                        dense_raw,
                        bm25_raw,
                        cached["meta"],
                        owned_qrels,
                    )
                )
            print(f"complementarity {query_id} {record['cell']}", flush=True)
        company_audits.append(_company_audit(corp, corpus, group, scope_cache))
        search._search.pop(corp, None)
        scope_cache.clear()

    per_query.sort(key=lambda item: item["query_id"])
    requirement_rows.sort(key=lambda item: (item["query_id"], item["requirement_id"]))
    dossiers.sort(key=lambda item: item["query_id"])
    dense_headline = aggregate_headline(dense_scored)
    bm25_headline = aggregate_headline(bm25_scored)
    cell_counts = dict(Counter(item["cell"] for item in per_query))
    for name in QUERY_CELLS:
        cell_counts.setdefault(name, 0)
    requirement_counts = dict(Counter(item["cell"] for item in requirement_rows))
    for name in REQUIREMENT_CELLS:
        requirement_counts.setdefault(name, 0)
    crossed = crossed_query_ids(requirement_rows)
    decision = decide_hybrid(
        cell_counts, requirement_counts, crossed_queries=len(crossed)
    )
    audit = _audit(
        root,
        directory,
        frozen,
        company_audits,
        dense_scored,
        bm25_scored,
        dense_headline,
        bm25_headline,
        per_query,
    )
    audit["dense_reproduction"] = _compare_stored_dense(root, dense_scored)
    payload = {
        "experiment_id": "v21_dev_dense_d2a_vs_bm25",
        "dataset_id": DATASET_ID,
        "dataset_sha256": frozen["dataset_sha256"],
        "parent_dataset_sha256": frozen["parent_dataset_sha256"],
        "population": "dev fully_answerable",
        "question_count": len(per_query),
        "test_questions_searched": 0,
        "test_used_for_tuning": False,
        "hybrid_implemented": False,
        "query_cells": {name: int(cell_counts[name]) for name in QUERY_CELLS},
        "requirement_cells": {name: int(requirement_counts[name]) for name in REQUIREMENT_CELLS},
        "crossed_query_ids": crossed,
        "hybrid_decision": decision,
        "dense_headline": dense_headline,
        "bm25_headline": bm25_headline,
        "headline_table": _headline_table(dense_headline, bm25_headline),
        "audit": audit,
    }
    output = _write_run(
        root,
        payload,
        dense_scored,
        bm25_scored,
        per_query,
        requirement_rows,
        dossiers,
        frozen,
    )
    payload["run_dir"] = str(output)
    return payload


def _index_scope(
    search: CatalogDenseSearch,
    corpus: Any,
    corp: str,
    receipts: tuple[str, ...],
) -> dict[str, Any]:
    loaded = search._search.get(corp)
    if loaded is None:
        raise UsageError(f"{corp} dense index was not loaded by the product search path")
    units, _matrix, scoring = _eligible_rows(loaded.index(), corpus.binding, receipts)
    chunks = bm25_chunks(units, scoring)
    index = Bm25Index(chunks)
    ids = [str(unit["retrieval_unit_id"]) for unit in units]
    if [str(chunk["retrieval_unit_id"]) for chunk in index.chunks] != ids:
        raise UsageError("BM25 document ids are not the eligible D2a ids")
    meta: dict[str, dict[str, Any]] = {}
    id_counts = Counter(ids)
    duplicate_ids = sorted(unit_id for unit_id, count in id_counts.items() if count > 1)
    for unit in units:
        meta.setdefault(str(unit["retrieval_unit_id"]), unit)
    return {
        "index": index,
        "meta": meta,
        "ids": ids,
        "id_set": set(ids),
        "duplicate_ids": duplicate_ids,
        "receipts": receipts,
        "manifest": index_manifest(
            index,
            scope={
                "corp_code": corp,
                "eligible_unit_count": len(ids),
                "effective_filing_count": len(receipts),
            },
        ),
    }


def _assert_hits_in_universe(
    query_id: str,
    label: str,
    hits: list[dict[str, Any]],
    cached: dict[str, Any],
    receipts: tuple[str, ...],
) -> None:
    allowed_receipts = set(receipts)
    for hit in hits:
        unit_id = str(hit.get("retrieval_unit_id") or hit.get("chunk_id") or "")
        if unit_id not in cached["id_set"]:
            raise UsageError(f"{query_id} {label} hit {unit_id} is outside the eligible units")
        receipt = hit.get("receipt_no")
        if receipt is not None and str(receipt) not in allowed_receipts:
            raise UsageError(f"{query_id} {label} hit is outside the filing scope")


def _assert_requirement_counts(
    query_id: str,
    dense_row: dict[str, Any],
    bm25_row: dict[str, Any],
    req_rows: list[dict[str, Any]],
) -> None:
    dense_covered = sum(1 for row in req_rows if row["cell"] in {"both", "dense_only"})
    bm25_covered = sum(1 for row in req_rows if row["cell"] in {"both", "bm25_only"})
    if dense_covered != int(dense_row["requirement_covered@10"]):
        raise UsageError(f"{query_id} dense requirement coverage drifted from the metric")
    if bm25_covered != int(bm25_row["requirement_covered@10"]):
        raise UsageError(f"{query_id} bm25 requirement coverage drifted from the metric")
    if len(req_rows) != int(dense_row["requirement_total"]):
        raise UsageError(f"{query_id} requirement denominator drifted")


def _query_record(
    question: dict[str, Any],
    dense_row: dict[str, Any],
    bm25_row: dict[str, Any],
    req_rows: list[dict[str, Any]],
    qrels: list[dict[str, Any]],
    eligible_ids: list[str],
    bm25_returned: int,
) -> dict[str, Any]:
    dense_hit = bool(dense_row["at_k"][str(HIT_K)]["hit"])
    bm25_hit = bool(bm25_row["at_k"][str(HIT_K)]["hit"])
    grades = _unit_grades(qrels)
    positive = {unit_id for unit_id, grade in grades.items() if grade >= 1}
    eligible = set(eligible_ids)
    outside = sorted(positive - eligible)
    dense_only_reqs = [row["requirement_id"] for row in req_rows if row["cell"] == "dense_only"]
    bm25_only_reqs = [row["requirement_id"] for row in req_rows if row["cell"] == "bm25_only"]
    missing = [row["requirement_id"] for row in req_rows if row["cell"] == "neither"]
    record = {
        "query_id": question["query_id"],
        "company": question.get("company_name"),
        "question": question.get("question"),
        "wording": question.get("wording_style"),
        "query_type": question.get("query_type"),
        "evidence_form": question.get("evidence_form"),
        "stratum": question.get("stratum_id"),
        "period_sensitive": bool(question.get("period_sensitive")),
        "requirement_count": len(req_rows),
        "requirement_band": "multi" if len(req_rows) > 1 else "single",
        "cell": query_cell(dense_hit, bm25_hit),
        "dense_first_relevant_rank": first_relevant_rank(dense_row),
        "bm25_first_relevant_rank": first_relevant_rank(bm25_row),
        "dense_relevant_unit_ids": relevant_ids_at(dense_row, HIT_K),
        "bm25_relevant_unit_ids": relevant_ids_at(bm25_row, HIT_K),
        "dense_only_requirements": dense_only_reqs,
        "bm25_only_requirements": bm25_only_reqs,
        "missing_requirements": missing,
        "crossed_requirements": bool(dense_only_reqs and bm25_only_reqs),
        "positive_unit_count": len(positive),
        "positive_outside_candidate_count": len(outside),
        "positive_units_outside_candidates": outside,
        "candidate_filter_blocks_all_positives": bool(positive) and len(outside) == len(positive),
        "bm25_returned_count": bm25_returned,
        "dense_returned_count": len(dense_row["hits"]),
        "company_name": question.get("company_name"),
        "wording_style": question.get("wording_style"),
        "stratum_id": question.get("stratum_id"),
    }
    record["failure_note"] = failure_note(record)
    return record


def _needs_dossier(record: dict[str, Any]) -> bool:
    if record["cell"] != CELL_BOTH:
        return True
    return bool(
        record["crossed_requirements"]
        or record["dense_only_requirements"]
        or record["bm25_only_requirements"]
    )


def _dossier(
    question: dict[str, Any],
    record: dict[str, Any],
    req_rows: list[dict[str, Any]],
    dense_raw: list[dict[str, Any]],
    bm25_raw: list[dict[str, Any]],
    meta: dict[str, dict[str, Any]],
    qrels: list[dict[str, Any]],
) -> dict[str, Any]:
    grades = _unit_grades(qrels)
    return {
        "query_id": record["query_id"],
        "cell": record["cell"],
        "company": record["company"],
        "question": record["question"],
        "wording": record["wording"],
        "query_type": record["query_type"],
        "evidence_form": record["evidence_form"],
        "stratum": record["stratum"],
        "period_sensitive": record["period_sensitive"],
        "failure_note": record["failure_note"],
        "dense_first_relevant_rank": record["dense_first_relevant_rank"],
        "bm25_first_relevant_rank": record["bm25_first_relevant_rank"],
        "requirements": [
            {
                "requirement_id": row["requirement_id"],
                "text": row["requirement_text"],
                "cell": row["cell"],
                "dense_unit_ids": row["dense_unit_ids"],
                "bm25_unit_ids": row["bm25_unit_ids"],
            }
            for row in req_rows
        ],
        "dense_top": [_hit_view(hit, grades, meta) for hit in dense_raw[:5]],
        "bm25_top": [_hit_view(hit, grades, meta) for hit in bm25_raw[:5]],
        "positive_units": _positive_views(grades, meta, record),
        "query_terms": surface_terms(str(question.get("question") or "")),
    }


def _positive_views(
    grades: dict[str, int],
    meta: dict[str, dict[str, Any]],
    record: dict[str, Any],
) -> list[dict[str, Any]]:
    dense_ids = set(record["dense_relevant_unit_ids"])
    bm25_ids = set(record["bm25_relevant_unit_ids"])
    views = []
    for unit_id, grade in sorted(grades.items()):
        if grade < 1:
            continue
        unit = meta.get(unit_id)
        text = "" if unit is None else str(unit.get("text") or "")
        views.append(
            {
                "retrieval_unit_id": unit_id,
                "relevance": grade,
                "in_candidate_set": unit is not None,
                "in_dense_at_10": unit_id in dense_ids,
                "in_bm25_at_10": unit_id in bm25_ids,
                "table_markers": table_markers(text),
                "query_terms_present": terms_in_text(
                    surface_terms(str(record["question"] or "")), text
                ),
                "excerpt": _excerpt(text),
                "section_path": [] if unit is None else list(unit.get("section_path") or []),
                "document_name": None if unit is None else unit.get("document_name"),
                "filing_period": None if unit is None else unit.get("filing_period"),
            }
        )
    return views


def _hit_view(
    hit: dict[str, Any],
    grades: dict[str, int],
    meta: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    unit_id = str(hit.get("retrieval_unit_id") or hit.get("chunk_id"))
    unit = meta.get(unit_id) or {}
    text = str(hit.get("text") or unit.get("text") or "")
    relevance = grades.get(unit_id)
    return {
        "rank": int(hit["rank"]),
        "retrieval_unit_id": unit_id,
        "score": hit.get("score"),
        "judged": unit_id in grades,
        "relevance": relevance,
        "table_markers": table_markers(text),
        "section_path": list(hit.get("section_path") or unit.get("section_path") or []),
        "document_name": hit.get("document_name") or unit.get("document_name"),
        "filing_period": hit.get("filing_period") or unit.get("filing_period"),
        "excerpt": _excerpt(text),
    }


def _company_audit(
    corp: str,
    corpus: Any,
    questions: list[dict[str, Any]],
    scope_cache: dict[tuple[str, tuple[str, ...]], dict[str, Any]],
) -> dict[str, Any]:
    scopes = [value for (company, _receipts), value in scope_cache.items() if company == corp]
    if len(scopes) != 1:
        raise UsageError(
            f"{corp} dev questions did not share one filing scope ({len(scopes)} scopes)"
        )
    scope = scopes[0]
    ids = scope["ids"]
    return {
        "corp_code": corp,
        "company_name": corpus.binding.company_name,
        "corpus_id": corpus.manifest.get("corpus_id"),
        "embedding_model": corpus.manifest.get("embedding_model"),
        "representation_version": corpus.manifest.get("representation_version"),
        "question_count": len(questions),
        "effective_filing_count": len(scope["receipts"]),
        "effective_receipt_nos": list(scope["receipts"]),
        "eligible_unit_count": len(ids),
        "unique_eligible_unit_count": len(set(ids)),
        "duplicate_retrieval_unit_ids_in_universe": scope["duplicate_ids"],
        "bm25_ids_equal_dense_eligible_ids": True,
        "bm25_index": scope["manifest"],
        "filters": dict(NULL_FILTERS),
    }


def _audit(
    root: Path,
    directory: Path,
    frozen: dict[str, Any],
    company_audits: list[dict[str, Any]],
    dense_scored: list[dict[str, Any]],
    bm25_scored: list[dict[str, Any]],
    dense_headline: dict[str, Any],
    bm25_headline: dict[str, Any],
    per_query: list[dict[str, Any]],
) -> dict[str, Any]:
    dense_coverage = float(dense_headline["at_k"]["10"]["judged_coverage"]["value"] or 0)
    bm25_coverage = float(bm25_headline["at_k"]["10"]["judged_coverage"]["value"] or 0)
    gap = abs(dense_coverage - bm25_coverage)
    revision = git_revision(root)
    return {
        "same_retrieval_unit_universe": all(
            item["bm25_ids_equal_dense_eligible_ids"] for item in company_audits
        ),
        "filing_policy_equal": True,
        "filing_policy": (
            "resolve_search_call / select_filings with as_of, period, and "
            "report_types unset. Same helper as the v2.1 Dense-D2a rescore. "
            "Correction originals in scope are suppressed. BM25 indexes only "
            "the retrieval units inside that receipt set."
        ),
        "bm25_scores_element_ids": False,
        "qrel_key": "(query_id, requirement_id, retrieval_unit_id)",
        "legacy_element_qrel_loaded": False,
        "query_preprocessing": {
            "shared_input": "question text with no expansion and no shared analyzer",
            "dense": QUERY_ENCODING_POLICY,
            "dense_model": DENSE_MODEL_ID,
            "bm25_tokenizer": TOKENIZER_VERSION,
            "bm25_tokenizer_rules": list(TOKENIZER_RULES),
            "difference": (
                "Both retrievers receive the same raw question. Dense encodes it "
                "with the existing BGE-M3 query encoder and no instruction. BM25 "
                "applies NFKC and ko-mixed-v1, including Hangul syllable bigrams, "
                "and saturates query term frequency at 1."
            ),
        },
        "document_text": (
            "BM25 tokenizes the D2a retrieval unit text. Dense ranks the stored "
            "D2a embedding of that same unit text. This run does not embed again."
        ),
        "bm25_parameters": {
            "k1": BM25_PARAMS["k1"],
            "b": BM25_PARAMS["b"],
            "idf": BM25_PARAMS["idf"],
            "tuned_in_this_experiment": False,
            "tuned_on_v21_test": False,
            "history": BM25_PARAMETER_HISTORY,
        },
        "top_k": TOP_K,
        "hit_k": HIT_K,
        "zero_score_omitted_from_bm25_topk": True,
        "companies": company_audits,
        "ranking_duplicates": {
            "dense": ranking_duplicates(dense_scored),
            "bm25": ranking_duplicates(bm25_scored),
        },
        "bm25_rankings_shorter_than_10": [
            {
                "query_id": row["query_id"],
                "bm25_returned_count": row["bm25_returned_count"],
            }
            for row in per_query
            if row["bm25_returned_count"] < HIT_K
        ],
        "positive_units_outside_candidates": [
            {
                "query_id": row["query_id"],
                "count": row["positive_outside_candidate_count"],
                "blocks_all": row["candidate_filter_blocks_all_positives"],
            }
            for row in per_query
            if row["positive_outside_candidate_count"]
        ],
        "window_states_at_10": {
            "dense": _mean_states(dense_scored),
            "bm25": _mean_states(bm25_scored),
        },
        "judged_coverage_at_10": {
            "dense": dense_coverage,
            "bm25": bm25_coverage,
            "absolute_gap": gap,
        },
        "judged_coverage_gap_flag": gap >= COVERAGE_GAP_FLAG,
        "unjudged_is_not_relevance_0": True,
        "dataset_file_sha256": {
            "freeze.json": _sha_file(directory / "freeze.json"),
            "questions.json": _sha_file(directory / "questions.json"),
            "qrels.json": _sha_file(directory / "qrels.json"),
            "requirements.json": _sha_file(directory / "requirements.json"),
        },
        "dataset_sha256": frozen["dataset_sha256"],
        "git_commit": revision["git_commit"],
        "git_dirty": revision["git_dirty"],
        "dataset_files_written": False,
    }


def _compare_stored_dense(root: Path, fresh: list[dict[str, Any]]) -> dict[str, Any]:
    """Reproducibility check against the existing v2.1 Dense-D2a run.

    Only dev question ids are read. Test rankings are not inputs to the decision.
    """
    path = (
        root
        / "evaluation"
        / "runs"
        / "multi_company_retrieval_v2_1_dense_d2a"
        / "per_question.json"
    )
    if not path.is_file():
        return {"compared": False, "reason": "stored dense run is absent"}
    stored = _read(path)["questions"]
    fresh_ids = {row["query_id"] for row in fresh}
    by_id = {row["query_id"]: row for row in stored if row["query_id"] in fresh_ids}
    mismatches = []
    for row in fresh:
        previous = by_id.get(row["query_id"])
        if previous is None:
            mismatches.append({"query_id": row["query_id"], "reason": "missing"})
            continue
        previous_ids = [hit["retrieval_unit_id"] for hit in previous["hits"]]
        fresh_ids_ranked = [hit["retrieval_unit_id"] for hit in row["hits"]]
        hit_changed = previous["at_k"]["10"]["hit"] != row["at_k"]["10"]["hit"]
        if previous_ids != fresh_ids_ranked or hit_changed:
            mismatches.append(
                {
                    "query_id": row["query_id"],
                    "stored_hit_at_10": previous["at_k"]["10"]["hit"],
                    "fresh_hit_at_10": row["at_k"]["10"]["hit"],
                }
            )
    return {
        "compared": True,
        "dev_questions_compared": len(fresh),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "role": "reproducibility of Dense-D2a on dev, not a tuning input",
    }


def _mean_states(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = Counter()
    for row in rows:
        totals.update(_window_state(row))
    count = len(rows) or 1
    keys = ("relevant", "irrelevant", "unjudged", "not_returned")
    return {key: totals[key] / count for key in keys}


def _window_state(scored: dict[str, Any]) -> dict[str, int]:
    hits = scored["hits"][:HIT_K]
    relevant = irrelevant = unjudged = 0
    for hit in hits:
        if not hit["judged"]:
            unjudged += 1
        elif hit["relevance"] is not None and int(hit["relevance"]) >= 1:
            relevant += 1
        else:
            irrelevant += 1
    return {
        "relevant": relevant,
        "irrelevant": irrelevant,
        "unjudged": unjudged,
        "not_returned": max(0, HIT_K - len(hits)),
    }


def _headline_table(
    dense: dict[str, Any], bm25: dict[str, Any]
) -> list[dict[str, Any]]:
    specs = (
        ("Hit@1", ("at_k", "1", "hit")),
        ("Hit@5", ("at_k", "5", "hit")),
        ("Hit@10", ("at_k", "10", "hit")),
        ("Recall@5 macro", ("recall@5_macro",)),
        ("Recall@5 micro", ("recall@5_micro",)),
        ("Recall@10 macro", ("recall@10_macro",)),
        ("Recall@10 micro", ("recall@10_micro",)),
        ("MRR", ("mrr",)),
        ("NDCG@10", ("at_k", "10", "ndcg_condensed")),
        ("preferred_hit@10", ("at_k", "10", "preferred_hit")),
        ("requirement_recall@10", ("requirement_recall@10",)),
        ("all_requirements_covered@10", ("all_requirements_covered@10",)),
        ("judged_coverage@10", ("at_k", "10", "judged_coverage")),
        ("judged_coverage@20", ("judged_coverage@20",)),
    )
    table = []
    for label, path in specs:
        left = _dig(dense, path)
        right = _dig(bm25, path)
        table.append(
            {
                "metric": label,
                "dense_numerator": left.get("numerator"),
                "dense_denominator": left.get("denominator"),
                "dense_value": left.get("value"),
                "bm25_numerator": right.get("numerator"),
                "bm25_denominator": right.get("denominator"),
                "bm25_value": right.get("value"),
                "role": "diagnostic" if label == "preferred_hit@10" else "headline",
                "ndcg_variant": "condensed" if label == "NDCG@10" else None,
            }
        )
    return table


def _write_run(
    root: Path,
    payload: dict[str, Any],
    dense_scored: list[dict[str, Any]],
    bm25_scored: list[dict[str, Any]],
    per_query: list[dict[str, Any]],
    requirement_rows: list[dict[str, Any]],
    dossiers: list[dict[str, Any]],
    frozen: dict[str, Any],
) -> Path:
    output = root / "evaluation" / "runs" / RUN_DIR_NAME
    if dataset_dir(root) in output.parents:
        raise UsageError("refusing to write the complementarity run into the dataset")
    output.mkdir(parents=True, exist_ok=True)
    config = _config(payload, frozen)
    subgroups = {
        "cells": {field: cell_breakdown(per_query, field) for field in _SUBGROUPS},
        "dense_metrics": {field: breakdown(dense_scored, field) for field in _SUBGROUPS},
        "bm25_metrics": {field: breakdown(bm25_scored, field) for field in _SUBGROUPS},
    }
    _dump(output / "experiment_config.json", config)
    _dump(
        output / "dense_dev.json",
        {"headline": payload["dense_headline"], "questions": dense_scored},
    )
    _dump(
        output / "bm25_dev.json",
        {"headline": payload["bm25_headline"], "questions": bm25_scored},
    )
    _dump(
        output / "headline_comparison.json",
        {
            "table": payload["headline_table"],
            "query_cells": payload["query_cells"],
            "requirement_cells": payload["requirement_cells"],
            "hybrid_decision": payload["hybrid_decision"],
            "ndcg_sensitivity": {
                "dense_unjudged_as_zero": payload["dense_headline"]["at_k"]["10"][
                    "ndcg_unjudged_as_zero"
                ],
                "bm25_unjudged_as_zero": payload["bm25_headline"]["at_k"]["10"][
                    "ndcg_unjudged_as_zero"
                ],
                "role": "sensitivity_not_headline",
            },
        },
    )
    _dump(output / "per_query_complementarity.json", {"queries": per_query})
    _write_csv(output / "per_query_complementarity.csv", per_query)
    _dump(
        output / "requirement_complementarity.json",
        {
            "cells": payload["requirement_cells"],
            "crossed_query_ids": payload["crossed_query_ids"],
            "requirements": requirement_rows,
        },
    )
    _dump(output / "subgroup_breakdown.json", subgroups)
    _dump(output / "representative_cases.json", {"cases": dossiers})
    _dump(output / "audit.json", payload["audit"])
    _dump(
        output / "failure_taxonomy.json",
        {
            "status": "unlabeled",
            "note": (
                "Cases are stored for reading. Category labels are not assigned "
                "from wording or evidence_form alone."
            ),
        },
    )
    report = render_report(payload, subgroups)
    (output / "report.md").write_text(report, encoding="utf-8")
    return output


def _config(payload: dict[str, Any], frozen: dict[str, Any]) -> dict[str, Any]:
    audit = payload["audit"]
    return {
        "experiment_id": payload["experiment_id"],
        "dataset_id": payload["dataset_id"],
        "dataset_sha256": payload["dataset_sha256"],
        "parent_dataset_sha256": payload["parent_dataset_sha256"],
        "population": payload["population"],
        "question_count": payload["question_count"],
        "split_seed": frozen.get("split_seed"),
        "metric_module": "disclosure_agent.evalset_v2_1.metrics",
        "metric_policy": frozen.get("metric_policy"),
        "top_k": TOP_K,
        "headline_ks": [1, 5, 10],
        "hit_complementarity_k": HIT_K,
        "test_questions_searched": 0,
        "test_used_for_tuning": False,
        "hybrid_implemented": False,
        "reranker": False,
        "query_expansion": False,
        "new_embedding": False,
        "dense": {
            "implementation": "product_catalog_dense_d2a",
            "entry": "retrieval_baseline._search_question",
            "model": DENSE_MODEL_ID,
            "query_encoding_policy": QUERY_ENCODING_POLICY,
            "representation_version": D2A_VERSION,
        },
        "bm25": {
            "implementation": "disclosure_agent.bm25.Bm25Index.search",
            "tokenizer_version": TOKENIZER_VERSION,
            "k1": BM25_PARAMS["k1"],
            "b": BM25_PARAMS["b"],
            "zero_score_omitted_from_topk": True,
            "indexed_text": "eligible D2a retrieval unit text",
            "parameter_history": BM25_PARAMETER_HISTORY,
        },
        "query_preprocessing": audit["query_preprocessing"],
        "filing_policy": audit["filing_policy"],
        "git_commit": audit["git_commit"],
        "git_dirty": audit["git_dirty"],
        "hybrid_decision_rule": HYBRID_DECISION_RULE,
    }


def render_report(payload: dict[str, Any], subgroups: dict[str, Any]) -> str:
    """Factual report. Manual failure labels are added after the cases are read."""
    cells = payload["query_cells"]
    reqs = payload["requirement_cells"]
    decision = payload["hybrid_decision"]
    lines = [
        "# v2.1 DEV Dense-D2a vs BM25 complementarity",
        "",
        "Hybrid는 구현하지 않았다. 비교 모집단은 fully_answerable dev "
        f"{payload['question_count']}문항이다. test는 검색하지 않았고 튜닝에 쓰지 않았다.",
        "",
        f"- dataset_sha256: `{payload['dataset_sha256']}`",
        f"- git_commit: `{payload['audit']['git_commit']}`",
        f"- git_dirty: {payload['audit']['git_dirty']}",
        "",
        "## Headline",
        "",
        _markdown_table(payload["headline_table"]),
        "",
        "## Query Hit@10 cells",
        "",
        f"- both hit: {cells[CELL_BOTH]}",
        f"- Dense only: {cells[CELL_DENSE]}",
        f"- BM25 only: {cells[CELL_BM25]}",
        f"- both miss: {cells[CELL_MISS]}",
        "",
        "## Requirement cells",
        "",
        f"- both: {reqs['both']}",
        f"- Dense only: {reqs['dense_only']}",
        f"- BM25 only: {reqs['bm25_only']}",
        f"- neither: {reqs['neither']}",
        f"- crossed queries: {', '.join(payload['crossed_query_ids']) or 'none'}",
        "",
        "## Pre-registered hybrid label",
        "",
        f"Primary label: {decision['primary']}.",
        "",
        decision["rule"],
        "",
        "## Subgroup Hit@10",
        "",
        "small_n은 문항 수가 5 미만이다. 이 표는 우열 순위가 아니다.",
        "",
    ]
    for field in _SUBGROUPS:
        lines.append(f"### {field}")
        lines.append("")
        lines.append(_subgroup_table(field, subgroups))
        lines.append("")
    coverage = payload["audit"]["judged_coverage_at_10"]
    lines.extend(
        [
            "## Coverage and duplicates",
            "",
            (
                f"- judged_coverage@10 dense {coverage['dense']}, "
                f"bm25 {coverage['bm25']}, gap {coverage['absolute_gap']}"
            ),
            f"- coverage gap flag: {payload['audit']['judged_coverage_gap_flag']}",
            (
                "- dense ranking duplicates: "
                f"{payload['audit']['ranking_duplicates']['dense']['duplicate_slots']}"
            ),
            (
                "- bm25 ranking duplicates: "
                f"{payload['audit']['ranking_duplicates']['bm25']['duplicate_slots']}"
            ),
            (
                "- BM25 rankings shorter than 10: "
                f"{len(payload['audit']['bm25_rankings_shorter_than_10'])}"
            ),
            (
                "- questions with a positive unit outside the candidate set: "
                f"{len(payload['audit']['positive_units_outside_candidates'])}"
            ),
            "",
            "## Failure taxonomy",
            "",
            "대표 사례의 본문을 읽기 전에는 범주를 붙이지 않는다.",
            "",
        ]
    )
    return "\n".join(lines)


def _subgroup_table(field: str, subgroups: dict[str, Any]) -> str:
    cells = subgroups["cells"][field]
    dense = subgroups["dense_metrics"][field]
    bm25 = subgroups["bm25_metrics"][field]
    header = (
        "| group | n | small_n | Dense Hit@10 | BM25 Hit@10 | both | "
        "dense only | bm25 only | both miss |"
    )
    sep = "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    body = [header, sep]
    for key in cells:
        dense_hit = dense[key]["at_k"]["10"]["hit"]
        bm25_hit = bm25[key]["at_k"]["10"]["hit"]
        item = cells[key]
        body.append(
            "| {group} | {n} | {small} | {dense} | {bm25} | {both} | "
            "{dense_only} | {bm25_only} | {miss} |".format(
                group=_cell(key),
                n=item["question_count"],
                small="yes" if item["small_n"] else "no",
                dense=_count_text(dense_hit),
                bm25=_count_text(bm25_hit),
                both=item["cells"][CELL_BOTH],
                dense_only=item["cells"][CELL_DENSE],
                bm25_only=item["cells"][CELL_BM25],
                miss=item["cells"][CELL_MISS],
            )
        )
    return "\n".join(body)


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    header = "| metric | Dense | BM25 |"
    sep = "| --- | --- | --- |"
    body = [header, sep]
    for row in rows:
        role = " (diagnostic)" if row["role"] == "diagnostic" else ""
        variant = " condensed" if row.get("ndcg_variant") else ""
        body.append(
            f"| {row['metric']}{variant}{role} | {_pair(row, 'dense')} | {_pair(row, 'bm25')} |"
        )
    return "\n".join(body)


def _pair(row: dict[str, Any], side: str) -> str:
    numerator = row[f"{side}_numerator"]
    denominator = row[f"{side}_denominator"]
    value = row[f"{side}_value"]
    rendered = "n/a" if value is None else f"{value:.4f}"
    if numerator is None:
        return rendered
    return f"{numerator}/{denominator} ({rendered})"


def _count_text(block: dict[str, Any]) -> str:
    if block.get("value") is None:
        return "n/a"
    return f"{block['numerator']}/{block['denominator']}"


def _one_sided_note(
    hit_name: str, hit_rank: int | None, miss_name: str, miss_rank: int | None
) -> str:
    hit_text = f"{hit_name} 첫 양성 rank는 {hit_rank}"
    if miss_rank is None:
        return f"{hit_text}이고 {miss_name} 반환 순위에는 양성 unit이 없다."
    return (
        f"{hit_text}이고 {miss_name} 첫 양성 rank는 {miss_rank}이라 "
        f"상위 {HIT_K} 밖이다."
    )


def _dig(payload: dict[str, Any], path: tuple[str, ...]) -> dict[str, Any]:
    cursor: Any = payload
    for key in path:
        cursor = cursor[key]
    if not isinstance(cursor, dict):
        raise UsageError("headline metric path is not a block")
    return cursor


def surface_terms(text: str) -> list[str]:
    """Diagnostic surface forms. This is not the BM25 tokenizer."""
    found: list[str] = []
    seen: set[str] = set()
    normalized = unicodedata.normalize("NFKC", text or "")
    for match in _SURFACE_RE.finditer(normalized):
        term = match.group(0)
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        found.append(term)
    return found


def terms_in_text(terms: list[str], text: str) -> list[str]:
    folded = unicodedata.normalize("NFKC", text or "").casefold()
    return [term for term in terms if term.casefold() in folded]


def table_markers(text: str) -> list[str]:
    marks = []
    if " | " in (text or ""):
        marks.append("pipe_row")
    if "단위=" in (text or ""):
        marks.append("unit_header")
    if "기간=" in (text or ""):
        marks.append("period_header")
    return marks


def _excerpt(text: str) -> str:
    compact = re.sub(r"\s+", " ", text or "").strip()
    if len(compact) <= EXCERPT_CHARS:
        return compact
    return compact[:EXCERPT_CHARS]


def _cell(value: str) -> str:
    return value.replace("|", "/")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "query_id",
        "company",
        "question",
        "wording",
        "query_type",
        "evidence_form",
        "stratum",
        "period_sensitive",
        "requirement_count",
        "cell",
        "dense_first_relevant_rank",
        "bm25_first_relevant_rank",
        "dense_relevant_unit_ids",
        "bm25_relevant_unit_ids",
        "missing_requirements",
        "failure_note",
    ]
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: _csv_value(row.get(key))
                    for key in columns
                }
            )


def _csv_value(value: Any) -> str:
    if isinstance(value, list):
        return "|".join(str(item) for item in value)
    if value is None:
        return ""
    return str(value)


def _dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

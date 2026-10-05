"""Union of existing retrievers plus rank-bucket tail samples.

Dense-D2a and the repository BM25 index are reused. No new ranker is trained.
"""

from __future__ import annotations

import random
from typing import Any

from disclosure_agent.bm25 import Bm25Index


def select_candidates(
    *,
    query_id: str,
    answerability: str,
    seed_unit_ids: list[str],
    dense_ranked: list[dict[str, Any]],
    bm25_ranked: list[dict[str, Any]],
    units_by_id: dict[str, dict[str, Any]],
    pooling: dict[str, Any],
) -> dict[str, Any]:
    """Deduplicate by retrieval unit. Keep every retriever's rank. Hide that from judges later."""
    reduced = answerability == "unanswerable"
    dense_n = int(pooling["unanswerable_dense_depth"] if reduced else pooling["dense_depth"])
    bm25_n = int(pooling["unanswerable_bm25_depth"] if reduced else pooling["bm25_depth"])
    buckets = [] if reduced and not pooling.get(
        "unanswerable_tail") else list(pooling["tail_buckets"])
    chosen: dict[str, dict[str, Any]] = {}

    def add(unit_id: str, source: dict[str, Any]) -> None:
        unit = units_by_id.get(unit_id)
        if unit is None:
            return
        row = chosen.get(unit_id)
        if row is None:
            row = {
                "retrieval_unit_id": unit_id,
                "element_ids": list(unit["element_ids"]),
                "text": unit["text"],
                "document_name": unit["document_name"],
                "filing_period": unit["filing_period"],
                "section_path": list(unit.get("section_path") or []),
                "receipt_no": unit["receipt_no"],
                "receipt_date": unit.get("receipt_date") or "",
                "sources": [],
            }
            chosen[unit_id] = row
        row["sources"].append(source)

    for unit_id in seed_unit_ids:
        add(unit_id, {"retriever": "seed", "rank": None, "score": None})
    for hit in dense_ranked[:dense_n]:
        add(
            str(hit["retrieval_unit_id"]),
            {
                "retriever": "dense_d2a",
                "rank": int(hit["rank"]),
                "score": _round_score(hit.get("score")),
            },
        )
    for hit in bm25_ranked[:bm25_n]:
        add(
            str(hit["retrieval_unit_id"]),
            {
                "retriever": "bm25_ko_mixed_v1",
                "rank": int(hit["rank"]),
                "score": _round_score(hit.get("score")),
            },
        )
    shortfall: list[str] = []
    already = set(chosen)
    for bucket in buckets:
        picked = _tail_sample(
            query_id, dense_ranked, already, bucket, int(pooling["tail_random_seed"]))
        if len(picked) < int(bucket["n"]):
            shortfall.append(str(bucket["name"]))
        for hit in picked:
            unit_id = str(hit["retrieval_unit_id"])
            add(
                unit_id,
                {
                    "retriever": "dense_d2a_tail",
                    "rank": int(hit["rank"]),
                    "score": _round_score(hit.get("score")),
                    "bucket": bucket["name"],
                },
            )
            already.add(unit_id)
    ordered = list(chosen.values())
    ordered.sort(key=lambda item: item["retrieval_unit_id"])
    rng = random.Random(f"{pooling['tail_random_seed']}:judge:{query_id}")
    rng.shuffle(ordered)
    for index, item in enumerate(ordered, start=1):
        item["candidate_id"] = f"C{index:02d}"
    return {
        "query_id": query_id,
        "candidates": ordered,
        "tail_shortfall": shortfall,
    }


def _tail_sample(
    query_id: str,
    ranked: list[dict[str, Any]],
    already: set[str],
    bucket: dict[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    start = int(bucket["start"])
    end = bucket.get("end")
    window = []
    for hit in ranked:
        rank = int(hit["rank"])
        if rank < start:
            continue
        if end is not None and rank > int(end):
            continue
        if str(hit["retrieval_unit_id"]) in already:
            continue
        window.append(hit)
    rng = random.Random(f"{seed}:{query_id}:{bucket['name']}")
    rng.shuffle(window)
    return window[: int(bucket["n"])]


def _round_score(value: Any) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def bm25_top(units: list[dict[str, Any]], query: str, depth: int) -> list[dict[str, Any]]:
    """Existing Korean BM25. Zero scores stay out of the pool, matching Bm25Index.search."""
    chunks = [
        {
            "chunk_id": unit["retrieval_unit_id"],
            "ordinal": int(unit.get("ordinal") or 0),
            "text": unit["text"],
            "retrieval_unit_id": unit["retrieval_unit_id"],
        }
        for unit in units
    ]
    index = Bm25Index(chunks)
    hits = index.search(query, depth)
    return [
        {
            "retrieval_unit_id": str(hit.get("retrieval_unit_id") or hit["chunk_id"]),
            "rank": int(hit["rank"]),
            "score": float(hit["score"]),
        }
        for hit in hits
    ]

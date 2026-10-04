"""Multi-company Dense-D2a retrieval baseline.

Scores the product search path: company index, correction-aware filing scope,
then exact Dense-D2a. It does not rewrite queries, change top-k, or add another
retriever. Gold is a source element, not a chunk id. Preferred filings are
diagnostic and do not decide Hit or Recall.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from disclosure_agent.agent.search import SearchDisclosuresRequest
from disclosure_agent.agent.search_catalog import CatalogDenseSearch, load_search_catalog
from disclosure_agent.agent.tool_args import resolve_search_call
from disclosure_agent.chunking import canonical_dumps, sha256_text
from disclosure_agent.dense import (
    DENSE_MODEL_ID,
    DENSE_MODEL_REVISION,
    DOCUMENT_ENCODING_POLICY,
    NORMALIZATION,
    QUERY_ENCODING_POLICY,
    SEARCH_METHOD,
    SIMILARITY_METRIC,
)
from disclosure_agent.errors import UsageError
from disclosure_agent.evaluation import git_revision
from disclosure_agent.retrieval_unit import D2A_VERSION

DATASET_ID = "multi_company_retrieval_v1"
SCHEMA_VERSION = 1
TOP_K = 20
HEADLINE_KS = (1, 5, 10)
SMALL_N = 5
INSPECTION_CHARS = 360


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_baseline_dataset(directory: Path) -> dict[str, Any]:
    snapshot = json.loads((directory / "snapshot.json").read_text(encoding="utf-8"))
    questions = json.loads((directory / "questions.json").read_text(encoding="utf-8"))
    evidence = json.loads((directory / "evidence.json").read_text(encoding="utf-8"))
    if snapshot.get("dataset_id") != DATASET_ID:
        raise UsageError(f"dataset_id must be {DATASET_ID}")
    if snapshot.get("schema_version") != SCHEMA_VERSION:
        raise UsageError("unexpected retrieval baseline schema_version")
    rows = questions.get("questions")
    labels = evidence.get("evidence")
    if not isinstance(rows, list) or not rows:
        raise UsageError("questions.json has no questions")
    if not isinstance(labels, list) or not labels:
        raise UsageError("evidence.json has no evidence")
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_ids: set[str] = set()
    for item in labels:
        evidence_id = str(item.get("evidence_id") or "")
        if not evidence_id or evidence_id in seen_ids:
            raise UsageError(f"duplicate or empty evidence_id {evidence_id!r}")
        seen_ids.add(evidence_id)
        relevance = item.get("relevance")
        if relevance not in {0, 1, 2}:
            raise UsageError(f"{evidence_id} relevance must be 0, 1, or 2")
        if not item.get("element_id") or not item.get("receipt_no"):
            raise UsageError(f"{evidence_id} needs element_id and receipt_no")
        if not item.get("quote"):
            raise UsageError(f"{evidence_id} needs a quote checked against the source")
        by_question[str(item["question_id"])].append(item)
    question_ids: set[str] = set()
    for question in rows:
        question_id = str(question.get("question_id") or "")
        if not question_id or question_id in question_ids:
            raise UsageError(f"duplicate or empty question_id {question_id!r}")
        question_ids.add(question_id)
        if question.get("subset") not in {"core", "filing_policy"}:
            raise UsageError(f"{question_id} subset must be core or filing_policy")
        owned = by_question.get(question_id) or []
        if not any(int(item["relevance"]) >= 1 for item in owned):
            raise UsageError(f"{question_id} has no relevance >= 1 evidence")
        elements = [str(item["element_id"]) for item in owned]
        if len(elements) != len(set(elements)):
            raise UsageError(f"{question_id} repeats an element_id")
    extra = set(by_question) - question_ids
    if extra:
        raise UsageError(f"evidence references unknown questions: {sorted(extra)}")
    return {
        "snapshot": snapshot,
        "questions": rows,
        "evidence": labels,
        "by_question": dict(by_question),
        "directory": directory,
    }


def dataset_sha256(dataset: dict[str, Any]) -> str:
    body = {
        "snapshot": dataset["snapshot"],
        "questions": dataset["questions"],
        "evidence": dataset["evidence"],
    }
    return sha256_text(canonical_dumps(body))


def score_question(
    question: dict[str, Any],
    evidence: list[dict[str, Any]],
    hits: list[dict[str, Any]],
    *,
    ks: tuple[int, ...] = HEADLINE_KS,
) -> dict[str, Any]:
    """Score one ranked list. Unlabeled units are relevance 0."""
    by_element = {str(item["element_id"]): item for item in evidence}
    relevant = [item for item in evidence if int(item["relevance"]) >= 1]
    preferred = [item for item in relevant if item.get("preferred")]
    annotated = [_annotate_hit(hit, by_element) for hit in hits]
    at_k = {}
    for k in ks:
        window = annotated[:k]
        found = _found_evidence(window)
        found_ids = {item["evidence_id"] for item in found}
        preferred_found = [item for item in preferred if item["evidence_id"] in found_ids]
        at_k[str(k)] = {
            "hit": bool(found),
            "recall": _ratio(len(found), len(relevant)),
            "relevant_found": len(found),
            "relevant_total": len(relevant),
            "preferred_hit": bool(preferred_found) if preferred else None,
            "ndcg": _ndcg(window, relevant, k),
        }
    return {
        "question_id": question["question_id"],
        "company_name": question.get("company_name"),
        "corp_code": question.get("corp_code"),
        "query_type": question.get("query_type"),
        "wording": question.get("wording"),
        "evidence_form": question.get("evidence_form"),
        "period_sensitive": bool(question.get("period_sensitive")),
        "subset": question.get("subset"),
        "at_k": at_k,
        "mrr": _mrr(annotated),
        "first_relevant_rank": _first_rank(annotated, minimum=1),
        "first_direct_rank": _first_rank(annotated, minimum=2),
        "first_preferred_rank": _first_preferred_rank(annotated),
        "duplicate_relevant_hits_at_10": _duplicate_relevant_hits(annotated[:10]),
        "hits": annotated,
    }


def aggregate(rows: list[dict[str, Any]], *, population: str) -> dict[str, Any]:
    """Macro-average questions. Micro recall keeps the pooled evidence counts."""
    ks = {}
    for k in HEADLINE_KS:
        key = str(k)
        hits = sum(1 for row in rows if row["at_k"][key]["hit"])
        preferred_rows = [row for row in rows if row["at_k"][key]["preferred_hit"] is not None]
        preferred_hits = sum(1 for row in preferred_rows if row["at_k"][key]["preferred_hit"])
        micro_found = sum(row["at_k"][key]["relevant_found"] for row in rows)
        micro_total = sum(row["at_k"][key]["relevant_total"] for row in rows)
        ks[key] = {
            "hit": _mean_block(hits, len(rows)),
            "recall_macro": _mean_ratio(
                [row["at_k"][key]["recall"]["value"] for row in rows]
            ),
            "recall_micro": _ratio(micro_found, micro_total),
            "ndcg": _mean_ratio([row["at_k"][key]["ndcg"] for row in rows]),
            "preferred_hit": _mean_block(preferred_hits, len(preferred_rows)),
        }
        ks[key]["preferred_hit"]["role"] = "diagnostic"
        ks[key]["preferred_hit"]["note"] = (
            "Preferred filing or section was retrieved. "
            "A miss here is not a retrieval failure when other valid evidence was retrieved."
        )
    return {
        "population": population,
        "question_count": len(rows),
        "small_n": len(rows) < SMALL_N,
        "mrr": _mean_ratio([row["mrr"] for row in rows]),
        "mrr_cutoff": TOP_K,
        "at_k": ks,
    }


def breakdown(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(field))].append(row)
    return {
        key: aggregate(value, population=f"{field}={key}")
        for key, value in sorted(groups.items())
    }


def run_baseline(
    *,
    dataset_dir: Path,
    output_dir: Path,
    root: Path | None = None,
) -> dict[str, Any]:
    base = root or repo_root()
    dataset = load_baseline_dataset(dataset_dir)
    catalog = load_search_catalog(base)
    _assert_corpus_pin(dataset["snapshot"], catalog)
    units = _load_unit_index(catalog)
    _assert_evidence_in_corpus(dataset, units)
    search = CatalogDenseSearch(catalog)
    scored = []
    scopes = []
    for question in dataset["questions"]:
        scope, hits = _search_question(question, catalog, search)
        scopes.append(scope)
        row = score_question(question, dataset["by_question"][question["question_id"]], hits)
        row["search_scope"] = scope
        row["expected_evidence"] = [
            _evidence_brief(item) for item in dataset["by_question"][question["question_id"]]
        ]
        scored.append(row)
    core = [row for row in scored if row["subset"] == "core"]
    policy = [row for row in scored if row["subset"] == "filing_policy"]
    revision = git_revision(base)
    payload = {
        "dataset_id": DATASET_ID,
        "dataset_sha256": dataset_sha256(dataset),
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": revision["git_commit"],
        "git_dirty": revision["git_dirty"],
        "retrieval": _retrieval_config(),
        "corpora": dataset["snapshot"]["corpora"],
        "filing_policy_rule": dataset["snapshot"]["filing_policy"],
        "metric_definitions": _metric_definitions(),
        "headline": aggregate(core, population="core"),
        "filing_policy_subset": aggregate(policy, population="filing_policy"),
        "breakdowns": {
            "company": breakdown(core, "company_name"),
            "query_type": breakdown(core, "query_type"),
            "wording": breakdown(core, "wording"),
            "evidence_form": breakdown(core, "evidence_form"),
            "period_sensitive": breakdown(core, "period_sensitive"),
        },
        "questions": [_public_question(row) for row in scored],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "metrics.json", payload)
    _write_json(output_dir / "per_question.json", {"questions": payload["questions"]})
    inspection = output_dir / "inspection.jsonl"
    with inspection.open("w", encoding="utf-8", newline="\n") as handle:
        for row in scored:
            handle.write(json.dumps(_inspection(row), ensure_ascii=False) + "\n")
    payload["paths"] = {
        "metrics": str(output_dir / "metrics.json"),
        "per_question": str(output_dir / "per_question.json"),
        "inspection": str(inspection),
    }
    return payload


def _search_question(
    question: dict[str, Any],
    catalog: Any,
    search: CatalogDenseSearch,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    corp = str(question["corp_code"])
    corpus = catalog.corpus(corp)
    if corpus is None:
        raise UsageError(f"{question['question_id']} company is not in the product catalog")
    filters = question.get("filters") or {}
    raw = {
        "query": question["text"],
        "company_id": corp,
        "period": filters.get("period"),
        "filing_date_from": None,
        "filing_date_to": None,
        "report_types": filters.get("report_types"),
        "as_of": filters.get("as_of"),
        "top_k": TOP_K,
    }
    tool = resolve_search_call(
        raw,
        company_id=corp,
        filings=corpus.filings,
        corpus_as_of=corpus.corpus_as_of,
        max_top_k=TOP_K,
        relations=catalog.relations_for(corp),
    )
    if tool.status != "ok" or tool.filings is None:
        raise UsageError(
            f"{question['question_id']} filing scope failed: {tool.status} {tool.message}"
        )
    receipts = tuple(item.receipt_no for item in tool.filings)
    result = search(
        SearchDisclosuresRequest(
            query=question["text"],
            company_id=corp,
            corp_code=corp,
            top_k=TOP_K,
            eligible_receipt_nos=receipts,
        )
    )
    scope = {
        "corp_code": corp,
        "index_id": str(corpus.manifest.get("corpus_id") or ""),
        "corpus_as_of": corpus.corpus_as_of.isoformat(),
        "as_of": filters.get("as_of"),
        "candidate_filing_count": len(corpus.filings),
        "effective_receipt_nos": list(receipts),
        "top_k": TOP_K,
    }
    hits = [_hit_record(hit) for hit in result.hits]
    return scope, hits


def _hit_record(hit: Any) -> dict[str, Any]:
    return {
        "rank": int(hit.rank),
        "score": float(hit.score),
        "retrieval_unit_id": hit.retrieval_unit_id,
        "receipt_no": hit.receipt_no,
        "document_name": hit.document_name,
        "filing_period": hit.filing_period,
        "section_path": list(hit.section_path),
        "source_element_ids": list(hit.source_element_ids),
        "text": hit.text,
    }


def _annotate_hit(
    hit: dict[str, Any], by_element: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    matched = []
    for element_id in hit.get("source_element_ids") or []:
        item = by_element.get(str(element_id))
        if item is not None:
            matched.append(item)
    relevance = max((int(item["relevance"]) for item in matched), default=0)
    return {
        "rank": int(hit["rank"]),
        "score": hit.get("score"),
        "retrieval_unit_id": hit.get("retrieval_unit_id"),
        "receipt_no": hit.get("receipt_no"),
        "document_name": hit.get("document_name"),
        "filing_period": hit.get("filing_period"),
        "section_path": list(hit.get("section_path") or []),
        "source_element_ids": list(hit.get("source_element_ids") or []),
        "text": hit.get("text") or "",
        "matched_evidence_ids": [item["evidence_id"] for item in matched],
        "matched_grades": {
            item["evidence_id"]: int(item["relevance"]) for item in matched
        },
        "relevance": relevance,
        "preferred": any(
            bool(item.get("preferred")) and int(item["relevance"]) >= 1 for item in matched
        ),
    }


def _found_evidence(hits: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Positive evidence introduced by the window. Relevance 0 labels do not count."""
    found: list[dict[str, str]] = []
    seen: set[str] = set()
    for hit in hits:
        if int(hit["relevance"]) < 1:
            continue
        for evidence_id in hit["matched_evidence_ids"]:
            if evidence_id in seen:
                continue
            grade = _grade_on_hit(hit, evidence_id)
            if grade < 1:
                continue
            seen.add(evidence_id)
            found.append({"evidence_id": evidence_id})
    return found


def _grade_on_hit(hit: dict[str, Any], evidence_id: str) -> int:
    grades = hit.get("matched_grades") or {}
    if evidence_id in grades:
        return int(grades[evidence_id])
    return int(hit["relevance"])


def _ndcg(hits: list[dict[str, Any]], relevant: list[dict[str, Any]], k: int) -> float:
    gains = []
    seen: set[str] = set()
    for hit in hits[:k]:
        fresh = [
            evidence_id
            for evidence_id in hit["matched_evidence_ids"]
            if evidence_id not in seen
        ]
        rel = 0
        for evidence_id in fresh:
            seen.add(evidence_id)
            grade = _relevance_of(evidence_id, relevant)
            if grade > rel:
                rel = grade
        gains.append(_gain(rel))
    ideal = [_gain(int(item["relevance"])) for item in relevant]
    ideal.sort(reverse=True)
    ideal = ideal[:k]
    idcg = _dcg(ideal)
    if idcg == 0:
        return 0.0
    return _dcg(gains) / idcg


def _relevance_of(evidence_id: str, relevant: list[dict[str, Any]]) -> int:
    for item in relevant:
        if item["evidence_id"] == evidence_id:
            return int(item["relevance"])
    return 0


def _gain(relevance: int) -> float:
    if relevance <= 0:
        return 0.0
    return float((2**relevance) - 1)


def _dcg(gains: list[float]) -> float:
    total = 0.0
    for index, gain in enumerate(gains):
        total += gain / math.log2(index + 2)
    return total


def _mrr(hits: list[dict[str, Any]]) -> float:
    for hit in hits[:TOP_K]:
        if int(hit["relevance"]) >= 1:
            return 1.0 / int(hit["rank"])
    return 0.0


def _first_rank(hits: list[dict[str, Any]], *, minimum: int) -> int | None:
    for hit in hits:
        if int(hit["relevance"]) >= minimum:
            return int(hit["rank"])
    return None


def _first_preferred_rank(hits: list[dict[str, Any]]) -> int | None:
    for hit in hits:
        if hit["preferred"] and int(hit["relevance"]) >= 1:
            return int(hit["rank"])
    return None


def _duplicate_relevant_hits(hits: list[dict[str, Any]]) -> int:
    seen: set[str] = set()
    duplicates = 0
    for hit in hits:
        ids = [item for item in hit["matched_evidence_ids"] if _kept(item)]
        if int(hit["relevance"]) < 1 or not ids:
            continue
        if all(item in seen for item in ids):
            duplicates += 1
        seen.update(ids)
    return duplicates


def _kept(evidence_id: str) -> bool:
    return bool(evidence_id)


def _ratio(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": None if denominator == 0 else numerator / denominator,
    }


def _mean_block(numerator: int, denominator: int) -> dict[str, Any]:
    block = _ratio(numerator, denominator)
    block["small_n"] = denominator < SMALL_N
    return block


def _mean_ratio(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"numerator": 0, "denominator": 0, "value": None, "small_n": True}
    return {
        "numerator": None,
        "denominator": len(values),
        "value": sum(values) / len(values),
        "small_n": len(values) < SMALL_N,
    }


def _assert_corpus_pin(snapshot: dict[str, Any], catalog: Any) -> None:
    for pin in snapshot["corpora"]:
        corpus = catalog.corpus(str(pin["corp_code"]))
        if corpus is None:
            raise UsageError(f"pinned corpus is not loaded: {pin['corp_code']}")
        actual = str(corpus.manifest.get("corpus_id") or "")
        if actual != pin["corpus_id"]:
            raise UsageError(f"corpus_id changed for {pin['corp_code']}: {actual}")
        if corpus.corpus_as_of.isoformat() != pin["corpus_as_of"]:
            raise UsageError(f"corpus_as_of changed for {pin['corp_code']}")
        counted = int(corpus.manifest.get("retrieval_unit_count") or -1)
        if counted != int(pin["retrieval_unit_count"]):
            raise UsageError(f"retrieval_unit_count changed for {pin['corp_code']}")
        if str(corpus.manifest.get("embedding_model") or "") != DENSE_MODEL_ID:
            raise UsageError(f"embedding model changed for {pin['corp_code']}")
        if str(corpus.manifest.get("representation_version") or "") != D2A_VERSION:
            raise UsageError(f"representation changed for {pin['corp_code']}")


def _load_unit_index(catalog: Any) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """element_id -> units, per company. Used only to prove gold text is indexed."""
    found: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for corpus in catalog.corpora:
        corp = corpus.binding.corp_code
        by_element: dict[str, list[dict[str, Any]]] = defaultdict(list)
        with corpus.binding.units_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                unit = json.loads(line)
                brief = {
                    "receipt_no": str(unit.get("receipt_no") or ""),
                    "text": str(unit.get("text") or ""),
                }
                for element_id in unit.get("source_element_ids") or []:
                    by_element[str(element_id)].append(brief)
        found[corp] = by_element
    return found


def _assert_evidence_in_corpus(dataset: dict[str, Any], units: dict[str, Any]) -> None:
    questions = {item["question_id"]: item for item in dataset["questions"]}
    for item in dataset["evidence"]:
        if int(item["relevance"]) < 1:
            continue
        question = questions[item["question_id"]]
        corp = str(question["corp_code"])
        matches = units.get(corp, {}).get(str(item["element_id"])) or []
        same_receipt = [unit for unit in matches if unit["receipt_no"] == item["receipt_no"]]
        if not same_receipt:
            raise UsageError(
                f"{item['evidence_id']} element is not in an indexed unit "
                f"for receipt {item['receipt_no']}"
            )
        quote = str(item["quote"])
        if not any(quote in unit["text"] for unit in same_receipt):
            raise UsageError(f"{item['evidence_id']} quote is not in the indexed unit text")


def _retrieval_config() -> dict[str, Any]:
    return {
        "implementation": "product_catalog_dense_d2a",
        "retriever_type": "dense",
        "representation_version": D2A_VERSION,
        "embedding_model": DENSE_MODEL_ID,
        "embedding_model_revision": DENSE_MODEL_REVISION,
        "query_encoding_model": DENSE_MODEL_ID,
        "query_encoding_policy": QUERY_ENCODING_POLICY,
        "document_encoding_policy": DOCUMENT_ENCODING_POLICY,
        "normalization": NORMALIZATION,
        "similarity_metric": SIMILARITY_METRIC,
        "search_method": SEARCH_METHOD,
        "top_k": TOP_K,
        "filters": (
            "Company is given. Query text is the raw question. "
            "Effective receipts use select_filings with no period and no report type, "
            "unless a question sets as_of. Correction originals are suppressed when the "
            "correction is already in scope. No query rewrite, hybrid, or reranker."
        ),
        "bm25": False,
        "hybrid": False,
        "rrf": False,
        "reranker": False,
    }


def _metric_definitions() -> dict[str, str]:
    return {
        "hit@k": (
            "1 when any evidence with relevance >= 1 has its element inside a top-k unit. "
            "Relevance 1 is a hit. Preferred filing is not required."
        ),
        "recall@k": (
            "Macro: unweighted mean of per-question |retrieved relevant evidence| / "
            "|relevant evidence|. Micro: the same counts pooled. "
            "A unit retrieves an evidence item when source_element_ids contains that element."
        ),
        "mrr": (
            f"Mean of 1/rank for the first relevance >= 1 unit. "
            f"Unfound inside the retrieved cutoff {TOP_K} is 0."
        ),
        "ndcg@k": (
            "Graded NDCG. Gain is 2^relevance - 1, so relevance 2 is 3 and relevance 1 is 1. "
            "A repeated element adds gain only the first time it appears. "
            "The ideal list is the question's relevant grades, sorted, truncated to k."
        ),
        "preferred_hit@k": (
            "Diagnostic. 1 when a preferred evidence element is inside top-k. "
            "Not part of Hit, Recall, MRR, or NDCG."
        ),
    }


def _evidence_brief(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_id": item["evidence_id"],
        "relevance": item["relevance"],
        "preferred": bool(item.get("preferred")),
        "receipt_no": item["receipt_no"],
        "filing_period": item.get("filing_period"),
        "document_name": item.get("document_name"),
        "section": item.get("section"),
        "element_id": item["element_id"],
        "quote": item["quote"],
    }


def _public_question(row: dict[str, Any]) -> dict[str, Any]:
    body = dict(row)
    hits = []
    for hit in row["hits"]:
        item = dict(hit)
        item["text"] = _clip(str(item.get("text") or ""))
        hits.append(item)
    body["hits"] = hits
    return body


def _inspection(row: dict[str, Any]) -> dict[str, Any]:
    flags = []
    if not row["at_k"]["10"]["hit"]:
        flags.append("evidence_missing_at_10")
    if row["first_direct_rank"] is None or int(row["first_direct_rank"]) > 10:
        flags.append("direct_evidence_missing_at_10")
    if row["at_k"]["10"]["preferred_hit"] is False:
        flags.append("preferred_missing_at_10")
    if row["first_relevant_rank"] is not None and int(row["first_relevant_rank"]) > 5:
        flags.append("first_relevant_after_5")
    if int(row["duplicate_relevant_hits_at_10"]) >= 3:
        flags.append("duplicate_relevant_hits")
    return {
        "question_id": row["question_id"],
        "auto_flags": flags,
        "first_relevant_rank": row["first_relevant_rank"],
        "first_direct_rank": row["first_direct_rank"],
        "first_preferred_rank": row["first_preferred_rank"],
        "expected_evidence": row["expected_evidence"],
        "top10": [_clip_hit(hit) for hit in row["hits"][:10]],
    }


def _clip_hit(hit: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": hit["rank"],
        "score": hit["score"],
        "relevance": hit["relevance"],
        "preferred": hit["preferred"],
        "matched_evidence_ids": hit["matched_evidence_ids"],
        "receipt_no": hit["receipt_no"],
        "filing_period": hit["filing_period"],
        "section": " > ".join(hit["section_path"]),
        "retrieval_unit_id": hit["retrieval_unit_id"],
        "text": _clip(str(hit["text"])),
    }


def _clip(text: str) -> str:
    compact = " ".join(text.split())
    if len(compact) <= INSPECTION_CHARS:
        return compact
    return compact[: INSPECTION_CHARS - 1] + "…"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score the product Dense-D2a retrieval baseline")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=repo_root() / "evaluation" / "datasets" / DATASET_ID,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root() / "evaluation" / "runs" / f"{DATASET_ID}_dense_d2a",
    )
    args = parser.parse_args(argv)
    run_baseline(dataset_dir=args.dataset, output_dir=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Gold source-span alignment and the fixed-snapshot BM25 baseline.

Gold labels are source spans (element id + paragraph or cell coordinates).
Chunk ids are never the answer key. Q01 and Q09 share one intent group.
Q10 is outside the positive Recall denominator.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from disclosure_agent.bm25 import (
    BM25_PARAMS,
    TOKENIZER_VERSION,
    Bm25Index,
    index_manifest,
    tokenize,
)
from disclosure_agent.chunking import (
    CHUNKER_VERSION,
    assert_search_eligible,
    chunk_elements,
    chunk_manifest_summary,
    chunker_config_hash,
    load_elements,
    load_json,
    member_for_document,
    sha256_text,
)
from disclosure_agent.errors import UsageError

KS = (1, 3, 5, 10, 20)
FAILURE_K = 5
CAUSE_PARSING = "parsing_loss"
CAUSE_CHUNKING = "chunking_loss"
CAUSE_INDEX = "index_eligibility_loss"
CAUSE_RETRIEVAL = "retrieval_ranking_failure"
CAUSE_ANSWERABILITY = "answerability_limitation"
CAUSE_CALCULATION = "calculation_issue"
CAUSE_EVALUATION = "evaluation_bug"
EVALUATION_SCHEMA_VERSION = 2
EVIDENCE_CLASSES = (
    "relevant_conflicting",
    "irrelevant",
    "context_only",
    "invalid_for_answer",
)
_HIT_CLASS = {
    "relevant_conflicting": "relevant_conflicting_evidence",
    "context_only": "context_only",
    "invalid_for_answer": "invalid_for_answer",
}
_POSITIVE_HIT_CLASS = "positive_answer_evidence"
_TEN = Decimal("0.0000000001")
_TWO = Decimal("0.01")


def load_dataset(directory: Path) -> dict[str, Any]:
    snapshot = load_json(directory / "snapshot.json")
    questions = load_json(directory / "questions.json")
    claims = load_json(directory / "claims.json")
    hard_negatives = load_json(directory / "hard_negatives.json")
    by_id = {claim["claim_id"]: claim for claim in claims["claims"]}
    if len(by_id) != len(claims["claims"]):
        raise UsageError("duplicate claim_id in the evaluation dataset")
    _validate_evidence_labels(hard_negatives["hard_negatives"])
    return {
        "snapshot": snapshot,
        "questions": questions["questions"],
        "claims": by_id,
        "hard_negatives": hard_negatives["hard_negatives"],
        "directory": str(directory),
    }


def verify_snapshot(dataset: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    assert_search_eligible(manifest)
    snap = dataset["snapshot"]
    expected = {
        "receipt_no": snap["receipt_no"],
        "source_sha256": snap["source_sha256"],
        "parser_version": snap["parser_version"],
        "preprocess_version": snap["preprocess_version"],
        "config_hash": snap["config_hash"],
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise UsageError(
                f"pinned snapshot {key} does not match the parse manifest "
                f"({value!r} != {manifest.get(key)!r}). Labels were not guessed."
            )
    scope = snap["index_scope"]
    member = member_for_document(manifest, scope["document_id"])
    if member.get("member_index") != scope["member_index"]:
        raise UsageError("index member_index does not match the pinned snapshot")
    if member.get("member_name") != scope["member_name"]:
        raise UsageError("index member_name does not match the pinned snapshot")
    if member.get("document_name") != scope["document_name_observed"]:
        raise UsageError(
            "observed document_name does not match the pinned snapshot. "
            "role=unknown is not used as a document-type label."
        )
    if member.get("byte_sha256") != snap.get("member_byte_sha256"):
        raise UsageError("ZIP member byte SHA-256 does not match the pinned snapshot")
    if manifest.get("element_count") != snap.get("element_count"):
        raise UsageError("element_count does not match the pinned snapshot")
    return member


def verify_gold_texts(dataset: dict[str, Any], elements: list[dict[str, Any]]) -> None:
    by_id = {element["element_id"]: element for element in elements}
    for claim in dataset["claims"].values():
        for span in claim.get("spans") or []:
            element = by_id.get(span["element_id"])
            if element is None:
                raise UsageError(
                    f"{claim['claim_id']} span {span['span_id']} "
                    "element is missing from elements.jsonl"
                )
            if span["kind"] == "paragraph":
                text = element.get("text") or ""
                if text != span.get("text"):
                    raise UsageError(f"{span['span_id']} paragraph text does not match the element")
                if span.get("char_end") != len(text):
                    raise UsageError(f"{span['span_id']} char_end does not match the paragraph")
                continue
            cells = {
                (int(cell["start_row"]), int(cell["start_col"])): cell.get("text") or ""
                for cell in (element.get("table") or {}).get("cells") or []
            }
            for wanted in span.get("cells") or []:
                key = (int(wanted["start_row"]), int(wanted["start_col"]))
                if key not in cells:
                    raise UsageError(f"{span['span_id']} cell {key} is missing")
                if cells[key] != wanted.get("text"):
                    raise UsageError(
                        f"{span['span_id']} cell {key} text does not match the element"
                    )


def positive_questions(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in dataset["questions"] if item.get("in_positive_recall_denominator")]


def claims_for_question(dataset: dict[str, Any], question: dict[str, Any]) -> list[dict[str, Any]]:
    found = []
    for claim_id in question.get("claim_ids") or []:
        claim = dataset["claims"].get(claim_id)
        if claim is None:
            raise UsageError(f"{question['question_id']} references missing claim {claim_id}")
        found.append(claim)
    return found


def required_claims(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [claim for claim in claims if claim.get("required")]


def evidence_paths_for_question(
    dataset: dict[str, Any], question: dict[str, Any]
) -> list[dict[str, Any]]:
    """OR paths that can answer the question. One path is enough.

    Questions without ``evidence_paths`` keep a single path of their required
    claims, so the legacy AND metric and the path metric match.
    """
    declared = question.get("evidence_paths")
    if declared:
        loaded = []
        for spec in declared:
            claims = []
            for claim_id in spec.get("claim_ids") or []:
                claim = dataset["claims"].get(claim_id)
                if claim is None:
                    raise UsageError(
                        f"{question['question_id']} evidence path {spec.get('path_id')} "
                        f"references missing claim {claim_id}"
                    )
                claims.append(claim)
            if not claims:
                raise UsageError(
                    f"{question['question_id']} evidence path {spec.get('path_id')} has no claims"
                )
            loaded.append(
                {
                    "path_id": spec["path_id"],
                    "claim_ids": [claim["claim_id"] for claim in claims],
                    "claims": claims,
                }
            )
        return loaded
    claims = required_claims(claims_for_question(dataset, question))
    if not claims:
        return []
    return [
        {
            "path_id": "required_claims",
            "claim_ids": [claim["claim_id"] for claim in claims],
            "claims": claims,
        }
    ]


def _unique_claims(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    found = []
    seen: set[str] = set()
    for claim in claims:
        claim_id = claim["claim_id"]
        if claim_id in seen:
            continue
        seen.add(claim_id)
        found.append(claim)
    return found


def select_evidence_path(path_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick one path. A fully covered path wins; otherwise the best partial path.

    Declared order breaks ties. Later paths are not added into the denominator.
    """
    if not path_rows:
        return None
    covered = [row for row in path_rows if row["path_set_satisfied"]]
    if covered:
        return covered[0]

    def sort_key(row: dict[str, Any]) -> tuple[float, int]:
        total = int(row["required_claims"])
        ratio = -1.0 if total == 0 else row["claim_set_full"] / total
        return (ratio, -path_rows.index(row))

    return max(path_rows, key=sort_key)


def coverage_level(sources: list[dict[str, Any]], claim: dict[str, Any]) -> str:
    statuses = [_span_status(sources, span) for span in claim.get("spans") or []]
    return _combine(statuses)


def _combine(statuses: list[str]) -> str:
    if not statuses:
        return "none"
    if all(status == "full" for status in statuses):
        return "full"
    if all(status == "none" for status in statuses):
        return "none"
    return "partial"


def _span_status(sources: list[dict[str, Any]], span: dict[str, Any]) -> str:
    matched = [source for source in sources if source.get("element_id") == span["element_id"]]
    if span["kind"] == "paragraph":
        intervals = [
            (int(source.get("char_start") or 0), int(source.get("char_end") or 0))
            for source in matched
            if source.get("kind") == "paragraph"
        ]
        return _interval_status(intervals, int(span.get("char_start") or 0), int(span["char_end"]))
    needed = {(int(cell["start_row"]), int(cell["start_col"])) for cell in span.get("cells") or []}
    if not needed:
        return "none"
    found: set[tuple[int, int]] = set()
    for source in matched:
        for cell in source.get("cells") or []:
            found.add((int(cell["start_row"]), int(cell["start_col"])))
    hit = needed & found
    if hit == needed:
        return "full"
    if hit:
        return "partial"
    return "none"


def _span_accumulator(span: dict[str, Any]) -> dict[str, Any]:
    if span["kind"] == "paragraph":
        return {
            "element_id": span["element_id"],
            "kind": "paragraph",
            "start": int(span.get("char_start") or 0),
            "end": int(span["char_end"]),
            "intervals": [],
        }
    return {
        "element_id": span["element_id"],
        "kind": "table_cells",
        "needed": {
            (int(cell["start_row"]), int(cell["start_col"])) for cell in span.get("cells") or []
        },
        "found": set(),
    }


def _absorb_span(state: dict[str, Any], sources: list[dict[str, Any]]) -> None:
    for source in sources:
        if source.get("element_id") != state["element_id"]:
            continue
        if state["kind"] == "paragraph":
            if source.get("kind") != "paragraph":
                continue
            state["intervals"].append(
                (int(source.get("char_start") or 0), int(source.get("char_end") or 0))
            )
            continue
        for cell in source.get("cells") or []:
            state["found"].add((int(cell["start_row"]), int(cell["start_col"])))


def _accumulator_status(state: dict[str, Any]) -> str:
    if state["kind"] == "paragraph":
        return _interval_status(state["intervals"], state["start"], state["end"])
    needed = state["needed"]
    if not needed:
        return "none"
    hit = needed & state["found"]
    if hit == needed:
        return "full"
    if hit:
        return "partial"
    return "none"


def _interval_status(intervals: list[tuple[int, int]], start: int, end: int) -> str:
    pieces = []
    for left, right in intervals:
        lo, hi = max(left, start), min(right, end)
        if hi > lo:
            pieces.append((lo, hi))
    if not pieces:
        return "none"
    pieces.sort()
    merged: list[list[int]] = []
    for lo, hi in pieces:
        if not merged or lo > merged[-1][1]:
            merged.append([lo, hi])
        else:
            merged[-1][1] = max(merged[-1][1], hi)
    if len(merged) == 1 and merged[0][0] <= start and merged[0][1] >= end:
        return "full"
    return "partial"


def chunk_sources(chunk: dict[str, Any]) -> list[dict[str, Any]]:
    return list(chunk.get("chunk_sources") or [])


def single_and_set(chunks: list[dict[str, Any]], claim: dict[str, Any]) -> dict[str, str]:
    single = "none"
    rank = {"none": 0, "partial": 1, "full": 2}
    for chunk in chunks:
        level = coverage_level(chunk_sources(chunk), claim)
        if rank[level] > rank[single]:
            single = level
        if single == "full":
            break
    pooled: list[dict[str, Any]] = []
    for chunk in chunks:
        pooled.extend(chunk_sources(chunk))
    return {"single_chunk": single, "chunk_set": coverage_level(pooled, claim)}


def operating_margin_change(operands: dict[str, str]) -> dict[str, Any]:
    """Deterministic margin change from the four raw won amounts. Not a retrieval score."""
    revenue_2025 = Decimal(operands["fy2025_revenue"])
    revenue_2024 = Decimal(operands["fy2024_revenue"])
    operating_2025 = Decimal(operands["fy2025_operating_income"])
    operating_2024 = Decimal(operands["fy2024_operating_income"])
    margin_2025 = operating_2025 / revenue_2025 * Decimal(100)
    margin_2024 = operating_2024 / revenue_2024 * Decimal(100)
    delta = margin_2025 - margin_2024
    display_2025 = margin_2025.quantize(_TWO, rounding=ROUND_HALF_UP)
    display_2024 = margin_2024.quantize(_TWO, rounding=ROUND_HALF_UP)
    display_delta = delta.quantize(_TWO, rounding=ROUND_HALF_UP)
    rounded_subtraction = display_2025 - display_2024
    return {
        "independent_of_retrieval": True,
        "formula": (
            "margin = operating_income / revenue * 100; "
            "delta_pp = fy2025_margin - fy2024_margin"
        ),
        "fy2025_margin_exact": format(margin_2025, "f"),
        "fy2024_margin_exact": format(margin_2024, "f"),
        "delta_pp_exact": format(delta, "f"),
        "published_10dp": {
            "fy2025": format(margin_2025.quantize(_TEN, rounding=ROUND_HALF_UP), "f"),
            "fy2024": format(margin_2024.quantize(_TEN, rounding=ROUND_HALF_UP), "f"),
            "delta_pp": format(delta.quantize(_TEN, rounding=ROUND_HALF_UP), "f"),
        },
        "display_2dp_half_up": {
            "fy2025": format(display_2025, "f"),
            "fy2024": format(display_2024, "f"),
            "delta_pp": format(display_delta, "f"),
        },
        "rounded_display_difference": format(rounded_subtraction, "f"),
        "used_rounded_display_difference": False,
    }


def q05_unique_amounts(claim_a: dict[str, Any], claim_b: dict[str, Any]) -> dict[str, Any]:
    """The two facility tables restate one plan. Totals are not added together."""
    totals_a = _total_cells(claim_a)
    totals_b = _total_cells(claim_b)
    return {
        "policy": "do_not_sum_progress_and_plan",
        "progress_totals": totals_a,
        "plan_totals": totals_b,
        "same_totals": totals_a == totals_b and bool(totals_a),
        "naive_sum_forbidden": True,
    }


def _total_cells(claim: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for span in claim.get("spans") or []:
        if span.get("element_key") not in {"E116", "E118"}:
            continue
        for cell in span.get("cells") or []:
            text = cell.get("text") or ""
            if text in {"400,000", "94,512", "305,488"}:
                texts.append(text)
    return texts


def hard_negative_ids(dataset: dict[str, Any]) -> set[str]:
    return {item["element_id"] for item in dataset["hard_negatives"]}


def _validate_evidence_labels(items: list[dict[str, Any]]) -> None:
    """Optional evidence_class on hard negatives. Missing class keeps the legacy label."""
    flags = {
        "relevant_conflicting": (True, False, True),
        "irrelevant": (False, False, False),
        "context_only": (True, False, False),
    }
    for item in items:
        evidence_class = item.get("evidence_class")
        if evidence_class is None:
            continue
        if evidence_class not in EVIDENCE_CLASSES:
            raise UsageError(
                f"{item.get('element_key')} evidence_class {evidence_class!r} is not supported"
            )
        topical = item.get("topical_relevance")
        answer_valid = item.get("answer_valid_evidence")
        conflicting = item.get("conflicting_evidence")
        if answer_valid is not False:
            raise UsageError(
                f"{item.get('element_key')} is a hard negative and cannot be answer-valid evidence"
            )
        if not item.get("classified_for"):
            raise UsageError(
                f"{item.get('element_key')} evidence_class requires classified_for question ids"
            )
        if evidence_class == "invalid_for_answer":
            if conflicting is not False:
                raise UsageError(
                    f"{item.get('element_key')} invalid_for_answer cannot also be conflicting"
                )
            continue
        expected = flags[evidence_class]
        if (topical, answer_valid, conflicting) != expected:
            raise UsageError(
                f"{item.get('element_key')} evidence flags do not match {evidence_class}"
            )


def evidence_labels_for_question(
    dataset: dict[str, Any], question: dict[str, Any]
) -> list[dict[str, Any]]:
    question_id = question["question_id"]
    found = []
    for item in dataset["hard_negatives"]:
        if item.get("evidence_class") not in EVIDENCE_CLASSES:
            continue
        if question_id not in item.get("classified_for", []):
            continue
        found.append(item)
    return found


def _first_element_body_rank(
    ranked_chunks: list[dict[str, Any]], element_id: str
) -> int | None:
    for rank, chunk in enumerate(ranked_chunks, start=1):
        if chunk_contains_element(chunk, element_id, body_only=True):
            return rank
    return None


def chunk_contains_element(
    chunk: dict[str, Any], element_id: str, *, body_only: bool = False
) -> bool:
    for source in chunk_sources(chunk):
        if source.get("element_id") != element_id:
            continue
        if body_only and source.get("role") != "body":
            continue
        return True
    return False


def load_eval_corpus(
    *,
    dataset_dir: Path,
    manifest_path: Path,
    elements_path: Path,
) -> dict[str, Any]:
    """Load the pinned snapshot, verify gold text, and chunk the indexed member."""
    dataset = load_dataset(dataset_dir)
    manifest = load_json(manifest_path)
    member = verify_snapshot(dataset, manifest)
    elements = load_elements(elements_path)
    verify_gold_texts(dataset, elements)
    document_id = dataset["snapshot"]["index_scope"]["document_id"]
    chunks = chunk_elements(elements, manifest=manifest, document_id=document_id)
    chunk_summary = chunk_manifest_summary(chunks)
    scope = {
        "corp_code": dataset["snapshot"]["corp_code"],
        "receipt_no": dataset["snapshot"]["receipt_no"],
        "source_sha256": dataset["snapshot"]["source_sha256"],
        "parser_version": manifest.get("parser_version"),
        "preprocess_version": manifest.get("preprocess_version"),
        "parser_config_hash": manifest.get("config_hash"),
        "document_id": document_id,
        "member_index": member.get("member_index"),
        "member_name": member.get("member_name"),
        "document_name_observed": member.get("document_name"),
        "indexed_members": [
            {
                "member_index": member.get("member_index"),
                "member_name": member.get("member_name"),
                "document_id": document_id,
                "document_name_observed": member.get("document_name"),
            }
        ],
        "excluded_members": dataset["snapshot"]["index_scope"]["excluded_members"],
        "chunk_count": chunk_summary["chunk_count"],
        "not_company_wide_performance": True,
        "eligibility": "parse_manifest.overall_status=success",
        "q09": "excluded from positive Recall; paraphrase of Q01",
        "q10": "excluded from positive Recall; no positive evidence",
    }
    return {
        "dataset": dataset,
        "dataset_dir": dataset_dir,
        "chunks": chunks,
        "document_id": document_id,
        "elements_by_id": {element["element_id"]: element for element in elements},
        "chunk_summary": chunk_summary,
        "scope": scope,
    }


def score_search(
    corpus: dict[str, Any],
    search_fn: Any,
    ks: tuple[int, ...] = KS,
) -> list[dict[str, Any]]:
    """Score every question. ``search_fn`` receives the raw question text."""
    dataset = corpus["dataset"]
    questions = []
    for question in dataset["questions"]:
        hits = search_fn(question["text"])
        questions.append(
            _score_question(
                dataset,
                question,
                corpus["chunks"],
                hits,
                ks,
                elements_by_id=corpus["elements_by_id"],
                indexed_document_id=corpus["document_id"],
            )
        )
    return questions


def reproducibility_record(
    corpus: dict[str, Any], retrieval: dict[str, Any]
) -> dict[str, Any]:
    scope = corpus["scope"]
    chunk_summary = corpus["chunk_summary"]
    revision = git_revision(Path(__file__).resolve().parents[2])
    return {
        "source": {
            "receipt_no": scope["receipt_no"],
            "source_sha256": scope["source_sha256"],
            "parser_version": scope["parser_version"],
            "preprocess_version": scope["preprocess_version"],
            "parser_config_hash": scope["parser_config_hash"],
            "indexed_members": scope["indexed_members"],
            "document_ids": chunk_summary["document_ids"],
        },
        "chunking": {
            "chunker_version": CHUNKER_VERSION,
            "chunker_config_hash": chunker_config_hash(),
            "artifact_sha256": chunk_summary["artifact_sha256"],
            "chunk_id_sha256": chunk_summary["chunk_id_sha256"],
            "chunk_count": chunk_summary["chunk_count"],
            "chunk_id_fields": [
                "chunker_version",
                "chunker_config_hash",
                "parser_config_hash",
                "source_sha256",
                "document_id",
                "ordinal",
                "text_hash",
                "sources.element_id",
                "sources.role",
                "sources.char_start",
                "sources.char_end",
                "sources.cells[start_row,start_col,rowspan,colspan]",
            ],
        },
        "retrieval": retrieval,
        "evaluation": {
            "dataset_id": corpus["dataset"]["snapshot"].get("dataset_id"),
            "dataset_sha256": dataset_file_hash(corpus["dataset_dir"]),
            "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
            "git_commit": revision["git_commit"],
            "git_dirty": revision["git_dirty"],
            "generated_at": datetime.now(UTC).isoformat(),
        },
    }


def write_eval_run(
    output_dir: Path,
    payload: dict[str, Any],
    chunks: list[dict[str, Any]],
    *,
    metric_keys: tuple[str, ...],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "metrics.json",
        {key: payload[key] for key in metric_keys},
    )
    _write_json(output_dir / "per_question.json", {"questions": payload["questions"]})
    chunk_path = output_dir / "chunks.jsonl"
    with chunk_path.open("w", encoding="utf-8", newline="\n") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False, sort_keys=True) + "\n")
    payload["paths"] = {
        "metrics": str(output_dir / "metrics.json"),
        "per_question": str(output_dir / "per_question.json"),
        "chunks": str(chunk_path),
    }
    return payload


def run_baseline(
    *,
    dataset_dir: Path,
    manifest_path: Path,
    elements_path: Path,
    output_dir: Path,
    ks: tuple[int, ...] = KS,
) -> dict[str, Any]:
    corpus = load_eval_corpus(
        dataset_dir=dataset_dir,
        manifest_path=manifest_path,
        elements_path=elements_path,
    )
    index = Bm25Index(corpus["chunks"])
    questions = score_search(
        corpus, lambda text: index.search(text, index.n), ks
    )
    retrieval = {
        "retriever_type": "bm25",
        "tokenizer_version": TOKENIZER_VERSION,
        "k1": BM25_PARAMS["k1"],
        "b": BM25_PARAMS["b"],
        "zero_score_omitted_from_topk": True,
    }
    reproducibility = reproducibility_record(corpus, retrieval)
    payload = {
        "retriever_type": "bm25",
        "dataset": str(dataset_dir),
        "dataset_sha256": reproducibility["evaluation"]["dataset_sha256"],
        "reproducibility": reproducibility,
        "scope": corpus["scope"],
        "chunker_version": CHUNKER_VERSION,
        "chunker_config_hash": chunker_config_hash(),
        "tokenizer_version": TOKENIZER_VERSION,
        "bm25": dict(BM25_PARAMS),
        "chunks": corpus["chunk_summary"],
        "index": index_manifest(index, scope=corpus["scope"]),
        "calculation_q07": _q07_calculation(corpus["dataset"]),
        "metrics": _aggregate(
            corpus["dataset"], questions, ks, index_scope=_metric_scope(corpus["scope"])
        ),
        "questions": questions,
        "failures": _failures(questions, FAILURE_K),
    }
    return write_eval_run(
        output_dir,
        payload,
        corpus["chunks"],
        metric_keys=(
            "retriever_type",
            "dataset",
            "dataset_sha256",
            "reproducibility",
            "scope",
            "chunker_version",
            "chunker_config_hash",
            "tokenizer_version",
            "bm25",
            "chunks",
            "index",
            "calculation_q07",
            "metrics",
            "failures",
        ),
    )


def _path_window_status(
    window: list[dict[str, Any]], path: dict[str, Any]
) -> dict[str, Any]:
    rows = []
    for claim in path["claims"]:
        levels = single_and_set(window, claim)
        rows.append({"claim_id": claim["claim_id"], **levels})
    full = bool(rows) and all(row["chunk_set"] == "full" for row in rows)
    single = bool(rows) and all(row["single_chunk"] == "full" for row in rows)
    return {
        "path_id": path["path_id"],
        "claims": rows,
        "claim_set_full": sum(row["chunk_set"] == "full" for row in rows),
        "required_claims": len(rows),
        "path_set_satisfied": full,
        "path_single_chunk_satisfied": single,
    }


def _role_claims(claims: list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
    made = []
    for claim in claims:
        for span in claim.get("spans") or []:
            if span.get("evidence_role", "value") != role:
                continue
            made.append({"claim_id": span["span_id"], "spans": [span]})
    return made


def _first_any_rank(
    ranked_chunks: list[dict[str, Any]], claims: list[dict[str, Any]]
) -> int | None:
    for rank, chunk in enumerate(ranked_chunks, start=1):
        sources = chunk_sources(chunk)
        if any(coverage_level(sources, claim) != "none" for claim in claims):
            return rank
    return None


def _first_set_full_rank(
    ranked_chunks: list[dict[str, Any]], claims: list[dict[str, Any]]
) -> int | None:
    if not claims:
        return None
    states = {
        claim["claim_id"]: [_span_accumulator(span) for span in claim.get("spans") or []]
        for claim in claims
    }
    done: dict[str, int | None] = {claim["claim_id"]: None for claim in claims}
    for rank, chunk in enumerate(ranked_chunks, start=1):
        sources = chunk_sources(chunk)
        for claim in claims:
            claim_id = claim["claim_id"]
            if done[claim_id] is not None:
                continue
            for state in states[claim_id]:
                _absorb_span(state, sources)
            if _combine([_accumulator_status(state) for state in states[claim_id]]) == "full":
                done[claim_id] = rank
        if all(done[claim["claim_id"]] is not None for claim in claims):
            return max(done[claim["claim_id"]] for claim in claims)
    return None


def _span_ranks(
    ranked_chunks: list[dict[str, Any]], claims: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for claim in claims:
        for span in claim.get("spans") or []:
            rows.append(
                {
                    "span_id": span["span_id"],
                    "claim_id": claim["claim_id"],
                    "evidence_role": span.get("evidence_role", "value"),
                    "first_full_rank": _first_set_full_rank(
                        ranked_chunks, [{"claim_id": span["span_id"], "spans": [span]}]
                    ),
                }
            )
    return rows


def _path_rank_report(
    ranked_chunks: list[dict[str, Any]], paths: list[dict[str, Any]]
) -> dict[str, Any]:
    report = {}
    for path in paths:
        claims = path["claims"]
        value_claims = _role_claims(claims, "value")
        context_claims = _role_claims(claims, "context")
        report[path["path_id"]] = {
            "claim_ids": list(path["claim_ids"]),
            "claim_count": len(claims),
            "first_any_rank": _first_any_rank(ranked_chunks, claims),
            "first_value_set_rank": _first_set_full_rank(ranked_chunks, value_claims),
            "first_context_set_rank": _first_set_full_rank(ranked_chunks, context_claims),
            "first_set_full_rank": _first_set_full_rank(ranked_chunks, claims),
            "spans": _span_ranks(ranked_chunks, claims),
        }
    return report


def _score_question(
    dataset: dict[str, Any],
    question: dict[str, Any],
    chunks: list[dict[str, Any]],
    hits: list[dict[str, Any]],
    ks: tuple[int, ...],
    *,
    elements_by_id: dict[str, dict[str, Any]] | None = None,
    indexed_document_id: str | None = None,
) -> dict[str, Any]:
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    claims = claims_for_question(dataset, question)
    required = required_claims(claims)
    paths = evidence_paths_for_question(dataset, question)
    evidence_claims = _unique_claims([claim for path in paths for claim in path["claims"]])
    if not evidence_claims:
        evidence_claims = required
    ranked_chunks = [by_id[hit["chunk_id"]] for hit in hits if hit["chunk_id"] in by_id]
    path_ranks = _path_rank_report(ranked_chunks, paths)
    labels = evidence_labels_for_question(dataset, question)
    first_conflict_rank = _first_labeled_body_rank(
        ranked_chunks, labels, evidence_class="relevant_conflicting"
    )
    at_k = {}
    for k in ks:
        window = ranked_chunks[:k]
        claim_rows = []
        for claim in required:
            levels = single_and_set(window, claim)
            claim_rows.append(
                {
                    "claim_id": claim["claim_id"],
                    "single_chunk": levels["single_chunk"],
                    "chunk_set": levels["chunk_set"],
                }
            )
        evidence_full = bool(claim_rows) and all(row["chunk_set"] == "full" for row in claim_rows)
        single_full = bool(claim_rows) and all(
            row["single_chunk"] == "full" for row in claim_rows
        )
        path_rows = [_path_window_status(window, path) for path in paths]
        selected = select_evidence_path(path_rows)
        satisfied = [row["path_id"] for row in path_rows if row["path_set_satisfied"]]
        at_k[str(k)] = {
            "claims": claim_rows,
            "claim_set_full": sum(row["chunk_set"] == "full" for row in claim_rows),
            "claim_single_full": sum(row["single_chunk"] == "full" for row in claim_rows),
            "required_claims": len(claim_rows),
            "evidence_set_satisfied": evidence_full,
            "evidence_single_chunk_satisfied": single_full,
            "evidence_paths": path_rows,
            "answer_evidence_success": bool(satisfied),
            "answer_evidence_path_id": None if not satisfied else satisfied[0],
            "satisfied_evidence_path_ids": satisfied,
            "selected_path_id": None if selected is None else selected["path_id"],
            "selected_path_claim_set_full": 0 if selected is None else selected["claim_set_full"],
            "selected_path_required_claims": (
                0 if selected is None else selected["required_claims"]
            ),
            "relevant_conflicting_hit": (
                first_conflict_rank is not None and first_conflict_rank <= k
            ),
        }
    hard_ids = hard_negative_ids(dataset)
    labeled_claims = _unique_claims([*claims, *evidence_claims])
    positive_element_ids = {
        span["element_id"]
        for claim in labeled_claims
        for span in claim.get("spans") or []
    }
    top5 = [
        _hit_view(hit, chunk, evidence_claims, hard_ids, labels)
        for hit, chunk in zip(hits[:5], ranked_chunks[:5], strict=False)
    ]
    top20 = [
        _hit_view(hit, chunk, evidence_claims, hard_ids, labels)
        for hit, chunk in zip(hits[:20], ranked_chunks[:20], strict=False)
    ]
    negatives = _negative_ranks(dataset, question, ranked_chunks, positive_element_ids)
    span_audit = audit_span_failures(
        chunks,
        question,
        claims,
        ranked_chunks,
        k=FAILURE_K,
        elements_by_id=elements_by_id,
        indexed_document_id=indexed_document_id,
    )
    diagnosis = _diagnosis(question, chunks, ranked_chunks, required, span_audit)
    ceiling = question["answerability"] == "supported"
    return {
        "question_id": question["question_id"],
        "intent_group": question["intent_group"],
        "paraphrase_of": question.get("paraphrase_of"),
        "independent_intent": question.get("independent_intent", True),
        "in_positive_recall_denominator": bool(question.get("in_positive_recall_denominator")),
        "answerability": question["answerability"],
        "answer_complete_ceiling": ceiling,
        "positive_claim_count": len(required),
        "evidence_path_ids": [path["path_id"] for path in paths],
        "abstention_generation_measured": False,
        "text": question["text"],
        "at_k": at_k,
        "answer_evidence": {
            "mode": "any_path" if question.get("evidence_paths") else "single_required_claim_path",
            "success_rule": "any_complete_evidence_path",
            "path_ranks": path_ranks,
            "deterministic_calculation_included": False,
        },
        "top5": top5,
        "top20": top20,
        "hard_negative": {
            "in_top5": any(row["in_top5"] for row in negatives),
            "hits": [row for row in negatives if row["in_top5"]],
        },
        "hard_negative_ranks": negatives,
        "evidence_classification": _evidence_classification(
            ranked_chunks,
            evidence_claims,
            path_ranks,
            negatives,
            at_k,
            labels,
        ),
        "span_audit": span_audit,
        "diagnosis": diagnosis,
        "amount_policy": question.get("amount_policy"),
        "notes": question.get("notes"),
        "calculation_independent_of_retrieval": question["question_id"] == "Q07",
    }


def _hit_view(
    hit: dict[str, Any],
    chunk: dict[str, Any],
    required: list[dict[str, Any]],
    hard_ids: set[str],
    labels: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    preview = chunk["text"].replace("\n", " ")
    if len(preview) > 240:
        preview = preview[:240]
    coverage = [
        {"claim_id": claim["claim_id"], **single_and_set([chunk], claim)}
        for claim in required
    ]
    return {
        "rank": hit["rank"],
        "score": round(hit["score"], 6),
        "chunk_id": hit["chunk_id"],
        "ordinal": hit["ordinal"],
        "section_path": list(chunk.get("section_path") or []),
        "sources": [_source_brief(source) for source in chunk_sources(chunk)],
        "positive_evidence": any(row["chunk_set"] != "none" for row in coverage),
        "hard_negative": any(
            source.get("element_id") in hard_ids and source.get("role") == "body"
            for source in chunk_sources(chunk)
        ),
        "evidence_classes": _hit_evidence_classes(chunk, required, labels or []),
        "claim_coverage": coverage,
        "preview": preview,
    }


def _hit_evidence_classes(
    chunk: dict[str, Any],
    evidence_claims: list[dict[str, Any]],
    labels: list[dict[str, Any]],
) -> list[str]:
    classes: list[str] = []
    if evidence_claims and any(
        coverage_level(chunk_sources(chunk), claim) != "none" for claim in evidence_claims
    ):
        classes.append(_POSITIVE_HIT_CLASS)
    for item in labels:
        mapped = _HIT_CLASS.get(item.get("evidence_class"))
        if mapped is None or mapped in classes:
            continue
        if chunk_contains_element(chunk, item["element_id"], body_only=True):
            classes.append(mapped)
    if not classes:
        classes.append("irrelevant")
    return classes


def _first_labeled_body_rank(
    ranked_chunks: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    *,
    evidence_class: str,
) -> int | None:
    ranks = [
        rank
        for item in labels
        if item.get("evidence_class") == evidence_class
        for rank in (_first_element_body_rank(ranked_chunks, item["element_id"]),)
        if rank is not None
    ]
    return None if not ranks else min(ranks)


def _evidence_classification(
    ranked_chunks: list[dict[str, Any]],
    evidence_claims: list[dict[str, Any]],
    path_ranks: dict[str, Any],
    negatives: list[dict[str, Any]],
    at_k: dict[str, Any],
    labels: list[dict[str, Any]],
) -> dict[str, Any]:
    complete = [
        row["first_set_full_rank"]
        for row in path_ranks.values()
        if row.get("first_set_full_rank") is not None
    ]
    watched: list[dict[str, Any]] = []
    seen: set[str] = set()
    for claim in evidence_claims:
        for span in claim.get("spans") or []:
            element_id = span["element_id"]
            if element_id in seen:
                continue
            seen.add(element_id)
            watched.append(
                {
                    "element_key": span.get("element_key"),
                    "element_id": element_id,
                    "role": "answer_evidence",
                    "first_body_rank": _first_element_body_rank(ranked_chunks, element_id),
                }
            )
    for item in labels:
        element_id = item["element_id"]
        if element_id in seen:
            continue
        seen.add(element_id)
        watched.append(
            {
                "element_key": item.get("element_key"),
                "element_id": element_id,
                "role": item.get("evidence_class"),
                "first_body_rank": _first_element_body_rank(ranked_chunks, element_id),
            }
        )
    return {
        "first_positive_answer_evidence_rank": _first_any_rank(ranked_chunks, evidence_claims),
        "first_complete_evidence_path_rank": None if not complete else min(complete),
        "relevant_conflicting_ranks": [
            {
                "element_key": row["element_key"],
                "element_id": row["element_id"],
                "rank": row["rank"],
                "chunk_id": row["chunk_id"],
            }
            for row in negatives
            if row.get("counts_as_relevant_conflict")
        ],
        "element_body_ranks": watched,
        "at_5": {
            "positive_answer_evidence": bool(at_k["5"]["answer_evidence_success"]),
            "relevant_conflicting_evidence": bool(at_k["5"]["relevant_conflicting_hit"]),
        },
    }


def _source_brief(source: dict[str, Any]) -> dict[str, Any]:
    cells = source.get("cells") or []
    return {
        "element_id": source.get("element_id"),
        "role": source.get("role"),
        "kind": source.get("kind"),
        "row_rects": source.get("row_rects"),
        "rows_contiguous": source.get("rows_contiguous"),
        "char_start": source.get("char_start"),
        "char_end": source.get("char_end"),
        "cells": [[cell["start_row"], cell["start_col"]] for cell in cells],
    }


def _negative_ranks(
    dataset: dict[str, Any],
    question: dict[str, Any],
    chunks: list[dict[str, Any]],
    positive_element_ids: set[str],
) -> list[dict[str, Any]]:
    """Rank every labeled hard negative. A hit is never positive evidence."""
    rows = []
    for item in dataset["hard_negatives"]:
        found = None
        for rank, chunk in enumerate(chunks, start=1):
            if chunk_contains_element(chunk, item["element_id"], body_only=True):
                found = (rank, chunk["chunk_id"])
                break
        rank = None if found is None else found[0]
        rows.append(
            {
                "element_key": item["element_key"],
                "element_id": item["element_id"],
                "rank": rank,
                "chunk_id": None if found is None else found[1],
                "in_top5": rank is not None and rank <= 5,
                "labeled_not_gold_for_this_question": question["question_id"]
                in item.get("not_gold_for", []),
                "counted_as_positive_evidence": item["element_id"] in positive_element_ids,
                "evidence_class": item.get("evidence_class"),
                "topical_relevance": item.get("topical_relevance"),
                "answer_valid_evidence": item.get("answer_valid_evidence"),
                "conflicting_evidence": item.get("conflicting_evidence"),
                "counts_as_relevant_conflict": (
                    item.get("evidence_class") == "relevant_conflicting"
                    and question["question_id"] in item.get("classified_for", [])
                ),
            }
        )
    return rows


def _diagnosis(
    question: dict[str, Any],
    chunks: list[dict[str, Any]],
    ranked_chunks: list[dict[str, Any]],
    required: list[dict[str, Any]],
    span_audit: list[dict[str, Any]],
) -> dict[str, Any]:
    claim_states = {
        claim["claim_id"]: [_span_accumulator(span) for span in claim.get("spans") or []]
        for claim in required
    }
    set_ranks = {claim["claim_id"]: None for claim in required}
    single_ranks = {claim["claim_id"]: None for claim in required}
    first_any = None
    for rank, chunk in enumerate(ranked_chunks, start=1):
        sources = chunk_sources(chunk)
        if first_any is None and required and any(
            coverage_level(sources, claim) != "none" for claim in required
        ):
            first_any = rank
        for claim in required:
            claim_id = claim["claim_id"]
            if single_ranks[claim_id] is None and coverage_level(sources, claim) == "full":
                single_ranks[claim_id] = rank
            if set_ranks[claim_id] is None:
                for state in claim_states[claim_id]:
                    _absorb_span(state, sources)
                if (
                    _combine([_accumulator_status(state) for state in claim_states[claim_id]])
                    == "full"
                ):
                    set_ranks[claim_id] = rank
    first_set = None
    if required and all(set_ranks[claim["claim_id"]] is not None for claim in required):
        first_set = max(set_ranks[claim["claim_id"]] for claim in required)
    full_chunk = None
    overlap: list[str] = []
    if required:
        full_chunk = _first_full_chunk(chunks, required[0])
        if full_chunk is not None:
            overlap = sorted(set(tokenize(question["text"])) & set(tokenize(full_chunk["text"])))
    claim_rows = []
    for claim in required:
        possible = any(
            coverage_level(chunk_sources(chunk), claim) == "full" for chunk in chunks
        )
        claim_rows.append(
            {
                "claim_id": claim["claim_id"],
                "first_single_full_rank": single_ranks[claim["claim_id"]],
                "first_set_full_rank": set_ranks[claim["claim_id"]],
                "single_chunk_possible": possible,
            }
        )
    categories = sorted({row["cause"] for row in span_audit})
    partial = question["answerability"] == "partial"
    if partial:
        categories = sorted({*categories, CAUSE_ANSWERABILITY})
    ranked_ids = {chunk["chunk_id"] for chunk in ranked_chunks}
    full_ids = [
        chunk["chunk_id"]
        for chunk in chunks
        if required and coverage_level(chunk_sources(chunk), required[0]) == "full"
    ]
    zero_score = bool(full_ids) and all(chunk_id not in ranked_ids for chunk_id in full_ids)
    return {
        "first_positive_evidence_rank": first_any,
        "first_evidence_set_rank": first_set,
        "zero_score_omitted": zero_score,
        "token_overlap_with_first_claim_full_chunk": len(overlap),
        "token_overlap_terms": overlap,
        "first_claim_full_chunk_id": None if full_chunk is None else full_chunk["chunk_id"],
        "claims": claim_rows,
        "failure_categories": categories,
        "answerability_limitation": partial,
        "out_of_scope": question["answerability"] == "out_of_scope",
        "abstention_generation_measured": False,
    }


def _first_full_chunk(
    chunks: list[dict[str, Any]], claim: dict[str, Any]
) -> dict[str, Any] | None:
    for chunk in chunks:
        if coverage_level(chunk_sources(chunk), claim) == "full":
            return chunk
    return None


def _q07_calculation(dataset: dict[str, Any]) -> dict[str, Any]:
    question = next(item for item in dataset["questions"] if item["question_id"] == "Q07")
    spec = question["calculation"]
    result = operating_margin_change(spec["operands"])
    if result["published_10dp"] != spec["published_10dp"]:
        raise UsageError("Q07 full-precision publication does not match the operand formula")
    if result["display_2dp_half_up"] != spec["display_2dp_half_up"]:
        raise UsageError("Q07 display rounding does not match the operand formula")
    result["matches_reviewed_publication"] = True
    result["operands"] = dict(spec["operands"])
    result["test_name"] = "deterministic_calculation"
    result["status"] = "success"
    result["uses_evidence_path"] = "derived_operands"
    result["retrieval_answer_evidence_included"] = False
    result["precision_note"] = (
        "Canonical display rounds the raw-won delta half-up to 2dp (6.64). "
        "Subtracting the source-rounded margins 18.82 and 12.17 yields 6.65. "
        "This test checks only the raw-won operand path."
    )
    return result


def _aggregate(
    dataset: dict[str, Any],
    questions: list[dict[str, Any]],
    ks: tuple[int, ...],
    *,
    index_scope: dict[str, Any],
) -> dict[str, Any]:
    positive = [item for item in questions if item["in_positive_recall_denominator"]]
    ceiling = [item for item in positive if item["answer_complete_ceiling"]]
    by_k = {}
    for k in ks:
        key = str(k)
        claim_hit = 0
        claim_single = 0
        claim_total = 0
        question_hit = 0
        question_single = 0
        answer_complete = 0
        source_ceiling = 0
        path_hit = 0
        path_total = 0
        path_question_hit = 0
        path_selections = []
        conflict_population = [
            item
            for item in questions
            if any(
                row.get("counts_as_relevant_conflict") for row in item["hard_negative_ranks"]
            )
        ]
        conflict_hit = 0
        for question in positive:
            block = question["at_k"][key]
            claim_hit += block["claim_set_full"]
            claim_single += block["claim_single_full"]
            claim_total += block["required_claims"]
            satisfied = bool(block["evidence_set_satisfied"])
            question_hit += int(satisfied)
            question_single += int(block["evidence_single_chunk_satisfied"])
            source_ceiling += int(satisfied)
            if satisfied and question["answer_complete_ceiling"]:
                answer_complete += 1
            path_hit += block["selected_path_claim_set_full"]
            path_total += block["selected_path_required_claims"]
            path_question_hit += int(block["answer_evidence_success"])
            path_selections.append(
                {
                    "question_id": question["question_id"],
                    "selected_path_id": block["selected_path_id"],
                    "satisfied_evidence_path_ids": block["satisfied_evidence_path_ids"],
                    "numerator": block["selected_path_claim_set_full"],
                    "denominator": block["selected_path_required_claims"],
                    "success": block["answer_evidence_success"],
                }
            )
        for question in conflict_population:
            conflict_hit += int(question["at_k"][key]["relevant_conflicting_hit"])
        conflict_ratio = _ratio(conflict_hit, len(conflict_population), index_scope)
        conflict_ratio["diagnostic"] = True
        conflict_ratio["not_a_quality_score"] = True
        conflict_ratio["question_ids"] = [
            item["question_id"] for item in conflict_population
        ]
        by_k[key] = {
            "claim_set_recall": _ratio(claim_hit, claim_total, index_scope),
            "claim_single_chunk_recall": _ratio(claim_single, claim_total, index_scope),
            "question_evidence_satisfied": _ratio(question_hit, len(positive), index_scope),
            "question_single_chunk_satisfied": _ratio(
                question_single, len(positive), index_scope
            ),
            "question_answer_complete": _ratio(answer_complete, len(ceiling), index_scope),
            "question_source_ceiling_satisfied": _ratio(
                source_ceiling, len(positive), index_scope
            ),
            "answer_evidence_path_recall": _ratio(path_hit, path_total, index_scope),
            "question_answer_evidence_satisfied": _ratio(
                path_question_hit, len(positive), index_scope
            ),
            "relevant_conflict_hit": conflict_ratio,
            "answer_evidence_path_selection": path_selections,
        }
    paraphrase = _paraphrase_delta(questions)
    taxonomy = _taxonomy(questions)
    return {
        "population": "one filing, fixed 10-question regression set, not company-wide performance",
        "positive_question_ids": [item["question_id"] for item in positive],
        "positive_question_count": len(positive),
        "answer_complete_question_ids": [item["question_id"] for item in ceiling],
        "answer_complete_excluded_question_ids": [
            item["question_id"] for item in positive if not item["answer_complete_ceiling"]
        ],
        "answer_complete_definition": (
            "Numerator counts supported questions whose required source spans are fully "
            "covered. Denominator is only answer_complete_ceiling=true. Partial questions "
            "cannot reach a complete answer, so they are outside this denominator."
        ),
        "source_ceiling_definition": (
            "Required source spans are covered. For a partial question this is the document "
            "ceiling of the supported claims. It is not a complete answer and it does not "
            "measure abstention, because there is no generation layer."
        ),
        "metric_semantics": {
            "claim_set_recall": (
                "Legacy AND of question.claim_ids whose claim.required is true. "
                "Alternative evidence paths are not in this denominator. "
                "On a single-path dataset this matches answer_evidence_path_recall."
            ),
            "question_evidence_satisfied": (
                "Legacy question-level AND of those required claims. "
                "Not the OR of evidence paths."
            ),
            "answer_evidence_path_recall": (
                "Each positive question contributes exactly one selected evidence path. "
                "The first fully covered path wins; otherwise the path with the highest "
                "claim-set coverage wins, and declared order breaks ties. "
                "The denominator is the sum of selected-path claim counts, so it can "
                "change with k and is not comparable to the legacy claim denominator."
            ),
            "question_answer_evidence_satisfied": (
                "A positive question succeeds when at least one evidence path is fully "
                "covered. Paths are OR, not AND. Denominator is the positive question count. "
                "This is answer-valid evidence, not topical relevance."
            ),
            "relevant_conflict_hit": (
                "Diagnostic only. Numerator is questions whose labeled "
                "relevant_conflicting element appears at rank <= k. "
                "Denominator is questions that carry that label. "
                "A hit is not retrieval success and a miss is not retrieval failure."
            ),
            "deterministic_calculation": (
                "Q07 calculation_q07 is a separate test. It uses the derived operand path "
                "and is not added into retrieval recall."
            ),
        },
        "intent_group_G01_counted_once": True,
        "q09_excluded_as_paraphrase": True,
        "q10_excluded_from_positive_recall": True,
        "index_scope": index_scope,
        "at_k": by_k,
        "failure_taxonomy_at_5": taxonomy,
        "question_diagnosis": [_question_summary(item) for item in questions],
        "paraphrase_q01_q09": paraphrase,
        "q10": _q10_summary(questions),
    }


def _paraphrase_delta(questions: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {item["question_id"]: item for item in questions}
    left, right = by_id["Q01"], by_id["Q09"]
    shared = left["at_k"]["5"]["claims"]
    other = right["at_k"]["5"]["claims"]
    same_claims = [row["claim_id"] for row in shared] == [row["claim_id"] for row in other]
    left_ids = {hit["chunk_id"] for hit in left["top5"]}
    right_ids = {hit["chunk_id"] for hit in right["top5"]}
    left20 = {hit["chunk_id"] for hit in left["top20"]}
    right20 = {hit["chunk_id"] for hit in right["top20"]}
    left_rank = left["diagnosis"]["first_positive_evidence_rank"]
    right_rank = right["diagnosis"]["first_positive_evidence_rank"]
    left_answer = left["evidence_classification"]["first_positive_answer_evidence_rank"]
    right_answer = right["evidence_classification"]["first_positive_answer_evidence_rank"]
    left_complete = left["evidence_classification"]["first_complete_evidence_path_rank"]
    right_complete = right["evidence_classification"]["first_complete_evidence_path_rank"]
    return {
        "shared_claim_ids": [row["claim_id"] for row in shared],
        "q09_uses_same_claim_ids": same_claims,
        "q01_top1": _top_id(left),
        "q09_top1": _top_id(right),
        "top1_changed": _top_id(left) != _top_id(right),
        "q01_first_positive_evidence_rank": left_rank,
        "q09_first_positive_evidence_rank": right_rank,
        "gold_rank_delta": (
            None if left_rank is None or right_rank is None else right_rank - left_rank
        ),
        "q01_first_positive_answer_evidence_rank": left_answer,
        "q09_first_positive_answer_evidence_rank": right_answer,
        "answer_evidence_rank_delta": (
            None
            if left_answer is None or right_answer is None
            else right_answer - left_answer
        ),
        "q01_first_complete_evidence_path_rank": left_complete,
        "q09_first_complete_evidence_path_rank": right_complete,
        "q01_token_overlap_with_first_claim_full_chunk": left["diagnosis"][
            "token_overlap_with_first_claim_full_chunk"
        ],
        "q09_token_overlap_with_first_claim_full_chunk": right["diagnosis"][
            "token_overlap_with_first_claim_full_chunk"
        ],
        "q09_zero_score_omitted": right["diagnosis"]["zero_score_omitted"],
        "top5_overlap_count": len(left_ids & right_ids),
        "top20_overlap_count": len(left20 & right20),
        "q01_evidence_set_satisfied_at_5": left["at_k"]["5"]["evidence_set_satisfied"],
        "q09_evidence_set_satisfied_at_5": right["at_k"]["5"]["evidence_set_satisfied"],
        "counted_as_independent_intents": 1,
    }


def _question_summary(question: dict[str, Any]) -> dict[str, Any]:
    diagnosis = question["diagnosis"]
    return {
        "question_id": question["question_id"],
        "answerability": question["answerability"],
        "answer_complete_ceiling": question["answer_complete_ceiling"],
        "in_positive_recall_denominator": question["in_positive_recall_denominator"],
        "positive_claim_count": question["positive_claim_count"],
        "first_positive_evidence_rank": diagnosis["first_positive_evidence_rank"],
        "first_evidence_set_rank": diagnosis["first_evidence_set_rank"],
        "zero_score_omitted": diagnosis["zero_score_omitted"],
        "token_overlap_with_first_claim_full_chunk": diagnosis[
            "token_overlap_with_first_claim_full_chunk"
        ],
        "failure_categories": diagnosis["failure_categories"],
        "claim_ranks": diagnosis["claims"],
        "hard_negative_ranks": question["hard_negative_ranks"],
        "evidence_set_satisfied_at_5": question["at_k"]["5"]["evidence_set_satisfied"],
        "evidence_set_satisfied_at_10": question["at_k"]["10"]["evidence_set_satisfied"],
        "evidence_set_satisfied_at_20": question["at_k"]["20"]["evidence_set_satisfied"],
        "answer_evidence_by_k": {
            key: {
                "success": question["at_k"][key]["answer_evidence_success"],
                "path_id": question["at_k"][key]["answer_evidence_path_id"],
                "satisfied_evidence_path_ids": question["at_k"][key][
                    "satisfied_evidence_path_ids"
                ],
                "selected_path_id": question["at_k"][key]["selected_path_id"],
            }
            for key in ("1", "3", "5", "10", "20")
        },
        "answer_evidence_path_ranks": question["answer_evidence"]["path_ranks"],
        "evidence_classification": question["evidence_classification"],
        "diagnostic_flags": _diagnostic_flags(question),
    }


def _q10_summary(questions: list[dict[str, Any]]) -> dict[str, Any]:
    question = next(item for item in questions if item["question_id"] == "Q10")
    return {
        "question_id": "Q10",
        "answerability": question["answerability"],
        "in_positive_recall_denominator": False,
        "positive_claim_count": question["positive_claim_count"],
        "reason": "no_positive_evidence",
        "search_hits_are_not_recall_failures": True,
        "abstention_required": True,
        "abstention_generation_measured": False,
    }


def _taxonomy(questions: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    details: dict[str, int] = {}
    for question in questions:
        if not question["in_positive_recall_denominator"]:
            continue
        for row in question.get("span_audit") or []:
            counts[row["cause"]] = counts.get(row["cause"], 0) + 1
            detail = row.get("detail") or ""
            if detail:
                details[detail] = details.get(detail, 0) + 1
    conflict_ids = [
        question["question_id"]
        for question in questions
        if "relevant_conflicting_evidence_in_top5" in _diagnostic_flags(question)
    ]
    return {
        "k": FAILURE_K,
        "unit": "missed required spans of positive questions",
        "causes": counts,
        "details": details,
        "categories": [
            CAUSE_PARSING,
            CAUSE_CHUNKING,
            CAUSE_INDEX,
            CAUSE_RETRIEVAL,
            CAUSE_ANSWERABILITY,
            CAUSE_CALCULATION,
            CAUSE_EVALUATION,
        ],
        "diagnostics": {
            "relevant_conflicting_evidence_in_top5": {
                "question_ids": conflict_ids,
                "note": (
                    "Topical conflicting evidence inside the top 5. "
                    "Not a missed-span cause and not a quality score."
                ),
            }
        },
    }


def _diagnostic_flags(question: dict[str, Any]) -> list[str]:
    flags = []
    if question["at_k"]["5"].get("relevant_conflicting_hit"):
        flags.append("relevant_conflicting_evidence_in_top5")
    return flags


def _metric_scope(scope: dict[str, Any]) -> dict[str, Any]:
    return {
        "receipt_no": scope["receipt_no"],
        "source_sha256": scope["source_sha256"],
        "member": scope["document_id"],
        "chunk_count": scope["chunk_count"],
        "excluded_from_positive_denominator": [
            "Q09 paraphrase duplicate intent",
            "Q10 no positive evidence",
        ],
    }


def git_revision(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return {"git_commit": None, "git_dirty": None}
    if commit.returncode != 0 or dirty.returncode != 0:
        return {"git_commit": None, "git_dirty": None}
    return {
        "git_commit": commit.stdout.strip() or None,
        "git_dirty": bool(dirty.stdout.strip()),
    }


def stable_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop the clock and output paths so two runs of the same inputs compare equal."""
    body = json.loads(json.dumps(payload))
    body.pop("paths", None)
    evaluation = body.get("reproducibility", {}).get("evaluation", {})
    evaluation.pop("generated_at", None)
    return body


def _top_id(question: dict[str, Any]) -> str | None:
    if not question["top5"]:
        return None
    return question["top5"][0]["chunk_id"]


def _ratio(
    numerator: int, denominator: int, index_scope: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": None if denominator == 0 else numerator / denominator,
        "index_scope": index_scope,
    }


def _failures(questions: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    found = []
    key = str(k)
    for question in questions:
        if not question["in_positive_recall_denominator"]:
            continue
        levels = {row["claim_id"]: row for row in question["at_k"][key]["claims"]}
        grouped: dict[str, list[dict[str, Any]]] = {}
        for span in question.get("span_audit") or []:
            grouped.setdefault(span["claim_id"], []).append(span)
        for claim_id, spans in grouped.items():
            causes = sorted({span["cause"] for span in spans})
            level = levels.get(claim_id) or {}
            found.append(
                {
                    "question_id": question["question_id"],
                    "claim_id": claim_id,
                    "k": k,
                    "chunk_set": level.get("chunk_set"),
                    "single_chunk": level.get("single_chunk"),
                    "causes": causes,
                    "spans": spans,
                }
            )
    return found


def audit_span_failures(
    chunks: list[dict[str, Any]],
    question: dict[str, Any],
    claims: list[dict[str, Any]],
    ranked_chunks: list[dict[str, Any]],
    *,
    k: int = 5,
    elements_by_id: dict[str, dict[str, Any]] | None = None,
    indexed_document_id: str | None = None,
) -> list[dict[str, Any]]:
    """Classify a missed span. Ranked chunks are the non-zero BM25 hits, best first."""
    window = ranked_chunks[:k]
    ranked_at = {chunk["chunk_id"]: rank for rank, chunk in enumerate(ranked_chunks, start=1)}
    rows = []
    for claim in required_claims(claims):
        if single_and_set(window, claim)["chunk_set"] == "full":
            continue
        for span in claim.get("spans") or []:
            cause, detail, in_index, in_topk, first_rank = _classify_span(
                chunks,
                window,
                ranked_at,
                span,
                k=k,
                elements_by_id=elements_by_id,
                indexed_document_id=indexed_document_id,
            )
            if cause is None:
                continue
            rows.append(
                {
                    "question_id": question["question_id"],
                    "claim_id": claim["claim_id"],
                    "span_id": span["span_id"],
                    "in_index": in_index,
                    "in_topk": in_topk,
                    "cause": cause,
                    "detail": detail,
                    "first_full_rank": first_rank,
                }
            )
    return rows


def _classify_span(
    chunks: list[dict[str, Any]],
    window: list[dict[str, Any]],
    ranked_at: dict[str, int],
    span: dict[str, Any],
    *,
    k: int,
    elements_by_id: dict[str, dict[str, Any]] | None,
    indexed_document_id: str | None,
) -> tuple[str | None, str | None, str, str, int | None]:
    element = None if elements_by_id is None else elements_by_id.get(span["element_id"])
    in_index = _span_status(
        [source for chunk in chunks for source in chunk_sources(chunk)], span
    )
    in_topk = _span_status(
        [source for chunk in window for source in chunk_sources(chunk)], span
    )
    if in_topk == "full":
        return None, None, in_index, in_topk, None
    if elements_by_id is not None and element is None:
        return CAUSE_PARSING, "element_missing_from_elements", in_index, in_topk, None
    if (
        element is not None
        and indexed_document_id is not None
        and element.get("document_id") not in {None, indexed_document_id}
    ):
        return CAUSE_INDEX, "element_document_not_indexed", in_index, in_topk, None
    if in_index != "full":
        return CAUSE_CHUNKING, f"in_index={in_index}", in_index, in_topk, None
    full_chunks = [
        chunk
        for chunk in chunks
        if _span_status(chunk_sources(chunk), span) == "full"
    ]
    ranks = [
        ranked_at[chunk["chunk_id"]]
        for chunk in full_chunks
        if chunk["chunk_id"] in ranked_at
    ]
    if full_chunks and not ranks:
        return CAUSE_RETRIEVAL, "zero_score_omitted", in_index, in_topk, None
    if ranks and min(ranks) > k:
        return CAUSE_RETRIEVAL, "ranked_outside_topk", in_index, in_topk, min(ranks)
    if ranks and min(ranks) <= k:
        return CAUSE_EVALUATION, "ranked_within_k_but_topk_not_full", in_index, in_topk, min(ranks)
    return CAUSE_RETRIEVAL, "split_across_chunks", in_index, in_topk, None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def compare_retrievers(
    baseline_questions: list[dict[str, Any]],
    candidate_questions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Pair two runs that share gold, chunks, and questions. Not a generalization claim."""
    candidate = {item["question_id"]: item for item in candidate_questions}
    rows = []
    for left in baseline_questions:
        right = candidate[left["question_id"]]
        left_class = left["evidence_classification"]
        right_class = right["evidence_classification"]
        left_conflict = _conflict_rank(left)
        right_conflict = _conflict_rank(right)
        rows.append(
            {
                "question_id": left["question_id"],
                "baseline_first_positive_answer_evidence_rank": left_class[
                    "first_positive_answer_evidence_rank"
                ],
                "candidate_first_positive_answer_evidence_rank": right_class[
                    "first_positive_answer_evidence_rank"
                ],
                "baseline_first_complete_evidence_path_rank": left_class[
                    "first_complete_evidence_path_rank"
                ],
                "candidate_first_complete_evidence_path_rank": right_class[
                    "first_complete_evidence_path_rank"
                ],
                "baseline_question_success_at_5": bool(
                    left["at_k"]["5"]["answer_evidence_success"]
                ),
                "candidate_question_success_at_5": bool(
                    right["at_k"]["5"]["answer_evidence_success"]
                ),
                "baseline_conflicting_evidence_rank": left_conflict,
                "candidate_conflicting_evidence_rank": right_conflict,
                "baseline_relevant_conflict_at_5": bool(
                    left["at_k"]["5"]["relevant_conflicting_hit"]
                ),
                "candidate_relevant_conflict_at_5": bool(
                    right["at_k"]["5"]["relevant_conflicting_hit"]
                ),
            }
        )
    return {
        "population": "one filing regression set; not a generalization result",
        "positive_rank_definition": (
            "first chunk that covers any answer-evidence span, including a partial path"
        ),
        "question_success_definition": "question_answer_evidence_satisfied: one full evidence path",
        "questions": rows,
    }


def _conflict_rank(question: dict[str, Any]) -> int | None:
    ranks = [
        row["rank"]
        for row in question["evidence_classification"]["relevant_conflicting_ranks"]
        if row["rank"] is not None
    ]
    return None if not ranks else min(ranks)


def dataset_file_hash(directory: Path) -> str:
    parts = []
    for name in ("snapshot.json", "questions.json", "claims.json", "hard_negatives.json"):
        parts.append(sha256_text((directory / name).read_text(encoding="utf-8")))
    return sha256_text("\n".join(parts))

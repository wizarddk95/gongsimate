"""Gold source-span alignment and the fixed-snapshot BM25 baseline.

Gold labels are source spans (element id + paragraph or cell coordinates).
Chunk ids are never the answer key. Q01 and Q09 share one intent group.
Q10 is outside the positive Recall denominator.
"""

from __future__ import annotations

import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from disclosure_agent.bm25 import BM25_PARAMS, TOKENIZER_VERSION, Bm25Index, index_manifest
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

KS = (1, 3, 5)
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


def run_baseline(
    *,
    dataset_dir: Path,
    manifest_path: Path,
    elements_path: Path,
    output_dir: Path,
    ks: tuple[int, ...] = KS,
) -> dict[str, Any]:
    dataset = load_dataset(dataset_dir)
    manifest = load_json(manifest_path)
    member = verify_snapshot(dataset, manifest)
    elements = load_elements(elements_path)
    verify_gold_texts(dataset, elements)
    document_id = dataset["snapshot"]["index_scope"]["document_id"]
    chunks = chunk_elements(elements, manifest=manifest, document_id=document_id)
    index = Bm25Index(chunks)
    max_k = max(ks)
    questions = []
    for question in dataset["questions"]:
        hits = index.search(question["text"], max_k)
        questions.append(_score_question(dataset, question, chunks, hits, ks))
    calculation = _q07_calculation(dataset)
    metrics = _aggregate(dataset, questions, ks)
    scope = {
        "corp_code": dataset["snapshot"]["corp_code"],
        "receipt_no": dataset["snapshot"]["receipt_no"],
        "source_sha256": dataset["snapshot"]["source_sha256"],
        "document_id": document_id,
        "member_index": member.get("member_index"),
        "member_name": member.get("member_name"),
        "document_name_observed": member.get("document_name"),
        "excluded_members": dataset["snapshot"]["index_scope"]["excluded_members"],
        "not_company_wide_performance": True,
        "eligibility": "parse_manifest.overall_status=success",
    }
    payload = {
        "dataset": str(dataset_dir),
        "dataset_sha256": dataset_file_hash(dataset_dir),
        "scope": scope,
        "chunker_version": CHUNKER_VERSION,
        "chunker_config_hash": chunker_config_hash(),
        "tokenizer_version": TOKENIZER_VERSION,
        "bm25": dict(BM25_PARAMS),
        "chunks": chunk_manifest_summary(chunks),
        "index": index_manifest(index, scope=scope),
        "calculation_q07": calculation,
        "metrics": metrics,
        "questions": questions,
        "failures": _failures(questions, ks[-1] if ks else 5),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "metrics.json", {key: payload[key] for key in (
        "dataset", "dataset_sha256", "scope", "chunker_version", "chunker_config_hash",
        "tokenizer_version", "bm25", "chunks", "index", "calculation_q07", "metrics", "failures",
    )})
    _write_json(output_dir / "per_question.json", {"questions": questions})
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


def _score_question(
    dataset: dict[str, Any],
    question: dict[str, Any],
    chunks: list[dict[str, Any]],
    hits: list[dict[str, Any]],
    ks: tuple[int, ...],
) -> dict[str, Any]:
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    claims = claims_for_question(dataset, question)
    required = required_claims(claims)
    ranked_chunks = [by_id[hit["chunk_id"]] for hit in hits if hit["chunk_id"] in by_id]
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
        at_k[str(k)] = {
            "claims": claim_rows,
            "claim_set_full": sum(row["chunk_set"] == "full" for row in claim_rows),
            "claim_single_full": sum(row["single_chunk"] == "full" for row in claim_rows),
            "required_claims": len(claim_rows),
            "evidence_set_satisfied": evidence_full,
            "evidence_single_chunk_satisfied": single_full,
        }
    top = []
    for hit, chunk in zip(hits[:5], ranked_chunks[:5], strict=False):
        preview = chunk["text"].replace("\n", " ")
        if len(preview) > 240:
            preview = preview[:240]
        top.append(
            {
                **hit,
                "score": round(hit["score"], 6),
                "preview": preview,
                "claim_coverage": [
                    {
                        "claim_id": claim["claim_id"],
                        **single_and_set([chunk], claim),
                    }
                    for claim in required
                ],
            }
        )
    negatives = _negative_ranks(dataset, question, ranked_chunks[:5])
    span_audit = audit_span_failures(chunks, question, claims, ranked_chunks, k=5)
    return {
        "question_id": question["question_id"],
        "intent_group": question["intent_group"],
        "paraphrase_of": question.get("paraphrase_of"),
        "independent_intent": question.get("independent_intent", True),
        "in_positive_recall_denominator": bool(question.get("in_positive_recall_denominator")),
        "answerability": question["answerability"],
        "answer_complete_ceiling": question["answerability"] == "supported",
        "text": question["text"],
        "at_k": at_k,
        "top5": top,
        "hard_negative": negatives,
        "span_audit": span_audit,
        "amount_policy": question.get("amount_policy"),
        "notes": question.get("notes"),
    }


def _negative_ranks(
    dataset: dict[str, Any], question: dict[str, Any], chunks: list[dict[str, Any]]
) -> dict[str, Any]:
    blocked = [
        item
        for item in dataset["hard_negatives"]
        if question["question_id"] in item.get("not_gold_for", [])
    ]
    hits = []
    for item in blocked:
        for rank, chunk in enumerate(chunks, start=1):
            if chunk_contains_element(chunk, item["element_id"], body_only=True):
                hits.append(
                    {
                        "element_key": item["element_key"],
                        "rank": rank,
                        "chunk_id": chunk["chunk_id"],
                    }
                )
    return {"in_top5": bool(hits), "hits": hits}


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
    return result


def _aggregate(
    dataset: dict[str, Any], questions: list[dict[str, Any]], ks: tuple[int, ...]
) -> dict[str, Any]:
    positive = [item for item in questions if item["in_positive_recall_denominator"]]
    by_k = {}
    for k in ks:
        key = str(k)
        claim_hit = 0
        claim_single = 0
        claim_total = 0
        question_hit = 0
        question_single = 0
        answer_complete = 0
        for question in positive:
            block = question["at_k"][key]
            claim_hit += block["claim_set_full"]
            claim_single += block["claim_single_full"]
            claim_total += block["required_claims"]
            question_hit += int(block["evidence_set_satisfied"])
            question_single += int(block["evidence_single_chunk_satisfied"])
            if block["evidence_set_satisfied"] and question["answer_complete_ceiling"]:
                answer_complete += 1
        by_k[key] = {
            "claim_set_recall": _ratio(claim_hit, claim_total),
            "claim_single_chunk_recall": _ratio(claim_single, claim_total),
            "question_evidence_satisfied": _ratio(question_hit, len(positive)),
            "question_single_chunk_satisfied": _ratio(question_single, len(positive)),
            "question_answer_complete": _ratio(answer_complete, len(positive)),
        }
    paraphrase = _paraphrase_delta(questions)
    return {
        "population": "one filing, fixed 10-question regression set, not company-wide performance",
        "positive_question_ids": [item["question_id"] for item in positive],
        "positive_question_count": len(positive),
        "intent_group_G01_counted_once": True,
        "q09_excluded_as_paraphrase": True,
        "q10_excluded_from_positive_recall": True,
        "at_k": by_k,
        "paraphrase_q01_q09": paraphrase,
        "q10": next(item["question_id"] for item in questions if item["question_id"] == "Q10"),
    }


def _paraphrase_delta(questions: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {item["question_id"]: item for item in questions}
    left, right = by_id["Q01"], by_id["Q09"]
    shared = left["at_k"]["5"]["claims"]
    other = right["at_k"]["5"]["claims"]
    same_claims = [row["claim_id"] for row in shared] == [row["claim_id"] for row in other]
    return {
        "shared_claim_ids": [row["claim_id"] for row in shared],
        "q09_uses_same_claim_ids": same_claims,
        "q01_top1": _top_id(left),
        "q09_top1": _top_id(right),
        "top1_changed": _top_id(left) != _top_id(right),
        "counted_as_independent_intents": 1,
    }


def _top_id(question: dict[str, Any]) -> str | None:
    if not question["top5"]:
        return None
    return question["top5"][0]["chunk_id"]


def _ratio(numerator: int, denominator: int) -> dict[str, int | float | None]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": None if denominator == 0 else numerator / denominator,
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
) -> list[dict[str, Any]]:
    """Separate a missed span into chunking vs retrieval using the full index."""
    window = ranked_chunks[:k]
    rows = []
    for claim in required_claims(claims):
        if single_and_set(window, claim)["chunk_set"] == "full":
            continue
        for span in claim.get("spans") or []:
            in_index = _span_status(
                [source for chunk in chunks for source in chunk_sources(chunk)], span
            )
            in_topk = _span_status(
                [source for chunk in window for source in chunk_sources(chunk)], span
            )
            if in_topk == "full":
                continue
            if in_index != "full":
                cause = "chunking"
            else:
                cause = "retrieval"
            rows.append(
                {
                    "question_id": question["question_id"],
                    "claim_id": claim["claim_id"],
                    "span_id": span["span_id"],
                    "in_index": in_index,
                    "in_topk": in_topk,
                    "cause": cause,
                }
            )
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def dataset_file_hash(directory: Path) -> str:
    parts = []
    for name in ("snapshot.json", "questions.json", "claims.json", "hard_negatives.json"):
        parts.append(sha256_text((directory / name).read_text(encoding="utf-8")))
    return sha256_text("\n".join(parts))

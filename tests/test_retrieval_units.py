"""Retrieval-unit projection, source lineage, and gold mapping."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from disclosure_agent.chunking import sha256_text
from disclosure_agent.context_budget import (
    EMBEDDING_MODEL_MAX_TOKENS,
    RETRIEVAL_UNIT_HARD_MAX_TOKENS,
    RETRIEVAL_UNIT_TARGET_TOKENS,
)
from disclosure_agent.evaluation import (
    _first_set_full_rank,
    coverage_level,
    load_dataset,
    required_claims,
)
from disclosure_agent.retrieval_unit import (
    D2A_VERSION,
    D2B_VERSION,
    build_retrieval_units,
    cell_slices_cover_text,
    retrieval_units_sha256,
    slot_diagnostic,
)

DATASET_V03 = Path(__file__).resolve().parents[1] / "evaluation" / "datasets" / "isu_petasys_v0_3"
D0_METRICS = (
    Path(__file__).resolve().parents[1]
    / "evaluation"
    / "runs"
    / "isu_petasys_v0_3_dense"
    / "metrics.json"
)
CHUNKS_PATH = (
    Path(__file__).resolve().parents[1]
    / "evaluation"
    / "runs"
    / "isu_petasys_v0_3_dense"
    / "chunks.jsonl"
)
ELEMENTS_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "parsed"
    / "dart"
    / "20260323000847"
    / "ff2ee7ca28c18ac6487ea4349cfde9d8dca4d03771c2a8a1bd37c9d81cfd7bb6"
    / "p2"
    / "2d75d2abcee58231"
    / "elements.jsonl"
)
E186 = "78be6e94020c610d2411a83f2187ca58f098c8d9f071da7c4d78cb9db0ae5b62"


def _chars(text: str) -> int:
    return len(text)


def _cell(row: int, col: int, text: str, *, rowspan: int = 1, colspan: int = 1) -> dict:
    return {
        "start_row": row,
        "start_col": col,
        "rowspan": rowspan,
        "colspan": colspan,
        "text": text,
    }


def _ref(cell: dict) -> dict:
    return {
        "start_row": cell["start_row"],
        "start_col": cell["start_col"],
        "rowspan": cell["rowspan"],
        "colspan": cell["colspan"],
        "text_sha256": sha256_text(cell["text"]),
    }


def _narrative(body: str, *, ordinal: int = 9) -> tuple[dict, dict]:
    cell = _cell(0, 0, body)
    element_id = "policy"
    header = "[section] 정책"
    chunk = {
        "chunk_id": f"narrative-{ordinal}",
        "ordinal": ordinal,
        "text": f"{header}\n{body}",
        "text_hash": sha256_text(f"{header}\n{body}"),
        "section_path": ["정책"],
        "chunk_sources": [
            {
                "element_id": element_id,
                "kind": "table",
                "role": "body",
                "cells": [_ref(cell)],
            }
        ],
    }
    elements = {element_id: {"element_id": element_id, "table": {"cells": [cell]}}}
    return chunk, elements


def _profitability_chunk() -> tuple[dict, dict]:
    title = "2) 수익성지표"
    header = [
        _cell(0, 0, "구 분", rowspan=2),
        _cell(0, 1, "산 식", rowspan=2),
        _cell(0, 2, "비 율", colspan=2),
        _cell(1, 2, "당 기"),
        _cell(1, 3, "전 기"),
    ]
    body = [
        _cell(2, 0, "매출액영업이익률"),
        _cell(2, 1, "영업이익──────── ×100　　　매출액"),
        _cell(2, 2, "18.82%"),
        _cell(2, 3, "12.17%"),
        _cell(3, 0, "빈행"),
        _cell(3, 1, ""),
        _cell(3, 2, ""),
        _cell(3, 3, ""),
    ]
    elements = {
        "title": {"element_id": "title", "text": title},
        "table": {"element_id": "table", "table": {"cells": header + body}},
    }
    chunk = {
        "chunk_id": "profit",
        "ordinal": 7,
        "text": "canonical pipe text that must stay untouched",
        "text_hash": sha256_text("canonical pipe text that must stay untouched"),
        "section_path": ["IV. 이사의 경영진단 및 분석의견"],
        "chunk_sources": [
            {
                "element_id": "title",
                "kind": "paragraph",
                "role": "context_title",
                "char_start": 0,
                "char_end": len(title),
                "cells": [],
            },
            {
                "element_id": "table",
                "kind": "table",
                "role": "header",
                "cells": [_ref(cell) for cell in header],
            },
            {
                "element_id": "table",
                "kind": "table",
                "role": "body",
                "cells": [_ref(cell) for cell in body],
            },
        ],
    }
    return chunk, elements


def test_d2a_is_deterministic_and_does_not_edit_the_canonical_chunk() -> None:
    body = "(1) 연결 정책입니다. (2) 별도 정책입니다."
    chunk, elements = _narrative(body)
    before = (chunk["text"], [dict(source) for source in chunk["chunk_sources"]])
    first = build_retrieval_units(
        [chunk],
        elements,
        mode="d2a",
        token_counter=_chars,
        hard_max_tokens=10_000,
        model_max_tokens=20_000,
    )
    second = build_retrieval_units(
        [chunk],
        elements,
        mode="d2a",
        token_counter=_chars,
        hard_max_tokens=10_000,
        model_max_tokens=20_000,
    )
    assert retrieval_units_sha256(first) == retrieval_units_sha256(second)
    assert first[0]["representation_version"] == D2A_VERSION
    assert first[0]["canonical_chunk_id"] == chunk["chunk_id"]
    assert first[0]["text"] == chunk["text"]
    assert first[0]["source_refs"] == chunk["chunk_sources"]
    assert (chunk["text"], chunk["chunk_sources"]) == before
    assert first[0]["embedding_truncated"] is False


def test_d2a_splits_numbered_sections_without_a_mid_token_chop() -> None:
    first_section = "(1) " + ("가" * 70)
    second_section = "(2) " + ("나" * 12)
    body = f"머리말입니다. {first_section}{second_section}"
    chunk, elements = _narrative(body)
    units = build_retrieval_units(
        [chunk],
        elements,
        mode="d2a",
        token_counter=_chars,
        target_tokens=40,
        hard_max_tokens=100,
        model_max_tokens=400,
    )
    assert len(units) >= 2
    assert all(unit["token_count"] <= 120 for unit in units)
    assert all(unit["embedding_truncated"] is False for unit in units)
    bodies = [unit["text"].split("\n", 1)[1] for unit in units]
    assert any(first_section in body for body in bodies)
    assert cell_slices_cover_text(units, body)
    assert chunk["text"].endswith(body)


def test_d2a_keeps_an_atomic_section_between_target_and_hard_max() -> None:
    section = "(1) " + ("가" * 60)
    body = section + " " + "(2) " + ("나" * 80)
    chunk, elements = _narrative(body)
    units = build_retrieval_units(
        [chunk],
        elements,
        mode="d2a",
        token_counter=_chars,
        target_tokens=40,
        hard_max_tokens=110,
        model_max_tokens=400,
    )
    matched = [unit for unit in units if section in unit["text"]]
    assert len(matched) == 1
    assert "(2)" not in matched[0]["text"]
    assert matched[0]["token_count"] > 30
    assert cell_slices_cover_text(units, body)


def test_d2a_uses_sentences_before_the_token_window() -> None:
    sentence = "가나다라마바사아자차입니다. "
    body = (sentence * 7) + "가나다라마바사아자차입니다."
    chunk, elements = _narrative(body)
    units = build_retrieval_units(
        [chunk],
        elements,
        mode="d2a",
        token_counter=_chars,
        target_tokens=40,
        hard_max_tokens=80,
        model_max_tokens=200,
    )
    assert len(units) > 1
    assert all(unit["token_count"] <= 80 for unit in units)
    assert all(unit["split_reason"] != "token_window" for unit in units)
    assert cell_slices_cover_text(units, body)


def test_d2a_token_window_is_only_the_fallback_and_does_not_truncate() -> None:
    body = "가" * 180
    chunk, elements = _narrative(body)
    units = build_retrieval_units(
        [chunk],
        elements,
        mode="d2a",
        token_counter=_chars,
        target_tokens=30,
        hard_max_tokens=50,
        model_max_tokens=80,
    )
    assert len(units) > 1
    assert all(unit["token_count"] <= 50 for unit in units)
    assert all(unit["embedding_truncated"] is False for unit in units)
    assert any(unit["split_reason"] == "token_window" for unit in units)
    assert cell_slices_cover_text(units, body)
    assert RETRIEVAL_UNIT_TARGET_TOKENS < RETRIEVAL_UNIT_HARD_MAX_TOKENS
    assert RETRIEVAL_UNIT_HARD_MAX_TOKENS < EMBEDDING_MODEL_MAX_TOKENS


def test_partial_cell_spans_cover_gold_only_as_a_union() -> None:
    claim = {
        "claim_id": "cell",
        "spans": [
            {
                "span_id": "cell-span",
                "element_id": "policy",
                "kind": "table_cells",
                "cells": [{"start_row": 0, "start_col": 0, "text": "가" * 40}],
            }
        ],
    }
    left = {
        "chunk_sources": [
            {
                "element_id": "policy",
                "kind": "table",
                "role": "body",
                "cells": [
                    {
                        "start_row": 0,
                        "start_col": 0,
                        "text_char_start": 0,
                        "text_char_end": 20,
                        "text_char_length": 40,
                    }
                ],
            }
        ]
    }
    right = {
        "chunk_sources": [
            {
                "element_id": "policy",
                "kind": "table",
                "role": "body",
                "cells": [
                    {
                        "start_row": 0,
                        "start_col": 0,
                        "text_char_start": 20,
                        "text_char_end": 40,
                        "text_char_length": 40,
                    }
                ],
            }
        ]
    }
    assert coverage_level(left["chunk_sources"], claim) == "partial"
    assert coverage_level(right["chunk_sources"], claim) == "partial"
    pooled = left["chunk_sources"] + right["chunk_sources"]
    assert coverage_level(pooled, claim) == "full"
    ranked = [
        {"claim_id": "left", "chunk_sources": left["chunk_sources"]},
        {"claim_id": "right", "chunk_sources": right["chunk_sources"]},
    ]
    assert _first_set_full_rank(ranked, [claim]) == 2


def test_d2b_row_text_is_deterministic_and_does_not_invent_fields() -> None:
    chunk, elements = _profitability_chunk()
    before = chunk["text"]
    first = build_retrieval_units([chunk], elements, mode="d2b", token_counter=_chars)
    second = build_retrieval_units([chunk], elements, mode="d2b", token_counter=_chars)
    assert retrieval_units_sha256(first) == retrieval_units_sha256(second)
    assert all(unit["representation_version"] == D2B_VERSION for unit in first)
    assert chunk["text"] == before
    row = next(unit for unit in first if "18.82%" in unit["text"])
    assert row["text"] == (
        "2) 수익성지표 | 매출액영업이익률 | 당 기=18.82% | 전 기=12.17%"
    )
    assert "×100" not in row["text"]
    assert "연결" not in row["text"]
    assert "단위=" not in row["text"]
    assert "기간=" not in row["text"]
    assert row["canonical_chunk_id"] == chunk["chunk_id"]
    coords = {
        (int(cell["start_row"]), int(cell["start_col"]))
        for source in row["source_refs"]
        for cell in source.get("cells") or []
    }
    assert (2, 1) in coords
    assert (2, 2) in coords
    assert (0, 2) in coords
    assert (1, 2) in coords
    residual = next(unit for unit in first if unit["split_reason"] == "table_row_residual")
    assert "빈행" in residual["text"]
    assert "18.82%" not in residual["text"]


def test_d2b_keeps_a_single_cell_narrative_as_one_canonical_unit() -> None:
    chunk, elements = _narrative("(1) 긴 정책 " + ("가" * 50))
    units = build_retrieval_units([chunk], elements, mode="d2b", token_counter=_chars)
    assert len(units) == 1
    assert units[0]["text"] == chunk["text"]
    assert units[0]["representation_type"] == "canonical_text"
    assert units[0]["eligibility_reason"] == "single_cell_narrative"
    assert units[0]["source_refs"] == chunk["chunk_sources"]


def test_d2b_copies_unit_and_period_only_from_source_roles() -> None:
    chunk, elements = _profitability_chunk()
    chunk["chunk_sources"].append(
        {
            "element_id": "unit",
            "kind": "paragraph",
            "role": "context_unit",
            "char_start": 0,
            "char_end": 8,
            "cells": [],
        }
    )
    chunk["chunk_sources"].append(
        {
            "element_id": "period",
            "kind": "paragraph",
            "role": "context_period",
            "char_start": 0,
            "char_end": 9,
            "cells": [],
        }
    )
    elements["unit"] = {"element_id": "unit", "text": "(단위 : 원)"}
    elements["period"] = {"element_id": "period", "text": "제54기"}
    units = build_retrieval_units([chunk], elements, mode="d2b", token_counter=_chars)
    row = next(unit for unit in units if "18.82%" in unit["text"])
    assert row["text"].endswith("기간=제54기 | 단위=(단위 : 원)")
    assert "FY2025" not in row["text"]
    assert "연결기준" not in row["text"]


def test_slot_diagnostic_counts_duplicate_canonical_sources() -> None:
    hits = [
        {"chunk_id": "ru-1", "canonical_chunk_id": "chunk-a"},
        {"chunk_id": "ru-2", "canonical_chunk_id": "chunk-a"},
        {"chunk_id": "ru-3", "canonical_chunk_id": "chunk-b"},
    ]
    report = slot_diagnostic(hits, ks=(1, 3))
    assert report["1"]["unique_canonical_chunks"] == 1
    assert report["1"]["duplicate_same_source_ru_count"] == 0
    assert report["3"]["unique_canonical_chunks"] == 2
    assert report["3"]["duplicate_same_source_ru_count"] == 1
    assert report["3"]["not_a_headline_metric"] is True


def test_e186_is_outside_required_claims_and_d0_headline_stays_locked() -> None:
    dataset = load_dataset(DATASET_V03)
    positive = [
        question
        for question in dataset["questions"]
        if question.get("in_positive_recall_denominator")
    ]
    assert len(positive) == 8
    claims = []
    for question in positive:
        owned = [
            claim
            for claim in dataset["claims"].values()
            if claim["question_id"] == question["question_id"]
        ]
        claims.extend(required_claims(owned))
    assert len(claims) == 27
    for claim in dataset["claims"].values():
        for span in claim["spans"]:
            assert span["element_id"] != E186
    if not D0_METRICS.exists():
        pytest.skip("D0 metrics artifact is local")
    metrics = json.loads(D0_METRICS.read_text(encoding="utf-8"))
    recall_5 = metrics["metrics"]["at_k"]["5"]["claim_set_recall"]
    recall_20 = metrics["metrics"]["at_k"]["20"]["claim_set_recall"]
    question_5 = metrics["metrics"]["at_k"]["5"]["question_answer_evidence_satisfied"]
    question_20 = metrics["metrics"]["at_k"]["20"]["question_answer_evidence_satisfied"]
    assert (recall_5["numerator"], recall_5["denominator"]) == (15, 27)
    assert (recall_20["numerator"], recall_20["denominator"]) == (25, 27)
    assert (question_5["numerator"], question_5["denominator"]) == (2, 8)
    assert (question_20["numerator"], question_20["denominator"]) == (6, 8)


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


@pytest.mark.skipif(
    not CHUNKS_PATH.exists() or not ELEMENTS_PATH.exists(),
    reason="pinned artifacts",
)
def test_pinned_oversized_cells_split_on_numbered_headings_and_rebuild() -> None:
    chunks = _load_jsonl(CHUNKS_PATH)
    elements = {row["element_id"]: row for row in _load_jsonl(ELEMENTS_PATH)}
    targets = [chunk for chunk in chunks if chunk["ordinal"] in {266, 1381}]
    assert len(targets) == 2
    units = build_retrieval_units(
        targets,
        elements,
        mode="d2a",
        token_counter=_chars,
        target_tokens=2500,
        hard_max_tokens=4000,
        model_max_tokens=20000,
    )
    for chunk in targets:
        group = [unit for unit in units if unit["canonical_ordinal"] == chunk["ordinal"]]
        assert len(group) > 1
        body = chunk["text"].split("\n", 1)[1]
        assert cell_slices_cover_text(group, body)
        assert all(unit["token_count"] <= 4000 for unit in group)
        markers = [f"({number})" for number in range(1, 8)]
        found = {marker: 0 for marker in markers}
        for unit in group:
            for marker in markers:
                found[marker] += unit["text"].count(marker)
        assert all(count >= 1 for count in found.values())

"""Retrieval representations, context budget, and Q07 precision paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from disclosure_agent.chunking import chunk_elements, load_elements, load_json, sha256_text
from disclosure_agent.context_budget import (
    CONTEXT_UNIT_BUDGET_TOKENS,
    EMBEDDING_MODEL_MAX_TOKENS,
    RETRIEVAL_UNIT_TARGET_TOKENS,
    assemble_context_unit,
)
from disclosure_agent.dense import MAX_SEQ_LENGTH
from disclosure_agent.evaluation import (
    load_dataset,
    precision_combination_valid,
    q07_precision_contract,
)
from disclosure_agent.representation import (
    D1_TARGET_ELEMENT_IDS,
    build_retrieval_representations,
    embedding_input_record,
    representation_sha256,
    serialize_table_chunk,
)

DATASET_V03 = Path(__file__).resolve().parents[1] / "evaluation" / "datasets" / "isu_petasys_v0_3"
PINNED = (
    Path("data/parsed/dart/20260323000847")
    / "ff2ee7ca28c18ac6487ea4349cfde9d8dca4d03771c2a8a1bd37c9d81cfd7bb6"
    / "p2"
    / "2d75d2abcee58231"
)
E2362 = "9b792cedd0b5cac5fb96d20943d300ae51dcf509b090e917367dad1152e341de"
E228 = "f55b61e36dd52bfcfeda562f3dcc163dd8bc4051f2176d3b248b13034f60f244"
E186 = "78be6e94020c610d2411a83f2187ca58f098c8d9f071da7c4d78cb9db0ae5b62"


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


def _sample_chunk() -> tuple[dict, dict]:
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
        _cell(2, 1, "영업이익 / 매출액 ×100"),
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
        "chunk_id": "chunk",
        "ordinal": 7,
        "text": "canonical pipe text",
        "text_hash": sha256_text("canonical pipe text"),
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


def test_structured_serialization_is_deterministic_and_keeps_lineage() -> None:
    chunk, elements = _sample_chunk()
    before = chunk["text"]
    sources = list(chunk["chunk_sources"])
    first = serialize_table_chunk(chunk, elements)
    second = serialize_table_chunk(chunk, elements)
    assert first == second
    assert "연결기준" not in first
    assert "[표] 2) 수익성지표" in first
    assert "[구 분] 매출액영업이익률" in first
    assert "[산 식] 영업이익 / 매출액 ×100" in first
    assert "[비 율 / 당 기] 18.82%" in first
    assert "[비 율 / 전 기] 12.17%" in first
    assert "[항목]" not in first
    assert chunk["text"] == before
    assert chunk["chunk_sources"] == sources
    records = build_retrieval_representations(
        [chunk],
        elements,
        mode="structured-table-v1",
        target_element_ids=frozenset({"table"}),
    )
    again = build_retrieval_representations(
        [chunk],
        elements,
        mode="structured-table-v1",
        target_element_ids=frozenset({"table"}),
    )
    assert representation_sha256(records) == representation_sha256(again)
    assert records[0]["source_refs"] == sources
    assert records[0]["canonical_text_hash"] == sha256_text(before)
    assert chunk["text"] == before


def test_blank_column_header_uses_item_field_without_inventing_scope() -> None:
    cells = [
        _cell(0, 0, ""),
        _cell(0, 1, "제 54 기"),
        _cell(1, 0, "영업이익(손실)"),
        _cell(1, 1, "204,724,531,402"),
    ]
    elements = {"table": {"element_id": "table", "table": {"cells": cells}}}
    chunk = {
        "ordinal": 1,
        "section_path": ["III"],
        "chunk_sources": [
            {
                "element_id": "table",
                "kind": "table",
                "role": "header",
                "cells": [_ref(cells[0]), _ref(cells[1])],
            },
            {
                "element_id": "table",
                "kind": "table",
                "role": "body",
                "cells": [_ref(cells[2]), _ref(cells[3])],
            },
        ],
    }
    text = serialize_table_chunk(chunk, elements)
    assert "[항목] 영업이익(손실)" in text
    assert "[제 54 기] 204,724,531,402" in text
    assert "연결" not in text


def test_oversized_representation_is_not_stored_as_a_prefix() -> None:
    text = "가" * 20000
    recorded = embedding_input_record(text, token_count=12000, max_tokens=MAX_SEQ_LENGTH)
    assert recorded["stored_text"] == text
    assert recorded["embedding_truncated"] is True
    assert recorded["embedding_model_max_tokens"] == EMBEDDING_MODEL_MAX_TOKENS
    assert recorded["stored_text_is_full_representation"] is True
    assert len(recorded["stored_text"]) > MAX_SEQ_LENGTH


def test_context_unit_does_not_paste_an_over_budget_retrieval_text() -> None:
    chunk = {
        "chunk_id": "big",
        "ordinal": 266,
        "text": "정책본문" * 5000,
        "section_path": ["3. 중요한 회계정책 (연결)"],
        "chunk_sources": [
            {
                "element_id": "policy",
                "kind": "table",
                "role": "body",
            }
        ],
    }
    unit = assemble_context_unit(chunk, token_count=12711)
    assert unit.over_budget is True
    assert unit.omitted_full_retrieval_text is True
    assert unit.text != chunk["text"]
    assert "정책본문" not in unit.text
    assert "element_id=policy" in unit.text
    assert unit.source_refs[0]["element_id"] == "policy"
    assert unit.budget_tokens == CONTEXT_UNIT_BUDGET_TOKENS
    assert RETRIEVAL_UNIT_TARGET_TOKENS < EMBEDDING_MODEL_MAX_TOKENS
    small = assemble_context_unit(
        {**chunk, "text": "짧은 근거"},
        token_count=40,
    )
    assert small.over_budget is False
    assert small.text == "짧은 근거"


def test_q07_precision_paths_are_not_interchangeable() -> None:
    question = next(
        item
        for item in load_dataset(DATASET_V03)["questions"]
        if item["question_id"] == "Q07"
    )
    contract = q07_precision_contract(question)
    reported = contract["reported_metric_precision_policy"]
    raw = contract["raw_operand_precision_policy"]
    assert reported["evidence_path"] == "direct_reported_metric"
    assert reported["delta_pp"] == "6.65"
    assert raw["evidence_path"] == "derived_operands"
    assert raw["delta_pp"] == "6.64"
    assert precision_combination_valid(question, "direct_reported_metric", "6.65")
    assert precision_combination_valid(question, "derived_operands", "6.64")
    assert not precision_combination_valid(question, "direct_reported_metric", "6.64")
    assert not precision_combination_valid(question, "derived_operands", "6.65")
    assert contract["generation_implemented"] is False
    assert "+6.65%p" in contract["both_paths_statement"]
    assert "+6.64%p" in contract["both_paths_statement"]
    paths = [path["path_id"] for path in question["evidence_paths"]]
    assert paths == ["direct_reported_metric", "derived_operands"]


def test_e186_stays_relevant_conflicting_and_not_answer_gold() -> None:
    dataset = load_dataset(DATASET_V03)
    label = next(item for item in dataset["hard_negatives"] if item["element_id"] == E186)
    assert label["evidence_class"] == "relevant_conflicting"
    assert label["answer_valid_evidence"] is False
    assert "Q07" in label["classified_for"]
    for claim in dataset["claims"].values():
        for span in claim["spans"]:
            assert span["element_id"] != E186


@pytest.mark.skipif(
    not (PINNED / "elements.jsonl").exists(),
    reason="pinned parse snapshot is local",
)
def test_pinned_d1_tables_keep_source_text_and_do_not_invent_scope() -> None:
    manifest = load_json(PINNED / "parse_manifest.json")
    elements = load_elements(PINNED / "elements.jsonl")
    document_id = "m0:20260323000847.xml"
    chunks = chunk_elements(elements, manifest=manifest, document_id=document_id)
    originals = [(chunk["text"], list(chunk["chunk_sources"])) for chunk in chunks]
    elements_by_id = {element["element_id"]: element for element in elements}
    records = build_retrieval_representations(
        chunks,
        elements_by_id,
        mode="structured-table-v1",
        target_element_ids=D1_TARGET_ELEMENT_IDS,
    )
    again = build_retrieval_representations(
        chunks,
        elements_by_id,
        mode="structured-table-v1",
        target_element_ids=D1_TARGET_ELEMENT_IDS,
    )
    assert representation_sha256(records) == representation_sha256(again)
    for chunk, (text, sources) in zip(chunks, originals, strict=True):
        assert chunk["text"] == text
        assert chunk["chunk_sources"] == sources
    structured = [record for record in records if record["type"] == "structured_table"]
    assert len(structured) == 8
    profitability = next(
        record
        for record in structured
        if any(source["element_id"] == E2362 for source in record["source_refs"])
    )
    assert "18.82%" in profitability["text"]
    assert "12.17%" in profitability["text"]
    assert "2) 수익성지표" in profitability["text"]
    assert "연결기준" not in profitability["text"]
    assert "연결 포괄손익계산서" not in profitability["text"]
    statement = [
        record
        for record in structured
        if any(
            source["element_id"] == E228 and source["role"] == "body"
            for source in record["source_refs"]
        )
    ]
    assert len(statement) == 7
    joined = "\n".join(record["text"] for record in statement)
    assert "연결 포괄손익계산서" in joined
    assert "204,724,531,402" in joined
    assert "101,878,836,049" in joined
    assert "(단위 : 원)" in joined
    for record in structured:
        assert record["source_refs"]
        assert record["text_hash"] == sha256_text(record["text"])

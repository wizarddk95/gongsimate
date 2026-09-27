"""Chunker, source-span alignment, and the pinned BM25 baseline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from disclosure_agent.bm25 import Bm25Index, tokenize
from disclosure_agent.chunking import canonical_dumps, chunk_elements, sha256_text
from disclosure_agent.errors import UsageError
from disclosure_agent.evaluation import (
    audit_span_failures,
    claims_for_question,
    coverage_level,
    hard_negative_ids,
    load_dataset,
    operating_margin_change,
    positive_questions,
    q05_unique_amounts,
    run_baseline,
    single_and_set,
)

DATASET = Path(__file__).resolve().parents[1] / "evaluation" / "datasets" / "isu_petasys_v0_1"
PINNED = (
    Path("data/parsed/dart/20260323000847")
    / "ff2ee7ca28c18ac6487ea4349cfde9d8dca4d03771c2a8a1bd37c9d81cfd7bb6"
    / "p2"
    / "2d75d2abcee58231"
)
DOC = "m0:a.xml"
SHA = "a" * 64


def _manifest(status: str = "success") -> dict:
    return {
        "overall_status": status,
        "receipt_no": "20260323000847",
        "source_sha256": SHA,
        "parser_version": "2",
        "preprocess_version": "2",
        "config_hash": "b" * 64,
        "element_count": 2,
        "members": [
            {
                "member_index": 0,
                "member_name": "a.xml",
                "document_id": DOC,
                "document_name": "사업보고서",
                "byte_sha256": "c" * 64,
            }
        ],
    }


def _locator() -> dict:
    return {"source_sha256": SHA, "member_index": 0, "member_name": "a.xml"}


def _cell(
    row: int,
    col: int,
    text: str,
    *,
    rowspan: int = 1,
    colspan: int = 1,
    role: str = "unknown",
    tag: str = "TD",
) -> dict:
    return {
        "start_row": row,
        "start_col": col,
        "rowspan": rowspan,
        "colspan": colspan,
        "text": text,
        "header_role": role,
        "tag": tag,
        "source_locator": {**_locator(), "cell": [row, col]},
    }


def _paragraph(order: int, text: str, section: list[str]) -> dict:
    return {
        "element_type": "paragraph",
        "order_index": order,
        "document_id": DOC,
        "element_id": f"p{order}",
        "section_path": section,
        "text": text,
        "source_locator": _locator(),
        "warnings": [],
    }


def _table(order: int, cells: list[dict], section: list[str], *, unit: str | None = None) -> dict:
    rows = max(cell["start_row"] + cell["rowspan"] for cell in cells)
    cols = max(cell["start_col"] + cell["colspan"] for cell in cells)
    element = {
        "element_type": "table",
        "order_index": order,
        "document_id": DOC,
        "element_id": f"t{order}",
        "section_path": section,
        "text": "table",
        "source_locator": _locator(),
        "warnings": [],
        "table": {"cells": cells, "row_count": rows, "col_count": cols},
    }
    if unit:
        element["unit_mentions"] = [
            {
                "raw_text": unit,
                "source_locator": cells[0]["source_locator"],
                "applies_to": None,
                "scope_status": "unknown",
            }
        ]
        element["unit_scope_status"] = "unknown"
    return element


def _split_fixture() -> list[dict]:
    section = ["II", "재무"]
    caption_cells = [
        _cell(0, 0, "제 54 기 2025.01.01 부터 2025.12.31 까지", tag="TE"),
        _cell(1, 0, "(단위 : 원)", tag="TE"),
    ]
    body = [_cell(0, 0, "과목", role="unknown"), _cell(0, 1, "제 54 기", role="unknown")]
    for row in range(1, 11):
        body.append(_cell(row, 0, f"항목{row}"))
        body.append(_cell(row, 1, f"{row},000,000"))
    return [
        _table(0, caption_cells, section, unit="(단위 : 원)"),
        _table(1, body, section),
    ]


def test_partial_manifest_is_not_search_eligible() -> None:
    with pytest.raises(UsageError, match="overall_status"):
        chunk_elements([], manifest=_manifest("partial"), document_id=DOC)


def test_chunk_rerun_is_deterministic() -> None:
    elements = _split_fixture()
    manifest = _manifest()
    first = chunk_elements(elements, manifest=manifest, document_id=DOC)
    second = chunk_elements(elements, manifest=manifest, document_id=DOC)
    assert canonical_dumps(first) == canonical_dumps(second)
    assert len({chunk["chunk_id"] for chunk in first}) == len(first)


def test_split_table_keeps_period_unit_and_limits_body_rows() -> None:
    chunks = chunk_elements(_split_fixture(), manifest=_manifest(), document_id=DOC)
    body_chunks = []
    for chunk in chunks:
        roles = {}
        for source in chunk["chunk_sources"]:
            if source["element_id"] == "t1" and source["role"] == "body":
                rows = {cell["start_row"] for cell in source["cells"]}
                body_chunks.append((chunk, rows))
            roles.setdefault(source["element_id"], set()).add(source["role"])
    assert len(body_chunks) >= 2
    for chunk, rows in body_chunks:
        assert rows <= {1, 2, 3, 4} or rows <= {5, 6, 7, 8} or rows <= {9, 10}
        assert max(rows) - min(rows) < 10
        coords = {
            (source["element_id"], cell["start_row"], cell["start_col"])
            for source in chunk["chunk_sources"]
            for cell in source.get("cells") or []
        }
        assert ("t0", 0, 0) in coords
        assert ("t0", 1, 0) in coords
        assert chunk["unit_applied"] is False
        assert any(mention["raw_text"] == "(단위 : 원)" for mention in chunk["unit_mentions"])
        assert chunk["unit_scope_status"] == "unknown"


def test_unknown_header_role_still_keeps_header_text() -> None:
    chunks = chunk_elements(_split_fixture(), manifest=_manifest(), document_id=DOC)
    header_cells = [
        cell
        for chunk in chunks
        for source in chunk["chunk_sources"]
        if source["element_id"] == "t1" and source["role"] in {"header", "repeated_header"}
        for cell in source["cells"]
        if cell["start_row"] == 0 and cell["start_col"] == 0
    ]
    assert header_cells
    assert header_cells[0]["header_role"] == "unknown"
    assert header_cells[0]["text_sha256"] == sha256_text("과목")


def test_merged_cell_span_is_preserved() -> None:
    section = ["II"]
    cells = [
        _cell(0, 0, "구분"),
        _cell(0, 1, "금액"),
        _cell(1, 0, "신규공장", rowspan=2),
        _cell(1, 1, "141,300"),
        _cell(2, 1, "211,000"),
    ]
    chunks = chunk_elements([_table(0, cells, section)], manifest=_manifest(), document_id=DOC)
    merged = [
        cell
        for chunk in chunks
        for source in chunk["chunk_sources"]
        for cell in source.get("cells") or []
        if cell["start_row"] == 1 and cell["start_col"] == 0
    ]
    assert merged[0]["rowspan"] == 2
    assert merged[0]["text_sha256"] == sha256_text("신규공장")


def test_long_paragraph_split_is_covered_by_the_set_not_one_piece() -> None:
    text = ("가나다라. " * 80).strip()
    element = _paragraph(0, text, ["I"])
    config = {
        "chunker_version": "1",
        "paragraph_target_chars": 50,
        "paragraph_max_chars": 80,
        "table_split_min_body_rows": 8,
        "table_body_rows_per_chunk": 4,
        "preface_paragraph_max_chars": 200,
        "caption_table_max_rows": 6,
        "caption_table_max_cols": 2,
        "header_detection": "th-or-band-above-first-amount",
        "unknown_header_role_means_missing_text": False,
        "repeat_unit_and_header_on_split": True,
        "link_preceding_scope_or_caption": True,
    }
    chunks = chunk_elements([element], manifest=_manifest(), document_id=DOC, config=config)
    assert len(chunks) > 1
    claim = {
        "spans": [
            {
                "span_id": "s",
                "element_id": "p0",
                "kind": "paragraph",
                "char_start": 0,
                "char_end": len(text),
            }
        ]
    }
    assert single_and_set(chunks, claim)["single_chunk"] != "full"
    assert single_and_set(chunks, claim)["chunk_set"] == "full"


def test_dataset_recall_denominator_and_shared_intent() -> None:
    dataset = load_dataset(DATASET)
    positive = [item["question_id"] for item in positive_questions(dataset)]
    assert positive == ["Q01", "Q02", "Q03", "Q04", "Q05", "Q06", "Q07", "Q08"]
    assert "Q09" not in positive
    assert "Q10" not in positive
    q01 = next(item for item in dataset["questions"] if item["question_id"] == "Q01")
    q09 = next(item for item in dataset["questions"] if item["question_id"] == "Q09")
    assert q01["claim_ids"] == q09["claim_ids"] == ["Q01-C1", "Q01-C2"]
    assert q09["independent_intent"] is False
    blocked = hard_negative_ids(dataset)
    for question_id in ("Q03", "Q06", "Q07"):
        question = next(item for item in dataset["questions"] if item["question_id"] == question_id)
        for claim in claims_for_question(dataset, question):
            for span in claim["spans"]:
                assert span["element_id"] not in blocked


def test_q07_calculation_uses_raw_values_not_rounded_displays() -> None:
    dataset = load_dataset(DATASET)
    question = next(item for item in dataset["questions"] if item["question_id"] == "Q07")
    spec = question["calculation"]
    result = operating_margin_change(spec["operands"])
    assert result["published_10dp"] == spec["published_10dp"]
    assert result["display_2dp_half_up"] == {
        "fy2025": "18.82",
        "fy2024": "12.17",
        "delta_pp": "6.64",
    }
    assert result["rounded_display_difference"] == "6.65"
    assert result["used_rounded_display_difference"] is False


def test_q05_tables_are_not_summed() -> None:
    dataset = load_dataset(DATASET)
    view = q05_unique_amounts(dataset["claims"]["Q05-C1"], dataset["claims"]["Q05-C2"])
    assert view["same_totals"] is True
    assert view["naive_sum_forbidden"] is True
    assert view["progress_totals"] == ["400,000", "94,512", "305,488"]


def test_controlling_interest_does_not_satisfy_consolidated_net_income() -> None:
    dataset = load_dataset(DATASET)
    claim = dataset["claims"]["Q06-C3"]
    e228 = next(span["element_id"] for span in claim["spans"] if span["element_key"] == "E228")
    e227 = next(span["element_id"] for span in claim["spans"] if span["element_key"] == "E227")
    substitute = [
        {
            "element_id": e228,
            "kind": "table",
            "cells": [{"start_row": 14, "start_col": 1, "text_sha256": "x"}],
        }
    ]
    assert coverage_level(substitute, claim) == "none"
    complete = []
    for span in claim["spans"]:
        complete.append(
            {
                "element_id": span["element_id"],
                "kind": span["kind"],
                "char_start": span.get("char_start"),
                "char_end": span.get("char_end"),
                "cells": span.get("cells") or [],
            }
        )
    assert coverage_level(complete, claim) == "full"
    only_value = [item for item in complete if item["element_id"] == e228]
    assert coverage_level(only_value, claim) == "partial"
    caption = [item for item in complete if item["element_id"] == e227]
    assert single_and_set(
        [{"chunk_sources": only_value}, {"chunk_sources": caption}],
        claim,
    ) == {"single_chunk": "partial", "chunk_set": "full"}


def test_span_audit_separates_chunking_from_retrieval() -> None:
    claim = {
        "claim_id": "C",
        "required": True,
        "spans": [
            {
                "span_id": "S",
                "element_id": "e",
                "kind": "table_cells",
                "cells": [{"start_row": 1, "start_col": 1}],
            }
        ],
    }
    question = {"question_id": "Q", "claim_ids": ["C"]}
    present = {
        "chunk_id": "keep",
        "ordinal": 0,
        "chunk_sources": [
            {
                "element_id": "e",
                "kind": "table",
                "cells": [{"start_row": 1, "start_col": 1}],
            }
        ],
    }
    other = {"chunk_id": "other", "ordinal": 1, "chunk_sources": []}
    missed = audit_span_failures([present, other], question, [claim], [other], k=1)
    assert missed[0]["cause"] == "retrieval"
    absent = audit_span_failures([other], question, [claim], [other], k=1)
    assert absent[0]["cause"] == "chunking"


def test_bm25_is_deterministic_and_drops_zero_scores() -> None:
    assert tokenize("매출액 FY2025") == tokenize("매출액 FY2025")
    chunks = [
        {"chunk_id": "b", "ordinal": 1, "text": "연결 매출액"},
        {"chunk_id": "a", "ordinal": 0, "text": "연결 매출액"},
        {"chunk_id": "c", "ordinal": 2, "text": "시설투자"},
    ]
    first = Bm25Index(chunks).search("매출액", 3)
    second = Bm25Index(chunks).search("매출액", 3)
    assert first == second
    assert [hit["chunk_id"] for hit in first] == ["a", "b"]
    assert Bm25Index(chunks).search("완전히다른질의xyz", 3) == []


@pytest.mark.skipif(
    not (PINNED / "elements.jsonl").exists(),
    reason="pinned parse snapshot is local",
)
def test_pinned_business_report_baseline(tmp_path: Path) -> None:
    manifest = PINNED / "parse_manifest.json"
    elements = PINNED / "elements.jsonl"
    first = run_baseline(
        dataset_dir=DATASET,
        manifest_path=manifest,
        elements_path=elements,
        output_dir=tmp_path / "run1",
    )
    second = run_baseline(
        dataset_dir=DATASET,
        manifest_path=manifest,
        elements_path=elements,
        output_dir=tmp_path / "run2",
    )
    assert first["chunks"] == second["chunks"]
    left = first["questions"][0]["top5"][0]["chunk_id"]
    right = second["questions"][0]["top5"][0]["chunk_id"]
    assert left == right
    assert first["scope"]["document_name_observed"] == "사업보고서"
    assert first["scope"]["member_index"] == 0
    assert first["metrics"]["positive_question_count"] == 8
    assert first["metrics"]["at_k"]["5"]["claim_set_recall"]["denominator"] == 27
    assert "Q10" not in first["metrics"]["positive_question_ids"]
    assert "Q09" not in first["metrics"]["positive_question_ids"]
    chunk_lines = (tmp_path / "run1" / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    stored = json.loads(chunk_lines[0])
    assert stored["lineage"]["document_id"] == "m0:20260323000847.xml"
    e227 = "9f1d0a924bf776e4107153d15dd62dfbd92be3d0b82c2fa604f665ae1bf01b0b"
    e228 = "f55b61e36dd52bfcfeda562f3dcc163dd8bc4051f2176d3b248b13034f60f244"
    e81 = "9381f1567fbe783565a43c392f71aae7fc7f6a4f491befcc005375b2f8a7b8d9"
    e82 = "f483b0b273de86536c7841d2c38a126125c78a8437346dce45b0a26f4e802711"
    e186 = "78be6e94020c610d2411a83f2187ca58f098c8d9f071da7c4d78cb9db0ae5b62"
    chunks = [json.loads(line) for line in chunk_lines]
    assert all(chunk["lineage"]["document_id"] == "m0:20260323000847.xml" for chunk in chunks)
    revenue = []
    for chunk in chunks:
        coords = {
            (source["element_id"], cell["start_row"], cell["start_col"], source["role"])
            for source in chunk["chunk_sources"]
            for cell in source.get("cells") or []
        }
        body_rows = {
            cell["start_row"]
            for source in chunk["chunk_sources"]
            if source["element_id"] == e228 and source["role"] == "body"
            for cell in source["cells"]
        }
        if 1 in body_rows:
            revenue.append(coords)
            assert len(body_rows) < 20
        if (e82, 2, 2, "body") in {(item[0], item[1], item[2], item[3]) for item in coords}:
            assert any(source["element_id"] == e81 for source in chunk["chunk_sources"])
            header = next(
                cell
                for source in chunk["chunk_sources"]
                if source["element_id"] == e82
                for cell in source["cells"]
                if cell["start_row"] == 1 and cell["start_col"] == 2
            )
            assert header["header_role"] == "unknown"
            assert header["text_sha256"] == sha256_text("품 목")
    assert revenue
    assert any((e227, 4, 0) == (item[0], item[1], item[2]) for item in revenue[0])
    assert any((e227, 1, 0) == (item[0], item[1], item[2]) for item in revenue[0])
    assert any(
        source["element_id"] == e186 and source["role"] == "body"
        for chunk in chunks
        for source in chunk["chunk_sources"]
    )
    q10 = next(item for item in first["questions"] if item["question_id"] == "Q10")
    assert q10["in_positive_recall_denominator"] is False
    assert q10["answerability"] == "out_of_scope"

"""Chunker, source-span alignment, and the pinned BM25 baseline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from disclosure_agent.bm25 import Bm25Index, tokenize
from disclosure_agent.chunking import (
    canonical_dumps,
    chunk_elements,
    chunks_artifact_sha256,
    sha256_text,
)
from disclosure_agent.errors import UsageError
from disclosure_agent.evaluation import (
    _score_question,
    _validate_evidence_labels,
    audit_span_failures,
    claims_for_question,
    compare_retrievers,
    coverage_level,
    hard_negative_ids,
    load_dataset,
    operating_margin_change,
    positive_questions,
    q05_unique_amounts,
    run_baseline,
    select_evidence_path,
    single_and_set,
    stable_payload,
)

DATASET = Path(__file__).resolve().parents[1] / "evaluation" / "datasets" / "isu_petasys_v0_1"
DATASET_V02 = Path(__file__).resolve().parents[1] / "evaluation" / "datasets" / "isu_petasys_v0_2"
DATASET_V03 = Path(__file__).resolve().parents[1] / "evaluation" / "datasets" / "isu_petasys_v0_3"
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
    assert chunks_artifact_sha256(first) == chunks_artifact_sha256(second)
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
    assert missed[0]["cause"] == "retrieval_ranking_failure"
    assert missed[0]["detail"] == "zero_score_omitted"
    absent = audit_span_failures([other], question, [claim], [other], k=1)
    assert absent[0]["cause"] == "chunking_loss"


def test_embedded_unit_phrase_stays_in_the_body() -> None:
    section = ["III"]
    cells = [
        _cell(0, 0, "(단위 : 원)"),
        _cell(1, 0, "과목"),
        _cell(1, 1, "제 54 기"),
        _cell(2, 0, "기본주당이익(손실) (단위 : 원)"),
        _cell(2, 1, "2,278"),
    ]
    chunks = chunk_elements([_table(9, cells, section)], manifest=_manifest(), document_id=DOC)
    roles = {
        (source["role"], cell["start_row"])
        for chunk in chunks
        for source in chunk["chunk_sources"]
        for cell in source.get("cells") or []
    }
    assert ("unit", 0) in roles
    assert ("body", 2) in roles
    assert ("unit", 2) not in roles
    assert ("repeated_unit", 2) not in roles


def test_noncontiguous_caption_rows_are_not_one_rectangle() -> None:
    section = ["III"]
    caption = [
        _cell(0, 0, "연결 포괄손익계산서"),
        _cell(1, 0, "제 54 기 2025.01.01 부터 2025.12.31 까지"),
        _cell(2, 0, "비교 표시"),
        _cell(3, 0, "(단위 : 원)"),
    ]
    body = [_cell(0, 0, "과목"), _cell(0, 1, "금액"), _cell(1, 0, "매출"), _cell(1, 1, "1")]
    chunks = chunk_elements(
        [_table(0, caption, section), _table(1, body, section)],
        manifest=_manifest(),
        document_id=DOC,
    )
    title = next(
        source
        for chunk in chunks
        for source in chunk["chunk_sources"]
        if source["element_id"] == "t0" and source["role"] == "context_title"
    )
    assert title["rows_contiguous"] is False
    assert title["rows"] is None
    assert title["row_rects"] == [{"start": 0, "end": 0}, {"start": 2, "end": 2}]
    assert all(cell["start_row"] != 1 for cell in title["cells"])


def test_q02_labels_sales_type_rather_than_two_product_groups() -> None:
    dataset = load_dataset(DATASET)
    statements = " ".join(dataset["claims"][f"Q02-C{index}"]["statement"] for index in range(1, 5))
    by_id = {item["question_id"]: item for item in dataset["questions"]}
    assert "매출유형" in statements
    assert "별도 제품군이 아니다" in dataset["claims"]["Q02-C2"]["statement"]
    assert by_id["Q02"]["answerability"] == "partial"
    assert by_id["Q04"]["answerability"] == "partial"
    assert by_id["Q08"]["answerability"] == "partial"


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
    assert stable_payload(first) == stable_payload(second)
    left = first["questions"][0]["top5"][0]["chunk_id"]
    right = second["questions"][0]["top5"][0]["chunk_id"]
    assert left == right
    assert first["scope"]["document_name_observed"] == "사업보고서"
    assert first["scope"]["member_index"] == 0
    assert first["scope"]["parser_version"] == "2"
    assert first["scope"]["parser_config_hash"].startswith("2d75d2ab")
    assert first["reproducibility"]["source"]["source_sha256"].startswith("ff2ee7ca")
    assert first["reproducibility"]["evaluation"]["git_commit"]
    assert first["chunks"]["artifact_sha256"]
    assert first["metrics"]["positive_question_count"] == 8
    assert first["metrics"]["at_k"]["5"]["claim_set_recall"]["denominator"] == 27
    for key in ("1", "3", "5", "10", "20"):
        block = first["metrics"]["at_k"][key]
        legacy = block["claim_set_recall"]
        path_recall = block["answer_evidence_path_recall"]
        assert legacy["numerator"] == path_recall["numerator"]
        assert (
            block["claim_set_recall"]["denominator"]
            == block["answer_evidence_path_recall"]["denominator"]
        )
        assert (
            block["question_evidence_satisfied"]["numerator"]
            == block["question_answer_evidence_satisfied"]["numerator"]
        )
    assert first["metrics"]["at_k"]["5"]["question_answer_complete"]["denominator"] == 5
    assert first["metrics"]["at_k"]["5"]["question_source_ceiling_satisfied"]["denominator"] == 8
    assert first["metrics"]["answer_complete_excluded_question_ids"] == ["Q02", "Q04", "Q08"]
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
    assert q10["positive_claim_count"] == 0
    assert q10["abstention_generation_measured"] is False
    q01 = next(item for item in first["questions"] if item["question_id"] == "Q01")
    q09 = next(item for item in first["questions"] if item["question_id"] == "Q09")
    assert q01["diagnosis"]["token_overlap_with_first_claim_full_chunk"] > 0
    assert q09["diagnosis"]["token_overlap_with_first_claim_full_chunk"] == 0
    assert q09["diagnosis"]["zero_score_omitted"] is True
    assert q09["diagnosis"]["first_evidence_set_rank"] is None
    for question in first["questions"]:
        for row in question["hard_negative_ranks"]:
            assert row["counted_as_positive_evidence"] is False


def test_v0_2_extends_only_q07_and_keeps_legacy_denominator() -> None:
    old = load_dataset(DATASET)
    new = load_dataset(DATASET_V02)
    assert new["snapshot"]["dataset_id"] == "isu_petasys_v0_2"
    assert new["snapshot"]["supersedes"] == "isu_petasys_v0_1"
    assert new["hard_negatives"] == old["hard_negatives"]
    old_questions = {item["question_id"]: item for item in old["questions"]}
    new_questions = {item["question_id"]: item for item in new["questions"]}
    for question_id, previous in old_questions.items():
        current = new_questions[question_id]
        assert current["text"] == previous["text"]
        if question_id != "Q07":
            assert current == previous
            continue
        assert current["claim_ids"] == ["Q07-C1", "Q07-C2", "Q07-C3", "Q07-C4"]
        assert [path["path_id"] for path in current["evidence_paths"]] == [
            "direct_reported_metric",
            "derived_operands",
        ]
        assert current["calculation"]["display_2dp_half_up"]["delta_pp"] == "6.64"
        assert current["precision_policy"]["direct_reported_metric"]["delta_pp"] == "6.65"
        assert current["precision_policy"]["calculation_test_uses"] == "derived_operands"
    for claim_id, claim in old["claims"].items():
        assert new["claims"][claim_id] == claim
    direct = [new["claims"]["Q07-A1"], new["claims"]["Q07-A2"]]
    assert [claim["required"] for claim in direct] == [False, False]
    texts = []
    for claim in direct:
        for span in claim["spans"]:
            for cell in span.get("cells") or []:
                texts.append(cell["text"])
            if span["kind"] == "paragraph":
                texts.append(span["text"])
    assert "18.82%" in texts
    assert "12.17%" in texts
    assert "매출액영업이익률" in texts
    assert any("(연결기준)" in text for text in texts)
    assert "2) 수익성지표" in texts
    blocked = hard_negative_ids(new)
    q07 = new_questions["Q07"]
    for path in q07["evidence_paths"]:
        for claim_id in path["claim_ids"]:
            for span in new["claims"][claim_id]["spans"]:
                assert span["element_id"] not in blocked
    result = operating_margin_change(q07["calculation"]["operands"])
    assert result["display_2dp_half_up"]["delta_pp"] == "6.64"
    assert result["rounded_display_difference"] == "6.65"
    assert result["used_rounded_display_difference"] is False


def test_one_evidence_path_satisfies_retrieval_without_the_other() -> None:
    direct = {
        "claim_id": "Q07-A1",
        "required": False,
        "spans": [
            {
                "span_id": "value",
                "element_id": "direct",
                "kind": "table_cells",
                "evidence_role": "value",
                "cells": [{"start_row": 2, "start_col": 2}],
            }
        ],
    }
    derived = {
        "claim_id": "Q07-C1",
        "required": True,
        "spans": [
            {
                "span_id": "operand",
                "element_id": "statement",
                "kind": "table_cells",
                "cells": [{"start_row": 1, "start_col": 1}],
            }
        ],
    }
    question = {
        "question_id": "Q07",
        "intent_group": "G07",
        "text": "영업이익률",
        "answerability": "supported",
        "in_positive_recall_denominator": True,
        "independent_intent": True,
        "claim_ids": ["Q07-C1"],
        "evidence_paths": [
            {"path_id": "direct_reported_metric", "claim_ids": ["Q07-A1"]},
            {"path_id": "derived_operands", "claim_ids": ["Q07-C1"]},
        ],
    }
    dataset = {
        "questions": [question],
        "claims": {"Q07-A1": direct, "Q07-C1": derived},
        "hard_negatives": [],
    }
    chunk = {
        "chunk_id": "direct-chunk",
        "ordinal": 0,
        "text": "매출액영업이익률 18.82%",
        "section_path": ["IV"],
        "chunk_sources": [
            {
                "element_id": "direct",
                "kind": "table",
                "role": "body",
                "cells": [{"start_row": 2, "start_col": 2}],
            }
        ],
    }
    hits = [{"chunk_id": "direct-chunk", "rank": 1, "score": 3.0, "ordinal": 0}]
    scored = _score_question(dataset, question, [chunk], hits, (1, 3, 5, 10, 20))
    top = scored["at_k"]["1"]
    assert top["evidence_set_satisfied"] is False
    assert top["answer_evidence_success"] is True
    assert top["answer_evidence_path_id"] == "direct_reported_metric"
    assert top["selected_path_required_claims"] == 1
    assert scored["answer_evidence"]["path_ranks"]["direct_reported_metric"][
        "first_value_set_rank"
    ] == 1
    assert scored["hard_negative_ranks"] == []
    neither = select_evidence_path(
        [
            {
                "path_id": "direct_reported_metric",
                "path_set_satisfied": False,
                "claim_set_full": 0,
                "required_claims": 2,
            },
            {
                "path_id": "derived_operands",
                "path_set_satisfied": False,
                "claim_set_full": 0,
                "required_claims": 4,
            },
        ]
    )
    assert neither is not None
    assert neither["path_id"] == "direct_reported_metric"


@pytest.mark.skipif(
    not (PINNED / "elements.jsonl").exists(),
    reason="pinned parse snapshot is local",
)
def test_pinned_v0_2_baseline_keeps_bm25_and_splits_q07(tmp_path: Path) -> None:
    payload = run_baseline(
        dataset_dir=DATASET_V02,
        manifest_path=PINNED / "parse_manifest.json",
        elements_path=PINNED / "elements.jsonl",
        output_dir=tmp_path / "v02",
    )
    assert payload["chunks"]["chunk_count"] == 2803
    assert payload["reproducibility"]["chunking"]["chunker_config_hash"].startswith("9720cb47")
    assert payload["tokenizer_version"] == "ko-mixed-v1"
    assert payload["bm25"]["k1"] == 1.2
    assert payload["bm25"]["b"] == 0.75
    assert payload["reproducibility"]["evaluation"]["dataset_id"] == "isu_petasys_v0_2"
    assert payload["scope"]["member_index"] == 0
    legacy = payload["metrics"]["at_k"]["5"]["claim_set_recall"]
    assert legacy["denominator"] == 27
    q07 = next(item for item in payload["questions"] if item["question_id"] == "Q07")
    ranks = q07["answer_evidence"]["path_ranks"]
    assert ranks["direct_reported_metric"]["first_value_set_rank"] == 1
    assert "derived_operands" in ranks
    for row in q07["hard_negative_ranks"]:
        assert row["element_key"] == "E186"
        assert row["counted_as_positive_evidence"] is False
    calc = payload["calculation_q07"]
    assert calc["status"] == "success"
    assert calc["retrieval_answer_evidence_included"] is False
    assert calc["display_2dp_half_up"]["delta_pp"] == "6.64"
    assert calc["rounded_display_difference"] == "6.65"
    assert payload["metrics"]["at_k"]["5"]["relevant_conflict_hit"]["denominator"] == 0


def test_evidence_class_rejects_answer_valid_hard_negative() -> None:
    with pytest.raises(UsageError):
        _validate_evidence_labels(
            [
                {
                    "element_key": "E1",
                    "evidence_class": "relevant_conflicting",
                    "topical_relevance": True,
                    "answer_valid_evidence": True,
                    "conflicting_evidence": True,
                    "classified_for": ["Q07"],
                }
            ]
        )


def test_v0_3_keeps_v0_2_answer_gold_and_classifies_conflict() -> None:
    old = load_dataset(DATASET_V02)
    new = load_dataset(DATASET_V03)
    assert new["snapshot"]["dataset_id"] == "isu_petasys_v0_3"
    assert new["snapshot"]["evaluation_schema_version"] == 2
    assert new["snapshot"]["gold_answer_evidence_same_as"] == "isu_petasys_v0_2"
    assert new["snapshot"]["supersedes"] == "isu_petasys_v0_2"
    assert new["questions"] == old["questions"]
    assert new["claims"] == old["claims"]
    label = new["hard_negatives"][0]
    assert label["element_key"] == "E186"
    assert label["evidence_class"] == "relevant_conflicting"
    assert label["topical_relevance"] is True
    assert label["answer_valid_evidence"] is False
    assert label["conflicting_evidence"] is True
    assert label["classified_for"] == ["Q03", "Q06", "Q07"]
    assert label["not_gold_for"] == old["hard_negatives"][0]["not_gold_for"]


def test_conflicting_hit_is_not_answer_evidence() -> None:
    claim = {
        "claim_id": "Q07-A1",
        "required": False,
        "spans": [
            {
                "span_id": "value",
                "element_id": "direct",
                "element_key": "E2362",
                "kind": "table_cells",
                "evidence_role": "value",
                "cells": [
                    {"start_row": 2, "start_col": 2},
                    {"start_row": 2, "start_col": 3},
                ],
            }
        ],
    }
    question = {
        "question_id": "Q07",
        "intent_group": "G07",
        "text": "연결 영업이익률",
        "answerability": "supported",
        "in_positive_recall_denominator": True,
        "independent_intent": True,
        "claim_ids": ["Q07-A1"],
        "evidence_paths": [
            {"path_id": "direct_reported_metric", "claim_ids": ["Q07-A1"]},
        ],
    }
    dataset = {
        "questions": [question],
        "claims": {"Q07-A1": claim},
        "hard_negatives": [
            {
                "element_key": "E186",
                "element_id": "e186",
                "evidence_class": "relevant_conflicting",
                "topical_relevance": True,
                "answer_valid_evidence": False,
                "conflicting_evidence": True,
                "classified_for": ["Q07"],
                "not_gold_for": ["Q07"],
            }
        ],
    }
    partial = {
        "chunk_id": "partial",
        "ordinal": 0,
        "text": "매출액영업이익률 18.82%",
        "section_path": ["IV"],
        "chunk_sources": [
            {
                "element_id": "direct",
                "kind": "table",
                "role": "body",
                "cells": [{"start_row": 2, "start_col": 2}],
            }
        ],
    }
    conflict = {
        "chunk_id": "conflict",
        "ordinal": 1,
        "text": "매출액 1,880,025",
        "section_path": ["II"],
        "chunk_sources": [
            {
                "element_id": "e186",
                "kind": "table",
                "role": "body",
                "cells": [{"start_row": 1, "start_col": 1}],
            }
        ],
    }
    hits = [
        {"chunk_id": "partial", "rank": 1, "score": 1.0, "ordinal": 0},
        {"chunk_id": "conflict", "rank": 2, "score": 0.9, "ordinal": 1},
    ]
    scored = _score_question(
        dataset, question, [partial, conflict], hits, (1, 3, 5, 10, 20)
    )
    assert scored["at_k"]["5"]["answer_evidence_success"] is False
    assert scored["at_k"]["5"]["relevant_conflicting_hit"] is True
    assert scored["top5"][0]["evidence_classes"] == ["positive_answer_evidence"]
    assert scored["top5"][1]["evidence_classes"] == ["relevant_conflicting_evidence"]
    assert scored["hard_negative_ranks"][0]["counted_as_positive_evidence"] is False
    assert scored["hard_negative_ranks"][0]["rank"] == 2
    classification = scored["evidence_classification"]
    assert classification["first_positive_answer_evidence_rank"] == 1
    assert classification["first_complete_evidence_path_rank"] is None
    assert classification["at_5"]["positive_answer_evidence"] is False
    assert classification["at_5"]["relevant_conflicting_evidence"] is True


def test_exact_dense_search_uses_normalized_dot_product() -> None:
    numpy = pytest.importorskip("numpy")
    from disclosure_agent.dense import exact_normalized_search

    chunks = [
        {"chunk_id": "a", "ordinal": 0},
        {"chunk_id": "b", "ordinal": 1},
        {"chunk_id": "c", "ordinal": 2},
    ]
    docs = numpy.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]], dtype=numpy.float32)
    query = numpy.array([0.6, 0.8], dtype=numpy.float32)
    hits = exact_normalized_search(docs, query, chunks)
    assert [hit["chunk_id"] for hit in hits] == ["c", "b", "a"]
    assert hits[0]["score"] == pytest.approx(1.0)
    tied = numpy.array([[1.0, 0.0], [1.0, 0.0]], dtype=numpy.float32)
    tied_hits = exact_normalized_search(
        tied,
        numpy.array([1.0, 0.0], dtype=numpy.float32),
        chunks[:2],
    )
    assert [hit["chunk_id"] for hit in tied_hits] == ["a", "b"]


def test_compare_retrievers_keeps_conflict_separate_from_success() -> None:
    def question(question_id: str, positive: int | None, conflict: int | None, success: bool):
        return {
            "question_id": question_id,
            "at_k": {
                "5": {
                    "answer_evidence_success": success,
                    "relevant_conflicting_hit": conflict is not None and conflict <= 5,
                }
            },
            "evidence_classification": {
                "first_positive_answer_evidence_rank": positive,
                "first_complete_evidence_path_rank": positive if success else None,
                "relevant_conflicting_ranks": [
                    {"element_key": "E186", "rank": conflict, "element_id": "e", "chunk_id": "c"}
                ]
                if conflict is not None
                else [],
            },
        }

    left = [question("Q07", 1, 2, False)]
    right = [question("Q07", 4, 3, False)]
    compared = compare_retrievers(left, right)
    row = compared["questions"][0]
    assert row["baseline_question_success_at_5"] is False
    assert row["candidate_question_success_at_5"] is False
    assert row["baseline_conflicting_evidence_rank"] == 2
    assert row["candidate_conflicting_evidence_rank"] == 3


@pytest.mark.skipif(
    not (PINNED / "elements.jsonl").exists(),
    reason="pinned parse snapshot is local",
)
def test_pinned_v0_3_bm25_keeps_ranks_and_records_conflict(tmp_path: Path) -> None:
    payload = run_baseline(
        dataset_dir=DATASET_V03,
        manifest_path=PINNED / "parse_manifest.json",
        elements_path=PINNED / "elements.jsonl",
        output_dir=tmp_path / "v03",
    )
    assert payload["retriever_type"] == "bm25"
    assert payload["chunks"]["chunk_count"] == 2803
    assert payload["chunks"]["artifact_sha256"].startswith("86d4832d")
    assert payload["reproducibility"]["chunking"]["chunker_config_hash"].startswith("9720cb47")
    assert payload["reproducibility"]["evaluation"]["evaluation_schema_version"] == 2
    assert payload["reproducibility"]["evaluation"]["dataset_id"] == "isu_petasys_v0_3"
    q07 = next(item for item in payload["questions"] if item["question_id"] == "Q07")
    ranks = q07["answer_evidence"]["path_ranks"]
    assert ranks["direct_reported_metric"]["first_value_set_rank"] == 1
    assert ranks["direct_reported_metric"]["first_set_full_rank"] == 144
    assert ranks["derived_operands"]["first_set_full_rank"] == 220
    conflict = q07["hard_negative_ranks"][0]
    assert conflict["element_key"] == "E186"
    assert conflict["rank"] == 2
    assert conflict["counted_as_positive_evidence"] is False
    assert conflict["counts_as_relevant_conflict"] is True
    assert q07["at_k"]["5"]["answer_evidence_success"] is False
    assert q07["at_k"]["20"]["answer_evidence_success"] is False
    assert q07["at_k"]["5"]["relevant_conflicting_hit"] is True
    assert q07["evidence_classification"]["at_5"]["positive_answer_evidence"] is False
    diagnostic = payload["metrics"]["at_k"]["5"]["relevant_conflict_hit"]
    assert diagnostic["denominator"] == 3
    assert diagnostic["not_a_quality_score"] is True
    assert diagnostic["question_ids"] == ["Q03", "Q06", "Q07"]
    q09 = next(item for item in payload["questions"] if item["question_id"] == "Q09")
    assert q09["diagnosis"]["zero_score_omitted"] is True

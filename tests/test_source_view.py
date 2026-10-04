"""Public source blocks stay tied to parsed cells, not flattened retrieval text."""

from __future__ import annotations

import json
from pathlib import Path

from disclosure_agent.agent.catalog import repo_root
from disclosure_agent.agent.context import evidence_prompt_payload
from disclosure_agent.agent.evidence import EvidenceBuilder
from disclosure_agent.agent.search import SearchHit
from disclosure_agent.agent.source_view import project_content_blocks

_NEEDLE = "사업부문 | 연결 회사명 | 주요 생산 및 판매제품 유형"


def test_pipe_text_stays_text_until_a_parsed_table_is_named() -> None:
    blocks = project_content_blocks(
        [
            {
                "element_id": "paragraph",
                "kind": "paragraph",
                "char_start": 0,
                "char_end": len(_NEEDLE),
            }
        ],
        {"paragraph": {"element_type": "paragraph", "text": _NEEDLE}},
    )
    assert blocks == [{"type": "text", "text": _NEEDLE}]


def test_table_block_uses_parsed_cells_and_keeps_spans() -> None:
    elements = {
        "table": {
            "element_type": "table",
            "cells": [
                _cell(0, 0, "사업부문"),
                _cell(0, 1, "연결 회사명"),
                _cell(1, 0, "PCB 생산과 판매", rowspan=4),
                _cell(1, 1, "한국 ㈜이수페타시스"),
            ],
        }
    }
    refs = [
        {
            "element_id": "table",
            "kind": "table",
            "role": "header",
            "cells": [_pointer(0, 0), _pointer(0, 1)],
        },
        {
            "element_id": "table",
            "kind": "table",
            "role": "body",
            "cells": [_pointer(1, 0, rowspan=4), _pointer(1, 1)],
        },
    ]
    blocks = project_content_blocks(refs, elements)
    assert [block["type"] for block in blocks] == ["table"]
    table = blocks[0]
    assert table["row_count"] == 5
    assert table["col_count"] == 2
    headers = [cell["text"] for cell in table["cells"] if cell["header"]]
    assert headers == ["사업부문", "연결 회사명"]
    span = next(cell for cell in table["cells"] if cell["text"] == "PCB 생산과 판매")
    assert span["rowspan"] == 4
    assert span["header"] is False
    assert "source_sha256" not in json.dumps(table)


def test_cell_hash_mismatch_does_not_invent_a_table() -> None:
    elements = {"table": {"element_type": "table", "cells": [_cell(0, 0, "사업부문")]}}
    blocks = project_content_blocks(
        [
            {
                "element_id": "table",
                "kind": "table",
                "role": "header",
                "cells": [_pointer(0, 0, digest="0" * 64)],
            }
        ],
        elements,
    )
    assert blocks == []


def test_prompt_payload_keeps_retrieval_text_and_omits_blocks() -> None:
    hit = _hit()
    evidence = EvidenceBuilder(
        element_index={"table": (Path("unused"), 0)},
    ).build([hit], company_name="이수페타시스")
    item = evidence[0]
    assert item.text == _NEEDLE
    payload = evidence_prompt_payload(item)
    assert "blocks" not in payload
    assert payload["text"] == _NEEDLE
    assert item.to_dict()["blocks"] == []
    assert item.to_dict()["evidence_id"] == item.evidence_id


def test_isu_business_table_restores_the_parsed_grid() -> None:
    unit, elements_path = _isu_business_unit()
    locators, index = _index(elements_path)
    builder = EvidenceBuilder(locators, index)
    hit = _hit_from_unit(unit)
    built = builder.build([hit], company_name="이수페타시스")
    item = built[0]
    assert item.text == unit["text"]
    assert item.receipt_no == unit["receipt_no"]
    assert item.company_name == "이수페타시스"
    assert "blocks" not in evidence_prompt_payload(item)
    table = next(block for block in item.blocks if block["type"] == "table")
    headers = [cell["text"] for cell in table["cells"] if cell["header"]]
    assert headers[:3] == ["사업부문", "연결 회사명", "주요 생산 및 판매제품 유형"]
    span = next(cell for cell in table["cells"] if cell["text"] == "PCB 생산과 판매")
    assert span["rowspan"] == 4
    assert span["start_row"] == 1
    assert span["start_col"] == 0
    prose = next(block for block in item.blocks if block["type"] == "text")
    assert "인쇄회로기판" in prose["text"]
    assert "|" not in prose["text"]
    joined = " ".join(cell["text"] for cell in table["cells"])
    assert " | " not in joined


def _cell(row: int, col: int, text: str, *, rowspan: int = 1) -> dict[str, object]:
    return {
        "start_row": row,
        "start_col": col,
        "rowspan": rowspan,
        "colspan": 1,
        "text": text,
        "header_role": "unknown",
        "tag": "TD",
    }


def _pointer(
    row: int,
    col: int,
    *,
    rowspan: int = 1,
    digest: str | None = None,
) -> dict[str, object]:
    import hashlib

    text = {
        (0, 0): "사업부문",
        (0, 1): "연결 회사명",
        (1, 0): "PCB 생산과 판매",
        (1, 1): "한국 ㈜이수페타시스",
    }[(row, col)]
    pointer: dict[str, object] = {
        "start_row": row,
        "start_col": col,
        "rowspan": rowspan,
        "colspan": 1,
        "tag": "TD",
        "header_role": "unknown",
        "text_sha256": digest or hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    return pointer


def _hit() -> SearchHit:
    return SearchHit(
        retrieval_unit_id="unit",
        score=0.1,
        rank=1,
        text=_NEEDLE,
        canonical_chunk_id="chunk",
        unit_index=0,
        source_element_ids=["missing"],
        source_refs=[],
        receipt_no="20240321001149",
        document_name="사업보고서",
        section_path=["II. 사업의 내용"],
        source_locator=[],
        report_name="사업보고서 (2023.12)",
        filing_period="2023.12",
        receipt_date="20240321",
        source_sha256="abc",
        corp_code="00107613",
    )


def _hit_from_unit(unit: dict[str, object]) -> SearchHit:
    return SearchHit(
        retrieval_unit_id=str(unit["retrieval_unit_id"]),
        score=0.2,
        rank=1,
        text=str(unit["text"]),
        canonical_chunk_id=str(unit["canonical_chunk_id"]),
        unit_index=int(unit["unit_index"]),
        source_element_ids=[str(item) for item in unit["source_element_ids"]],
        source_refs=list(unit["source_refs"]),
        receipt_no=str(unit["receipt_no"]),
        document_name=str(unit["document_name"]),
        section_path=[str(item) for item in unit["section_path"]],
        source_locator=[],
        report_name=str(unit.get("report_name") or ""),
        filing_period=str(unit.get("filing_period") or ""),
        receipt_date=str(unit.get("receipt_date") or ""),
        source_sha256=str(unit.get("source_sha256") or ""),
        corp_code=str(unit.get("corp_code") or ""),
    )


def _isu_business_unit() -> tuple[dict[str, object], Path]:
    root = repo_root()
    index = root / "data" / "product" / "indexes" / "isu_petasys_periodic_d2a"
    units = index / "retrieval_units.jsonl"
    corpus = index / "corpus.json"
    if not units.is_file() or not corpus.is_file():
        raise AssertionError("이수페타시스 product index is required for this source check")
    found = None
    with units.open(encoding="utf-8") as handle:
        for line in handle:
            if _NEEDLE not in line:
                continue
            row = json.loads(line)
            if _NEEDLE in str(row.get("text") or ""):
                found = row
                break
    if found is None:
        raise AssertionError("이수페타시스 business table unit was not found")
    manifest = json.loads(corpus.read_text(encoding="utf-8"))
    member = next(
        item for item in manifest["members"] if item["receipt_no"] == found["receipt_no"]
    )
    path = root / str(member["elements_path"])
    if not path.is_file():
        raise AssertionError(f"parsed elements are missing: {path}")
    return found, path


def _index(path: Path):
    from disclosure_agent.agent.evidence import load_element_index

    return load_element_index(path)

"""Parser contracts: fixtures plus pinned-source safety. Not a quality claim about every filing."""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pytest

from disclosure_agent.cli import run_parse_cli
from disclosure_agent.errors import ParseLockTimeout, UsageError
from disclosure_agent.parsing import (
    PARSER_VERSION,
    ParseInputError,
    _bind_tree,
    _local,
    _locator_for,
    config_hash,
    element_text_at,
    parse_receipt,
    pin_source,
    result_dir,
)

RECEIPT = "20240101000001"


def _xml(body: str) -> str:
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<DOCUMENT xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">\n'
        "<DOCUMENT-NAME>테스트보고서</DOCUMENT-NAME>\n"
        "<COMPANY-NAME>가나다</COMPANY-NAME>\n"
        f"<BODY>\n{body}\n</BODY>\n</DOCUMENT>\n"
    )


def _zip(members: list[tuple[str, str | bytes]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        for name, payload in members:
            data = payload if isinstance(payload, bytes) else payload.encode("utf-8")
            archive.writestr(name, data)
    return buf.getvalue()


def _store(raw: Path, payload: bytes, *, receipt: str = RECEIPT, sha: str | None = None) -> str:
    digest = sha or hashlib.sha256(payload).hexdigest()
    dest = raw / "dart" / receipt
    version = dest / "versions" / digest
    version.mkdir(parents=True)
    (version / "document.zip").write_bytes(payload)
    (version / "version.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "receipt_no": receipt,
                "sha256": digest,
                "size_bytes": len(payload),
                "listing": {"report_nm": "사업보고서 (2025.12)", "corp_name": "가나다"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (dest / "current.json").write_text(
        json.dumps({"receipt_no": receipt, "current_sha256": digest}),
        encoding="utf-8",
    )
    return digest


def _elements(dest: Path) -> list[dict]:
    path = next(dest.rglob("elements.jsonl"))
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_inline_newline_heading_order_and_repeat(tmp_path: Path) -> None:
    body = """
    <SECTION-1>
      <TITLE>I. 개요</TITLE>
      <P>같은 문장입니다.</P>
      <P>앞문장 <SPAN>중간</SPAN>끝문장
줄바꿈</P>
      <P>같은 문장입니다.</P>
      <TABLE>
        <TBODY>
          <TR>
            <TH ROWSPAN="2">항목</TH>
            <TH COLSPAN="2">금액</TH>
          </TR>
          <TR>
            <TD>(단위: 원)</TD>
            <TD></TD>
          </TR>
          <TR>
            <TD>-</TD>
            <TD>0</TD>
            <TD>(1,200)</TD>
          </TR>
        </TBODY>
      </TABLE>
      <P>표 밖 문단</P>
    </SECTION-1>
    """
    digest = _store(tmp_path / "raw", _zip([(f"{RECEIPT}.xml", _xml(body))]))
    result = parse_receipt(RECEIPT, tmp_path / "raw", tmp_path / "parsed")
    assert result["status"] == "success"
    assert result["source_sha256"] == digest
    elements = _elements(tmp_path / "parsed")
    kinds = [item["element_type"] for item in elements]
    assert kinds == ["heading", "paragraph", "paragraph", "paragraph", "table", "paragraph"]
    assert elements[0]["text"] == "I. 개요"
    assert elements[0]["section_path"] == []
    assert elements[1]["section_path"] == ["I. 개요"]
    assert elements[1]["text"] == "같은 문장입니다."
    assert elements[3]["text"] == "같은 문장입니다."
    assert elements[1]["element_id"] != elements[3]["element_id"]
    assert elements[2]["text"] == "앞문장 중간끝문장\n줄바꿈"
    table = elements[4]["table"]
    texts = [cell["text"] for cell in table["cells"]]
    assert texts == ["항목", "금액", "(단위: 원)", "", "-", "0", "(1,200)"]
    assert table["cells"][0]["rowspan"] == 2
    assert table["cells"][1]["colspan"] == 2
    assert table["cells"][0]["header_role"] == "header"
    assert table["cells"][4]["header_role"] == "unknown"
    paragraphs = [item["text"] for item in elements if item["element_type"] == "paragraph"]
    assert "표 밖 문단" in paragraphs
    assert all("항목" not in item for item in paragraphs)
    xml = _xml(body)
    assert element_text_at(xml, elements[2]) == elements[2]["text"]
    again = parse_receipt(RECEIPT, tmp_path / "raw", tmp_path / "parsed")
    assert again["reused"] is True
    assert _elements(tmp_path / "parsed")[2]["element_id"] == elements[2]["element_id"]


def test_nested_table_not_copied_and_angle_text_kept(tmp_path: Path) -> None:
    body = """
    <P>본문 &amp; 기호</P>
    <P>변경안 &lt;신설&gt; 유지</P>
    <TABLE>
      <TBODY>
        <TR>
          <TD>바깥<TABLE><TBODY><TR><TD>안쪽</TD><TD>-</TD></TR></TBODY></TABLE></TD>
          <TD>0</TD>
        </TR>
      </TBODY>
    </TABLE>
    """
    # The stored original contains a literal <신설>, which is not a tag.
    raw_xml = _xml(body).replace("&lt;신설&gt;", "<신설>")
    members = [(f"{RECEIPT}.xml", raw_xml), (f"{RECEIPT}_note.bin", b"\xff\xfe")]
    _store(tmp_path / "raw", _zip(members))
    result = parse_receipt(RECEIPT, tmp_path / "raw", tmp_path / "parsed")
    assert result["status"] == "success"
    elements = _elements(tmp_path / "parsed")
    paragraphs = [item["text"] for item in elements if item["element_type"] == "paragraph"]
    assert "본문 & 기호" in paragraphs
    assert "변경안 <신설> 유지" in paragraphs
    tables = [item for item in elements if item["element_type"] == "table"]
    assert len(tables) == 2
    outer = tables[0]
    inner = tables[1]
    assert outer["table"]["cells"][0]["text"] == "바깥"
    assert outer["table"]["cells"][0]["nested_table_element_ids"] == [inner["element_id"]]
    assert "안쪽" not in outer["text"]
    manifest_path = next((tmp_path / "parsed").rglob("parse_manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    statuses = {item["member_name"]: item["status"] for item in manifest["members"]}
    assert statuses[f"{RECEIPT}.xml"] == "parsed"
    assert statuses[f"{RECEIPT}_note.bin"] == "excluded"
    assert manifest["source_metadata"]["listing"]["corp_name"] == "가나다"
    assert "report_fiscal_year" not in json.dumps(manifest)


def test_multiple_nested_tables_in_one_cell_are_all_linked(tmp_path: Path) -> None:
    body = """
    <TABLE>
      <TBODY>
        <TR>
          <TD>사업부문
            <TABLE><TBODY><TR><TD>매출</TD></TR></TBODY></TABLE>
            <TABLE><TBODY><TR><TD>고객</TD></TR></TBODY></TABLE>
          </TD>
        </TR>
      </TBODY>
    </TABLE>
    """
    _store(tmp_path / "raw", _zip([(f"{RECEIPT}.xml", _xml(body))]))
    result = parse_receipt(RECEIPT, tmp_path / "raw", tmp_path / "parsed")
    assert result["status"] == "success"
    tables = [item for item in _elements(tmp_path / "parsed") if item["element_type"] == "table"]
    assert len(tables) == 3
    outer, sales, customers = tables
    assert [cell["text"] for cell in sales["table"]["cells"]] == ["매출"]
    assert [cell["text"] for cell in customers["table"]["cells"]] == ["고객"]
    assert outer["table"]["cells"][0]["nested_table_element_ids"] == [
        sales["element_id"],
        customers["element_id"],
    ]
    assert "multiple_nested_tables_in_one_cell" in outer["warnings"]
    assert "nested_table_locator" not in outer["table"]["cells"][0]
    assert "nested_table_locators" not in outer["table"]["cells"][0]


def test_oversized_span_is_clamped_without_allocating_the_declared_grid(
    tmp_path: Path,
) -> None:
    body = """
    <P>앞</P>
    <TABLE>
      <TBODY>
        <TR><TD ROWSPAN="99999" COLSPAN="99999">큰칸</TD><TD>옆</TD></TR>
        <TR><TD>다음</TD></TR>
      </TBODY>
    </TABLE>
    <P>뒤</P>
    """
    _store(tmp_path / "raw", _zip([(f"{RECEIPT}.xml", _xml(body))]))
    result = parse_receipt(RECEIPT, tmp_path / "raw", tmp_path / "parsed")
    assert result["status"] == "partial"
    assert result["search_eligible"] is False
    manifest = json.loads(
        next((tmp_path / "parsed").rglob("parse_manifest.json")).read_text(encoding="utf-8")
    )
    assert "table_span_clamped" in manifest["members"][0]["warnings"]
    elements = _elements(tmp_path / "parsed")
    paragraphs = [item["text"] for item in elements if item["element_type"] == "paragraph"]
    assert paragraphs == ["앞", "뒤"]
    table = next(item for item in elements if item["element_type"] == "table")
    cell = table["table"]["cells"][0]
    assert cell["text"] == "큰칸"
    assert cell["declared_rowspan"] == 99999
    assert cell["declared_colspan"] == 99999
    assert cell["rowspan"] <= 1000
    assert cell["colspan"] <= 100
    assert cell["rowspan"] * cell["colspan"] <= 50_000
    assert "table_span_clamped" in table["warnings"]


def test_image_recorded_and_locator_roundtrip(tmp_path: Path) -> None:
    body = """
    <COVER>
      <COVER-TITLE>사 업 보 고 서</COVER-TITLE>
      <IMAGE><IMG>확인.jpg</IMG><IMG-CAPTION>확인</IMG-CAPTION></IMAGE>
    </COVER>
    """
    xml = _xml(body)
    _store(tmp_path / "raw", _zip([(f"{RECEIPT}.xml", xml)]))
    parse_receipt(RECEIPT, tmp_path / "raw", tmp_path / "parsed")
    elements = _elements(tmp_path / "parsed")
    assert elements[0]["element_type"] == "heading"
    assert elements[0]["text"] == "사 업 보 고 서"
    image = elements[1]
    assert image["element_type"] == "image"
    assert "image_not_interpreted" in image["warnings"]
    assert element_text_at(xml, image) == image["text"]


def test_pin_ignores_later_current_change(tmp_path: Path) -> None:
    first = _zip([(f"{RECEIPT}.xml", _xml("<P>첫번째</P>"))])
    second = _zip([(f"{RECEIPT}.xml", _xml("<P>두번째</P>"))])
    raw = tmp_path / "raw"
    digest = _store(raw, first)
    other = hashlib.sha256(second).hexdigest()
    version = raw / "dart" / RECEIPT / "versions" / other
    version.mkdir()
    (version / "document.zip").write_bytes(second)
    (version / "version.json").write_text(
        json.dumps({"receipt_no": RECEIPT, "sha256": other, "size_bytes": len(second)}),
        encoding="utf-8",
    )
    source = pin_source(raw, RECEIPT, None)
    assert source.source_sha256 == digest
    (raw / "dart" / RECEIPT / "current.json").write_text(
        json.dumps({"current_sha256": other}),
        encoding="utf-8",
    )
    result = parse_receipt(RECEIPT, raw, tmp_path / "parsed", source_sha256=source.source_sha256)
    assert result["source_sha256"] == digest
    assert _elements(tmp_path / "parsed")[0]["text"] == "첫번째"
    other_result = parse_receipt(
        RECEIPT, raw, tmp_path / "parsed", source_sha256=other
    )
    assert other_result["source_sha256"] == other
    assert other_result["dest"] != result["dest"]


def test_damaged_inputs_rejected(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    payload = _zip([(f"{RECEIPT}.xml", _xml("<P>정상</P>"))])
    digest = _store(raw, payload)
    version = raw / "dart" / RECEIPT / "versions" / digest
    (version / "document.zip").write_bytes(b"not-a-zip")
    with pytest.raises(ParseInputError):
        parse_receipt(RECEIPT, raw, tmp_path / "parsed")

    entity_xml = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<!DOCTYPE foo [<!ENTITY xxe SYSTEM "http://127.0.0.1/secret">]>\n'
        "<DOCUMENT><BODY><P>x</P></BODY></DOCUMENT>\n"
    )
    digest2 = _store(tmp_path / "raw2", _zip([(f"{RECEIPT}.xml", entity_xml)]))
    result = parse_receipt(RECEIPT, tmp_path / "raw2", tmp_path / "parsed2")
    assert result["status"] == "failed"
    manifest = json.loads(
        next((tmp_path / "parsed2").rglob("parse_manifest.json")).read_text(encoding="utf-8")
    )
    assert manifest["members"][0]["reason"] == "external_entity_or_doctype_rejected"
    assert digest2


def test_hash_mismatch_and_usage(tmp_path: Path) -> None:
    payload = _zip([(f"{RECEIPT}.xml", _xml("<P>ㄱ</P>"))])
    raw = tmp_path / "raw"
    digest = _store(raw, payload)
    meta_path = raw / "dart" / RECEIPT / "versions" / digest / "version.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["sha256"] = "a" * 64
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ParseInputError):
        parse_receipt(RECEIPT, raw, tmp_path / "parsed", source_sha256=digest)
    with pytest.raises(UsageError):
        parse_receipt("12", raw, tmp_path / "parsed")


def test_size_limit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    payload = _zip([(f"{RECEIPT}.xml", _xml("<P>ㄱ</P>"))])
    _store(tmp_path / "raw", payload)
    monkeypatch.setattr("disclosure_agent.parsing.MAX_MEMBER_BYTES", 10)
    # validate_zip still passes; member limit is separate.
    result = parse_receipt(RECEIPT, tmp_path / "raw", tmp_path / "parsed")
    assert result["status"] == "failed"
    assert result["parser_version"] == PARSER_VERSION


def test_interrupt_does_not_publish_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _zip([(f"{RECEIPT}.xml", _xml("<P>중단</P>"))])
    digest = _store(tmp_path / "raw", payload)
    foreign = result_dir(tmp_path / "parsed", RECEIPT, digest).parent / ".tmp-other-run"
    foreign.mkdir(parents=True)
    (foreign / "marker.txt").write_text("keep\n", encoding="utf-8")

    def boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("stopped")

    monkeypatch.setattr("disclosure_agent.parsing.os.rename", boom)
    with pytest.raises(RuntimeError, match="stopped"):
        parse_receipt(RECEIPT, tmp_path / "raw", tmp_path / "parsed")
    assert list((tmp_path / "parsed").rglob("COMPLETE")) == []
    assert (foreign / "marker.txt").read_text(encoding="utf-8") == "keep\n"


def test_declared_encoding_is_not_replaced(tmp_path: Path) -> None:
    xml = _xml("<P>한글</P>").encode("euc-kr")
    xml = xml.replace(b'encoding="utf-8"', b'encoding="euc-kr"')
    _store(tmp_path / "raw", _zip([(f"{RECEIPT}.xml", xml)]))
    parse_receipt(RECEIPT, tmp_path / "raw", tmp_path / "parsed")
    assert _elements(tmp_path / "parsed")[0]["text"] == "한글"
    bad = _xml("<P>한글</P>").encode("euc-kr")
    _store(tmp_path / "raw2", _zip([(f"{RECEIPT}.xml", bad)]))
    result = parse_receipt(RECEIPT, tmp_path / "raw2", tmp_path / "parsed2")
    assert result["status"] == "failed"


def test_unknown_tag_is_not_silent_success(tmp_path: Path) -> None:
    body = "<P>유지</P><NOTE>빠진설명</NOTE>"
    _store(tmp_path / "raw", _zip([(f"{RECEIPT}.xml", _xml(body))]))
    result = parse_receipt(RECEIPT, tmp_path / "raw", tmp_path / "parsed")
    assert result["status"] == "partial"
    assert result["search_eligible"] is False
    texts = [item["text"] for item in _elements(tmp_path / "parsed")]
    assert "유지" in texts
    assert "빠진설명" in texts
    manifest = json.loads(
        next((tmp_path / "parsed").rglob("parse_manifest.json")).read_text(encoding="utf-8")
    )
    member_warnings = manifest["members"][0]["warnings"]
    assert any(item.startswith("unsupported_tag:NOTE") for item in member_warnings)


def test_literal_repair_keeps_meaning_without_double_escape(tmp_path: Path) -> None:
    raw_xml = _xml("<P>변경안 &lt;신설&gt; &amp; 유지</P>\n<P>라벨 <신설> 끝</P>")
    _store(tmp_path / "raw", _zip([(f"{RECEIPT}.xml", raw_xml)]))
    result = parse_receipt(RECEIPT, tmp_path / "raw", tmp_path / "parsed")
    assert result["status"] == "success"
    texts = [item["text"] for item in _elements(tmp_path / "parsed")]
    assert "변경안 <신설> & 유지" in texts
    assert "라벨 <신설> 끝" in texts
    report = json.loads(
        next((tmp_path / "parsed").rglob("preprocess_report.json")).read_text(encoding="utf-8")
    )
    changes = report["members"][0]["changes"]
    assert any(
        item["rule"] == "literal_angle_allowlist" and item["before"] == "<신설>" for item in changes
    )
    assert not any("&amp;lt;" in item["after"] or "&amp;amp;" in item["after"] for item in changes)


def test_corrupt_complete_result_is_not_reused(tmp_path: Path) -> None:
    _store(tmp_path / "raw", _zip([(f"{RECEIPT}.xml", _xml("<P>정상</P>"))]))
    first = parse_receipt(RECEIPT, tmp_path / "raw", tmp_path / "parsed")
    dest = Path(first["dest"])
    (dest / "elements.jsonl").write_text("{not-json\n", encoding="utf-8")
    second = parse_receipt(RECEIPT, tmp_path / "raw", tmp_path / "parsed")
    assert second["reused"] is False
    assert second["status"] == "success"
    assert _elements(tmp_path / "parsed")[0]["text"] == "정상"
    assert list(dest.parent.glob("quarantine/*"))


def test_full_config_hash_mismatch_is_not_reused(tmp_path: Path) -> None:
    _store(tmp_path / "raw", _zip([(f"{RECEIPT}.xml", _xml("<P>정상</P>"))]))
    first = parse_receipt(RECEIPT, tmp_path / "raw", tmp_path / "parsed")
    dest = Path(first["dest"])
    manifest_path = dest / "parse_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config_hash"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ParseInputError, match="full config_hash"):
        parse_receipt(RECEIPT, tmp_path / "raw", tmp_path / "parsed")
    assert (dest / "COMPLETE").is_file()
    assert dest == result_dir(tmp_path / "parsed", RECEIPT, first["source_sha256"])
    assert dest.name == config_hash()[:16]


def test_confirmed_cell_and_caption_literals_stay_success(tmp_path: Path) -> None:
    body = (
        "<P><연결기준></P>"
        "<P><SPAN><별도기준></SPAN></P>"
        "<TABLE><TR><TD><중단영업손실></TD><TD></TD></TR></TABLE>"
        "<P><표1></P><P><표2></P>"
    )
    _store(tmp_path / "raw", _zip([(f"{RECEIPT}.xml", _xml(body))]))
    result = parse_receipt(RECEIPT, tmp_path / "raw", tmp_path / "parsed")
    assert result["status"] == "success"
    assert result["search_eligible"] is True
    elements = _elements(tmp_path / "parsed")
    texts = [item["text"] for item in elements]
    assert "<연결기준>" in texts
    assert "<별도기준>" in texts
    assert "<표1>" in texts
    assert "<표2>" in texts
    table = next(item for item in elements if item["element_type"] == "table")
    assert table["table"]["cells"][0]["text"] == "<중단영업손실>"
    assert table["table"]["cells"][1]["text"] == ""


def test_correction_wrapper_stays_partial(tmp_path: Path) -> None:
    body = "<CORRECTION><TITLE>정 정 신 고 (보고)</TITLE><P>본문</P></CORRECTION>"
    _store(tmp_path / "raw", _zip([(f"{RECEIPT}.xml", _xml(body))]))
    result = parse_receipt(RECEIPT, tmp_path / "raw", tmp_path / "parsed")
    assert result["status"] == "partial"
    assert result["search_eligible"] is False
    elements = _elements(tmp_path / "parsed")
    by_text = {item["text"]: item for item in elements}
    assert by_text["정 정 신 고 (보고)"]["element_type"] == "heading"
    assert by_text["정 정 신 고 (보고)"]["section_path"] == []
    assert by_text["본문"]["section_path"] == ["정 정 신 고 (보고)"]


def test_partial_reuse_keeps_status_and_warnings(tmp_path: Path) -> None:
    body = "<P>유지</P><NOTE>설명</NOTE>"
    _store(tmp_path / "raw", _zip([(f"{RECEIPT}.xml", _xml(body))]))
    first = parse_receipt(RECEIPT, tmp_path / "raw", tmp_path / "parsed")
    second = parse_receipt(RECEIPT, tmp_path / "raw", tmp_path / "parsed")
    assert first["status"] == "partial"
    assert second["reused"] is True
    assert second["status"] == "partial"
    assert second["search_eligible"] is False
    assert second["warnings"] == first["warnings"]
    assert second["warnings"]


# Expectations below were copied from the stored ZIP text, not from parser output.
# Offsets are Unicode code points in the decoded member, before preprocessing.
_RAW = Path("data/raw/dart")


def test_stored_original_fidelity_is_not_a_locator_roundtrip() -> None:
    zip_path = (
        _RAW
        / "20260323000847"
        / "versions"
        / "ff2ee7ca28c18ac6487ea4349cfde9d8dca4d03771c2a8a1bd37c9d81cfd7bb6"
        / "document.zip"
    )
    if not zip_path.is_file():
        pytest.skip("stored original is not in this checkout")
    with zipfile.ZipFile(zip_path) as archive:
        raw = archive.read("20260323000847.xml").decode("utf-8")
    assert raw[83212:83212 + len("<신설>")] == "<신설>"
    assert raw[91720:91720 + len("<연결기준_2025년>")] == "<연결기준_2025년>"
    empty_end = raw.find("</TABLE>", 463568)
    empty = raw[463568:empty_end]
    assert "<TD" in empty
    assert "<IMG" not in empty and "<IMAGE" not in empty
    assert empty.count("<TABLE") == 1
    result = parse_receipt(
        "20260323000847",
        Path("data/raw"),
        Path("data/parsed"),
        source_sha256="ff2ee7ca28c18ac6487ea4349cfde9d8dca4d03771c2a8a1bd37c9d81cfd7bb6",
    )
    elements = _elements(Path(result["dest"]))
    texts = [item["text"] for item in elements]
    cell_texts = [
        cell["text"]
        for item in elements
        if item["element_type"] == "table"
        for cell in item["table"]["cells"]
    ]
    assert "<신설>" in cell_texts
    assert "<연결기준_2025년>" in texts
    assert "<신설>" in raw
    assert "<연결기준_2025년>" in raw
    tables = [
        item for item in _elements(Path(result["dest"])) if item["element_type"] == "table"
    ]
    empty_tables = [
        item
        for item in tables
        if item["table"]["cells"] and all(not cell["text"] for cell in item["table"]["cells"])
    ]
    assert empty_tables
    assert result["status"] == "success"
    others = (
        (
            "20260515002006",
            "1609c4d4db119563e98c507ea939fc91c1e1e0b32f72f3eb0f102479fd7acb8c",
            "20260515002006.xml",
            "<연결기준_2026년 1분기>",
        ),
        (
            "20260814002283",
            "c58fbe472b2cafe15e0865b64774be69a482c4f76e89a5469de3f50ddc70a218",
            "20260814002283.xml",
            "<연결기준_2026년 반기>",
        ),
    )
    for receipt, digest, member, needle in others:
        path = _RAW / receipt / "versions" / digest / "document.zip"
        with zipfile.ZipFile(path) as archive:
            member_text = archive.read(member).decode("utf-8")
        assert needle in member_text
        parsed = parse_receipt(
            receipt, Path("data/raw"), Path("data/parsed"), source_sha256=digest
        )
        parsed_texts = [item["text"] for item in _elements(Path(parsed["dest"]))]
        assert needle in parsed_texts
        assert parsed["status"] == "success"


def test_unsupported_wrapper_heading_does_not_escape(tmp_path: Path) -> None:
    body = """
    <SECTION-1>
      <TITLE>부모 제목</TITLE>
      <NOTE>
        <TITLE>래퍼 내부 제목</TITLE>
        <P>래퍼 문단</P>
        <SECTION-2>
          <TITLE>중첩 제목</TITLE>
          <P>중첩 문단</P>
        </SECTION-2>
      </NOTE>
      <P>래퍼 뒤 형제</P>
    </SECTION-1>
    <P>다음 구역</P>
    """
    _store(tmp_path / "raw", _zip([(f"{RECEIPT}.xml", _xml(body))]))
    result = parse_receipt(RECEIPT, tmp_path / "raw", tmp_path / "parsed")
    assert result["status"] == "partial"
    assert result["search_eligible"] is False
    paths = {
        item["text"]: item["section_path"]
        for item in _elements(tmp_path / "parsed")
        if item["element_type"] in {"heading", "paragraph"}
    }
    assert paths["부모 제목"] == []
    assert paths["래퍼 내부 제목"] == ["부모 제목"]
    assert paths["래퍼 문단"] == ["부모 제목", "래퍼 내부 제목"]
    assert paths["중첩 제목"] == ["부모 제목", "래퍼 내부 제목"]
    assert paths["중첩 문단"] == ["부모 제목", "래퍼 내부 제목", "중첩 제목"]
    assert paths["래퍼 뒤 형제"] == ["부모 제목"]
    assert paths["다음 구역"] == []
    other = "20240101000002"
    _store(
        tmp_path / "raw",
        _zip([(f"{other}.xml", _xml("<P>다른 문서</P>"))]),
        receipt=other,
    )
    second = parse_receipt(other, tmp_path / "raw", tmp_path / "parsed")
    assert second["status"] == "success"
    assert _elements(Path(second["dest"]))[0]["section_path"] == []
    assert _elements(Path(second["dest"]))[0]["text"] == "다른 문서"


def test_crossed_tree_contexts_resolve_their_own_nodes() -> None:
    xml_a = _xml("<P>ALPHA-ONLY</P>")
    xml_b = _xml("<P>BETA-ONLY</P>")
    root_a, ctx_a = _bind_tree(
        xml_a, source_sha256="a" * 64, member_index=0, member_name="same.xml"
    )
    root_b, ctx_b = _bind_tree(
        xml_b, source_sha256="b" * 64, member_index=1, member_name="same.xml"
    )
    paragraph_a = next(node for node in root_a.iter() if _local(node.tag) == "P")
    paragraph_b = next(node for node in root_b.iter() if _local(node.tag) == "P")
    locator_a = _locator_for(paragraph_a, ctx_a)
    locator_b = _locator_for(paragraph_b, ctx_b)
    assert (
        element_text_at(xml_a, {"element_type": "paragraph", "source_locator": locator_a})
        == "ALPHA-ONLY"
    )
    assert (
        element_text_at(xml_b, {"element_type": "paragraph", "source_locator": locator_b})
        == "BETA-ONLY"
    )
    assert locator_a["member_name"] == locator_b["member_name"] == "same.xml"
    assert locator_a["member_index"] == 0
    assert locator_b["member_index"] == 1
    assert locator_a["source_sha256"] != locator_b["source_sha256"]
    with pytest.raises(ParseInputError, match="structural path"):
        _locator_for(paragraph_a, ctx_b)


def test_same_member_name_locators_follow_member_index(tmp_path: Path) -> None:
    xml_zero = _xml("<P>MEMBER-ZERO</P>")
    xml_one = _xml("<P>MEMBER-ONE</P>")
    _store(tmp_path / "raw", _zip([("same.xml", xml_zero), ("same.xml", xml_one)]))
    parse_receipt(RECEIPT, tmp_path / "raw", tmp_path / "parsed")
    paragraphs = [
        item for item in _elements(tmp_path / "parsed") if item["element_type"] == "paragraph"
    ]
    assert [item["text"] for item in paragraphs] == ["MEMBER-ZERO", "MEMBER-ONE"]
    assert paragraphs[0]["source_locator"]["member_index"] == 0
    assert paragraphs[1]["source_locator"]["member_index"] == 1
    assert paragraphs[0]["source_locator"]["member_name"] == "same.xml"
    assert paragraphs[0]["document_id"] != paragraphs[1]["document_id"]
    assert element_text_at(xml_zero, paragraphs[0]) == "MEMBER-ZERO"
    assert element_text_at(xml_one, paragraphs[1]) == "MEMBER-ONE"


def test_unit_mentions_are_not_applied(tmp_path: Path) -> None:
    body = """
    <P>앞 문단 (단위 : 백만원)</P>
    <TABLE>
      <TBODY>
        <TR><TD>(단위: 백만원, %)</TD><TD>100</TD></TR>
        <TR><TD>매출</TD><TD>1</TD></TR>
      </TBODY>
    </TABLE>
    <TABLE>
      <TBODY>
        <TR><TD>(단위: 원)</TD><TD>(단위: 주)</TD></TR>
      </TBODY>
    </TABLE>
    <TABLE>
      <TBODY>
        <TR><TD>수량</TD><TD>3</TD></TR>
      </TBODY>
    </TABLE>
    """
    _store(tmp_path / "raw", _zip([(f"{RECEIPT}.xml", _xml(body))]))
    result = parse_receipt(RECEIPT, tmp_path / "raw", tmp_path / "parsed")
    assert result["status"] == "success"
    elements = _elements(tmp_path / "parsed")
    paragraph = next(item for item in elements if item["element_type"] == "paragraph")
    tables = [item for item in elements if item["element_type"] == "table"]
    assert paragraph["text"] == "앞 문단 (단위 : 백만원)"
    assert paragraph["unit_mentions"][0]["raw_text"] == "(단위 : 백만원)"
    assert paragraph["unit_mentions"][0]["applies_to"] is None
    assert paragraph["unit_mentions"][0]["scope_status"] == "unknown"
    assert paragraph["unit_scope_status"] == "unknown"
    mixed, split, plain = tables
    assert [cell["text"] for cell in mixed["table"]["cells"]] == [
        "(단위: 백만원, %)",
        "100",
        "매출",
        "1",
    ]
    assert mixed["unit_mentions"][0]["raw_text"] == "(단위: 백만원, %)"
    assert mixed["unit_mentions"][0]["scope_status"] == "unresolved"
    assert mixed["unit_mentions"][0]["applies_to"] is None
    assert "multiplier" not in mixed["unit_mentions"][0]
    assert mixed["unit_scope_status"] == "unresolved"
    assert "(단위 : 백만원)" not in [item["raw_text"] for item in mixed["unit_mentions"]]
    assert "unit" not in mixed["table"]["cells"][1]
    assert {item["raw_text"] for item in split["unit_mentions"]} == {"(단위: 원)", "(단위: 주)"}
    assert all(item["applies_to"] is None for item in split["unit_mentions"])
    assert split["unit_scope_status"] == "unresolved"
    assert "unit_mentions" not in plain
    assert "unit_scope_status" not in plain


def test_other_run_temp_is_not_a_completed_result(tmp_path: Path) -> None:
    payload = _zip([(f"{RECEIPT}.xml", _xml("<P>본문</P>"))])
    digest = _store(tmp_path / "raw", payload)
    dest = result_dir(tmp_path / "parsed", RECEIPT, digest)
    foreign = dest.parent / ".tmp-other-run"
    foreign.mkdir(parents=True)
    (foreign / "COMPLETE").write_text("ok\n", encoding="utf-8")
    (foreign / "elements.jsonl").write_text('{"text":"다른 실행"}\n', encoding="utf-8")
    result = parse_receipt(RECEIPT, tmp_path / "raw", tmp_path / "parsed")
    assert result["reused"] is False
    assert result["dest"] == str(dest)
    assert _elements(dest)[0]["text"] == "본문"
    assert (foreign / "COMPLETE").is_file()
    kept = (foreign / "elements.jsonl").read_text(encoding="utf-8")
    assert kept.startswith('{"text":"다른 실행"}')


def test_concurrent_processes_publish_one_and_reuse(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    parsed = tmp_path / "parsed"
    payload = _zip([(f"{RECEIPT}.xml", _xml("<P>경합</P>"))])
    digest = _store(raw, payload)
    dest = result_dir(parsed, RECEIPT, digest)
    foreign = dest.parent / ".tmp-other-run"
    foreign.mkdir(parents=True)
    (foreign / "COMPLETE").write_text("not-this-run\n", encoding="utf-8")
    script = tmp_path / "race_parse.py"
    script.write_text(
        "\n".join(
            [
                "import time",
                "from pathlib import Path",
                "import disclosure_agent.parsing as parsing",
                "original = parsing._parse_zip",
                "def slow(*args, **kwargs):",
                "    time.sleep(0.6)",
                "    return original(*args, **kwargs)",
                "parsing._parse_zip = slow",
                "result = parsing.parse_receipt(",
                f"    {RECEIPT!r}, Path({str(raw)!r}), Path({str(parsed)!r}),",
                "    lock_timeout_seconds=20,",
                ")",
                "print('reused=' + str(result['reused']))",
                "print('status=' + result['status'])",
                "print('dest=' + result['dest'])",
            ]
        ),
        encoding="utf-8",
    )
    procs = [
        subprocess.Popen(
            [sys.executable, str(script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    outputs: list[str] = []
    for proc in procs:
        try:
            stdout, stderr = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise AssertionError("concurrent parse exceeded 30s") from None
        assert proc.returncode == 0, stderr
        outputs.append(stdout)
    reused = [
        line for text in outputs for line in text.splitlines() if line.startswith("reused=")
    ]
    assert sorted(reused) == ["reused=False", "reused=True"]
    assert all("status=success" in text for text in outputs)
    assert {line for text in outputs for line in text.splitlines() if line.startswith("dest=")} == {
        f"dest={dest}"
    }
    assert (dest / "COMPLETE").is_file()
    assert _elements(dest)[0]["text"] == "경합"
    temps = [path for path in dest.parent.iterdir() if path.name.startswith(".tmp-")]
    assert temps == [foreign]
    assert (foreign / "COMPLETE").read_text(encoding="utf-8") == "not-this-run\n"
    assert not (dest.parent / f".{dest.name}.lock").exists()


def test_lock_timeout_is_not_a_parse_failure(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    parsed = tmp_path / "parsed"
    payload = _zip([(f"{RECEIPT}.xml", _xml("<P>대기</P>"))])
    digest = _store(raw, payload)
    dest = result_dir(parsed, RECEIPT, digest)
    lock_path = dest.parent / f".{dest.name}.lock"
    lock_path.parent.mkdir(parents=True)
    holder = tmp_path / "hold_lock.py"
    holder.write_text(
        "\n".join(
            [
                "import os",
                "import time",
                f"fd = os.open({str(lock_path)!r}, os.O_CREAT | os.O_EXCL | os.O_WRONLY)",
                'os.write(fd, b"pid=hold\\n")',
                "time.sleep(2)",
                "os.close(fd)",
                f"os.unlink({str(lock_path)!r})",
            ]
        ),
        encoding="utf-8",
    )
    proc = subprocess.Popen([sys.executable, str(holder)])
    deadline = time.monotonic() + 5
    while not lock_path.exists():
        assert time.monotonic() < deadline
        time.sleep(0.05)
    zip_path = raw / "dart" / RECEIPT / "versions" / digest / "document.zip"
    before = zip_path.read_bytes()
    with pytest.raises(ParseLockTimeout, match="not a parse failure"):
        parse_receipt(RECEIPT, raw, parsed, lock_timeout_seconds=0.3)
    assert not dest.exists()
    assert zip_path.read_bytes() == before
    proc.wait(timeout=10)
    assert proc.returncode == 0
    assert not lock_path.exists()


def test_parse_cli_distinguishes_partial_and_reuses_diagnostics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    body = "<CORRECTION><TITLE>정정</TITLE><P>본문 전체는 출력하지 않는다</P></CORRECTION>"
    _store(tmp_path / "raw", _zip([(f"{RECEIPT}.xml", _xml(body))]))
    code = run_parse_cli(
        [
            "--receipt-no",
            RECEIPT,
            "--raw-dir",
            str(tmp_path / "raw"),
            "--output-dir",
            str(tmp_path / "parsed"),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    lines = captured.out.splitlines()
    assert lines[0] == "status: partial"
    assert lines[1] == "search_eligible: False"
    assert lines[2] == "quality_note: 처리 완료, 검색 입력 제외"
    assert "preprocess_version:" in captured.out
    assert "manifest:" in captured.out
    assert "reused: False" in captured.out
    assert "unsupported_tag:CORRECTION" in captured.out
    assert "본문 전체는 출력하지 않는다" not in captured.out
    again = run_parse_cli(
        [
            "--receipt-no",
            RECEIPT,
            "--raw-dir",
            str(tmp_path / "raw"),
            "--output-dir",
            str(tmp_path / "parsed"),
        ]
    )
    reused = capsys.readouterr().out
    assert again == 0
    assert reused.splitlines()[0] == "status: partial"
    assert "search_eligible: False" in reused
    assert "quality_note: 처리 완료, 검색 입력 제외" in reused
    assert "reused: True" in reused
    assert "manifest:" in reused


def test_parse_cli_shows_nonfatal_warnings_and_truncates(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    notes = "".join(f"<NOTE>n{index}</NOTE>" for index in range(20))
    _store(tmp_path / "raw", _zip([(f"{RECEIPT}.xml", _xml("<P>유지</P>" + notes))]))
    code = run_parse_cli(
        [
            "--receipt-no",
            RECEIPT,
            "--raw-dir",
            str(tmp_path / "raw"),
            "--output-dir",
            str(tmp_path / "parsed"),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "status: partial" in out
    assert "warnings_omitted:" in out
    assert "warnings_detail:" in out
    assert "count=" in out
    literal = "20240101000003"
    _store(
        tmp_path / "raw",
        _zip([(f"{literal}.xml", _xml("<P>라벨 <신설> 끝</P>"))]),
        receipt=literal,
    )
    ok = run_parse_cli(
        [
            "--receipt-no",
            literal,
            "--raw-dir",
            str(tmp_path / "raw"),
            "--output-dir",
            str(tmp_path / "parsed"),
        ]
    )
    success = capsys.readouterr().out
    assert ok == 0
    assert success.splitlines()[0] == "status: success"
    assert "search_eligible: True" in success
    assert "safe_literal_repairs_applied" in success


def test_parse_cli_lock_timeout_exit_is_not_parse_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "raw"
    parsed = tmp_path / "parsed"
    payload = _zip([(f"{RECEIPT}.xml", _xml("<P>대기</P>"))])
    digest = _store(raw, payload)
    dest = result_dir(parsed, RECEIPT, digest)
    lock_path = dest.parent / f".{dest.name}.lock"
    lock_path.parent.mkdir(parents=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        def _short(*args: object, **kwargs: object) -> dict:
            kwargs["lock_timeout_seconds"] = 0.2
            return parse_receipt(*args, **kwargs)  # type: ignore[arg-type, return-value]

        monkeypatch.setattr("disclosure_agent.cli.parse_receipt", _short)
        code = run_parse_cli(
            [
                "--receipt-no",
                RECEIPT,
                "--raw-dir",
                str(raw),
                "--output-dir",
                str(parsed),
            ]
        )
    finally:
        os.close(fd)
        lock_path.unlink(missing_ok=True)
    captured = capsys.readouterr()
    assert code == 3
    assert "not a parse failure" in captured.err
    assert "status:" not in captured.out
    assert not dest.exists()

"""Discovery / corp mapping / list pagination contracts with mocked HTTP."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import httpx
import pytest

from disclosure_agent.discovery import (
    CorpRecord,
    discover_filings,
    refresh_corp_cache,
    resolve_corp,
)
from disclosure_agent.errors import CorpMappingError, RateLimitError


def corp_xml_zip(rows: list[dict[str, str]]) -> bytes:
    body = ['<?xml version="1.0" encoding="UTF-8"?>', "<result>"]
    for row in rows:
        body.append("<list>")
        for key, value in row.items():
            body.append(f"<{key}>{value}</{key}>")
        body.append("</list>")
    body.append("</result>")
    xml = "\n".join(body).encode()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("CORPCODE.xml", xml)
    return buf.getvalue()


def list_payload(
    *,
    status: str = "000",
    message: str = "정상",
    page_no: int = 1,
    total_page: int = 1,
    items: list[dict] | None = None,
) -> dict:
    return {
        "status": status,
        "message": message,
        "page_no": page_no,
        "page_count": 100,
        "total_count": len(items or []),
        "total_page": total_page,
        "list": items or [],
    }


def item(rcept_no: str, report_nm: str = "사업보고서") -> dict:
    return {
        "corp_code": "00126380",
        "corp_name": "삼성전자",
        "stock_code": "005930",
        "report_nm": report_nm,
        "rcept_no": rcept_no,
        "flr_nm": "삼성전자",
        "rcept_dt": "20240331",
        "rm": "",
        "corp_cls": "Y",
    }


@pytest.fixture
def corp(tmp_path: Path) -> tuple[CorpRecord, dict]:
    cache = {
        "schema_version": 1,
        "fetched_at": "2026-09-21T00:00:00+00:00",
        "companies": [
            {
                "corp_code": "00126380",
                "corp_name": "삼성전자",
                "corp_eng_name": "SAMSUNG",
                "stock_code": "005930",
                "modify_date": "20240101",
            }
        ],
    }
    path = tmp_path / "dart" / "corp_codes.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(cache), encoding="utf-8")
    record = CorpRecord(
        corp_code="00126380",
        corp_name="삼성전자",
        corp_eng_name="SAMSUNG",
        stock_code="005930",
        modify_date="20240101",
    )
    return record, cache


def test_stock_code_maps_and_paginated_receipts_flow(tmp_path: Path, corp: tuple) -> None:
    record, cache = corp
    pages = {
        1: list_payload(
            page_no=1,
            total_page=2,
            items=[item("20240331000001"), item("20240331000002")],
        ),
        2: list_payload(
            page_no=2,
            total_page=2,
            items=[item("20240331000002"), item("20240331000003", "분기보고서")],
        ),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        page_no = int(request.url.params.get("page_no", "1"))
        return httpx.Response(200, json=pages[page_no])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    resolved, _ = resolve_corp(tmp_path, stock_code="005930", refresh_if_missing=False)
    assert resolved.corp_code == "00126380"
    manifest = discover_filings(
        "key",
        corp=record,
        bgn_de="20240101",
        end_de="20241231",
        cache_meta=cache,
        client=client,
    )
    assert manifest.list_complete is True
    assert manifest.selected_receipt_nos == [
        "20240331000001",
        "20240331000002",
        "20240331000003",
    ]
    assert manifest.duplicate_receipt_nos == ["20240331000002"]
    assert manifest.items[2]["report_nm"] == "분기보고서"


def test_empty_list_is_complete(tmp_path: Path, corp: tuple) -> None:
    record, cache = corp

    def empty_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=list_payload(status="013", message="없음", items=[]))

    client = httpx.Client(transport=httpx.MockTransport(empty_handler))
    empty = discover_filings(
        "key",
        corp=record,
        bgn_de="20240101",
        end_de="20240102",
        cache_meta=cache,
        client=client,
    )
    assert empty.list_complete is True
    assert empty.selected_receipt_nos == []
    assert empty.items == []


def test_page_failure_is_not_complete(tmp_path: Path, corp: tuple) -> None:
    record, cache = corp

    def rate_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=list_payload(status="020", message="제한"))

    client = httpx.Client(transport=httpx.MockTransport(rate_handler))
    with pytest.raises(RateLimitError):
        discover_filings(
            "key",
            corp=record,
            bgn_de="20240101",
            end_de="20240102",
            cache_meta=cache,
            client=client,
        )


def test_max_filings_defers_without_marking_failed(tmp_path: Path, corp: tuple) -> None:
    record, cache = corp

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=list_payload(
                items=[
                    item("20240331000001"),
                    item("20240331000002"),
                    item("20240331000003"),
                ]
            ),
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    manifest = discover_filings(
        "key",
        corp=record,
        bgn_de="20240101",
        end_de="20241231",
        cache_meta=cache,
        max_filings=1,
        client=client,
    )
    assert manifest.list_complete is True
    assert manifest.selected_receipt_nos == ["20240331000001"]
    assert manifest.deferred_receipt_nos == ["20240331000002", "20240331000003"]


def test_ambiguous_stock_code_refuses_guess(tmp_path: Path) -> None:
    cache = {
        "schema_version": 1,
        "fetched_at": "2026-09-21T00:00:00+00:00",
        "companies": [
            {
                "corp_code": "00000001",
                "corp_name": "A",
                "corp_eng_name": "",
                "stock_code": "123456",
                "modify_date": "",
            },
            {
                "corp_code": "00000002",
                "corp_name": "B",
                "corp_eng_name": "",
                "stock_code": "123456",
                "modify_date": "",
            },
        ],
    }
    path = tmp_path / "dart" / "corp_codes.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(cache), encoding="utf-8")
    with pytest.raises(CorpMappingError, match="ambiguous"):
        resolve_corp(tmp_path, stock_code="123456", refresh_if_missing=False)


def test_refresh_corp_cache_records_fetched_at(tmp_path: Path) -> None:
    zip_bytes = corp_xml_zip(
        [
            {
                "corp_code": "00126380",
                "corp_name": "삼성전자",
                "corp_eng_name": "SAMSUNG",
                "stock_code": "005930",
                "modify_date": "20240101",
            }
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert "corpCode.xml" in str(request.url)
        return httpx.Response(200, content=zip_bytes)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    cache = refresh_corp_cache(tmp_path, "key", client=client, force=True)
    assert cache["count"] == 1
    assert cache["fetched_at"]
    assert (tmp_path / "dart" / "corp_codes.json").is_file()

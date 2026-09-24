"""Corp-code mapping cache and disclosure-list discovery."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from disclosure_agent.dart import (
    DisclosureListItem,
    extract_corp_code_xml_from_zip,
    fetch_corp_code_zip,
    fetch_disclosure_list_page,
    parse_corp_code_xml,
)
from disclosure_agent.errors import CorpMappingError, UsageError

CACHE_SCHEMA = 1


@dataclass(frozen=True)
class CorpRecord:
    corp_code: str
    corp_name: str
    corp_eng_name: str
    stock_code: str
    modify_date: str


@dataclass
class DiscoveryManifest:
    schema_version: int
    run_id: str
    source: str
    started_at: str
    finished_at: str | None
    list_complete: bool
    filters: dict[str, Any]
    corp: dict[str, Any]
    items: list[dict[str, Any]] = field(default_factory=list)
    selected_receipt_nos: list[str] = field(default_factory=list)
    deferred_receipt_nos: list[str] = field(default_factory=list)
    duplicate_receipt_nos: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def corp_cache_path(cache_dir: Path) -> Path:
    return cache_dir / "dart" / "corp_codes.json"


def load_corp_cache(cache_dir: Path) -> dict[str, Any] | None:
    path = corp_cache_path(cache_dir)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or data.get("schema_version") != CACHE_SCHEMA:
        return None
    return data


def refresh_corp_cache(
    cache_dir: Path,
    api_key: str,
    *,
    client: httpx.Client | None = None,
    force: bool = True,
) -> dict[str, Any]:
    """Download and cache official corpCode mapping. Records fetched_at."""
    existing = load_corp_cache(cache_dir)
    if existing and not force:
        return existing
    zip_bytes = fetch_corp_code_zip(api_key, client=client)
    xml_bytes = extract_corp_code_xml_from_zip(zip_bytes)
    rows = parse_corp_code_xml(xml_bytes)
    fetched_at = datetime.now(UTC).isoformat()
    payload = {
        "schema_version": CACHE_SCHEMA,
        "source_endpoint": "https://opendart.fss.or.kr/api/corpCode.xml",
        "fetched_at": fetched_at,
        "count": len(rows),
        "companies": rows,
    }
    path = corp_cache_path(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return payload


def resolve_corp(
    cache_dir: Path,
    *,
    stock_code: str | None = None,
    corp_code: str | None = None,
    api_key: str | None = None,
    client: httpx.Client | None = None,
    refresh_if_missing: bool = True,
) -> tuple[CorpRecord, dict[str, Any]]:
    """Resolve exactly one company. Never picks arbitrarily among multiples."""
    if bool(stock_code) == bool(corp_code):
        raise UsageError("provide exactly one of --stock-code or --corp-code")

    cache = load_corp_cache(cache_dir)
    if cache is None:
        if not refresh_if_missing:
            raise CorpMappingError("corp code cache is missing; run dart-corp-codes --refresh")
        if not api_key:
            raise UsageError("OPENDART_API_KEY is required to download corp code mapping")
        cache = refresh_corp_cache(cache_dir, api_key, client=client, force=True)

    companies = cache.get("companies") or []
    if not isinstance(companies, list):
        raise CorpMappingError("corp code cache is invalid")

    if corp_code:
        matches = [
            c
            for c in companies
            if isinstance(c, dict) and str(c.get("corp_code") or "").strip() == corp_code
        ]
        label = f"corp_code={corp_code}"
    else:
        assert stock_code is not None
        stock = stock_code.strip()
        matches = [
            c
            for c in companies
            if isinstance(c, dict) and str(c.get("stock_code") or "").strip() == stock
        ]
        label = f"stock_code={stock}"

    if not matches:
        raise CorpMappingError(f"no company matched {label}")
    if len(matches) > 1:
        raise CorpMappingError(
            f"ambiguous mapping for {label}: {len(matches)} candidates (refusing to guess)"
        )
    row = matches[0]
    record = CorpRecord(
        corp_code=str(row.get("corp_code") or "").strip(),
        corp_name=str(row.get("corp_name") or "").strip(),
        corp_eng_name=str(row.get("corp_eng_name") or "").strip(),
        stock_code=str(row.get("stock_code") or "").strip(),
        modify_date=str(row.get("modify_date") or "").strip(),
    )
    if not record.corp_code:
        raise CorpMappingError(f"matched row for {label} is missing corp_code")
    return record, cache


def discover_filings(
    api_key: str,
    *,
    corp: CorpRecord,
    bgn_de: str,
    end_de: str,
    cache_meta: dict[str, Any],
    max_filings: int | None = None,
    page_count: int = 100,
    client: httpx.Client | None = None,
    run_id: str | None = None,
) -> DiscoveryManifest:
    """Paginate disclosure list. Empty list ≠ page failure ≠ truncated by max_filings."""
    rid = run_id or uuid4().hex
    started = datetime.now(UTC).isoformat()
    filters = {
        "corp_code": corp.corp_code,
        "bgn_de": bgn_de,
        "end_de": end_de,
        "last_reprt_at": "N",
        "pblntf_ty": "A",
        "page_count": page_count,
        "max_filings": max_filings,
    }
    manifest = DiscoveryManifest(
        schema_version=1,
        run_id=rid,
        source="dart",
        started_at=started,
        finished_at=None,
        list_complete=False,
        filters=filters,
        corp={
            "corp_code": corp.corp_code,
            "corp_name": corp.corp_name,
            "stock_code": corp.stock_code,
            "mapping_fetched_at": cache_meta.get("fetched_at"),
        },
    )

    seen: set[str] = set()
    ordered: list[DisclosureListItem] = []
    duplicates: list[str] = []
    page_no = 1
    total_page = 1

    try:
        while page_no <= total_page:
            page = fetch_disclosure_list_page(
                api_key,
                corp_code=corp.corp_code,
                bgn_de=bgn_de,
                end_de=end_de,
                page_no=page_no,
                page_count=page_count,
                last_reprt_at="N",
                pblntf_ty="A",
                client=client,
            )
            total_page = max(page.total_page, 1)
            for item in page.items:
                if item.rcept_no in seen:
                    duplicates.append(item.rcept_no)
                    continue
                seen.add(item.rcept_no)
                ordered.append(item)
            if page.status == "013" and page_no == 1:
                break
            page_no += 1
        manifest.list_complete = True
    except Exception as exc:
        manifest.notes.append(f"list_incomplete: {type(exc).__name__}: {exc}")
        manifest.list_complete = False
        manifest.finished_at = datetime.now(UTC).isoformat()
        manifest.duplicate_receipt_nos = duplicates
        manifest.items = [_item_dict(item) for item in ordered]
        raise

    manifest.duplicate_receipt_nos = duplicates
    manifest.items = [_item_dict(item) for item in ordered]
    receipt_nos = [item.rcept_no for item in ordered]
    if max_filings is not None and max_filings >= 0:
        manifest.selected_receipt_nos = receipt_nos[:max_filings]
        manifest.deferred_receipt_nos = receipt_nos[max_filings:]
        if manifest.deferred_receipt_nos:
            manifest.notes.append(
                f"deferred_by_max_filings:{len(manifest.deferred_receipt_nos)}"
            )
    else:
        manifest.selected_receipt_nos = receipt_nos
        manifest.deferred_receipt_nos = []
    manifest.finished_at = datetime.now(UTC).isoformat()
    return manifest


def write_manifest(output_dir: Path, manifest: DiscoveryManifest) -> Path:
    runs = output_dir / "dart" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    path = runs / f"{manifest.run_id}-manifest.json"
    path.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def validate_ymd(value: str, *, name: str) -> str:
    text = value.strip()
    if len(text) != 8 or not text.isdigit():
        raise UsageError(f"{name} must be YYYYMMDD")
    try:
        datetime.strptime(text, "%Y%m%d")
    except ValueError as exc:
        raise UsageError(f"{name} must be a valid calendar date YYYYMMDD") from exc
    return text


def _item_dict(item: DisclosureListItem) -> dict[str, Any]:
    return {
        "corp_code": item.corp_code,
        "corp_name": item.corp_name,
        "stock_code": item.stock_code,
        "report_nm": item.report_nm,
        "rcept_no": item.rcept_no,
        "flr_nm": item.flr_nm,
        "rcept_dt": item.rcept_dt,
        "rm": item.rm,
        "corp_cls": item.corp_cls,
    }

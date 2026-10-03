"""Which stored filings may enter the product Dense-D2a index.

Downloaded ZIP, a completed parse, and a retrieval row are different states.
A filing is index-eligible only when the current parser contract says
``overall_status == success`` and it has a periodic report member. Audit
reports stay out, matching the existing single-filing index scope. A partial
correction is not substituted for the original.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from disclosure_agent.agent.catalog import DenseD2aBinding
from disclosure_agent.agent.errors import RetrievalFailure
from disclosure_agent.agent.temporal import FilingPeriod, filing_period_from_listing, period_label
from disclosure_agent.dense import DENSE_MODEL_ID
from disclosure_agent.parsing import (
    PARSER_VERSION,
    PREPROCESS_VERSION,
    SEARCH_ELIGIBLE_STATUSES,
    config_hash,
    result_dir,
)
from disclosure_agent.retrieval_unit import D2A_VERSION

PERIODIC_DOCUMENT_NAMES = frozenset({"사업보고서", "반기보고서", "분기보고서"})
AUDIT_DOCUMENT_NAMES = frozenset({"감사보고서", "연결감사보고서"})
PRODUCT_INDEX_RELATIVE = Path("data/product/indexes/isu_petasys_periodic_d2a")
CORPUS_AS_OF_DEFINITION = (
    "corpus_as_of는 이번 색인에 넣은 공시의 current.json last_success_verified_at 중 "
    "가장 늦은 시각을 Asia/Seoul로 바꾼 값이다. request_time도 아니고, "
    "가장 최근 접수일도 아니다. 이 시각 뒤에 DART에 올라온 공시는 이 색인에 없다."
)
INCLUSION_RULE = (
    "current SHA의 ZIP이 있고, 현재 parser/preprocess/config의 COMPLETE parse가 "
    "overall_status=success이며, 멤버 document_name이 사업보고서·반기보고서·분기보고서인 "
    "경우만 색인한다. 감사보고서와 연결감사보고서는 제외한다. "
    "overall_status가 success가 아니면 접수번호 전체를 제외하고 원본 ZIP은 지우지 않는다."
)


@dataclass(frozen=True)
class IndexedMember:
    corp_code: str
    company_name: str
    receipt_no: str
    report_name: str
    report_type: str
    receipt_date: str
    period_end_year: int | None
    period_end_month: int | None
    remark: str
    source_sha256: str
    document_id: str
    document_name: str
    member_index: int
    member_name: str
    elements_path: Path
    manifest_path: Path
    last_success_verified_at: str


@dataclass(frozen=True)
class FilingInventoryRow:
    receipt_no: str
    report_name: str
    report_type: str
    receipt_date: str
    period_label: str
    remark: str
    corp_code: str
    company_name: str
    source_sha256: str
    size_bytes: int | None
    last_result: str
    last_success_verified_at: str
    downloaded: bool
    parse_status: str
    complete: bool
    element_count: int | None
    index_eligible: bool
    index_reason: str
    periodic_documents: str
    excluded_members: str


@dataclass(frozen=True)
class CorpusPlan:
    rows: tuple[FilingInventoryRow, ...]
    members: tuple[IndexedMember, ...]
    corpus_as_of: datetime | None


@dataclass(frozen=True)
class ProductCorpus:
    binding: DenseD2aBinding
    filings: tuple[FilingPeriod, ...]
    corpus_as_of: datetime
    elements_paths: tuple[Path, ...]
    manifest: dict[str, object]


def product_index_dir(root: Path) -> Path:
    return root / PRODUCT_INDEX_RELATIVE


def plan_corpus(root: Path) -> CorpusPlan:
    """Read stored filings. Does not chunk, embed, or call DART."""
    raw_root = root / "data" / "raw" / "dart"
    parsed_root = root / "data" / "parsed"
    if not raw_root.is_dir():
        return CorpusPlan(rows=(), members=(), corpus_as_of=None)
    rows: list[FilingInventoryRow] = []
    members: list[IndexedMember] = []
    for dest in sorted(path for path in raw_root.iterdir() if path.is_dir()):
        if not re.fullmatch(r"\d{14}", dest.name):
            continue
        row, filing_members = _inspect_receipt(dest, parsed_root)
        rows.append(row)
        members.extend(filing_members)
    verified = [
        _parse_verified_at(item.last_success_verified_at)
        for item in members
        if item.last_success_verified_at
    ]
    corpus_as_of = max(verified) if verified else None
    return CorpusPlan(rows=tuple(rows), members=tuple(members), corpus_as_of=corpus_as_of)


def load_product_corpus(root: Path) -> ProductCorpus:
    """Load the built multi-filing index. A missing build is a retrieval failure."""
    index_dir = product_index_dir(root)
    manifest_path = index_dir / "corpus.json"
    if not manifest_path.is_file():
        raise RetrievalFailure(
            "product periodic index is missing: "
            f"{manifest_path}. Run python -m disclosure_agent.product_index"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("representation_version") != D2A_VERSION:
        raise RetrievalFailure("product index is not Dense-D2a")
    units_path = index_dir / "retrieval_units.jsonl"
    embeddings_path = index_dir / "embeddings.f32"
    members = manifest.get("members")
    if not isinstance(members, list) or not members:
        raise RetrievalFailure("product index manifest has no members")
    filings: list[FilingPeriod] = []
    elements: list[Path] = []
    seen: set[str] = set()
    for member in members:
        if not isinstance(member, dict):
            raise RetrievalFailure("product index member is not an object")
        receipt_no = str(member["receipt_no"])
        element_path = root / str(member["elements_path"])
        elements.append(element_path)
        if receipt_no in seen:
            continue
        seen.add(receipt_no)
        filings.append(_filing_from_member(member))
    first = members[0]
    if not isinstance(first, dict):
        raise RetrievalFailure("product index member is not an object")
    binding = DenseD2aBinding(
        corp_code=str(manifest["corp_code"]),
        company_name=str(manifest["company_name"]),
        units_path=units_path,
        embeddings_path=embeddings_path,
        elements_path=elements[0] if elements else None,
        receipt_no=str(first["receipt_no"]),
        document_name=str(first.get("document_name") or ""),
        representation_version=D2A_VERSION,
        embedding_model=str(manifest.get("embedding_model") or DENSE_MODEL_ID),
        elements_paths=tuple(elements),
    )
    as_of_raw = str(manifest.get("corpus_as_of") or "")
    if not as_of_raw:
        raise RetrievalFailure("product index manifest has no corpus_as_of")
    corpus_as_of = datetime.fromisoformat(as_of_raw)
    if corpus_as_of.tzinfo is None:
        raise RetrievalFailure("product index corpus_as_of is naive")
    return ProductCorpus(
        binding=binding,
        filings=tuple(filings),
        corpus_as_of=corpus_as_of,
        elements_paths=tuple(elements),
        manifest=manifest,
    )


def _inspect_receipt(
    dest: Path,
    parsed_root: Path,
) -> tuple[FilingInventoryRow, list[IndexedMember]]:
    receipt_no = dest.name
    current_path = dest / "current.json"
    if not current_path.is_file():
        return _row(
            receipt_no,
            downloaded=False,
            parse_status="missing",
            complete=False,
            index_eligible=False,
            index_reason="current.json is missing",
        ), []
    current = json.loads(current_path.read_text(encoding="utf-8"))
    sha = str(current.get("current_sha256") or "")
    version_path = dest / "versions" / sha / "version.json"
    zip_path = dest / "versions" / sha / "document.zip"
    listing: dict[str, object] = {}
    if version_path.is_file():
        meta = json.loads(version_path.read_text(encoding="utf-8"))
        raw_listing = meta.get("listing")
        if isinstance(raw_listing, dict):
            listing = raw_listing
    if listing:
        period = filing_period_from_listing(listing)
    else:
        period = FilingPeriod(
            receipt_no=receipt_no,
            report_name="",
            report_type="",
            receipt_date="",
            period_end_year=None,
            period_end_month=None,
            remark="",
        )
    downloaded = zip_path.is_file() and bool(sha)
    base = dict(
        receipt_no=receipt_no,
        report_name=period.report_name,
        report_type=period.report_type,
        receipt_date=period.receipt_date,
        period_label=period_label(period),
        remark=period.remark,
        corp_code=str(listing.get("corp_code") or ""),
        company_name=str(listing.get("corp_name") or ""),
        source_sha256=sha,
        size_bytes=current.get("current_size_bytes"),
        last_result=str(current.get("last_result") or ""),
        last_success_verified_at=str(current.get("last_success_verified_at") or ""),
        downloaded=downloaded,
    )
    if not downloaded:
        return _row(
            **base,
            parse_status="missing",
            complete=False,
            element_count=None,
            index_eligible=False,
            index_reason="current ZIP is missing",
        ), []
    manifest_dir = result_dir(parsed_root, receipt_no, sha)
    manifest_path = manifest_dir / "parse_manifest.json"
    complete = (manifest_dir / "COMPLETE").is_file()
    if not manifest_path.is_file() or not complete:
        return _row(
            **base,
            parse_status="missing" if not manifest_path.is_file() else "unfinished",
            complete=complete,
            element_count=None,
            index_eligible=False,
            index_reason="current parser COMPLETE result is missing",
        ), []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = str(manifest.get("overall_status") or "")
    element_count = manifest.get("element_count")
    count = int(element_count) if isinstance(element_count, int) else None
    contract_reason = _contract_reason(manifest)
    periodic, excluded_names = _split_members(manifest)
    if contract_reason:
        return _row(
            **base,
            parse_status=status or "unknown",
            complete=True,
            element_count=count,
            index_eligible=False,
            index_reason=contract_reason,
            periodic_documents=",".join(item["document_name"] for item in periodic),
            excluded_members=",".join(excluded_names),
        ), []
    if not periodic:
        return _row(
            **base,
            parse_status=status,
            complete=True,
            element_count=count,
            index_eligible=False,
            index_reason="no periodic report member",
            excluded_members=",".join(excluded_names),
        ), []
    elements_path = manifest_dir / "elements.jsonl"
    indexed: list[IndexedMember] = []
    for member in periodic:
        indexed.append(
            IndexedMember(
                corp_code=str(listing.get("corp_code") or ""),
                company_name=str(listing.get("corp_name") or ""),
                receipt_no=receipt_no,
                report_name=period.report_name,
                report_type=period.report_type,
                receipt_date=period.receipt_date,
                period_end_year=period.period_end_year,
                period_end_month=period.period_end_month,
                remark=period.remark,
                source_sha256=sha,
                document_id=str(member.get("document_id") or ""),
                document_name=str(member.get("document_name") or ""),
                member_index=int(member.get("member_index") or 0),
                member_name=str(member.get("member_name") or ""),
                elements_path=elements_path,
                manifest_path=manifest_path,
                last_success_verified_at=str(current.get("last_success_verified_at") or ""),
            )
        )
    return _row(
        **base,
        parse_status=status,
        complete=True,
        element_count=count,
        index_eligible=True,
        index_reason="periodic member of a success parse",
        periodic_documents=",".join(item.document_name for item in indexed),
        excluded_members=",".join(excluded_names),
    ), indexed


def _contract_reason(manifest: dict[str, object]) -> str:
    status = str(manifest.get("overall_status") or "")
    if status not in SEARCH_ELIGIBLE_STATUSES:
        warnings = _warning_text(manifest)
        detail = f"overall_status={status or 'missing'}"
        if warnings:
            detail = f"{detail}; {warnings}"
        return detail
    if str(manifest.get("parser_version") or "") != PARSER_VERSION:
        return f"parser_version={manifest.get('parser_version')}"
    if str(manifest.get("preprocess_version") or "") != PREPROCESS_VERSION:
        return f"preprocess_version={manifest.get('preprocess_version')}"
    if str(manifest.get("config_hash") or "") != config_hash():
        return "config_hash does not match the current parser"
    return ""


def _warning_text(manifest: dict[str, object]) -> str:
    found: list[str] = []
    for item in manifest.get("warnings") or []:
        if isinstance(item, str) and item not in found:
            found.append(item)
    members = manifest.get("members")
    if isinstance(members, list):
        for member in members:
            if not isinstance(member, dict):
                continue
            for item in member.get("warnings") or []:
                if isinstance(item, str) and item not in found:
                    found.append(item)
    return ",".join(found[:8])


def _split_members(
    manifest: dict[str, object],
) -> tuple[list[dict[str, object]], list[str]]:
    periodic: list[dict[str, object]] = []
    excluded: list[str] = []
    members = manifest.get("members")
    if not isinstance(members, list):
        return periodic, excluded
    for member in members:
        if not isinstance(member, dict):
            continue
        name = str(member.get("document_name") or "")
        status = str(member.get("status") or "")
        if name in PERIODIC_DOCUMENT_NAMES and status == "parsed":
            periodic.append(member)
            continue
        if name in AUDIT_DOCUMENT_NAMES:
            excluded.append(name)
            continue
        if name:
            excluded.append(f"{name}:{status or 'unknown'}")
    return periodic, excluded


def _row(
    receipt_no: str,
    *,
    report_name: str = "",
    report_type: str = "",
    receipt_date: str = "",
    period_label: str = "",
    remark: str = "",
    corp_code: str = "",
    company_name: str = "",
    source_sha256: str = "",
    size_bytes: object = None,
    last_result: str = "",
    last_success_verified_at: str = "",
    downloaded: bool,
    parse_status: str,
    complete: bool,
    element_count: int | None = None,
    index_eligible: bool,
    index_reason: str,
    periodic_documents: str = "",
    excluded_members: str = "",
) -> FilingInventoryRow:
    size = size_bytes if isinstance(size_bytes, int) else None
    return FilingInventoryRow(
        receipt_no=receipt_no,
        report_name=report_name,
        report_type=report_type,
        receipt_date=receipt_date,
        period_label=period_label,
        remark=remark,
        corp_code=corp_code,
        company_name=company_name,
        source_sha256=source_sha256,
        size_bytes=size,
        last_result=last_result,
        last_success_verified_at=last_success_verified_at,
        downloaded=downloaded,
        parse_status=parse_status,
        complete=complete,
        element_count=element_count,
        index_eligible=index_eligible,
        index_reason=index_reason,
        periodic_documents=periodic_documents,
        excluded_members=excluded_members,
    )


def _parse_verified_at(value: str) -> datetime:
    moment = datetime.fromisoformat(value)
    if moment.tzinfo is None:
        raise ValueError("last_success_verified_at must be timezone-aware")
    return moment


def _filing_from_member(member: dict[str, object]) -> FilingPeriod:
    year = member.get("period_end_year")
    month = member.get("period_end_month")
    return FilingPeriod(
        receipt_no=str(member["receipt_no"]),
        report_name=str(member.get("report_name") or ""),
        report_type=str(member.get("report_type") or ""),
        receipt_date=str(member.get("receipt_date") or ""),
        period_end_year=int(year) if isinstance(year, int) else None,
        period_end_month=int(month) if isinstance(month, int) else None,
        remark=str(member.get("remark") or ""),
        source_sha256=str(member.get("source_sha256") or ""),
        corp_code=str(member.get("corp_code") or ""),
        document_name=str(member.get("document_name") or ""),
    )

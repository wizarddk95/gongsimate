"""Stored filing-list metadata.

The source of truth is the DART listing copied onto the raw version
(``report_nm``, ``rcept_dt``, ``rcept_no``, ``rm``). A parenthetical
``(YYYY.MM)`` in ``report_nm`` is that listing's period label. The receipt
date is not turned into a fiscal year.

Question wording is not interpreted here. Which of these filings a search
may use is decided by validated ``search_disclosures`` arguments.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from disclosure_agent.agent.clock import SEOUL

_PERIOD_RE = re.compile(r"\((\d{4})\.(\d{2})\)\s*$")


@dataclass(frozen=True)
class FilingPeriod:
    """One stored listing. Missing period labels stay empty."""

    receipt_no: str
    report_name: str
    report_type: str
    receipt_date: str
    period_end_year: int | None
    period_end_month: int | None
    remark: str
    source_sha256: str = ""
    corp_code: str = ""
    document_name: str = ""


def filing_period_from_listing(listing: Mapping[str, object]) -> FilingPeriod:
    """Copy listing fields. Parse a period only when ``report_nm`` has one."""
    report_name = str(listing.get("report_nm") or "")
    match = _PERIOD_RE.search(report_name.strip())
    year = int(match.group(1)) if match else None
    month = int(match.group(2)) if match else None
    report_type = _PERIOD_RE.sub("", report_name).strip() or report_name
    return FilingPeriod(
        receipt_no=str(listing.get("rcept_no") or ""),
        report_name=report_name,
        report_type=report_type,
        receipt_date=str(listing.get("rcept_dt") or ""),
        period_end_year=year,
        period_end_month=month,
        remark=str(listing.get("rm") or ""),
        source_sha256=str(listing.get("source_sha256") or ""),
        corp_code=str(listing.get("corp_code") or ""),
        document_name=str(listing.get("document_name") or ""),
    )


def period_label(period: FilingPeriod) -> str:
    """Listing period as ``YYYY.MM``. Empty when the listing has no label."""
    if period.period_end_year is None or period.period_end_month is None:
        return ""
    return f"{period.period_end_year}.{period.period_end_month:02d}"


def load_indexed_filing(root: Path) -> FilingPeriod:
    """Read the single-filing Dense-D2a snapshot listing. Does not invent a year."""
    snapshot_path = root / "evaluation" / "datasets" / "isu_petasys_v0_3" / "snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    receipt_no = str(snapshot["receipt_no"])
    version_path = (
        root
        / "data"
        / "raw"
        / "dart"
        / receipt_no
        / "versions"
        / str(snapshot["source_sha256"])
        / "version.json"
    )
    if not version_path.is_file():
        raise ValueError(f"indexed filing listing is missing: {version_path}")
    meta = json.loads(version_path.read_text(encoding="utf-8"))
    listing = meta.get("listing")
    if not isinstance(listing, dict):
        raise ValueError("indexed filing version has no listing object")
    period = filing_period_from_listing(listing)
    return FilingPeriod(
        receipt_no=period.receipt_no,
        report_name=period.report_name,
        report_type=period.report_type,
        receipt_date=period.receipt_date,
        period_end_year=period.period_end_year,
        period_end_month=period.period_end_month,
        remark=period.remark,
        source_sha256=str(snapshot["source_sha256"]),
        corp_code=str(snapshot.get("corp_code") or ""),
        document_name=str(snapshot.get("index_scope", {}).get("document_name_observed") or ""),
    )


def receipt_on_or_before(period: FilingPeriod, moment: datetime | None) -> bool:
    """Keep a filing whose receipt date is on or before ``moment``.

    A missing receipt date is kept. Dropping it would invent an exclusion.
    """
    if moment is None:
        return True
    raw = period.receipt_date
    if len(raw) != 8 or not raw.isdigit():
        return True
    filed = date(int(raw[0:4]), int(raw[4:6]), int(raw[6:8]))
    return filed <= moment.astimezone(SEOUL).date()


def filing_catalog_row(period: FilingPeriod) -> dict[str, str | None]:
    """Compact listing metadata for a tool planner. No body text."""
    return {
        "receipt_no": period.receipt_no,
        "report_name": period.report_name,
        "report_type": period.report_type,
        "filing_period": period_label(period) or None,
        "receipt_date": period.receipt_date,
        "source_sha256": period.source_sha256 or None,
    }

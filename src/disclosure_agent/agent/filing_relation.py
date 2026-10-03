"""Links between an original filing and a later correction filing.

The raw ZIP is never deleted. Parsing ``<CORRECTION>`` does not by itself
prove which receipt this document replaces. A link is ``confirmed`` only when
the correction text names the original receipt. A unique match on the same
company, the document's first-submission date, and the named report is
``inferred``. Anything else stays ``unverified`` and is not dropped from search.
"""

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from disclosure_agent.agent.temporal import FilingPeriod, receipt_on_or_before

RelationConfidence = Literal["confirmed", "inferred", "unverified"]

_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")
_DOC_RE = re.compile(r"정정대상 공시서류\s*:\s*([^\s]+)")
_DATE_RE = re.compile(r"최초제출일\s*:\s*(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일")
_RECEIPT_RE = re.compile(r"(?<!\d)(20\d{12})(?!\d)")


@dataclass(frozen=True, slots=True)
class CorrectionPointer:
    """Facts copied out of the correction notice. Not a resolved link."""

    document_name: str | None
    first_submitted_on: str | None
    named_receipt_nos: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FilingRelation:
    correction_receipt_no: str
    original_receipt_no: str | None
    confidence: RelationConfidence
    evidence: tuple[str, ...]
    corp_code: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "correction_receipt_no": self.correction_receipt_no,
            "original_receipt_no": self.original_receipt_no,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "corp_code": self.corp_code,
        }


def correction_pointer(xml_text: str) -> CorrectionPointer | None:
    """Read the correction notice. Returns None when the wrapper is absent."""
    start = xml_text.find("<CORRECTION")
    if start < 0:
        return None
    end = xml_text.find("</CORRECTION>", start)
    block = xml_text[start:] if end < 0 else xml_text[start:end]
    plain = _SPACE_RE.sub(" ", _TAG_RE.sub(" ", block))
    document = _DOC_RE.search(plain)
    dated = _DATE_RE.search(plain)
    first = None
    if dated:
        year, month, day = (int(part) for part in dated.groups())
        first = f"{year:04d}{month:02d}{day:02d}"
    named = tuple(dict.fromkeys(_RECEIPT_RE.findall(block)))
    return CorrectionPointer(
        document_name=None if document is None else document.group(1),
        first_submitted_on=first,
        named_receipt_nos=named,
    )


def relate_filings(
    filings: tuple[FilingPeriod, ...] | list[FilingPeriod],
    pointers: dict[str, CorrectionPointer],
) -> tuple[FilingRelation, ...]:
    """Match each pointer to at most one stored original.

    Title year and report type alone never produce ``inferred``. The
    first-submission date has to agree with the original receipt date, and
    the named document has to agree with the original report name.
    """
    relations: list[FilingRelation] = []
    for receipt_no, pointer in pointers.items():
        correction = next((item for item in filings if item.receipt_no == receipt_no), None)
        corp_code = "" if correction is None else correction.corp_code
        named = [
            item.receipt_no
            for item in filings
            if item.receipt_no in pointer.named_receipt_nos and item.receipt_no != receipt_no
        ]
        if len(named) == 1:
            relations.append(
                FilingRelation(
                    correction_receipt_no=receipt_no,
                    original_receipt_no=named[0],
                    confidence="confirmed",
                    evidence=("named_receipt_no",),
                    corp_code=corp_code,
                )
            )
            continue
        candidates = [
            item
            for item in filings
            if item.receipt_no != receipt_no and _same_original(item, correction, pointer)
        ]
        if len(candidates) == 1 and pointer.first_submitted_on and pointer.document_name:
            relations.append(
                FilingRelation(
                    correction_receipt_no=receipt_no,
                    original_receipt_no=candidates[0].receipt_no,
                    confidence="inferred",
                    evidence=("first_submission_date", "document_name", "corp_code"),
                    corp_code=corp_code,
                )
            )
            continue
        relations.append(
            FilingRelation(
                correction_receipt_no=receipt_no,
                original_receipt_no=None,
                confidence="unverified",
                evidence=("insufficient_link",),
                corp_code=corp_code,
            )
        )
    return tuple(relations)


def suppressed_originals(
    filings: tuple[FilingPeriod, ...] | list[FilingPeriod],
    relations: tuple[FilingRelation, ...] | list[FilingRelation],
    *,
    as_of: datetime | None,
) -> frozenset[str]:
    """Originals replaced by a correction that is already in this pool.

    ``unverified`` links do not suppress anything. A correction filed after
    ``as_of`` does not suppress the original that was current at that time.
    """
    present = {item.receipt_no: item for item in filings}
    dropped: set[str] = set()
    for relation in relations:
        if relation.confidence not in {"confirmed", "inferred"}:
            continue
        original = relation.original_receipt_no
        if not original or original not in present:
            continue
        correction = present.get(relation.correction_receipt_no)
        if correction is None:
            continue
        if as_of is not None and not receipt_on_or_before(correction, as_of):
            continue
        dropped.add(original)
    return frozenset(dropped)


def load_relations(path: Path) -> tuple[FilingRelation, ...]:
    if not path.is_file():
        return ()
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("relations") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return ()
    found: list[FilingRelation] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        confidence = str(row.get("confidence") or "")
        if confidence not in {"confirmed", "inferred", "unverified"}:
            continue
        correction = str(row.get("correction_receipt_no") or "")
        if not correction:
            continue
        original = row.get("original_receipt_no")
        evidence = row.get("evidence") or ()
        found.append(
            FilingRelation(
                correction_receipt_no=correction,
                original_receipt_no=None if not original else str(original),
                confidence=confidence,  # type: ignore[arg-type]
                evidence=tuple(str(item) for item in evidence),
                corp_code=str(row.get("corp_code") or ""),
            )
        )
    return tuple(found)


def pointer_from_zip(zip_path: Path) -> CorrectionPointer | None:
    if not zip_path.is_file():
        return None
    with zipfile.ZipFile(zip_path) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".xml")]
        found: list[CorrectionPointer] = []
        for name in names:
            text = archive.read(name).decode("utf-8", errors="replace")
            pointer = correction_pointer(text)
            if pointer is not None:
                found.append(pointer)
    if not found:
        return None
    periodic = {"사업보고서", "반기보고서", "분기보고서"}
    for pointer in found:
        if pointer.document_name in periodic:
            return pointer
    return found[0]


def _same_original(
    candidate: FilingPeriod,
    correction: FilingPeriod | None,
    pointer: CorrectionPointer,
) -> bool:
    if correction is not None and correction.corp_code and candidate.corp_code:
        if candidate.corp_code != correction.corp_code:
            return False
    if not pointer.first_submitted_on or candidate.receipt_date != pointer.first_submitted_on:
        return False
    if not pointer.document_name:
        return False
    if candidate.report_name.startswith("[기재정정]"):
        return False
    return pointer.document_name in candidate.report_name

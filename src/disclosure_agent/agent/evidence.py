"""Turn retrieval hits into source-grounded evidence before answer generation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from disclosure_agent.agent.search import SearchHit
from disclosure_agent.agent.settings import EXCERPT_CHARS
from disclosure_agent.agent.source_view import element_view, project_content_blocks


@dataclass
class Evidence:
    evidence_id: str
    retrieval_unit_id: str
    receipt_no: str
    document_name: str
    company_name: str
    section_path: list[str]
    element_ids: list[str]
    excerpt: str
    source_locator: list[dict[str, Any]]
    text: str
    report_name: str = ""
    filing_period: str = ""
    receipt_date: str = ""
    source_sha256: str = ""
    corp_code: str = ""
    blocks: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "retrieval_unit_id": self.retrieval_unit_id,
            "receipt_no": self.receipt_no,
            "document_name": self.document_name,
            "company_name": self.company_name,
            "report_name": self.report_name,
            "filing_period": self.filing_period,
            "receipt_date": self.receipt_date,
            "source_sha256": self.source_sha256,
            "corp_code": self.corp_code,
            "section_path": list(self.section_path),
            "element_ids": list(self.element_ids),
            "excerpt": self.excerpt,
            "text": self.text,
            "source_locator": self.source_locator,
            "blocks": self.blocks,
        }


class EvidenceBuilder:
    def __init__(
        self,
        element_locators: dict[str, dict[str, Any]] | None = None,
        element_index: dict[str, tuple[Path, int]] | None = None,
    ) -> None:
        self.element_locators = element_locators or {}
        self.element_index = element_index or {}
        self._views: dict[str, dict[str, Any]] = {}

    @classmethod
    def from_elements_path(cls, path: Path | None) -> EvidenceBuilder:
        if path is None or not path.is_file():
            return cls({})
        locators, index = load_element_index(path)
        return cls(locators, index)

    @classmethod
    def from_elements_paths(cls, paths: tuple[Path, ...] | list[Path]) -> EvidenceBuilder:
        locators: dict[str, dict[str, Any]] = {}
        index: dict[str, tuple[Path, int]] = {}
        for path in paths:
            if path.is_file():
                loaded_locators, loaded_index = load_element_index(path)
                locators.update(loaded_locators)
                index.update(loaded_index)
        return cls(locators, index)

    def build(self, hits: list[SearchHit], *, company_name: str) -> list[Evidence]:
        needed: list[str] = []
        for hit in hits:
            needed.extend(hit.source_element_ids)
        self._ensure_views(needed)
        evidence: list[Evidence] = []
        seen: set[str] = set()
        for hit in hits:
            evidence_id = _evidence_id(hit.retrieval_unit_id)
            if evidence_id in seen:
                evidence_id = f"{evidence_id}-{hit.rank}"
            seen.add(evidence_id)
            locator = _attach_element_locators(hit.source_locator, self.element_locators)
            evidence.append(
                Evidence(
                    evidence_id=evidence_id,
                    retrieval_unit_id=hit.retrieval_unit_id,
                    receipt_no=hit.receipt_no,
                    document_name=hit.document_name,
                    company_name=company_name,
                    section_path=list(hit.section_path),
                    element_ids=list(hit.source_element_ids),
                    excerpt=_excerpt(hit.text),
                    source_locator=locator,
                    text=hit.text,
                    report_name=hit.report_name,
                    filing_period=hit.filing_period,
                    receipt_date=hit.receipt_date,
                    source_sha256=hit.source_sha256,
                    corp_code=hit.corp_code,
                    blocks=project_content_blocks(hit.source_refs, self._views),
                )
            )
        return evidence

    def _ensure_views(self, element_ids: list[str]) -> None:
        grouped: dict[Path, list[tuple[str, int]]] = {}
        for element_id in element_ids:
            if element_id in self._views:
                continue
            located = self.element_index.get(element_id)
            if located is None:
                self._views[element_id] = {}
                continue
            path, offset = located
            grouped.setdefault(path, []).append((element_id, offset))
        for path, rows in grouped.items():
            with path.open("rb") as handle:
                for element_id, offset in rows:
                    handle.seek(offset)
                    line = handle.readline()
                    if not line.strip():
                        self._views[element_id] = {}
                        continue
                    row = json.loads(line)
                    self._views[element_id] = element_view(row) if isinstance(row, dict) else {}


def load_element_locators(path: Path) -> dict[str, dict[str, Any]]:
    locators, _index = load_element_index(path)
    return locators


def load_element_index(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, tuple[Path, int]]]:
    """Locator objects, plus a byte offset for each element.

    Display cells are read later, and only for elements a citation actually uses.
    """
    locators: dict[str, dict[str, Any]] = {}
    index: dict[str, tuple[Path, int]] = {}
    with path.open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            element_id = row.get("element_id")
            if not element_id:
                continue
            key = str(element_id)
            index[key] = (path, offset)
            locator = row.get("source_locator")
            if isinstance(locator, dict):
                locators[key] = locator
    return locators, index


def _evidence_id(retrieval_unit_id: str) -> str:
    digest = hashlib.sha256(retrieval_unit_id.encode("utf-8")).hexdigest()[:16]
    return f"ev-{digest}"


_HIDDEN_LINE_PREFIXES = ("[section]", "[context_omitted", "[source]")


def visible_excerpt(text: str) -> str:
    """User-facing passage. Retrieval headers and debug lines stay out."""
    kept: list[str] = []
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(_HIDDEN_LINE_PREFIXES):
            continue
        kept.append(stripped)
    if not kept:
        return ""
    return _excerpt(" ".join(kept))


def _excerpt(text: str) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= EXCERPT_CHARS:
        return collapsed
    return collapsed[: EXCERPT_CHARS - 1] + "…"


def _attach_element_locators(
    spans: list[dict[str, Any]],
    locators: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    attached: list[dict[str, Any]] = []
    for span in spans:
        row = dict(span)
        element_id = str(span.get("element_id") or "")
        locator = locators.get(element_id)
        if locator is not None:
            row["element_source_locator"] = locator
        attached.append(row)
    return attached

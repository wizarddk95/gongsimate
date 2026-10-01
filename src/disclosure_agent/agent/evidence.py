"""Turn retrieval hits into source-grounded evidence before answer generation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from disclosure_agent.agent.search import SearchHit
from disclosure_agent.agent.settings import EXCERPT_CHARS


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "retrieval_unit_id": self.retrieval_unit_id,
            "receipt_no": self.receipt_no,
            "document_name": self.document_name,
            "company_name": self.company_name,
            "section_path": list(self.section_path),
            "element_ids": list(self.element_ids),
            "excerpt": self.excerpt,
            "source_locator": self.source_locator,
        }


class EvidenceBuilder:
    def __init__(self, element_locators: dict[str, dict[str, Any]] | None = None) -> None:
        self.element_locators = element_locators or {}

    @classmethod
    def from_elements_path(cls, path: Path | None) -> EvidenceBuilder:
        if path is None or not path.is_file():
            return cls({})
        return cls(load_element_locators(path))

    def build(self, hits: list[SearchHit], *, company_name: str) -> list[Evidence]:
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
                )
            )
        return evidence


def load_element_locators(path: Path) -> dict[str, dict[str, Any]]:
    locators: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            element_id = row.get("element_id")
            locator = row.get("source_locator")
            if element_id and isinstance(locator, dict):
                locators[str(element_id)] = locator
    return locators


def _evidence_id(retrieval_unit_id: str) -> str:
    digest = hashlib.sha256(retrieval_unit_id.encode("utf-8")).hexdigest()[:16]
    return f"ev-{digest}"


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

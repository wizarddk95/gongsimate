"""Citations may point only at evidence ids from the current run."""

from __future__ import annotations

from dataclasses import dataclass

from disclosure_agent.agent.errors import CitationValidationFailure


@dataclass(frozen=True)
class Citation:
    citation_id: str
    evidence_id: str

    def to_dict(self) -> dict[str, str]:
        return {"citation_id": self.citation_id, "evidence_id": self.evidence_id}


def assign_citation_ids(evidence_ids: list[str]) -> list[Citation]:
    """Assign user-facing numbers from first appearance. The model does not choose them."""
    assigned: list[Citation] = []
    seen: dict[str, str] = {}
    for evidence_id in evidence_ids:
        if evidence_id in seen:
            continue
        citation_id = str(len(seen) + 1)
        seen[evidence_id] = citation_id
        assigned.append(Citation(citation_id=citation_id, evidence_id=evidence_id))
    return assigned


class CitationValidator:
    def validate(self, citations: list[Citation], evidence_ids: set[str]) -> None:
        for citation in citations:
            if citation.evidence_id not in evidence_ids:
                raise CitationValidationFailure(
                    f"citation {citation.citation_id} references {citation.evidence_id}, "
                    "which is not in the current run"
                )

"""Separate canonical source, retrieval unit, and generation context.

These sizes are different contracts:

- embedding model limit: what the encoder can accept
- retrieval unit target: how large a search string should be
- context unit budget: how much cited evidence is assembled for an answer

The 2803-chunk BGE-M3 audit (nearest-rank) is the basis for the targets.
p50 is 51 tokens, p99 is 558, and 2,761 of 2,803 chunks are at or below 512.
Forty-two chunks exceed 512. Five exceed 1,024. Two 1x1 accounting-policy
cells exceed 8,192 (ordinal 266 = 12,711 and ordinal 1381 = 11,815). Those
two cells are not gold for the current questions.

512 is the size a split piece should approach. It is not a chop threshold:
a structurally atomic piece may sit between 512 and the hard max.
1,024 is the hard max because it is twice the target, it is the first audit
threshold that isolates the five-chunk tail, and it stays far below the
8,192 model limit. Pieces between 513 and 1,024 stay whole. Anything above
1,024 must be split. A retrieval unit is not sized to the model maximum.
"""

from __future__ import annotations

from dataclasses import dataclass

EMBEDDING_MODEL_MAX_TOKENS = 8192
RETRIEVAL_UNIT_TARGET_TOKENS = 512
RETRIEVAL_UNIT_HARD_MAX_TOKENS = 1024
CONTEXT_UNIT_BUDGET_TOKENS = 512


@dataclass(frozen=True)
class CanonicalSource:
    """Citation record. Element and cell coordinates live here."""

    chunk_id: str
    ordinal: int
    text: str
    source_refs: tuple[dict, ...]


@dataclass(frozen=True)
class RetrievalUnit:
    """Search string. It may differ from the canonical text."""

    chunk_id: str
    ordinal: int
    version: str
    representation_type: str
    text: str
    source_refs: tuple[dict, ...]
    embedding_token_count: int | None = None
    embedding_truncated: bool | None = None
    embedding_model_max_tokens: int | None = None


@dataclass(frozen=True)
class ContextUnit:
    """Evidence selected for an answer. Not a promise to paste the retrieval unit."""

    chunk_id: str
    ordinal: int
    text: str
    source_refs: tuple[dict, ...]
    token_count: int
    budget_tokens: int
    over_budget: bool
    omitted_full_retrieval_text: bool


def assemble_context_unit(
    chunk: dict,
    *,
    token_count: int,
    budget_tokens: int = CONTEXT_UNIT_BUDGET_TOKENS,
) -> ContextUnit:
    """Keep provenance when the retrieval text is over the context budget.

    Over-budget text is not copied into the context unit. A later assembler
    can select rows from ``source_refs``. This function does not call a model.
    """
    sources = tuple(chunk.get("chunk_sources") or [])
    if token_count <= budget_tokens:
        return ContextUnit(
            chunk_id=chunk["chunk_id"],
            ordinal=int(chunk["ordinal"]),
            text=chunk.get("text") or "",
            source_refs=sources,
            token_count=token_count,
            budget_tokens=budget_tokens,
            over_budget=False,
            omitted_full_retrieval_text=False,
        )
    section = " > ".join(chunk.get("section_path") or [])
    lines = ["[context_omitted_over_budget]"]
    if section:
        lines.append(f"[section] {section}")
    for source in sources:
        lines.append(
            "[source] "
            f"element_id={source.get('element_id')} "
            f"role={source.get('role')} "
            f"kind={source.get('kind')}"
        )
    return ContextUnit(
        chunk_id=chunk["chunk_id"],
        ordinal=int(chunk["ordinal"]),
        text="\n".join(lines),
        source_refs=sources,
        token_count=token_count,
        budget_tokens=budget_tokens,
        over_budget=True,
        omitted_full_retrieval_text=True,
    )

"""Retrieval representations derived from canonical chunks.

Canonical chunk text and ``chunk_sources`` stay the citation record.
A representation is a search string. It does not replace source coordinates.
"""

from __future__ import annotations

from typing import Any

from disclosure_agent.chunking import canonical_dumps, sha256_text
from disclosure_agent.errors import UsageError

CANONICAL_REPRESENTATION_VERSION = "canonical-v1"
STRUCTURED_TABLE_VERSION = "structured-table-v1"
BLANK_HEADER_FIELD = "항목"

# D1 population. The serializer does not read these ids.
# E228 is the consolidated statement of comprehensive income.
# E2362 is the profitability-indicator table. Its consolidated scope
# sentence is a different element and is not inserted here.
D1_STRUCTURED_TABLES = (
    {
        "element_key": "E228",
        "element_id": "f55b61e36dd52bfcfeda562f3dcc163dd8bc4051f2176d3b248b13034f60f244",
    },
    {
        "element_key": "E2362",
        "element_id": "9b792cedd0b5cac5fb96d20943d300ae51dcf509b090e917367dad1152e341de",
    },
)
D1_TARGET_ELEMENT_IDS = frozenset(item["element_id"] for item in D1_STRUCTURED_TABLES)

_HEADER_ROLES = {"header", "repeated_header"}
_UNIT_ROLES = {"unit", "repeated_unit", "context_unit"}
_TITLE_ROLES = {"context_title"}
_PERIOD_ROLES = {"context_period"}
_BODY_ROLES = {"body"}


def embedding_input_record(
    text: str, *, token_count: int, max_tokens: int
) -> dict[str, Any]:
    """Record truncation. The stored string stays the full representation."""
    if token_count < 0 or max_tokens < 1:
        raise UsageError("token counts must be non-negative and max_tokens >= 1")
    if text is None:
        raise UsageError("retrieval representation text is missing")
    return {
        "stored_text": text,
        "embedding_token_count": token_count,
        "embedding_truncated": token_count > max_tokens,
        "embedding_model_max_tokens": max_tokens,
        "stored_text_is_full_representation": True,
    }


def source_refs(chunk: dict[str, Any]) -> list[dict[str, Any]]:
    return list(chunk.get("chunk_sources") or [])


def chunk_body_element_ids(chunk: dict[str, Any]) -> set[str]:
    found = set()
    for source in source_refs(chunk):
        if source.get("kind") == "table" and source.get("role") in _BODY_ROLES:
            found.add(source["element_id"])
    return found


def uses_structured_table(chunk: dict[str, Any], target_element_ids: frozenset[str]) -> bool:
    return bool(chunk_body_element_ids(chunk) & target_element_ids)


def representation_sha256(records: list[dict[str, Any]]) -> str:
    body = "\n".join(canonical_dumps(_hash_record(record)) for record in records)
    return sha256_text(body)


def build_retrieval_representations(
    chunks: list[dict[str, Any]],
    elements_by_id: dict[str, dict[str, Any]],
    *,
    mode: str,
    target_element_ids: frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    """Return one representation per canonical chunk, in the same order."""
    if mode not in {"canonical", "structured-table-v1"}:
        raise UsageError(f"unknown retrieval representation mode {mode!r}")
    targets = D1_TARGET_ELEMENT_IDS if target_element_ids is None else target_element_ids
    records = []
    for chunk in chunks:
        structured = mode == "structured-table-v1" and uses_structured_table(chunk, targets)
        text = (
            serialize_table_chunk(chunk, elements_by_id)
            if structured
            else chunk.get("text") or ""
        )
        if structured and text == (chunk.get("text") or ""):
            raise UsageError(
                f"structured representation did not change chunk ordinal {chunk['ordinal']}"
            )
        version = STRUCTURED_TABLE_VERSION if structured else CANONICAL_REPRESENTATION_VERSION
        kind = "structured_table" if structured else "canonical_text"
        records.append(
            {
                "chunk_id": chunk["chunk_id"],
                "ordinal": chunk["ordinal"],
                "version": version,
                "type": kind,
                "text": text,
                "text_hash": sha256_text(text),
                "source_refs": source_refs(chunk),
                "canonical_text_hash": chunk.get("text_hash"),
            }
        )
    return records


def serialize_table_chunk(
    chunk: dict[str, Any], elements_by_id: dict[str, dict[str, Any]]
) -> str:
    """Field-style text from cells already attached to this chunk.

    Scope words are copied only when a source cell or preface contains them.
    """
    title_lines: list[str] = []
    period_lines: list[str] = []
    unit_lines: list[str] = []
    header_cells: list[dict[str, Any]] = []
    body_cells: list[dict[str, Any]] = []
    for source in source_refs(chunk):
        role = source.get("role")
        if source.get("kind") != "table":
            text = _paragraph_text(source, elements_by_id)
            if not text:
                continue
            if role in _TITLE_ROLES:
                title_lines.append(text)
            elif role in _PERIOD_ROLES:
                period_lines.append(text)
            elif role in _UNIT_ROLES:
                unit_lines.append(text)
            continue
        cells = _table_cells(source, elements_by_id)
        if role in _TITLE_ROLES:
            title_lines.extend(_nonempty(cells))
        elif role in _PERIOD_ROLES:
            period_lines.extend(_nonempty(cells))
        elif role in _UNIT_ROLES:
            unit_lines.extend(_nonempty(cells))
        elif role in _HEADER_ROLES:
            header_cells.extend(cells)
        elif role in _BODY_ROLES:
            body_cells.extend(cells)
    if not body_cells:
        raise UsageError(
            f"structured table chunk ordinal {chunk.get('ordinal')} has no body cells"
        )
    labels = _column_labels(header_cells)
    lines: list[str] = []
    section = " > ".join(chunk.get("section_path") or [])
    if section:
        lines.append(f"[section] {section}")
    for text in _unique(title_lines):
        lines.append(f"[표] {text}")
    for text in _unique(period_lines):
        lines.append(f"[기간] {text}")
    for text in _unique(unit_lines):
        lines.append(f"[단위] {text}")
    blocks = _body_blocks(body_cells, labels)
    if not blocks:
        raise UsageError(
            f"structured table chunk ordinal {chunk.get('ordinal')} has no non-empty body cells"
        )
    rendered = []
    if lines:
        rendered.append("\n".join(lines))
    rendered.extend(blocks)
    return "\n\n".join(rendered)


def _hash_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": record["chunk_id"],
        "ordinal": record["ordinal"],
        "version": record["version"],
        "type": record["type"],
        "text_hash": record["text_hash"],
        "source_refs": record["source_refs"],
    }


def _paragraph_text(source: dict[str, Any], elements_by_id: dict[str, dict[str, Any]]) -> str:
    element = elements_by_id[source["element_id"]]
    text = element.get("text") or ""
    start = int(source.get("char_start") or 0)
    end = int(source.get("char_end") or 0)
    return _flat(text[start:end])


def _table_cells(
    source: dict[str, Any], elements_by_id: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    element = elements_by_id[source["element_id"]]
    lookup = {_cell_key(cell): cell for cell in (element.get("table") or {}).get("cells") or []}
    materialized = []
    for ref in source.get("cells") or []:
        key = _cell_key(ref)
        cell = lookup.get(key)
        if cell is None:
            raise UsageError(
                f"chunk source cell {key} is missing from element {source['element_id']}"
            )
        raw = cell.get("text") or ""
        digest = sha256_text(raw)
        if digest != ref.get("text_sha256"):
            raise UsageError(
                "cell text does not match chunk source lineage "
                f"element={source['element_id']} cell={key}"
            )
        materialized.append(
            {
                "start_row": key[0],
                "start_col": key[1],
                "rowspan": key[2],
                "colspan": key[3],
                "text": raw,
            }
        )
    materialized.sort(key=lambda cell: (cell["start_row"], cell["start_col"]))
    return materialized


def _cell_key(cell: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(cell["start_row"]),
        int(cell["start_col"]),
        int(cell.get("rowspan") or 1),
        int(cell.get("colspan") or 1),
    )


def _flat(text: str) -> str:
    return (text or "").replace("\n", " ").strip()


def _nonempty(cells: list[dict[str, Any]]) -> list[str]:
    return [text for cell in cells if (text := _flat(cell["text"]))]


def _unique(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    kept = []
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        kept.append(line)
    return kept


def _covers(cell: dict[str, Any], row: int, col: int) -> bool:
    return (
        cell["start_row"] <= row < cell["start_row"] + cell["rowspan"]
        and cell["start_col"] <= col < cell["start_col"] + cell["colspan"]
    )


def _column_labels(header_cells: list[dict[str, Any]]) -> dict[int, str]:
    if not header_cells:
        return {}
    rows = sorted({cell["start_row"] for cell in header_cells})
    last_col = max(cell["start_col"] + cell["colspan"] - 1 for cell in header_cells)
    labels = {}
    for col in range(last_col + 1):
        parts: list[str] = []
        for row in rows:
            for cell in header_cells:
                if not _covers(cell, row, col):
                    continue
                text = _flat(cell["text"])
                if text and (not parts or parts[-1] != text):
                    parts.append(text)
        labels[col] = " / ".join(parts)
    return labels


def _body_blocks(body_cells: list[dict[str, Any]], labels: dict[int, str]) -> list[str]:
    by_row: dict[int, list[dict[str, Any]]] = {}
    for cell in body_cells:
        by_row.setdefault(cell["start_row"], []).append(cell)
    blocks = []
    for row in sorted(by_row):
        fields = []
        for cell in sorted(by_row[row], key=lambda item: item["start_col"]):
            text = _flat(cell["text"])
            if not text:
                continue
            label = labels.get(cell["start_col"]) or BLANK_HEADER_FIELD
            fields.append(f"[{label}] {text}")
        if fields:
            blocks.append("\n".join(fields))
    return blocks

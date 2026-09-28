"""Deterministic chunker over parsed disclosure elements.

Search input is a parse manifest with ``overall_status == success``.
``COMPLETE`` and the CLI exit code are not eligibility signals.

``header_role == "unknown"`` is the parser policy for ``TD``/``TE``. It does not
mean the cell has no header text. Column-header rows are chosen from ``TH`` /
``THEAD`` and from the position of the row above the first numeric body row.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from disclosure_agent.errors import UsageError
from disclosure_agent.parsing import SEARCH_ELIGIBLE_STATUSES

CHUNKER_VERSION = "1"
SCHEMA_VERSION = 1
UNIT_ROW_POLICY = "entire-cell-parenthesized-unit-phrase"

# chunk_id is the SHA-256 of the canonical JSON for this identity.
# section_path, unit mentions, and row_rects are not separate inputs:
# section_path is already inside text_hash, and row_rects are derived from cells.
CHUNK_ID_FIELDS = (
    "chunker_version",
    "chunker_config_hash",
    "parser_config_hash",
    "source_sha256",
    "document_id",
    "ordinal",
    "text_hash",
    "sources.element_id",
    "sources.role",
    "sources.char_start",
    "sources.char_end",
    "sources.cells[start_row,start_col,rowspan,colspan]",
)

_PURE_UNIT_RE = re.compile(r"^\s*\(\s*단위\s*[:：][^)]*\)\s*$")
_AMOUNT_RE = re.compile(
    r"(?<![\d.])(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d{7,}|\d+\.\d+%|\d+%)"
)

DEFAULT_CHUNKER_CONFIG: dict[str, Any] = {
    "chunker_version": CHUNKER_VERSION,
    "paragraph_target_chars": 800,
    "paragraph_max_chars": 4000,
    "table_split_min_body_rows": 8,
    "table_body_rows_per_chunk": 4,
    "preface_paragraph_max_chars": 200,
    "caption_table_max_rows": 6,
    "caption_table_max_cols": 2,
    "header_detection": "th-or-band-above-first-amount",
    "unknown_header_role_means_missing_text": False,
    "repeat_unit_and_header_on_split": True,
    "link_preceding_scope_or_caption": True,
    "unit_row": UNIT_ROW_POLICY,
}


def canonical_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def chunker_config_hash(config: dict[str, Any] | None = None) -> str:
    body = dict(DEFAULT_CHUNKER_CONFIG if config is None else config)
    return sha256_text(canonical_dumps(body))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_elements(path: Path) -> list[dict[str, Any]]:
    elements: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise UsageError(f"elements JSONL line {line_no} is not JSON") from exc
            if not isinstance(item, dict):
                raise UsageError(f"elements JSONL line {line_no} is not an object")
            elements.append(item)
    return elements


def assert_search_eligible(manifest: dict[str, Any]) -> None:
    status = manifest.get("overall_status")
    if status not in SEARCH_ELIGIBLE_STATUSES:
        raise UsageError(
            "search input requires parse_manifest overall_status='success', "
            f"got {status!r}. COMPLETE and the process exit code are not used."
        )


def member_for_document(manifest: dict[str, Any], document_id: str) -> dict[str, Any]:
    for member in manifest.get("members") or []:
        if isinstance(member, dict) and member.get("document_id") == document_id:
            return member
    raise UsageError(f"document_id {document_id!r} is not in the parse manifest")


def chunk_elements(
    elements: list[dict[str, Any]],
    *,
    manifest: dict[str, Any],
    document_id: str,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Chunk one parsed document. Element order inside the document is kept."""
    assert_search_eligible(manifest)
    resolved = dict(DEFAULT_CHUNKER_CONFIG if config is None else config)
    unit_row = resolved.get("unit_row", UNIT_ROW_POLICY)
    if unit_row != UNIT_ROW_POLICY:
        raise UsageError(
            f"unsupported unit_row policy {unit_row!r}. "
            f"Expected {UNIT_ROW_POLICY!r}."
        )
    resolved["unit_row"] = unit_row
    member = member_for_document(manifest, document_id)
    selected = _select_elements(elements, manifest, document_id)
    config_hash = chunker_config_hash(resolved)
    lineage = {
        "receipt_no": manifest.get("receipt_no"),
        "source_sha256": manifest.get("source_sha256"),
        "parser_version": manifest.get("parser_version"),
        "preprocess_version": manifest.get("preprocess_version"),
        "parser_config_hash": manifest.get("config_hash"),
        "document_id": document_id,
        "member_index": member.get("member_index"),
        "member_name": member.get("member_name"),
        "document_name_observed": member.get("document_name"),
        "chunker_version": resolved["chunker_version"],
        "chunker_config_hash": config_hash,
    }
    raw: list[dict[str, Any]] = []
    index = 0
    while index < len(selected):
        element = selected[index]
        kind = element.get("element_type")
        if kind == "heading":
            raw.extend(_paragraph_chunks([element], selected, index, resolved))
            index += 1
            continue
        if kind == "paragraph":
            group, index = _take_paragraph_group(selected, index, resolved)
            raw.extend(_paragraph_chunks(group, selected, index - len(group), resolved))
            continue
        if kind == "table":
            raw.extend(_table_chunks(selected, index, resolved))
            index += 1
            continue
        raw.extend(_leaf_chunks(element))
        index += 1
    return [_finalize(item, ordinal, lineage) for ordinal, item in enumerate(raw)]


def chunks_artifact_sha256(chunks: list[dict[str, Any]]) -> str:
    """Hash the full canonical chunk records, not only chunk ids."""
    return sha256_text("\n".join(canonical_dumps(chunk) for chunk in chunks))


def chunk_manifest_summary(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [chunk["chunk_id"] for chunk in chunks]
    return {
        "schema_version": SCHEMA_VERSION,
        "chunker_version": CHUNKER_VERSION,
        "chunk_count": len(chunks),
        "chunk_id_sha256": sha256_text("\n".join(ids)),
        "artifact_sha256": chunks_artifact_sha256(chunks),
        "document_ids": sorted({chunk["lineage"]["document_id"] for chunk in chunks}),
        "chunker_config_hash": None if not chunks else chunks[0]["chunker_config_hash"],
    }


def _select_elements(
    elements: list[dict[str, Any]], manifest: dict[str, Any], document_id: str
) -> list[dict[str, Any]]:
    source_sha = manifest.get("source_sha256")
    selected: list[dict[str, Any]] = []
    for element in elements:
        if element.get("document_id") != document_id:
            continue
        locator = element.get("source_locator") or {}
        located = locator.get("source_sha256")
        if located and source_sha and located != source_sha:
            raise UsageError("element source_sha256 does not match the parse manifest")
        selected.append(element)
    selected.sort(key=lambda item: (item.get("order_index", 0), item.get("element_id", "")))
    return selected


def _take_paragraph_group(
    elements: list[dict[str, Any]], start: int, config: dict[str, Any]
) -> tuple[list[dict[str, Any]], int]:
    first = elements[start]
    section = first.get("section_path")
    group = [first]
    total = len(first.get("text") or "")
    target = int(config["paragraph_target_chars"])
    limit = int(config["paragraph_max_chars"])
    index = start + 1
    if total > limit:
        return group, index
    while index < len(elements):
        nxt = elements[index]
        if nxt.get("element_type") != "paragraph" or nxt.get("section_path") != section:
            break
        nxt_len = len(nxt.get("text") or "")
        if total >= target or total + nxt_len > limit:
            break
        group.append(nxt)
        total += nxt_len
        index += 1
    return group, index


def _paragraph_chunks(
    group: list[dict[str, Any]],
    elements: list[dict[str, Any]],
    start: int,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    del elements, start
    limit = int(config["paragraph_max_chars"])
    if len(group) == 1 and len(group[0].get("text") or "") > limit:
        pieces = _split_ranges(group[0]["text"], limit)
        return [_one_paragraph_piece(group[0], piece) for piece in pieces]
    sources = [
        _paragraph_source(element, 0, len(element.get("text") or ""), "body") for element in group
    ]
    parts = [item.get("text") or "" for item in group]
    text = _render_paragraphs(group[0].get("section_path") or [], parts)
    mentions = _mentions_for_paragraphs(group)
    return [
        {
            "section_path": list(group[0].get("section_path") or []),
            "text": text,
            "chunk_sources": sources,
            "unit_mentions": mentions,
        }
    ]


def _one_paragraph_piece(element: dict[str, Any], piece: tuple[int, int]) -> dict[str, Any]:
    start, end = piece
    text_piece = (element.get("text") or "")[start:end]
    role = "body"
    return {
        "section_path": list(element.get("section_path") or []),
        "text": _render_paragraphs(element.get("section_path") or [], [text_piece]),
        "chunk_sources": [_paragraph_source(element, start, end, role)],
        "unit_mentions": _mentions_for_paragraphs([element]) if start == 0 else [],
    }


def _split_ranges(text: str, limit: int) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + limit)
        if end < len(text):
            window = text[start:end]
            candidates = [window.rfind("\n"), window.rfind("다. "), window.rfind(". ")]
            cut = max(candidates)
            if cut >= limit // 3:
                end = start + cut + 1
        if end <= start:
            end = min(len(text), start + limit)
        ranges.append((start, end))
        start = end
    return ranges


def _paragraph_source(element: dict[str, Any], start: int, end: int, role: str) -> dict[str, Any]:
    return {
        "element_id": element["element_id"],
        "element_type": element.get("element_type"),
        "order_index": element.get("order_index"),
        "document_id": element.get("document_id"),
        "kind": "paragraph",
        "role": role,
        "char_start": start,
        "char_end": end,
        "cells": [],
    }


def _table_chunks(
    elements: list[dict[str, Any]], index: int, config: dict[str, Any]
) -> list[dict[str, Any]]:
    element = elements[index]
    table = element.get("table") or {}
    cells = list(table.get("cells") or [])
    row_count = int(table.get("row_count") or 0)
    bands = _row_bands(cells, row_count)
    if config.get("link_preceding_scope_or_caption"):
        preface = _preface(elements, index, config)
    else:
        preface = None
    groups = _body_groups(bands["body"], cells, int(config["table_body_rows_per_chunk"]))
    split = len(bands["body"]) >= int(config["table_split_min_body_rows"]) and len(groups) > 1
    if not split:
        groups = [bands["body"]] if bands["body"] else [[]]
    chunks: list[dict[str, Any]] = []
    for group in groups:
        if not group and not bands["unit"] and not bands["header"] and not cells:
            continue
        chunks.append(_one_table_chunk(element, cells, bands, group, preface, repeated=split))
    if not chunks:
        chunks.append(_one_table_chunk(element, cells, bands, [], preface, repeated=False))
    return chunks


def _one_table_chunk(
    element: dict[str, Any],
    cells: list[dict[str, Any]],
    bands: dict[str, list[int]],
    body_rows: list[int],
    preface: tuple[str, dict[str, Any]] | None,
    *,
    repeated: bool,
) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    lines: list[str] = []
    included_mentions: list[dict[str, Any]] = []
    if preface is not None:
        kind, prev = preface
        if kind == "paragraph":
            text = prev.get("text") or ""
            role = "context_scope" if "연결기준" in text or "별도기준" in text else "context_title"
            sources.append(_paragraph_source(prev, 0, len(text), role))
            lines.append(text)
            included_mentions.extend(_mentions_for_paragraphs([prev]))
        else:
            prev_cells = list((prev.get("table") or {}).get("cells") or [])
            grouped = _group_caption_cells(prev, prev_cells)
            sources.extend(grouped["sources"])
            lines.extend(grouped["lines"])
            included_mentions.extend(grouped["mentions"])
    header_role = "repeated_header" if repeated else "header"
    unit_role = "repeated_unit" if repeated else "unit"
    for role, rows in ((unit_role, bands["unit"]), (header_role, bands["header"])):
        block = _cells_on_rows(cells, rows)
        if not block:
            continue
        sources.append(_table_source(element, role, block))
        lines.extend(_render_rows(block, rows))
        included_mentions.extend(_mentions_for_cells(element, block))
    body_cells = _cells_on_rows(cells, body_rows)
    if body_cells:
        sources.append(_table_source(element, "body", body_cells))
        lines.extend(_render_rows(body_cells, body_rows))
        included_mentions.extend(_mentions_for_cells(element, body_cells))
    text = _render_paragraphs(element.get("section_path") or [], lines)
    return {
        "section_path": list(element.get("section_path") or []),
        "text": text,
        "chunk_sources": sources,
        "unit_mentions": _dedupe_mentions(included_mentions),
    }


def _leaf_chunks(element: dict[str, Any]) -> list[dict[str, Any]]:
    text = element.get("text") or ""
    if not text and element.get("element_type") != "image":
        return []
    return [
        {
            "section_path": list(element.get("section_path") or []),
            "text": _render_paragraphs(element.get("section_path") or [], [text]),
            "chunk_sources": [_paragraph_source(element, 0, len(text), "body")],
            "unit_mentions": _mentions_for_paragraphs([element]),
        }
    ]


def _row_bands(cells: list[dict[str, Any]], row_count: int) -> dict[str, list[int]]:
    by_row: dict[int, list[dict[str, Any]]] = {}
    for cell in cells:
        by_row.setdefault(int(cell["start_row"]), []).append(cell)
    unit_rows = [row for row in range(row_count) if _is_unit_row(by_row.get(row, []))]
    first_amount: int | None = None
    for row in range(row_count):
        if row in unit_rows:
            continue
        if any(_AMOUNT_RE.search(cell.get("text") or "") for cell in by_row.get(row, [])):
            first_amount = row
            break
    header_rows: list[int] = []
    body_rows: list[int] = []
    label_band = first_amount is None
    for row in range(row_count):
        if row in unit_rows:
            continue
        row_cells = by_row.get(row, [])
        if first_amount is not None and row < first_amount:
            header_rows.append(row)
            continue
        if label_band and _looks_like_label_row(row_cells):
            header_rows.append(row)
            continue
        label_band = False
        body_rows.append(row)
    return {"unit": unit_rows, "header": header_rows, "body": body_rows}


def _looks_like_label_row(cells: list[dict[str, Any]]) -> bool:
    """Short non-numeric rows can be headers even when header_role is unknown.

    Used only when the table has no amount-like cell to end the header band.
    A long cell, including a project name or a narrative result, ends the band.
    """
    texts = [(cell.get("text") or "").strip() for cell in cells]
    texts = [text for text in texts if text]
    if not texts or any(_AMOUNT_RE.search(text) for text in texts):
        return False
    return max(len(text) for text in texts) <= 24


def _is_unit_text(text: str) -> bool:
    """A unit caption is the whole cell, such as ``(단위 : 원)``.

    A label that merely contains that phrase, such as ``수권주식수(단위: 주)``
    or ``기본주당이익(손실) (단위 : 원)``, stays a body or header cell.
    """
    return _PURE_UNIT_RE.fullmatch(text or "") is not None


def _is_unit_row(cells: list[dict[str, Any]]) -> bool:
    return any(_is_unit_text(cell.get("text") or "") for cell in cells)


def _body_groups(
    body_rows: list[int], cells: list[dict[str, Any]], max_rows: int
) -> list[list[int]]:
    if not body_rows:
        return []
    groups: list[list[int]] = []
    current: list[int] = []
    open_until = -1
    for row in body_rows:
        span_end = row
        for cell in cells:
            if int(cell["start_row"]) == row:
                span_end = max(span_end, int(cell["start_row"]) + int(cell.get("rowspan") or 1) - 1)
        if current and row > open_until and len(current) >= max_rows:
            groups.append(current)
            current = [row]
            open_until = span_end
        else:
            current.append(row)
            open_until = max(open_until, span_end)
    if current:
        groups.append(current)
    return groups


def _preface(
    elements: list[dict[str, Any]], index: int, config: dict[str, Any]
) -> tuple[str, dict[str, Any]] | None:
    if index == 0:
        return None
    prev = elements[index - 1]
    current = elements[index]
    if prev.get("section_path") != current.get("section_path"):
        return None
    if prev.get("element_type") == "paragraph":
        text = prev.get("text") or ""
        if text and len(text) <= int(config["preface_paragraph_max_chars"]):
            return ("paragraph", prev)
        return None
    if prev.get("element_type") != "table":
        return None
    table = prev.get("table") or {}
    rows = int(table.get("row_count") or 0)
    cols = int(table.get("col_count") or 0)
    if rows > int(config["caption_table_max_rows"]) or cols > int(config["caption_table_max_cols"]):
        return None
    if not any(_is_unit_text(cell.get("text") or "") for cell in table.get("cells") or []):
        return None
    return ("caption_table", prev)


def _group_caption_cells(
    element: dict[str, Any], cells: list[dict[str, Any]]
) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    order = ["context_title", "context_period", "context_unit"]
    for cell in cells:
        role = _caption_role(cell.get("text") or "")
        buckets.setdefault(role, []).append(cell)
    sources = []
    lines: list[str] = []
    used: list[dict[str, Any]] = []
    for role in order:
        block = buckets.get(role) or []
        if not block:
            continue
        block.sort(key=lambda cell: (cell["start_row"], cell["start_col"]))
        sources.append(_table_source(element, role, block))
        rows = sorted({int(cell["start_row"]) for cell in block})
        lines.extend(_render_rows(block, rows))
        used.extend(block)
    return {
        "sources": sources,
        "lines": lines,
        "mentions": _mentions_for_cells(element, used),
    }


def _caption_role(text: str) -> str:
    if _is_unit_text(text):
        return "context_unit"
    if "부터" in text and "까지" in text:
        return "context_period"
    return "context_title"


def _cells_on_rows(cells: list[dict[str, Any]], rows: list[int]) -> list[dict[str, Any]]:
    wanted = set(rows)
    found = [cell for cell in cells if int(cell["start_row"]) in wanted]
    found.sort(key=lambda cell: (int(cell["start_row"]), int(cell["start_col"])))
    return found


def _row_rects(rows: list[int]) -> list[dict[str, int]]:
    unique = sorted(set(rows))
    if not unique:
        return []
    rects: list[dict[str, int]] = []
    start = previous = unique[0]
    for row in unique[1:]:
        if row == previous + 1:
            previous = row
            continue
        rects.append({"start": start, "end": previous})
        start = previous = row
    rects.append({"start": start, "end": previous})
    return rects


def _table_source(
    element: dict[str, Any], role: str, cells: list[dict[str, Any]]
) -> dict[str, Any]:
    rects = _row_rects([int(cell["start_row"]) for cell in cells])
    contiguous = len(rects) == 1
    return {
        "element_id": element["element_id"],
        "element_type": "table",
        "order_index": element.get("order_index"),
        "document_id": element.get("document_id"),
        "kind": "table",
        "role": role,
        "rows": dict(rects[0]) if contiguous else None,
        "row_rects": rects,
        "rows_contiguous": contiguous,
        "cells": [_cell_ref(cell) for cell in cells],
    }


def _cell_ref(cell: dict[str, Any]) -> dict[str, Any]:
    return {
        "start_row": int(cell["start_row"]),
        "start_col": int(cell["start_col"]),
        "rowspan": int(cell.get("rowspan") or 1),
        "colspan": int(cell.get("colspan") or 1),
        "tag": cell.get("tag"),
        "header_role": cell.get("header_role"),
        "text_sha256": sha256_text(cell.get("text") or ""),
    }


def _render_rows(cells: list[dict[str, Any]], rows: list[int]) -> list[str]:
    by_row: dict[int, list[dict[str, Any]]] = {}
    for cell in cells:
        by_row.setdefault(int(cell["start_row"]), []).append(cell)
    lines = []
    for row in rows:
        ordered = sorted(by_row.get(row, []), key=lambda cell: int(cell["start_col"]))
        parts = [(cell.get("text") or "").replace("\n", " ").strip() for cell in ordered]
        lines.append(" | ".join(parts))
    return lines


def _render_paragraphs(section: list[str], parts: list[str]) -> str:
    body = [part for part in parts if part]
    path = " > ".join(section)
    lines = [f"[section] {path}"] if path else []
    lines.extend(body)
    return "\n".join(lines).strip()


def _mentions_for_paragraphs(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mentions: list[dict[str, Any]] = []
    for element in elements:
        mentions.extend(element.get("unit_mentions") or [])
    return _dedupe_mentions(mentions)


def _mentions_for_cells(
    element: dict[str, Any], cells: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    locators = [cell.get("source_locator") for cell in cells if cell.get("source_locator")]
    found = []
    for mention in element.get("unit_mentions") or []:
        if mention.get("source_locator") in locators:
            found.append(mention)
    return found


def _dedupe_mentions(mentions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for mention in mentions:
        key = canonical_dumps(mention)
        if key in seen:
            continue
        seen.add(key)
        unique.append(mention)
    return unique


def _finalize(item: dict[str, Any], ordinal: int, lineage: dict[str, Any]) -> dict[str, Any]:
    text = item["text"]
    text_hash = sha256_text(text)
    sources = item["chunk_sources"]
    identity = {
        "v": lineage["chunker_version"],
        "chunker_config_hash": lineage["chunker_config_hash"],
        "parser_config_hash": lineage["parser_config_hash"],
        "source_sha256": lineage["source_sha256"],
        "document_id": lineage["document_id"],
        "ordinal": ordinal,
        "text_hash": text_hash,
        "sources": [
            {
                "element_id": source["element_id"],
                "role": source["role"],
                "char_start": source.get("char_start"),
                "char_end": source.get("char_end"),
                "cells": [
                    [cell["start_row"], cell["start_col"], cell["rowspan"], cell["colspan"]]
                    for cell in source.get("cells") or []
                ],
            }
            for source in sources
        ],
    }
    mentions = item.get("unit_mentions") or []
    chunk: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "chunk_id": sha256_text(canonical_dumps(identity)),
        "ordinal": ordinal,
        "text": text,
        "text_hash": text_hash,
        "chunker_version": lineage["chunker_version"],
        "chunker_config_hash": lineage["chunker_config_hash"],
        "lineage": dict(lineage),
        "section_path": item["section_path"],
        "chunk_sources": sources,
        "unit_applied": False,
    }
    if mentions:
        chunk["unit_mentions"] = mentions
        statuses = {mention.get("scope_status") for mention in mentions}
        chunk["unit_scope_status"] = "unresolved" if "unresolved" in statuses else "unknown"
    return chunk

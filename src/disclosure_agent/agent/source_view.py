"""User-facing source blocks for a retrieval unit.

The retrieval ``text`` stays the flattened search representation. This module
does not parse that string. A table is shown only when source refs name cells
and the parsed element still has those cells.
"""

from __future__ import annotations

import hashlib
from typing import Any

_HEADER_ROLES = {"header", "repeated_header"}


def project_content_blocks(
    source_refs: list[dict[str, Any]] | None,
    elements: dict[str, dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Ordered text and table blocks for one evidence item.

    Table cells keep the parsed row, column, and span. Coordinates are shifted
    so the first included row and column start at zero. Spans are not inferred
    from punctuation in the retrieval text.
    """
    refs = [ref for ref in source_refs or [] if isinstance(ref, dict)]
    views = elements or {}
    consumed = [False] * len(refs)
    blocks: list[dict[str, Any]] = []
    for index, ref in enumerate(refs):
        if consumed[index]:
            continue
        element_id = str(ref.get("element_id") or "")
        if _is_table(ref) and element_id:
            group = [ref]
            consumed[index] = True
            for later in range(index + 1, len(refs)):
                other = refs[later]
                if consumed[later] or not _is_table(other):
                    continue
                if str(other.get("element_id") or "") != element_id:
                    continue
                group.append(other)
                consumed[later] = True
            table = _table_block(group, views.get(element_id))
            if table is not None:
                blocks.append(table)
            continue
        consumed[index] = True
        text = _text_block(ref, views.get(element_id))
        if text:
            blocks.append({"type": "text", "text": text})
    return blocks


def element_view(row: dict[str, Any]) -> dict[str, Any]:
    """Display fields for one parsed element. File hashes and locators stay out."""
    kind = str(row.get("element_type") or "")
    view: dict[str, Any] = {
        "element_type": kind,
        "text": "" if kind == "table" else str(row.get("text") or ""),
    }
    table = row.get("table")
    if not isinstance(table, dict):
        return view
    cells = []
    for cell in table.get("cells") or []:
        if not isinstance(cell, dict):
            continue
        cells.append(
            {
                "start_row": _span_int(cell.get("start_row")),
                "start_col": _span_int(cell.get("start_col")),
                "rowspan": max(1, _span_int(cell.get("rowspan"), default=1)),
                "colspan": max(1, _span_int(cell.get("colspan"), default=1)),
                "text": str(cell.get("text") or ""),
                "header_role": str(cell.get("header_role") or ""),
                "tag": str(cell.get("tag") or ""),
            }
        )
    view["cells"] = cells
    return view


def _table_block(
    refs: list[dict[str, Any]],
    element: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not element:
        return None
    by_position = {
        (int(cell["start_row"]), int(cell["start_col"])): cell
        for cell in element.get("cells") or []
        if isinstance(cell, dict)
    }
    chosen: dict[tuple[int, int], dict[str, Any]] = {}
    for ref in refs:
        header = str(ref.get("role") or "") in _HEADER_ROLES
        for pointer in ref.get("cells") or []:
            if not isinstance(pointer, dict):
                continue
            key = (_span_int(pointer.get("start_row")), _span_int(pointer.get("start_col")))
            source = by_position.get(key)
            if source is None:
                continue
            digest = pointer.get("text_sha256")
            if isinstance(digest, str) and digest and digest != _sha256(source["text"]):
                continue
            cell_header = (
                header
                or str(source.get("header_role") or "") == "header"
                or str(pointer.get("header_role") or "") == "header"
                or str(source.get("tag") or "").upper() == "TH"
                or str(pointer.get("tag") or "").upper() == "TH"
            )
            current = chosen.get(key)
            if current is None:
                chosen[key] = {
                    "start_row": key[0],
                    "start_col": key[1],
                    "rowspan": max(1, _span_int(pointer.get("rowspan"), default=source["rowspan"])),
                    "colspan": max(1, _span_int(pointer.get("colspan"), default=source["colspan"])),
                    "text": source["text"],
                    "header": cell_header,
                }
            elif cell_header:
                current["header"] = True
    if not chosen:
        return None
    cells = list(chosen.values())
    min_row = min(cell["start_row"] for cell in cells)
    min_col = min(cell["start_col"] for cell in cells)
    shifted = []
    for cell in cells:
        shifted.append(
            {
                "start_row": cell["start_row"] - min_row,
                "start_col": cell["start_col"] - min_col,
                "rowspan": cell["rowspan"],
                "colspan": cell["colspan"],
                "text": cell["text"],
                "header": cell["header"],
            }
        )
    shifted.sort(key=lambda item: (item["start_row"], item["start_col"]))
    return {
        "type": "table",
        "row_count": max(cell["start_row"] + cell["rowspan"] for cell in shifted),
        "col_count": max(cell["start_col"] + cell["colspan"] for cell in shifted),
        "cells": shifted,
    }


def _text_block(ref: dict[str, Any], element: dict[str, Any] | None) -> str:
    if _is_table(ref) or not element:
        return ""
    text = str(element.get("text") or "")
    start = ref.get("char_start")
    end = ref.get("char_end")
    if isinstance(start, int) and isinstance(end, int) and 0 <= start <= end <= len(text):
        text = text[start:end]
    return text.strip()


def _is_table(ref: dict[str, Any]) -> bool:
    kind = ref.get("kind") or ref.get("element_type")
    return kind == "table"


def _span_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return value


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

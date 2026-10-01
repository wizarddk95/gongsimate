"""Retrieval units projected from immutable canonical chunks.

A retrieval unit is the string the encoder searches. It is not a new canonical
chunk and it is not the context unit a later answer would cite. Gold stays on
source coordinates. Callers map a unit back through ``source_refs``.
"""

from __future__ import annotations

import copy
import math
import re
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from disclosure_agent.chunking import canonical_dumps, sha256_text
from disclosure_agent.context_budget import (
    EMBEDDING_MODEL_MAX_TOKENS,
    RETRIEVAL_UNIT_HARD_MAX_TOKENS,
    RETRIEVAL_UNIT_TARGET_TOKENS,
)
from disclosure_agent.errors import UsageError
from disclosure_agent.representation import _flat, _table_cells

D2A_VERSION = "retrieval-unit-d2a-v1"
D2B_VERSION = "retrieval-unit-d2b-v1"

_HEADER_ROLES = {"header", "repeated_header"}
_UNIT_ROLES = {"unit", "repeated_unit", "context_unit"}
_TITLE_ROLES = {"context_title"}
_PERIOD_ROLES = {"context_period"}
_LABEL_HEADERS = {"구분", "항목", "과목", "계정과목", "계정"}
_FORMULA_HEADERS = {"산식", "계산식", "공식", "산출식"}
_NUMBERED_RE = re.compile(
    r"(?:(?<=^)|(?<=[\s.!?。]))\((\d{1,2})\)(?=\s*[가-힣A-Za-z「『\[])"
)
_SUBSECTION_RE = re.compile(
    r"(?:(?<=[가-힣])|(?<=[\s.!?。]))(\d{1,2})\)(?=[가-힣A-Za-z])"
)
_AMOUNT_VALUE_RE = re.compile(r"^[\d,.\s%원()\-]+$")

TokenCounter = Callable[[str], int]


def build_retrieval_units(
    chunks: list[dict[str, Any]],
    elements_by_id: dict[str, dict[str, Any]],
    *,
    mode: str,
    token_counter: TokenCounter,
    target_tokens: int = RETRIEVAL_UNIT_TARGET_TOKENS,
    hard_max_tokens: int = RETRIEVAL_UNIT_HARD_MAX_TOKENS,
    model_max_tokens: int = EMBEDDING_MODEL_MAX_TOKENS,
) -> list[dict[str, Any]]:
    """Project canonical chunks into retrieval units without editing the chunks."""
    if mode not in {"d2a", "d2b"}:
        raise UsageError(f"unknown retrieval unit mode {mode!r}")
    if target_tokens < 1 or hard_max_tokens < target_tokens or model_max_tokens < hard_max_tokens:
        raise UsageError("token limits must satisfy 1 <= target <= hard max <= model max")
    snapshot = [
        (chunk["chunk_id"], chunk.get("text"), copy.deepcopy(chunk.get("chunk_sources")))
        for chunk in chunks
    ]
    version = D2A_VERSION if mode == "d2a" else D2B_VERSION
    units: list[dict[str, Any]] = []
    for chunk in chunks:
        if mode == "d2a":
            pieces = _d2a_pieces(
                chunk,
                elements_by_id,
                token_counter,
                target_tokens=target_tokens,
                hard_max_tokens=hard_max_tokens,
            )
        else:
            pieces = _d2b_pieces(chunk, elements_by_id)
        units.extend(
            _finish_pieces(
                chunk,
                pieces,
                version=version,
                token_counter=token_counter,
                model_max_tokens=model_max_tokens,
                hard_max_tokens=hard_max_tokens if mode == "d2a" else None,
            )
        )
    for chunk, (chunk_id, text, sources) in zip(chunks, snapshot, strict=True):
        if chunk["chunk_id"] != chunk_id or chunk.get("text") != text:
            raise UsageError("retrieval unit projection edited a canonical chunk")
        if chunk.get("chunk_sources") != sources:
            raise UsageError("retrieval unit projection edited canonical source refs")
    return units


def retrieval_units_sha256(units: list[dict[str, Any]]) -> str:
    rows = [
        {
            "retrieval_unit_id": unit["retrieval_unit_id"],
            "canonical_chunk_id": unit["canonical_chunk_id"],
            "unit_index": unit["unit_index"],
            "representation_version": unit["representation_version"],
            "representation_hash": unit["representation_hash"],
            "source_refs": unit["source_refs"],
        }
        for unit in units
    ]
    body = "\n".join(canonical_dumps(row) for row in rows)
    return sha256_text(body)


def scoring_chunks(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Shape units so the existing source-span scorer can rank them.

    ``chunk_id`` is the retrieval unit id. Gold still reads ``chunk_sources``.
    """
    shaped = []
    for unit in units:
        shaped.append(
            {
                "chunk_id": unit["retrieval_unit_id"],
                "ordinal": unit["canonical_ordinal"],
                "unit_index": unit["unit_index"],
                "text": unit["text"],
                "section_path": list(unit.get("section_path") or []),
                "chunk_sources": unit["source_refs"],
                "canonical_chunk_id": unit["canonical_chunk_id"],
                "retrieval_unit_id": unit["retrieval_unit_id"],
            }
        )
    return shaped


def summarize_retrieval_units(
    units: list[dict[str, Any]],
    *,
    target_tokens: int,
    hard_max_tokens: int,
    model_max_tokens: int,
) -> dict[str, Any]:
    counts = [int(unit["token_count"]) for unit in units]
    ordered = sorted(counts)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        grouped[unit["canonical_chunk_id"]].append(unit)
    split_rows = []
    for group in grouped.values():
        if len(group) < 2:
            continue
        group.sort(key=lambda unit: int(unit["unit_index"]))
        split_rows.append(
            {
                "canonical_chunk_id": group[0]["canonical_chunk_id"],
                "canonical_ordinal": group[0]["canonical_ordinal"],
                "retrieval_unit_count": len(group),
                "split_reasons": sorted({unit["split_reason"] for unit in group}),
                "token_counts": [int(unit["token_count"]) for unit in group],
                "max_tokens": max(int(unit["token_count"]) for unit in group),
                "spans": [_span_brief(unit) for unit in group],
            }
        )
    split_rows.sort(key=lambda row: int(row["canonical_ordinal"]))
    type_counts: dict[str, int] = defaultdict(int)
    reason_counts: dict[str, int] = defaultdict(int)
    for unit in units:
        type_counts[unit["representation_type"]] += 1
        reason_counts[unit["split_reason"]] += 1
    return {
        "retrieval_unit_count": len(units),
        "canonical_chunk_count": len(grouped),
        "one_to_n_count": len(split_rows),
        "one_to_n_canonical_chunks": split_rows,
        "representation_type_counts": dict(sorted(type_counts.items())),
        "split_reason_counts": dict(sorted(reason_counts.items())),
        "token_policy": {
            "target_tokens": target_tokens,
            "hard_max_tokens": hard_max_tokens,
            "embedding_model_max_tokens": model_max_tokens,
            "percentile_definition": "nearest-rank: index = ceil(p/100*n)-1",
        },
        "max_tokens": ordered[-1] if ordered else 0,
        "p50": _nearest(ordered, 50),
        "p90": _nearest(ordered, 90),
        "p95": _nearest(ordered, 95),
        "p99": _nearest(ordered, 99),
        "thresholds": {
            ">512": sum(count > 512 for count in counts),
            ">1024": sum(count > 1024 for count in counts),
            ">2048": sum(count > 2048 for count in counts),
            ">8192": sum(count > 8192 for count in counts),
        },
        "embedding_truncated": sum(bool(unit["embedding_truncated"]) for unit in units),
    }


def slot_diagnostic(
    ranked_hits: list[dict[str, Any]], ks: tuple[int, ...] = (1, 3, 5, 10, 20)
) -> dict[str, Any]:
    """Count canonical sources occupying top-k retrieval-unit slots. Not a headline."""
    report = {}
    for k in ks:
        window = ranked_hits[:k]
        identities = [
            hit.get("canonical_chunk_id") or hit.get("chunk_id") for hit in window
        ]
        unique = len(set(identities))
        report[str(k)] = {
            "retrieval_unit_slots": len(window),
            "unique_canonical_chunks": unique,
            "duplicate_same_source_ru_count": len(window) - unique,
            "role": "diagnostic",
            "not_a_headline_metric": True,
        }
    return report


def compact_row_examples(units: list[dict[str, Any]]) -> dict[str, Any]:
    """One real serialization from each requested shape. Missing shapes stay null."""
    rows = [unit for unit in units if unit["representation_type"] == "table_row"]

    def take(predicate: Any) -> dict[str, Any] | None:
        found = next((unit for unit in rows if predicate(unit)), None)
        return None if found is None else _example_brief(found)

    simple = take(
        lambda unit: "단위=" not in unit["text"]
        and "기간=" not in unit["text"]
        and unit["text"].count("=") == 1
    )
    multi = take(
        lambda unit: "단위=" not in unit["text"]
        and "기간=" not in unit["text"]
        and unit["text"].count("=") >= 3
    )
    ineligible = next(
        (
            unit
            for unit in units
            if unit.get("eligibility_reason") == "single_cell_narrative"
        ),
        None,
    )
    return {
        "simple_table": simple,
        "multi_column": multi,
        "period_header": take(lambda unit: "기간=" in unit["text"]),
        "unit_included": take(lambda unit: "단위=" in unit["text"]),
        "not_eligible": None if ineligible is None else _example_brief(ineligible),
    }


def _example_brief(unit: dict[str, Any]) -> dict[str, Any]:
    return {
        "canonical_ordinal": unit["canonical_ordinal"],
        "canonical_chunk_id": unit["canonical_chunk_id"],
        "retrieval_unit_id": unit["retrieval_unit_id"],
        "representation_type": unit["representation_type"],
        "split_reason": unit["split_reason"],
        "eligibility_reason": unit.get("eligibility_reason"),
        "text": unit["text"],
        "source_element_ids": unit["source_element_ids"],
    }


def cell_slices_cover_text(units: list[dict[str, Any]], cell_text: str) -> bool:
    """True when partial cell slices from one chunk reconstruct the flattened cell."""
    slices = []
    for unit in units:
        for source in unit["source_refs"]:
            for cell in source.get("cells") or []:
                if "text_char_start" not in cell:
                    continue
                slices.append((int(cell["text_char_start"]), int(cell["text_char_end"])))
    if not slices:
        return False
    slices.sort()
    if slices[0][0] != 0 or slices[-1][1] != len(cell_text):
        return False
    for left, right in zip(slices, slices[1:], strict=False):
        if left[1] != right[0]:
            return False
    rebuilt = "".join(cell_text[start:end] for start, end in slices)
    return rebuilt == cell_text


def scope_inheritance_audit(
    elements: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    *,
    scope_element_id: str,
    title_element_id: str,
    table_element_id: str,
) -> dict[str, Any]:
    """Structural facts for one preface/table pair. This does not propagate scope."""
    by_id = {element["element_id"]: element for element in elements}
    scope = by_id[scope_element_id]
    title = by_id[title_element_id]
    table = by_id[table_element_id]
    ordered = sorted(elements, key=lambda element: int(element["order_index"]))
    scope_order = int(scope["order_index"])
    table_order = int(table["order_index"])
    between = [
        _element_brief(element)
        for element in ordered
        if scope_order < int(element["order_index"]) < table_order
    ]
    following_heading = next(
        (
            _element_brief(element)
            for element in ordered
            if int(element["order_index"]) > scope_order
            and element.get("element_type") == "heading"
        ),
        None,
    )
    heading_span = []
    if following_heading is not None:
        heading_order = int(following_heading["order_index"])
        heading_span = [
            _element_brief(element)
            for element in ordered
            if scope_order < int(element["order_index"]) < heading_order
        ]
    citations = {
        "scope": _chunk_citations(chunks, scope_element_id),
        "title": _chunk_citations(chunks, title_element_id),
        "table": _chunk_citations(chunks, table_element_id),
    }
    scope_path = _locator_steps(scope)
    table_path = _locator_steps(table)
    common = 0
    for left, right in zip(scope_path, table_path, strict=False):
        if left != right:
            break
        common += 1
    return {
        "implemented_propagation": False,
        "scope_element": _element_brief(scope),
        "title_element": _element_brief(title),
        "table_element": _element_brief(table),
        "same_section_path": scope.get("section_path") == table.get("section_path"),
        "section_path": scope.get("section_path"),
        "parent_element_id": {
            "scope": scope.get("parent_element_id"),
            "table": table.get("parent_element_id"),
        },
        "xml_relation": {
            "tree": (scope.get("source_locator") or {}).get("tree"),
            "scope_path": scope_path,
            "table_path": table_path,
            "shared_prefix_length": common,
            "same_node": scope_path == table_path,
            "table_is_descendant_of_scope_node": False,
            "relation": "later_sibling_under_shared_section_not_parent_child",
        },
        "elements_between_scope_and_table": between,
        "immediate_predecessor_of_table_order": table_order - 1,
        "immediate_predecessor_is_scope": table_order - 1 == scope_order,
        "next_heading_after_scope": following_heading,
        "elements_from_scope_until_next_heading": {
            "count": len(heading_span),
            "table_count": sum(row["element_type"] == "table" for row in heading_span),
            "element_types": [row["element_type"] for row in heading_span],
        },
        "chunk_citations": citations,
        "inherited_context_candidates": [
            {
                "source_ref": title_element_id,
                "context_type": "immediate_preface_title",
                "value": title.get("text"),
                "propagation_rule": "immediate_predecessor_paragraph_to_following_table",
                "effective_range": {
                    "target": "the single following table element",
                    "element_id": table_element_id,
                },
                "deterministic": True,
                "already_implemented_by": "chunker preface link, not by this session",
                "implemented_in_this_session": False,
            },
            {
                "source_ref": scope_element_id,
                "context_type": "scope_declaration",
                "value": scope.get("text"),
                "propagation_rule": "no_explicit_structural_container",
                "effective_range": None,
                "deterministic": False,
                "implemented_in_this_session": False,
                "rejected_alternative": (
                    "Do not copy a scope word onto every later element because a "
                    "sentence contains it. The next heading is not a container for "
                    "this sentence: the span until that heading has no field that "
                    "says which tables the sentence governs."
                ),
            },
        ],
    }


def _d2a_pieces(
    chunk: dict[str, Any],
    elements_by_id: dict[str, dict[str, Any]],
    token_counter: TokenCounter,
    *,
    target_tokens: int,
    hard_max_tokens: int,
) -> list[dict[str, Any]]:
    text = chunk.get("text") or ""
    if token_counter(text) <= hard_max_tokens:
        return [_passthrough_piece(chunk, split_reason="passthrough")]
    narrative = _single_cell_narrative(chunk)
    if narrative is not None:
        pieces = _split_single_cell(
            chunk,
            narrative,
            elements_by_id,
            token_counter,
            target_tokens=target_tokens,
            hard_max_tokens=hard_max_tokens,
        )
        if pieces is not None:
            return pieces
    row_pieces = _split_table_rows(
        chunk,
        elements_by_id,
        token_counter,
        target_tokens=target_tokens,
        hard_max_tokens=hard_max_tokens,
    )
    if row_pieces is not None:
        return row_pieces
    return _split_rendered_text(
        chunk,
        elements_by_id,
        token_counter,
        target_tokens=target_tokens,
        hard_max_tokens=hard_max_tokens,
    )


def _d2b_pieces(
    chunk: dict[str, Any], elements_by_id: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    decision = _table_row_decision(chunk, elements_by_id)
    if decision is None:
        piece = _passthrough_piece(chunk, split_reason="passthrough")
        piece["eligibility_reason"] = _ineligible_reason(chunk, elements_by_id)
        return [piece]
    return [
        {
            "text": row["text"],
            "source_refs": row["source_refs"],
            "split_reason": row["split_reason"],
            "representation_type": row["representation_type"],
            "eligibility_reason": row["eligibility_reason"],
        }
        for row in decision["rows"]
    ]


def _passthrough_piece(chunk: dict[str, Any], *, split_reason: str) -> dict[str, Any]:
    return {
        "text": chunk.get("text") or "",
        "source_refs": copy.deepcopy(list(chunk.get("chunk_sources") or [])),
        "split_reason": split_reason,
        "representation_type": "canonical_text",
        "eligibility_reason": split_reason,
    }


def _single_cell_narrative(chunk: dict[str, Any]) -> dict[str, Any] | None:
    sources = list(chunk.get("chunk_sources") or [])
    table_sources = [source for source in sources if source.get("kind") == "table"]
    if len(table_sources) != 1:
        return None
    source = table_sources[0]
    cells = list(source.get("cells") or [])
    if source.get("role") != "body" or len(cells) != 1:
        return None
    if any(item.get("role") in _HEADER_ROLES for item in sources):
        return None
    return source


def _split_single_cell(
    chunk: dict[str, Any],
    source: dict[str, Any],
    elements_by_id: dict[str, dict[str, Any]],
    token_counter: TokenCounter,
    *,
    target_tokens: int,
    hard_max_tokens: int,
) -> list[dict[str, Any]] | None:
    header, body = _section_and_body(chunk)
    cells = _table_cells(source, elements_by_id)
    if len(cells) != 1:
        return None
    cell_text = _flat(cells[0]["text"])
    if body != cell_text:
        return None
    groups = _bounded_groups(
        body,
        header,
        token_counter,
        target_tokens=target_tokens,
        hard_max_tokens=hard_max_tokens,
    )
    pieces = []
    for group in groups:
        start, end = group["start"], group["end"]
        cell_ref = copy.deepcopy(source["cells"][0])
        cell_ref["text_char_start"] = start
        cell_ref["text_char_end"] = end
        cell_ref["text_char_length"] = len(cell_text)
        narrowed = copy.deepcopy(source)
        narrowed["cells"] = [cell_ref]
        pieces.append(
            {
                "text": _render(header, body[start:end]),
                "source_refs": [narrowed],
                "split_reason": group["reason"],
                "representation_type": "oversized_split",
                "eligibility_reason": "oversized_single_cell",
            }
        )
    return pieces


def _split_table_rows(
    chunk: dict[str, Any],
    elements_by_id: dict[str, dict[str, Any]],
    token_counter: TokenCounter,
    *,
    target_tokens: int,
    hard_max_tokens: int,
) -> list[dict[str, Any]] | None:
    segments = _line_segments(chunk, elements_by_id)
    if segments is None:
        return None
    body_rows = [segment for segment in segments if segment["role"] == "body"]
    if len(body_rows) < 2:
        return None
    prefix_segments = [segment for segment in segments if segment["role"] != "body"]
    header = "\n".join(segment["text"] for segment in prefix_segments if segment["text"])
    prefix_sources = [
        copy.deepcopy(segment["source"])
        for segment in prefix_segments
        if segment.get("source") is not None
    ]
    pieces: list[dict[str, Any]] = []
    short_rows: list[dict[str, Any]] = []

    def flush_short() -> None:
        if not short_rows:
            return
        pieces.extend(
            _pack_table_rows(
                short_rows,
                header,
                prefix_sources,
                token_counter,
                target_tokens=target_tokens,
            )
        )
        short_rows.clear()

    for segment in body_rows:
        line = segment["text"]
        if token_counter(_render(header, line)) <= hard_max_tokens:
            short_rows.append(segment)
            continue
        flush_short()
        pieces.extend(
            _split_long_table_row(
                segment,
                header,
                prefix_sources,
                token_counter,
                target_tokens=target_tokens,
                hard_max_tokens=hard_max_tokens,
            )
        )
    flush_short()
    return pieces


def _pack_table_rows(
    body_rows: list[dict[str, Any]],
    header: str,
    prefix_sources: list[dict[str, Any]],
    token_counter: TokenCounter,
    *,
    target_tokens: int,
) -> list[dict[str, Any]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for segment in body_rows:
        if not current:
            current = [segment]
            continue
        trial = "\n".join(item["text"] for item in current + [segment])
        if token_counter(_render(header, trial)) <= target_tokens:
            current.append(segment)
        else:
            groups.append(current)
            current = [segment]
    if current:
        groups.append(current)
    pieces = []
    for group in groups:
        body_sources = [_narrow_row_source(segment) for segment in group]
        pieces.append(
            {
                "text": _render(header, "\n".join(item["text"] for item in group)),
                "source_refs": copy.deepcopy(prefix_sources) + body_sources,
                "split_reason": "table_row_group",
                "representation_type": "oversized_split",
                "eligibility_reason": "oversized_table_rows",
            }
        )
    return pieces


def _narrow_row_source(segment: dict[str, Any]) -> dict[str, Any]:
    narrowed = copy.deepcopy(segment["source"])
    row = int(segment["row"])
    narrowed["cells"] = [
        cell for cell in narrowed.get("cells") or [] if int(cell["start_row"]) == row
    ]
    narrowed["rows"] = {"start": row, "end": row}
    narrowed["row_rects"] = [{"start": row, "end": row}]
    narrowed["rows_contiguous"] = True
    return narrowed


def _split_long_table_row(
    segment: dict[str, Any],
    header: str,
    prefix_sources: list[dict[str, Any]],
    token_counter: TokenCounter,
    *,
    target_tokens: int,
    hard_max_tokens: int,
) -> list[dict[str, Any]]:
    line = segment["text"]
    cell_spans = _joined_cell_spans(segment["cells"])
    covered = cell_spans[-1][2] if cell_spans else 0
    if covered != len(line):
        raise UsageError("table row text does not match its cell spans")
    groups = _bounded_groups(
        line,
        header,
        token_counter,
        target_tokens=target_tokens,
        hard_max_tokens=hard_max_tokens,
    )
    source = segment["source"]
    pieces = []
    for group in groups:
        cited = []
        for cell, start, end in cell_spans:
            overlap_start = max(group["start"], start)
            overlap_end = min(group["end"], end)
            if overlap_end <= overlap_start:
                continue
            ref = _matching_cell_ref(source, cell)
            if overlap_start == start and overlap_end == end:
                cited.append(copy.deepcopy(ref))
                continue
            partial = copy.deepcopy(ref)
            partial["text_char_start"] = overlap_start - start
            partial["text_char_end"] = overlap_end - start
            partial["text_char_length"] = end - start
            cited.append(partial)
        narrowed = copy.deepcopy(source)
        narrowed["cells"] = cited
        row = int(segment["row"])
        narrowed["rows"] = {"start": row, "end": row}
        narrowed["row_rects"] = [{"start": row, "end": row}]
        narrowed["rows_contiguous"] = True
        pieces.append(
            {
                "text": _render(header, line[group["start"] : group["end"]]),
                "source_refs": copy.deepcopy(prefix_sources) + [narrowed],
                "split_reason": group["reason"],
                "representation_type": "oversized_split",
                "eligibility_reason": "oversized_table_row",
            }
        )
    return pieces


def _joined_cell_spans(
    cells: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], int, int]]:
    ordered = sorted(cells, key=lambda cell: int(cell["start_col"]))
    spans = []
    cursor = 0
    for index, cell in enumerate(ordered):
        flat = _flat(cell["text"])
        spans.append((cell, cursor, cursor + len(flat)))
        cursor += len(flat)
        if index + 1 < len(ordered):
            cursor += len(" | ")
    return spans


def _matching_cell_ref(source: dict[str, Any], cell: dict[str, Any]) -> dict[str, Any]:
    for ref in source.get("cells") or []:
        if int(ref["start_row"]) == int(cell["start_row"]) and int(ref["start_col"]) == int(
            cell["start_col"]
        ):
            return ref
    raise UsageError("rendered table cell has no source ref")


def _split_rendered_text(
    chunk: dict[str, Any],
    elements_by_id: dict[str, dict[str, Any]],
    token_counter: TokenCounter,
    *,
    target_tokens: int,
    hard_max_tokens: int,
) -> list[dict[str, Any]]:
    header, body = _section_and_body(chunk)
    paragraph_sources = [
        source
        for source in chunk.get("chunk_sources") or []
        if source.get("kind") == "paragraph"
    ]
    if len(paragraph_sources) == 1:
        source = paragraph_sources[0]
        element = elements_by_id[source["element_id"]]
        raw = element.get("text") or ""
        expected = raw[int(source.get("char_start") or 0) : int(source.get("char_end") or 0)]
        if body == expected:
            groups = _bounded_groups(
                body,
                header,
                token_counter,
                target_tokens=target_tokens,
                hard_max_tokens=hard_max_tokens,
            )
            pieces = []
            base = int(source.get("char_start") or 0)
            for group in groups:
                narrowed = copy.deepcopy(source)
                narrowed["char_start"] = base + group["start"]
                narrowed["char_end"] = base + group["end"]
                pieces.append(
                    {
                        "text": _render(header, body[group["start"] : group["end"]]),
                        "source_refs": [narrowed],
                        "split_reason": group["reason"],
                        "representation_type": "oversized_split",
                        "eligibility_reason": "oversized_paragraph",
                    }
                )
            return pieces
    raise UsageError(
        "oversized chunk ordinal "
        f"{chunk.get('ordinal')} has no structural span mapping for a bounded retrieval unit"
    )


def _bounded_groups(
    body: str,
    header: str,
    token_counter: TokenCounter,
    *,
    target_tokens: int,
    hard_max_tokens: int,
) -> list[dict[str, Any]]:
    atoms = _explode_span(
        body,
        0,
        len(body),
        "numbered_heading",
        header,
        token_counter,
        hard_max_tokens=hard_max_tokens,
        first_cut=True,
    )
    if not atoms or atoms[0]["start"] != 0 or atoms[-1]["end"] != len(body):
        raise UsageError("retrieval unit split did not cover the source text")
    for left, right in zip(atoms, atoms[1:], strict=False):
        if left["end"] != right["start"]:
            raise UsageError("retrieval unit split left a gap in the source text")
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    def _count(group: list[dict[str, Any]]) -> int:
        return token_counter(_render(header, body[group[0]["start"] : group[-1]["end"]]))

    for atom in atoms:
        if token_counter(_render(header, body[atom["start"] : atom["end"]])) > hard_max_tokens:
            raise UsageError("an atomic retrieval piece still exceeds the hard token max")
        if not current:
            current = [atom]
            continue
        trial = current + [atom]
        if _count(trial) <= target_tokens:
            current = trial
        else:
            groups.append(current)
            current = [atom]
    if current:
        groups.append(current)
    packed = []
    for group in groups:
        reasons = {atom["reason"] for atom in group}
        reason = next(iter(reasons)) if len(reasons) == 1 else "packed_structural"
        packed.append(
            {"start": group[0]["start"], "end": group[-1]["end"], "reason": reason}
        )
    return packed


def _explode_span(
    body: str,
    start: int,
    end: int,
    reason: str,
    header: str,
    token_counter: TokenCounter,
    *,
    hard_max_tokens: int,
    first_cut: bool,
) -> list[dict[str, Any]]:
    piece = body[start:end]
    if token_counter(_render(header, piece)) <= hard_max_tokens and not first_cut:
        return [{"start": start, "end": end, "reason": reason}]
    splitters: list[tuple[str, list[tuple[int, int]] | None]] = []
    if first_cut:
        splitters.append(("numbered_heading", _sequence_spans(piece, _NUMBERED_RE)))
    splitters.extend(
        [
            ("subsection_marker", _sequence_spans(piece, _SUBSECTION_RE)),
            ("paragraph", _paragraph_spans(piece)),
            ("sentence", _sentence_spans(piece)),
        ]
    )
    for split_reason, spans in splitters:
        if not spans or len(spans) < 2:
            continue
        if spans[0][0] != 0 or spans[-1][1] != len(piece):
            continue
        exploded: list[dict[str, Any]] = []
        for local_start, local_end in spans:
            if local_end <= local_start:
                continue
            exploded.extend(
                _explode_span(
                    body,
                    start + local_start,
                    start + local_end,
                    split_reason,
                    header,
                    token_counter,
                    hard_max_tokens=hard_max_tokens,
                    first_cut=False,
                )
            )
        if len(exploded) >= 2 and not (
            len(exploded) == 1 and exploded[0]["start"] == start and exploded[0]["end"] == end
        ):
            return exploded
    return _token_windows(
        body,
        start,
        end,
        header,
        token_counter,
        hard_max_tokens=hard_max_tokens,
    )


def _sequence_spans(text: str, pattern: re.Pattern[str]) -> list[tuple[int, int]] | None:
    matches = []
    expected = 1
    for match in pattern.finditer(text):
        number = int(match.group(1))
        if number != expected:
            continue
        matches.append(match)
        expected += 1
    if len(matches) < 2:
        return None
    spans: list[tuple[int, int]] = []
    first = matches[0].start()
    if first > 0:
        spans.append((0, first))
    for index, match in enumerate(matches):
        span_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        spans.append((match.start(), span_end))
    if spans[0][0] != 0 or spans[-1][1] != len(text):
        return None
    for left, right in zip(spans, spans[1:], strict=False):
        if left[1] != right[0]:
            return None
    return spans


def _paragraph_spans(text: str) -> list[tuple[int, int]] | None:
    separator = "\n\n" if "\n\n" in text else "\n" if "\n" in text else ""
    if not separator:
        return None
    parts = text.split(separator)
    if len(parts) < 2:
        return None
    spans = []
    cursor = 0
    for index, part in enumerate(parts):
        span_end = cursor + len(part)
        if index < len(parts) - 1:
            span_end += len(separator)
        spans.append((cursor, span_end))
        cursor = span_end
    if cursor != len(text):
        return None
    return spans


def _sentence_spans(text: str) -> list[tuple[int, int]] | None:
    cuts = []
    for index, char in enumerate(text):
        if char not in ".?!。":
            continue
        prev = text[index - 1] if index else ""
        nxt = text[index + 1] if index + 1 < len(text) else ""
        if prev.isdigit():
            continue
        if nxt and not nxt.isspace():
            continue
        cuts.append(index + 1)
    if not cuts:
        return None
    spans = []
    cursor = 0
    for cut in cuts:
        if cut <= cursor:
            continue
        spans.append((cursor, cut))
        cursor = cut
    if cursor < len(text):
        spans.append((cursor, len(text)))
    if len(spans) < 2:
        return None
    return spans


def _token_windows(
    body: str,
    start: int,
    end: int,
    header: str,
    token_counter: TokenCounter,
    *,
    hard_max_tokens: int,
) -> list[dict[str, Any]]:
    windows = []
    cursor = start
    while cursor < end:
        lo = cursor + 1
        hi = end
        best = cursor
        while lo <= hi:
            mid = (lo + hi) // 2
            if token_counter(_render(header, body[cursor:mid])) <= hard_max_tokens:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        if best <= cursor:
            raise UsageError("token fallback cannot place a retrieval unit under the hard max")
        cut = best
        if best < end:
            window = body[cursor:best]
            whitespace = window.rfind(" ")
            if whitespace >= max(1, len(window) // 4):
                candidate = cursor + whitespace + 1
                if cursor < candidate <= best:
                    cut = candidate
        windows.append({"start": cursor, "end": cut, "reason": "token_window"})
        cursor = cut
    return windows


def _table_row_decision(
    chunk: dict[str, Any], elements_by_id: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    sources = list(chunk.get("chunk_sources") or [])
    header_sources = [source for source in sources if source.get("role") in _HEADER_ROLES]
    body_sources = [
        source
        for source in sources
        if source.get("kind") == "table" and source.get("role") == "body"
    ]
    if not header_sources or not body_sources:
        return None
    header_cells: list[dict[str, Any]] = []
    for source in header_sources:
        header_cells.extend(_table_cells(source, elements_by_id))
    labels = _leaf_labels(header_cells)
    if not any(labels.values()):
        return None
    body_cells: list[dict[str, Any]] = []
    body_source_cells: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for source in body_sources:
        materialized = _table_cells(source, elements_by_id)
        body_cells.extend(materialized)
        for cell in source.get("cells") or []:
            body_source_cells[int(cell["start_row"])].append(cell)
    by_row: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for cell in body_cells:
        by_row[int(cell["start_row"])].append(cell)
    nonempty_rows = [
        row
        for row, cells in by_row.items()
        if any(_compact(cell["text"]) for cell in cells)
    ]
    if not nonempty_rows:
        return None
    context = _context_lines(sources, elements_by_id)
    rendered_rows = []
    for row in sorted(nonempty_rows):
        source_refs = _row_source_refs(sources, body_source_cells[row], context)
        record = _row_record(by_row[row], labels, context)
        if record is None:
            rendered_rows.append(
                {
                    "text": _render(_section_header(chunk), _render_row_line(by_row[row])),
                    "source_refs": [
                        ref
                        for ref in source_refs
                        if ref.get("role") == "body"
                    ],
                    "row": row,
                    "split_reason": "table_row_residual",
                    "representation_type": "table_row_residual",
                    "eligibility_reason": "residual_non_record_row",
                }
            )
            continue
        rendered_rows.append(
            {
                "text": record,
                "source_refs": source_refs,
                "row": row,
                "split_reason": "table_row",
                "representation_type": "table_row",
                "eligibility_reason": "header_aligned_row",
            }
        )
    if not any(row["representation_type"] == "table_row" for row in rendered_rows):
        return None
    return {"rows": rendered_rows}


def _ineligible_reason(
    chunk: dict[str, Any], elements_by_id: dict[str, dict[str, Any]]
) -> str:
    if _single_cell_narrative(chunk) is not None:
        return "single_cell_narrative"
    sources = list(chunk.get("chunk_sources") or [])
    if not any(source.get("role") in _HEADER_ROLES for source in sources):
        return "no_header"
    has_body = any(
        source.get("role") == "body" and source.get("kind") == "table" for source in sources
    )
    if not has_body:
        return "no_body"
    if _table_row_decision(chunk, elements_by_id) is None:
        return "row_not_independent_record"
    return "eligible"


def _row_record(
    row_cells: list[dict[str, Any]],
    labels: dict[int, str],
    context: dict[str, list[str]],
) -> str | None:
    label_column = _label_column(row_cells, labels)
    fields = []
    for cell in sorted(row_cells, key=lambda item: int(item["start_col"])):
        column = int(cell["start_col"])
        if label_column is not None and column == label_column:
            continue
        value = _compact(cell["text"])
        if not value:
            continue
        leaf = labels.get(column, "")
        if not leaf or _norm_header(leaf) in _FORMULA_HEADERS:
            continue
        fields.append(f"{leaf}={value}")
    if not fields:
        return None
    parts: list[str] = []
    if context["titles"]:
        parts.append(" / ".join(context["titles"]))
    if label_column is not None:
        label_text = ""
        for cell in row_cells:
            if int(cell["start_col"]) == label_column:
                label_text = _compact(cell["text"])
                break
        if label_text:
            parts.append(label_text)
    parts.extend(fields)
    if context["periods"]:
        parts.append("기간=" + " / ".join(context["periods"]))
    if context["units"]:
        parts.append("단위=" + " / ".join(context["units"]))
    return " | ".join(parts)


def _label_column(row_cells: list[dict[str, Any]], labels: dict[int, str]) -> int | None:
    labeled = [
        column
        for column, label in sorted(labels.items())
        if _norm_header(label) in _LABEL_HEADERS
    ]
    if labeled:
        return labeled[0]
    nonempty = [
        cell
        for cell in sorted(row_cells, key=lambda item: int(item["start_col"]))
        if _compact(cell["text"])
    ]
    if not nonempty:
        return None
    first = nonempty[0]
    if _looks_like_amount(first["text"]):
        return None
    return int(first["start_col"])


def _row_source_refs(
    sources: list[dict[str, Any]],
    row_cell_refs: list[dict[str, Any]],
    context: dict[str, list[str]],
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    used_roles = set()
    if context["titles"]:
        used_roles.update(_TITLE_ROLES)
    if context["periods"]:
        used_roles.update(_PERIOD_ROLES)
    if context["units"]:
        used_roles.update(_UNIT_ROLES)
    for source in sources:
        role = source.get("role")
        if role in _HEADER_ROLES or role in used_roles:
            kept.append(copy.deepcopy(source))
            continue
        if role != "body" or source.get("kind") != "table":
            continue
        wanted = {
            (
                int(cell["start_row"]),
                int(cell["start_col"]),
                int(cell.get("rowspan") or 1),
                int(cell.get("colspan") or 1),
            )
            for cell in row_cell_refs
        }
        selected = [
            cell
            for cell in source.get("cells") or []
            if (
                int(cell["start_row"]),
                int(cell["start_col"]),
                int(cell.get("rowspan") or 1),
                int(cell.get("colspan") or 1),
            )
            in wanted
        ]
        if not selected:
            continue
        narrowed = copy.deepcopy(source)
        narrowed["cells"] = selected
        rows = sorted({int(cell["start_row"]) for cell in selected})
        narrowed["rows"] = {"start": rows[0], "end": rows[-1]} if len(rows) == 1 else None
        narrowed["row_rects"] = [{"start": row, "end": row} for row in rows]
        narrowed["rows_contiguous"] = len(rows) == 1
        kept.append(narrowed)
    return kept


def _context_lines(
    sources: list[dict[str, Any]], elements_by_id: dict[str, dict[str, Any]]
) -> dict[str, list[str]]:
    titles: list[str] = []
    periods: list[str] = []
    units: list[str] = []
    for source in sources:
        role = source.get("role")
        texts = _source_texts(source, elements_by_id)
        if role in _TITLE_ROLES:
            titles.extend(texts)
        elif role in _PERIOD_ROLES:
            periods.extend(texts)
        elif role in _UNIT_ROLES:
            units.extend(texts)
    return {
        "titles": _unique(titles),
        "periods": _unique(periods),
        "units": _unique(units),
    }


def _source_texts(
    source: dict[str, Any], elements_by_id: dict[str, dict[str, Any]]
) -> list[str]:
    if source.get("kind") != "table":
        element = elements_by_id[source["element_id"]]
        raw = element.get("text") or ""
        text = _compact(raw[int(source.get("char_start") or 0) : int(source.get("char_end") or 0)])
        return [text] if text else []
    cells = _table_cells(source, elements_by_id)
    return [_compact(cell["text"]) for cell in cells if _compact(cell["text"])]


def _leaf_labels(header_cells: list[dict[str, Any]]) -> dict[int, str]:
    if not header_cells:
        return {}
    last_col = max(
        int(cell["start_col"]) + int(cell.get("colspan") or 1) - 1 for cell in header_cells
    )
    rows = sorted({int(cell["start_row"]) for cell in header_cells})
    labels = {}
    for column in range(last_col + 1):
        leaf = ""
        for row in rows:
            for cell in header_cells:
                if _covers(cell, row, column) and _compact(cell["text"]):
                    leaf = _compact(cell["text"])
        labels[column] = leaf
    return labels


def _covers(cell: dict[str, Any], row: int, column: int) -> bool:
    return (
        int(cell["start_row"]) <= row < int(cell["start_row"]) + int(cell.get("rowspan") or 1)
        and int(cell["start_col"]) <= column
        < int(cell["start_col"]) + int(cell.get("colspan") or 1)
    )


def _line_segments(
    chunk: dict[str, Any], elements_by_id: dict[str, dict[str, Any]]
) -> list[dict[str, Any]] | None:
    segments: list[dict[str, Any]] = []
    path = " > ".join(chunk.get("section_path") or [])
    if path:
        segments.append(
            {"text": f"[section] {path}", "role": "section", "source": None, "row": None}
        )
    for source in chunk.get("chunk_sources") or []:
        if source.get("kind") != "table":
            element = elements_by_id[source["element_id"]]
            raw = element.get("text") or ""
            text = raw[int(source.get("char_start") or 0) : int(source.get("char_end") or 0)]
            if text:
                segments.append(
                    {"text": text, "role": source.get("role"), "source": source, "row": None}
                )
            continue
        cells = _table_cells(source, elements_by_id)
        rows = sorted({int(cell["start_row"]) for cell in cells})
        by_row: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for cell in cells:
            by_row[int(cell["start_row"])].append(cell)
        for row in rows:
            ordered = sorted(by_row[row], key=lambda cell: int(cell["start_col"]))
            line = " | ".join(_flat(cell["text"]) for cell in ordered)
            if not line:
                continue
            segments.append(
                {
                    "text": line,
                    "role": source.get("role"),
                    "source": source,
                    "row": row,
                    "cells": ordered,
                }
            )
    rendered = "\n".join(segment["text"] for segment in segments if segment["text"]).strip()
    if rendered != (chunk.get("text") or ""):
        return None
    return segments


def _section_header(chunk: dict[str, Any]) -> str:
    path = " > ".join(chunk.get("section_path") or [])
    return f"[section] {path}" if path else ""


def _render_row_line(row_cells: list[dict[str, Any]]) -> str:
    ordered = sorted(row_cells, key=lambda cell: int(cell["start_col"]))
    return " | ".join(_flat(cell["text"]) for cell in ordered)


def _section_and_body(chunk: dict[str, Any]) -> tuple[str, str]:
    path = " > ".join(chunk.get("section_path") or [])
    header = f"[section] {path}" if path else ""
    text = chunk.get("text") or ""
    if header and text.startswith(header + "\n"):
        return header, text[len(header) + 1 :]
    if header and text == header:
        return header, ""
    return "", text


def _render(header: str, body: str) -> str:
    if header and body:
        return f"{header}\n{body}"
    return header or body


def _finish_pieces(
    chunk: dict[str, Any],
    pieces: list[dict[str, Any]],
    *,
    version: str,
    token_counter: TokenCounter,
    model_max_tokens: int,
    hard_max_tokens: int | None,
) -> list[dict[str, Any]]:
    units = []
    for index, piece in enumerate(pieces):
        text = piece["text"]
        digest = sha256_text(text)
        source_refs = piece["source_refs"]
        identity = {
            "representation_version": version,
            "canonical_chunk_id": chunk["chunk_id"],
            "unit_index": index,
            "text_hash": digest,
            "source_refs": source_refs,
        }
        token_count = int(token_counter(text))
        if hard_max_tokens is not None and token_count > model_max_tokens:
            raise UsageError(
                f"D2a retrieval unit for ordinal {chunk.get('ordinal')} still exceeds "
                f"the embedding model limit ({token_count} > {model_max_tokens})"
            )
        if hard_max_tokens is not None and token_count > hard_max_tokens:
            raise UsageError(
                f"D2a unit for ordinal {chunk.get('ordinal')} exceeds hard max "
                f"({token_count} > {hard_max_tokens})"
            )
        element_ids = []
        for source in source_refs:
            element_id = source.get("element_id")
            if element_id and element_id not in element_ids:
                element_ids.append(element_id)
        units.append(
            {
                "retrieval_unit_id": sha256_text(canonical_dumps(identity)),
                "canonical_chunk_id": chunk["chunk_id"],
                "canonical_ordinal": int(chunk["ordinal"]),
                "unit_index": index,
                "representation_version": version,
                "representation_type": piece["representation_type"],
                "representation_hash": digest,
                "text": text,
                "token_count": token_count,
                "embedding_truncated": token_count > model_max_tokens,
                "embedding_model_max_tokens": model_max_tokens,
                "source_refs": source_refs,
                "source_element_ids": element_ids,
                "split_reason": piece["split_reason"],
                "eligibility_reason": piece.get("eligibility_reason"),
                "section_path": list(chunk.get("section_path") or []),
            }
        )
    return units


def _span_brief(unit: dict[str, Any]) -> dict[str, Any]:
    spans = []
    for source in unit["source_refs"]:
        brief: dict[str, Any] = {
            "element_id": source.get("element_id"),
            "kind": source.get("kind"),
            "role": source.get("role"),
            "char_start": source.get("char_start"),
            "char_end": source.get("char_end"),
        }
        cells = []
        for cell in source.get("cells") or []:
            item = {
                "start_row": cell.get("start_row"),
                "start_col": cell.get("start_col"),
            }
            if "text_char_start" in cell:
                item["text_char_start"] = cell["text_char_start"]
                item["text_char_end"] = cell["text_char_end"]
                item["text_char_length"] = cell.get("text_char_length")
            cells.append(item)
        brief["cells"] = cells
        spans.append(brief)
    return {
        "unit_index": unit["unit_index"],
        "retrieval_unit_id": unit["retrieval_unit_id"],
        "token_count": unit["token_count"],
        "split_reason": unit["split_reason"],
        "source_spans": spans,
    }


def _element_brief(element: dict[str, Any]) -> dict[str, Any]:
    table = element.get("table") or {}
    return {
        "element_id": element["element_id"],
        "order_index": int(element["order_index"]),
        "element_type": element.get("element_type"),
        "section_path": element.get("section_path"),
        "text_preview": _compact(element.get("text") or "")[:120],
        "row_count": table.get("row_count"),
        "col_count": table.get("col_count"),
    }


def _chunk_citations(chunks: list[dict[str, Any]], element_id: str) -> list[dict[str, Any]]:
    found = []
    for chunk in chunks:
        roles = [
            source.get("role")
            for source in chunk.get("chunk_sources") or []
            if source.get("element_id") == element_id
        ]
        if not roles:
            continue
        found.append(
            {
                "ordinal": chunk.get("ordinal"),
                "chunk_id": chunk.get("chunk_id"),
                "roles": roles,
            }
        )
    return found


def _locator_steps(element: dict[str, Any]) -> list[dict[str, Any]]:
    locator = element.get("source_locator") or {}
    return [
        {"tag": step.get("tag"), "index": step.get("index")}
        for step in locator.get("path") or []
    ]


def _nearest(ordered: list[int], percent: float) -> int:
    if not ordered:
        return 0
    index = max(0, math.ceil(percent / 100 * len(ordered)) - 1)
    return ordered[index]


def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _norm_header(text: str) -> str:
    return re.sub(r"\s+", "", _compact(text))


def _looks_like_amount(text: str) -> bool:
    flat = _compact(text)
    return bool(flat) and _AMOUNT_VALUE_RE.fullmatch(flat) is not None and any(
        char.isdigit() for char in flat
    )


def _unique(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    kept = []
    for line in lines:
        if not line or line in seen:
            continue
        seen.add(line)
        kept.append(line)
    return kept

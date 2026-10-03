"""Retrieval units projected from immutable canonical chunks.

A retrieval unit is the string the encoder searches. It is not a new canonical
chunk and it is not the context unit a later answer would cite. Gold stays on
source coordinates. Callers map a unit back through ``source_refs``.
"""

from __future__ import annotations

import copy
import math
import re
from collections import Counter, defaultdict
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
D2C_VERSION = "retrieval-unit-d2c-v1"

# The preserved D2b run changed representation and candidate granularity together.
# It is not evidence for a representation-only effect. Artifacts stay in place.
D2B_ORIGINAL_HYPOTHESIS_VALIDITY = {
    "recommended_name": "D2b-row-atomic-confounded",
    "preserved_run": "evaluation/runs/isu_petasys_v0_3_dense_d2b",
    "artifacts_overwritten": False,
    "experimental_validity_for_original_hypothesis": False,
    "original_hypothesis": (
        "Holding the retrieval unit and the candidate corpus fixed, "
        "what happens to BGE-M3 dense retrieval if only the table "
        "representation changes to a compact deterministic form?"
    ),
    "actual_simultaneous_changes": [
        "table_chunk_decomposed_into_row_level_retrieval_units",
        "retrieval_unit_count_increased",
        "candidate_granularity_changed",
        "compact_serialization_applied",
    ],
    "confounders": [
        "retrieval_granularity_changed",
        "candidate_count_changed",
        "duplicate_same_source_competition",
    ],
    "not_evidence_for": (
        "the effect of compact table representation at a fixed retrieval granularity"
    ),
}

_HEADER_ROLES = {"header", "repeated_header"}
_UNIT_ROLES = {"unit", "repeated_unit", "context_unit"}
_TITLE_ROLES = {"context_title"}
_PERIOD_ROLES = {"context_period"}
_LABEL_HEADERS = {"구분", "항목", "과목", "계정과목", "계정"}
_FORMULA_HEADERS = {"산식", "계산식", "공식", "산출식"}
_SCOPE_ROLES = {"context_scope"}
_COMPACT_KNOWN_ROLES = (
    _HEADER_ROLES
    | _UNIT_ROLES
    | _TITLE_ROLES
    | _PERIOD_ROLES
    | _SCOPE_ROLES
    | {"body"}
)
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
    if mode == "d2c":
        units, _baseline, _report = project_representation_only(
            chunks,
            elements_by_id,
            token_counter=token_counter,
            target_tokens=target_tokens,
            hard_max_tokens=hard_max_tokens,
            model_max_tokens=model_max_tokens,
        )
        return units
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


def project_representation_only(
    chunks: list[dict[str, Any]],
    elements_by_id: dict[str, dict[str, Any]],
    *,
    token_counter: TokenCounter,
    target_tokens: int = RETRIEVAL_UNIT_TARGET_TOKENS,
    hard_max_tokens: int = RETRIEVAL_UNIT_HARD_MAX_TOKENS,
    model_max_tokens: int = EMBEDDING_MODEL_MAX_TOKENS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Rewrite eligible D2a table text without adding or removing a candidate.

    One D2a retrieval unit stays one unit. A newline inside the new text is
    serialization syntax, not a candidate boundary.
    """
    baseline = build_retrieval_units(
        chunks,
        elements_by_id,
        mode="d2a",
        token_counter=token_counter,
        target_tokens=target_tokens,
        hard_max_tokens=hard_max_tokens,
        model_max_tokens=model_max_tokens,
    )
    rewritten = apply_compact_table_representation(
        baseline,
        elements_by_id,
        token_counter=token_counter,
        hard_max_tokens=hard_max_tokens,
        model_max_tokens=model_max_tokens,
    )
    assert_representation_only_topology(baseline, rewritten)
    report = representation_change_summary(baseline, rewritten)
    if not report["retrieval_unit_count_matches_baseline"]:
        raise UsageError("representation-only run changed the retrieval unit count")
    return rewritten, baseline, report


def apply_compact_table_representation(
    baseline: list[dict[str, Any]],
    elements_by_id: dict[str, dict[str, Any]],
    *,
    token_counter: TokenCounter,
    hard_max_tokens: int,
    model_max_tokens: int,
) -> list[dict[str, Any]]:
    """Return a 1:1 copy of ``baseline`` with eligible table text replaced."""
    rewritten = []
    for unit in baseline:
        decision = _compact_table_decision(
            unit,
            elements_by_id,
            token_counter,
            hard_max_tokens=hard_max_tokens,
            model_max_tokens=model_max_tokens,
        )
        clone = copy.deepcopy(unit)
        clone["baseline_representation_hash"] = unit["representation_hash"]
        clone["baseline_representation_version"] = unit["representation_version"]
        clone["representation_eligibility"] = decision["reason"]
        clone["omitted_context_scope"] = decision["omitted_context_scope"]
        clone["representation_changed"] = decision["text"] is not None
        if decision["text"] is None:
            rewritten.append(clone)
            continue
        text = decision["text"]
        clone["text"] = text
        clone["representation_hash"] = sha256_text(text)
        clone["representation_version"] = D2C_VERSION
        clone["representation_type"] = "compact_table"
        clone["token_count"] = decision["token_count"]
        clone["embedding_truncated"] = decision["token_count"] > model_max_tokens
        if clone["token_count"] > hard_max_tokens or clone["embedding_truncated"]:
            raise UsageError(
                "compact table text exceeded the retrieval unit limit "
                f"ordinal={unit.get('canonical_ordinal')} tokens={clone['token_count']}"
            )
        rewritten.append(clone)
    return rewritten


def assert_representation_only_topology(
    baseline: list[dict[str, Any]], rewritten: list[dict[str, Any]]
) -> None:
    """Fail unless the candidate identity and source spans match D2a."""
    if len(baseline) != len(rewritten):
        raise UsageError(
            "representation-only topology changed retrieval unit count: "
            f"{len(baseline)} -> {len(rewritten)}"
        )
    baseline_ids = [unit["retrieval_unit_id"] for unit in baseline]
    rewritten_ids = [unit["retrieval_unit_id"] for unit in rewritten]
    if baseline_ids != rewritten_ids or set(baseline_ids) != set(rewritten_ids):
        raise UsageError("representation-only run changed retrieval_unit_id values")
    for left, right in zip(baseline, rewritten, strict=True):
        for key in ("canonical_chunk_id", "canonical_ordinal", "unit_index", "split_reason"):
            if left[key] != right[key]:
                raise UsageError(
                    f"representation-only run changed {key} "
                    f"for retrieval unit {left['retrieval_unit_id']}"
                )
        if left["source_refs"] != right["source_refs"]:
            raise UsageError(
                "representation-only run changed source spans "
                f"for retrieval unit {left['retrieval_unit_id']}"
            )
        if right["text"] == left["text"]:
            if right["representation_hash"] != left["representation_hash"]:
                raise UsageError("unchanged text changed representation_hash")
            if right["representation_version"] != left["representation_version"]:
                raise UsageError("unchanged text changed representation_version")
            continue
        if right["representation_version"] != D2C_VERSION:
            raise UsageError("changed table text kept the D2a representation version")
        if right["representation_hash"] == left["representation_hash"]:
            raise UsageError("changed table text kept the D2a representation hash")
        if right.get("retrieval_unit_id") != left["retrieval_unit_id"]:
            raise UsageError("changed table text minted a new retrieval unit id")


def representation_change_summary(
    baseline: list[dict[str, Any]], rewritten: list[dict[str, Any]]
) -> dict[str, Any]:
    reasons: dict[str, int] = defaultdict(int)
    changed_chunks: set[str] = set()
    changed_tables: set[str] = set()
    changed = 0
    passthrough = 0
    omitted_scope = 0
    for left, right in zip(baseline, rewritten, strict=True):
        reason = str(right.get("representation_eligibility") or "missing")
        reasons[reason] += 1
        if right.get("omitted_context_scope"):
            omitted_scope += 1
        if right["text"] == left["text"]:
            passthrough += 1
            continue
        changed += 1
        changed_chunks.add(right["canonical_chunk_id"])
        for source in right["source_refs"]:
            if source.get("kind") == "table" and source.get("role") in {"body", "repeated_body"}:
                changed_tables.add(source["element_id"])
    return {
        "retrieval_unit_count": len(rewritten),
        "baseline_retrieval_unit_count": len(baseline),
        "retrieval_unit_count_matches_baseline": len(rewritten) == len(baseline),
        "changed_retrieval_unit_count": changed,
        "passthrough_retrieval_unit_count": passthrough,
        "changed_canonical_chunk_count": len(changed_chunks),
        "changed_table_element_count": len(changed_tables),
        "eligibility_reason_counts": dict(sorted(reasons.items())),
        "units_with_omitted_context_scope": omitted_scope,
        "independent_variable": "table_retrieval_unit_text",
        "fixed": [
            "retrieval_unit_count",
            "retrieval_unit_id",
            "canonical_chunk_id",
            "source_refs",
            "unit_index",
            "split_reason",
        ],
    }


def compact_representation_examples(
    units: list[dict[str, Any]], *, limit: int = 5
) -> list[dict[str, Any]]:
    """Deterministic real serializations. Selection does not read gold ids."""
    changed = [unit for unit in units if unit.get("representation_changed")]

    def take(predicate: Any) -> dict[str, Any] | None:
        found = next((unit for unit in changed if predicate(unit)), None)
        if found is None:
            return None
        return _compact_example(found)

    selected = [
        take(lambda unit: unit["text"].count("\n") >= 2 and "단위=" in unit["text"]),
        take(lambda unit: "기간=" in unit["text"] and "단위=" in unit["text"]),
        take(lambda unit: "단위=" in unit["text"]),
        take(lambda unit: "기간=" in unit["text"]),
        take(lambda unit: unit["text"].count("\n") >= 3),
        take(lambda unit: "=" in unit["text"] and "\n" not in unit["text"]),
        take(lambda unit: " | " in unit["text"]),
    ]
    examples = []
    seen: set[str] = set()
    for item in selected:
        if item is None or item["retrieval_unit_id"] in seen:
            continue
        seen.add(item["retrieval_unit_id"])
        examples.append(item)
        if len(examples) >= limit:
            return examples
    for unit in changed:
        if unit["retrieval_unit_id"] in seen:
            continue
        seen.add(unit["retrieval_unit_id"])
        examples.append(_compact_example(unit))
        if len(examples) >= limit:
            break
    return examples


def _compact_example(unit: dict[str, Any]) -> dict[str, Any]:
    text = unit["text"]
    stored = text if len(text) <= 1200 else text[:1200]
    return {
        "canonical_ordinal": unit["canonical_ordinal"],
        "retrieval_unit_id": unit["retrieval_unit_id"],
        "representation_type": unit["representation_type"],
        "representation_eligibility": unit.get("representation_eligibility"),
        "split_reason": unit["split_reason"],
        "token_count": unit["token_count"],
        "text_chars": len(text),
        "text_truncated_for_report": len(text) > 1200,
        "text": stored,
    }


def _compact_table_decision(
    unit: dict[str, Any],
    elements_by_id: dict[str, dict[str, Any]],
    token_counter: TokenCounter,
    *,
    hard_max_tokens: int,
    model_max_tokens: int,
) -> dict[str, Any]:
    """Decide whether this one unit can be rewritten. Never splits the unit."""
    empty = {
        "reason": "not_table",
        "text": None,
        "token_count": None,
        "omitted_context_scope": False,
    }
    sources = list(unit.get("source_refs") or [])
    if _has_partial_cell(sources):
        return {**empty, "reason": "partial_cell_slice"}
    table_body = [
        source
        for source in sources
        if source.get("kind") == "table" and source.get("role") == "body"
    ]
    if not table_body:
        return empty
    if _has_mixed_paragraph(sources):
        return {**empty, "reason": "mixed_paragraph_body"}
    unknown = _unknown_source_role(sources)
    if unknown is not None:
        return {**empty, "reason": f"unhandled_source_role:{unknown}"}
    omitted_scope = _has_nonempty_role(sources, elements_by_id, _SCOPE_ROLES)
    header_sources = [source for source in sources if source.get("role") in _HEADER_ROLES]
    if not header_sources:
        return {**empty, "reason": "no_header", "omitted_context_scope": omitted_scope}
    header_cells = _materialize_sources(header_sources, elements_by_id)
    body_cells = _materialize_sources(table_body, elements_by_id)
    if _origin_conflict(header_cells):
        return {
            **empty,
            "reason": "conflicting_header_cell",
            "omitted_context_scope": omitted_scope,
        }
    if not _header_is_reliable(header_cells):
        return {**empty, "reason": "header_not_reliable", "omitted_context_scope": omitted_scope}
    column_cells, record_cells = _split_header_cells(header_cells)
    paths = _header_paths(column_cells)
    if not any(paths.values()):
        return {**empty, "reason": "no_leaf_header", "omitted_context_scope": omitted_scope}
    rendered = _render_compact_table(
        unit,
        sources,
        elements_by_id,
        body_cells,
        paths,
        record_cells,
    )
    if rendered is None:
        return {**empty, "reason": "table_not_lossless", "omitted_context_scope": omitted_scope}
    text, reason = rendered
    if reason != "eligible_compact_table":
        return {**empty, "reason": reason, "omitted_context_scope": omitted_scope}
    if text == (unit.get("text") or ""):
        return {
            "reason": "compact_identical_to_baseline",
            "text": None,
            "token_count": None,
            "omitted_context_scope": omitted_scope,
        }
    token_count = int(token_counter(text))
    if token_count > hard_max_tokens or token_count > model_max_tokens:
        return {
            "reason": "compact_exceeds_hard_max",
            "text": None,
            "token_count": None,
            "omitted_context_scope": omitted_scope,
        }
    return {
        "reason": "eligible_compact_table",
        "text": text,
        "token_count": token_count,
        "omitted_context_scope": omitted_scope,
    }


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
        occupancy = Counter(identities)
        report[str(k)] = {
            "retrieval_unit_slots": len(window),
            "unique_canonical_chunks": unique,
            "duplicate_same_source_ru_count": len(window) - unique,
            "max_same_canonical_occupancy": max(occupancy.values()) if occupancy else 0,
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
        if "cells" not in segment:
            pieces.extend(
                _split_long_body_line(
                    segment,
                    header,
                    prefix_sources,
                    token_counter,
                    target_tokens=target_tokens,
                    hard_max_tokens=hard_max_tokens,
                )
            )
            continue
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
    row = segment.get("row")
    # A non-table body line packed beside table rows has no row index.
    # Keep its source span. Do not invent a row, and do not drop the line.
    if row is None:
        return narrowed
    row = int(row)
    narrowed["cells"] = [
        cell for cell in narrowed.get("cells") or [] if int(cell["start_row"]) == row
    ]
    narrowed["rows"] = {"start": row, "end": row}
    narrowed["row_rects"] = [{"start": row, "end": row}]
    narrowed["rows_contiguous"] = True
    return narrowed


def _split_long_body_line(
    segment: dict[str, Any],
    header: str,
    prefix_sources: list[dict[str, Any]],
    token_counter: TokenCounter,
    *,
    target_tokens: int,
    hard_max_tokens: int,
) -> list[dict[str, Any]]:
    """Split a non-table body line that was packed with table rows."""
    line = segment["text"]
    source = segment.get("source")
    if not isinstance(source, dict):
        raise UsageError("oversized body line has no source span")
    groups = _bounded_groups(
        line,
        header,
        token_counter,
        target_tokens=target_tokens,
        hard_max_tokens=hard_max_tokens,
    )
    base = int(source.get("char_start") or 0)
    pieces = []
    for group in groups:
        narrowed = copy.deepcopy(source)
        narrowed["char_start"] = base + group["start"]
        narrowed["char_end"] = base + group["end"]
        pieces.append(
            {
                "text": _render(header, line[group["start"] : group["end"]]),
                "source_refs": copy.deepcopy(prefix_sources) + [narrowed],
                "split_reason": group["reason"],
                "representation_type": "oversized_split",
                "eligibility_reason": "oversized_body_line",
            }
        )
    return pieces


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
    """Split one oversized span. An explicit stack keeps long filings off the call stack.

    The first span of a numbered or sentence cut can itself stay over the hard
    max for hundreds of levels. Recursion hit Python's stack on a half-year
    report and died inside the token counter.
    """
    pending: list[tuple[int, int, str, bool]] = [(start, end, reason, first_cut)]
    leaves: list[dict[str, Any]] = []
    while pending:
        start, end, reason, first_cut = pending.pop()
        piece = body[start:end]
        if not piece:
            continue
        if token_counter(_render(header, piece)) <= hard_max_tokens and not first_cut:
            leaves.append({"start": start, "end": end, "reason": reason})
            continue
        chosen = _shrinking_children(piece, first_cut=first_cut)
        if chosen:
            for local_start, local_end, split_reason in reversed(chosen):
                pending.append(
                    (start + local_start, start + local_end, split_reason, False)
                )
            continue
        leaves.extend(
            _token_windows(
                body,
                start,
                end,
                header,
                token_counter,
                hard_max_tokens=hard_max_tokens,
            )
        )
    return leaves


def _shrinking_children(
    piece: str, *, first_cut: bool
) -> list[tuple[int, int, str]] | None:
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
    width = len(piece)
    for split_reason, spans in splitters:
        if not spans or len(spans) < 2:
            continue
        if spans[0][0] != 0 or spans[-1][1] != width:
            continue
        children: list[tuple[int, int, str]] = []
        for local_start, local_end in spans:
            if local_end <= local_start:
                continue
            if local_end - local_start >= width:
                children = []
                break
            children.append((local_start, local_end, split_reason))
        if len(children) >= 2:
            return children
    return None


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


def _has_partial_cell(sources: list[dict[str, Any]]) -> bool:
    for source in sources:
        for cell in source.get("cells") or []:
            if "text_char_start" in cell or "text_char_end" in cell:
                return True
    return False


def _has_mixed_paragraph(sources: list[dict[str, Any]]) -> bool:
    return any(
        source.get("kind") != "table" and source.get("role") == "body" for source in sources
    )


def _unknown_source_role(sources: list[dict[str, Any]]) -> str | None:
    for source in sources:
        role = source.get("role")
        if role not in _COMPACT_KNOWN_ROLES:
            return str(role)
    return None


def _has_nonempty_role(
    sources: list[dict[str, Any]],
    elements_by_id: dict[str, dict[str, Any]],
    roles: set[str],
) -> bool:
    for source in sources:
        if source.get("role") not in roles:
            continue
        if _source_texts(source, elements_by_id):
            return True
    return False


def _materialize_sources(
    sources: list[dict[str, Any]], elements_by_id: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for source in sources:
        element = elements_by_id[source["element_id"]]
        catalog = {
            (int(cell["start_row"]), int(cell["start_col"])): cell
            for cell in (element.get("table") or {}).get("cells") or []
        }
        for cell in _table_cells(source, elements_by_id):
            origin = catalog.get((int(cell["start_row"]), int(cell["start_col"])), {})
            cell["header_role"] = origin.get("header_role")
            key = (
                int(cell["start_row"]),
                int(cell["start_col"]),
                int(cell["rowspan"]),
                int(cell["colspan"]),
                _compact(cell["text"]),
            )
            if key in seen:
                continue
            seen.add(key)
            cells.append(cell)
    return cells


def _header_is_reliable(header_cells: list[dict[str, Any]]) -> bool:
    """A header band is a column header, not the first data row.

    TH cells, a real span, or an explicit label header are structural evidence.
    A flat inferred TD row is not enough: the chunker can park the first record there.
    """
    if any(cell.get("header_role") == "header" for cell in header_cells):
        return True
    if any(
        int(cell.get("colspan") or 1) > 1 or int(cell.get("rowspan") or 1) > 1
        for cell in header_cells
    ):
        return True
    return any(_norm_header(cell.get("text") or "") in _LABEL_HEADERS for cell in header_cells)


def _split_header_cells(
    header_cells: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Separate column titles from data rows that share the header role."""
    rows = sorted({int(cell["start_row"]) for cell in header_cells})
    by_row: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for cell in header_cells:
        by_row[int(cell["start_row"])].append(cell)
    column_cells = list(by_row[rows[0]])
    record_cells: list[dict[str, Any]] = []
    for row in rows[1:]:
        if _is_subheader_row(by_row[row], column_cells):
            column_cells.extend(by_row[row])
            continue
        record_cells.extend(by_row[row])
    return column_cells, record_cells


def _is_subheader_row(row_cells: list[dict[str, Any]], column_cells: list[dict[str, Any]]) -> bool:
    nonempty = [cell for cell in row_cells if _compact(cell["text"])]
    if not nonempty:
        return True
    for cell in nonempty:
        if any(
            int(existing["start_row"]) == int(cell["start_row"])
            and int(existing["start_col"]) == int(cell["start_col"])
            for existing in column_cells
        ):
            continue
        if not _has_colspan_parent(cell, column_cells):
            return False
    return True


def _has_colspan_parent(cell: dict[str, Any], column_cells: list[dict[str, Any]]) -> bool:
    start = int(cell["start_col"])
    span = int(cell.get("colspan") or 1)
    for column in range(start, start + span):
        found = False
        for ancestor in column_cells:
            if int(ancestor["start_row"]) >= int(cell["start_row"]):
                continue
            if int(ancestor.get("colspan") or 1) <= 1:
                continue
            origin = int(ancestor["start_col"])
            if origin <= column < origin + int(ancestor.get("colspan") or 1):
                found = True
                break
        if not found:
            return False
    return True


def _origin_conflict(cells: list[dict[str, Any]]) -> bool:
    seen: dict[tuple[int, int], str] = {}
    for cell in cells:
        origin = (int(cell["start_row"]), int(cell["start_col"]))
        text = _compact(cell["text"])
        if origin in seen and seen[origin] != text:
            return True
        seen[origin] = text
    return False


def _header_paths(header_cells: list[dict[str, Any]]) -> dict[int, str]:
    if not header_cells:
        return {}
    rows = sorted({int(cell["start_row"]) for cell in header_cells})
    last_col = max(
        int(cell["start_col"]) + int(cell.get("colspan") or 1) - 1 for cell in header_cells
    )
    labels: dict[int, str] = {}
    for column in range(last_col + 1):
        parts: list[str] = []
        for row in rows:
            for cell in header_cells:
                if not _covers(cell, row, column):
                    continue
                text = _compact(cell["text"])
                if text and (not parts or parts[-1] != text):
                    parts.append(text)
        labels[column] = " / ".join(parts)
    return labels


def _explicit_label_column(paths: dict[int, str]) -> int | None:
    for column, path in sorted(paths.items()):
        leaf = path.split(" / ")[-1] if path else ""
        if _norm_header(leaf) in _LABEL_HEADERS or _norm_header(path) in _LABEL_HEADERS:
            return column
    if paths and not paths.get(0) and any(path for column, path in paths.items() if column != 0):
        return 0
    return None


def _render_compact_row(
    nonempty: list[dict[str, Any]], paths: dict[int, str], label_column: int | None
) -> tuple[str | None, str]:
    if len(nonempty) == 1 and int(nonempty[0].get("colspan") or 1) > 1:
        return _compact(nonempty[0]["text"]), "ok"
    label = ""
    fields: list[str] = []
    for cell in nonempty:
        text = _compact(cell["text"])
        start = int(cell["start_col"])
        span = int(cell.get("colspan") or 1)
        covered: list[str] = []
        for column in range(start, start + span):
            path = paths.get(column, "")
            if path and path not in covered:
                covered.append(path)
        if len(covered) > 1:
            return None, "ambiguous_colspan"
        path = covered[0] if covered else ""
        if label_column is not None and start == label_column and span == 1:
            if paths.get(start) == "" and _looks_like_amount(text):
                return None, "unlabeled_body_cell"
            label = text
            continue
        if not path:
            return None, "unlabeled_body_cell"
        fields.append(f"{path}={text}")
    if label and fields:
        return " | ".join([label, *fields]), "ok"
    if label:
        return label, "ok"
    if fields:
        return " | ".join(fields), "ok"
    return None, "unlabeled_body_cell"


def _render_compact_table(
    unit: dict[str, Any],
    sources: list[dict[str, Any]],
    elements_by_id: dict[str, dict[str, Any]],
    body_cells: list[dict[str, Any]],
    paths: dict[int, str],
    record_cells: list[dict[str, Any]] | None = None,
) -> tuple[str, str] | None:
    del unit
    if _origin_conflict(body_cells) or _origin_conflict(record_cells or []):
        return "", "conflicting_body_cell"
    by_row: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for cell in [*(record_cells or []), *body_cells]:
        by_row[int(cell["start_row"])].append(cell)
    label_column = _explicit_label_column(paths)
    row_lines: list[str] = []
    nonempty_count = 0
    for row in sorted(by_row):
        ordered = sorted(by_row[row], key=lambda cell: int(cell["start_col"]))
        nonempty = [cell for cell in ordered if _compact(cell["text"])]
        nonempty_count += len(nonempty)
        if not nonempty:
            continue
        line, status = _render_compact_row(nonempty, paths, label_column)
        if status != "ok" or line is None:
            return "", status
        row_lines.append(line)
    if nonempty_count == 0 or not row_lines:
        return "", "no_nonempty_body"
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
    parts: list[str] = []
    parts.extend(_unique(titles))
    parts.extend(row_lines)
    parts.extend(f"기간={text}" for text in _unique(periods))
    parts.extend(f"단위={text}" for text in _unique(units))
    text = "\n".join(parts)
    for cell in [*(record_cells or []), *body_cells]:
        copied = _compact(cell["text"])
        if copied and copied not in text:
            return "", "table_not_lossless"
    for copied in [*_unique(titles), *_unique(periods), *_unique(units)]:
        if copied not in text:
            return "", "table_not_lossless"
    return text, "eligible_compact_table"


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

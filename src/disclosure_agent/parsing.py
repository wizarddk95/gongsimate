"""Parse a fixed DART original ZIP into heading, paragraph, and table elements.

Supported input is dart4 XML (``DOCUMENT``) as stored by the sync layer.
HTML members are recorded as unsupported until a real sample is observed.
Preprocessing escapes only confirmed literal angles (for example ``<신설>``)
and bare ``&`` outside existing entities. Unknown XML tags stay in the tree
and make the member partial; they are not rewritten into text and then
reported as success. A failed strict parse is not repaired into a clean success.

``COMPLETE`` means the output files were published. ``overall_status`` is parse
quality. Exit code 0 for ``partial`` is not a full success. A later search
pipeline must accept only ``success`` until a separate quality policy exists.
The manifest is that record. Human-readable CLI text is not.

Locator paths live on the tree context of one ZIP member. ``parse_receipt``,
``resolve_locator``, and ``element_text_at`` do not read a process-global path
map. Their signatures are unchanged. ``resolve_locator`` walks ``path`` on a
private tree of the member text the caller passes. Same-named members are
distinguished by ``member_index`` on the locator.

Publication of one result directory is serialized with a local exclusive-create
lock. The collection sync lock is not used.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import time
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4
from xml.etree import ElementTree as ET

from disclosure_agent.dart import validate_zip
from disclosure_agent.errors import (
    DartZipError,
    IngestError,
    ParseLockTimeout,
    SizeLimitError,
    UsageError,
)
from disclosure_agent.storage import ORIGINAL_FILENAME, sha256_bytes

PARSER_VERSION = "2"
PREPROCESS_VERSION = "3"
SCHEMA_VERSION = 2
TREE_NAME = "dart4-xml-elementtree"
# Search may read a published result only when quality is success.
# COMPLETE and exit code 0 do not make partial eligible.
SEARCH_ELIGIBLE_STATUSES = frozenset({"success"})
# Unicode code points in the decoded member text, before preprocessing.
# Not a ZIP byte offset and not a locator path.
PREPROCESS_LOCATION = "unicode_code_point_offset_in_decoded_member_text"
MAX_MEMBER_BYTES = 30 * 1024 * 1024
MAX_ELEMENTS = 200_000
MAX_DEPTH = 80
# A single oversized ROWSPAN/COLSPAN must not allocate a dense grid.
MAX_ROWSPAN = 1_000
MAX_COLSPAN = 100
MAX_TABLE_CELLS = 50_000

HEADING_TAGS = frozenset({"COVER-TITLE", "TITLE"})
CELL_TAGS = frozenset({"TD", "TH", "TU", "TE"})
CONTAINER_TAGS = frozenset(
    {
        "DOCUMENT",
        "BODY",
        "COVER",
        "SECTION-1",
        "SECTION-2",
        "SECTION-3",
        "LIBRARY",
        "TABLE-GROUP",
        "SUMMARY",
    }
)
KNOWN_TAGS = frozenset(
    {
        "A",
        "BODY",
        "COL",
        "COLGROUP",
        "COMPANY-NAME",
        "COVER",
        "COVER-TITLE",
        "DOCUMENT",
        "DOCUMENT-NAME",
        "EXTRACTION",
        "FORMULA-VERSION",
        "IMAGE",
        "IMG",
        "IMG-CAPTION",
        "LIBRARY",
        "P",
        "PGBRK",
        "SECTION-1",
        "SECTION-2",
        "SECTION-3",
        "SPAN",
        "SUMMARY",
        "TABLE",
        "TABLE-GROUP",
        "TBODY",
        "TD",
        "TE",
        "TH",
        "THEAD",
        "TITLE",
        "TR",
        "TU",
        # Correction notice wrapper. Parsing it does not decide which filing
        # supersedes another. That link lives in the filing-relation layer.
        "CORRECTION",
    }
)

PARSER_CONFIG: dict[str, Any] = {
    "format": "dart4-xml",
    "tree": TREE_NAME,
    "whitespace": "preserve-internal-strip-edges-newlines-kept",
    "heading_tags": sorted(HEADING_TAGS),
    "cell_tags": sorted(CELL_TAGS),
    "header_role": "th-or-thead-ancestor-only",
    "preprocess_version": PREPROCESS_VERSION,
    "preprocess_location": PREPROCESS_LOCATION,
    "literal_angle_allowlist": [
        "신설",
        "연결기준",
        "별도기준",
        "연결기준_",
        "별도기준_",
        "중단영업손실",
        "표1",
        "표2",
    ],
    "non_tag_angle_escape": "allowlist-or-unclosed-nonregistry-name",
    "structural_wrappers": ["CORRECTION"],
    "image_policy": "record-filename-and-caption-no-ocr",
    "nested_table_policy": "owned-child-table-no-value-repeat",
    "locator_context": "per-member-tree-not-process-global",
    "unit_policy": "preserve-raw-text-record-mentions-do-not-apply",
    "max_member_bytes": MAX_MEMBER_BYTES,
    "max_elements": MAX_ELEMENTS,
    "max_depth": MAX_DEPTH,
    "max_rowspan": MAX_ROWSPAN,
    "max_colspan": MAX_COLSPAN,
    "max_table_cells": MAX_TABLE_CELLS,
}

# Exact tokens confirmed as text in stored originals. They stay on the
# allowlist so the repair rule name does not change. Other unclosed
# non-registry names are literals by structure, not by this list.
# <CORRECTION> is a known wrapper and is not a literal angle.
LITERAL_ANGLE_EXACT = frozenset(
    {"신설", "연결기준", "별도기준", "중단영업손실", "표1", "표2"}
)
LITERAL_ANGLE_PREFIXES = ("연결기준_", "별도기준_")
_TAG_AT = re.compile(
    r"<\?.*?\?>|<!--.*?-->|<!\[CDATA\[.*?\]\]>|"
    r"</(?P<close>[A-Za-z][A-Za-z0-9-]*)\s*>|"
    r"<(?P<open>[A-Za-z][A-Za-z0-9-]*)"
    r"(?P<rest>(?=[\s>/])(?:\"[^\"]*\"|'[^']*'|[^<>])*?)?/?>",
    re.DOTALL,
)
_LITERAL_ANGLE = re.compile(r"<([^<>\n]{1,80})>")
_ENTITY = re.compile(r"&(?:#\d+|#x[0-9A-Fa-f]+|[A-Za-z][A-Za-z0-9]+);")
_XML_DECL_ENC = re.compile(br"""encoding\s*=\s*["']([A-Za-z0-9._-]+)["']""", re.I)
# Parenthetical unit labels observed in stored filings, such as "(단위 : 백만원, %)".
_UNIT_LABEL = re.compile(r"\([^()\n]{0,40}단위[^()\n]{0,40}\)")
_DEFAULT_PARSE_LOCK_TIMEOUT_SECONDS = 300.0


class ParseInputError(IngestError):
    """The pinned original cannot be read as a parse input."""


@dataclass
class PinnedSource:
    receipt_no: str
    source_sha256: str
    pin_origin: str
    zip_path: Path
    version_meta: dict[str, Any]
    zip_bytes: bytes


@dataclass
class _Repair:
    rule: str
    offset: int
    before: str
    after: str
    safe: bool


@dataclass
class _Preprocess:
    text: str
    repairs: list[_Repair] = field(default_factory=list)
    unsupported_tags: list[str] = field(default_factory=list)

    @property
    def uncertain(self) -> bool:
        return any(not item.safe for item in self.repairs) or bool(self.unsupported_tags)


@dataclass
class _Emit:
    elements: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    empty_paragraphs: int = 0
    page_breaks: int = 0
    images: int = 0
    escaped_angles: int = 0
    preprocess: _Preprocess | None = None


def config_hash() -> str:
    payload = json.dumps(PARSER_CONFIG, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_receipt(
    receipt_no: str,
    raw_dir: Path,
    output_dir: Path,
    *,
    source_sha256: str | None = None,
    now: datetime | None = None,
    lock_timeout_seconds: float = _DEFAULT_PARSE_LOCK_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Parse one pinned ZIP. Does not call DART or re-read current.json.

    Reuse, quarantine, and publication of this result directory share one
    local lock. After the wait, a completed result is checked again and
    reused. ``ParseLockTimeout`` is not a parse failure and does not rewrite
    the original.
    """
    started = _aware(now or datetime.now(UTC))
    source = pin_source(raw_dir, receipt_no, source_sha256)
    dest = result_dir(output_dir, source.receipt_no, source.source_sha256)
    lock = _ParseOutputLock(_parse_lock_path(dest))
    lock.acquire(timeout_seconds=lock_timeout_seconds)
    try:
        existing = _reusable_result(dest, source)
        if existing is not None:
            return existing

        emit_started = started.isoformat()
        try:
            manifest, elements, preview, report = _parse_zip(source, started_at=emit_started)
        except (ParseInputError, DartZipError, SizeLimitError):
            raise
        manifest["completed_at"] = _aware(datetime.now(UTC)).isoformat()
        _publish(dest, manifest, elements, preview, report)
        return _result_payload(manifest, dest, reused=False)
    finally:
        lock.release()


def pin_source(raw_dir: Path, receipt_no: str, source_sha256: str | None) -> PinnedSource:
    receipt_no = receipt_no.strip()
    if not re.fullmatch(r"\d{14}", receipt_no):
        raise UsageError("receipt number must be exactly 14 digits")
    dest = raw_dir / "dart" / receipt_no
    if not dest.is_dir():
        raise ParseInputError(f"no stored original for receipt {receipt_no}")

    pin_origin = "argument"
    digest = source_sha256.strip() if source_sha256 else None
    if digest is None:
        current = _read_json(dest / "current.json")
        if not isinstance(current, dict) or not current.get("current_sha256"):
            raise ParseInputError("current.json is missing current_sha256")
        digest = str(current["current_sha256"])
        pin_origin = "current.json"
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise UsageError("source sha256 must be 64 lowercase hex characters")

    version_dir = dest / "versions" / digest
    if ".tmp-" in digest or any(part.startswith(".tmp-") for part in version_dir.parts):
        raise ParseInputError("refusing to parse a temporary version directory")
    zip_path = version_dir / ORIGINAL_FILENAME
    meta_path = version_dir / "version.json"
    if not zip_path.is_file() or not meta_path.is_file():
        raise ParseInputError("pinned version is missing document.zip or version.json")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        zip_bytes = zip_path.read_bytes()
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ParseInputError("pinned version is unreadable") from exc
    if not isinstance(meta, dict):
        raise ParseInputError("version.json must be an object")
    if meta.get("receipt_no") != receipt_no:
        raise ParseInputError("version receipt_no does not match the request")
    if meta.get("sha256") != digest or sha256_bytes(zip_bytes) != digest:
        raise ParseInputError("pinned ZIP hash does not match version.json")
    if meta.get("size_bytes") != len(zip_bytes):
        raise ParseInputError("pinned ZIP size does not match version.json")
    validate_zip(zip_bytes)
    return PinnedSource(
        receipt_no=receipt_no,
        source_sha256=digest,
        pin_origin=pin_origin,
        zip_path=zip_path,
        version_meta=meta,
        zip_bytes=zip_bytes,
    )


def result_dir(output_dir: Path, receipt_no: str, source_sha256: str) -> Path:
    # Directory key is a prefix of config_hash so Windows paths stay usable.
    # The full hash is stored in parse_manifest.json.
    return (
        output_dir
        / "dart"
        / receipt_no
        / source_sha256
        / f"p{PARSER_VERSION}"
        / config_hash()[:16]
    )


def resolve_locator(xml_text: str, locator: dict[str, Any]) -> ET.Element:
    """Re-find a node by ``locator["path"]`` on a private tree of ``xml_text``.

    Compatible with the previous signature. This call does not read another
    parse's path map. Pass the member text the locator was built from.
    ``member_index`` on the locator distinguishes same-named ZIP members;
    this function does not open the ZIP.
    """
    root, _ctx = _tree_for_locator(xml_text, locator)
    return _node_at(root, locator)


def element_text_at(xml_text: str, element: dict[str, Any]) -> str:
    """Return normalized text for a paragraph/heading, or the table's own text.

    Uses a private tree for ``xml_text``, same signature as before.
    """
    locator = element["source_locator"]
    root, ctx = _tree_for_locator(xml_text, locator)
    node = _node_at(root, locator)
    if element["element_type"] == "table":
        return _table_payload(node, locator, ctx)["text"]
    if element["element_type"] == "image":
        return _image_text(node)
    return _normalize(_inline_text(node))


# --- internal -----------------------------------------------------------------


def _parse_zip(
    source: PinnedSource, *, started_at: str
) -> tuple[dict[str, Any], list[dict], str, dict[str, Any]]:
    members: list[dict[str, Any]] = []
    elements: list[dict[str, Any]] = []
    preview_parts: list[str] = []
    package_warnings: list[str] = []
    cfg = config_hash()
    listing = source.version_meta.get("listing")
    if listing is not None and not isinstance(listing, dict):
        listing = None

    with zipfile.ZipFile(io.BytesIO(source.zip_bytes)) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        duplicate_names = len(names) != len(set(names))
        for index, info in enumerate(infos):
            member, member_elements = _parse_member(
                archive,
                info,
                index=index,
                source=source,
                config=cfg,
            )
            members.append(member)
            elements.extend(member_elements)
            if member_elements:
                preview_parts.append(_preview_member(member, member_elements))

    report = _preprocess_report(source, members)
    for member in members:
        member.pop("preprocess_changes", None)

    if duplicate_names:
        package_warnings.append("zip_member_names_are_not_unique")

    statuses = [item["status"] for item in members if not item.get("is_directory")]
    overall = _overall_status(statuses)
    if overall == "partial":
        package_warnings.append("one_or_more_members_were_not_fully_parsed")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "receipt_no": source.receipt_no,
        "source_sha256": source.source_sha256,
        "pin_origin": source.pin_origin,
        "parser_version": PARSER_VERSION,
        "preprocess_version": PREPROCESS_VERSION,
        "config_hash": cfg,
        "config": PARSER_CONFIG,
        "started_at": started_at,
        "completed_at": None,
        "overall_status": overall,
        "warnings": package_warnings,
        "scope": {
            "supported": "dart4 XML headings (TITLE, COVER-TITLE), paragraphs (P), tables",
            "excluded": [
                "image bytes / OCR",
                "HTML members (none observed in the pinned samples)",
                "page numbers (PGBRK is a break marker, not a page)",
                "financial account mapping",
            ],
        },
        "element_count": len(elements),
        "members": members,
        "source_metadata": {
            "origin": f"versions/{source.source_sha256}/version.json",
            "listing": listing,
        },
    }
    preview = _preview_document(source, manifest, preview_parts)
    return manifest, elements, preview, report


def _parse_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    index: int,
    source: PinnedSource,
    config: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    name = info.filename.replace("\\", "/")
    base = {
        "member_index": index,
        "member_name": name,
        "compressed_size": info.compress_size,
        "uncompressed_size": info.file_size,
        "is_directory": name.endswith("/") or info.is_dir(),
    }
    if base["is_directory"]:
        return {
            **base,
            "status": "skipped",
            "reason": "directory",
            "element_count": 0,
        }, []
    if _unsafe_zip_name(name):
        return {
            **base,
            "status": "failed",
            "reason": "zip_path_escape",
            "element_count": 0,
            "warnings": ["zip_path_escape"],
        }, []
    if info.file_size > MAX_MEMBER_BYTES:
        return {
            **base,
            "status": "failed",
            "reason": "member_too_large",
            "element_count": 0,
            "warnings": ["member_too_large"],
        }, []
    try:
        raw = archive.read(info)
    except (zipfile.BadZipFile, RuntimeError) as exc:
        return {
            **base,
            "status": "failed",
            "reason": f"unreadable_member:{exc.__class__.__name__}",
            "element_count": 0,
            "warnings": ["unreadable_member"],
        }, []
    byte_hash = sha256_bytes(raw)
    base["byte_sha256"] = byte_hash
    if len(raw) != info.file_size:
        return {
            **base,
            "status": "failed",
            "reason": "uncompressed_size_mismatch",
            "element_count": 0,
            "warnings": ["uncompressed_size_mismatch"],
        }, []

    binary_kind = _non_text_kind(raw, name)
    if binary_kind is not None:
        return {
            **base,
            "status": "excluded",
            "format": binary_kind,
            "encoding": None,
            "reason": "non_text_member",
            "element_count": 0,
            "warnings": [],
        }, []

    try:
        text, encoding = _decode_member(raw)
    except UnicodeError as exc:
        return {
            **base,
            "status": "failed",
            "format": "unknown",
            "encoding": None,
            "reason": f"decode_error:{exc.__class__.__name__}",
            "element_count": 0,
            "warnings": ["decode_error"],
        }, []

    if _has_doctype_or_entity(text):
        return {
            **base,
            "status": "failed",
            "format": "xml",
            "encoding": encoding,
            "reason": "external_entity_or_doctype_rejected",
            "element_count": 0,
            "warnings": ["doctype_or_entity_rejected"],
        }, []

    if not _looks_like_dart_xml(text):
        return {
            **base,
            "status": "unsupported",
            "format": _sniff_format(text, name),
            "encoding": encoding,
            "reason": "not_dart4_xml",
            "element_count": 0,
            "warnings": ["unsupported_member"],
        }, []

    prepared = _preprocess_xml(text, member_index=index)
    try:
        root, ctx = _bind_tree(
            prepared.text,
            source_sha256=source.source_sha256,
            member_index=index,
            member_name=name,
        )
    except ET.ParseError as exc:
        return {
            **base,
            "status": "failed",
            "format": "xml",
            "encoding": encoding,
            "reason": f"xml_not_well_formed:{exc}",
            "element_count": 0,
            "warnings": ["strict_parse_failed"],
            "preprocess": _preprocess_summary(prepared),
        }, []

    document_id = f"m{index}:{name}"
    state = _Emit(escaped_angles=sum(1 for item in prepared.repairs if item.safe))
    state.preprocess = prepared
    meta = _document_meta(root)
    _walk(
        root,
        section=[],
        state=state,
        source=source,
        config=config,
        document_id=document_id,
        depth=0,
        inside_table=False,
        ctx=ctx,
    )
    warnings = list(state.warnings)
    if any(item.safe for item in prepared.repairs):
        warnings.append("safe_literal_repairs_applied")
    if prepared.uncertain:
        warnings.append("uncertain_xml_structure")
    for tag in sorted(set(prepared.unsupported_tags)):
        warnings.append(f"unsupported_tag:{tag}")
    status = "partial" if _member_partial(warnings) else "parsed"
    member = {
        **base,
        "status": status,
        "format": "dart4-xml",
        "encoding": encoding,
        "role": "unknown",
        "document_id": document_id,
        "document_name": meta.get("document_name"),
        "company_name": meta.get("company_name"),
        "element_count": len(state.elements),
        "empty_paragraphs_omitted": state.empty_paragraphs,
        "page_breaks_omitted": state.page_breaks,
        "images_recorded": state.images,
        "non_tag_angles_escaped": sum(
            1 for item in prepared.repairs if item.rule != "bare_ampersand"
        ),
        "preprocess": _preprocess_summary(prepared),
        "preprocess_changes": [
            {
                "rule": item.rule,
                "offset": item.offset,
                "before": item.before,
                "after": item.after,
                "safe": item.safe,
            }
            for item in prepared.repairs
        ],
        "warnings": warnings,
        "reason": None,
    }
    return member, state.elements


def _walk(
    node: ET.Element,
    *,
    section: list[str],
    state: _Emit,
    source: PinnedSource,
    config: str,
    document_id: str,
    depth: int,
    inside_table: bool,
    ctx: _TreeContext,
) -> None:
    if depth > MAX_DEPTH:
        state.warnings.append("max_depth_exceeded")
        return
    if len(state.elements) > MAX_ELEMENTS:
        state.warnings.append("max_elements_exceeded")
        return
    tag = _local(node.tag)
    if tag not in KNOWN_TAGS:
        # Still unsupported: a better section_path does not make this success.
        state.warnings.append(f"unsupported_tag:{tag}")
        direct = _normalize(node.text or "")
        if direct:
            _append_element(
                state,
                element_type="paragraph",
                text=direct,
                section=list(section),
                node=node,
                source=source,
                config=config,
                document_id=document_id,
                extra={},
                warnings=[f"unsupported_tag:{tag}"],
                ctx=ctx,
            )
        _walk_following(
            node,
            section=section,
            state=state,
            source=source,
            config=config,
            document_id=document_id,
            depth=depth,
            inside_table=inside_table,
            ctx=ctx,
            emit_tails=True,
            tail_warning=f"unsupported_tag_tail:{tag}",
        )
        return
    if tag == "PGBRK":
        state.page_breaks += 1
        return
    if tag == "TABLE":
        _emit_table(
            node,
            section=section,
            state=state,
            source=source,
            config=config,
            document_id=document_id,
            nest_level=1,
            ctx=ctx,
        )
        return
    if tag == "IMAGE":
        _emit_image(
            node,
            section=section,
            state=state,
            source=source,
            config=config,
            document_id=document_id,
            ctx=ctx,
        )
        return
    if tag in HEADING_TAGS and not inside_table:
        text = _normalize(_inline_text(node))
        if text:
            _append_element(
                state,
                element_type="heading",
                text=text,
                section=list(section),
                node=node,
                source=source,
                config=config,
                document_id=document_id,
                extra={},
                ctx=ctx,
            )
            section = [*section, text]
        else:
            state.empty_paragraphs += 1
        return
    if tag == "P" and not inside_table:
        text = _normalize(_inline_text(node))
        if text:
            _append_element(
                state,
                element_type="paragraph",
                text=text,
                section=list(section),
                node=node,
                source=source,
                config=config,
                document_id=document_id,
                extra={},
                ctx=ctx,
            )
        else:
            state.empty_paragraphs += 1
        return
    if tag in CELL_TAGS or tag in {"TR", "THEAD", "TBODY", "COLGROUP"}:
        return
    _walk_following(
        node,
        section=section,
        state=state,
        source=source,
        config=config,
        document_id=document_id,
        depth=depth,
        inside_table=inside_table,
        ctx=ctx,
        emit_tails=False,
        tail_warning="",
    )


def _walk_following(
    node: ET.Element,
    *,
    section: list[str],
    state: _Emit,
    source: PinnedSource,
    config: str,
    document_id: str,
    depth: int,
    inside_table: bool,
    ctx: _TreeContext,
    emit_tails: bool,
    tail_warning: str,
) -> None:
    """Pass heading context to later siblings without mutating the caller's list.

    ``section = [*section, heading]`` rebinds this frame only. A wrapper's
    titles therefore do not attach to siblings outside the wrapper or to the
    next document.
    """
    for child in list(node):
        child_tag = _local(child.tag)
        before = len(section)
        _walk(
            child,
            section=section,
            state=state,
            source=source,
            config=config,
            document_id=document_id,
            depth=depth + 1,
            inside_table=inside_table,
            ctx=ctx,
        )
        if child_tag in HEADING_TAGS and len(section) == before:
            heading = _normalize(_inline_text(child))
            if heading:
                section = [*section, heading]
        if not emit_tails:
            continue
        tail = _normalize(child.tail or "")
        if tail:
            _append_element(
                state,
                element_type="paragraph",
                text=tail,
                section=list(section),
                node=child,
                source=source,
                config=config,
                document_id=document_id,
                extra={},
                warnings=[tail_warning],
                ctx=ctx,
            )


def _emit_table(
    node: ET.Element,
    *,
    section: list[str],
    state: _Emit,
    source: PinnedSource,
    config: str,
    document_id: str,
    nest_level: int,
    ctx: _TreeContext,
) -> None:
    payload = _table_payload(node, _locator_for(node, ctx), ctx)
    warnings = list(payload.pop("warnings"))
    nested_nodes: list[ET.Element] = payload.pop("nested_nodes")
    if nest_level > 2:
        warnings.append("nested_table_deeper_than_observed_samples")
        state.warnings.append("nested_table_deeper_than_observed_samples")
    for item in warnings:
        if item in _TABLE_DIMENSION_WARNINGS and item not in state.warnings:
            state.warnings.append(item)
    element = _append_element(
        state,
        element_type="table",
        text=payload["text"],
        section=list(section),
        node=node,
        source=source,
        config=config,
        document_id=document_id,
        extra={"table": {key: payload[key] for key in ("cells", "row_count", "col_count")}},
        warnings=warnings,
        ctx=ctx,
    )
    mentions: list[dict[str, Any]] = []
    for cell in element["table"]["cells"]:
        mentions.extend(_unit_mentions(cell["text"], cell["source_locator"]))
    if mentions:
        element["unit_mentions"] = mentions
        element["unit_scope_status"] = _unit_scope_status(mentions)
        element["content_hash"] = _content_hash(element)
    for nested in nested_nodes:
        _emit_table(
            nested,
            section=section,
            state=state,
            source=source,
            config=config,
            document_id=document_id,
            nest_level=nest_level + 1,
            ctx=ctx,
        )
        child_id = state.elements[-1]["element_id"]
        _attach_nested(element, nested, child_id, ctx)
        element["content_hash"] = _content_hash(element)


def _attach_nested(
    table_element: dict[str, Any], nested: ET.Element, child_id: str, ctx: _TreeContext
) -> None:
    nested_loc = _locator_for(nested, ctx)
    for cell in table_element["table"]["cells"]:
        locators = cell.get("nested_table_locators")
        if not locators:
            continue
        for index, locator in enumerate(locators):
            if locator != nested_loc:
                continue
            cell.setdefault("nested_table_element_ids", []).append(child_id)
            del locators[index]
            if not locators:
                cell.pop("nested_table_locators", None)
            return


def _emit_image(
    node: ET.Element,
    *,
    section: list[str],
    state: _Emit,
    source: PinnedSource,
    config: str,
    document_id: str,
    ctx: _TreeContext,
) -> None:
    text = _image_text(node)
    filename = None
    caption = None
    for child in node.iter():
        tag = _local(child.tag)
        if tag == "IMG" and filename is None:
            filename = _normalize(child.text or "")
        if tag == "IMG-CAPTION" and caption is None:
            caption = _normalize(_inline_text(child))
    _append_element(
        state,
        element_type="image",
        text=text,
        section=list(section),
        node=node,
        source=source,
        config=config,
        document_id=document_id,
        extra={"image": {"filename": filename, "caption": caption}},
        warnings=["image_not_interpreted"],
        ctx=ctx,
    )
    state.images += 1


def _append_element(
    state: _Emit,
    *,
    element_type: str,
    text: str,
    section: list[str],
    node: ET.Element,
    source: PinnedSource,
    config: str,
    document_id: str,
    extra: dict[str, Any],
    ctx: _TreeContext,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    locator = _locator_for(node, ctx)
    order_index = len(state.elements)
    element_warnings = list(warnings or [])
    body: dict[str, Any] = {
        "element_type": element_type,
        "order_index": order_index,
        "document_id": document_id,
        "section_path": section,
        "text": text,
        "source_locator": locator,
        "warnings": element_warnings,
        **extra,
    }
    if element_type == "paragraph":
        mentions = _unit_mentions(text, locator)
        if mentions:
            body["unit_mentions"] = mentions
            body["unit_scope_status"] = _unit_scope_status(mentions)
    body["content_hash"] = _content_hash(body)
    body["element_id"] = _element_id(
        source_sha256=source.source_sha256,
        config=config,
        document_id=document_id,
        order_index=order_index,
        element_type=element_type,
        locator=locator,
    )
    state.elements.append(body)
    return body


def _table_payload(
    node: ET.Element, table_locator: dict[str, Any], ctx: _TreeContext
) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    warnings: list[str] = []
    nested_nodes: list[ET.Element] = []
    occupied: dict[tuple[int, int], bool] = {}
    row = 0
    stopped = False
    for tr in _direct_rows(node):
        if stopped:
            break
        col = 0
        for cell in list(tr):
            tag = _local(cell.tag)
            if tag not in CELL_TAGS:
                continue
            while occupied.get((row, col)):
                col += 1
                if col > MAX_TABLE_CELLS:
                    warnings.append("table_cell_limit_exceeded")
                    stopped = True
                    break
            if stopped or len(cells) >= MAX_TABLE_CELLS:
                warnings.append("table_cell_limit_exceeded")
                stopped = True
                break
            declared_rowspan = _positive_int(cell.attrib.get("ROWSPAN"), default=1)
            declared_colspan = _positive_int(cell.attrib.get("COLSPAN"), default=1)
            rowspan, colspan, clamped = _bounded_span(
                declared_rowspan, declared_colspan, occupied_count=len(occupied)
            )
            if clamped:
                warnings.append("table_span_clamped")
            for dr in range(rowspan):
                for dc in range(colspan):
                    slot = (row + dr, col + dc)
                    if slot in occupied and (dr or dc):
                        warnings.append("merged_cell_overlap")
                    if dr or dc:
                        occupied[slot] = True
                    elif slot in occupied:
                        warnings.append("cell_slot_overlap")
                    else:
                        occupied[slot] = True
            nested = _nested_tables(cell)
            for item in nested:
                nested_nodes.append(item)
            text = _cell_text(cell)
            header_role = "header" if tag == "TH" or _in_thead(cell, node) else "unknown"
            cell_locator = _locator_for(cell, ctx)
            record: dict[str, Any] = {
                "start_row": row,
                "start_col": col,
                "rowspan": rowspan,
                "colspan": colspan,
                "text": text,
                "header_role": header_role,
                "source_locator": cell_locator,
                "tag": tag,
            }
            if clamped:
                record["declared_rowspan"] = declared_rowspan
                record["declared_colspan"] = declared_colspan
            if nested:
                record["nested_table_locators"] = [_locator_for(item, ctx) for item in nested]
                if len(nested) > 1:
                    warnings.append("multiple_nested_tables_in_one_cell")
            cells.append(record)
            col += colspan
        row += 1
    warnings = list(dict.fromkeys(warnings))
    row_count = 0
    col_count = 0
    for cell in cells:
        row_count = max(row_count, cell["start_row"] + cell["rowspan"])
        col_count = max(col_count, cell["start_col"] + cell["colspan"])
    preview = _table_preview_text(cells, row_count, col_count)
    return {
        "cells": cells,
        "row_count": row_count,
        "col_count": col_count,
        "text": preview,
        "warnings": warnings,
        "nested_nodes": nested_nodes,
        "locator": table_locator,
    }


def _direct_rows(table: ET.Element) -> Iterator[ET.Element]:
    """Yield TR elements that belong to this table, not to a nested table."""

    def walk(node: ET.Element, nested: bool) -> Iterator[ET.Element]:
        for child in list(node):
            tag = _local(child.tag)
            if tag == "TABLE":
                continue
            if tag == "TR" and not nested:
                yield child
                continue
            yield from walk(child, nested)

    yield from walk(table, False)


def _nested_tables(cell: ET.Element) -> list[ET.Element]:
    found: list[ET.Element] = []

    def walk(node: ET.Element) -> None:
        for child in list(node):
            if _local(child.tag) == "TABLE":
                found.append(child)
                continue
            walk(child)

    walk(cell)
    return found


def _in_thead(cell: ET.Element, table: ET.Element) -> bool:
    parent_map = {child: parent for parent in table.iter() for child in list(parent)}
    node: ET.Element | None = cell
    while node is not None and node is not table:
        if _local(node.tag) == "THEAD":
            return True
        node = parent_map.get(node)
    return False


def _cell_text(cell: ET.Element) -> str:
    parts: list[str] = []

    def walk(node: ET.Element) -> None:
        tag = _local(node.tag)
        if tag == "TABLE":
            return
        if tag == "P":
            text = _normalize(_inline_text(node))
            if text:
                if parts:
                    parts.append("\n")
                parts.append(text)
            return
        if node.text:
            parts.append(node.text)
        for child in list(node):
            if _local(child.tag) == "TABLE":
                if child.tail:
                    parts.append(child.tail)
                continue
            walk(child)
            if child.tail:
                parts.append(child.tail)

    if cell.text and cell.text.strip() and not any(_local(c.tag) == "P" for c in list(cell)):
        parts.append(cell.text)
        for child in list(cell):
            if _local(child.tag) == "TABLE":
                if child.tail:
                    parts.append(child.tail)
                continue
            if _local(child.tag) in {"SPAN", "A"}:
                parts.append(_inline_text(child))
            else:
                walk(child)
            if child.tail:
                parts.append(child.tail)
    else:
        walk(cell)
    return _normalize("".join(parts))


def _bounded_span(
    declared_rowspan: int, declared_colspan: int, *, occupied_count: int
) -> tuple[int, int, bool]:
    """Clamp a merge so the covered slots stay inside the table cell budget."""
    rowspan = min(declared_rowspan, MAX_ROWSPAN)
    colspan = min(declared_colspan, MAX_COLSPAN)
    budget = max(1, MAX_TABLE_CELLS - occupied_count)
    clamped = rowspan != declared_rowspan or colspan != declared_colspan
    if rowspan * colspan > budget:
        colspan = min(colspan, budget)
        rowspan = min(rowspan, max(1, budget // colspan))
        clamped = True
    return rowspan, colspan, clamped


_TABLE_DIMENSION_WARNINGS = frozenset({"table_span_clamped", "table_cell_limit_exceeded"})


def _table_preview_text(cells: list[dict[str, Any]], row_count: int, col_count: int) -> str:
    if not cells:
        return ""
    if row_count * col_count > MAX_TABLE_CELLS:
        return "\n".join(cell["text"] for cell in cells)
    grid = [["" for _ in range(col_count)] for _ in range(row_count)]
    for cell in cells:
        grid[cell["start_row"]][cell["start_col"]] = cell["text"]
    lines = ["\t".join(row) for row in grid]
    return "\n".join(lines)


def _image_text(node: ET.Element) -> str:
    filename = ""
    caption = ""
    for child in node.iter():
        tag = _local(child.tag)
        if tag == "IMG" and not filename:
            filename = _normalize(child.text or "")
        elif tag == "IMG-CAPTION" and not caption:
            caption = _normalize(_inline_text(child))
    if filename and caption:
        return f"{filename}\n{caption}"
    return filename or caption


def _inline_text(node: ET.Element) -> str:
    parts: list[str] = []
    if node.text:
        parts.append(node.text)
    for child in list(node):
        parts.append(_inline_text(child))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts)


def _normalize(text: str) -> str:
    value = text.replace("\r\n", "\n").replace("\r", "\n")
    return value.strip()


def _document_meta(root: ET.Element) -> dict[str, str | None]:
    document_name = None
    company_name = None
    for child in list(root):
        tag = _local(child.tag)
        if tag == "DOCUMENT-NAME" and document_name is None:
            document_name = _normalize(child.text or "") or None
        if tag == "COMPANY-NAME" and company_name is None:
            company_name = _normalize(child.text or "") or None
    return {"document_name": document_name, "company_name": company_name}


@dataclass
class _TreeContext:
    """Locator paths for one decoded ZIP member. Not process-global."""

    source_sha256: str
    member_index: int
    member_name: str
    paths: dict[int, list[dict[str, Any]]] = field(default_factory=dict)


def _locator_for(node: ET.Element, ctx: _TreeContext) -> dict[str, Any]:
    path = ctx.paths.get(id(node))
    if path is None:
        raise ParseInputError("element is missing a structural path")
    return {
        "tree": TREE_NAME,
        "source_sha256": ctx.source_sha256,
        "member_index": ctx.member_index,
        "member_name": ctx.member_name,
        "path": path,
    }


def _bind_tree(
    text: str,
    *,
    source_sha256: str,
    member_index: int,
    member_name: str,
) -> tuple[ET.Element, _TreeContext]:
    root = ET.fromstring(text, parser=ET.XMLParser())
    ctx = _TreeContext(
        source_sha256=source_sha256,
        member_index=member_index,
        member_name=member_name,
    )
    _annotate_paths(root, [{"tag": _local(root.tag), "index": 0}], ctx)
    return root, ctx


def _annotate_paths(node: ET.Element, path: list[dict[str, Any]], ctx: _TreeContext) -> None:
    ctx.paths[id(node)] = path
    for index, child in enumerate(list(node)):
        _annotate_paths(child, [*path, {"tag": _local(child.tag), "index": index}], ctx)


def _tree_for_locator(xml_text: str, locator: dict[str, Any]) -> tuple[ET.Element, _TreeContext]:
    member_index = locator.get("member_index")
    if not isinstance(member_index, int):
        member_index = 0
    prepared = _preprocess_xml(xml_text, member_index=member_index)
    return _bind_tree(
        prepared.text,
        source_sha256=str(locator.get("source_sha256") or ""),
        member_index=member_index,
        member_name=str(locator.get("member_name") or ""),
    )


def _node_at(root: ET.Element, locator: dict[str, Any]) -> ET.Element:
    node: ET.Element = root
    path = locator.get("path")
    if not isinstance(path, list):
        raise ParseInputError("locator path is missing")
    if not path or path[0].get("tag") != _local(root.tag) or path[0].get("index") != 0:
        raise ParseInputError("locator does not start at the document root")
    for step in path[1:]:
        tag = step.get("tag")
        index = step.get("index")
        children = list(node)
        if not isinstance(index, int) or index < 0 or index >= len(children):
            raise ParseInputError("locator child index is out of range")
        child = children[index]
        if _local(child.tag) != tag:
            raise ParseInputError("locator tag does not match the tree")
        node = child
    return node


def _unit_mentions(text: str, locator: dict[str, Any]) -> list[dict[str, Any]]:
    """Record observed parenthetical unit labels without applying them.

    A label inside a cell is not a table-wide unit. A label in a paragraph is
    not attached to a neighboring table. Mixed labels such as
    ``(단위: 백만원, %)`` stay one raw string and are ``unresolved`` rather than
    a single multiplier. No label means the key is omitted (absence), which is
    different from a mention whose scope is still unknown.
    """
    mentions: list[dict[str, Any]] = []
    for match in _UNIT_LABEL.finditer(text):
        raw = match.group(0)
        inner = raw[1:-1]
        mixed = "," in inner or "，" in inner
        mentions.append(
            {
                "raw_text": raw,
                "source_locator": locator,
                "applies_to": None,
                "scope_status": "unresolved" if mixed else "unknown",
            }
        )
    return mentions


def _unit_scope_status(mentions: list[dict[str, Any]]) -> str:
    if any(item["scope_status"] == "unresolved" for item in mentions):
        return "unresolved"
    if len({item["raw_text"] for item in mentions}) > 1:
        return "unresolved"
    return "unknown"


def _preprocess_xml(text: str, *, member_index: int) -> _Preprocess:
    """Rewrite safe literals. Unknown ASCII tags stay structural.

    A non-ASCII angle is literal text when it has no matching close tag.
    A matching close tag that is not in the DART tag registry stays uncertain.
    """
    balanced = _balanced_nonregistry_names(text)
    pieces: list[str] = []
    repairs: list[_Repair] = []
    unsupported: list[str] = []
    cursor = 0
    for match in _TAG_AT.finditer(text):
        segment = text[cursor : match.start()]
        pieces.append(
            _rewrite_text(segment, base=cursor, repairs=repairs, balanced=balanced)
        )
        token = match.group(0)
        name = (match.group("open") or match.group("close") or "").upper()
        if token.startswith(("<?", "<!--", "<![CDATA[")):
            pieces.append(token)
        elif name in KNOWN_TAGS:
            pieces.append(_escape_ampersands_tracked(token, base=match.start(), repairs=repairs))
        else:
            unsupported.append(name)
            pieces.append(_escape_ampersands_tracked(token, base=match.start(), repairs=repairs))
            repairs.append(
                _Repair(
                    rule="unsupported_tag_kept",
                    offset=match.start(),
                    before=token[:80],
                    after=token[:80],
                    safe=False,
                )
            )
        cursor = match.end()
    pieces.append(
        _rewrite_text(text[cursor:], base=cursor, repairs=repairs, balanced=balanced)
    )
    del member_index
    return _Preprocess(text="".join(pieces), repairs=repairs, unsupported_tags=unsupported)


def _balanced_nonregistry_names(text: str) -> frozenset[str]:
    """Names that occur as both ``<name>`` and ``</name>`` and are not ASCII tags.

    ``_TAG_AT`` already classifies ASCII tags. A Hangul (or mixed) name with
    both ends can still be markup, so it is not silently treated as text.
    """
    opens: set[str] = set()
    closes: set[str] = set()
    for match in _LITERAL_ANGLE.finditer(text):
        raw = match.group(1).strip()
        closing = raw.startswith("/")
        inner = raw[1:].strip() if closing else raw
        if not inner or any(char in inner for char in " \t=/\"'"):
            continue
        if not _xml_name(inner):
            continue
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9-]*", inner):
            continue
        if closing:
            closes.add(inner)
        else:
            opens.add(inner)
    return frozenset(opens & closes)


def _rewrite_text(
    text: str,
    *,
    base: int,
    repairs: list[_Repair],
    balanced: frozenset[str],
) -> str:
    pieces: list[str] = []
    cursor = 0
    for match in _LITERAL_ANGLE.finditer(text):
        pieces.append(
            _escape_ampersands_tracked(
                text[cursor : match.start()], base=base + cursor, repairs=repairs
            )
        )
        inner = match.group(1)
        token = match.group(0)
        if _is_safe_literal(inner, balanced):
            after = "&lt;" + token[1:]
            repairs.append(
                _Repair(
                    rule=_literal_rule(inner, balanced),
                    offset=base + match.start(),
                    before=token,
                    after=after,
                    safe=True,
                )
            )
            pieces.append(after)
        elif _xml_name(inner.strip()):
            after = "&lt;" + token[1:]
            repairs.append(
                _Repair(
                    rule="uncertain_named_angle",
                    offset=base + match.start(),
                    before=token,
                    after=after,
                    safe=False,
                )
            )
            pieces.append(after)
        else:
            after = "&lt;" + token[1:]
            repairs.append(
                _Repair(
                    rule="non_name_angle",
                    offset=base + match.start(),
                    before=token,
                    after=after,
                    safe=True,
                )
            )
            pieces.append(after)
        cursor = match.end()
    tail = text[cursor:]
    pieces.append(_escape_loose_lt(tail, base=base + cursor, repairs=repairs))
    return "".join(pieces)


def _escape_loose_lt(text: str, *, base: int, repairs: list[_Repair]) -> str:
    out: list[str] = []
    cursor = 0
    for index, char in enumerate(text):
        if char != "<":
            continue
        if index > cursor:
            out.append(
                _escape_ampersands_tracked(
                    text[cursor:index], base=base + cursor, repairs=repairs
                )
            )
        repairs.append(
            _Repair(rule="bare_lt", offset=base + index, before="<", after="&lt;", safe=True)
        )
        out.append("&lt;")
        cursor = index + 1
    out.append(_escape_ampersands_tracked(text[cursor:], base=base + cursor, repairs=repairs))
    return "".join(out)


def _is_allowlisted(inner: str) -> bool:
    stripped = inner.strip()
    if stripped in LITERAL_ANGLE_EXACT:
        return True
    return any(stripped.startswith(prefix) for prefix in LITERAL_ANGLE_PREFIXES)


def _literal_rule(inner: str, balanced: frozenset[str]) -> str:
    if _is_allowlisted(inner):
        return "literal_angle_allowlist"
    stripped = inner.strip()
    if stripped and _xml_name(stripped) and stripped not in balanced:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9-]*", stripped):
            return "literal_unclosed_angle"
    return "non_name_angle"


def _is_safe_literal(inner: str, balanced: frozenset[str]) -> bool:
    if _is_allowlisted(inner):
        return True
    stripped = inner.strip()
    if not stripped:
        return True
    if not _xml_name(stripped):
        return True
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9-]*", stripped):
        return False
    return stripped not in balanced


def _xml_name(value: str) -> bool:
    if not value or value[:1] in "/!?":
        return False
    return re.fullmatch(r"[^\W\d][\w.:-]*", value, flags=re.UNICODE) is not None


def _escape_ampersands_tracked(text: str, *, base: int, repairs: list[_Repair]) -> str:
    out: list[str] = []
    cursor = 0
    for match in _ENTITY.finditer(text):
        raw = text[cursor : match.start()]
        out.append(_replace_bare_amp(raw, base=base + cursor, repairs=repairs))
        out.append(match.group(0))
        cursor = match.end()
    out.append(_replace_bare_amp(text[cursor:], base=base + cursor, repairs=repairs))
    return "".join(out)


def _replace_bare_amp(text: str, *, base: int, repairs: list[_Repair]) -> str:
    if "&" not in text:
        return text
    out: list[str] = []
    cursor = 0
    for index, char in enumerate(text):
        if char != "&":
            continue
        out.append(text[cursor:index])
        repairs.append(
            _Repair(
                rule="bare_ampersand",
                offset=base + index,
                before="&",
                after="&amp;",
                safe=True,
            )
        )
        out.append("&amp;")
        cursor = index + 1
    out.append(text[cursor:])
    return "".join(out)


def _decode_member(raw: bytes) -> tuple[str, str]:
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig"), "utf-8"
    head = raw[:240]
    match = _XML_DECL_ENC.search(head)
    if match:
        encoding = match.group(1).decode("ascii")
        return raw.decode(encoding), encoding
    return raw.decode("utf-8"), "utf-8"


def _looks_like_dart_xml(text: str) -> bool:
    sample = text.lstrip("\ufeff").lstrip()
    if sample.startswith("<?xml"):
        sample = sample.split("?>", 1)[-1].lstrip()
    return sample.startswith("<DOCUMENT") or sample.startswith("<document")


def _sniff_format(text: str, name: str) -> str:
    lowered = text.lstrip()[:200].lower()
    suffix = Path(name).suffix.lower()
    if "<html" in lowered or suffix in {".htm", ".html"}:
        return "html"
    if name.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp")):
        return "image"
    if lowered.startswith("<?xml") or lowered.startswith("<"):
        return "xml"
    return suffix.lstrip(".") or "binary"


def _has_doctype_or_entity(text: str) -> bool:
    head = text[:5000].upper()
    return "<!DOCTYPE" in head or "<!ENTITY" in head


def _unsafe_zip_name(name: str) -> bool:
    if name.startswith("/") or name.startswith("\\"):
        return True
    parts = Path(name).parts
    return ".." in parts


def _member_partial(warnings: list[str]) -> bool:
    blocking = {
        "max_depth_exceeded",
        "max_elements_exceeded",
        "nested_table_deeper_than_observed_samples",
        "strict_parse_failed",
        "uncertain_xml_structure",
        *_TABLE_DIMENSION_WARNINGS,
    }
    if any(item in blocking for item in warnings):
        return True
    return any(item.startswith("unsupported_tag") for item in warnings)


def _preprocess_summary(prepared: _Preprocess) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for item in prepared.repairs:
        counts[item.rule] = counts.get(item.rule, 0) + 1
    return {
        "preprocess_version": PREPROCESS_VERSION,
        "location": PREPROCESS_LOCATION,
        "safe_repair_count": sum(1 for item in prepared.repairs if item.safe),
        "uncertain_count": sum(1 for item in prepared.repairs if not item.safe),
        "rule_counts": counts,
        "unsupported_tags": sorted(set(prepared.unsupported_tags)),
    }


def _non_text_kind(raw: bytes, name: str) -> str | None:
    suffix = Path(name).suffix.lower()
    if suffix in {".xml", ".html", ".htm", ".txt", ".xhtml"}:
        return None
    stripped = raw.lstrip()
    if stripped.startswith((b"<?xml", b"<DOCUMENT", b"<html", b"<HTML", b"\xef\xbb\xbf")):
        return None
    if suffix in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff"}:
        return "image"
    return "binary"


def _overall_status(statuses: list[str]) -> str:
    if not statuses:
        return "failed"
    interesting = [item for item in statuses if item not in {"skipped", "excluded"}]
    if not interesting:
        return "unsupported"
    if any(item == "failed" for item in interesting):
        if any(item == "parsed" for item in interesting):
            return "partial"
        return "failed"
    if any(item == "partial" for item in interesting):
        return "partial"
    if all(item == "unsupported" for item in interesting):
        return "unsupported"
    if any(item == "unsupported" for item in interesting):
        return "partial"
    if all(item == "parsed" for item in interesting):
        return "success"
    return "partial"


def _element_id(
    *,
    source_sha256: str,
    config: str,
    document_id: str,
    order_index: int,
    element_type: str,
    locator: dict[str, Any],
) -> str:
    raw = json.dumps(
        {
            "v": PARSER_VERSION,
            "source_sha256": source_sha256,
            "config_hash": config,
            "document_id": document_id,
            "order_index": order_index,
            "element_type": element_type,
            "source_locator": locator,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _content_hash(body: dict[str, Any]) -> str:
    semantic = {
        "element_type": body["element_type"],
        "section_path": body["section_path"],
        "text": body["text"],
        "warnings": body["warnings"],
    }
    if "table" in body:
        semantic["table"] = body["table"]
    if "image" in body:
        semantic["image"] = body["image"]
    if "unit_mentions" in body:
        semantic["unit_mentions"] = body["unit_mentions"]
        semantic["unit_scope_status"] = body.get("unit_scope_status")
    raw = json.dumps(semantic, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _preview_document(source: PinnedSource, manifest: dict[str, Any], parts: list[str]) -> str:
    lines = [
        f"# {source.receipt_no}",
        "",
        f"- source_sha256: `{source.source_sha256}`",
        f"- parser_version: `{PARSER_VERSION}`",
        f"- config_hash: `{manifest['config_hash']}`",
        f"- overall_status: `{manifest['overall_status']}`",
        "",
        "병합으로 덮인 칸은 값을 반복하지 않습니다. 빈 칸은 관측된 빈 셀이거나 병합 슬롯입니다.",
        "",
        *parts,
    ]
    return "\n".join(lines).rstrip() + "\n"


def _preview_member(member: dict[str, Any], elements: list[dict[str, Any]]) -> str:
    lines = [
        f"## {member['member_name']}",
        "",
        f"role={member.get('role')} document_name={member.get('document_name')}",
        "",
    ]
    for element in elements:
        if element["element_type"] == "heading":
            level = min(6, 1 + len(element["section_path"]))
            lines.append(f"{'#' * level} {element['text']}")
            lines.append("")
        elif element["element_type"] == "paragraph":
            lines.append(element["text"])
            lines.append("")
        elif element["element_type"] == "table":
            lines.append(_markdown_table(element["table"]))
            lines.append("")
        elif element["element_type"] == "image":
            lines.append(f"[image] {element['text']}")
            lines.append("")
    return "\n".join(lines)


def _markdown_table(table: dict[str, Any]) -> str:
    rows = table["row_count"]
    cols = table["col_count"]
    if rows == 0 or cols == 0:
        return "(empty table)"
    if rows * cols > MAX_TABLE_CELLS:
        lines = ["(table grid omitted: dimension limit)"]
        lines.extend(cell["text"] for cell in table["cells"])
        return "\n".join(lines)
    grid = [["" for _ in range(cols)] for _ in range(rows)]
    for cell in table["cells"]:
        grid[cell["start_row"]][cell["start_col"]] = cell["text"].replace("\n", " ")
    rendered = []
    for row in grid:
        rendered.append("| " + " | ".join(item.replace("|", "\\|") for item in row) + " |")
    return "\n".join(rendered)


def _preprocess_report(source: PinnedSource, members: list[dict[str, Any]]) -> dict[str, Any]:
    recorded = []
    for member in members:
        changes = member.get("preprocess_changes") or []
        if not changes and not (member.get("preprocess") or {}).get("unsupported_tags"):
            continue
        recorded.append(
            {
                "member_index": member.get("member_index"),
                "member_name": member.get("member_name"),
                "summary": member.get("preprocess"),
                "changes": changes,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "preprocess_version": PREPROCESS_VERSION,
        "location": PREPROCESS_LOCATION,
        "source_sha256": source.source_sha256,
        "safe_repair_does_not_force_partial": True,
        "members": recorded,
    }


def _member_summaries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in manifest.get("members") or []:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "member_index": item.get("member_index"),
                "member_name": item.get("member_name"),
                "status": item.get("status"),
                "warnings": list(item.get("warnings") or []),
                "reason": item.get("reason"),
            }
        )
    return rows


def _result_payload(manifest: dict[str, Any], dest: Path, *, reused: bool) -> dict[str, Any]:
    return {
        "status": manifest["overall_status"],
        "receipt_no": manifest["receipt_no"],
        "source_sha256": manifest["source_sha256"],
        "dest": str(dest),
        "parser_version": manifest.get("parser_version"),
        "preprocess_version": manifest.get("preprocess_version"),
        "config_hash": manifest.get("config_hash"),
        "element_count": manifest.get("element_count"),
        "warnings": list(manifest.get("warnings") or []),
        "members": _member_summaries(manifest),
        "manifest": str(dest / "parse_manifest.json"),
        "reused": reused,
        "search_eligible": manifest.get("overall_status") in SEARCH_ELIGIBLE_STATUSES,
    }


def _parse_lock_path(dest: Path) -> Path:
    """Lock the published directory key (config-hash prefix), not the sync lock."""
    return dest.parent / f".{dest.name}.lock"


class _ParseOutputLock:
    """Exclusive-create lock for one parse result directory.

    Local filesystem only. Windows NTFS accepts ``O_CREAT|O_EXCL``. The
    descriptor stays open until release. A crashed holder leaves the file;
    age does not delete it, and this class never removes a lock it did not
    acquire. Not a distributed lock, and not the collection ``.sync.lock``.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.fd: int | None = None

    def acquire(self, timeout_seconds: float) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + max(0.0, timeout_seconds)
        while True:
            try:
                self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise ParseLockTimeout(
                        "timed out waiting for the local parse-output lock; "
                        "this is not a damaged original and not a parse failure. "
                        f"Lock file: {self.path}. "
                        "It is not removed automatically. "
                        "Remove it only when no process holds it."
                    ) from None
                time.sleep(0.05)
                continue
            os.write(self.fd, f"pid={os.getpid()}\n".encode())
            return

    def release(self) -> None:
        if self.fd is None:
            return
        try:
            os.close(self.fd)
        except OSError:
            pass
        self.fd = None
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass


def _publish(
    dest: Path,
    manifest: dict[str, Any],
    elements: list[dict[str, Any]],
    preview: str,
    report: dict[str, Any],
) -> None:
    """Publish this run's temp directory. Does not delete other runs' temps."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / f".tmp-{os.getpid()}-{uuid4().hex[:12]}"
    try:
        tmp.mkdir()
        (tmp / "parse_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with (tmp / "elements.jsonl").open("w", encoding="utf-8") as handle:
            for element in elements:
                handle.write(json.dumps(element, ensure_ascii=False) + "\n")
        (tmp / "preview.md").write_text(preview, encoding="utf-8")
        (tmp / "preprocess_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (tmp / "COMPLETE").write_text("ok\n", encoding="utf-8")
        if dest.exists():
            if (dest / "COMPLETE").is_file():
                raise ParseInputError("refusing to overwrite a completed parse result")
            shutil.rmtree(dest)
        os.rename(tmp, dest)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def _reusable_result(dest: Path, source: PinnedSource) -> dict[str, Any] | None:
    if not (dest / "COMPLETE").is_file():
        return None
    try:
        manifest = _load_complete_manifest(dest, source)
    except (ParseInputError, OSError, UnicodeError, json.JSONDecodeError):
        _quarantine_result(dest, reason="unreadable_or_inconsistent")
        return None
    if manifest.get("config_hash") != config_hash():
        raise ParseInputError(
            "full config_hash differs from this directory; refusing to reuse or overwrite"
        )
    if manifest.get("overall_status") not in {"success", "partial"}:
        _quarantine_result(dest, reason="completed_but_not_reusable_status")
        return None
    return _result_payload(manifest, dest, reused=True)


def _load_complete_manifest(dest: Path, source: PinnedSource) -> dict[str, Any]:
    """Check that a published result is readable and internally consistent.

    Confirms the required files exist and can be read, that the manifest and
    preprocess report are JSON objects, and that schema, source hash, parser
    version, and preprocess version match. Every ``elements.jsonl`` line must
    be JSON, and the line count must equal ``element_count``.

    This does not store or compare a content digest. A rewrite that keeps the
    same line count and still parses as JSON is not detected. A ``.tmp-*``
    directory is not this result and is not treated as ``COMPLETE``.
    """
    required = (
        "parse_manifest.json",
        "elements.jsonl",
        "preview.md",
        "preprocess_report.json",
        "COMPLETE",
    )
    for name in required:
        path = dest / name
        if not path.is_file():
            raise ParseInputError(f"missing {name}")
        path.read_bytes()
    manifest = _read_json(dest / "parse_manifest.json")
    report = _read_json(dest / "preprocess_report.json")
    if not isinstance(manifest, dict) or not isinstance(report, dict):
        raise ParseInputError("manifest or preprocess report is not readable JSON")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ParseInputError("schema_version does not match")
    if manifest.get("source_sha256") != source.source_sha256:
        raise ParseInputError("existing parse result points at a different source hash")
    if manifest.get("parser_version") != PARSER_VERSION:
        raise ParseInputError("parser_version does not match")
    if manifest.get("preprocess_version") != PREPROCESS_VERSION:
        raise ParseInputError("preprocess_version does not match")
    if report.get("preprocess_version") != PREPROCESS_VERSION:
        raise ParseInputError("preprocess report version does not match")
    if report.get("source_sha256") != source.source_sha256:
        raise ParseInputError("preprocess report source hash does not match")
    preview = (dest / "preview.md").read_text(encoding="utf-8")
    if not preview.strip():
        raise ParseInputError("preview.md is empty")
    count = 0
    for line in (dest / "elements.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        json.loads(line)
        count += 1
    if count != manifest.get("element_count"):
        raise ParseInputError("existing elements.jsonl does not match the manifest count")
    return manifest


def _quarantine_result(dest: Path, *, reason: str) -> None:
    if not dest.exists():
        return
    quarantine = dest.parent / "quarantine"
    quarantine.mkdir(parents=True, exist_ok=True)
    target = quarantine / f"{dest.name}-{uuid4().hex[:8]}"
    os.rename(dest, target)
    (target / "QUARANTINE.txt").write_text(reason + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _local(tag: str) -> str:
    if not isinstance(tag, str):
        return str(tag)
    return tag.rsplit("}", 1)[-1]


def _positive_int(value: str | None, *, default: int) -> int:
    if value is None or not str(value).strip():
        return default
    try:
        parsed = int(str(value).strip())
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value

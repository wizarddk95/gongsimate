# ruff: noqa: E501
"""Index parse-success filings without touching the Isu Petasys product index.

Partial filings stay out. ``assert_search_eligible`` records the existing gate.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from disclosure_agent.agent.catalog import repo_root
from disclosure_agent.agent.clock import request_time_context
from disclosure_agent.agent.corpus import (
    CORPUS_AS_OF_DEFINITION,
    INCLUSION_RULE,
    PERIODIC_DOCUMENT_NAMES,
    IndexedMember,
)
from disclosure_agent.agent.temporal import filing_period_from_listing
from disclosure_agent.chunking import assert_search_eligible, load_json
from disclosure_agent.dense import (
    DENSE_MODEL_ID,
    EMBEDDING_DIMENSION,
    _bge_token_counter,
    _restore_tokenizer_limit,
    encode_texts,
    load_dense_model,
    resolve_device,
)
from disclosure_agent.errors import UsageError
from disclosure_agent.product_index import _project_member
from disclosure_agent.retrieval_unit import D2A_VERSION

ROOT = repo_root()
OUT = ROOT / "evaluation" / "generalization_smoke"
ISU_INDEX = ROOT / "data" / "product" / "indexes" / "isu_petasys_periodic_d2a" / "corpus.json"
INDEX_ROOT = ROOT / "data" / "product" / "indexes" / "generalization"


def main() -> None:
    isu_before = ISU_INDEX.stat().st_mtime if ISU_INDEX.is_file() else None
    summary = json.loads((OUT / "parse_summary.json").read_text(encoding="utf-8"))
    blocked = _record_partial_gate(summary["rows"])
    eligible = [
        row
        for row in summary["rows"]
        if row.get("parse_status") == "success" and row.get("role") in {"primary", "correction"}
    ]
    by_company: dict[str, list[dict[str, Any]]] = {}
    for row in eligible:
        by_company.setdefault(str(row["corp_code"]), []).append(row)
    pending = {
        corp_code: [row for row in rows if not _already_indexed(corp_code, str(row["receipt_no"]))]
        for corp_code, rows in by_company.items()
    }
    pending = {corp_code: rows for corp_code, rows in pending.items() if rows}
    if not pending:
        print("no new success filings to encode", flush=True)
        return
    device = resolve_device("auto")
    batch_size = 32 if device == "cuda" else 8
    print(f"loading BAAI/bge-m3 on {device} batch_size={batch_size}", flush=True)
    started = time.perf_counter()
    model = load_dense_model(device)
    load_s = time.perf_counter() - started
    token_counter = _bge_token_counter(model.tokenizer)
    written = []
    try:
        for corp_code, rows in pending.items():
            written.append(_index_company(corp_code, rows, model, token_counter, batch_size))
    finally:
        _restore_tokenizer_limit(model.tokenizer)
    isu_after = ISU_INDEX.stat().st_mtime if ISU_INDEX.is_file() else None
    payload = {
        "device": device,
        "batch_size": batch_size,
        "model_load_seconds": round(load_s, 3),
        "isu_product_index_mtime_unchanged": isu_before == isu_after,
        "partial_chunk_gate": blocked,
        "indexes": written,
    }
    path = OUT / "index_summary.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path} companies={len(written)}", flush=True)


def _record_partial_gate(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    blocked = []
    for row in rows:
        if row.get("parse_status") == "success":
            continue
        manifest_path = Path(str(row.get("manifest") or ""))
        if not manifest_path.is_file():
            continue
        manifest = load_json(manifest_path)
        periodic = [
            member
            for member in manifest.get("members") or []
            if isinstance(member, dict) and member.get("document_name") in PERIODIC_DOCUMENT_NAMES
        ]
        if not periodic:
            continue
        try:
            assert_search_eligible(manifest)
            reason = "assert_search_eligible accepted a non-success manifest"
        except UsageError as exc:
            reason = str(exc)
        blocked.append(
            {
                "corp_code": str(row.get("corp_code") or ""),
                "receipt_no": str(row.get("receipt_no") or ""),
                "report_nm": str(row.get("report_nm") or ""),
                "parse_status": str(row.get("parse_status") or ""),
                "chunk_result": reason,
            }
        )
        print(f"chunk gate {row.get('corp_name')} {row.get('report_nm')}", flush=True)
    return blocked


def _index_company(
    corp_code: str,
    rows: list[dict[str, Any]],
    model: Any,
    token_counter: Any,
    batch_size: int,
) -> dict[str, Any]:
    members = tuple(_indexed_member(row) for row in rows)
    units: list[dict[str, Any]] = []
    member_rows = []
    for member in members:
        projected, skipped = _project_member(member, token_counter)
        units.extend(projected)
        member_rows.append(
            {
                "corp_code": member.corp_code,
                "company_name": member.company_name,
                "receipt_no": member.receipt_no,
                "report_name": member.report_name,
                "report_type": member.report_type,
                "receipt_date": member.receipt_date,
                "period_end_year": member.period_end_year,
                "period_end_month": member.period_end_month,
                "filing_period": (
                    f"{member.period_end_year}.{member.period_end_month:02d}"
                    if member.period_end_year and member.period_end_month
                    else ""
                ),
                "remark": member.remark,
                "source_sha256": member.source_sha256,
                "document_id": member.document_id,
                "document_name": member.document_name,
                "elements_path": member.elements_path.resolve().relative_to(ROOT.resolve()).as_posix(),
                "retrieval_unit_count": len(projected),
                "excluded_chunks": skipped,
                "product_eligible": True,
            }
        )
        print(f"  {member.report_name} units={len(projected)}", flush=True)
    corp_codes = {str(unit.get("corp_code") or "") for unit in units}
    if corp_codes != {corp_code}:
        raise SystemExit(f"cross-company units in {corp_code}: {sorted(corp_codes)}")
    started = time.perf_counter()
    import numpy as np

    matrix = np.zeros((len(units), EMBEDDING_DIMENSION), dtype=np.float32)
    texts = [str(unit.get("text") or "") for unit in units]
    encoded = encode_texts(model, texts, batch_size, show_progress_bar=True)
    for index, vector in enumerate(encoded):
        matrix[index] = vector
    encode_seconds = round(time.perf_counter() - started, 3)
    dest = INDEX_ROOT / corp_code
    prior_units, prior_matrix, prior_members = _load_existing(dest)
    if prior_units:
        units = prior_units + units
        member_rows = prior_members + member_rows
        matrix = np.vstack([prior_matrix, matrix])
    verified = [
        datetime.fromisoformat(member.last_success_verified_at)
        for member in members
        if member.last_success_verified_at
    ]
    if prior_members:
        for row in prior_members:
            stamp = str(row.get("last_success_verified_at") or "")
            if stamp:
                verified.append(datetime.fromisoformat(stamp))
    corpus_as_of = request_time_context(max(verified))["iso"]
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / "retrieval_units.jsonl").open("w", encoding="utf-8") as handle:
        for unit in units:
            handle.write(json.dumps(unit, ensure_ascii=False) + "\n")
    matrix.astype("float32", copy=False).tofile(dest / "embeddings.f32")
    manifest = {
        "schema_version": 1,
        "corpus_id": f"generalization_{corp_code}",
        "representation_version": D2A_VERSION,
        "embedding_model": DENSE_MODEL_ID,
        "corp_code": corp_code,
        "company_name": members[0].company_name,
        "corpus_as_of": corpus_as_of,
        "corpus_as_of_definition": CORPUS_AS_OF_DEFINITION,
        "inclusion_rule": INCLUSION_RULE,
        "product_index_note": (
            "Smoke index of overall_status=success filings only. "
            "The Isu Petasys product index was not rewritten."
        ),
        "retrieval_unit_count": len(units),
        "members": member_rows,
    }
    (dest / "corpus.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "corp_code": corp_code,
        "company_name": members[0].company_name,
        "index_dir": dest.resolve().relative_to(ROOT.resolve()).as_posix(),
        "filings": len(member_rows),
        "retrieval_unit_count": len(units),
        "encode_seconds": encode_seconds,
        "corpus_as_of": corpus_as_of,
        "embedding_model": DENSE_MODEL_ID,
        "receipt_nos": [member.receipt_no for member in members],
    }


def _already_indexed(corp_code: str, receipt_no: str) -> bool:
    manifest_path = INDEX_ROOT / corp_code / "corpus.json"
    if not manifest_path.is_file():
        return False
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return any(
        isinstance(member, dict) and str(member.get("receipt_no") or "") == receipt_no
        for member in payload.get("members") or []
    )


def _load_existing(dest: Path) -> tuple[list[dict[str, Any]], Any, list[dict[str, Any]]]:
    import numpy as np

    manifest_path = dest / "corpus.json"
    units_path = dest / "retrieval_units.jsonl"
    matrix_path = dest / "embeddings.f32"
    if not manifest_path.is_file() or not units_path.is_file() or not matrix_path.is_file():
        return [], None, []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    units = [
        json.loads(line)
        for line in units_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matrix = np.fromfile(matrix_path, dtype=np.float32)
    if matrix.size != len(units) * EMBEDDING_DIMENSION:
        raise SystemExit(f"existing matrix rows do not match units in {dest}")
    matrix = matrix.reshape(len(units), EMBEDDING_DIMENSION)
    members = [member for member in manifest.get("members") or [] if isinstance(member, dict)]
    return units, matrix, members


def _indexed_member(row: dict[str, Any]) -> IndexedMember:
    receipt_no = str(row["receipt_no"])
    sha = str(row["source_sha256"])
    version_path = ROOT / "data" / "raw" / "dart" / receipt_no / "versions" / sha / "version.json"
    current_path = ROOT / "data" / "raw" / "dart" / receipt_no / "current.json"
    version = json.loads(version_path.read_text(encoding="utf-8"))
    current = json.loads(current_path.read_text(encoding="utf-8"))
    listing = version.get("listing") if isinstance(version.get("listing"), dict) else {}
    period = filing_period_from_listing(listing)
    manifest_path = Path(str(row["manifest"]))
    manifest = load_json(manifest_path)
    periodic = [
        member
        for member in manifest.get("members") or []
        if isinstance(member, dict)
        and member.get("document_name") in PERIODIC_DOCUMENT_NAMES
        and member.get("status") == "parsed"
    ]
    if len(periodic) != 1:
        names = [
            member.get("document_name")
            for member in manifest.get("members") or []
            if isinstance(member, dict)
        ]
        raise SystemExit(f"{receipt_no} expected one parsed periodic member, saw {names}")
    member = periodic[0]
    return IndexedMember(
        corp_code=str(row["corp_code"]),
        company_name=str(row["corp_name"]),
        receipt_no=receipt_no,
        report_name=period.report_name,
        report_type=period.report_type,
        receipt_date=period.receipt_date,
        period_end_year=period.period_end_year,
        period_end_month=period.period_end_month,
        remark=period.remark,
        source_sha256=sha,
        document_id=str(member.get("document_id") or ""),
        document_name=str(member.get("document_name") or ""),
        member_index=int(member.get("member_index") or 0),
        member_name=str(member.get("member_name") or ""),
        elements_path=manifest_path.parent / "elements.jsonl",
        manifest_path=manifest_path,
        last_success_verified_at=str(current.get("last_success_verified_at") or ""),
    )


if __name__ == "__main__":
    main()

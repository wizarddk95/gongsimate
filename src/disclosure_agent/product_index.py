"""Build the product Dense-D2a index from index-eligible periodic filings.

The locked evaluation run stays a single FY2025 baseline. This writer does
not overwrite it. FY2025 vectors are copied when the retrieval unit id and
text still match that run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from disclosure_agent.agent.catalog import repo_root
from disclosure_agent.agent.clock import request_time_context
from disclosure_agent.agent.corpus import (
    CORPUS_AS_OF_DEFINITION,
    INCLUSION_RULE,
    IndexedMember,
    plan_corpus,
    product_index_dir,
)
from disclosure_agent.chunking import chunk_elements, load_elements, load_json
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
from disclosure_agent.retrieval_unit import D2A_VERSION, build_retrieval_units

_REUSE_RUN = Path("evaluation/runs/isu_petasys_v0_3_dense_d2a")


def build_product_index(
    root: Path | None = None,
    *,
    device: str = "auto",
    batch_size: int | None = None,
) -> tuple[dict[str, Any], Any]:
    """Chunk, project D2a units, and encode. Returns the manifest and the model."""
    base = root or repo_root()
    plan = plan_corpus(base)
    if plan.corpus_as_of is None or not plan.members:
        raise UsageError("no index-eligible periodic filing with a verified timestamp")
    companies = {member.corp_code for member in plan.members}
    if len(companies) != 1:
        raise UsageError(f"product index expected one company, got {sorted(companies)}")
    resolved = resolve_device(device)
    if batch_size is None:
        batch_size = 32 if resolved == "cuda" else 8
    if batch_size < 1:
        raise UsageError("batch_size must be >= 1")
    print(f"loading BAAI/bge-m3 on {resolved} batch_size={batch_size}", flush=True)
    model = load_dense_model(resolved)
    token_counter = _bge_token_counter(model.tokenizer)
    try:
        units, member_rows, projection_excluded = _project_members(plan.members, token_counter)
    finally:
        _restore_tokenizer_limit(model.tokenizer)
    if not units:
        raise UsageError("projection produced no retrieval units")
    reuse_units, reuse_matrix = _load_reuse(base)
    matrix, reuse_info = _encode(model, units, reuse_units, reuse_matrix, batch_size)
    manifest = _write_index(
        base,
        units,
        matrix,
        member_rows,
        projection_excluded,
        plan,
        reuse_info,
    )
    print(
        f"wrote {manifest['retrieval_unit_count']} retrieval units "
        f"across {len(manifest['members'])} filings",
        flush=True,
    )
    return manifest, model


def _project_members(
    members: tuple[IndexedMember, ...],
    token_counter: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
    units: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for member in members:
        print(f"project {member.receipt_no} {member.report_name}", flush=True)
        try:
            projected, skipped = _project_member(member, token_counter)
        except (UsageError, TypeError, ValueError, KeyError) as exc:
            excluded.append(
                {
                    "receipt_no": member.receipt_no,
                    "report_name": member.report_name,
                    "reason": f"projection failed: {type(exc).__name__}: {exc}",
                }
            )
            print(f"  excluded: {type(exc).__name__}: {exc}", flush=True)
            continue
        units.extend(projected)
        record = _member_record(member, len(projected))
        record["excluded_chunks"] = skipped
        rows.append(record)
        print(f"  retrieval_units={len(projected)} skipped_chunks={len(skipped)}", flush=True)
    return units, rows, excluded


def _project_member(
    member: IndexedMember, token_counter: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = load_json(member.manifest_path)
    elements = load_elements(member.elements_path)
    chunks = chunk_elements(elements, manifest=manifest, document_id=member.document_id)
    elements_by_id = {
        str(element["element_id"]): element
        for element in elements
        if element.get("element_id")
    }
    projected, skipped = _project_chunks(chunks, elements_by_id, token_counter)
    label = period_label_of(member)
    for unit in projected:
        unit["corp_code"] = member.corp_code
        unit["receipt_no"] = member.receipt_no
        unit["report_name"] = member.report_name
        unit["report_type"] = member.report_type
        unit["document_name"] = member.document_name
        unit["document_id"] = member.document_id
        unit["filing_period"] = label
        unit["period_end_year"] = member.period_end_year
        unit["period_end_month"] = member.period_end_month
        unit["receipt_date"] = member.receipt_date
        unit["source_sha256"] = member.source_sha256
        unit["remark"] = member.remark
    return projected, skipped


def _project_chunks(
    chunks: list[dict[str, Any]],
    elements_by_id: dict[str, Any],
    token_counter: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        return (
            build_retrieval_units(
                chunks,
                elements_by_id,
                mode="d2a",
                token_counter=token_counter,
            ),
            [],
        )
    except UsageError:
        projected: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for chunk in chunks:
            try:
                projected.extend(
                    build_retrieval_units(
                        [chunk],
                        elements_by_id,
                        mode="d2a",
                        token_counter=token_counter,
                    )
                )
            except UsageError as exc:
                skipped.append(
                    {
                        "ordinal": chunk.get("ordinal"),
                        "chunk_id": chunk.get("chunk_id"),
                        "reason": str(exc),
                    }
                )
        if not projected:
            reason = skipped[0]["reason"] if skipped else "no chunks"
            raise UsageError(f"projection failed for every chunk: {reason}") from None
        return projected, skipped


def period_label_of(member: IndexedMember) -> str:
    from disclosure_agent.agent.temporal import FilingPeriod, period_label

    return period_label(
        FilingPeriod(
            receipt_no=member.receipt_no,
            report_name=member.report_name,
            report_type=member.report_type,
            receipt_date=member.receipt_date,
            period_end_year=member.period_end_year,
            period_end_month=member.period_end_month,
            remark=member.remark,
        )
    )


def _encode(
    model: Any,
    units: list[dict[str, Any]],
    reuse_units: list[dict[str, Any]],
    reuse_matrix: Any,
    batch_size: int,
) -> tuple[Any, dict[str, int]]:
    import numpy as np

    by_id = {str(unit["retrieval_unit_id"]): index for index, unit in enumerate(reuse_units)}
    matrix = np.zeros((len(units), EMBEDDING_DIMENSION), dtype=np.float32)
    pending_at: list[int] = []
    pending_text: list[str] = []
    reused = 0
    for index, unit in enumerate(units):
        source = by_id.get(str(unit["retrieval_unit_id"]))
        if (
            source is not None
            and reuse_matrix is not None
            and reuse_units[source].get("text") == unit.get("text")
        ):
            matrix[index] = reuse_matrix[source]
            reused += 1
            continue
        pending_at.append(index)
        pending_text.append(str(unit.get("text") or ""))
    print(f"encode {len(pending_text)} units, reuse {reused}", flush=True)
    if pending_text:
        encoded = encode_texts(model, pending_text, batch_size, show_progress_bar=True)
        for index, vector in zip(pending_at, encoded, strict=True):
            matrix[index] = vector
    return matrix, {"reused_row_count": reused, "encoded_row_count": len(pending_text)}


def _load_reuse(root: Path) -> tuple[list[dict[str, Any]], Any]:
    run = root / _REUSE_RUN
    units_path = run / "retrieval_units.jsonl"
    embeddings_path = run / "embeddings.f32"
    if not units_path.is_file() or not embeddings_path.is_file():
        return [], None
    import numpy as np

    units = [
        json.loads(line)
        for line in units_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    raw = np.fromfile(embeddings_path, dtype=np.float32)
    if raw.size != len(units) * EMBEDDING_DIMENSION:
        return [], None
    return units, raw.reshape(len(units), EMBEDDING_DIMENSION)


def _write_index(
    root: Path,
    units: list[dict[str, Any]],
    matrix: Any,
    member_rows: list[dict[str, Any]],
    projection_excluded: list[dict[str, str]],
    plan: Any,
    reuse_info: dict[str, int],
) -> dict[str, Any]:
    dest = product_index_dir(root)
    dest.mkdir(parents=True, exist_ok=True)
    units_path = dest / "retrieval_units.jsonl"
    with units_path.open("w", encoding="utf-8") as handle:
        for unit in units:
            handle.write(json.dumps(unit, ensure_ascii=False) + "\n")
    embeddings_path = dest / "embeddings.f32"
    matrix.astype("float32", copy=False).tofile(embeddings_path)
    company = plan.members[0]
    manifest = {
        "schema_version": 1,
        "corpus_id": "isu_petasys_periodic_d2a",
        "representation_version": D2A_VERSION,
        "embedding_model": DENSE_MODEL_ID,
        "corp_code": company.corp_code,
        "company_name": company.company_name,
        "corpus_as_of": request_time_context(plan.corpus_as_of)["iso"],
        "corpus_as_of_definition": CORPUS_AS_OF_DEFINITION,
        "inclusion_rule": INCLUSION_RULE,
        "retrieval_unit_count": len(units),
        "reuse": reuse_info,
        "members": member_rows,
        "excluded_filings": [
            {
                "receipt_no": row.receipt_no,
                "report_name": row.report_name,
                "parse_status": row.parse_status,
                "reason": row.index_reason,
            }
            for row in plan.rows
            if not row.index_eligible
        ],
        "projection_excluded": projection_excluded,
        "inventory": [_inventory_dict(row) for row in plan.rows],
    }
    (dest / "corpus.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _member_record(member: IndexedMember, unit_count: int) -> dict[str, Any]:
    root = repo_root()
    return {
        "corp_code": member.corp_code,
        "company_name": member.company_name,
        "receipt_no": member.receipt_no,
        "report_name": member.report_name,
        "report_type": member.report_type,
        "receipt_date": member.receipt_date,
        "period_end_year": member.period_end_year,
        "period_end_month": member.period_end_month,
        "filing_period": period_label_of(member),
        "remark": member.remark,
        "source_sha256": member.source_sha256,
        "document_id": member.document_id,
        "document_name": member.document_name,
        "member_index": member.member_index,
        "member_name": member.member_name,
        "elements_path": _relative(root, member.elements_path),
        "retrieval_unit_count": unit_count,
    }


def _inventory_dict(row: Any) -> dict[str, Any]:
    return {
        "receipt_no": row.receipt_no,
        "report_name": row.report_name,
        "report_type": row.report_type,
        "receipt_date": row.receipt_date,
        "period_label": row.period_label,
        "remark": row.remark,
        "source_sha256": row.source_sha256,
        "downloaded": row.downloaded,
        "parse_status": row.parse_status,
        "complete": row.complete,
        "element_count": row.element_count,
        "index_eligible": row.index_eligible,
        "index_reason": row.index_reason,
        "periodic_documents": row.periodic_documents,
        "excluded_members": row.excluded_members,
        "last_success_verified_at": row.last_success_verified_at,
    }


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    manifest, model = build_product_index(repo_root())
    if "--probe" in args:
        from disclosure_agent.product_probe import run_temporal_probe

        run_temporal_probe(repo_root(), model)


if __name__ == "__main__":
    main()

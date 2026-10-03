"""Product pointer at the locked Dense-D2a corpus. This module does not write it."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from disclosure_agent.dense import DENSE_MODEL_ID
from disclosure_agent.retrieval_unit import D2A_VERSION

# The repository's canonical company id is the OpenDART corp_code.
# Snapshot identity below is read at runtime; this fallback matches that snapshot
# when the file is unavailable to a unit test that injects its own binding.
ISU_PETASYS_CORP_CODE = "00107613"


@dataclass(frozen=True)
class DenseD2aBinding:
    corp_code: str
    company_name: str
    units_path: Path
    embeddings_path: Path
    elements_path: Path | None
    receipt_no: str
    document_name: str
    representation_version: str
    embedding_model: str
    elements_paths: tuple[Path, ...] = ()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_dense_d2a_binding(root: Path | None = None) -> DenseD2aBinding:
    """Identity fields come from the v0.3 snapshot. Retrieval units stay in the D2a run."""
    base = root or repo_root()
    snapshot_path = base / "evaluation" / "datasets" / "isu_petasys_v0_3" / "snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    scope = snapshot["index_scope"]
    receipt_no = str(snapshot["receipt_no"])
    elements = (
        base
        / "data"
        / "parsed"
        / "dart"
        / receipt_no
        / str(snapshot["source_sha256"])
        / "p2"
        / str(snapshot["config_hash"])[:16]
        / "elements.jsonl"
    )
    run = base / "evaluation" / "runs" / "isu_petasys_v0_3_dense_d2a"
    return DenseD2aBinding(
        corp_code=str(snapshot["corp_code"]),
        company_name=str(snapshot["corp_name"]),
        units_path=run / "retrieval_units.jsonl",
        embeddings_path=run / "embeddings.f32",
        elements_path=elements if elements.is_file() else None,
        receipt_no=receipt_no,
        document_name=str(scope["document_name_observed"]),
        representation_version=D2A_VERSION,
        embedding_model=DENSE_MODEL_ID,
    )

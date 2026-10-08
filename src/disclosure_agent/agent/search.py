"""search_disclosures: Dense-D2a exact search behind a tool boundary.

Document vectors are the locked D2a embedding matrix. The query vector uses the
same BGE-M3 encoder contract as the dense baseline. Gold and evaluation labels
are not read and are stripped if a unit happens to carry them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from disclosure_agent.agent.catalog import DenseD2aBinding
from disclosure_agent.agent.errors import RetrievalFailure
from disclosure_agent.agent.settings import DEFAULT_RETRIEVAL_TOP_K
from disclosure_agent.dense import (
    EMBEDDING_DIMENSION,
    encode_texts,
    exact_normalized_search,
    load_dense_model,
    resolve_device,
)
from disclosure_agent.retrieval_unit import D2A_VERSION, scoring_chunks

_FORBIDDEN_KEYS = {
    "gold",
    "gold_spans",
    "gold_answer",
    "evidence_class",
    "question_id",
    "hard_negative",
    "in_positive_recall_denominator",
    "dataset_id",
    "label",
}
_SPAN_KEYS = (
    "document_id",
    "element_id",
    "char_start",
    "char_end",
    "kind",
    "role",
    "order_index",
)


class QueryEncoder(Protocol):
    def encode(self, text: str) -> Any: ...


@dataclass
class SearchDisclosuresRequest:
    query: str
    company_id: str
    corp_code: str | None = None
    fiscal_year: int | None = None
    period: str | None = None
    as_of_date: str | None = None
    top_k: int = DEFAULT_RETRIEVAL_TOP_K
    eligible_receipt_nos: tuple[str, ...] | None = None


@dataclass
class SearchHit:
    retrieval_unit_id: str
    score: float
    rank: int
    text: str
    canonical_chunk_id: str
    unit_index: int
    source_element_ids: list[str]
    source_refs: list[dict[str, Any]]
    receipt_no: str
    document_name: str
    section_path: list[str]
    source_locator: list[dict[str, Any]]
    report_name: str = ""
    filing_period: str = ""
    receipt_date: str = ""
    source_sha256: str = ""
    corp_code: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SearchDisclosuresResult:
    hits: list[SearchHit]
    retriever_type: str
    representation_version: str
    embedding_model: str
    company_id: str
    corp_code: str
    fiscal_year: int | None = None
    retrieval_policy: str = ""
    retrieval_edit: str | None = None
    policy_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


@dataclass
class DenseD2aIndex:
    units: list[dict[str, Any]]
    matrix: Any
    binding: DenseD2aBinding
    scoring: list[dict[str, Any]] = field(init=False)
    by_id: dict[str, dict[str, Any]] = field(init=False)

    def __post_init__(self) -> None:
        self.scoring = scoring_chunks(self.units)
        self.by_id = {str(unit["retrieval_unit_id"]): unit for unit in self.units}


class BgeM3QueryEncoder:
    """Lazy BGE-M3 query encoder. The model loads once on this instance.

    Document embeddings are not recomputed. A later request on the same
    search object does not load the encoder again. Startup warmup is a TODO.
    """

    def __init__(self) -> None:
        self._model: Any = None

    def encode(self, text: str) -> Any:
        if self._model is None:
            self._model = load_dense_model(resolve_device("auto"))
        return encode_texts(self._model, [text], batch_size=1)[0]


class DenseD2aSearch:
    """Tool implementation bound to one company's Dense-D2a index."""

    def __init__(
        self,
        binding: DenseD2aBinding,
        encoder: QueryEncoder | None = None,
        *,
        index: DenseD2aIndex | None = None,
    ) -> None:
        self.binding = binding
        self.encoder = encoder or BgeM3QueryEncoder()
        self._index = index

    def __call__(self, request: SearchDisclosuresRequest) -> SearchDisclosuresResult:
        return search_disclosures(request, self.index(), self.encoder, self.binding)

    def index(self) -> DenseD2aIndex:
        if self._index is None:
            self._index = load_dense_d2a_index(self.binding)
        return self._index


def search_disclosures(
    request: SearchDisclosuresRequest,
    index: DenseD2aIndex,
    encoder: QueryEncoder,
    binding: DenseD2aBinding,
) -> SearchDisclosuresResult:
    """Rank the D2a matrix. ``fiscal_year`` is recorded and is not a hard filter.

    ``eligible_receipt_nos`` restricts the matrix before ranking. ``None`` searches
    every row. An empty tuple searches nothing.
    """
    corp_code = request.corp_code or request.company_id
    if corp_code != binding.corp_code or request.company_id != binding.corp_code:
        raise RetrievalFailure(
            "search_disclosures company_id does not match the Dense-D2a index"
        )
    if request.top_k < 1:
        raise RetrievalFailure("top_k must be >= 1")
    units, matrix, scoring = _eligible_rows(index, binding, request.eligible_receipt_nos)
    if not units:
        return SearchDisclosuresResult(
            hits=[],
            retriever_type="dense",
            representation_version=binding.representation_version,
            embedding_model=binding.embedding_model,
            company_id=binding.corp_code,
            corp_code=binding.corp_code,
            fiscal_year=request.fiscal_year,
        )
    try:
        vector = encoder.encode(request.query)
        ranked = exact_normalized_search(matrix, vector, scoring)
    except RetrievalFailure:
        raise
    except Exception as exc:
        raise RetrievalFailure(f"dense search failed: {type(exc).__name__}") from exc
    hits: list[SearchHit] = []
    for item in ranked[: request.top_k]:
        unit_id = str(item.get("retrieval_unit_id") or item["chunk_id"])
        unit = next((row for row in units if str(row["retrieval_unit_id"]) == unit_id), None)
        if unit is None:
            raise RetrievalFailure(f"ranked id {unit_id} is not in the D2a unit catalog")
        hits.append(_project_hit(unit, item, binding))
    return SearchDisclosuresResult(
        hits=hits,
        retriever_type="dense",
        representation_version=binding.representation_version,
        embedding_model=binding.embedding_model,
        company_id=binding.corp_code,
        corp_code=binding.corp_code,
        fiscal_year=request.fiscal_year,
    )


def load_dense_d2a_index(binding: DenseD2aBinding) -> DenseD2aIndex:
    if binding.representation_version != D2A_VERSION:
        raise RetrievalFailure(
            f"refusing non-D2a binding {binding.representation_version}"
        )
    if not binding.units_path.is_file():
        raise RetrievalFailure(f"D2a retrieval units are missing: {binding.units_path}")
    if not binding.embeddings_path.is_file():
        raise RetrievalFailure(f"D2a embeddings are missing: {binding.embeddings_path}")
    units = _load_units(binding.units_path)
    matrix = _load_matrix(binding.embeddings_path, len(units))
    return DenseD2aIndex(units=units, matrix=matrix, binding=binding)


def _load_units(path: Any) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RetrievalFailure("D2a retrieval unit row is not an object")
            if row.get("representation_version") != D2A_VERSION:
                raise RetrievalFailure("retrieval unit is not D2a")
            units.append(row)
    if not units:
        raise RetrievalFailure("D2a retrieval unit artifact is empty")
    return units


def _load_matrix(path: Any, rows: int) -> Any:
    import numpy as np

    raw = np.fromfile(path, dtype=np.float32)
    expected = rows * EMBEDDING_DIMENSION
    if raw.size != expected:
        raise RetrievalFailure(
            f"D2a embedding rows do not match retrieval units ({raw.size} != {expected})"
        )
    return raw.reshape(rows, EMBEDDING_DIMENSION)


def _eligible_rows(
    index: DenseD2aIndex,
    binding: DenseD2aBinding,
    receipt_nos: tuple[str, ...] | None,
) -> tuple[list[dict[str, Any]], Any, list[dict[str, Any]]]:
    if receipt_nos is None:
        return index.units, index.matrix, index.scoring
    allowed = set(receipt_nos)
    keep = [
        offset
        for offset, unit in enumerate(index.units)
        if _unit_receipt(unit, binding) in allowed
    ]
    if not keep:
        return [], index.matrix[:0], []
    import numpy as np

    units = [index.units[offset] for offset in keep]
    scoring = [index.scoring[offset] for offset in keep]
    matrix = np.asarray(index.matrix)[keep]
    return units, matrix, scoring


def _unit_receipt(unit: dict[str, Any], binding: DenseD2aBinding) -> str:
    receipt = unit.get("receipt_no")
    if receipt:
        return str(receipt)
    return binding.receipt_no


def _project_hit(
    unit: dict[str, Any],
    ranked: dict[str, Any],
    binding: DenseD2aBinding,
) -> SearchHit:
    source_refs = _scrub(list(unit.get("source_refs") or []))
    element_ids = [str(item) for item in unit.get("source_element_ids") or []]
    section = [str(item) for item in unit.get("section_path") or []]
    filing_period = str(unit.get("filing_period") or "")
    if not filing_period:
        year = unit.get("period_end_year")
        month = unit.get("period_end_month")
        if isinstance(year, int) and isinstance(month, int):
            filing_period = f"{year}.{month:02d}"
    return SearchHit(
        retrieval_unit_id=str(unit["retrieval_unit_id"]),
        score=float(ranked["score"]),
        rank=int(ranked["rank"]),
        text=str(unit.get("text") or ""),
        canonical_chunk_id=str(unit.get("canonical_chunk_id") or ""),
        unit_index=int(unit.get("unit_index") or 0),
        source_element_ids=element_ids,
        source_refs=source_refs,
        receipt_no=str(unit.get("receipt_no") or binding.receipt_no),
        document_name=str(unit.get("document_name") or binding.document_name),
        section_path=section,
        source_locator=[_span_locator(ref) for ref in source_refs if isinstance(ref, dict)],
        report_name=str(unit.get("report_name") or ""),
        filing_period=filing_period,
        receipt_date=str(unit.get("receipt_date") or ""),
        source_sha256=str(unit.get("source_sha256") or ""),
        corp_code=str(unit.get("corp_code") or binding.corp_code),
    )


def _span_locator(ref: dict[str, Any]) -> dict[str, Any]:
    return {key: ref[key] for key in _SPAN_KEYS if key in ref}


def _scrub(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _scrub(item)
            for key, item in value.items()
            if key not in _FORBIDDEN_KEYS and not str(key).startswith("gold")
        }
    if isinstance(value, list):
        return [_scrub(item) for item in value]
    return value

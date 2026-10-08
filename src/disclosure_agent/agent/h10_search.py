"""Product search: BM25-primary Dense-head on the Dense-D2a unit universe.

The order is ``bm25_primary_dense_head`` from the offline H10 experiment.
This module does not train, fuse scores, or add a reranker. A request that
cannot supply the fixed candidate window is ``h10_unavailable``. Dense-D2a
order is not substituted for that request.
"""

from __future__ import annotations

from dataclasses import replace

from disclosure_agent.agent.corpus import ProductCorpus
from disclosure_agent.agent.errors import RetrievalFailure
from disclosure_agent.agent.search import (
    SearchDisclosuresRequest,
    SearchDisclosuresResult,
    SearchHit,
    _eligible_rows,
    _project_hit,
)
from disclosure_agent.agent.search_catalog import CatalogDenseSearch, SearchCatalog
from disclosure_agent.agent.settings import (
    PRODUCT_RETRIEVAL_POLICY,
    PRODUCT_RETRIEVAL_SHORT_LIST_POLICY,
    PRODUCT_RETRIEVAL_SUPPORT,
)
from disclosure_agent.bm25 import Bm25Index
from disclosure_agent.errors import UsageError
from disclosure_agent.evalset_v2_1.asymmetric_fusion import (
    CANDIDATE_N,
    PROTECTED_PREFIX,
    bm25_primary_dense_head,
)
from disclosure_agent.evalset_v2_1.complementarity import bm25_chunks
from disclosure_agent.retrieval_unit import D2A_VERSION

SUPPORT_SCOPE = PRODUCT_RETRIEVAL_SUPPORT


class H10Search:
    """Rank one company's effective filings with the locked H10 placement."""

    def __init__(self, catalog: SearchCatalog, encoder: object | None = None) -> None:
        self.catalog = catalog
        self.dense = CatalogDenseSearch(catalog, encoder)  # type: ignore[arg-type]
        self._bm25: dict[tuple[str, tuple[str, ...] | None], Bm25Index] = {}

    def __call__(self, request: SearchDisclosuresRequest) -> SearchDisclosuresResult:
        corpus = self.catalog.corpus(request.company_id)
        if corpus is None:
            raise RetrievalFailure("search company is not in the product catalog")
        if request.corp_code and request.corp_code != corpus.binding.corp_code:
            raise RetrievalFailure("search company_id does not match corp_code")
        wide = replace(request, top_k=CANDIDATE_N)
        dense_result = self.dense(wide)
        units, _matrix, scoring = self._eligible(corpus, request)
        if len(units) < CANDIDATE_N or len(dense_result.hits) < CANDIDATE_N:
            return _unavailable(request, corpus, "candidate_n_not_met")
        try:
            index = self._bm25_index(
                corpus.binding.corp_code,
                request.eligible_receipt_nos,
                units,
                scoring,
            )
        except UsageError as exc:
            raise RetrievalFailure("BM25 units do not match the dense retrieval units") from exc
        bm25_ranked = index.search(request.query, CANDIDATE_N)
        if len(bm25_ranked) < CANDIDATE_N:
            return _unavailable(request, corpus, "bm25_candidate_n_not_met")
        dense_rows = _dense_rows(dense_result.hits)
        bm25_rows = _bm25_rows(bm25_ranked, units)
        try:
            placed = bm25_primary_dense_head(dense_rows, bm25_rows)
        except UsageError as exc:
            return _unavailable(request, corpus, f"placement_rejected:{exc}")
        fused = placed["hits"]
        prefix = [hit["retrieval_unit_id"] for hit in fused[:PROTECTED_PREFIX]]
        bm25_prefix = [row["retrieval_unit_id"] for row in bm25_rows[:PROTECTED_PREFIX]]
        if prefix != bm25_prefix:
            return _unavailable(request, corpus, "prefix_left_bm25")
        by_id = {str(unit["retrieval_unit_id"]): unit for unit in units}
        dense_scores = {hit.retrieval_unit_id: hit.score for hit in dense_result.hits}
        bm25_scores = {
            str(hit.get("retrieval_unit_id") or hit["chunk_id"]): float(hit["score"])
            for hit in bm25_ranked
        }
        hits: list[SearchHit] = []
        for item in fused[: request.top_k]:
            unit_id = str(item["retrieval_unit_id"])
            unit = by_id.get(unit_id)
            if unit is None:
                return _unavailable(request, corpus, "fused_id_missing_from_units")
            elements = [str(value) for value in unit.get("source_element_ids") or []]
            if unit_id in set(elements):
                raise RetrievalFailure("retrieval unit id collapsed into a source element id")
            if elements and list(item.get("source_element_ids") or []) not in ([], elements):
                if list(item.get("source_element_ids") or []) != elements:
                    return _unavailable(request, corpus, "source_element_ids_disagree")
            score = bm25_scores.get(unit_id, dense_scores.get(unit_id, 0.0))
            hits.append(
                _project_hit(unit, {"score": score, "rank": int(item["rank"])}, corpus.binding)
            )
        return SearchDisclosuresResult(
            hits=hits,
            retriever_type=PRODUCT_RETRIEVAL_POLICY,
            representation_version=corpus.binding.representation_version,
            embedding_model=corpus.binding.embedding_model,
            company_id=corpus.binding.corp_code,
            corp_code=corpus.binding.corp_code,
            fiscal_year=request.fiscal_year,
            retrieval_policy=PRODUCT_RETRIEVAL_POLICY,
            retrieval_edit=str(placed["edit"]),
            policy_reason=None,
        )

    def _eligible(
        self,
        corpus: ProductCorpus,
        request: SearchDisclosuresRequest,
    ) -> tuple[list[dict], object, list[dict]]:
        search = self.dense._search.get(corpus.binding.corp_code)
        if search is None:
            raise RetrievalFailure("dense index was not loaded for the H10 company")
        return _eligible_rows(search.index(), corpus.binding, request.eligible_receipt_nos)

    def _bm25_index(
        self,
        corp_code: str,
        receipts: tuple[str, ...] | None,
        units: list[dict],
        scoring: list[dict],
    ) -> Bm25Index:
        key = (corp_code, None if receipts is None else tuple(receipts))
        cached = self._bm25.get(key)
        if cached is not None:
            return cached
        chunks = bm25_chunks(units, scoring)
        index = Bm25Index(chunks)
        self._bm25[key] = index
        return index


def _dense_rows(hits: list[SearchHit]) -> list[dict]:
    rows = []
    for index, hit in enumerate(hits[:CANDIDATE_N], start=1):
        rows.append(
            {
                "rank": index,
                "retrieval_unit_id": hit.retrieval_unit_id,
                "source_element_ids": list(hit.source_element_ids),
            }
        )
    return rows


def _bm25_rows(ranked: list[dict], units: list[dict]) -> list[dict]:
    elements = {
        str(unit["retrieval_unit_id"]): [str(item) for item in unit.get("source_element_ids") or []]
        for unit in units
    }
    rows = []
    for index, hit in enumerate(ranked[:CANDIDATE_N], start=1):
        unit_id = str(hit.get("retrieval_unit_id") or hit["chunk_id"])
        rows.append(
            {
                "rank": index,
                "retrieval_unit_id": unit_id,
                "source_element_ids": list(elements.get(unit_id) or []),
            }
        )
    return rows


def _unavailable(
    request: SearchDisclosuresRequest,
    corpus: ProductCorpus,
    reason: str,
) -> SearchDisclosuresResult:
    return SearchDisclosuresResult(
        hits=[],
        retriever_type=PRODUCT_RETRIEVAL_SHORT_LIST_POLICY,
        representation_version=corpus.binding.representation_version or D2A_VERSION,
        embedding_model=corpus.binding.embedding_model,
        company_id=corpus.binding.corp_code,
        corp_code=corpus.binding.corp_code,
        fiscal_year=request.fiscal_year,
        retrieval_policy=PRODUCT_RETRIEVAL_SHORT_LIST_POLICY,
        retrieval_edit=None,
        policy_reason=reason,
    )

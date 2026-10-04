"""Company indexes and the filing set a search is allowed to rank.

Physical indexes stay immutable. This module chooses one company's index,
then the filings that are effective for the validated tool arguments.
It does not read the user question and does not rank text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from disclosure_agent.agent.clock import request_time_context
from disclosure_agent.agent.corpus import ProductCorpus, discover_index_dirs, load_index_corpus
from disclosure_agent.agent.errors import RetrievalFailure
from disclosure_agent.agent.evidence import Evidence, EvidenceBuilder
from disclosure_agent.agent.filing_relation import (
    FilingRelation,
    load_relations,
    suppressed_originals,
)
from disclosure_agent.agent.search import (
    BgeM3QueryEncoder,
    DenseD2aSearch,
    QueryEncoder,
    SearchDisclosuresRequest,
    SearchDisclosuresResult,
    SearchHit,
)
from disclosure_agent.agent.temporal import FilingPeriod, receipt_on_or_before
from disclosure_agent.agent.tool_args import ToolResolution


@dataclass(frozen=True)
class SearchCatalog:
    """Indexes discovered from ``data/product/indexes``, keyed by corp_code."""

    root: Path
    corpora: tuple[ProductCorpus, ...]
    relations: tuple[FilingRelation, ...] = ()
    _by_corp: dict[str, ProductCorpus] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        by_corp: dict[str, ProductCorpus] = {}
        for corpus in self.corpora:
            corp_code = corpus.binding.corp_code
            if not corp_code:
                raise RetrievalFailure(f"product index has no corp_code: {corpus.index_dir}")
            previous = by_corp.get(corp_code)
            if previous is not None:
                raise RetrievalFailure(
                    f"two product indexes claim corp_code {corp_code}: "
                    f"{previous.index_dir} and {corpus.index_dir}"
                )
            by_corp[corp_code] = corpus
        object.__setattr__(self, "_by_corp", by_corp)

    def indexed_corp_codes(self) -> set[str]:
        return set(self._by_corp)

    def corpus(self, corp_code: str) -> ProductCorpus | None:
        return self._by_corp.get(corp_code)

    def relations_for(self, corp_code: str) -> tuple[FilingRelation, ...]:
        """Links that name this company, or that point at one of its filings."""
        corpus = self.corpus(corp_code)
        receipts = set() if corpus is None else {item.receipt_no for item in corpus.filings}
        kept: list[FilingRelation] = []
        for relation in self.relations:
            if relation.corp_code:
                if relation.corp_code == corp_code:
                    kept.append(relation)
                continue
            if relation.correction_receipt_no in receipts:
                kept.append(relation)
        return tuple(kept)

    def index_label(self, corpus: ProductCorpus) -> str:
        try:
            return corpus.index_dir.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return corpus.index_dir.as_posix()


_CORPUS_DESCRIPTION = (
    "아래 기업의 공시 원문만 읽어 답합니다. "
    "목록에 없는 기업은 검색하지 않습니다. "
    "데이터 기준 시각은 회사마다 다릅니다."
)


def public_corpus(catalog: SearchCatalog | None) -> dict[str, Any]:
    """Company names and per-company freshness. Paths and index ids stay out."""
    companies: list[dict[str, Any]] = []
    if catalog is not None:
        for corpus in catalog.corpora:
            companies.append(
                {
                    "company_name": corpus.binding.company_name,
                    "corpus_as_of": request_time_context(corpus.corpus_as_of)["iso"],
                    "filing_count": len(corpus.filings),
                }
            )
    companies.sort(key=lambda item: str(item["company_name"]))
    return {
        "basis": "indexed_disclosure_text",
        "description": _CORPUS_DESCRIPTION,
        "companies": companies,
    }


def load_search_catalog(root: Path) -> SearchCatalog:
    """Read every built product index and the stored filing relations."""
    corpora = tuple(load_index_corpus(root, index_dir) for index_dir in discover_index_dirs(root))
    relations = load_relations(root / "data" / "filing_relations" / "relations.json")
    return SearchCatalog(root=root, corpora=corpora, relations=relations)


def search_scope_record(
    catalog: SearchCatalog,
    corpus: ProductCorpus,
    tool: ToolResolution,
    relations: tuple[FilingRelation, ...],
) -> dict[str, Any]:
    """Debug metadata. This is not a public observation."""
    candidates = corpus.filings
    effective = () if tool.filings is None else tool.filings
    effective_ids = {item.receipt_no for item in effective}
    as_of = None if tool.resolved is None else tool.resolved.as_of
    hidden = suppressed_originals(candidates, relations, as_of=as_of)
    excluded: list[dict[str, str]] = []
    if tool.resolved is not None:
        for item in candidates:
            if item.receipt_no in effective_ids:
                continue
            excluded.append(_exclusion(item, hidden, relations, as_of, corpus.corpus_as_of))
    period = None
    if tool.resolved is not None and tool.resolved.arguments.period is not None:
        period = tool.resolved.arguments.period.model_dump()
    return {
        "company_id": corpus.binding.corp_code,
        "corp_code": corpus.binding.corp_code,
        "index_id": str(corpus.manifest.get("corpus_id") or ""),
        "index_dir": catalog.index_label(corpus),
        "retrieval_unit_count": corpus.manifest.get("retrieval_unit_count"),
        "corpus_as_of": request_time_context(corpus.corpus_as_of)["iso"],
        "requested_period": period,
        "as_of": None if tool.resolved is None else tool.resolved.arguments.as_of,
        "candidate_filing_count": len(candidates),
        "candidate_receipt_nos": [item.receipt_no for item in candidates],
        "effective_receipt_nos": [item.receipt_no for item in effective],
        "excluded": excluded,
    }


class CatalogDenseSearch:
    """Route a validated request to that company's index and reject foreign hits."""

    def __init__(self, catalog: SearchCatalog, encoder: QueryEncoder | None = None) -> None:
        self.catalog = catalog
        self.encoder = encoder or BgeM3QueryEncoder()
        self._search: dict[str, DenseD2aSearch] = {}

    def __call__(self, request: SearchDisclosuresRequest) -> SearchDisclosuresResult:
        corp_code = request.company_id
        corpus = self.catalog.corpus(corp_code)
        if corpus is None:
            raise RetrievalFailure("search company is not in the product catalog")
        if request.corp_code and request.corp_code != corpus.binding.corp_code:
            raise RetrievalFailure("search company_id does not match corp_code")
        search = self._search.get(corp_code)
        if search is None:
            search = DenseD2aSearch(corpus.binding, self.encoder)
            self._search[corp_code] = search
        result = search(request)
        receipts = request.eligible_receipt_nos
        allowed = None if receipts is None else set(receipts)
        for hit in result.hits:
            if hit.corp_code != corpus.binding.corp_code:
                raise RetrievalFailure("retrieval hit belongs to another company")
            if allowed is not None and hit.receipt_no not in allowed:
                raise RetrievalFailure("retrieval hit is outside the effective filing set")
        return result


class CatalogEvidenceBuilder:
    """Load element locators for the company that was actually retrieved."""

    def __init__(self, catalog: SearchCatalog) -> None:
        self.catalog = catalog
        self._builders: dict[str, EvidenceBuilder] = {}

    def build(self, hits: list[SearchHit], *, company_name: str) -> list[Evidence]:
        corp_code = next((hit.corp_code for hit in hits if hit.corp_code), "")
        return self._builder(corp_code).build(hits, company_name=company_name)

    def _builder(self, corp_code: str) -> EvidenceBuilder:
        cached = self._builders.get(corp_code)
        if cached is not None:
            return cached
        corpus = self.catalog.corpus(corp_code)
        if corpus is None:
            builder = EvidenceBuilder()
        else:
            builder = EvidenceBuilder.from_elements_paths(corpus.elements_paths)
        self._builders[corp_code] = builder
        return builder


def _exclusion(
    period: FilingPeriod,
    hidden: frozenset[str],
    relations: tuple[FilingRelation, ...],
    as_of: datetime | None,
    corpus_as_of: datetime,
) -> dict[str, str]:
    if period.receipt_no in hidden:
        relation = next(
            (
                item
                for item in relations
                if item.original_receipt_no == period.receipt_no
                and item.confidence in {"confirmed", "inferred"}
            ),
            None,
        )
        row = {
            "receipt_no": period.receipt_no,
            "reason": "superseded_by_correction",
        }
        if relation is not None:
            row["correction_receipt_no"] = relation.correction_receipt_no
            row["confidence"] = relation.confidence
        return row
    if as_of is not None and not receipt_on_or_before(period, as_of):
        return {"receipt_no": period.receipt_no, "reason": "after_as_of"}
    if not receipt_on_or_before(period, corpus_as_of):
        return {"receipt_no": period.receipt_no, "reason": "after_corpus_as_of"}
    return {"receipt_no": period.receipt_no, "reason": "outside_query_filter"}

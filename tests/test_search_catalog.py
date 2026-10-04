"""Product search uses one catalog for every built company index."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from disclosure_agent.agent.catalog import repo_root
from disclosure_agent.agent.company import CompanyRecord, CompanyRegistry
from disclosure_agent.agent.corpus import ProductCorpus, load_index_corpus
from disclosure_agent.agent.errors import RetrievalFailure
from disclosure_agent.agent.evidence import EvidenceBuilder
from disclosure_agent.agent.filing_relation import FilingRelation
from disclosure_agent.agent.planner import RawQueryPlanner, ScriptedPlanner
from disclosure_agent.agent.record import RunLog
from disclosure_agent.agent.runtime import AgentRuntime, build_default_runtime
from disclosure_agent.agent.search_catalog import (
    CatalogDenseSearch,
    SearchCatalog,
    load_search_catalog,
    search_scope_record,
)
from disclosure_agent.agent.temporal import FilingPeriod
from disclosure_agent.agent.tool_args import resolve_search_call
from disclosure_agent.dense import EMBEDDING_DIMENSION
from disclosure_agent.retrieval_unit import D2A_VERSION

DAESANG = "00121941"
ORIGINAL_2024 = "20250319001095"
CORRECTION_2024 = "20250321000555"
INDEXED = {
    "00107613",
    DAESANG,
    "00126380",
    "00126955",
    "00989619",
}


class _Encoder:
    def encode(self, text: str) -> object:
        del text
        vector = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
        vector[0] = 1.0
        return vector


def _annual(company_id: str, year: int, as_of: str | None = None) -> dict[str, object]:
    return {
        "query": "매출",
        "company_id": company_id,
        "period": {
            "year": year,
            "quarter": None,
            "half": None,
            "mode": "exact",
            "year_from": None,
            "year_to": None,
        },
        "filing_date_from": None,
        "filing_date_to": None,
        "report_types": ["annual"],
        "as_of": as_of,
        "top_k": 3,
    }


def _period(
    receipt: str,
    report_type: str,
    document_name: str,
    received: str,
    *,
    corp: str,
    year: int = 2024,
    month: int = 12,
) -> FilingPeriod:
    return FilingPeriod(
        receipt_no=receipt,
        report_name=f"{report_type} ({year}.{month:02d})",
        report_type=report_type,
        receipt_date=received,
        period_end_year=year,
        period_end_month=month,
        remark="",
        corp_code=corp,
        document_name=document_name,
    )


def _write_index(
    root: Path,
    folder: str,
    corp: str,
    name: str,
    filings: tuple[FilingPeriod, ...],
    *,
    hostile_corp: str | None = None,
) -> None:
    dest = root / "data" / "product" / "indexes" / folder
    dest.mkdir(parents=True)
    members = []
    units = []
    for filing in filings:
        members.append(
            {
                "corp_code": filing.corp_code,
                "company_name": name,
                "receipt_no": filing.receipt_no,
                "report_name": filing.report_name,
                "report_type": filing.report_type,
                "receipt_date": filing.receipt_date,
                "period_end_year": filing.period_end_year,
                "period_end_month": filing.period_end_month,
                "remark": filing.remark,
                "source_sha256": "abc",
                "document_name": filing.document_name,
                "elements_path": "missing/elements.jsonl",
            }
        )
        units.append(
            {
                "retrieval_unit_id": f"id-{filing.receipt_no}",
                "canonical_chunk_id": f"chunk-{filing.receipt_no}",
                "canonical_ordinal": 1,
                "unit_index": 0,
                "representation_version": D2A_VERSION,
                "text": f"{name} {filing.receipt_no} 매출",
                "source_refs": [],
                "source_element_ids": [],
                "section_path": [],
                "receipt_no": filing.receipt_no,
                "document_name": filing.document_name,
                "report_name": filing.report_name,
                "report_type": filing.report_type,
                "filing_period": f"{filing.period_end_year}.{filing.period_end_month:02d}",
                "period_end_year": filing.period_end_year,
                "period_end_month": filing.period_end_month,
                "receipt_date": filing.receipt_date,
                "source_sha256": "abc",
                "corp_code": hostile_corp or filing.corp_code,
            }
        )
    manifest = {
        "schema_version": 1,
        "corpus_id": f"test_{corp}",
        "representation_version": D2A_VERSION,
        "embedding_model": "BAAI/bge-m3",
        "corp_code": corp,
        "company_name": name,
        "corpus_as_of": "2026-10-02T00:00:00+09:00",
        "retrieval_unit_count": len(units),
        "members": members,
    }
    (dest / "corpus.json").write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )
    with (dest / "retrieval_units.jsonl").open("w", encoding="utf-8") as handle:
        for unit in units:
            handle.write(json.dumps(unit, ensure_ascii=False) + "\n")
    matrix = np.zeros((len(units), EMBEDDING_DIMENSION), dtype=np.float32)
    for index in range(len(units)):
        matrix[index, 0] = 1.0
    matrix.tofile(dest / "embeddings.f32")


def _runtime(
    catalog: SearchCatalog,
    registry: CompanyRegistry,
    planner: object,
    log_path: Path,
    search: CatalogDenseSearch,
) -> AgentRuntime:
    return AgentRuntime(
        registry=registry,
        search=search,
        evidence_builder=EvidenceBuilder(),
        planner=planner,  # type: ignore[arg-type]
        run_log=RunLog(log_path),
        catalog=catalog,
    )


def _record(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8").splitlines()[-1])


def test_listing_prefix_keeps_the_corrected_annual() -> None:
    original = _period(
        ORIGINAL_2024,
        "사업보고서",
        "사업보고서",
        "20250319",
        corp=DAESANG,
    )
    correction = _period(
        CORRECTION_2024,
        "[기재정정]사업보고서",
        "사업보고서",
        "20250321",
        corp=DAESANG,
    )
    relation = FilingRelation(
        correction_receipt_no=CORRECTION_2024,
        original_receipt_no=ORIGINAL_2024,
        confidence="inferred",
        evidence=("first_submission_date", "document_name", "corp_code"),
        corp_code=DAESANG,
    )
    current = resolve_search_call(
        _annual(DAESANG, 2024),
        company_id=DAESANG,
        filings=(original, correction),
        corpus_as_of=_as_of(),
        max_top_k=20,
        relations=(relation,),
    )
    assert current.status == "ok"
    assert [item.receipt_no for item in current.filings or ()] == [CORRECTION_2024]


def test_unverified_relation_keeps_the_original(tmp_path: Path) -> None:
    original = _period("20250319001095", "사업보고서", "사업보고서", "20250319", corp=DAESANG)
    correction = _period(
        "20250321000555",
        "[기재정정]사업보고서",
        "사업보고서",
        "20250321",
        corp=DAESANG,
    )
    _write_index(tmp_path, DAESANG, DAESANG, "대상", (original, correction))
    relation = FilingRelation(
        correction_receipt_no=correction.receipt_no,
        original_receipt_no=original.receipt_no,
        confidence="unverified",
        evidence=("insufficient_link",),
        corp_code=DAESANG,
    )
    catalog = SearchCatalog(
        tmp_path,
        (load_index_corpus(tmp_path, tmp_path / "data" / "product" / "indexes" / DAESANG),),
        (relation,),
    )
    corpus = catalog.corpus(DAESANG)
    assert corpus is not None
    tool = resolve_search_call(
        _annual(DAESANG, 2024),
        company_id=DAESANG,
        filings=corpus.filings,
        corpus_as_of=corpus.corpus_as_of,
        max_top_k=20,
        relations=catalog.relations_for(DAESANG),
    )
    chosen = {item.receipt_no for item in tool.filings or ()}
    assert chosen == {original.receipt_no, correction.receipt_no}
    scope = search_scope_record(catalog, corpus, tool, catalog.relations_for(DAESANG))
    assert all(row["reason"] != "superseded_by_correction" for row in scope["excluded"])
    units_path = tmp_path / "data" / "product" / "indexes" / DAESANG / "retrieval_units.jsonl"
    units = units_path.read_text(encoding="utf-8")
    assert original.receipt_no in units
    assert correction.receipt_no in units


def test_two_companies_do_not_leak(tmp_path: Path) -> None:
    alpha_filing = _period(
        "20260318000001", "사업보고서", "사업보고서", "20260318", corp="10000001", year=2025
    )
    beta_filing = _period(
        "20260318000002", "사업보고서", "사업보고서", "20260318", corp="10000002", year=2025
    )
    _write_index(tmp_path, "10000001", "10000001", "검색알파", (alpha_filing,))
    _write_index(tmp_path, "10000002", "10000002", "검색베타", (beta_filing,))
    catalog = load_search_catalog(tmp_path)
    registry = CompanyRegistry.from_records(
        [
            CompanyRecord("10000001", "검색알파", "111111"),
            CompanyRecord("10000002", "검색베타", "222222"),
        ],
        indexed_corp_codes=catalog.indexed_corp_codes(),
    )
    search = CatalogDenseSearch(catalog, _Encoder())
    planner = ScriptedPlanner([_annual("10000001", 2025)])
    log = tmp_path / "runs.jsonl"
    runtime = _runtime(catalog, registry, planner, log, search)
    events = list(runtime.stream("검색알파의 매출은?"))
    record = _record(log)
    assert events[-1].data["status"] == "completed"
    assert "search_scopes" not in events[-1].data
    calls = record["tool_calls"]
    assert isinstance(calls, list) and calls
    eligible = set(calls[0]["eligible_receipt_nos"])
    assert eligible == {alpha_filing.receipt_no}
    assert beta_filing.receipt_no not in eligible
    assert set(calls[0]["retrieved_corp_codes"]) == {"10000001"}
    assert beta_filing.receipt_no not in calls[0]["retrieved_receipt_nos"]
    scope = calls[0]["search_scope"]
    assert scope["index_id"] == "test_10000001"
    assert scope["effective_receipt_nos"] == [alpha_filing.receipt_no]
    assert {item.corp_code for item in planner.contexts[0].filings} == {"10000001"}


def test_foreign_unit_inside_a_company_index_is_rejected(tmp_path: Path) -> None:
    filing = _period(
        "20260318000001", "사업보고서", "사업보고서", "20260318", corp="10000001", year=2025
    )
    _write_index(
        tmp_path,
        "10000001",
        "10000001",
        "검색알파",
        (filing,),
        hostile_corp="10000002",
    )
    catalog = load_search_catalog(tmp_path)
    search = CatalogDenseSearch(catalog, _Encoder())
    from disclosure_agent.agent.search import SearchDisclosuresRequest

    with pytest.raises(RetrievalFailure, match="another company"):
        search(
            SearchDisclosuresRequest(
                query="매출",
                company_id="10000001",
                corp_code="10000001",
                eligible_receipt_nos=(filing.receipt_no,),
                top_k=3,
            )
        )


def test_no_matching_filing_does_not_search(tmp_path: Path) -> None:
    filing = _period(
        "20260318000001", "사업보고서", "사업보고서", "20260318", corp="10000001", year=2025
    )
    _write_index(tmp_path, "10000001", "10000001", "검색알파", (filing,))
    catalog = load_search_catalog(tmp_path)
    seen: list[object] = []

    def search(request: object) -> object:
        seen.append(request)
        raise AssertionError("retrieval should not run")

    registry = CompanyRegistry.from_records(
        [CompanyRecord("10000001", "검색알파", "111111")],
        indexed_corp_codes=catalog.indexed_corp_codes(),
    )
    log = tmp_path / "none.jsonl"
    runtime = AgentRuntime(
        registry=registry,
        search=search,  # type: ignore[arg-type]
        evidence_builder=EvidenceBuilder(),
        planner=ScriptedPlanner([_annual("10000001", 1991)]),
        run_log=RunLog(log),
        catalog=catalog,
    )
    events = list(runtime.stream("검색알파 1991년 매출은?"))
    assert seen == []
    assert events[-1].data["status"] == "no_matching_filing"
    assert events[-1].data["abstain_reason"] == "no_matching_filing"
    record = _record(log)
    assert record["search_called"] is False
    scope = record["search_scopes"]
    assert isinstance(scope, list) and scope[0]["effective_receipt_nos"] == []
    assert scope[0]["candidate_filing_count"] == 1
    text = events[-1].data["answer"]["text"]
    assert "DART" not in text


def test_duplicate_corp_code_is_rejected(tmp_path: Path) -> None:
    filing = _period(
        "20260318000001", "사업보고서", "사업보고서", "20260318", corp="10000001", year=2025
    )
    _write_index(tmp_path, "one", "10000001", "검색알파", (filing,))
    _write_index(tmp_path, "two", "10000001", "검색알파", (filing,))
    with pytest.raises(RetrievalFailure, match="two product indexes"):
        load_search_catalog(tmp_path)


def _real_catalog() -> SearchCatalog:
    root = repo_root()
    if not (root / "data" / "product" / "indexes").is_dir():
        pytest.skip("product indexes are not in this checkout")
    catalog = load_search_catalog(root)
    if not INDEXED <= catalog.indexed_corp_codes():
        pytest.skip("the five company indexes are not all built")
    return catalog


def test_product_catalog_maps_five_companies_without_name_branches() -> None:
    catalog = _real_catalog()
    expected = {
        "00107613": "data/product/indexes/isu_petasys_periodic_d2a",
        "00126380": "data/product/indexes/generalization/00126380",
        "00126955": "data/product/indexes/generalization/00126955",
        "00989619": "data/product/indexes/generalization/00989619",
        DAESANG: "data/product/indexes/generalization/00121941",
    }
    for corp_code, label in expected.items():
        corpus = catalog.corpus(corp_code)
        assert corpus is not None
        assert catalog.index_label(corpus) == label
        assert corpus.binding.corp_code == corp_code
        assert corpus.filings
        assert {item.corp_code for item in corpus.filings} == {corp_code}


def test_daesang_correction_views_and_physical_history() -> None:
    catalog = _real_catalog()
    corpus = catalog.corpus(DAESANG)
    assert corpus is not None
    relations = catalog.relations_for(DAESANG)
    current = _scope(catalog, corpus, 2024, None, relations)
    before = _scope(catalog, corpus, 2024, "2025-03-20", relations)
    after = _scope(catalog, corpus, 2024, "2025-03-22", relations)
    assert current["effective_receipt_nos"] == [CORRECTION_2024]
    assert before["effective_receipt_nos"] == [ORIGINAL_2024]
    assert after["effective_receipt_nos"] == [CORRECTION_2024]
    suppressed = {
        row["receipt_no"]
        for row in current["excluded"]
        if row["reason"] == "superseded_by_correction"
    }
    assert ORIGINAL_2024 in suppressed
    assert ORIGINAL_2024 in current["candidate_receipt_nos"]
    raw = repo_root() / "data" / "raw" / "dart" / ORIGINAL_2024
    assert raw.is_dir()
    assert any(raw.rglob("document.zip"))
    stored = json.loads((corpus.index_dir / "corpus.json").read_text(encoding="utf-8"))
    indexed = {member["receipt_no"] for member in stored["members"]}
    assert ORIGINAL_2024 in indexed
    assert CORRECTION_2024 in indexed


def test_product_runtime_retrieves_each_indexed_company(tmp_path: Path) -> None:
    catalog = _real_catalog()
    root = repo_root()
    registry = CompanyRegistry.from_corp_cache(
        root / "data" / "cache",
        indexed_corp_codes=catalog.indexed_corp_codes(),
    )
    search = CatalogDenseSearch(catalog, _Encoder())
    questions = {
        "삼성전자": ("00126380", 2025, "20260310002820"),
        "삼양식품": ("00126955", 2025, "20260318001082"),
        "알테오젠": ("00989619", 2025, "20260323001596"),
        "이수페타시스": ("00107613", 2025, "20260323000847"),
        "대상": (DAESANG, 2024, CORRECTION_2024),
    }
    for name, (corp_code, year, receipt) in questions.items():
        resolved = registry.resolve_mention(name)
        assert resolved.status == "resolved"
        assert resolved.corp_code == corp_code
        log = tmp_path / f"{corp_code}.jsonl"
        runtime = _runtime(
            catalog,
            registry,
            ScriptedPlanner([_annual(corp_code, year)]),
            log,
            search,
        )
        events = list(runtime.stream(f"{name} {year}년 매출은?"))
        record = _record(log)
        assert record["search_called"] is True
        assert record["index_id"]
        calls = record["tool_calls"]
        assert isinstance(calls, list) and calls
        eligible = set(calls[0]["eligible_receipt_nos"])
        assert receipt in eligible
        own_corpus = catalog.corpus(corp_code)
        assert own_corpus is not None
        own = {item.receipt_no for item in own_corpus.filings}
        others: set[str] = set()
        for code in catalog.indexed_corp_codes():
            if code == corp_code:
                continue
            other = catalog.corpus(code)
            assert other is not None
            others.update(item.receipt_no for item in other.filings)
        assert eligible <= own
        assert eligible.isdisjoint(others)
        assert set(calls[0]["retrieved_corp_codes"]) == {corp_code}
        assert set(calls[0]["retrieved_receipt_nos"]) <= eligible
        assert calls[0]["search_scope"]["index_id"]
        assert "search_scopes" not in events[-1].data
        assert {item.corp_code for item in runtime.planner.contexts[0].filings} == {corp_code}  # type: ignore[attr-defined]


def test_historical_as_of_retrieves_the_original(tmp_path: Path) -> None:
    catalog = _real_catalog()
    registry = CompanyRegistry.from_corp_cache(
        repo_root() / "data" / "cache",
        indexed_corp_codes=catalog.indexed_corp_codes(),
    )
    search = CatalogDenseSearch(catalog, _Encoder())
    log = tmp_path / "asof.jsonl"
    runtime = _runtime(
        catalog,
        registry,
        ScriptedPlanner([_annual(DAESANG, 2024, "2025-03-20")]),
        log,
        search,
    )
    list(runtime.stream("대상 2024년 매출은?"))
    record = _record(log)
    calls = record["tool_calls"]
    assert isinstance(calls, list)
    assert set(calls[0]["eligible_receipt_nos"]) == {ORIGINAL_2024}
    assert set(calls[0]["retrieved_receipt_nos"]) == {ORIGINAL_2024}
    assert set(calls[0]["retrieved_corp_codes"]) == {DAESANG}


def test_known_company_without_an_index_is_not_searched(tmp_path: Path) -> None:
    catalog = _real_catalog()
    registry = CompanyRegistry.from_corp_cache(
        repo_root() / "data" / "cache",
        indexed_corp_codes=catalog.indexed_corp_codes(),
    )
    seen: list[object] = []

    def search(request: object) -> object:
        seen.append(request)
        raise AssertionError("retrieval should not run")

    log = tmp_path / "sdi.jsonl"
    runtime = AgentRuntime(
        registry=registry,
        search=search,  # type: ignore[arg-type]
        evidence_builder=EvidenceBuilder(),
        planner=RawQueryPlanner(),
        run_log=RunLog(log),
        catalog=catalog,
    )
    events = list(runtime.stream("삼성SDI 매출은?"))
    assert seen == []
    assert events[-1].data["status"] == "company_unsupported"
    assert events[-1].data["abstain_reason"] == "unsupported_company"
    text = events[-1].data["answer"]["text"]
    assert "지원" in text
    assert "색인" not in text
    assert "DART" not in text


def test_default_runtime_uses_the_catalog(tmp_path: Path) -> None:
    catalog = _real_catalog()
    runtime = build_default_runtime(repo_root(), run_log_path=tmp_path / "default.jsonl")
    assert runtime.catalog is not None
    assert INDEXED <= runtime.catalog.indexed_corp_codes()
    assert set(runtime.registry.indexed_corp_codes) == catalog.indexed_corp_codes()
    resolved = runtime.registry.resolve_mention("삼성전자")
    assert resolved.status == "resolved"
    assert resolved.corp_code == "00126380"


def _scope(
    catalog: SearchCatalog,
    corpus: ProductCorpus,
    year: int,
    as_of: str | None,
    relations: tuple[FilingRelation, ...],
) -> dict[str, object]:
    tool = resolve_search_call(
        _annual(corpus.binding.corp_code, year, as_of),
        company_id=corpus.binding.corp_code,
        filings=corpus.filings,
        corpus_as_of=corpus.corpus_as_of,
        max_top_k=20,
        relations=relations,
    )
    assert tool.status == "ok"
    return search_scope_record(catalog, corpus, tool, relations)


def _as_of():
    from datetime import datetime

    from disclosure_agent.agent.clock import SEOUL

    return datetime(2026, 10, 2, tzinfo=SEOUL)

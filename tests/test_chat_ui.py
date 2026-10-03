"""Minimal chat UI: static page, public corpus, and citation payload."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from fastapi.testclient import TestClient
from test_agent_product import events_of, make_runtime

from disclosure_agent.agent.api import create_app
from disclosure_agent.agent.catalog import DenseD2aBinding, repo_root
from disclosure_agent.agent.clock import SEOUL
from disclosure_agent.agent.corpus import ProductCorpus
from disclosure_agent.agent.search_catalog import SearchCatalog, public_corpus
from disclosure_agent.agent.temporal import FilingPeriod

_STATIC = repo_root() / "src" / "disclosure_agent" / "agent" / "static"
_COMPANY_NAMES = ("이수페타시스", "삼성전자", "삼양식품", "알테오젠")


def _corpus(name: str, corp_code: str, day: int, secret: Path) -> ProductCorpus:
    binding = DenseD2aBinding(
        corp_code=corp_code,
        company_name=name,
        units_path=secret / "retrieval_units.jsonl",
        embeddings_path=secret / "embeddings.f32",
        elements_path=secret / "elements.jsonl",
        receipt_no="20260323000847",
        document_name="사업보고서",
        representation_version="retrieval-unit-d2a-v1",
        embedding_model="BAAI/bge-m3",
    )
    filing = FilingPeriod(
        receipt_no="20260323000847",
        report_name="사업보고서",
        report_type="사업보고서",
        receipt_date="20260323",
        period_end_year=2025,
        period_end_month=12,
        remark="",
        source_sha256="abc",
        corp_code=corp_code,
    )
    return ProductCorpus(
        binding=binding,
        filings=(filing,),
        corpus_as_of=datetime(2026, 9, day, 10, 26, 24, tzinfo=SEOUL),
        elements_paths=(secret / "elements.jsonl",),
        manifest={"index_dir": str(secret)},
        index_dir=secret,
    )


def _catalog() -> SearchCatalog:
    secret = Path(r"C:\secret\indexes\hidden_product_index")
    return SearchCatalog(
        root=secret,
        corpora=(
            _corpus("이수페타시스", "00107613", 25, secret / "isu"),
            _corpus("삼성전자", "00126380", 2, secret / "sec"),
        ),
    )


def test_public_corpus_lists_each_company_without_index_paths() -> None:
    payload = public_corpus(_catalog())
    blob = json.dumps(payload, ensure_ascii=False)
    assert "secret" not in blob
    assert "embeddings" not in blob
    assert "index_dir" not in blob
    assert "corp_code" not in blob
    assert payload["basis"] == "indexed_disclosure_text"
    assert "목록에 없는 기업" in payload["description"]
    rows = payload["companies"]
    assert [row["company_name"] for row in rows] == ["삼성전자", "이수페타시스"]
    assert rows[0]["corpus_as_of"].startswith("2026-09-02")
    assert rows[1]["corpus_as_of"].startswith("2026-09-25")
    assert rows[0]["corpus_as_of"] != rows[1]["corpus_as_of"]
    assert set(rows[0]) == {"company_name", "corpus_as_of", "filing_count"}
    assert public_corpus(None)["companies"] == []


def test_corpus_route_and_chat_page(tmp_path: Path) -> None:
    del tmp_path
    runtime = make_runtime([])
    runtime.catalog = _catalog()
    client = TestClient(create_app(runtime))

    page = client.get("/")
    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]
    assert 'id="ask-form"' in page.text
    assert "/static/chat.js" in page.text

    script = client.get("/static/chat.js")
    style = client.get("/static/chat.css")
    assert script.status_code == 200
    assert style.status_code == 200
    assert client.get("/static/secret.txt").status_code == 404
    assert client.get("/static/chat.js/../../api.py").status_code == 404

    corpus = client.get("/api/corpus")
    assert corpus.status_code == 200
    body = corpus.json()
    assert [row["company_name"] for row in body["companies"]] == ["삼성전자", "이수페타시스"]
    encoded = json.dumps(body, ensure_ascii=False)
    assert "secret" not in encoded
    assert "embeddings" not in encoded

    runtime.catalog = None
    empty = client.get("/api/corpus")
    assert empty.status_code == 200
    assert empty.json()["companies"] == []


def test_static_ui_does_not_hardcode_the_supported_companies() -> None:
    for name in ("index.html", "chat.js", "chat.css"):
        text = (_STATIC / name).read_text(encoding="utf-8")
        for company in _COMPANY_NAMES:
            assert company not in text
        assert "data/product" not in text
        assert "embeddings.f32" not in text
        assert "2026-09-25" not in text
    script = (_STATIC / "chat.js").read_text(encoding="utf-8")
    assert "/api/corpus" in script
    assert "citation_id" in script
    assert "evidence_id" in script
    assert "페이지" not in script


def test_citation_event_uses_the_same_evidence_as_the_completed_answer() -> None:
    found = events_of(make_runtime([]), "이수페타시스의 주요 사업이 뭐야?")
    completed = found[-1].data
    citation = next(event.data for event in found if event.event == "citation")
    answer_citation = completed["answer"]["citations"][0]
    evidence = completed["evidence"][0]
    assert citation["citation_id"] == answer_citation["citation_id"] == "1"
    assert citation["evidence_id"] == answer_citation["evidence_id"] == evidence["evidence_id"]
    assert citation["text"] == evidence["text"]
    assert citation["company_name"] == evidence["company_name"] == "이수페타시스"
    assert citation["receipt_no"] == evidence["receipt_no"] == "20260323000847"
    blob = json.dumps(citation, ensure_ascii=False)
    assert "data/product" not in blob
    assert "embeddings" not in blob


def test_chat_page_still_answers_without_a_catalog() -> None:
    client = TestClient(create_app(make_runtime([])))
    response = client.post("/ask", json={"message": "안녕"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "run.completed" in response.text
    corpus = client.get("/api/corpus")
    assert corpus.json()["companies"] == []

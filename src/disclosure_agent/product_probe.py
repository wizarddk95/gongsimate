"""Run the temporal regressions against the built index without calling OpenAI."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from disclosure_agent.agent.answer import GeneratedAnswer
from disclosure_agent.agent.citation import Citation
from disclosure_agent.agent.clock import SEOUL
from disclosure_agent.agent.company import CompanyRecord, CompanyRegistry
from disclosure_agent.agent.corpus import load_product_corpus, product_index_dir
from disclosure_agent.agent.evidence import EvidenceBuilder
from disclosure_agent.agent.record import RunLog
from disclosure_agent.agent.runtime import AgentRuntime
from disclosure_agent.agent.search import DenseD2aSearch
from disclosure_agent.dense import encode_texts

PROBE_QUESTIONS = (
    ("A_realized_2025", "이수페타시스의 2025년 매출이 얼마야?"),
    ("B_historical_2025", "이수페타시스의 2025년 매출 증가 이유가 뭐야?"),
    ("C_latest_performance", "이수페타시스의 최근 실적은?"),
    ("D_latest_periodic", "이수페타시스의 주요 사업이 뭐야?"),
    ("E_realized_2035", "2035년 이수페타시스 확정 매출 알려줘"),
    ("F_forecast_2035", "이수페타시스가 2035년 매출 목표를 제시했어?"),
    (
        "G_historical_2024_charter",
        "이수페타시스가 2024년에 정관상 배당 기준일을 바꾼 이유는 뭐야?",
    ),
)


class _ProbeGenerator:
    """Records whether generation ran. The text is not an answer."""

    generator_type = "probe"
    model = "probe"
    sdk_version = "probe"

    def __init__(self) -> None:
        self.calls = 0

    def generate(
        self,
        question: str,
        evidence: list[Any],
        *,
        company_name: str = "",
        fiscal_year: int | None = None,
        request_time: object = None,
        corpus_as_of: object = None,
    ) -> GeneratedAnswer:
        del question, company_name, fiscal_year, request_time, corpus_as_of
        self.calls += 1
        if not evidence:
            return GeneratedAnswer(text="probe-empty", abstain=True, generator_type="probe")
        return GeneratedAnswer(
            text="probe",
            citations=[Citation(citation_id="1", evidence_id=evidence[0].evidence_id)],
            generator_type="probe",
        )


class _ReadyEncoder:
    def __init__(self, model: Any) -> None:
        self._model = model

    def encode(self, text: str) -> Any:
        return encode_texts(self._model, [text], batch_size=1)[0]


def run_temporal_probe(root: Path, model: Any) -> dict[str, Any]:
    product = load_product_corpus(root)
    registry = CompanyRegistry.from_records(
        [CompanyRecord(product.binding.corp_code, product.binding.company_name, "007660", "")],
        indexed_corp_codes={product.binding.corp_code},
    )
    encoder = _ReadyEncoder(model)
    search = DenseD2aSearch(product.binding, encoder)
    evidence_builder = EvidenceBuilder.from_elements_paths(product.elements_paths)
    clock = datetime(2026, 10, 1, 20, 56, tzinfo=SEOUL)
    log_path = product_index_dir(root) / "temporal_probe_runs.jsonl"
    if log_path.exists():
        log_path.unlink()
    rows = []
    for key, question in PROBE_QUESTIONS:
        generator = _ProbeGenerator()
        log = RunLog(log_path)
        runtime = AgentRuntime(
            registry=registry,
            search=search,
            evidence_builder=evidence_builder,
            answer_generator=generator,
            run_log=log,
            filings=product.filings,
            corpus_as_of=product.corpus_as_of,
            clock=lambda moment=clock: moment,
        )
        result = runtime.run(question)
        terminal = result["terminal"]
        record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
        rows.append(
            {
                "key": key,
                "question": question,
                "status": terminal.data.get("status") or terminal.data.get("code"),
                "planner": record.get("planner"),
                "tool_statuses": [call.get("status") for call in record.get("tool_calls") or []],
                "eligible_receipt_nos": record.get("eligible_receipt_nos"),
                "search_called": record.get("search_called"),
                "retrieved_filing_counts": record.get("retrieved_filing_counts"),
                "context_filing_counts": record.get("context_filing_counts"),
                "generation_called": generator.calls > 0,
                "citation_count": len(terminal.data.get("answer", {}).get("citations") or []),
                "raw_user_query": record.get("raw_user_query"),
                "request_time": record.get("request_time"),
            }
        )
        print(key, rows[-1]["status"], rows[-1]["tool_statuses"], flush=True)
    payload = {"questions": rows}
    path = product_index_dir(root) / "temporal_probe.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload

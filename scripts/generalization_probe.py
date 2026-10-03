# ruff: noqa: E501
"""Planner probes for four issuers, and E2E only where a success index exists.

The process OPENAI_API_KEY is replaced by the project .env value when they differ.
The key itself is not printed.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any

from disclosure_agent.agent.catalog import DenseD2aBinding, repo_root
from disclosure_agent.agent.clock import request_time_context, seoul_now
from disclosure_agent.agent.company import CompanyRegistry
from disclosure_agent.agent.evidence import EvidenceBuilder
from disclosure_agent.agent.llm import OpenAIAnswerClient, OpenAIAnswerGenerator
from disclosure_agent.agent.planner import OpenAISearchPlanner, PlannerContext
from disclosure_agent.agent.record import RunLog
from disclosure_agent.agent.runtime import AgentRuntime
from disclosure_agent.agent.search import DenseD2aSearch
from disclosure_agent.agent.settings import load_openai_config
from disclosure_agent.agent.temporal import FilingPeriod, filing_period_from_listing
from disclosure_agent.agent.tool_args import arguments_record, resolve_search_call
from disclosure_agent.cli import load_dotenv_files
from disclosure_agent.retrieval_unit import D2A_VERSION

ROOT = repo_root()
OUT = ROOT / "evaluation" / "generalization_smoke"
INDEX_ROOT = ROOT / "data" / "product" / "indexes" / "generalization"
E2E_KEYS = ("latest", "business", "annual", "forecast")


def main() -> None:
    source = _prefer_dotenv_openai()
    load_dotenv_files()
    config = load_openai_config()
    if not config.configured:
        raise SystemExit("OPENAI_API_KEY is not configured")
    print(f"openai_key_source={source} model={config.model}", flush=True)
    companies = json.loads((OUT / "companies.json").read_text(encoding="utf-8"))["companies"]
    parsed = json.loads((OUT / "parse_summary.json").read_text(encoding="utf-8"))["rows"]
    eligible = {
        (row["corp_code"], row["receipt_no"])
        for row in parsed
        if row.get("parse_status") == "success" and row.get("role") == "primary"
    }
    client = OpenAIAnswerClient(config)
    planner = OpenAISearchPlanner(client)
    clock = request_time_context(seoul_now())
    planning_rows = []
    e2e_rows = []
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    started = time.perf_counter()
    for company in companies:
        filings = _filings(company)
        corpus_as_of = _corpus_as_of(company, filings)
        print(f"plan {company['stock_name']}", flush=True)
        for key, question in _questions(company):
            row = _plan_one(
                planner,
                company,
                key,
                question,
                filings,
                clock,
                corpus_as_of,
                eligible,
            )
            _add_usage(usage, row.get("usage"))
            planning_rows.append(row)
            print(
                f"  {key} calls={row['tool_called']} second={row['second_tool_call']} "
                f"status={row['final_status']} filing={row['first_matched_filing']}",
                flush=True,
            )
        index_dir = INDEX_ROOT / company["corp_code"]
        if (index_dir / "corpus.json").is_file():
            e2e_rows.extend(
                _e2e_company(company, client, config, clock, eligible, usage)
            )
        else:
            e2e_rows.append(
                {
                    "company": company["stock_name"],
                    "corp_code": company["corp_code"],
                    "e2e": "not_run",
                    "reason": "no overall_status=success filing, so no product index",
                }
            )
    payload = {
        "model": config.model,
        "openai_key_source": source,
        "request_time": clock,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "usage": usage,
        "planning": planning_rows,
        "e2e": e2e_rows,
        "note": (
            "Planning uses the downloaded primary listing catalog. "
            "E2E uses only overall_status=success filings. "
            "2025 revenue and the reason question are development probes, not a locked benchmark."
        ),
    }
    path = OUT / "probe_summary.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}", flush=True)


def _questions(company: dict[str, Any]) -> list[tuple[str, str]]:
    name = company["stock_name"]
    return [
        ("latest", f"{name}의 최근 실적은?"),
        ("business", f"{name}의 주요 사업이 뭐야?"),
        ("annual", f"{name}의 2025년 매출은?"),
        ("historical", f"{name}의 2025년 매출이 늘어난 이유는?"),
        ("quarter", f"{name}의 2026년 1분기 매출은?"),
        ("forecast", f"{name}의 향후 매출 목표가 있어?"),
    ]


def _plan_one(
    planner: OpenAISearchPlanner,
    company: dict[str, Any],
    key: str,
    question: str,
    filings: tuple[FilingPeriod, ...],
    clock: dict[str, str],
    corpus_as_of: datetime,
    eligible: set[tuple[str, str]],
) -> dict[str, Any]:
    prior: list[dict[str, Any]] = []
    turns = []
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    latency_ms = 0.0
    for _turn in range(2):
        action = planner.next_action(
            PlannerContext(
                question=question,
                request_time=clock,
                corpus_as_of=request_time_context(corpus_as_of)["iso"],
                company_id=company["corp_code"],
                company_name=company["corp_name"],
                filings=filings,
                prior_results=tuple(prior),
                max_top_k=20,
            )
        )
        _add_usage(usage, action.usage)
        if isinstance(action.latency_ms, (int, float)):
            latency_ms += float(action.latency_ms)
        if not action.requests_tool:
            break
        resolution = resolve_search_call(
            action.raw,
            company_id=company["corp_code"],
            filings=filings,
            corpus_as_of=corpus_as_of,
            max_top_k=20,
        )
        matched = [] if resolution.filings is None else [_filing_row(item) for item in resolution.filings]
        turn = {
            "arguments": None
            if resolution.resolved is None
            else arguments_record(resolution.resolved),
            "raw": action.raw,
            "validation_status": resolution.status,
            "validation_message": resolution.message,
            "matched_filings": matched,
        }
        turns.append(turn)
        prior.append(
            {
                "status": "success" if resolution.status == "ok" else resolution.status,
                "message": "metadata filter matched these filings; chunk text is omitted in this check",
                "matched_filings": matched,
                "hits": [],
            }
        )
    first = turns[0]["matched_filings"][0]["report_name"] if turns and turns[0]["matched_filings"] else None
    return {
        "company": company["stock_name"],
        "corp_code": company["corp_code"],
        "key": key,
        "question": question,
        "probe_kind": "development_probe" if key in {"historical", "forecast"} else "smoke",
        "tool_called": bool(turns),
        "second_tool_call": len(turns) > 1,
        "turns": turns,
        "first_matched_filing": first,
        "first_matched_product_eligible": bool(
            turns
            and turns[0]["matched_filings"]
            and (company["corp_code"], turns[0]["matched_filings"][0]["receipt_no"]) in eligible
        ),
        "final_status": turns[-1]["validation_status"] if turns else "no_tool",
        "latency_ms": round(latency_ms, 3),
        "usage": usage,
        "latest_available": _latest_label(filings),
    }


def _e2e_company(
    company: dict[str, Any],
    client: OpenAIAnswerClient,
    config: Any,
    clock: dict[str, str],
    eligible: set[tuple[str, str]],
    usage: dict[str, int],
) -> list[dict[str, Any]]:
    del clock
    index_dir = INDEX_ROOT / company["corp_code"]
    manifest = json.loads((index_dir / "corpus.json").read_text(encoding="utf-8"))
    filings = tuple(_period_from_member(member) for member in manifest["members"])
    elements = tuple(ROOT / str(member["elements_path"]) for member in manifest["members"])
    binding = DenseD2aBinding(
        corp_code=str(manifest["corp_code"]),
        company_name=str(manifest["company_name"]),
        units_path=index_dir / "retrieval_units.jsonl",
        embeddings_path=index_dir / "embeddings.f32",
        elements_path=elements[0],
        receipt_no=str(manifest["members"][0]["receipt_no"]),
        document_name=str(manifest["members"][0]["document_name"]),
        representation_version=D2A_VERSION,
        embedding_model=str(manifest["embedding_model"]),
        elements_paths=elements,
    )
    registry = CompanyRegistry.from_corp_cache(
        ROOT / "data" / "cache",
        indexed_corp_codes={company["corp_code"]},
    )
    log_path = OUT / f"e2e_{company['corp_code']}.jsonl"
    if log_path.exists():
        log_path.unlink()
    corpus_as_of = datetime.fromisoformat(str(manifest["corpus_as_of"]))
    rows = []
    search = DenseD2aSearch(binding)
    evidence_builder = EvidenceBuilder.from_elements_paths(elements)
    for key, question in _questions(company):
        if key not in E2E_KEYS:
            continue
        print(f"e2e {company['stock_name']} {key}", flush=True)
        runtime = AgentRuntime(
            registry=registry,
            search=search,
            evidence_builder=evidence_builder,
            answer_generator=OpenAIAnswerGenerator(client),
            run_log=RunLog(log_path),
            redact_secrets=(config.api_key,),
            filings=filings,
            corpus_as_of=corpus_as_of,
            planner=OpenAISearchPlanner(client),
        )
        started = time.perf_counter()
        result = runtime.run(question)
        elapsed = time.perf_counter() - started
        record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
        _add_usage(usage, {"input_tokens": record.get("input_tokens") or 0, "output_tokens": record.get("output_tokens") or 0, "total_tokens": record.get("total_tokens") or 0})
        _add_usage(usage, record.get("planner_usage"))
        terminal = result["terminal"].data
        evidence = terminal.get("evidence") or []
        citations = (terminal.get("answer") or {}).get("citations") or []
        leaked = [
            item.get("corp_code")
            for item in evidence
            if item.get("corp_code") and item.get("corp_code") != company["corp_code"]
        ]
        receipt_mismatch = [
            item.get("receipt_no")
            for item in evidence
            if (company["corp_code"], item.get("receipt_no")) not in eligible
        ]
        rows.append(
            {
                "company": company["stock_name"],
                "corp_code": company["corp_code"],
                "key": key,
                "question": question,
                "status": terminal.get("status"),
                "action": terminal.get("action"),
                "resolved_company": record.get("resolved_company"),
                "tool_calls": [
                    {
                        "status": call.get("status"),
                        "arguments": call.get("arguments"),
                        "matched_filings": call.get("matched_filings"),
                        "retrieved_receipt_nos": call.get("retrieved_receipt_nos"),
                    }
                    for call in record.get("tool_calls") or []
                ],
                "citation_count": len(citations),
                "evidence": [
                    {
                        "evidence_id": item.get("evidence_id"),
                        "corp_code": item.get("corp_code"),
                        "receipt_no": item.get("receipt_no"),
                        "report_name": item.get("report_name"),
                        "filing_period": item.get("filing_period"),
                        "section_path": item.get("section_path"),
                        "source_locator": item.get("source_locator"),
                        "source_sha256": item.get("source_sha256"),
                    }
                    for item in evidence
                ],
                "citation_evidence_ids": [item.get("evidence_id") for item in citations],
                "citation_ids_in_run": all(
                    item.get("evidence_id") in {row.get("evidence_id") for row in evidence}
                    for item in citations
                ),
                "cross_company_corp_codes": leaked,
                "receipt_not_eligible": receipt_mismatch,
                "answer_excerpt": str((terminal.get("answer") or {}).get("text") or "")[:500],
                "retrieval_ms": record.get("retrieval_ms"),
                "generation_ms": record.get("generation_ms"),
                "total_ms": round(elapsed * 1000, 3),
                "input_tokens": record.get("input_tokens"),
                "output_tokens": record.get("output_tokens"),
            }
        )
    return rows


def _filings(company: dict[str, Any]) -> tuple[FilingPeriod, ...]:
    periods = []
    for item in company["primary_filings"]:
        listing = {
            "report_nm": item["report_nm"],
            "rcept_no": item["receipt_no"],
            "rcept_dt": item["rcept_dt"],
            "rm": item.get("rm") or "",
            "corp_code": company["corp_code"],
        }
        periods.append(filing_period_from_listing(listing))
    return tuple(periods)


def _period_from_member(member: dict[str, Any]) -> FilingPeriod:
    year = member.get("period_end_year")
    month = member.get("period_end_month")
    return FilingPeriod(
        receipt_no=str(member["receipt_no"]),
        report_name=str(member.get("report_name") or ""),
        report_type=str(member.get("report_type") or ""),
        receipt_date=str(member.get("receipt_date") or ""),
        period_end_year=int(year) if isinstance(year, int) else None,
        period_end_month=int(month) if isinstance(month, int) else None,
        remark=str(member.get("remark") or ""),
        source_sha256=str(member.get("source_sha256") or ""),
        corp_code=str(member.get("corp_code") or ""),
        document_name=str(member.get("document_name") or ""),
    )


def _corpus_as_of(company: dict[str, Any], filings: tuple[FilingPeriod, ...]) -> datetime:
    moments = []
    for period in filings:
        current = ROOT / "data" / "raw" / "dart" / period.receipt_no / "current.json"
        if not current.is_file():
            continue
        raw = str(json.loads(current.read_text(encoding="utf-8")).get("last_success_verified_at") or "")
        if raw:
            moments.append(datetime.fromisoformat(raw))
    if not moments:
        raise SystemExit(f"no verified timestamp for {company['corp_code']}")
    return max(moments)


def _latest_label(filings: tuple[FilingPeriod, ...]) -> str:
    labeled = [item for item in filings if item.period_end_year and item.period_end_month]
    if not labeled:
        return ""
    latest = max(labeled, key=lambda item: (item.period_end_year or 0, item.period_end_month or 0, item.receipt_date))
    return latest.report_name


def _filing_row(period: FilingPeriod) -> dict[str, str]:
    label = ""
    if period.period_end_year and period.period_end_month:
        label = f"{period.period_end_year}.{period.period_end_month:02d}"
    return {
        "receipt_no": period.receipt_no,
        "report_name": period.report_name,
        "filing_period": label,
    }


def _add_usage(total: dict[str, int], usage: object) -> None:
    if not isinstance(usage, dict):
        return
    for key in total:
        value = usage.get(key)
        if isinstance(value, int):
            total[key] += value


def _prefer_dotenv_openai() -> str:
    dotenv: dict[str, str] = {}
    path = ROOT / ".env"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            name, _, value = stripped.partition("=")
            dotenv[name.strip()] = value.strip().strip('"').strip("'")
    file_key = dotenv.get("OPENAI_API_KEY", "")
    process = os.environ.get("OPENAI_API_KEY", "")
    if file_key and process and process != file_key:
        os.environ["OPENAI_API_KEY"] = file_key
        return "dotenv_replaced_unequal_process_value"
    if file_key and not process:
        return "dotenv"
    if process:
        return "process"
    return "missing"


if __name__ == "__main__":
    main()

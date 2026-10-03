"""Run the development tool-planning set against the configured OpenAI model.

Writes arguments, matched filings, and grades. Does not write the API key.
"""

from __future__ import annotations

import json
import time
from typing import Any

from disclosure_agent.agent.catalog import repo_root
from disclosure_agent.agent.clock import request_time_context, seoul_now
from disclosure_agent.agent.corpus import load_product_corpus
from disclosure_agent.agent.planner import OpenAISearchPlanner, PlannerContext
from disclosure_agent.agent.settings import load_openai_config
from disclosure_agent.agent.temporal import FilingPeriod
from disclosure_agent.agent.tool_args import resolve_search_call
from disclosure_agent.agent.tool_planning_eval import grade_case, load_cases, parse_calls
from disclosure_agent.cli import load_dotenv_files


def main() -> None:
    root = repo_root()
    load_dotenv_files()
    config = load_openai_config()
    if not config.configured:
        raise SystemExit("OPENAI_API_KEY is not configured")
    from disclosure_agent.agent.llm import OpenAIAnswerClient

    cases = load_cases(root / "evaluation" / "tool_planning" / "dev_cases.json")
    product = load_product_corpus(root)
    latest = _latest_period(product.filings)
    planner = OpenAISearchPlanner(OpenAIAnswerClient(config))
    clock = request_time_context(seoul_now())
    corpus_as_of = request_time_context(product.corpus_as_of)["iso"]
    rows = []
    started = time.perf_counter()
    for case in cases:
        rows.append(
            _run_case(
                case,
                planner=planner,
                filings=product.filings,
                clock=clock,
                corpus_as_of=corpus_as_of,
                corpus_clock=product.corpus_as_of,
                company_id=product.binding.corp_code,
                company_name=product.binding.company_name,
                latest_period=latest,
            )
        )
    summary = _summary(rows)
    payload = {
        "model": config.model,
        "request_time": clock,
        "corpus_as_of": corpus_as_of,
        "latest_period": {"year": latest[0], "month": latest[1]},
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "summary": summary,
        "cases": rows,
    }
    out = root / "evaluation" / "tool_planning" / "latest_run.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {out}")


def _run_case(
    case: dict[str, Any],
    *,
    planner: OpenAISearchPlanner,
    filings: tuple[FilingPeriod, ...],
    clock: dict[str, str],
    corpus_as_of: str,
    corpus_clock: object,
    company_id: str,
    company_name: str,
    latest_period: tuple[int, int],
) -> dict[str, Any]:
    prior: list[dict[str, Any]] = []
    raw_calls: list[Any] = []
    turns = []
    first_action: str | None = None
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    latency_ms = 0.0
    for _turn in range(2):
        try:
            action = planner.next_action(
                PlannerContext(
                    question=case["question"],
                    request_time=clock,
                    corpus_as_of=corpus_as_of,
                    company_id=company_id,
                    company_name=company_name,
                    filings=filings,
                    prior_results=tuple(prior),
                    max_top_k=20,
                )
            )
        except Exception as exc:
            grade = grade_case(case, [], invalid_calls=1, latest_period=latest_period)
            return {
                "id": case["id"],
                "category": case["category"],
                "question": case["question"],
                "pass": False,
                "errors": ["tool_failure", *grade["errors"]],
                "tool_called": False,
                "second_tool_call": False,
                "turns": [],
                "latency_ms": round(latency_ms, 3),
                "usage": usage,
                "failure": type(exc).__name__,
            }
        if action.usage:
            for key in usage:
                if isinstance(action.usage.get(key), int):
                    usage[key] += action.usage[key]
        if isinstance(action.latency_ms, (int, float)):
            latency_ms += float(action.latency_ms)
        if first_action is None:
            first_action = action.kind
        if action.kind in {"answer_directly", "ask_clarification"}:
            turns.append(
                {
                    "action": action.kind,
                    "text": action.user_text,
                    "company_mention": action.company_mention,
                    "latency_ms": action.latency_ms,
                    "usage": action.usage,
                }
            )
            break
        if not action.requests_tool:
            break
        raw_calls.append(action.raw)
        resolution = resolve_search_call(
            action.raw,
            company_id=company_id,
            filings=filings,
            corpus_as_of=corpus_clock,
            max_top_k=20,
        )
        matched = [] if resolution.filings is None else [
            {
                "receipt_no": item.receipt_no,
                "report_name": item.report_name,
                "filing_period": f"{item.period_end_year}.{item.period_end_month:02d}"
                if item.period_end_year and item.period_end_month
                else "",
            }
            for item in resolution.filings
        ]
        summary = {
            "status": "success" if resolution.status == "ok" else resolution.status,
            "message": resolution.message,
            "matched_filings": matched,
            "hits": [],
        }
        if resolution.status == "ok":
            summary["message"] = (
                "metadata filter matched these filings; chunk text is omitted in this check"
            )
        prior.append(summary)
        turns.append(
            {
                "arguments": action.raw,
                "schema_error": action.schema_error,
                "status": summary["status"],
                "matched_filings": matched,
                "latency_ms": action.latency_ms,
                "usage": action.usage,
            }
        )
    parsed, invalid = parse_calls(raw_calls)
    grade = grade_case(
        case,
        parsed,
        invalid_calls=invalid,
        latest_period=latest_period,
        action=first_action,
    )
    return {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "action": first_action,
        "pass": grade["pass"],
        "errors": grade["errors"],
        "tool_called": bool(raw_calls),
        "second_tool_call": len(raw_calls) > 1,
        "turns": turns,
        "latency_ms": round(latency_ms, 3),
        "usage": usage,
    }


def _latest_period(filings: tuple[FilingPeriod, ...]) -> tuple[int, int]:
    keyed = [
        (item.period_end_year, item.period_end_month)
        for item in filings
        if item.period_end_year is not None and item.period_end_month is not None
    ]
    if not keyed:
        raise ValueError("product corpus has no filing period")
    return max(keyed)


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors: dict[str, int] = {}
    for row in rows:
        for error in row["errors"]:
            errors[error] = errors.get(error, 0) + 1
    passed = sum(1 for row in rows if row["pass"])
    tokens = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for row in rows:
        for key in tokens:
            tokens[key] += int(row["usage"].get(key) or 0)
    return {
        "cases": len(rows),
        "passed": passed,
        "failed": len(rows) - passed,
        "errors": errors,
        "second_tool_calls": sum(1 for row in rows if row["second_tool_call"]),
        "usage": tokens,
        "latency_ms": round(sum(float(row["latency_ms"]) for row in rows), 3),
    }


if __name__ == "__main__":
    main()

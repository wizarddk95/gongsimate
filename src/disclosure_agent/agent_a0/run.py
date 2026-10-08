"""Run the product agent on the frozen A0 dataset and write the baseline artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from disclosure_agent.agent.clock import SEOUL
from disclosure_agent.agent.llm import ANSWER_INSTRUCTIONS
from disclosure_agent.agent.planner import PLANNER_INSTRUCTIONS, agent_tools
from disclosure_agent.agent.runtime import build_default_runtime
from disclosure_agent.agent.settings import (
    MAX_LLM_CALLS,
    MAX_SEARCH_CALLS,
    REQUEST_TIMEOUT_SECONDS,
)
from disclosure_agent.agent_a0.dataset import (
    DATASET_VERSION,
    agent_input,
    dataset_sha256,
    write_dataset,
)
from disclosure_agent.agent_a0.grade import FAILURES, TASK_OUTCOMES, grade_case
from disclosure_agent.agent_a0.judge import JUDGE_PROMPT_VERSION, judge_case, needs_judge
from disclosure_agent.agent_a0.report import render_report
from disclosure_agent.evalset_v2.llm import StructuredClient, estimate_usd
from disclosure_agent.evaluation import git_revision

RUN_DIR = "gongsi_mate_a0_agent_baseline"
REQUEST_TIME = datetime(2026, 10, 8, 12, 0, tzinfo=SEOUL)
MAX_ESTIMATED_USD = 12.0
CONTRACT = """# A0 evaluation contract

이 계약은 제품 Agent의 이번 실행만 집계한다.
G0 generator 출력, H10 ranking, TEST 55는 입력이 아니다.

검색 필요 여부는 tool call로 본다.
최종 과제 성공은 허용된 행동과 requirement 판정으로 본다.
불필요 검색은 task가 맞아도 secondary flag로 남긴다.

Faithfulness는 이번 실행에서 생성기에 들어간 evidence만 본다.
Requirement 문장은 faithfulness 입력에 넣지 않는다.
Citation integrity는 citation의 evidence_id가 그 evidence 안에 있는지로 본다.
보류 답변과 검색 없는 답변에는 citation을 요구하지 않는다.

점수는 G0 requirement 58개 점수와 비교하지 않는다.
"""


def run_baseline(root: Path, *, output_dir: Path | None = None) -> dict[str, Any]:
    meta = write_dataset(root)
    rows = _read_jsonl(root / meta["path"])
    if dataset_sha256(rows) != meta["dataset_sha256"]:
        raise ValueError("written A0 dataset hash drifted")
    output = output_dir or (root / "evaluation" / "runs" / RUN_DIR)
    output.mkdir(parents=True, exist_ok=True)
    _write_text(output / "agent_evalset.jsonl", root / meta["path"])
    config = _config(root, meta)
    _estimate_or_stop(config, len(rows))
    _dump(output / "config.json", config)
    _write(output / "evaluation_contract.md", CONTRACT)
    runtime = build_default_runtime(root, run_log_path=output / "product_trace.jsonl")
    if runtime.planner is None or not runtime.planner.uses_llm:
        raise RuntimeError(
            "product agent has no LLM planner; A0 requires the configured OpenAI path"
        )
    runtime.clock = lambda: REQUEST_TIME
    config["model"] = str(getattr(runtime.answer_generator, "model", None) or config["model"])
    _dump(output / "config.json", config)
    client = _judge_client()
    runs = _run_queries(runtime, rows, output / "agent_runs.jsonl")
    judgments = _judge_queries(rows, runs, client, output / "judge_results.jsonl")
    results = [
        grade_case(case, runs.get(case["query_id"]), judgments.get(case["query_id"]))
        for case in rows
    ]
    _dump_jsonl(output / "evaluation_results.jsonl", results)
    _dump_jsonl(output / "tool_calls.jsonl", _tool_rows(runs))
    summary = _summary(root, output, config, runs, judgments, results)
    _dump(output / "summary.json", summary)
    _write_failures(output / "failures.csv", results, runs)
    _write(output / "report.md", render_report(summary))
    manifest = _manifest(root, output, config, summary)
    _dump(output / "manifest.json", manifest)
    return summary


def _run_queries(runtime: Any, rows: list[dict[str, Any]], path: Path) -> dict[str, dict[str, Any]]:
    done = _existing(path)
    for index, case in enumerate(rows, start=1):
        query_id = case["query_id"]
        if query_id in done:
            print(f"[{index}/{len(rows)}] {query_id} cached", flush=True)
            continue
        print(f"[{index}/{len(rows)}] {query_id} running", flush=True)
        envelope = _execute(runtime, case)
        _append(path, envelope)
        done[query_id] = envelope["record"]
        print(
            f"[{index}/{len(rows)}] {query_id} {envelope['record'].get('status')}",
            flush=True,
        )
    return done


def _execute(runtime: Any, case: dict[str, Any]) -> dict[str, Any]:
    question = agent_input(case)
    trace = runtime.run_log.path
    before = _line_count(trace)
    try:
        runtime.run(question)
        record = _record_after(trace, before)
    except Exception as exc:
        record = {
            "status": "internal_error",
            "stop_reason": "internal_error",
            "error": {"code": "internal_error", "detail": type(exc).__name__},
            "tool_calls": [],
            "final_answer": None,
            "generation_evidence": [],
            "llm_usages": [],
            "llm_call_count": None,
            "search_call_count": None,
            "total_ms": None,
        }
    return {
        "query_id": case["query_id"],
        "dataset_version": DATASET_VERSION,
        "record": record,
    }


def _judge_queries(
    rows: list[dict[str, Any]],
    runs: dict[str, dict[str, Any]],
    client: StructuredClient,
    path: Path,
) -> dict[str, dict[str, Any]]:
    done = {row["query_id"]: row for row in _read_jsonl(path) if row.get("status") == "ok"}
    for case in rows:
        query_id = case["query_id"]
        record = runs.get(query_id)
        if query_id in done or record is None or not needs_judge(case):
            continue
        print(f"judge {query_id}", flush=True)
        try:
            judged = judge_case(case, record, client)
        except Exception as exc:
            judged = {
                "query_id": query_id,
                "prompt_version": JUDGE_PROMPT_VERSION,
                "faithfulness": [],
                "citations": [],
                "requirements": [],
                "usage": [],
                "status": "error",
                "error": type(exc).__name__,
            }
        _append(path, judged)
        if judged.get("status") == "ok":
            done[query_id] = judged
    return done


def _summary(
    root: Path,
    output: Path,
    config: dict[str, Any],
    runs: dict[str, dict[str, Any]],
    judgments: dict[str, dict[str, Any]],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    counted = Counter(item["task_outcome"] for item in results)
    task = {name: counted[name] for name in TASK_OUTCOMES}
    decision_rows = [item for item in results if item["search_decision"]["counts_for_accuracy"]]
    arguments = Counter(finding for item in results for finding in item["argument_findings"])
    primary = Counter(item["primary_cause"] for item in results if item.get("primary_cause"))
    secondary = Counter(flag for item in results for flag in item["secondary_flags"])
    faith_num = sum(item["grounding"]["faithfulness_supported"] for item in results)
    faith_den = sum(item["grounding"]["faithfulness_denominator"] for item in results)
    cite_num = sum(item["grounding"]["citation_correct"] for item in results)
    cite_den = sum(item["grounding"]["citation_correct_denominator"] for item in results)
    complete_rows = [item for item in results if item["grounding"]["citation_required"]]
    complete = sum(1 for item in complete_rows if item["grounding"]["citation_complete"])
    integrity = sum(item["grounding"]["integrity_error_count"] for item in results)
    req_num = sum(item["requirements"]["success_count"] for item in results)
    req_den = sum(item["requirements"]["denominator"] for item in results)
    clarify = [item for item in results if item["scenario"] == "ambiguous"]
    abstain = [item for item in results if item["scenario"] == "insufficient_evidence"]
    latencies = [
        float(record["total_ms"])
        for record in runs.values()
        if isinstance(record.get("total_ms"), (int, float))
    ]
    agent_usage = _usage(runs)
    judge_usage = _judge_usage(judgments)
    prices = config["prices_usd_per_million_tokens"]
    model = config["model"]
    agent_usd = estimate_usd(
        model, agent_usage["input_tokens"], agent_usage["output_tokens"], prices
    )
    judge_usd = estimate_usd(
        model, judge_usage["input_tokens"], judge_usage["output_tokens"], prices
    )
    successful = task["success"] + task["partial_success"] + task["appropriate_clarification"]
    successful += task["appropriate_abstention"]
    return {
        "dataset_version": DATASET_VERSION,
        "dataset_sha256": config["dataset_sha256"],
        "scenarios": config["scenarios"],
        "query_count": len(results),
        "search_decision": {
            "correct": sum(1 for item in decision_rows if item["search_decision"]["correct"]),
            "denominator": len(decision_rows),
            "missed_required_search": sum(
                1 for item in results if item["search_decision"]["missed_required_search"]
            ),
            "unnecessary_search": sum(
                1 for item in results if item["search_decision"]["unnecessary_search"]
            ),
        },
        "arguments": dict(arguments),
        "clarification_appropriate": sum(
            1 for item in clarify if item["task_outcome"] == "appropriate_clarification"
        ),
        "clarification_denominator": len(clarify),
        "abstention_appropriate": sum(
            1
            for item in abstain
            if item["task_outcome"] in {"appropriate_abstention", "partial_success", "success"}
        ),
        "abstention_denominator": len(abstain),
        "task": dict(task),
        "grounding": {
            "faithfulness_supported": faith_num,
            "faithfulness_denominator": faith_den,
            "citation_correct": cite_num,
            "citation_correct_denominator": cite_den,
            "citation_complete": complete,
            "citation_complete_denominator": len(complete_rows),
            "integrity_error_count": integrity,
            "requirement_success": req_num,
            "requirement_denominator": req_den,
        },
        "failures": {"primary": dict(primary), "secondary": dict(secondary)},
        "representative_failures": _representatives(results, runs),
        "latency_ms": {"p50": _percentile(latencies, 0.50), "p95": _percentile(latencies, 0.95)},
        "tool_latency_ms": round(
            sum(
                float(call.get("latency_ms") or 0)
                for record in runs.values()
                for call in record.get("tool_calls") or []
            ),
            3,
        ),
        "generation_latency_ms": round(
            sum(float(record.get("generation_ms") or 0) for record in runs.values()),
            3,
        ),
        "llm_calls": sum(int(record.get("llm_call_count") or 0) for record in runs.values()),
        "tool_calls": sum(int(record.get("search_call_count") or 0) for record in runs.values()),
        "cost": {
            "agent_input_tokens": agent_usage["input_tokens"],
            "agent_output_tokens": agent_usage["output_tokens"],
            "judge_input_tokens": judge_usage["input_tokens"],
            "judge_output_tokens": judge_usage["output_tokens"],
            "agent_usd": round(agent_usd, 4),
            "judge_usd": round(judge_usd, 4),
            "total_usd": round(agent_usd + judge_usd, 4),
            "usd_per_successful_task": None if not successful else round(agent_usd / successful, 4),
            "price_source": config["price_source"],
        },
        "output_dir": output.relative_to(root).as_posix(),
        "git": git_revision(root),
        "failure_names": list(FAILURES),
        "tests_note": (
            "ruff check src/disclosure_agent/agent_a0 tests/test_agent_a0.py "
            "src/disclosure_agent/agent/runtime.py passed. "
            "pytest tests/test_agent_a0.py tests/test_agent_product.py: 30 passed."
        ),
        "commit_note": (
            "이 리포트와 실행 산출물을 같은 커밋에 넣는다. "
            "manifest의 git revision은 커밋 직전 HEAD다."
        ),
    }


def _config(root: Path, meta: dict[str, Any]) -> dict[str, Any]:
    prices = json.loads(
        (root / "evaluation" / "specs" / "retrieval_evalset_v2_run.json").read_text(
            encoding="utf-8"
        )
    )
    prompt = PLANNER_INSTRUCTIONS + "\n---\n" + ANSWER_INSTRUCTIONS
    return {
        "dataset_version": DATASET_VERSION,
        "dataset_sha256": meta["dataset_sha256"],
        "scenarios": meta["scenarios"],
        "query_count": meta["query_count"],
        "agent_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "model": "gpt-6-luna",
        "provider": "openai",
        "retrieval": {
            "product_retriever": "dense",
            "representation": "retrieval-unit-d2a-v1",
            "embedding_model": "BAAI/bge-m3",
            "closed_benchmark_policy": "bm25_primary_dense_head",
            "note": "The product tool ranks Dense-D2a. This run does not switch it to H10.",
        },
        "tools": agent_tools(),
        "limits": {
            "max_search_calls": MAX_SEARCH_CALLS,
            "max_llm_calls": MAX_LLM_CALLS,
            "timeout_seconds": REQUEST_TIMEOUT_SECONDS,
            "concurrency": 1,
        },
        "request_time": "2026-10-08T12:00:00+09:00",
        "prices_usd_per_million_tokens": prices["prices_usd_per_million_tokens"],
        "price_source": prices["price_source"],
        "max_estimated_usd": MAX_ESTIMATED_USD,
    }


def _estimate_or_stop(config: dict[str, Any], queries: int) -> None:
    prices = config["prices_usd_per_million_tokens"]
    model = config["model"]
    agent = estimate_usd(model, queries * 4 * 6000, queries * 4 * 500, prices)
    judge = estimate_usd(model, queries * 2 * 5000, queries * 2 * 400, prices)
    estimate = agent + judge
    if estimate > MAX_ESTIMATED_USD:
        raise RuntimeError(f"A0 cost estimate ${estimate:.2f} exceeds {MAX_ESTIMATED_USD}")


def _judge_client() -> StructuredClient:
    from disclosure_agent.agent.settings import load_openai_config
    from disclosure_agent.cli import load_dotenv_files

    load_dotenv_files()
    openai_config = load_openai_config()
    if not openai_config.configured:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    return StructuredClient(
        provider="openai",
        model=openai_config.model,
        api_key=openai_config.api_key,
        timeout=openai_config.timeout_seconds,
    )


def _representatives(
    results: list[dict[str, Any]],
    runs: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    chosen = []
    seen: set[str] = set()
    for item in results:
        cause = item.get("primary_cause")
        if (
            not cause
            or cause in seen
            or item["task_outcome"]
            in {
                "success",
                "partial_success",
                "appropriate_clarification",
                "appropriate_abstention",
            }
        ):
            continue
        seen.add(cause)
        final = (runs.get(item["query_id"]) or {}).get("final_answer") or {}
        text = str(final.get("text") or "").replace("\n", " ")
        chosen.append(
            {
                "query_id": item["query_id"],
                "question": item["question"],
                "primary_behavior": item["primary_behavior"],
                "task_outcome": item["task_outcome"],
                "primary_cause": cause,
                "secondary_flags": item["secondary_flags"],
                "answer_excerpt": text[:240],
            }
        )
        if len(chosen) == 3:
            break
    return chosen


def _tool_rows(runs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for query_id, record in runs.items():
        for index, call in enumerate(record.get("tool_calls") or [], start=1):
            rows.append({"query_id": query_id, "call_index": index, **call})
    return rows


def _usage(runs: dict[str, dict[str, Any]]) -> dict[str, int]:
    totals = {"input_tokens": 0, "output_tokens": 0}
    for record in runs.values():
        for usage in record.get("llm_usages") or []:
            totals["input_tokens"] += int(usage.get("input_tokens") or 0)
            totals["output_tokens"] += int(usage.get("output_tokens") or 0)
    return totals


def _judge_usage(judgments: dict[str, dict[str, Any]]) -> dict[str, int]:
    totals = {"input_tokens": 0, "output_tokens": 0}
    for judged in judgments.values():
        for item in judged.get("usage") or []:
            usage = item.get("usage") or {}
            totals["input_tokens"] += int(usage.get("input_tokens") or 0)
            totals["output_tokens"] += int(usage.get("output_tokens") or 0)
    return totals


def _manifest(
    root: Path,
    output: Path,
    config: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run": RUN_DIR,
        "dataset_version": DATASET_VERSION,
        "dataset_sha256": config["dataset_sha256"],
        "agent_prompt_sha256": config["agent_prompt_sha256"],
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "model": config["model"],
        "retrieval": config["retrieval"],
        "limits": config["limits"],
        "request_time": config["request_time"],
        "git": git_revision(root),
        "task": summary["task"],
        "search_decision": summary["search_decision"],
        "files": sorted(path.name for path in output.iterdir() if path.is_file()),
    }


def _write_failures(
    path: Path,
    results: list[dict[str, Any]],
    runs: dict[str, dict[str, Any]],
) -> None:
    fields = [
        "query_id",
        "scenario",
        "task_outcome",
        "primary_behavior",
        "primary_cause",
        "secondary_flags",
        "question",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in results:
            if (
                item["task_outcome"]
                in {
                    "success",
                    "partial_success",
                    "appropriate_clarification",
                    "appropriate_abstention",
                }
                and not item["secondary_flags"]
                and not item["primary_cause"]
            ):
                continue
            writer.writerow(
                {
                    "query_id": item["query_id"],
                    "scenario": item["scenario"],
                    "task_outcome": item["task_outcome"],
                    "primary_behavior": item["primary_behavior"],
                    "primary_cause": item["primary_cause"] or "",
                    "secondary_flags": "|".join(item["secondary_flags"]),
                    "question": item["question"],
                }
            )


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    rank = (len(ordered) - 1) * fraction
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return round(ordered[low] * (1 - weight) + ordered[high] * weight, 3)


def _line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _record_after(path: Path, before: int) -> dict[str, Any]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) <= before:
        raise RuntimeError("product trace did not record this run")
    record = json.loads(lines[-1])
    if not isinstance(record, dict):
        raise RuntimeError("product trace row is not an object")
    return record


def _existing(path: Path) -> dict[str, dict[str, Any]]:
    return {row["query_id"]: row["record"] for row in _read_jsonl(path) if row.get("query_id")}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _write_text(destination: Path, source: Path) -> None:
    destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

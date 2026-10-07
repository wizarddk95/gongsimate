"""Run G0 on the frozen H10 DEV top-10 and write the evaluation artifacts.

Retrieval is not executed. A resumed run reuses a completed call with the same
input hash. Transport retries send the same prompt again.
"""

from __future__ import annotations

import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from disclosure_agent.cli import load_dotenv_files
from disclosure_agent.errors import UsageError
from disclosure_agent.evalset_v2.llm import CallLog, StructuredClient, estimate_usd
from disclosure_agent.evalset_v2.spec import load_run_config
from disclosure_agent.evalset_v2_1.grounded_generation import (
    BASELINE_ID,
    CITATION_INSTRUCTIONS,
    CITATION_PROMPT_VERSION,
    DECODING,
    EVALUATOR_VERSION,
    FAILURE_COLUMNS,
    FAITHFULNESS_INSTRUCTIONS,
    FAITHFULNESS_PROMPT_VERSION,
    GENERATOR_INSTRUCTIONS,
    GENERATOR_PROMPT_VERSION,
    REQUIREMENT_INSTRUCTIONS,
    REQUIREMENT_PROMPT_VERSION,
    RUN_DIR_NAME,
    CitationOutput,
    FaithfulnessOutput,
    G0Generation,
    RequirementOutput,
    assemble_query_result,
    call_input_hash,
    citation_payload,
    failure_rows,
    faithfulness_payload,
    generation_jsonl_row,
    judge_jsonl_row,
    load_g0_packets,
    prompt_sha256,
    render_report,
    requirement_payload,
    resolve_g0_models,
    result_jsonl_row,
    summarize_results,
)
from disclosure_agent.evaluation import git_revision


def run_g0(root: Path) -> dict[str, Any]:
    load_dotenv_files()
    models = resolve_g0_models(dict(os.environ))
    if not models["api_key_present"]:
        raise UsageError("OPENAI_API_KEY is empty")
    population = load_g0_packets(root)
    packets = population["packets"]
    verification = population["verification"]
    prices, budget = _prices(root)
    estimate = _preflight_usd(packets, prices, models)
    if estimate is not None and estimate > budget:
        raise UsageError(
            f"G0 cost estimate ${estimate} exceeds max_estimated_usd {budget}"
        )
    output = root / "evaluation" / "runs" / RUN_DIR_NAME
    output.mkdir(parents=True, exist_ok=True)
    key = os.environ["OPENAI_API_KEY"].strip()
    generator = StructuredClient(
        provider=models["provider"],
        model=models["generator_model"],
        api_key=key,
        timeout=models["timeout_seconds"],
    )
    if models["judge_model"] == models["generator_model"]:
        judge = generator
    else:
        judge = StructuredClient(
            provider=models["provider"],
            model=models["judge_model"],
            api_key=key,
            timeout=models["timeout_seconds"],
        )
    logs = {
        "generation": CallLog(output / "calls_generation.jsonl"),
        "faithfulness": CallLog(output / "calls_faithfulness.jsonl"),
        "requirements": CallLog(output / "calls_requirements.jsonl"),
        "citations": CallLog(output / "calls_citations.jsonl"),
    }
    generations = _map(
        models["workers"],
        lambda packet: _generate(packet, logs["generation"], generator, models, prices),
        packets,
    )
    by_generation = {row["query_id"]: row for row in generations}
    faithfulness = _map(
        models["workers"],
        lambda packet: _faithfulness(
            packet, by_generation[packet["query_id"]], logs["faithfulness"], judge, models, prices
        ),
        packets,
    )
    requirements = _map(
        models["workers"],
        lambda packet: _requirements(
            packet, by_generation[packet["query_id"]], logs["requirements"], judge, models, prices
        ),
        packets,
    )
    citations = _map(
        models["workers"],
        lambda packet: _citations(
            packet, by_generation[packet["query_id"]], logs["citations"], judge, models, prices
        ),
        packets,
    )
    by_faith = {row["query_id"]: row for row in faithfulness}
    by_requirements = {row["query_id"]: row for row in requirements}
    by_citations = {row["query_id"]: row for row in citations}
    results = [
        assemble_query_result(
            packet,
            by_generation[packet["query_id"]],
            by_faith[packet["query_id"]],
            by_requirements[packet["query_id"]],
            by_citations[packet["query_id"]],
        )
        for packet in packets
    ]
    summary = summarize_results(results, verification)
    _write_artifacts(output, models, verification, estimate, budget, results, summary, root)
    return summary


def _generate(packet, log, client, models, prices) -> dict[str, Any]:
    record = _complete(
        log,
        client,
        stage="g0_generate",
        prompt_version=GENERATOR_PROMPT_VERSION,
        instructions=GENERATOR_INSTRUCTIONS,
        payload=packet["model_payload"],
        schema=G0Generation,
        models=models,
        prices=prices,
        evaluator_version=None,
    )
    print(
        f"g0 generation {packet['query_id']} {record['parse_status']}",
        flush=True,
    )
    return {"query_id": packet["query_id"], **record}


def _faithfulness(packet, generation, log, client, models, prices) -> dict[str, Any]:
    if generation["parse_status"] != "ok":
        return {
            "query_id": packet["query_id"],
            **_skipped(
                "generation did not parse",
                models,
                FAITHFULNESS_PROMPT_VERSION,
                FAITHFULNESS_INSTRUCTIONS,
            ),
        }
    record = _complete(
        log,
        client,
        stage="g0_faithfulness",
        prompt_version=FAITHFULNESS_PROMPT_VERSION,
        instructions=FAITHFULNESS_INSTRUCTIONS,
        payload=faithfulness_payload(packet, generation),
        schema=FaithfulnessOutput,
        models=models,
        prices=prices,
        evaluator_version=EVALUATOR_VERSION,
    )
    print(f"g0 faithfulness {packet['query_id']} {record['parse_status']}", flush=True)
    return {"query_id": packet["query_id"], **record}


def _requirements(packet, generation, log, client, models, prices) -> dict[str, Any]:
    if generation["parse_status"] != "ok":
        return {
            "query_id": packet["query_id"],
            **_skipped(
                "generation did not parse",
                models,
                REQUIREMENT_PROMPT_VERSION,
                REQUIREMENT_INSTRUCTIONS,
            ),
        }
    record = _complete(
        log,
        client,
        stage="g0_requirements",
        prompt_version=REQUIREMENT_PROMPT_VERSION,
        instructions=REQUIREMENT_INSTRUCTIONS,
        payload=requirement_payload(packet, generation),
        schema=RequirementOutput,
        models=models,
        prices=prices,
        evaluator_version=EVALUATOR_VERSION,
    )
    print(f"g0 requirements {packet['query_id']} {record['parse_status']}", flush=True)
    return {"query_id": packet["query_id"], **record}


def _citations(packet, generation, log, client, models, prices) -> dict[str, Any]:
    if generation["parse_status"] != "ok":
        return {
            "query_id": packet["query_id"],
            **_skipped(
                "generation did not parse",
                models,
                CITATION_PROMPT_VERSION,
                CITATION_INSTRUCTIONS,
            ),
        }
    payload = citation_payload(packet, generation)
    if payload is None:
        print(f"g0 citations {packet['query_id']} skipped", flush=True)
        return {
            "query_id": packet["query_id"],
            **_skipped(
                "no valid citation pair",
                models,
                CITATION_PROMPT_VERSION,
                CITATION_INSTRUCTIONS,
            ),
        }
    record = _complete(
        log,
        client,
        stage="g0_citations",
        prompt_version=CITATION_PROMPT_VERSION,
        instructions=CITATION_INSTRUCTIONS,
        payload=payload,
        schema=CitationOutput,
        models=models,
        prices=prices,
        evaluator_version=EVALUATOR_VERSION,
    )
    print(f"g0 citations {packet['query_id']} {record['parse_status']}", flush=True)
    return {"query_id": packet["query_id"], **record}


def _complete(
    log: CallLog,
    client: StructuredClient,
    *,
    stage: str,
    prompt_version: str,
    instructions: str,
    payload: dict[str, Any],
    schema: type,
    models: dict[str, Any],
    prices: dict[str, Any],
    evaluator_version: str | None,
) -> dict[str, Any]:
    key = call_input_hash(
        stage=stage,
        provider=client.provider,
        model=client.model,
        prompt_version=prompt_version,
        instructions=instructions,
        payload=payload,
        schema_name=schema.__name__,
    )
    existing = log.get(key)
    if existing is not None:
        return _record_from_log(existing)
    last_error = "not started"
    attempts = 0
    for attempt in range(int(models["max_transport_retries"])):
        attempts += 1
        try:
            output, usage, latency_ms = client.complete(
                instructions=instructions,
                payload=payload,
                schema=schema,
            )
        except Exception as exc:
            last_error = _safe_error(exc)
            log.append(
                _log_row(
                    stage=stage,
                    key=key,
                    status="error",
                    output=None,
                    usage=None,
                    latency_ms=None,
                    cost=None,
                    error=last_error,
                    retry_count=attempt,
                    client=client,
                    prompt_version=prompt_version,
                    instructions=instructions,
                    evaluator_version=evaluator_version,
                )
            )
            if attempt + 1 < int(models["max_transport_retries"]):
                time.sleep(min(2**attempt, 8))
            continue
        cost = _cost(client.model, usage, prices)
        row = _log_row(
            stage=stage,
            key=key,
            status="ok",
            output=output,
            usage=usage,
            latency_ms=latency_ms,
            cost=cost,
            error=None,
            retry_count=attempt,
            client=client,
            prompt_version=prompt_version,
            instructions=instructions,
            evaluator_version=evaluator_version,
        )
        log.append(row)
        return _record_from_log(row)
    return _failed(
        last_error,
        client,
        prompt_version,
        instructions,
        evaluator_version,
        key,
        attempts,
    )


def _log_row(
    *,
    stage: str,
    key: str,
    status: str,
    output: dict[str, Any] | None,
    usage: dict[str, int] | None,
    latency_ms: float | None,
    cost: float | None,
    error: str | None,
    retry_count: int,
    client: StructuredClient,
    prompt_version: str,
    instructions: str,
    evaluator_version: str | None,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "status": status,
        "input_hash": key,
        "provider": client.provider,
        "model": client.model,
        "prompt_version": prompt_version,
        "prompt_sha256": prompt_sha256(instructions),
        "evaluator_version": evaluator_version,
        "decoding": dict(DECODING),
        "output": output,
        "usage": usage,
        "latency_ms": latency_ms,
        "estimated_cost_usd": cost,
        "error": error,
        "retry_count": retry_count,
    }


def _record_from_log(row: dict[str, Any]) -> dict[str, Any]:
    output = row.get("output") if row.get("status") == "ok" else None
    record = {
        "parse_status": "ok" if row.get("status") == "ok" else "failed",
        "output": output,
        "raw": output,
        "provider": row.get("provider"),
        "model": row.get("model"),
        "prompt_version": row.get("prompt_version"),
        "prompt_sha256": row.get("prompt_sha256"),
        "evaluator_version": row.get("evaluator_version"),
        "decoding": row.get("decoding") or dict(DECODING),
        "latency_ms": row.get("latency_ms"),
        "usage": row.get("usage"),
        "estimated_cost_usd": row.get("estimated_cost_usd"),
        "error": row.get("error"),
        "input_sha256": row.get("input_hash"),
        "retry_count": row.get("retry_count"),
        "reason": "",
        "answer_relevance": None,
    }
    if isinstance(output, dict):
        record["reason"] = str(output.get("reason") or "")
        record["answer_relevance"] = output.get("answer_relevance")
    return record


def _failed(
    error: str,
    client: StructuredClient,
    prompt_version: str,
    instructions: str,
    evaluator_version: str | None,
    key: str,
    attempts: int,
) -> dict[str, Any]:
    return {
        "parse_status": "failed",
        "output": None,
        "raw": None,
        "provider": client.provider,
        "model": client.model,
        "prompt_version": prompt_version,
        "prompt_sha256": prompt_sha256(instructions),
        "evaluator_version": evaluator_version,
        "decoding": dict(DECODING),
        "latency_ms": None,
        "usage": None,
        "estimated_cost_usd": None,
        "error": error,
        "input_sha256": key,
        "retry_count": attempts,
        "reason": "",
        "answer_relevance": None,
    }


def _skipped(
    reason: str, models: dict[str, Any], prompt_version: str, instructions: str
) -> dict[str, Any]:
    model = (
        models["generator_model"]
        if prompt_version == GENERATOR_PROMPT_VERSION
        else models["judge_model"]
    )
    return {
        "parse_status": "skipped",
        "output": None,
        "raw": None,
        "provider": models["provider"],
        "model": model,
        "prompt_version": prompt_version,
        "prompt_sha256": prompt_sha256(instructions),
        "evaluator_version": EVALUATOR_VERSION,
        "decoding": dict(DECODING),
        "latency_ms": None,
        "usage": None,
        "estimated_cost_usd": None,
        "error": None,
        "input_sha256": None,
        "retry_count": 0,
        "reason": reason,
        "answer_relevance": None,
    }


def _map(workers: int, function, packets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if workers == 1 or len(packets) < 2:
        return [function(packet) for packet in packets]
    done: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(function, packet): packet["query_id"] for packet in packets}
        for future in as_completed(futures):
            done[futures[future]] = future.result()
    return [done[packet["query_id"]] for packet in packets]


def _prices(root: Path) -> tuple[dict[str, Any], float]:
    config = load_run_config(root)
    return config["prices_usd_per_million_tokens"], float(config["max_estimated_usd"])


def _preflight_usd(
    packets: list[dict[str, Any]], prices: dict[str, Any], models: dict[str, Any]
) -> float | None:
    context = sum(int(packet["context_token_count"]) for packet in packets)
    planned = (
        (models["generator_model"], context, len(packets) * 800),
        (models["judge_model"], context, len(packets) * 500),
        (models["judge_model"], max(context // 5, 1), len(packets) * 400),
        (models["judge_model"], max(context // 2, 1), len(packets) * 400),
    )
    total = 0.0
    for model, input_tokens, output_tokens in planned:
        if model not in prices:
            return None
        total += estimate_usd(model, input_tokens, output_tokens, prices)
    return round(total, 4)


def _cost(model: str, usage: dict[str, int] | None, prices: dict[str, Any]) -> float | None:
    if not isinstance(usage, dict):
        return None
    if "input_tokens" not in usage or "output_tokens" not in usage:
        return None
    if model not in prices:
        return None
    return round(
        estimate_usd(model, int(usage["input_tokens"]), int(usage["output_tokens"]), prices),
        6,
    )


def _safe_error(exc: BaseException) -> str:
    text = " ".join(str(exc).split())[:200]
    lowered = text.lower()
    if "api_key" in lowered or "sk-" in text:
        return type(exc).__name__
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _write_artifacts(
    output: Path,
    models: dict[str, Any],
    verification: dict[str, Any],
    estimate: float | None,
    budget: float,
    results: list[dict[str, Any]],
    summary: dict[str, Any],
    root: Path,
) -> None:
    git = git_revision(root)
    manifest = {
        "baseline": BASELINE_ID,
        "dataset_id": verification["dataset_id"],
        "dataset_sha256": verification["dataset_sha256"],
        "population": "dev fully_answerable",
        "question_count": verification["question_count"],
        "requirement_count": verification["requirement_count"],
        "retrieval_run_id": verification["h10_run"],
        "retrieval_policy": verification["retrieval_policy"],
        "retrieval_reexecuted": False,
        "test_questions_generated": 0,
        "generator_provider": models["provider"],
        "generator_model": models["generator_model"],
        "judge_provider": models["provider"],
        "judge_model": models["judge_model"],
        "prompt_versions": {
            "generator": GENERATOR_PROMPT_VERSION,
            "faithfulness": FAITHFULNESS_PROMPT_VERSION,
            "requirements": REQUIREMENT_PROMPT_VERSION,
            "citations": CITATION_PROMPT_VERSION,
        },
        "prompt_sha256": {
            "generator": prompt_sha256(GENERATOR_INSTRUCTIONS),
            "faithfulness": prompt_sha256(FAITHFULNESS_INSTRUCTIONS),
            "requirements": prompt_sha256(REQUIREMENT_INSTRUCTIONS),
            "citations": prompt_sha256(CITATION_INSTRUCTIONS),
        },
        "evaluator_version": EVALUATOR_VERSION,
        "decoding": dict(DECODING),
        "frozen_retrieval": verification,
        "git_commit": git.get("git_commit"),
        "git_dirty": git.get("git_dirty"),
    }
    config = {
        "provider": models["provider"],
        "generator_model": models["generator_model"],
        "judge_model": models["judge_model"],
        "timeout_seconds": models["timeout_seconds"],
        "workers": models["workers"],
        "max_transport_retries": models["max_transport_retries"],
        "decoding": dict(DECODING),
        "preflight_estimated_usd": estimate,
        "max_estimated_usd": budget,
        "retrieval_reexecuted": False,
        "test_questions_generated": 0,
    }
    _dump(output / "manifest.json", manifest)
    _dump(output / "config.json", config)
    _jsonl(
        output / "generations.jsonl",
        (generation_jsonl_row(row, verification) for row in results),
    )
    _jsonl(
        output / "judge_faithfulness.jsonl",
        (judge_jsonl_row(row, "judge_faithfulness") for row in results),
    )
    _jsonl(
        output / "judge_requirements.jsonl",
        (judge_jsonl_row(row, "judge_requirements") for row in results),
    )
    _jsonl(
        output / "judge_citations.jsonl",
        (judge_jsonl_row(row, "judge_citations") for row in results),
    )
    _jsonl(output / "results.jsonl", (result_jsonl_row(row) for row in results))
    public_summary = dict(summary)
    _dump(output / "summary.json", public_summary)
    _write_failures(output / "failures.csv", failure_rows(results))
    (output / "report.md").write_text(render_report(summary), encoding="utf-8")


def _dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_failures(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FAILURE_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv(row.get(key)) for key in FAILURE_COLUMNS})


def _csv(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)

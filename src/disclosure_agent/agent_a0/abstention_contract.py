"""Rescore the stored A0 run with the abstention contract. Does not call a judge.

The baseline directory is read and left in place. Corrected rows go to a
separate run directory.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from disclosure_agent.agent_a0.grade import TASK_OUTCOMES, grade_case
from disclosure_agent.agent_a0.run import CONTRACT, RUN_DIR
from disclosure_agent.errors import UsageError

CORRECTED_RUN_DIR = "gongsi_mate_a0_abstention_corrected"
CONTRACT_VERSION = "a0-abstention-contract-v1"


def write_correction(root: Path, output_dir: Path | None = None) -> dict[str, Any]:
    source = root / "evaluation" / "runs" / RUN_DIR
    destination = output_dir or (root / "evaluation" / "runs" / CORRECTED_RUN_DIR)
    if destination.resolve() == source.resolve():
        raise UsageError("corrected A0 evaluation cannot overwrite the baseline run")
    cases = _read_jsonl(source / "agent_evalset.jsonl")
    traces = {
        str(row.get("raw_user_query") or ""): row
        for row in _read_jsonl(source / "product_trace.jsonl")
    }
    judgments = {
        str(row.get("query_id") or ""): row for row in _read_jsonl(source / "judge_results.jsonl")
    }
    raw_results = {
        str(row.get("query_id") or ""): row
        for row in _read_jsonl(source / "evaluation_results.jsonl")
    }
    corrected: list[dict[str, Any]] = []
    for case in cases:
        record = traces.get(case["question"])
        judgment = judgments.get(case["query_id"])
        corrected.append(grade_case(case, record, judgment))
    summary = _summary(corrected, raw_results)
    destination.mkdir(parents=True, exist_ok=True)
    _write(destination / "evaluation_contract.md", CONTRACT)
    _jsonl(destination / "evaluation_results.jsonl", corrected)
    _dump(destination / "summary.json", summary)
    _dump(
        destination / "manifest.json",
        {
            "contract_version": CONTRACT_VERSION,
            "source_run": RUN_DIR,
            "judge_recalled": 0,
            "generator_factual_flag_used": False,
            "source_artifacts_overwritten": False,
        },
    )
    (destination / "report.md").write_text(_report(summary), encoding="utf-8")
    return summary


def _summary(
    corrected: list[dict[str, Any]],
    raw_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    counted = Counter(item["task_outcome"] for item in corrected)
    task = {name: counted[name] for name in TASK_OUTCOMES}
    raw_counted = Counter(
        str(item.get("task_outcome") or "") for item in raw_results.values()
    )
    faith_num = sum(item["grounding"]["faithfulness_supported"] for item in corrected)
    faith_den = sum(item["grounding"]["faithfulness_denominator"] for item in corrected)
    raw_faith_den = sum(
        int((item.get("grounding") or {}).get("faithfulness_denominator") or 0)
        for item in raw_results.values()
    )
    changes = []
    for item in corrected:
        previous = raw_results.get(item["query_id"]) or {}
        if previous.get("task_outcome") != item["task_outcome"] or previous.get(
            "primary_cause"
        ) != item.get("primary_cause"):
            changes.append(
                {
                    "query_id": item["query_id"],
                    "raw_task_outcome": previous.get("task_outcome"),
                    "corrected_task_outcome": item["task_outcome"],
                    "raw_primary_cause": previous.get("primary_cause"),
                    "corrected_primary_cause": item.get("primary_cause"),
                    "factual_claim_count": item["grounding"]["factual_claim_count"],
                }
            )
    abstain = [item for item in corrected if item["scenario"] == "insufficient_evidence"]
    return {
        "contract_version": CONTRACT_VERSION,
        "source_run": RUN_DIR,
        "query_count": len(corrected),
        "judge_recalled": 0,
        "task": task,
        "raw_task": {name: raw_counted[name] for name in TASK_OUTCOMES},
        "faithfulness": {"numerator": faith_num, "denominator": faith_den},
        "raw_faithfulness_denominator": raw_faith_den,
        "abstention_appropriate": sum(
            1
            for item in abstain
            if item["task_outcome"] in {"appropriate_abstention", "partial_success", "success"}
        ),
        "abstention_denominator": len(abstain),
        "outcome_changes": changes,
    }


def _report(summary: dict[str, Any]) -> str:
    lines = [
        "# A0 abstention contract correction",
        "",
        f"Source `{summary['source_run']}`. Judge recalled {summary['judge_recalled']}.",
        "Generator factual flags were not read.",
        "",
        "## Task",
        "",
        "| outcome | raw | corrected |",
        "| --- | ---: | ---: |",
    ]
    for name in TASK_OUTCOMES:
        lines.append(
            f"| {name} | {summary['raw_task'][name]} | {summary['task'][name]} |"
        )
    faith = summary["faithfulness"]
    lines.extend(
        [
            "",
            (
                f"Faithfulness denominator {summary['raw_faithfulness_denominator']} -> "
                f"{faith['denominator']}. Numerator {faith['numerator']}."
            ),
            (
                f"Abstention appropriateness {summary['abstention_appropriate']}/"
                f"{summary['abstention_denominator']}."
            ),
            "",
            "## Outcome changes",
            "",
        ]
    )
    if not summary["outcome_changes"]:
        lines.append("No task outcome or primary cause changed.")
    else:
        for row in summary["outcome_changes"]:
            lines.append(
                f"- `{row['query_id']}`: {row['raw_task_outcome']}/{row['raw_primary_cause']} -> "
                f"{row['corrected_task_outcome']}/{row['corrected_primary_cause']}."
            )
    lines.append("")
    return "\n".join(lines)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write(path: Path, text: str) -> None:
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")

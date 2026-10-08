# ruff: noqa: E501
"""Release smoke of the product runtime after A1.

Calls ``AgentRuntime.run`` on eight existing questions and four exposure
probes. It does not score retrieval, compare models, or rewrite A0 artifacts.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

from disclosure_agent.agent.errors import user_message
from disclosure_agent.agent.h10_search import H10Search
from disclosure_agent.agent.planner import ScriptedPlanner
from disclosure_agent.agent.runtime import build_default_runtime
from disclosure_agent.agent.search import SearchDisclosuresRequest
from disclosure_agent.agent.settings import (
    DEFAULT_GENERATION_CONTEXT_TOKEN_BUDGET,
    DEFAULT_GENERATION_MAX_EVIDENCE,
    DEFAULT_RETRIEVAL_TOP_K,
    MAX_LLM_CALLS,
    MAX_SEARCH_CALLS,
    PRODUCT_RETRIEVAL_POLICY,
    PRODUCT_RETRIEVAL_SHORT_LIST_POLICY,
    REQUEST_TIMEOUT_SECONDS,
    AgentLimits,
)
from disclosure_agent.evalset_v2.spec import repo_root
from disclosure_agent.evalset_v2_1.asymmetric_fusion import (
    CANDIDATE_N,
    MARGINAL_SLOT,
    PROTECTED_PREFIX,
)

RUN_DIR_NAME = "gongsi_mate_release_smoke"
A0_DIRS = (
    "gongsi_mate_a0_agent_baseline",
    "gongsi_mate_a0_abstention_corrected",
)
PRICES = {"input": 2.0, "output": 8.0}
UNGROUNDED_DRAFT = "이수페타시스의 주요 사업은 반도체 웨이퍼 제조입니다."
AMOUNT_RE = re.compile(r"\d{1,3}(?:,\d{3})+|\d+\s*(?:원|억|조)")

SCENARIOS: tuple[dict[str, Any], ...] = (
    {
        "id": "no_search",
        "source_query_id": "A0-direct-01",
        "situation": "검색이 필요 없는 제품 설명",
    },
    {
        "id": "clarify",
        "source_query_id": "A0-clarify-03",
        "situation": "회사가 여러 곳으로 읽히는 질문",
    },
    {
        "id": "cited_search",
        "source_query_id": "V2-00107613-business_paragraph-02",
        "situation": "공시에서 확인해야 하는 기업 사실",
    },
    {
        "id": "additional_search",
        "source_query_id": "V2-00107613-multi_element-01",
        "situation": "한 보고서의 두 요소를 함께 묻는 질문",
    },
    {
        "id": "missing_period",
        "source_query_id": "A0-abstain-01",
        "situation": "색인에 없는 과거 기간의 수치",
    },
    {
        "id": "future_confirmed",
        "source_query_id": "A0-abstain-02",
        "situation": "아직 오지 않은 확정 실적",
    },
    {
        "id": "unsupported_company",
        "source_query_id": "A0-abstain-03",
        "situation": "색인되지 않은 기업",
    },
    {
        "id": "primary_business",
        "source_query_id": None,
        "question": "이수페타시스의 주요 사업이 뭐야?",
        "situation": "회사와 사업 내용이 있는 짧은 질문",
    },
)


def main() -> None:
    root = repo_root()
    out = root / "evaluation" / "runs" / RUN_DIR_NAME
    _guard_output(out)
    out.mkdir(parents=True, exist_ok=True)
    before = _a0_hashes(root)
    questions = _load_questions(root)
    policy = _policy_snapshot(root)
    (out / "policy.json").write_text(
        json.dumps(policy, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    runtime = build_default_runtime(root, run_log_path=out / "agent_runs.jsonl")
    original_planner = runtime.planner
    original_search = runtime.search
    original_limits = runtime.limits
    original_timeout = runtime.timeout_seconds
    rows: list[dict[str, Any]] = []
    try:
        for spec in SCENARIOS:
            question = spec.get("question") or questions[spec["source_query_id"]]
            print(f"scenario {spec['id']}", flush=True)
            row = _run_case(
                runtime,
                scenario_id=spec["id"],
                situation=spec["situation"],
                source_query_id=spec.get("source_query_id"),
                question=question,
                kind="scenario",
            )
            row["verdict"] = _scenario_verdict(spec["id"], row)
            rows.append(row)
            _append_jsonl(out / "runs.jsonl", row)
            print(
                f"  {row['verdict']} status={row['status']} policy={row['retrieval_policy']} top_k={row['top_k']}",
                flush=True,
            )
        exposures = [
            _probe_h10(runtime, original_search),
            _probe_evidence_required(runtime),
            _probe_budget(runtime, original_search),
            _probe_timeout(runtime, original_planner),
        ]
    finally:
        runtime.planner = original_planner
        runtime.search = original_search
        runtime.limits = original_limits
        runtime.timeout_seconds = original_timeout
    for row in exposures:
        _append_jsonl(out / "exposures.jsonl", row)
        print(f"exposure {row['id']} {row['verdict']}", flush=True)
    after = _a0_hashes(root)
    summary = {
        "run_dir": RUN_DIR_NAME,
        "git_head": policy["git_head"],
        "product_retrieval_policy": PRODUCT_RETRIEVAL_POLICY,
        "scenarios": [
            {
                "id": row["id"],
                "verdict": row["verdict"],
                "status": row["status"],
                "retrieval_policy": row["retrieval_policy"],
                "top_k": row["top_k"],
            }
            for row in rows
        ],
        "exposures": [
            {"id": row["id"], "verdict": row["verdict"], "user_message": row["user_message"]}
            for row in exposures
        ],
        "observed_top_k": sorted({value for row in rows for value in row["top_k"]}),
        "estimated_usd": round(sum(row["estimated_usd"] for row in rows + exposures), 6),
        "a0_unchanged": before == after,
        "a0_hashes": before,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out / "table.md").write_text(_report(policy, rows, exposures, summary), encoding="utf-8")
    print(json.dumps({"estimated_usd": summary["estimated_usd"], "a0_unchanged": summary["a0_unchanged"]}, ensure_ascii=False), flush=True)


def _guard_output(out: Path) -> None:
    if out.name in A0_DIRS or any(name in out.as_posix() for name in A0_DIRS):
        raise SystemExit("release smoke must not write an A0 directory")


def _a0_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in A0_DIRS:
        directory = root / "evaluation" / "runs" / name
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                hashes[path.relative_to(root).as_posix()] = digest
    return hashes


def _load_questions(root: Path) -> dict[str, str]:
    path = root / "evaluation" / "datasets" / "gongsi_mate_a0_dev" / "agent_evalset.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {row["query_id"]: row["question"] for row in rows}


def _policy_snapshot(root: Path) -> dict[str, Any]:
    return {
        "git_head": _git_head(root),
        "product_retrieval_policy": PRODUCT_RETRIEVAL_POLICY,
        "short_list_policy": PRODUCT_RETRIEVAL_SHORT_LIST_POLICY,
        "search_implementation": "disclosure_agent.agent.h10_search.H10Search",
        "candidate_n": CANDIDATE_N,
        "protected_prefix": PROTECTED_PREFIX,
        "marginal_slot": MARGINAL_SLOT,
        "server_max_top_k": DEFAULT_RETRIEVAL_TOP_K,
        "generation_context_token_budget": DEFAULT_GENERATION_CONTEXT_TOKEN_BUDGET,
        "generation_max_evidence": DEFAULT_GENERATION_MAX_EVIDENCE,
        "max_search_calls": MAX_SEARCH_CALLS,
        "max_llm_calls": MAX_LLM_CALLS,
        "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
        "dense_head_window": (
            "H10은 Dense와 BM25를 각각 20개 뽑은 뒤, BM25 1–9위를 유지한다. "
            "Dense 1위가 BM25 상위 10위 안에 없으면 그 후보를 융합 10위에 넣는다. "
            "이미 BM25 1–10위에 있으면 순서는 BM25 그대로다. "
            "검색 응답은 그 목록의 앞 top_k개다. "
            "따라서 융합 10위의 Dense 후보는 planner가 넘긴 top_k가 10 이상일 때만 검색 응답에 남는다. "
            "그 뒤 답변 창은 그 순서를 유지한 채 4096토큰 또는 20개에 걸릴 때까지 앞에서 자른다. "
            "앞 9개의 프롬프트 비용이 4096을 넘으면 10위는 모델 입력에도 들어가지 않는다."
        ),
        "prices_usd_per_million_tokens": PRICES,
        "price_source": "evaluation/specs/retrieval_evalset_v2_run.json gpt-6-luna",
    }


def _git_head(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _run_case(
    runtime: Any,
    *,
    scenario_id: str,
    situation: str,
    source_query_id: str | None,
    question: str,
    kind: str,
) -> dict[str, Any]:
    try:
        result = runtime.run(question)
    except Exception as exc:
        return {
            "id": scenario_id,
            "kind": kind,
            "situation": situation,
            "source_query_id": source_query_id,
            "question": question,
            "verdict": "FAIL",
            "status": "exception",
            "event": "exception",
            "user_message": "",
            "ui_label": "",
            "answer_text": "",
            "citations": [],
            "evidence": [],
            "citation_integrity": {"ok": False, "missing_evidence": [], "dangling_markers": []},
            "retrieval_policy": None,
            "retrieval_edit": None,
            "retrieval_policy_reason": None,
            "top_k": [],
            "llm_call_count": None,
            "search_call_count": None,
            "latency_ms": None,
            "estimated_usd": 0.0,
            "tokens": {},
            "exception": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "fallback": None,
            "abstain_reason": None,
            "tool_calls": [],
        }
    terminal = result["terminal"]
    record = _last_record(runtime.run_log.path)
    data = terminal.data if isinstance(terminal.data, dict) else {}
    answer = data.get("answer") if isinstance(data.get("answer"), dict) else {}
    evidence = _slim_evidence(data.get("evidence") or record.get("generation_evidence") or [])
    citations = answer.get("citations") or []
    surface = _user_surface(terminal.event, data)
    tokens = _tokens(record)
    tool_calls = [_slim_tool(call) for call in record.get("tool_calls") or []]
    return {
        "id": scenario_id,
        "kind": kind,
        "situation": situation,
        "source_query_id": source_query_id,
        "question": question,
        "status": data.get("status") or data.get("code") or record.get("status"),
        "event": terminal.event,
        "action": data.get("action"),
        "user_message": surface["message"],
        "ui_label": surface["label"],
        "ui_kind": surface["kind"],
        "answer_text": answer.get("text") or "",
        "limitations": answer.get("limitations") or [],
        "citations": citations,
        "evidence": evidence,
        "citation_integrity": _citation_integrity(answer.get("text") or "", citations, evidence),
        "retrieval_policy": record.get("retrieval_policy"),
        "retrieval_edit": record.get("retrieval_edit"),
        "retrieval_policy_reason": record.get("retrieval_policy_reason"),
        "top_k": [call["top_k"] for call in tool_calls if call.get("top_k") is not None],
        "planner_top_k": [
            (action.get("arguments") or {}).get("top_k")
            for action in record.get("planner_actions") or []
            if isinstance(action.get("arguments"), dict) and action["arguments"].get("top_k") is not None
        ],
        "llm_call_count": record.get("llm_call_count"),
        "search_call_count": record.get("search_call_count"),
        "latency_ms": record.get("latency_ms"),
        "estimated_usd": _usd(tokens),
        "tokens": tokens,
        "exception": None if record.get("error") is None else record.get("error"),
        "fallback": record.get("answer_stop") or record.get("evidence_guard"),
        "abstain_reason": data.get("abstain_reason") or record.get("abstain_reason"),
        "resolved_company": data.get("resolved_company") or record.get("resolved_company"),
        "tool_calls": tool_calls,
        "context_evidence_count": record.get("context_evidence_count"),
        "context_token_count": record.get("context_token_count"),
    }


def _probe_h10(runtime: Any, search: H10Search) -> dict[str, Any]:
    runtime.limits = AgentLimits()
    runtime.timeout_seconds = REQUEST_TIMEOUT_SECONDS
    runtime.planner = ScriptedPlanner(
        [
            {
                "query": "주요 사업",
                "company_id": "00107613",
                "period": None,
                "filing_date_from": None,
                "filing_date_to": None,
                "report_types": None,
                "as_of": None,
                "top_k": 10,
            }
        ],
        uses_llm=False,
    )

    def short_list(request: SearchDisclosuresRequest):
        return search(replace(request, eligible_receipt_nos=()))

    runtime.search = short_list
    row = _run_case(
        runtime,
        scenario_id="h10_unavailable",
        situation="후보 창이 20개보다 짧을 때의 사용자 문장",
        source_query_id=None,
        question="이수페타시스의 주요 사업이 뭐야?",
        kind="exposure",
    )
    row["verdict"] = _exposure_verdict(
        "h10_unavailable",
        row,
        expect_policy=PRODUCT_RETRIEVAL_SHORT_LIST_POLICY,
    )
    runtime.search = search
    return row


def _probe_evidence_required(runtime: Any) -> dict[str, Any]:
    runtime.limits = AgentLimits()
    runtime.timeout_seconds = REQUEST_TIMEOUT_SECONDS
    runtime.planner = ScriptedPlanner(
        [
            {"action": "answer_directly", "text": UNGROUNDED_DRAFT, "company_mention": None},
            {"action": "answer_directly", "text": UNGROUNDED_DRAFT, "company_mention": None},
        ],
        uses_llm=False,
    )
    row = _run_case(
        runtime,
        scenario_id="evidence_required",
        situation="근거 없는 기업 사실 초안이 두 번 거절될 때의 사용자 문장",
        source_query_id=None,
        question="이수페타시스의 주요 사업이 뭐야?",
        kind="exposure",
    )
    expected = "확인된 공시 근거가 없어 그 사실을 단정하지 않습니다."
    ok = (
        row["abstain_reason"] == "evidence_required"
        and expected in row["answer_text"]
        and UNGROUNDED_DRAFT not in row["answer_text"]
        and row["citation_integrity"]["ok"]
        and not row["citations"]
    )
    row["verdict"] = "PASS" if ok else "FAIL"
    return row


def _probe_budget(runtime: Any, search: Any) -> dict[str, Any]:
    runtime.search = search
    runtime.timeout_seconds = REQUEST_TIMEOUT_SECONDS
    runtime.limits = AgentLimits(max_search_calls=1, max_llm_calls=0)
    runtime.planner = ScriptedPlanner(
        [
            {
                "query": "주요 사업",
                "company_id": "00107613",
                "period": None,
                "filing_date_from": None,
                "filing_date_to": None,
                "report_types": None,
                "as_of": None,
                "top_k": 10,
            }
        ],
        uses_llm=False,
    )
    row = _run_case(
        runtime,
        scenario_id="answer_budget_exhausted",
        situation="검색 뒤 답변 모델 호출이 남아 있지 않을 때의 사용자 문장",
        source_query_id=None,
        question="이수페타시스의 주요 사업이 뭐야?",
        kind="exposure",
    )
    text = row["answer_text"]
    limitations = " ".join(row.get("limitations") or [])
    showed = "답변 모델 호출 한도" in text or "답변 모델 호출 한도" in limitations
    leaked = "[section]" in text or "traceback" in text.lower() or "retrieval_unit" in text
    ok = (
        row["fallback"] == "generation_budget_fallback"
        and showed
        and not leaked
        and row["citation_integrity"]["ok"]
    )
    row["verdict"] = "PASS" if ok else "FAIL"
    return row


def _probe_timeout(runtime: Any, planner: Any) -> dict[str, Any]:
    runtime.planner = planner
    runtime.limits = AgentLimits()
    runtime.timeout_seconds = -1
    row = _run_case(
        runtime,
        scenario_id="request_timeout",
        situation="요청 제한 시간이 지난 뒤의 사용자 문장",
        source_query_id=None,
        question="안녕하세요",
        kind="exposure",
    )
    ok = (
        row["event"] == "error"
        and row["status"] == "request_timeout"
        and row["user_message"] == user_message("request_timeout")
    )
    row["verdict"] = "PASS" if ok else "FAIL"
    return row


def _scenario_verdict(scenario_id: str, row: dict[str, Any]) -> str:
    if row.get("exception") and row.get("status") == "exception":
        return "FAIL"
    if row["event"] == "error" and row["status"] not in {"request_timeout"}:
        return "FAIL"
    if not row["citation_integrity"]["ok"]:
        return "FAIL"
    if scenario_id == "no_search":
        return "PASS" if row["search_call_count"] == 0 and row["status"] == "completed" else "FAIL"
    if scenario_id == "clarify":
        clarified = row["action"] == "ask_clarification" or row["status"] in {
            "company_ambiguous",
            "company_not_found",
        }
        return "PASS" if clarified and row["search_call_count"] == 0 else "FAIL"
    if scenario_id == "unsupported_company":
        held = row["status"] in {
            "company_unsupported",
            "unsupported_company",
            "abstained",
            "company_ambiguous",
            "company_not_found",
        } or row["action"] == "ask_clarification"
        invented = bool(AMOUNT_RE.search(row["answer_text"]))
        return "PASS" if held and not invented and row["search_call_count"] == 0 else "FAIL"
    if scenario_id in {"missing_period", "future_confirmed"}:
        invented = bool(AMOUNT_RE.search(row["answer_text"]))
        if invented and row["status"] == "completed" and row["action"] != "abstain":
            return "FAIL"
        return "PASS"
    if scenario_id in {"cited_search", "additional_search", "primary_business"}:
        if row["action"] == "answer_directly" and not row["citations"]:
            return "FAIL"
        policies = {call.get("retrieval_policy") for call in row["tool_calls"]}
        policies.discard(None)
        if row["search_call_count"] and policies and policies != {PRODUCT_RETRIEVAL_POLICY}:
            if PRODUCT_RETRIEVAL_SHORT_LIST_POLICY in policies:
                return "KNOWN LIMITATION"
            return "FAIL"
        if row["status"] in {"completed", "abstained", "retrieval_no_result"}:
            return "PASS"
        return "FAIL"
    return "FAIL"


def _exposure_verdict(name: str, row: dict[str, Any], *, expect_policy: str) -> str:
    if row["retrieval_policy"] != expect_policy:
        return "FAIL"
    if not row["citation_integrity"]["ok"]:
        return "FAIL"
    if name == "h10_unavailable":
        distinct = (
            row["status"] == "retrieval_unavailable"
            and row["abstain_reason"] == "h10_unavailable"
            and "없다고 확인된 것은 아닙니다" in row["answer_text"]
            and "h10_unavailable" not in row["answer_text"]
            and "찾지 못했습니다" not in row["answer_text"]
        )
        return "PASS" if distinct else "FAIL"
    return "FAIL"


def _user_surface(event: str, data: dict[str, Any]) -> dict[str, str]:
    if event == "error":
        return {
            "kind": "error",
            "label": "",
            "message": str(data.get("message") or user_message("internal_error")),
        }
    action = data.get("action")
    status = data.get("status")
    reason = data.get("abstain_reason")
    answer = data.get("answer") if isinstance(data.get("answer"), dict) else {}
    text = str(answer.get("text") or "")
    stop = data.get("answer_stop")
    cited = bool(answer.get("citations"))
    if action == "ask_clarification":
        label, kind = "확인이 필요합니다", "clarification"
    elif status in {"company_unsupported", "unsupported_company"} or reason == "unsupported_company":
        label, kind = "지원 범위 밖", "unsupported"
    elif status == "retrieval_unavailable" or reason == "h10_unavailable":
        label, kind = "검색을 마치지 못했습니다", "unavailable"
    elif stop == "generation_budget_fallback" or reason == "answer_budget_exhausted":
        label, kind = ("일부만 확인했습니다", "partial") if cited else ("답변을 마치지 못했습니다", "limited")
    elif status == "no_matching_filing" or reason == "no_matching_filing":
        label, kind = "해당 조건의 공시를 찾지 못함", "missing"
    elif status == "retrieval_no_result" or reason == "no_results":
        label, kind = "해당하는 내용을 찾지 못함", "empty"
    elif reason == "insufficient_evidence" or status == "abstained":
        label, kind = "공시에서 확인하지 못함", "abstained"
    elif action == "answer_directly":
        label, kind = "", "direct"
    elif action == "search_disclosures" and status == "completed":
        label, kind = "공시 원문 기반", "grounded"
    elif action == "abstain":
        label, kind = "답변을 보류했습니다", "abstained"
    else:
        label, kind = "", "answer"
    return {"kind": kind, "label": label, "message": text}


def _citation_integrity(
    text: str,
    citations: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    evidence_ids = {item.get("evidence_id") for item in evidence}
    missing = [
        item.get("evidence_id")
        for item in citations
        if item.get("evidence_id") not in evidence_ids
    ]
    citation_ids = {str(item.get("citation_id")) for item in citations}
    dangling = [mark for mark in re.findall(r"\[(\d+)\]", text) if mark not in citation_ids]
    return {
        "ok": not missing and not dangling,
        "missing_evidence": missing,
        "dangling_markers": dangling,
    }


def _tokens(record: dict[str, Any]) -> dict[str, int]:
    incoming = 0
    outgoing = 0
    for item in record.get("llm_usages") or []:
        if isinstance(item, dict):
            incoming += int(item.get("input_tokens") or 0)
            outgoing += int(item.get("output_tokens") or 0)
    return {"input_tokens": incoming, "output_tokens": outgoing}


def _usd(tokens: dict[str, int]) -> float:
    return round(
        tokens["input_tokens"] / 1_000_000 * PRICES["input"]
        + tokens["output_tokens"] / 1_000_000 * PRICES["output"],
        6,
    )


def _slim_evidence(items: list[Any]) -> list[dict[str, Any]]:
    slim = []
    for item in items:
        if not isinstance(item, dict):
            continue
        slim.append(
            {
                "evidence_id": item.get("evidence_id"),
                "retrieval_unit_id": item.get("retrieval_unit_id"),
                "receipt_no": item.get("receipt_no"),
                "report_name": item.get("report_name"),
                "filing_period": item.get("filing_period"),
                "section_path": item.get("section_path"),
                "excerpt": str(item.get("text") or item.get("excerpt") or "")[:240],
            }
        )
    return slim


def _slim_tool(call: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": call.get("status"),
        "top_k": call.get("top_k"),
        "query": call.get("query"),
        "retrieval_policy": call.get("retrieval_policy"),
        "retrieval_edit": call.get("retrieval_edit"),
        "retrieval_policy_reason": call.get("retrieval_policy_reason"),
        "latency_ms": call.get("latency_ms"),
        "retrieval_unit_ids": call.get("retrieval_unit_ids") or [],
        "evidence_ids": call.get("evidence_ids") or [],
    }


def _last_record(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return json.loads(lines[-1])


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _report(
    policy: dict[str, Any],
    rows: list[dict[str, Any]],
    exposures: list[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    lines = [
        "# GongsiMate release smoke",
        "",
        f"HEAD `{policy['git_head']}`. 제품 검색 정책 `{policy['product_retrieval_policy']}`.",
        f"A0 디렉터리 해시 유지: {summary['a0_unchanged']}.",
        "",
        "## 시나리오",
        "",
        "| id | verdict | status | policy | top_k | llm | search | ms | usd |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {id} | {verdict} | {status} | {policy} | {top_k} | {llm} | {search} | {ms} | {usd} |".format(
                id=row["id"],
                verdict=row["verdict"],
                status=row["status"],
                policy=row["retrieval_policy"] or "",
                top_k=",".join(str(item) for item in row["top_k"]) or "-",
                llm=row["llm_call_count"],
                search=row["search_call_count"],
                ms=row["latency_ms"],
                usd=row["estimated_usd"],
            )
        )
    lines.extend(["", "## 사용자 노출", ""])
    for row in exposures:
        lines.append(
            f"- `{row['id']}` {row['verdict']}. status `{row['status']}`. "
            f"ui `{row['ui_label'] or row['ui_kind']}`. message: {row['user_message'][:180]}"
        )
    lines.extend(
        [
            "",
            "## Dense head 창",
            "",
            policy["dense_head_window"],
            "",
            f"관측 top_k: {summary['observed_top_k']}.",
            f"추정 비용 USD {summary['estimated_usd']}. 청구서가 아니다.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()

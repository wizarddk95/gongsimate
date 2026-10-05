"""Stage runner.

Sampling is not rewritten after it exists. Retrieval starts only after questions exist.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from disclosure_agent.agent.search import BgeM3QueryEncoder, _eligible_rows, load_dense_d2a_index
from disclosure_agent.agent.search_catalog import load_search_catalog
from disclosure_agent.bm25 import Bm25Index
from disclosure_agent.chunking import canonical_dumps, sha256_text
from disclosure_agent.cli import load_dotenv_files
from disclosure_agent.dense import exact_normalized_search
from disclosure_agent.errors import UsageError
from disclosure_agent.evalset_v2 import PROMPT_VERSION
from disclosure_agent.evalset_v2.corpus_sample import effective_receipts
from disclosure_agent.evalset_v2.generate import (
    COMPOUND_QUERY_TYPES,
    decompose_question,
    generate_slot,
    map_slots,
)
from disclosure_agent.evalset_v2.judge import agreement_summary, judge_candidate
from disclosure_agent.evalset_v2.llm import (
    CallLog,
    StructuredClient,
    assert_budget,
    planned_cost,
)
from disclosure_agent.evalset_v2.metrics import (
    aggregate_headline,
    breakdown,
    build_qrels,
    filter_headline,
    score_question,
    sufficiency,
)
from disclosure_agent.evalset_v2.pool import select_candidates
from disclosure_agent.evalset_v2.spec import (
    dataset_dir,
    load_run_config,
    load_spec,
    spec_sha256,
)
from disclosure_agent.evaluation import git_revision
from disclosure_agent.retrieval_baseline import TOP_K, _search_question

__all__ = ["sample_stage", "estimate_stage", "run_all"]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_dumps(payload) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sample_stage(root: Path) -> dict[str, Any]:
    """Freeze the sampling manifest. Refuses to overwrite an existing manifest."""
    from disclosure_agent.evalset_v2.corpus_sample import (
        build_sampling_manifest,
        load_company_units,
        manifest_is_retrieval_free,
    )

    spec = load_spec(root)
    directory = dataset_dir(root)
    manifest_path = directory / "sampling_manifest.json"
    if manifest_path.is_file():
        existing = read_json(manifest_path)
        if existing.get("spec_sha256") != spec_sha256(spec):
            raise UsageError("population spec changed after the sampling manifest was frozen")
        return existing
    catalog = load_search_catalog(root)
    units = {
        str(company["corp_code"]): load_company_units(catalog, spec, str(company["corp_code"]))
        for company in spec["companies"]
    }
    manifest = build_sampling_manifest(units, spec)
    if not manifest_is_retrieval_free(manifest):
        raise UsageError("sampling manifest contains retrieval fields")
    directory.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, manifest)
    write_json(directory / "corpus_support.json", manifest["support"])
    return manifest


def estimate_stage(root: Path) -> dict[str, Any]:
    spec = load_spec(root)
    config = load_run_config(root)
    manifest = _manifest(root, spec)
    counts = upper_call_counts(manifest, spec, config)
    models = {role: item["expected_model"] for role, item in spec["judges"].items()}
    plan = planned_cost(counts, models, config)
    assert_budget(plan, config)
    plan["call_counts"] = counts
    plan["note"] = (
        "Upper bound before pooling. Dollars use configured estimates, not an invoice. "
        "Judge calls are one candidate at a time."
    )
    return plan


def run_all(root: Path) -> dict[str, Any]:
    spec = load_spec(root)
    manifest = sample_stage(root)
    plan = estimate_stage(root)
    print(
        f"estimate calls={plan['calls']} usd={plan['estimated_usd']} "
        f"source={plan['price_source']}",
        flush=True,
    )
    questions = generate_stage(root, spec, manifest)
    requirements = decompose_stage(root, spec, manifest, questions)
    pools = pool_stage(root, spec, manifest, questions)
    _check_actual_judge_budget(root, spec, pools)
    judgments = judge_stage(root, spec, questions, requirements, pools)
    qrels = assemble_stage(root, spec, manifest, questions, requirements, judgments)
    frozen = freeze_stage(root, spec, manifest, questions, requirements, qrels, judgments)
    metrics = evaluate_stage(root, spec, questions, requirements, qrels, frozen)
    return {"freeze": frozen["dataset_sha256"], "headline": metrics["headline_test"]}


def generate_stage(
    root: Path, spec: dict[str, Any], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    path = dataset_dir(root) / "questions.json"
    if path.is_file():
        payload = read_json(path)
        if payload.get(
            "spec_sha256") == spec_sha256(
                spec) and len(payload.get("questions") or []) == len(manifest["slots"]):
            return payload["questions"]
    config = load_run_config(root)
    routes = _routes(spec)
    client = _client(routes["generator"], config)
    log = CallLog(dataset_dir(root) / "calls" / "generate.jsonl")
    slots = manifest["slots"]

    def work(slot: dict[str, Any]) -> dict[str, Any]:
        row = generate_slot(
            slot,
            log,
            client,
            max_retries=int(config["max_retries"]),
            reasoning_setting=str(config["reasoning_setting"]),
        )
        print(f"generate {row['query_id']}", flush=True)
        return row

    questions = map_slots(slots, work, workers=int(config["workers"]))
    write_json(
        path, {"spec_sha256": spec_sha256(
            spec), "prompt_version": PROMPT_VERSION, "questions": questions})
    return questions


def decompose_stage(
    root: Path,
    spec: dict[str, Any],
    manifest: dict[str, Any],
    questions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    path = dataset_dir(root) / "requirements.json"
    if path.is_file():
        payload = read_json(path)
        if payload.get("spec_sha256") == spec_sha256(spec) and payload.get("requirements"):
            return payload["requirements"]
    config = load_run_config(root)
    routes = _routes(spec)
    client = _client(routes["decomposer"], config)
    log = CallLog(dataset_dir(root) / "calls" / "decompose.jsonl")
    by_id = {slot["query_id"]: slot for slot in manifest["slots"]}
    rows: list[dict[str, Any]] = []
    for question in questions:
        slot = by_id[question["query_id"]]
        if question.get("draft_requirements") or question["query_type"] not in COMPOUND_QUERY_TYPES:
            rows.extend(
                decompose_question(
                    question,
                    slot,
                    log,
                    client,
                    max_retries=int(config["max_retries"]),
                    reasoning_setting=str(config["reasoning_setting"]),
                )
            )
            continue
        rows.extend(
            decompose_question(
                question,
                slot,
                log,
                client,
                max_retries=int(config["max_retries"]),
                reasoning_setting=str(config["reasoning_setting"]),
            )
        )
        print(f"decompose {question['query_id']}", flush=True)
    write_json(
        path,
        {"spec_sha256": spec_sha256(spec), "prompt_version": PROMPT_VERSION, "requirements": rows},
    )
    return rows


def pool_stage(
    root: Path,
    spec: dict[str, Any],
    manifest: dict[str, Any],
    questions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    path = dataset_dir(root) / "pools.json"
    if path.is_file():
        payload = read_json(path)
        if payload.get(
            "spec_sha256") == spec_sha256(
                spec) and len(payload.get("pools") or []) == len(questions):
            return payload["pools"]
    catalog = load_search_catalog(root)
    encoder = BgeM3QueryEncoder()
    by_company: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for question in questions:
        by_company[str(question["corp_code"])].append(question)
    pools: list[dict[str, Any]] = []
    for corp, group in by_company.items():
        corpus = catalog.corpus(corp)
        if corpus is None:
            raise UsageError(f"{corp} is not in the product catalog")
        receipts = tuple(sorted(effective_receipts(catalog, corp)))
        index = load_dense_d2a_index(corpus.binding)
        units, matrix, scoring = _eligible_rows(index, corpus.binding, receipts)
        by_id = {str(unit["retrieval_unit_id"]): _unit_view(unit) for unit in units}
        bm25 = Bm25Index(
            [
                {
                    "chunk_id": unit["retrieval_unit_id"],
                    "ordinal": int(unit.get("ordinal") or 0),
                    "text": unit["text"],
                    "retrieval_unit_id": unit["retrieval_unit_id"],
                }
                for unit in by_id.values()
            ]
        )
        depth = max(
            int(spec["pooling"]["bm25_depth"]), int(spec["pooling"]["unanswerable_bm25_depth"]))
        for question in group:
            vector = encoder.encode(question["question"])
            ranked = exact_normalized_search(matrix, vector, scoring)
            dense_ranked = [
                {
                    "retrieval_unit_id": str(hit.get("retrieval_unit_id") or hit["chunk_id"]),
                    "rank": int(hit["rank"]),
                    "score": float(hit["score"]),
                }
                for hit in ranked
            ]
            bm25_ranked = [
                {
                    "retrieval_unit_id": str(hit.get("retrieval_unit_id") or hit["chunk_id"]),
                    "rank": int(hit["rank"]),
                    "score": float(hit["score"]),
                }
                for hit in bm25.search(question["question"], depth)
            ]
            pools.append(
                select_candidates(
                    query_id=question["query_id"],
                    answerability=question["answerability"],
                    seed_unit_ids=list(question.get("seed_unit_ids") or []),
                    dense_ranked=dense_ranked,
                    bm25_ranked=bm25_ranked,
                    units_by_id=by_id,
                    pooling=spec["pooling"],
                )
            )
            print(
                f"pool {question['query_id']} candidates={len(
                    pools[-1]['candidates'])}", flush=True)
        del index, matrix
    write_json(
        path,
        {
            "spec_sha256": spec_sha256(spec),
            "retrievers": spec["pooling"]["retrievers"],
            "pools": pools,
        },
    )
    return pools


def judge_stage(
    root: Path,
    spec: dict[str, Any],
    questions: list[dict[str, Any]],
    requirements: list[dict[str, Any]],
    pools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    path = dataset_dir(root) / "judgments.jsonl"
    expected = sum(len(item["candidates"]) for item in pools)
    if path.is_file():
        existing = _read_jsonl(path)
        if len(existing) == expected and all(row.get("query_id") for row in existing):
            return existing
    config = load_run_config(root)
    routes = _routes(spec)
    judge_a = _client(routes["judge_a"], config)
    judge_b = _client(routes["judge_b"], config)
    adjudicator = _client(routes["adjudicator"], config)
    directory = dataset_dir(root) / "calls"
    log_a = CallLog(directory / "judge_a.jsonl")
    log_b = CallLog(directory / "judge_b.jsonl")
    log_c = CallLog(directory / "adjudicate.jsonl")
    requirements_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in requirements:
        requirements_by_query[item["query_id"]].append(item)
    questions_by_id = {item["query_id"]: item for item in questions}
    jobs = []
    for pool in pools:
        question = questions_by_id[pool["query_id"]]
        reqs = requirements_by_query[pool["query_id"]]
        for candidate in pool["candidates"]:
            jobs.append((question, reqs, candidate))

    def work(job: tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
        question, reqs, candidate = job
        row = judge_candidate(
            query_id=question["query_id"],
            question=question["question"],
            requirements=reqs,
            candidate=candidate,
            judge_a=judge_a,
            judge_b=judge_b,
            adjudicator=adjudicator,
            log_a=log_a,
            log_b=log_b,
            log_c=log_c,
            max_retries=int(config["max_retries"]),
            reasoning_setting=str(config["reasoning_setting"]),
        )
        print(f"judge {row['query_id']} {row['candidate_id']} {row['final_source']}", flush=True)
        return row

    rows = map_slots(jobs, work, workers=int(config["workers"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def assemble_stage(
    root: Path,
    spec: dict[str, Any],
    manifest: dict[str, Any],
    questions: list[dict[str, Any]],
    requirements: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    qrels = build_qrels(judgments, questions)
    agreement = agreement_summary(judgments)
    write_json(
        dataset_dir(root) / "qrels.json",
        {
            "spec_sha256": spec_sha256(spec),
            "manifest_slots": len(manifest["slots"]),
            "agreement": agreement,
            "qrels": qrels,
        },
    )
    return qrels


def freeze_stage(
    root: Path,
    spec: dict[str, Any],
    manifest: dict[str, Any],
    questions: list[dict[str, Any]],
    requirements: list[dict[str, Any]],
    qrels: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
) -> dict[str, Any]:
    public_questions = [
        {key: value for key, value in question.items() if key != "draft_requirements"}
        for question in questions
    ]
    body = {
        "dataset_id": spec["dataset_id"],
        "spec_sha256": spec_sha256(spec),
        "prompt_version": PROMPT_VERSION,
        "label_source": spec["label_source"],
        "human_review": False,
        "slot_ids": [slot["query_id"] for slot in manifest["slots"]],
        "seed_units": {
            slot["query_id"]: [seed["retrieval_unit_id"] for seed in slot["seeds"]]
            for slot in manifest["slots"]
        },
        "questions": public_questions,
        "requirements": requirements,
        "qrels": qrels,
    }
    digest = sha256_text(canonical_dumps(body))
    agreement = agreement_summary(judgments)
    usage = _usage(dataset_dir(root) / "calls")
    frozen = {
        "dataset_id": spec["dataset_id"],
        "dataset_sha256": digest,
        "spec_sha256": spec_sha256(spec),
        "prompt_version": PROMPT_VERSION,
        "label_source": "LLM-judged evaluation set",
        "human_review": False,
        "limitations": list(spec["limitations"]),
        "dev_test_rule": spec["split_rule"],
        "test_holdout": (
            "The test split is for a later frozen comparison. "
            "Do not tune a retriever by repeatedly inspecting this test split."
        ),
        "agreement": agreement,
        "usage": usage,
        "question_count": len(public_questions),
        "qrel_count": len(qrels),
    }
    directory = dataset_dir(root)
    write_json(directory / "freeze.json", frozen)
    write_json(directory / "questions_frozen.json", {"questions": public_questions})
    return frozen


def evaluate_stage(
    root: Path,
    spec: dict[str, Any],
    questions: list[dict[str, Any]],
    requirements: list[dict[str, Any]],
    qrels: list[dict[str, Any]],
    frozen: dict[str, Any],
) -> dict[str, Any]:
    """Run the unchanged product Dense-D2a path. This does not edit retrieval code."""
    catalog = load_search_catalog(root)
    from disclosure_agent.agent.search_catalog import CatalogDenseSearch

    search = CatalogDenseSearch(catalog)
    qrels_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in qrels:
        qrels_by_query[row["query_id"]].append(row)
    scored = []
    for question in questions:
        scope, hits = _search_question(
            {
                "question_id": question["query_id"],
                "corp_code": question["corp_code"],
                "text": question["question"],
                "filters": {"as_of": None, "period": None, "report_types": None},
            },
            catalog,
            search,
        )
        row = score_question(
            question,
            qrels_by_query.get(question["query_id"]) or [],
            requirements,
            hits,
        )
        for item, hit in zip(row["hits"], hits, strict=False):
            item["document_name"] = hit.get("document_name")
            item["filing_period"] = hit.get("filing_period")
            item["text"] = str(hit.get("text") or "")[:240]
        row["question"] = question["question"]
        row["search_scope"] = scope
        scored.append(row)
        print(f"eval {question['query_id']}", flush=True)
    headline_rows = filter_headline(questions, scored)
    dev_rows = [
        row
        for row in scored
        if row["answerability"] == "fully_answerable" and row["split"] == "dev"
    ]
    partial_rows = [row for row in scored if row["answerability"] == "partially_answerable"]
    revision = git_revision(root)
    payload = {
        "dataset_id": spec["dataset_id"],
        "dataset_sha256": frozen["dataset_sha256"],
        "label_source": "LLM-judged evaluation set",
        "human_review": False,
        "git_commit": revision["git_commit"],
        "git_dirty": revision["git_dirty"],
        "retrieval": {
            "implementation": "product_catalog_dense_d2a",
            "retriever_type": "dense",
            "top_k": TOP_K,
            "product_code_changed": False,
            "query": "original question text",
        },
        "metric_policy": spec["metric_policy"],
        "headline_test": aggregate_headline(headline_rows),
        "dev_fully_answerable": aggregate_headline(dev_rows),
        "all_fully_answerable": aggregate_headline(
            [row for row in scored if row["answerability"] == "fully_answerable"]
        ),
        "partial_not_in_headline": aggregate_headline(partial_rows) if partial_rows else None,
        "unanswerable_sufficiency": sufficiency(scored, qrels),
        "breakdowns_test": {
            "company": breakdown(headline_rows, "company_name"),
            "query_type": breakdown(headline_rows, "query_type"),
            "evidence_form": breakdown(headline_rows, "evidence_form"),
            "wording": breakdown(headline_rows, "wording_style"),
            "period_sensitive": breakdown(headline_rows, "period_sensitive"),
            "multi_evidence": breakdown(headline_rows, "multi_evidence"),
            "stratum": breakdown(headline_rows, "stratum_id"),
        },
        "agreement": frozen.get("agreement"),
        "usage": frozen.get("usage"),
        "examples": _examples(scored),
        "questions": scored,
    }
    output = root / "evaluation" / "runs" / "multi_company_retrieval_v2_dense_d2a"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "metrics.json", payload)
    write_json(output / "per_question.json", {"questions": scored})
    (output / "summary.md").write_text(_summary(payload), encoding="utf-8")
    return payload


def upper_call_counts(
    manifest: dict[str, Any],
    spec: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, int]:
    pooling = spec["pooling"]
    tail = sum(int(bucket["n"]) for bucket in pooling["tail_buckets"])
    full_cap = int(pooling["dense_depth"]) + int(pooling["bm25_depth"]) + tail + 2
    thin_cap = int(
        pooling["unanswerable_dense_depth"]) + int(pooling["unanswerable_bm25_depth"]) + 1
    generate = len(manifest["slots"])
    decompose = 0
    judge = 0
    for slot in manifest["slots"]:
        compound = slot["assigned_query_type"] in COMPOUND_QUERY_TYPES
        if slot["answerability"] == "fully_answerable" and compound:
            decompose += 1
        judge += thin_cap if slot["answerability"] == "unanswerable" else full_cap
    adjudicate = int(round(judge * float(config["assumed_adjudication_rate"])))
    return {
        "generate": generate,
        "decompose": decompose,
        "judge_a": judge,
        "judge_b": judge,
        "adjudicate": adjudicate,
    }


def _check_actual_judge_budget(
    root: Path, spec: dict[str, Any], pools: list[dict[str, Any]]) -> None:
    config = load_run_config(root)
    judge = sum(len(item["candidates"]) for item in pools)
    adjudicate = int(round(judge * float(config["assumed_adjudication_rate"])))
    models = {role: item["expected_model"] for role, item in spec["judges"].items()}
    plan = planned_cost(
        {
            "generate": 0,
            "decompose": 0,
            "judge_a": judge,
            "judge_b": judge,
            "adjudicate": adjudicate,
        },
        models,
        config,
    )
    assert_budget(plan, config)


def _manifest(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    path = dataset_dir(root) / "sampling_manifest.json"
    if not path.is_file():
        raise UsageError("sampling manifest is missing; run the sample stage first")
    manifest = read_json(path)
    if manifest.get("spec_sha256") != spec_sha256(spec):
        raise UsageError("sampling manifest does not match the population spec")
    return manifest


def _routes(spec: dict[str, Any]) -> dict[str, dict[str, str]]:
    load_dotenv_files()
    routes = {}
    for role, item in spec["judges"].items():
        actual = os.environ.get(str(item["model_env"]), "").strip()
        expected = str(item["expected_model"])
        if actual != expected:
            raise UsageError(
                f"{item['model_env']} must be {expected} for this frozen spec, got {actual!r}")
        key_name = "GEMINI_API_KEY" if item["provider"] == "gemini" else "OPENAI_API_KEY"
        routes[role] = {
            "provider": str(item["provider"]),
            "model": actual,
            "api_key": os.environ.get(key_name, "").strip(),
        }
    return routes


def _client(route: dict[str, str], config: dict[str, Any]) -> StructuredClient:
    timeout = float(
        config.get("timeout_seconds") or os.environ.get("OPENAI_TIMEOUT_SECONDS", "90") or "90")
    return StructuredClient(
        provider=route["provider"],
        model=route["model"],
        api_key=route["api_key"],
        timeout=timeout,
    )


def _unit_view(unit: dict[str, Any]) -> dict[str, Any]:
    return {
        "retrieval_unit_id": str(unit["retrieval_unit_id"]),
        "element_ids": [str(item) for item in unit.get("source_element_ids") or []],
        "text": str(unit.get("text") or ""),
        "document_name": str(unit.get("document_name") or ""),
        "filing_period": str(unit.get("filing_period") or ""),
        "section_path": [str(item) for item in unit.get("section_path") or []],
        "receipt_no": str(unit.get("receipt_no") or ""),
        "receipt_date": str(unit.get("receipt_date") or ""),
        "ordinal": int(unit.get("canonical_ordinal") or 0),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _usage(directory: Path) -> dict[str, Any]:
    totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "errors": 0})
    if not directory.is_dir():
        return {}
    for path in sorted(directory.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            bucket = totals[path.stem]
            if row.get("status") == "ok":
                bucket["calls"] += 1
                usage = row.get("usage") or {}
                bucket["input_tokens"] += int(usage.get("input_tokens") or 0)
                bucket["output_tokens"] += int(usage.get("output_tokens") or 0)
            else:
                bucket["errors"] += 1
    return dict(totals)


def _examples(scored: list[dict[str, Any]]) -> dict[str, Any]:
    test = [
        row
        for row in scored
        if row["split"] == "test" and row["answerability"] == "fully_answerable"
    ]

    def first(predicate: Any) -> dict[str, Any] | None:
        for row in test:
            if predicate(row):
                return {
                    "query_id": row["query_id"],
                    "question": row["question"],
                    "query_type": row["query_type"],
                    "company_name": row["company_name"],
                    "hit@10": row["at_k"]["10"]["hit"],
                    "mrr": row["mrr"],
                }
        return None

    return {
        "hit_at_1": first(lambda row: row["at_k"]["1"]["hit"]),
        "miss_at_10": first(lambda row: not row["at_k"]["10"]["hit"]),
        "multi_evidence": first(lambda row: row["multi_evidence"]),
        "unanswerable": next(
            (
                {
                    "query_id": row["query_id"],
                    "question": row["question"],
                    "company_name": row["company_name"],
                }
                for row in scored
                if row["answerability"] == "unanswerable"
            ),
            None,
        ),
    }


def _summary(payload: dict[str, Any]) -> str:
    headline = payload["headline_test"]
    hit10 = headline["at_k"]["10"]["hit"]
    lines = [
        "# multi_company_retrieval_v2 Dense-D2a",
        "",
        "LLM-judged evaluation set. 사람 검수는 없다. v1 31문항 결과와 같은 계열이 아니다.",
        "",
        f"- dataset_sha256: `{payload['dataset_sha256']}`",
        f"- test questions in headline: {headline['question_count']}",
        f"- excluded, no positive qrel: {headline['excluded_count']}",
        f"- Hit@10: {hit10['numerator']}/{hit10['denominator']}",
        f"- MRR: {headline['mrr']['value']}",
        f"- NDCG@10 condensed: {headline['at_k']['10']['ndcg_condensed']['value']}",
        (
            "- preferred_hit@10: "
            f"{headline['at_k']['10']['preferred_hit']['numerator']}/"
            f"{headline['at_k']['10']['preferred_hit']['denominator']}"
        ),
        f"- judged_coverage@10: {headline['at_k']['10']['judged_coverage']['value']}",
        f"- judged_coverage@20: {headline.get('judged_coverage@20', {}).get('value')}",
        "",
        "테스트 분할은 이후 비교용이다. 이 분할을 반복해서 보며 검색기를 고르지 않는다.",
        "",
    ]
    return "\n".join(lines)


def dataset_sha256(body: dict[str, Any]) -> str:
    return sha256_text(canonical_dumps(body))

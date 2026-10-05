"""Re-score the unchanged product Dense-D2a path on the frozen v2.1 qrels."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from disclosure_agent.agent.search_catalog import CatalogDenseSearch, load_search_catalog
from disclosure_agent.chunking import canonical_dumps
from disclosure_agent.evalset_v2_1 import DATASET_ID
from disclosure_agent.evalset_v2_1.freeze import dataset_dir, parent_dir
from disclosure_agent.evalset_v2_1.metrics import (
    aggregate_headline,
    breakdown,
    filter_headline,
    score_question,
    sufficiency,
)
from disclosure_agent.evaluation import git_revision
from disclosure_agent.retrieval_baseline import TOP_K, _search_question

RUN_DIR_NAME = "multi_company_retrieval_v2_1_dense_d2a"


def evaluate_dense(root: Path) -> dict[str, Any]:
    directory = dataset_dir(root)
    frozen = _read(directory / "freeze.json")
    questions = _read(directory / "questions.json")["questions"]
    requirements = _read(directory / "requirements.json")["requirements"]
    qrels = _read(directory / "qrels.json")["qrels"]
    v2_questions = _read(parent_dir(root) / "questions_frozen.json")["questions"]
    v2_split = {row["query_id"]: row.get("split") for row in v2_questions}
    catalog = load_search_catalog(root)
    search = CatalogDenseSearch(catalog)
    qrels_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in qrels:
        qrels_by_query[row["query_id"]].append(row)
    scored = []
    for question in questions:
        _scope, hits = _search_question(
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
        row["legacy_v2_split"] = v2_split.get(question["query_id"])
        scored.append(row)
        print(f"eval {question['query_id']}", flush=True)
    headline_rows = filter_headline(questions, scored)
    dev_rows = [
        row
        for row in scored
        if row["answerability"] == "fully_answerable" and row["split"] == "dev"
    ]
    partial_rows = [row for row in scored if row["answerability"] == "partially_answerable"]
    legacy_rows = [
        row
        for row in scored
        if row["answerability"] == "fully_answerable" and row["legacy_v2_split"] == "test"
    ]
    revision = git_revision(root)
    payload = {
        "dataset_id": DATASET_ID,
        "dataset_sha256": frozen["dataset_sha256"],
        "parent_dataset_sha256": frozen["parent_dataset_sha256"],
        "label_source": frozen["label_source"],
        "human_review": False,
        "git_commit": revision["git_commit"],
        "git_dirty": revision["git_dirty"],
        "retrieval": {
            "implementation": "product_catalog_dense_d2a",
            "retriever_type": "dense",
            "top_k": TOP_K,
            "product_code_changed": False,
            "query": "original question text",
            "comparison_to_v2": (
                "Same retriever. Differences are the split, the qrel unit, and the metric."
            ),
        },
        "metric_policy": frozen["metric_policy"],
        "headline_test": aggregate_headline(headline_rows),
        "dev_fully_answerable": aggregate_headline(dev_rows),
        "diagnostic_v2_test_ids_with_v21_metrics": {
            "role": (
                "Same questions as the v2 test split, scored with v2.1 unit metrics. "
                "Not a second benchmark and not a retrieval improvement."
            ),
            "metrics": aggregate_headline(legacy_rows),
        },
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
        "questions": scored,
    }
    output = root / "evaluation" / "runs" / RUN_DIR_NAME
    output.mkdir(parents=True, exist_ok=True)
    _write(output / "metrics.json", payload)
    _write(output / "per_question.json", {"questions": scored})
    (output / "summary.md").write_text(_summary(payload), encoding="utf-8")
    return payload


def _summary(payload: dict[str, Any]) -> str:
    headline = payload["headline_test"]
    hit1 = headline["at_k"]["1"]["hit"]
    hit5 = headline["at_k"]["5"]["hit"]
    hit10 = headline["at_k"]["10"]["hit"]
    recall5 = headline["recall@5_micro"]
    recall10 = headline["recall@10_micro"]
    lines = [
        "# multi_company_retrieval_v2_1 Dense-D2a",
        "",
        "같은 Dense-D2a를 v2.1 정의로 다시 잰 값이다. 검색기가 나아진 결과가 아니다.",
        "",
        f"- dataset_sha256: `{payload['dataset_sha256']}`",
        f"- test questions in headline: {headline['question_count']}",
        f"- excluded, no positive unit: {headline['excluded_count']}",
        f"- Hit@1: {hit1['numerator']}/{hit1['denominator']}",
        f"- Hit@5: {hit5['numerator']}/{hit5['denominator']}",
        f"- Hit@10: {hit10['numerator']}/{hit10['denominator']}",
        f"- Recall@5 micro: {recall5['numerator']}/{recall5['denominator']}",
        f"- Recall@10 micro: {recall10['numerator']}/{recall10['denominator']}",
        f"- Recall@5 macro: {headline['recall@5_macro']['value']}",
        f"- Recall@10 macro: {headline['recall@10_macro']['value']}",
        f"- MRR: {headline['mrr']['value']}",
        f"- NDCG@10 condensed: {headline['at_k']['10']['ndcg_condensed']['value']}",
        (
            "- NDCG@10 unjudged-as-zero: "
            f"{headline['at_k']['10']['ndcg_unjudged_as_zero']['value']}"
        ),
        (
            "- preferred_hit@10: "
            f"{headline['at_k']['10']['preferred_hit']['numerator']}/"
            f"{headline['at_k']['10']['preferred_hit']['denominator']}"
        ),
        f"- judged_coverage@10: {headline['at_k']['10']['judged_coverage']['value']}",
        f"- judged_coverage@20: {headline['judged_coverage@20']['value']}",
        "",
        "dev는 다음 검색 실험용이다. test로 가중치를 맞추지 않는다.",
        "",
    ]
    return "\n".join(lines)


def _read(path: Path) -> Any:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: Any) -> None:
    path.write_text(canonical_dumps(payload) + "\n", encoding="utf-8")

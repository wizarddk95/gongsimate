"""Load and check the frozen population specification.

Quotas are a list of strata, not a Cartesian product of every axis.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from disclosure_agent.chunking import canonical_dumps, sha256_text
from disclosure_agent.errors import UsageError

ANSWERABILITIES = frozenset(
    {"fully_answerable", "partially_answerable", "unanswerable"}
)
QUERY_TYPES = frozenset(
    {
        "single_fact",
        "single_numeric",
        "time_series_comparison",
        "composition_ratio",
        "causal_explanation",
        "table_lookup",
        "calculation",
        "multi_evidence",
        "cross_filing_comparison",
    }
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def spec_path(root: Path | None = None) -> Path:
    base = root or repo_root()
    return base / "evaluation" / "specs" / "retrieval_population_v2.json"


def run_config_path(root: Path | None = None) -> Path:
    base = root or repo_root()
    return base / "evaluation" / "specs" / "retrieval_evalset_v2_run.json"


def dataset_dir(root: Path | None = None) -> Path:
    base = root or repo_root()
    return base / "evaluation" / "datasets" / "multi_company_retrieval_v2"


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise UsageError(f"{path} must be an object")
    return payload


def load_spec(root: Path | None = None) -> dict[str, Any]:
    payload = load_json(spec_path(root))
    validate_spec(payload)
    return payload


def load_run_config(root: Path | None = None) -> dict[str, Any]:
    payload = load_json(run_config_path(root))
    if payload.get("max_calls", 0) < 1:
        raise UsageError("max_calls must be >= 1")
    if float(payload.get("max_estimated_usd", 0)) <= 0:
        raise UsageError("max_estimated_usd must be > 0")
    return payload


def spec_sha256(spec: dict[str, Any]) -> str:
    return sha256_text(canonical_dumps(spec))


def validate_spec(spec: dict[str, Any]) -> None:
    """Reject a Cartesian quota and a population claim the five companies cannot support."""
    if spec.get("dataset_id") != "multi_company_retrieval_v2":
        raise UsageError("dataset_id must be multi_company_retrieval_v2")
    if spec.get("frozen_before_retrieval") is not True:
        raise UsageError("population spec must be frozen before retrieval")
    companies = spec.get("companies")
    if not isinstance(companies, list) or len(companies) != 5:
        raise UsageError("spec is pinned to the five indexed companies")
    codes = [str(item.get("corp_code") or "") for item in companies]
    if len(set(codes)) != 5 or any(not re.fullmatch(r"\d{8}", code) for code in codes):
        raise UsageError("company corp_code values must be five distinct 8-digit codes")
    if any(
        item.get(
            "size_annotation") and "quota" not in str(
                item.get("size_annotation")) for item in companies):
        raise UsageError("size annotations must say they are not a quota")
    strata = spec.get("answerable_strata")
    if not isinstance(strata, list) or not strata:
        raise UsageError("answerable_strata is empty")
    ids: set[str] = set()
    per_company = 0
    topics: set[str] = set()
    forms: set[str] = set()
    types: set[str] = set()
    for item in strata:
        stratum_id = str(item.get("stratum_id") or "")
        if not stratum_id or stratum_id in ids:
            raise UsageError(f"duplicate or empty stratum_id {stratum_id!r}")
        ids.add(stratum_id)
        n = int(item.get("n_per_company") or 0)
        if n < 1:
            raise UsageError(f"{stratum_id} n_per_company must be >= 1")
        per_company += n
        topics.add(str(item.get("topic_bucket")))
        forms.add(str(item.get("evidence_form")))
        allowed = item.get("allowed_query_types")
        if not isinstance(allowed, list) or not allowed:
            raise UsageError(f"{stratum_id} needs allowed_query_types")
        unknown = set(allowed) - QUERY_TYPES
        if unknown:
            raise UsageError(f"{stratum_id} unknown query types {sorted(unknown)}")
        types.update(allowed)
    cartesian = len(topics) * len(forms) * len(types) * len(ANSWERABILITIES) * len(companies)
    if len(strata) >= cartesian:
        raise UsageError("answerable strata must not be the Cartesian product")
    partial = int(spec.get("partial_per_company") or 0)
    unanswerable_rows = spec.get("unanswerable_per_company")
    if partial < 1 or not isinstance(unanswerable_rows, list) or not unanswerable_rows:
        raise UsageError(
            "partial and unanswerable quotas are required and are not retrieval-headline items")
    unanswerable = sum(int(item.get("n") or 0) for item in unanswerable_rows)
    total = (per_company + partial + unanswerable) * len(companies)
    expected = spec.get("expected_counts") or {}
    if int(expected.get("total") or -1) != total:
        raise UsageError(
            f"expected_counts.total is {expected.get('total')} but quotas sum to {total}")
    if not 100 <= total <= 200:
        raise UsageError(f"total queries {total} are outside 100..200")
    if spec.get("human_review") is not False:
        raise UsageError("this dataset is LLM-judged and must not claim human review")
    if spec.get("label_source") != "LLM-judged evaluation set":
        raise UsageError("label_source must say LLM-judged evaluation set")
    policy = spec.get("metric_policy") or {}
    if "null" not in str(policy.get("unjudged")):
        raise UsageError("metric policy must keep unjudged as null")
    if "not used" not in str(policy.get("legacy_v1_unlabeled_as_zero")):
        raise UsageError("v1 unlabeled-as-zero must not be the v2 headline")

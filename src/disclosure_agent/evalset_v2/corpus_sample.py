"""Stratified evidence seeds drawn from indexed units before any retriever runs."""

from __future__ import annotations

import random
import re
from collections import defaultdict
from typing import Any

from disclosure_agent.agent.search_catalog import SearchCatalog
from disclosure_agent.agent.tool_args import resolve_search_call
from disclosure_agent.chunking import sha256_text
from disclosure_agent.errors import UsageError

_CAUSE_RE = re.compile(r"때문에|사유|이유는|기인|영향으로|증가 이유|감소 이유")
_YEAR_RE = re.compile(r"20\d{2}")


def body_text(text: str) -> str:
    raw = text or ""
    if raw.startswith("[section]"):
        parts = raw.split("\n", 1)
        return parts[1] if len(parts) == 2 else ""
    return raw


def topic_bucket(section_path: list[str], rules: list[dict[str, Any]]) -> str | None:
    joined = " / ".join(section_path)
    for rule in rules:
        if re.search(str(rule["pattern"]), joined):
            return str(rule["bucket"])
    return None


def surface_form(body: str) -> str:
    return "table" if " | " in body else "paragraph"


def prefix_hash(body: str, n_chars: int) -> str:
    compact = re.sub(r"\s+", " ", body).strip().lower()
    return sha256_text(compact[:n_chars])


def effective_receipts(catalog: SearchCatalog, corp_code: str) -> set[str]:
    corpus = catalog.corpus(corp_code)
    if corpus is None:
        raise UsageError(f"{corp_code} is not in the product catalog")
    tool = resolve_search_call(
        {
            "query": "population scope",
            "company_id": corp_code,
            "period": None,
            "filing_date_from": None,
            "filing_date_to": None,
            "report_types": None,
            "as_of": None,
            "top_k": 20,
        },
        company_id=corp_code,
        filings=corpus.filings,
        corpus_as_of=corpus.corpus_as_of,
        max_top_k=20,
        relations=catalog.relations_for(corp_code),
    )
    if tool.status != "ok" or tool.filings is None:
        raise UsageError(f"{corp_code} filing scope failed: {tool.status} {tool.message}")
    return {item.receipt_no for item in tool.filings}


def load_company_units(
    catalog: SearchCatalog,
    spec: dict[str, Any],
    corp_code: str,
) -> list[dict[str, Any]]:
    """Read retrieval-unit text. Does not load embeddings or rank a query."""
    corpus = catalog.corpus(corp_code)
    if corpus is None:
        raise UsageError(f"{corp_code} is not in the product catalog")
    allowed = effective_receipts(catalog, corp_code)
    rules = list(spec["topic_rules"])
    text_filter = spec["text_filter"]
    minimum = int(text_filter["min_body_chars"])
    maximum = int(text_filter["max_body_chars"])
    prefix_n = int(text_filter["dedupe_prefix_chars"])
    company = next(item for item in spec["companies"] if item["corp_code"] == corp_code)
    rows: list[dict[str, Any]] = []
    with corpus.binding.units_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            unit = json_loads(line)
            receipt = str(unit.get("receipt_no") or "")
            if receipt not in allowed:
                continue
            body = body_text(str(unit.get("text") or ""))
            if not minimum <= len(body) <= maximum:
                continue
            section = [str(item) for item in unit.get("section_path") or []]
            bucket = topic_bucket(section, rules)
            if bucket is None:
                continue
            element_ids = [str(item) for item in unit.get("source_element_ids") or []]
            if not element_ids:
                continue
            period = str(unit.get("filing_period") or "")
            rows.append(
                {
                    "corp_code": corp_code,
                    "company_name": corpus.binding.company_name,
                    "user_facing_name": company["user_facing_name"],
                    "retrieval_unit_id": str(unit["retrieval_unit_id"]),
                    "element_ids": element_ids,
                    "text": str(unit.get("text") or ""),
                    "body": body,
                    "document_name": str(unit.get("document_name") or ""),
                    "report_name": str(unit.get("report_name") or ""),
                    "filing_period": period,
                    "period_end_year": unit.get("period_end_year"),
                    "receipt_no": receipt,
                    "receipt_date": str(unit.get("receipt_date") or ""),
                    "section_path": section,
                    "topic_bucket": bucket,
                    "surface_form": surface_form(body),
                    "n_elements": len(element_ids),
                    "prefix_hash": prefix_hash(body, prefix_n),
                    "has_cause_language": bool(_CAUSE_RE.search(body)),
                }
            )
    return rows


def json_loads(line: str) -> dict[str, Any]:
    import json

    payload = json.loads(line)
    if not isinstance(payload, dict):
        raise UsageError("retrieval unit row is not an object")
    return payload


def support_counts(units: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    periods: dict[str, set[str]] = defaultdict(set)
    for unit in units:
        key = f"{unit['topic_bucket']}|{unit['surface_form']}"
        counts[key] += 1
        if unit["n_elements"] >= 2 and unit["topic_bucket"] == "business":
            counts["business|multi_element"] += 1
        periods[unit["topic_bucket"]].add(unit["filing_period"])
    for topic, values in periods.items():
        counts[f"{topic}|periods"] = len(values)
    return dict(sorted(counts.items()))


def assert_absent_year(units: list[dict[str, Any]], year: int) -> None:
    for unit in units:
        period = str(unit.get("filing_period") or "")
        if period.startswith(str(year)):
            raise UsageError(f"absent_year {year} is present in filing_period {period}")


def build_sampling_manifest(
    units_by_company: dict[str, list[dict[str, Any]]],
    spec: dict[str, Any],
) -> dict[str, Any]:
    """Allocate seeds. The random stream is seeded from the spec and does not see rankings."""
    from disclosure_agent.evalset_v2.spec import spec_sha256

    rng = random.Random(int(spec["random_seed"]))
    slots: list[dict[str, Any]] = []
    support: dict[str, Any] = {}
    for company in spec["companies"]:
        corp = str(company["corp_code"])
        units = units_by_company.get(corp) or []
        assert_absent_year(units, int(spec["absent_year"]))
        support[corp] = {
            "user_facing_name": company["user_facing_name"],
            "eligible_units": len(units),
            "counts": support_counts(units),
        }
        used: set[str] = set()
        for stratum in spec["answerable_strata"]:
            picked = _sample_stratum(units, stratum, used, rng)
            _assign(slots, picked, stratum, spec, company)
        partial_units = _take_tables(units, used, int(spec["partial_per_company"]), rng)
        for index, unit in enumerate(partial_units, start=1):
            slots.append(
                _slot(
                    company,
                    stratum_id="partial",
                    answerability="partially_answerable",
                    evidence_form="mixed",
                    query_type="time_series_comparison",
                    wording_style="layperson",
                    seeds=[_seed(unit)],
                    reason_code="period_out_of_corpus",
                    index=index,
                    spec=spec,
                )
            )
        causal_pool = [
            unit
            for unit in units
            if unit["surface_form"] == "table"
            and not unit["has_cause_language"]
            and unit["prefix_hash"] not in used
        ]
        causal = _round_robin(causal_pool, 1, rng)
        if len(causal) != 1:
            raise UsageError(
                f"{corp} has no table without cause language for an unanswerable question")
        used.add(causal[0]["prefix_hash"])
        slots.append(
            _slot(
                company,
                stratum_id="unanswerable_causal",
                answerability="unanswerable",
                evidence_form="table",
                query_type="causal_explanation",
                wording_style="layperson",
                seeds=[_seed(causal[0])],
                reason_code="causal_not_in_source",
                index=1,
                spec=spec,
            )
        )
        slots.append(
            _slot(
                company,
                stratum_id="unanswerable_period",
                answerability="unanswerable",
                evidence_form="paragraph",
                query_type="single_numeric",
                wording_style="layperson",
                seeds=[],
                reason_code="period_out_of_corpus",
                index=1,
                spec=spec,
            )
        )
    _check_totals(slots, spec)
    split_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for slot in slots:
        split_counts[str(slot["answerability"])][str(slot["split"])] += 1
    return {
        "dataset_id": spec["dataset_id"],
        "spec_sha256": spec_sha256(spec),
        "random_seed": spec["random_seed"],
        "split_rule": spec["split_rule"],
        "realized_splits": {key: dict(value) for key, value in split_counts.items()},
        "built_from": "effective filing units only; no retriever ranks",
        "support": support,
        "slots": slots,
    }


def _sample_stratum(
    units: list[dict[str, Any]],
    stratum: dict[str, Any],
    used: set[str],
    rng: random.Random,
) -> list[dict[str, Any]]:
    n = int(stratum["n_per_company"])
    if stratum.get("pair_periods"):
        pairs = _period_pairs(units, stratum, used, n, rng)
        for left, right in pairs:
            used.add(left["prefix_hash"])
            used.add(right["prefix_hash"])
        return pairs
    pool = _stratum_pool(units, stratum, used)
    picked = _round_robin(pool, n, rng)
    if len(picked) != n:
        raise UsageError(
            f"{stratum['stratum_id']} needed {n} units and found {len(picked)} "
            f"in topic {stratum['topic_bucket']}"
        )
    for unit in picked:
        used.add(unit["prefix_hash"])
    return picked


def _stratum_pool(
    units: list[dict[str, Any]],
    stratum: dict[str, Any],
    used: set[str],
) -> list[dict[str, Any]]:
    minimum = int(stratum.get("min_elements") or 1)
    form = str(stratum["evidence_form"])
    topic = str(stratum["topic_bucket"])
    pool = []
    for unit in units:
        if unit["topic_bucket"] != topic or unit["prefix_hash"] in used:
            continue
        if form == "multi_element":
            if unit["n_elements"] < minimum:
                continue
        elif unit["surface_form"] != form:
            continue
        if form == "paragraph" and unit["n_elements"] > 3:
            continue
        pool.append(unit)
    return pool


def _period_pairs(
    units: list[dict[str, Any]],
    stratum: dict[str, Any],
    used: set[str],
    n: int,
    rng: random.Random,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    topic = str(stratum["topic_bucket"])
    by_period: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        if unit["topic_bucket"] != topic or unit["surface_form"] != "table":
            continue
        if unit["prefix_hash"] in used:
            continue
        by_period[unit["filing_period"]].append(unit)
    periods = sorted(period for period, rows in by_period.items() if rows)
    if len(periods) < 2:
        raise UsageError(f"{topic} does not have two filing periods for cross_period")
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    cursor = 0
    while len(pairs) < n:
        if cursor + 1 >= len(periods):
            break
        left_period = periods[cursor]
        right_period = periods[cursor + 1]
        lefts = _round_robin(by_period[left_period], 1, rng)
        rights = _round_robin(by_period[right_period], 1, rng)
        if not lefts or not rights:
            break
        pairs.append((lefts[0], rights[0]))
        left_hash = lefts[0]["prefix_hash"]
        right_hash = rights[0]["prefix_hash"]
        by_period[left_period] = [
            unit for unit in by_period[left_period] if unit["prefix_hash"] != left_hash
        ]
        by_period[right_period] = [
            unit for unit in by_period[right_period] if unit["prefix_hash"] != right_hash
        ]
        cursor += 2
        if cursor + 1 >= len(periods):
            cursor = 0
    if len(pairs) != n:
        raise UsageError(f"cross_period needed {n} pairs and built {len(pairs)}")
    return pairs


def _take_tables(
    units: list[dict[str, Any]],
    used: set[str],
    n: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    pool = [
        unit
        for unit in units
        if unit["surface_form"] == "table" and unit["prefix_hash"] not in used
    ]
    picked = _round_robin(pool, n, rng)
    if len(picked) != n:
        raise UsageError(f"partial questions needed {n} unused tables and found {len(picked)}")
    for unit in picked:
        used.add(unit["prefix_hash"])
    return picked


def _round_robin(
    pool: list[dict[str, Any]],
    n: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Spread seeds across document type and period. Shuffle only inside a group."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for unit in pool:
        groups[(unit["document_name"], unit["filing_period"])].append(unit)
    for key in groups:
        groups[key].sort(key=lambda item: item["retrieval_unit_id"])
        rng.shuffle(groups[key])
    keys = sorted(groups)
    picked: list[dict[str, Any]] = []
    while len(picked) < n and any(groups[key] for key in keys):
        for key in keys:
            if not groups[key]:
                continue
            picked.append(groups[key].pop())
            if len(picked) == n:
                break
    return picked


def _assign(
    slots: list[dict[str, Any]],
    picked: list[Any],
    stratum: dict[str, Any],
    spec: dict[str, Any],
    company: dict[str, Any],
) -> None:
    cycle = list(spec["wording_cycle"])
    allowed = list(stratum["allowed_query_types"])
    for index, item in enumerate(picked, start=1):
        seeds = item if isinstance(item, tuple) else (item,)
        if stratum.get("pair_periods"):
            seeds = item
        query_type = allowed[(index - 1) % len(allowed)]
        slots.append(
            _slot(
                company,
                stratum_id=str(stratum["stratum_id"]),
                answerability="fully_answerable",
                evidence_form=str(stratum["evidence_form"]),
                query_type=query_type,
                wording_style=cycle[(index - 1) % len(cycle)],
                seeds=[_seed(unit) for unit in seeds],
                reason_code=None,
                index=index,
                spec=spec,
            )
        )


def _seed(unit: dict[str, Any]) -> dict[str, Any]:
    return {
        "retrieval_unit_id": unit["retrieval_unit_id"],
        "element_ids": list(unit["element_ids"]),
        "text": unit["text"],
        "document_name": unit["document_name"],
        "report_name": unit["report_name"],
        "filing_period": unit["filing_period"],
        "receipt_no": unit["receipt_no"],
        "receipt_date": unit["receipt_date"],
        "section_path": list(unit["section_path"]),
        "topic_bucket": unit["topic_bucket"],
        "surface_form": unit["surface_form"],
    }


def _slot(
    company: dict[str, Any],
    *,
    stratum_id: str,
    answerability: str,
    evidence_form: str,
    query_type: str,
    wording_style: str,
    seeds: list[dict[str, Any]],
    reason_code: str | None,
    index: int,
    spec: dict[str, Any],
) -> dict[str, Any]:
    corp = str(company["corp_code"])
    query_id = f"V2-{corp}-{stratum_id}-{index:02d}"
    test_n = int(round(int(_stratum_n(spec, stratum_id)) * (1.0 - float(spec["dev_fraction"]))))
    # index is 1-based inside the company stratum. test takes the first test_n slots.
    split = "test" if index <= max(test_n, 1) else "dev"
    if int(_stratum_n(spec, stratum_id)) == 1:
        split = "test"
    return {
        "query_id": query_id,
        "corp_code": corp,
        "company_name": company["user_facing_name"],
        "stratum_id": stratum_id,
        "answerability": answerability,
        "evidence_form": evidence_form,
        "assigned_query_type": query_type,
        "wording_style": wording_style,
        "split": split,
        "reason_code": reason_code,
        "absent_year": spec["absent_year"] if reason_code == "period_out_of_corpus" else None,
        "seeds": seeds,
    }


def _stratum_n(spec: dict[str, Any], stratum_id: str) -> int:
    for item in spec["answerable_strata"]:
        if item["stratum_id"] == stratum_id:
            return int(item["n_per_company"])
    if stratum_id == "partial":
        return int(spec["partial_per_company"])
    return 1


def _check_totals(slots: list[dict[str, Any]], spec: dict[str, Any]) -> None:
    counts = {
        "fully_answerable": 0,
        "partially_answerable": 0,
        "unanswerable": 0,
    }
    ids: set[str] = set()
    for slot in slots:
        if slot["query_id"] in ids:
            raise UsageError(f"duplicate query_id {slot['query_id']}")
        ids.add(slot["query_id"])
        counts[str(slot["answerability"])] += 1
    expected = spec["expected_counts"]
    if counts["fully_answerable"] != int(expected["answerable"]):
        raise UsageError(f"answerable count {counts['fully_answerable']}")
    if counts["partially_answerable"] != int(expected["partially_answerable"]):
        raise UsageError("partial count drifted")
    if counts["unanswerable"] != int(expected["unanswerable"]):
        raise UsageError("unanswerable count drifted")
    if _YEAR_RE.search("2018") and int(spec["absent_year"]) != 2018:
        raise UsageError("absent year check failed")


def manifest_is_retrieval_free(manifest: dict[str, Any]) -> bool:
    """Sampling output must not carry retriever names, ranks, or scores."""
    banned = {"rank", "score", "retriever", "bm25", "dense"}
    for slot in manifest.get("slots") or []:
        for key in slot:
            if key in banned:
                return False
        for seed in slot.get("seeds") or []:
            if banned & set(seed):
                return False
    return True

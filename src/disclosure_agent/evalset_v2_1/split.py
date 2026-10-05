"""Deterministic stratified dev/test split for the frozen v2 questions.

The assignment reads question and requirement annotations only.
Rank, score, hit, and every other retrieval result are not inputs.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from typing import Any

from disclosure_agent.errors import UsageError

SPLIT_SEED = 20261005
TEST_SIZE = 55
DEV_SIZE = 45

# Marginal targets. This is not a Cartesian product of the axes.
PRIMARY_FIELDS = ("wording_style", "query_type", "evidence_form")
EXACT_FIELDS = ("wording_style", "query_type")
REPAIR_FIELDS = ("stratum_id", "period_sensitive", "multi_evidence", "evidence_form")
SECONDARY_FIELDS = ("stratum_id", "period_sensitive", "multi_evidence")
BALANCE_FIELDS = PRIMARY_FIELDS + SECONDARY_FIELDS
FIELD_WEIGHT = {
    "wording_style": 4.0,
    "query_type": 4.0,
    "evidence_form": 2.0,
    "stratum_id": 1.0,
    "period_sensitive": 1.0,
    "multi_evidence": 1.0,
}
ANNOTATION_FIELDS = (
    "query_id",
    "corp_code",
    "wording_style",
    "query_type",
    "evidence_form",
    "stratum_id",
    "period_sensitive",
)


def assign_fully_answerable_splits(
    questions: list[dict[str, Any]],
    requirements: list[dict[str, Any]],
    *,
    test_size: int = TEST_SIZE,
    seed: int = SPLIT_SEED,
) -> dict[str, str]:
    """Return query_id -> ``dev`` or ``test`` for fully answerable questions.

    Company counts are a hard cap: with 20 questions per company and a 55/45
    global target, each company contributes 11 test and 9 dev questions.
    Other axes are marginal Hamilton targets. The smallest query types are
    placed first so a category of size at least 2 is not left in one split
    by accident. The seed breaks exact score ties and nothing else.
    """
    rows = [_annotation(question, _multi_evidence(requirements)) for question in questions]
    dev_size = len(rows) - test_size
    if test_size <= 0 or dev_size <= 0:
        raise UsageError("split sizes must both be positive")
    if len({row["query_id"] for row in rows}) != len(rows):
        raise UsageError("duplicate query_id in split input")
    targets = _split_targets(rows, test_size)
    company_test = _hamilton(_counts(rows, "corp_code"), test_size)
    company_dev = {
        key: _counts(rows, "corp_code")[key] - company_test[key] for key in company_test
    }
    order = sorted(
        rows,
        key=lambda row: (
            _counts(rows, "query_type")[row["query_type"]],
            _counts(rows, "wording_style")[row["wording_style"]],
            row["query_id"],
        ),
    )
    assigned: dict[str, str] = {}
    used = {split: {field: Counter() for field in BALANCE_FIELDS} for split in ("test", "dev")}
    company_used: dict[str, Counter[str]] = defaultdict(Counter)
    for row in order:
        split = _choose_split(
            row,
            targets=targets,
            used=used,
            company_test=company_test,
            company_dev=company_dev,
            company_used=company_used,
            seed=seed,
        )
        assigned[row["query_id"]] = split
        company_used[row["corp_code"]][split] += 1
        for field in BALANCE_FIELDS:
            used[split][field][row[field]] += 1
    _repair_primary(rows, assigned, targets)
    _repair_secondary(rows, assigned, targets)
    _validate(rows, assigned, targets, company_test, test_size)
    return assigned


def _annotation(question: dict[str, Any], multi: set[str]) -> dict[str, str]:
    if question.get("answerability") != "fully_answerable":
        raise UsageError("split input must be fully_answerable questions")
    missing = [field for field in ANNOTATION_FIELDS if field not in question]
    if missing:
        raise UsageError(f"{question.get('query_id')} is missing {missing}")
    query_id = str(question["query_id"])
    return {
        "query_id": query_id,
        "corp_code": str(question["corp_code"]),
        "wording_style": str(question["wording_style"]),
        "query_type": str(question["query_type"]),
        "evidence_form": str(question["evidence_form"]),
        "stratum_id": str(question["stratum_id"]),
        "period_sensitive": str(bool(question["period_sensitive"])),
        "multi_evidence": "true" if query_id in multi else "false",
    }


def _multi_evidence(requirements: list[dict[str, Any]]) -> set[str]:
    counts: Counter[str] = Counter()
    for row in requirements:
        if row.get("answerability") == "fully_answerable":
            counts[str(row["query_id"])] += 1
    return {query_id for query_id, count in counts.items() if count > 1}


def _counts(rows: list[dict[str, str]], field: str) -> dict[str, int]:
    return dict(Counter(row[field] for row in rows))


def _split_targets(
    rows: list[dict[str, str]], test_size: int
) -> dict[str, dict[str, dict[str, int]]]:
    targets: dict[str, dict[str, dict[str, int]]] = {"test": {}, "dev": {}}
    for field in BALANCE_FIELDS:
        counts = _counts(rows, field)
        test_counts = _hamilton(counts, test_size)
        targets["test"][field] = test_counts
        targets["dev"][field] = {
            key: counts[key] - test_counts[key] for key in counts
        }
    return targets


def _hamilton(counts: dict[str, int], total: int) -> dict[str, int]:
    """Largest-remainder apportionment. Ties break by category name."""
    population = sum(counts.values())
    if population <= 0:
        raise UsageError("cannot apportion an empty population")
    floors: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for key, count in counts.items():
        exact = count * total / population
        floors[key] = int(exact)
        remainders.append((exact - floors[key], key))
    short = total - sum(floors.values())
    remainders.sort(key=lambda item: (-item[0], item[1]))
    for _, key in remainders[:short]:
        floors[key] += 1
    return floors


def _choose_split(
    row: dict[str, str],
    *,
    targets: dict[str, dict[str, int]],
    used: dict[str, dict[str, Counter[str]]],
    company_test: dict[str, int],
    company_dev: dict[str, int],
    company_used: dict[str, Counter[str]],
    seed: int,
) -> str:
    caps = {
        "test": company_test[row["corp_code"]],
        "dev": company_dev[row["corp_code"]],
    }
    feasible = [
        split for split in ("test", "dev") if company_used[row["corp_code"]][split] < caps[split]
    ]
    if not feasible:
        raise UsageError(f"no split capacity left for {row['query_id']}")

    def rank(split: str) -> tuple[float, float, str]:
        overflow = 0.0
        pressure = 0.0
        for field in BALANCE_FIELDS:
            weight = FIELD_WEIGHT[field]
            value = row[field]
            target = targets[split][field].get(value, 0)
            current = used[split][field][value]
            if current >= target:
                overflow += weight * (1 + current - target)
            elif target > 0:
                pressure += weight * (current / target)
        return (overflow, pressure, _tie(seed, row["query_id"], split))

    return min(feasible, key=rank)


def _tie(seed: int, query_id: str, split: str) -> str:
    body = f"{seed}|{query_id}|{split}".encode()
    return hashlib.sha256(body).hexdigest()


def _primary_gap(
    rows: list[dict[str, str]],
    assigned: dict[str, str],
    targets: dict[str, dict[str, dict[str, int]]],
) -> str:
    parts = []
    for field in EXACT_FIELDS:
        for value, target in sorted(targets["test"][field].items()):
            actual = sum(
                1
                for row in rows
                if row[field] == value and assigned[row["query_id"]] == "test"
            )
            if actual != target:
                parts.append(f"{field}={value} test {actual}/{target}")
    return ", ".join(parts)


def _require_both_sides(
    rows: list[dict[str, str]], assigned: dict[str, str], field: str
) -> None:
    for value in {row[field] for row in rows}:
        total = sum(1 for row in rows if row[field] == value)
        if total < 2:
            continue
        counts = Counter(assigned[row["query_id"]] for row in rows if row[field] == value)
        if counts["test"] == 0 or counts["dev"] == 0:
            raise UsageError(f"{field}={value} is missing from one split")


def _require_target(
    rows: list[dict[str, str]],
    assigned: dict[str, str],
    targets: dict[str, dict[str, dict[str, int]]],
    field: str,
) -> None:
    _require_both_sides(rows, assigned, field)
    for value, target in targets["test"][field].items():
        actual = sum(
            1 for row in rows if row[field] == value and assigned[row["query_id"]] == "test"
        )
        if actual != target:
            raise UsageError(f"{field}={value} test {actual} != target {target}")


def _primary_l1(
    rows: list[dict[str, str]],
    assigned: dict[str, str],
    targets: dict[str, dict[str, dict[str, int]]],
) -> int:
    distance = 0
    for field in EXACT_FIELDS:
        for value, target in targets["test"][field].items():
            actual = sum(
                1
                for row in rows
                if row[field] == value and assigned[row["query_id"]] == "test"
            )
            distance += abs(actual - target)
    return distance


def _repair_primary(
    rows: list[dict[str, str]],
    assigned: dict[str, str],
    targets: dict[str, dict[str, dict[str, int]]],
) -> None:
    """Swap within a company until primary marginals match. No retrieval fields."""
    by_company: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_company[row["corp_code"]].append(row)
    for _ in range(len(rows)):
        if _primary_l1(rows, assigned, targets) == 0:
            return
        best: tuple[int, str, str] | None = None
        for group in by_company.values():
            for left in group:
                for right in group:
                    if assigned[left["query_id"]] == assigned[right["query_id"]]:
                        continue
                    if left["query_id"] >= right["query_id"]:
                        continue
                    pair = (left["query_id"], right["query_id"])
                    assigned[pair[0]], assigned[pair[1]] = assigned[pair[1]], assigned[pair[0]]
                    distance = _primary_l1(rows, assigned, targets)
                    assigned[pair[0]], assigned[pair[1]] = assigned[pair[1]], assigned[pair[0]]
                    candidate = (distance, pair[0], pair[1])
                    if best is None or candidate < best:
                        best = candidate
        current = _primary_l1(rows, assigned, targets)
        if best is None or best[0] >= current:
            raise UsageError(
                "primary split targets cannot be met by a within-company swap: "
                + _primary_gap(rows, assigned, targets)
            )
        assigned[best[1]], assigned[best[2]] = assigned[best[2]], assigned[best[1]]
    if _primary_l1(rows, assigned, targets) != 0:
        raise UsageError("primary split targets were not met")


def _field_l1(
    rows: list[dict[str, str]],
    assigned: dict[str, str],
    targets: dict[str, dict[str, dict[str, int]]],
    fields: tuple[str, ...],
) -> int:
    distance = 0
    for field in fields:
        for value, target in targets["test"][field].items():
            actual = sum(
                1
                for row in rows
                if row[field] == value and assigned[row["query_id"]] == "test"
            )
            distance += abs(actual - target)
    return distance


def _repair_secondary(
    rows: list[dict[str, str]],
    assigned: dict[str, str],
    targets: dict[str, dict[str, dict[str, int]]],
) -> None:
    """Move section and period margins without touching wording or query type."""
    by_company: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_company[row["corp_code"]].append(row)
    exact = _field_l1(rows, assigned, targets, EXACT_FIELDS)
    for _ in range(len(rows)):
        current = _field_l1(rows, assigned, targets, REPAIR_FIELDS)
        if current == 0:
            return
        best: tuple[int, str, str] | None = None
        for group in by_company.values():
            for left in group:
                for right in group:
                    if assigned[left["query_id"]] == assigned[right["query_id"]]:
                        continue
                    if left["query_id"] >= right["query_id"]:
                        continue
                    pair = (left["query_id"], right["query_id"])
                    assigned[pair[0]], assigned[pair[1]] = assigned[pair[1]], assigned[pair[0]]
                    if _field_l1(rows, assigned, targets, EXACT_FIELDS) != exact:
                        assigned[pair[0]], assigned[pair[1]] = assigned[pair[1]], assigned[pair[0]]
                        continue
                    distance = _field_l1(rows, assigned, targets, REPAIR_FIELDS)
                    assigned[pair[0]], assigned[pair[1]] = assigned[pair[1]], assigned[pair[0]]
                    candidate = (distance, pair[0], pair[1])
                    if best is None or candidate < best:
                        best = candidate
        if best is None or best[0] >= current:
            return
        assigned[best[1]], assigned[best[2]] = assigned[best[2]], assigned[best[1]]


def _validate(
    rows: list[dict[str, str]],
    assigned: dict[str, str],
    targets: dict[str, dict[str, int]],
    company_test: dict[str, int],
    test_size: int,
) -> None:
    if sum(1 for split in assigned.values() if split == "test") != test_size:
        raise UsageError("test split did not land on the requested size")
    by_company: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_company[row["corp_code"]][assigned[row["query_id"]]] += 1
    for corp, target in company_test.items():
        if by_company[corp]["test"] != target:
            raise UsageError(f"{corp} test count {by_company[corp]['test']} != {target}")
    for field in ("wording_style", "query_type"):
        _require_target(rows, assigned, targets, field)
    _require_both_sides(rows, assigned, "evidence_form")

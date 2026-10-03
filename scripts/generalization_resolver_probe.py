# ruff: noqa: E501
"""Reproduce the CompanyResolver collision and record alias checks.

Uses the corp-code cache only. Does not call a model and does not change resolution.
"""

from __future__ import annotations

import json
from pathlib import Path

from disclosure_agent.agent.company import CompanyRegistry

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evaluation" / "generalization_smoke"
CACHE = ROOT / "data" / "cache"


def _space_name(name: str) -> str:
    if len(name) < 4:
        return name
    midpoint = len(name) // 2
    return f"{name[:midpoint]} {name[midpoint:]}"


def main() -> None:
    companies = json.loads((OUT / "companies.json").read_text(encoding="utf-8"))["companies"]
    indexed = {company["corp_code"] for company in companies}
    indexed.add("00107613")
    registry = CompanyRegistry.from_corp_cache(CACHE, indexed_corp_codes=indexed)
    collision_question = "이수페타시스 요즘 실적 어때?"
    mentions = registry.extract_mentions(collision_question)
    per_mention = []
    for mention in mentions:
        per_mention.append(_resolution(registry.resolve_mention(mention)))
    combined = registry.resolve_mentions(mentions)
    rows = [_case(registry, collision_question, note="prior E2E collision")]
    for company in companies:
        names = [
            ("legal_name", company["corp_name"]),
            ("stock_name", company["stock_name"]),
            ("compact", company["corp_name"].replace("(주)", "").replace("주식회사", "").strip()),
            ("spaced", _space_name(company["stock_name"])),
        ]
        if company.get("corp_name_eng"):
            names.append(("english", company["corp_name_eng"]))
        for kind, surface in names:
            question = f"{surface} 최근 실적은?"
            rows.append(_case(registry, question, note=kind, expected=company["corp_code"]))
        rows.append(
            _case(
                registry,
                f"{company['stock_name']} 요즘 실적 어때?",
                note="adverb collision with this company",
                expected=company["corp_code"],
            )
        )
    payload = {
        "indexed_corp_codes": sorted(indexed),
        "collision": {
            "question": collision_question,
            "mentions": mentions,
            "per_mention": per_mention,
            "combined": _resolution(combined),
        },
        "alias_cases": rows,
    }
    path = OUT / "resolver_after.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["collision"], ensure_ascii=False, indent=2))
    failed = [
        row for row in rows if row.get("expected") and row.get("corp_code") != row["expected"]
    ]
    print(f"alias_cases={len(rows)} unexpected={len(failed)}")
    for row in failed:
        print(row["question"], row["status"], row.get("company_name"), row.get("candidates"))


def _case(registry: CompanyRegistry, question: str, *, note: str, expected: str | None = None) -> dict[str, object]:
    mentions = registry.extract_mentions(question)
    resolution = registry.resolve_mentions(mentions)
    payload = _resolution(resolution)
    payload["question"] = question
    payload["note"] = note
    payload["mentions"] = mentions
    if expected:
        payload["expected"] = expected
    return payload


def _resolution(resolution: object) -> dict[str, object]:
    status = getattr(resolution, "status", None)
    return {
        "status": status,
        "corp_code": getattr(resolution, "corp_code", None),
        "company_name": getattr(resolution, "company_name", None),
        "mention": getattr(resolution, "mention", None),
        "candidate_total": getattr(resolution, "candidate_total", None),
        "candidates": [
            {"corp_code": company.corp_code, "corp_name": company.corp_name, "stock_code": company.stock_code}
            for company in getattr(resolution, "candidates", ())
        ],
    }


if __name__ == "__main__":
    main()

# ruff: noqa: E501
"""Parse the filings chosen by scripts/generalization_select.py.

Reads stored ZIPs only. Does not call OpenDART and does not change the parser.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from disclosure_agent.parsing import parse_receipt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evaluation" / "generalization_smoke"
RAW = ROOT / "data" / "raw"
PARSED = ROOT / "data" / "parsed"


def main() -> None:
    payload = json.loads((OUT / "companies.json").read_text(encoding="utf-8"))
    rows = []
    for company in payload["companies"]:
        for outcome in company["download_outcomes"]:
            if outcome.get("status") == "failed":
                rows.append(
                    {
                        "corp_code": company["corp_code"],
                        "corp_name": company["corp_name"],
                        "receipt_no": outcome["rcept_no"],
                        "report_nm": outcome.get("report_nm"),
                        "role": outcome.get("role"),
                        "downloaded": False,
                        "parse_status": "not_parsed",
                        "reason": outcome.get("error_type"),
                    }
                )
                continue
            print(f"parse {company['corp_name']} {outcome.get('report_nm')}", flush=True)
            result = parse_receipt(
                outcome["rcept_no"],
                RAW,
                PARSED,
                source_sha256=outcome.get("sha256"),
            )
            manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
            warnings = _warnings(manifest)
            rows.append(
                {
                    "stratum": company["stratum"],
                    "corp_code": company["corp_code"],
                    "corp_name": company["corp_name"],
                    "stock_code": company["stock_code"],
                    "receipt_no": outcome["rcept_no"],
                    "report_nm": outcome.get("report_nm"),
                    "role": outcome.get("role"),
                    "source_sha256": result.get("source_sha256"),
                    "downloaded": True,
                    "parse_status": manifest.get("overall_status"),
                    "search_eligible": manifest.get("overall_status") == "success",
                    "reused": result.get("reused"),
                    "element_count": manifest.get("element_count"),
                    "member_count": len(manifest.get("members") or []),
                    "warnings": warnings,
                    "warning_patterns": _patterns(warnings),
                    "members": [
                        {
                            "member_index": member.get("member_index"),
                            "member_name": member.get("member_name"),
                            "document_name": member.get("document_name"),
                            "status": member.get("status"),
                            "warnings": member.get("warnings") or [],
                        }
                        for member in manifest.get("members") or []
                        if isinstance(member, dict)
                    ],
                    "manifest": result.get("manifest"),
                }
            )
            print(
                f"  {manifest.get('overall_status')} elements={manifest.get('element_count')} "
                f"warnings={len(warnings)}",
                flush=True,
            )
    summary = {
        "filings": len(rows),
        "by_status": dict(Counter(str(row.get("parse_status")) for row in rows)),
        "patterns": dict(Counter(pattern for row in rows for pattern in row.get("warning_patterns") or [])),
        "rows": rows,
    }
    path = OUT / "parse_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"filings": summary["filings"], "by_status": summary["by_status"], "patterns": summary["patterns"]}, ensure_ascii=False))
    print(f"wrote {path}")


def _warnings(manifest: dict[str, Any]) -> list[str]:
    found: list[str] = []
    for item in manifest.get("warnings") or []:
        if isinstance(item, str) and item not in found:
            found.append(item)
    for member in manifest.get("members") or []:
        if not isinstance(member, dict):
            continue
        for item in member.get("warnings") or []:
            if isinstance(item, str) and item not in found:
                found.append(item)
    return found


def _patterns(warnings: list[str]) -> list[str]:
    patterns = []
    for warning in warnings:
        if warning.startswith("unsupported_tag"):
            patterns.append("unsupported_tag")
        elif warning == "uncertain_xml_structure":
            patterns.append("uncertain_xml_structure")
        elif "table" in warning or "span" in warning or "cell" in warning:
            patterns.append("table_span_or_size")
        elif warning in {"max_depth_exceeded", "max_elements_exceeded"}:
            patterns.append("size_limit")
        else:
            patterns.append(warning)
    return sorted(set(patterns))


if __name__ == "__main__":
    main()

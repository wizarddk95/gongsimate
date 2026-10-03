# ruff: noqa: E501
"""Choose four non-financial issuers for the generalization smoke.

Criteria are written before any filing is downloaded. Parser status is not an
input. A company is skipped only for a pre-registered access reason: market,
financial industry, missing periodic coverage, or a primary filing that the
existing ingestion size limit rejects.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import httpx

from disclosure_agent.agent.company import normalize_name
from disclosure_agent.cli import load_dotenv_files, require_api_key
from disclosure_agent.dart import redact_secret
from disclosure_agent.discovery import CorpRecord, discover_filings
from disclosure_agent.errors import FatalBatchError, RateLimitError, SizeLimitError
from disclosure_agent.lock import output_run_lock
from disclosure_agent.sync import sync_receipt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evaluation" / "generalization_smoke"
CACHE = ROOT / "data" / "cache" / "dart" / "corp_codes.json"
RAW = ROOT / "data" / "raw"
BGN_DE = "20240101"
END_DE = "20261002"
ISU_STOCK = "007660"
PERIOD_RE = re.compile(r"\((\d{4})\.(\d{2})\)\s*$")
FINANCIAL_NAME = ("은행", "증권", "보험", "금융", "카드", "캐피탈", "저축", "리츠", "REIT", "reit")
COLLISION = {
    "요즘",
    "미래",
    "대상",
    "한국",
    "우리",
    "하나",
    "대신",
    "대성",
    "동서",
    "태양",
    "하늘",
    "제일",
    "중앙",
    "동양",
    "한양",
    "서울",
    "아시아",
    "넥스트",
    "테크",
    "바이오",
    "소프트",
    "미디어",
    "엔터",
    "나무",
    "사람",
    "바다",
    "최근",
    "최신",
    "현재",
    "실적",
    "매출",
}
MARKET = {"Y": "KOSPI", "K": "KOSDAQ", "N": "KONEX", "E": "OTHER"}
CRITERIA = {
    "frozen_before_download": True,
    "as_of_calendar": "2026-10-02",
    "receipt_window": {"bgn_de": BGN_DE, "end_de": END_DE},
    "excluded_industries": "DART induty_code prefix 64/65/66, or a name containing 은행/증권/보험/금융/카드/캐피탈/저축/리츠",
    "excluded_development_corpus": "stock_code 007660 이수페타시스",
    "size_proxy": "Naver mobile marketValue ranking fetched during this run. Unit is 억원.",
    "strata": {
        "A": "KOSPI non-financial with the highest market cap that passes the access gates. Industry is the DART induty 2-digit prefix.",
        "B": "KOSPI non-financial, not A, whose market cap is inside 0.5x-2x of 007660. If that band has no access-eligible company, expand once to 0.25x-4x. Among the band, pick the smallest absolute log distance, requiring a different induty prefix from A.",
        "C": "KOSDAQ non-financial with the highest market cap and an induty prefix different from A and B.",
        "D": "Listed non-financial whose normalized corp name is an exact member of the collision lexicon. Prefer the one with the most correction periodic filings, then the lowest stock code. If none pass the access gates, try listed names of length 2 in stock-code order, at most 20 list calls, and require at least one correction.",
    },
    "coverage": "At least two distinct non-correction 사업보고서 periods, one non-correction 반기보고서, and one non-correction 분기보고서 inside the receipt window.",
    "primary_filings": "Two latest annual originals, one latest half original, one latest quarter original. Corrections of those same type+period are downloaded for observation and do not replace this set.",
    "access_reject_only": [
        "not in corp cache",
        "financial industry",
        "market class mismatch",
        "missing periodic coverage",
        "primary filing SizeLimitError or remote download failure",
        "RateLimitError aborts the run",
    ],
    "not_used": "parser overall_status, chunk count, retrieval quality",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selection_criteria.json").write_text(
        json.dumps(CRITERIA, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    load_dotenv_files()
    api_key = require_api_key()
    by_stock = _load_listed()
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    with httpx.Client(timeout=30.0, follow_redirects=True, headers=headers) as web:
        kospi = _market_rows(web, "KOSPI", pages=5)
        kosdaq = _market_rows(web, "KOSDAQ", pages=3)
        isu_cap = _cap_of(kospi, ISU_STOCK)
        if isu_cap is None:
            isu_cap = _integration_cap(web, ISU_STOCK)
    print(f"kospi_rows={len(kospi)} kosdaq_rows={len(kosdaq)} isu_cap_eok={isu_cap}", flush=True)
    if isu_cap is None:
        raise SystemExit("007660 market cap was not found")
    rejections: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    with httpx.Client(timeout=30.0, follow_redirects=False) as dart:
        chosen_a = _pick_ranked(
            kospi,
            by_stock,
            dart,
            api_key,
            stratum="A",
            corp_cls="Y",
            blocked_industries=set(),
            limit=8,
            rejections=rejections,
        )
        if chosen_a is None:
            _write(selected, rejections, isu_cap, kospi, kosdaq)
            raise SystemExit("stratum A was not filled")
        selected.append(chosen_a)
        industry_a = chosen_a["induty_prefix"]
        chosen_b = _pick_mid(
            kospi,
            by_stock,
            dart,
            api_key,
            isu_cap=isu_cap,
            blocked_industries={industry_a},
            rejections=rejections,
        )
        if chosen_b is None:
            _write(selected, rejections, isu_cap, kospi, kosdaq)
            raise SystemExit("stratum B was not filled")
        selected.append(chosen_b)
        chosen_c = _pick_ranked(
            kosdaq,
            by_stock,
            dart,
            api_key,
            stratum="C",
            corp_cls="K",
            blocked_industries={industry_a, chosen_b["induty_prefix"]},
            limit=12,
            rejections=rejections,
        )
        if chosen_c is None:
            _write(selected, rejections, isu_cap, kospi, kosdaq)
            raise SystemExit("stratum C was not filled")
        selected.append(chosen_c)
        chosen_d = _pick_edge(
            by_stock,
            dart,
            api_key,
            blocked_codes={item["corp_code"] for item in selected},
            rejections=rejections,
        )
        if chosen_d is None:
            _write(selected, rejections, isu_cap, kospi, kosdaq)
            raise SystemExit("stratum D was not filled")
        selected.append(chosen_d)
    _write(selected, rejections, isu_cap, kospi, kosdaq)
    print("selected", ", ".join(item["corp_name"] for item in selected), flush=True)


def _pick_ranked(
    rows: list[dict[str, Any]],
    by_stock: dict[str, dict[str, Any]],
    dart: httpx.Client,
    api_key: str,
    *,
    stratum: str,
    corp_cls: str,
    blocked_industries: set[str],
    limit: int,
    rejections: list[dict[str, Any]],
) -> dict[str, Any] | None:
    tried = 0
    for row in rows:
        if tried >= limit:
            break
        if row["stock_code"] == ISU_STOCK:
            continue
        tried += 1
        chosen = _consider(
            row,
            by_stock,
            dart,
            api_key,
            stratum=stratum,
            required_cls=corp_cls,
            blocked_industries=blocked_industries,
            rejections=rejections,
            rationale=_ranked_reason(stratum, row),
        )
        if chosen is not None:
            return chosen
    return None


def _pick_mid(
    rows: list[dict[str, Any]],
    by_stock: dict[str, dict[str, Any]],
    dart: httpx.Client,
    api_key: str,
    *,
    isu_cap: int,
    blocked_industries: set[str],
    rejections: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for low, high, label in ((0.5, 2.0, "0.5x-2x"), (0.25, 4.0, "0.25x-4x")):
        band = [
            row
            for row in rows
            if row["stock_code"] != ISU_STOCK and low * isu_cap <= row["market_cap_eok"] <= high * isu_cap
        ]
        band.sort(key=lambda row: abs(_log_ratio(row["market_cap_eok"], isu_cap)))
        print(f"mid band {label} candidates={len(band)}", flush=True)
        for row in band[:12]:
            chosen = _consider(
                row,
                by_stock,
                dart,
                api_key,
                stratum="B",
                required_cls="Y",
                blocked_industries=blocked_industries,
                rejections=rejections,
                rationale=(
                    f"KOSPI mid band {label} of 007660 market cap; "
                    f"closest remaining non-financial with a different industry prefix"
                ),
                extra={"mid_band": label, "isu_market_cap_eok": isu_cap},
            )
            if chosen is not None:
                return chosen
    return None


def _pick_edge(
    by_stock: dict[str, dict[str, Any]],
    dart: httpx.Client,
    api_key: str,
    *,
    blocked_codes: set[str],
    rejections: list[dict[str, Any]],
) -> dict[str, Any] | None:
    pool = []
    for stock, row in by_stock.items():
        name = normalize_name(str(row.get("corp_name") or ""))
        if name in COLLISION:
            pool.append((stock, row, name))
    pool.sort(key=lambda item: item[0])
    ranked: list[tuple[int, str, dict[str, Any], dict[str, Any]]] = []
    for stock, row, name in pool:
        if str(row.get("corp_code") or "") in blocked_codes:
            continue
        profile = _profile(dart, api_key, row)
        if profile is None:
            rejections.append({"stratum": "D", "stock_code": stock, "reason": "company.json failed"})
            continue
        if _financial(profile, str(row.get("corp_name") or "")):
            rejections.append(
                {
                    "stratum": "D",
                    "stock_code": stock,
                    "corp_name": profile.get("corp_name"),
                    "reason": "financial",
                }
            )
            continue
        filings = _list_filings(dart, api_key, profile)
        if filings is None:
            rejections.append({"stratum": "D", "stock_code": stock, "reason": "list failed"})
            continue
        plan = _plan_filings(filings)
        if plan is None:
            rejections.append(
                {
                    "stratum": "D",
                    "stock_code": stock,
                    "corp_name": profile.get("corp_name"),
                    "reason": "missing periodic coverage",
                    "coverage": _coverage(filings),
                }
            )
            continue
        ranked.append((plan["correction_count_window"], stock, profile, plan))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    for corrections, stock, profile, plan in ranked:
        chosen = _download_candidate(
            profile,
            plan,
            api_key,
            stratum="D",
            rationale=(
                "normalized corp name is an exact collision-lexicon word; "
                f"correction periodic filings in window={corrections}"
            ),
            edge_case="company name collides with a common Korean word; correction filings observed in the list",
            rejections=rejections,
            extra={"collision_name": normalize_name(str(profile.get("corp_name") or ""))},
        )
        if chosen is not None:
            return chosen
    print("edge lexicon exhausted; fallback length-2 names", flush=True)
    checked = 0
    shorts = [
        row
        for row in by_stock.values()
        if len(normalize_name(str(row.get("corp_name") or ""))) == 2
        and str(row.get("corp_code") or "") not in blocked_codes
        and normalize_name(str(row.get("corp_name") or "")) not in COLLISION
    ]
    shorts.sort(key=lambda row: str(row.get("stock_code") or ""))
    for row in shorts:
        if checked >= 20:
            break
        checked += 1
        profile = _profile(dart, api_key, row)
        if profile is None or _financial(profile, str(row.get("corp_name") or "")):
            continue
        filings = _list_filings(dart, api_key, profile)
        if filings is None:
            continue
        plan = _plan_filings(filings)
        if plan is None or plan["correction_count_window"] < 1:
            continue
        chosen = _download_candidate(
            profile,
            plan,
            api_key,
            stratum="D",
            rationale="fallback: length-2 listed name with at least one correction periodic filing",
            edge_case="short corp name plus a correction filing",
            rejections=rejections,
        )
        if chosen is not None:
            return chosen
    return None


def _consider(
    row: dict[str, Any],
    by_stock: dict[str, dict[str, Any]],
    dart: httpx.Client,
    api_key: str,
    *,
    stratum: str,
    required_cls: str,
    blocked_industries: set[str],
    rejections: list[dict[str, Any]],
    rationale: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    cached = by_stock.get(row["stock_code"])
    if cached is None:
        rejections.append(
            {"stratum": stratum, "stock_code": row["stock_code"], "stock_name": row["stock_name"], "reason": "not in corp cache"}
        )
        return None
    if any(word in str(cached.get("corp_name") or "") for word in FINANCIAL_NAME):
        rejections.append(
            {
                "stratum": stratum,
                "stock_code": row["stock_code"],
                "corp_name": cached.get("corp_name"),
                "reason": "financial name",
            }
        )
        return None
    profile = _profile(dart, api_key, cached)
    if profile is None:
        rejections.append({"stratum": stratum, "stock_code": row["stock_code"], "reason": "company.json failed"})
        return None
    corp_cls = str(profile.get("corp_cls") or "")
    if corp_cls != required_cls:
        rejections.append(
            {
                "stratum": stratum,
                "stock_code": row["stock_code"],
                "corp_name": profile.get("corp_name"),
                "reason": f"corp_cls={corp_cls} required={required_cls}",
            }
        )
        return None
    if _financial(profile, str(profile.get("corp_name") or "")):
        rejections.append(
            {
                "stratum": stratum,
                "stock_code": row["stock_code"],
                "corp_name": profile.get("corp_name"),
                "induty_code": profile.get("induty_code"),
                "reason": "financial",
            }
        )
        return None
    prefix = _induty_prefix(str(profile.get("induty_code") or ""))
    if prefix in blocked_industries:
        rejections.append(
            {
                "stratum": stratum,
                "stock_code": row["stock_code"],
                "corp_name": profile.get("corp_name"),
                "reason": f"industry prefix {prefix} already used",
            }
        )
        return None
    filings = _list_filings(dart, api_key, profile)
    if filings is None:
        rejections.append({"stratum": stratum, "stock_code": row["stock_code"], "reason": "list failed"})
        return None
    plan = _plan_filings(filings)
    if plan is None:
        rejections.append(
            {
                "stratum": stratum,
                "stock_code": row["stock_code"],
                "corp_name": profile.get("corp_name"),
                "reason": "missing periodic coverage",
                "coverage": _coverage(filings),
            }
        )
        return None
    edge = "standard periodic mix; no name collision assumed"
    return _download_candidate(
        profile,
        plan,
        api_key,
        stratum=stratum,
        rationale=rationale,
        edge_case=edge,
        rejections=rejections,
        extra={
            "market_cap_eok": row["market_cap_eok"],
            "market_cap_rank_in_fetch": row["rank"],
            "naver_name": row["stock_name"],
            **(extra or {}),
        },
    )


def _download_candidate(
    profile: dict[str, Any],
    plan: dict[str, Any],
    api_key: str,
    *,
    stratum: str,
    rationale: str,
    edge_case: str,
    rejections: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    name = str(profile.get("corp_name") or "")
    print(f"download stratum={stratum} {name}", flush=True)
    outcomes = []
    try:
        with output_run_lock(RAW):
            for item in plan["download"]:
                listing = _listing(item)
                result = sync_receipt(
                    str(item["rcept_no"]),
                    RAW,
                    api_key,
                    listing=listing,
                )
                safe_error = redact_secret(result.outcome.error_message or "", api_key)
                outcomes.append(
                    {
                        "rcept_no": result.outcome.receipt_no,
                        "report_nm": item.get("report_nm"),
                        "status": result.outcome.status,
                        "sha256": result.outcome.sha256,
                        "size_bytes": result.outcome.size_bytes,
                        "error_type": result.outcome.error_type,
                        "error_message": safe_error or None,
                        "role": item.get("_role"),
                    }
                )
                print(
                    f"  {result.outcome.status} {item.get('report_nm')} {result.outcome.receipt_no}",
                    flush=True,
                )
                if result.outcome.status == "failed" and result.outcome.error_type in {
                    "RateLimitError",
                    "FatalBatchError",
                }:
                    raise FatalBatchError(safe_error or result.outcome.error_type or "batch aborted")
                if result.outcome.status == "failed" and item.get("_role") == "primary":
                    rejections.append(
                        {
                            "stratum": stratum,
                            "corp_name": name,
                            "stock_code": profile.get("stock_code"),
                            "reason": "primary download failed",
                            "error_type": result.outcome.error_type,
                            "receipt_no": result.outcome.receipt_no,
                        }
                    )
                    return None
                time.sleep(0.2)
    except RateLimitError:
        raise
    except FatalBatchError:
        raise
    except SizeLimitError as exc:
        rejections.append(
            {
                "stratum": stratum,
                "corp_name": name,
                "reason": "size limit",
                "detail": redact_secret(str(exc), api_key),
            }
        )
        return None
    prefix = _induty_prefix(str(profile.get("induty_code") or ""))
    return {
        "stratum": stratum,
        "corp_code": str(profile.get("corp_code") or ""),
        "corp_name": name,
        "corp_name_eng": str(profile.get("corp_name_eng") or ""),
        "stock_name": str(profile.get("stock_name") or ""),
        "stock_code": str(profile.get("stock_code") or ""),
        "corp_cls": str(profile.get("corp_cls") or ""),
        "market": MARKET.get(str(profile.get("corp_cls") or ""), str(profile.get("corp_cls") or "")),
        "induty_code": str(profile.get("induty_code") or ""),
        "induty_prefix": prefix,
        "acc_mt": str(profile.get("acc_mt") or ""),
        "selection_rationale": rationale,
        "edge_case": edge_case,
        "coverage": plan["coverage"],
        "correction_count_window": plan["correction_count_window"],
        "corrections": plan["corrections"],
        "primary_filings": plan["primary"],
        "download_outcomes": outcomes,
        **(extra or {}),
    }


def _profile(client: httpx.Client, api_key: str, cached: dict[str, Any]) -> dict[str, Any] | None:
    corp_code = str(cached.get("corp_code") or "")
    try:
        response = client.get(
            "https://opendart.fss.or.kr/api/company.json",
            params={"crtfc_key": api_key, "corp_code": corp_code},
        )
        response.raise_for_status()
    except httpx.HTTPError:
        print(f"company.json http error corp_code={corp_code}", flush=True)
        return None
    payload = response.json()
    if str(payload.get("status") or "") != "000":
        print(
            f"company.json {corp_code} status={payload.get('status')} message={payload.get('message')}",
            flush=True,
        )
        return None
    payload["corp_code"] = corp_code
    if not payload.get("stock_code"):
        payload["stock_code"] = cached.get("stock_code")
    time.sleep(0.15)
    return payload


def _list_filings(
    client: httpx.Client,
    api_key: str,
    profile: dict[str, Any],
) -> list[dict[str, Any]] | None:
    corp = CorpRecord(
        corp_code=str(profile.get("corp_code") or ""),
        corp_name=str(profile.get("corp_name") or ""),
        corp_eng_name=str(profile.get("corp_name_eng") or ""),
        stock_code=str(profile.get("stock_code") or ""),
        modify_date="",
    )
    try:
        manifest = discover_filings(
            api_key,
            corp=corp,
            bgn_de=BGN_DE,
            end_de=END_DE,
            cache_meta={},
            client=client,
        )
    except (RateLimitError, FatalBatchError):
        raise
    except Exception as exc:
        print(f"list failed {corp.corp_name}: {type(exc).__name__}", flush=True)
        return None
    if not manifest.list_complete:
        return None
    return list(manifest.items)


def _plan_filings(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    coverage = _coverage(items)
    if coverage["annual_periods"] < 2 or coverage["half"] < 1 or coverage["quarter"] < 1:
        return None
    annual = _latest_originals(items, "annual", 2)
    half = _latest_originals(items, "half", 1)
    quarter = _latest_originals(items, "quarter", 1)
    if len(annual) < 2 or not half or not quarter:
        return None
    primary = [*annual, *half, *quarter]
    keys = {(item["_kind"], item["_period"]) for item in primary}
    corrections = []
    for item in items:
        kind, corrected, period = _classify(item)
        if corrected and period and (kind, period) in keys:
            copy = dict(item)
            copy["_role"] = "correction"
            copy["_kind"] = kind
            copy["_period"] = period
            corrections.append(copy)
    download = []
    for item in primary:
        copy = dict(item)
        copy["_role"] = "primary"
        download.append(copy)
    download.extend(corrections)
    return {
        "coverage": coverage,
        "primary": [_public_filing(item) for item in primary],
        "corrections": [_public_filing(item) for item in corrections],
        "correction_count_window": coverage["corrections"],
        "download": download,
    }


def _coverage(items: list[dict[str, Any]]) -> dict[str, int]:
    annual_periods = {
        _classify(item)[2]
        for item in items
        if _classify(item)[:2] == ("annual", False) and _classify(item)[2]
    }
    return {
        "annual_periods": len(annual_periods),
        "half": sum(1 for item in items if _classify(item)[:2] == ("half", False)),
        "quarter": sum(1 for item in items if _classify(item)[:2] == ("quarter", False)),
        "corrections": sum(1 for item in items if _classify(item)[1] and _classify(item)[0] != "other"),
        "listed": len(items),
    }


def _latest_originals(items: list[dict[str, Any]], kind: str, count: int) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for item in items:
        item_kind, corrected, period = _classify(item)
        if item_kind != kind or corrected or not period:
            continue
        current = grouped.get(period)
        if current is None or str(item.get("rcept_dt") or "") > str(current.get("rcept_dt") or ""):
            chosen = dict(item)
            chosen["_kind"] = kind
            chosen["_period"] = period
            grouped[period] = chosen
    ordered = sorted(grouped.values(), key=lambda item: str(item["_period"]), reverse=True)
    return ordered[:count]


def _classify(item: dict[str, Any]) -> tuple[str, bool, str]:
    name = str(item.get("report_nm") or "")
    if "분기보고서" in name:
        kind = "quarter"
    elif "반기보고서" in name:
        kind = "half"
    elif "사업보고서" in name:
        kind = "annual"
    else:
        kind = "other"
    match = PERIOD_RE.search(name.strip())
    period = f"{match.group(1)}.{match.group(2)}" if match else ""
    return kind, "정정" in name, period


def _public_filing(item: dict[str, Any]) -> dict[str, str]:
    return {
        "receipt_no": str(item.get("rcept_no") or ""),
        "report_nm": str(item.get("report_nm") or ""),
        "rcept_dt": str(item.get("rcept_dt") or ""),
        "period": str(item.get("_period") or ""),
        "kind": str(item.get("_kind") or ""),
        "role": str(item.get("_role") or "primary"),
        "rm": str(item.get("rm") or ""),
    }


def _listing(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "corp_code": item.get("corp_code"),
        "corp_name": item.get("corp_name"),
        "stock_code": item.get("stock_code"),
        "report_nm": item.get("report_nm"),
        "rcept_no": item.get("rcept_no"),
        "flr_nm": item.get("flr_nm"),
        "rcept_dt": item.get("rcept_dt"),
        "rm": item.get("rm"),
        "corp_cls": item.get("corp_cls"),
    }


def _financial(profile: dict[str, Any], name: str) -> bool:
    if any(word.lower() in name.lower() for word in FINANCIAL_NAME):
        return True
    prefix = _induty_prefix(str(profile.get("induty_code") or ""))
    return prefix in {"64", "65", "66"}


def _induty_prefix(code: str) -> str:
    digits = "".join(char for char in code if char.isdigit())
    if len(digits) >= 2:
        return digits[:2]
    return code.strip()


def _ranked_reason(stratum: str, row: dict[str, Any]) -> str:
    if stratum == "A":
        return (
            "highest fetched KOSPI market cap among non-financial issuers that passed the access gates; "
            f"accepted at fetched rank {row['rank']}"
        )
    return (
        "highest fetched KOSDAQ market cap among non-financial issuers with an unused industry prefix; "
        f"accepted at fetched rank {row['rank']}"
    )


def _log_ratio(cap: int, base: int) -> float:
    import math

    return math.log(cap / base)


def _load_listed() -> dict[str, dict[str, Any]]:
    payload = json.loads(CACHE.read_text(encoding="utf-8"))
    found: dict[str, dict[str, Any]] = {}
    for row in payload.get("companies") or []:
        if not isinstance(row, dict):
            continue
        stock = str(row.get("stock_code") or "").strip()
        if len(stock) != 6 or not stock.isdigit():
            continue
        current = found.get(stock)
        if current is None or str(row.get("modify_date") or "") >= str(current.get("modify_date") or ""):
            found[stock] = row
    return found


def _market_rows(client: httpx.Client, market: str, *, pages: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for page in range(1, pages + 1):
        response = client.get(
            f"https://m.stock.naver.com/api/stocks/marketValue/{market}",
            params={"page": page, "pageSize": 100},
        )
        response.raise_for_status()
        stocks = response.json().get("stocks") or []
        if not stocks:
            break
        for stock in stocks:
            code = str(stock.get("itemCode") or "")
            cap = _parse_eok(stock.get("marketValue"))
            if len(code) != 6 or cap is None:
                continue
            rows.append(
                {
                    "stock_code": code,
                    "stock_name": str(stock.get("stockName") or ""),
                    "market_cap_eok": cap,
                    "rank": len(rows) + 1,
                    "market": market,
                }
            )
    return rows


def _cap_of(rows: list[dict[str, Any]], stock_code: str) -> int | None:
    for row in rows:
        if row["stock_code"] == stock_code:
            return int(row["market_cap_eok"])
    return None


def _integration_cap(client: httpx.Client, stock_code: str) -> int | None:
    response = client.get(f"https://m.stock.naver.com/api/stock/{stock_code}/integration")
    response.raise_for_status()
    for item in response.json().get("totalInfos") or []:
        if isinstance(item, dict) and item.get("code") == "marketValue":
            return _parse_korean_eok(str(item.get("value") or ""))
    return None


def _parse_eok(value: object) -> int | None:
    text = str(value or "").replace(",", "").strip()
    if not text.isdigit():
        return None
    return int(text)


def _parse_korean_eok(text: str) -> int | None:
    compact = text.replace(",", "").replace(" ", "")
    if not compact:
        return None
    total = 0
    match_jo = re.search(r"(\d+)조", compact)
    match_eok = re.search(r"(\d+)억", compact)
    if match_jo:
        total += int(match_jo.group(1)) * 10000
    if match_eok:
        total += int(match_eok.group(1))
    return total or None


def _write(
    selected: list[dict[str, Any]],
    rejections: list[dict[str, Any]],
    isu_cap: int,
    kospi: list[dict[str, Any]],
    kosdaq: list[dict[str, Any]],
) -> None:
    payload = {
        "isu_market_cap_eok": isu_cap,
        "market_fetch": {
            "source": "https://m.stock.naver.com/api/stocks/marketValue/{KOSPI|KOSDAQ}",
            "kospi_rows": len(kospi),
            "kosdaq_rows": len(kosdaq),
            "top_kospi": kospi[:5],
            "top_kosdaq": kosdaq[:5],
        },
        "companies": selected,
        "rejections": rejections,
    }
    path = OUT / "companies.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()

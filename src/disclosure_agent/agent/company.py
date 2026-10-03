"""Gazetteer mention extraction and deterministic company resolution.

The extractor returns surface mentions only. ``company_id`` is assigned here,
and only when the registry contains exactly one match. A mention is never
resolved by guessing among several candidates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from disclosure_agent.discovery import load_corp_cache

ResolutionStatus = Literal["resolved", "ambiguous", "not_found", "unsupported", "no_mention"]

_CORP_MARKERS = ("주식회사", "(주)", "㈜", "（주）")
_PARTICLES = (
    "에서는",
    "에서도",
    "부터는",
    "으로는",
    "로서는",
    "했나요",
    "인가요",
    "했어",
    "에서",
    "으로",
    "에게",
    "까지",
    "부터",
    "처럼",
    "보다",
    "이라고",
    "라고",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "와",
    "과",
    "도",
    "만",
    "에",
    "로",
    "의",
    "요",
    "까",
    "죠",
)
_STOPLIST = {
    "안녕",
    "안녕하세요",
    "매출",
    "실적",
    "영업이익",
    "사업",
    "공시",
    "보고서",
    "제품",
    "위험",
    "부채",
    "배당",
    "수주",
    "원인",
    "이유",
    "증가",
    "감소",
    "확정",
    "주요",
    "알려",
    "알려줘",
    "뭐야",
    "무엇",
    "왜",
    "얼마",
    "현재",
    "올해",
    "내년",
    "질문",
    "회사",
    "기업",
    "내용",
    "개요",
    "연결",
    "별도",
    "기준",
    "사업보고서",
    "rag",
}
_DISCLOSURE_HINTS = (
    "매출",
    "실적",
    "영업이익",
    "사업",
    "공시",
    "보고서",
    "제품",
    "위험",
    "부채",
    "배당",
    "수주",
    "원인",
    "이유",
    "증가",
    "감소",
    "점유",
)
_SMALLTALK_EXACT = {"안녕", "안녕하세요", "하이", "hello", "hi"}
_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣&]+")
_YEAR_RE = re.compile(r"(20\d{2})")


@dataclass(frozen=True, slots=True)
class CompanyRecord:
    corp_code: str
    corp_name: str
    stock_code: str
    corp_eng_name: str = ""


@dataclass(frozen=True, slots=True)
class Resolution:
    status: ResolutionStatus
    company_id: str | None = None
    corp_code: str | None = None
    company_name: str | None = None
    stock_code: str | None = None
    mention: str | None = None
    candidates: tuple[CompanyRecord, ...] = ()
    candidate_total: int = 0

    def public_candidates(self, limit: int = 8) -> list[dict[str, str]]:
        """Names a user can choose. Internal ids stay off this list."""
        rows = []
        for company in self.candidates[:limit]:
            row = {"company_name": company.corp_name}
            if company.stock_code:
                row["stock_code"] = company.stock_code
            rows.append(row)
        return rows


def normalize_name(value: str) -> str:
    text = value.strip().casefold()
    text = re.sub(r"\s+", "", text)
    for marker in _CORP_MARKERS:
        text = text.replace(marker, "")
    return text


def is_smalltalk(message: str) -> bool:
    folded = normalize_name(message).strip("!?.~ ")
    return folded in _SMALLTALK_EXACT or folded.startswith("안녕")


def is_disclosure_question(message: str) -> bool:
    folded = normalize_name(message)
    return any(hint in folded for hint in _DISCLOSURE_HINTS)


def mentioned_years(message: str) -> list[str]:
    return _YEAR_RE.findall(message)


def _strip_particles(token: str) -> str:
    changed = True
    while changed and len(token) > 2:
        changed = False
        for particle in _PARTICLES:
            remain = len(token) - len(particle)
            if remain >= 2 and token.endswith(particle):
                token = token[:remain]
                changed = True
                break
    return token


def _tokens(message: str) -> list[str]:
    tokens = []
    for raw in _TOKEN_RE.findall(message):
        token = _strip_particles(normalize_name(raw))
        if len(token) >= 2:
            tokens.append(token)
    return tokens


def _prefer_longer(names: list[str]) -> list[str]:
    kept: list[str] = []
    for name in sorted(set(names), key=len, reverse=True):
        if any(name in longer for longer in kept):
            continue
        kept.append(name)
    return kept


class CompanyRegistry:
    """Name, stock-code, and corp-code lookup over an explicit company list."""

    def __init__(
        self,
        companies: list[CompanyRecord],
        *,
        indexed_corp_codes: set[str] | None = None,
    ) -> None:
        self.indexed_corp_codes = set(indexed_corp_codes or ())
        self._by_corp: dict[str, list[CompanyRecord]] = {}
        self._by_stock: dict[str, list[CompanyRecord]] = {}
        self._by_name: dict[str, list[CompanyRecord]] = {}
        self._by_eng: dict[str, list[CompanyRecord]] = {}
        self._by_first: dict[str, list[str]] = {}
        self._by_prefix2: dict[str, list[CompanyRecord]] = {}
        for company in companies:
            if not company.corp_code or not company.corp_name:
                continue
            self._by_corp.setdefault(company.corp_code, []).append(company)
            if company.stock_code:
                self._by_stock.setdefault(company.stock_code, []).append(company)
            name = normalize_name(company.corp_name)
            if len(name) >= 2:
                self._by_name.setdefault(name, []).append(company)
                self._by_first.setdefault(name[0], []).append(name)
                self._by_prefix2.setdefault(name[:2], []).append(company)
            eng = normalize_name(company.corp_eng_name)
            if len(eng) >= 3:
                self._by_eng.setdefault(eng, []).append(company)

    @classmethod
    def from_records(
        cls,
        companies: list[CompanyRecord],
        *,
        indexed_corp_codes: set[str] | None = None,
    ) -> CompanyRegistry:
        return cls(companies, indexed_corp_codes=indexed_corp_codes)

    @classmethod
    def from_corp_cache(
        cls,
        cache_dir: Path,
        *,
        indexed_corp_codes: set[str] | None = None,
        fallback: list[CompanyRecord] | None = None,
    ) -> CompanyRegistry:
        """Load ``data/cache/dart/corp_codes.json`` when it is present.

        A missing cache uses ``fallback`` and does not call OpenDART.
        """
        payload = load_corp_cache(cache_dir)
        if payload is None:
            return cls(list(fallback or ()), indexed_corp_codes=indexed_corp_codes)
        companies: list[CompanyRecord] = []
        for row in payload.get("companies") or []:
            if not isinstance(row, dict):
                continue
            companies.append(
                CompanyRecord(
                    corp_code=str(row.get("corp_code") or "").strip(),
                    corp_name=str(row.get("corp_name") or "").strip(),
                    stock_code=str(row.get("stock_code") or "").strip(),
                    corp_eng_name=str(row.get("corp_eng_name") or "").strip(),
                )
            )
        return cls(companies, indexed_corp_codes=indexed_corp_codes)

    def extract_mentions(self, message: str) -> list[str]:
        """Return surface mentions. This method does not assign company ids.

        Adjacent tokens are joined so a spaced name such as ``삼성 전자`` can
        match ``삼성전자``. Resolution of that string is ``resolve_mention``.
        """
        found: list[str] = []
        prefix_mentions: list[str] = []
        tokens = _tokens(message)
        found.extend(self._joined_mentions(tokens))
        for token in tokens:
            if token in self._by_stock or token in self._by_corp:
                found.append(token)
                continue
            if token.isdigit():
                continue
            contained = self._names_in_token(token)
            if contained:
                found.extend(contained)
                continue
            if token in _STOPLIST or len(token) < 2:
                continue
            if self._prefix_candidates(token):
                prefix_mentions.append(token)
        if found:
            return _prefer_longer(found)
        return _prefer_longer(prefix_mentions)

    def _joined_mentions(self, tokens: list[str]) -> list[str]:
        found: list[str] = []
        width = min(4, len(tokens))
        for window in range(width, 1, -1):
            for index in range(len(tokens) - window + 1):
                joined = "".join(tokens[index : index + window])
                if (
                    joined in self._by_name
                    or joined in self._by_eng
                    or joined in self._by_stock
                    or joined in self._by_corp
                ):
                    found.append(joined)
        return found

    def resolve_mention(self, mention: str) -> Resolution:
        key = normalize_name(mention)
        if not key:
            return Resolution(status="not_found", mention=mention)
        exact = _unique_companies(self._exact_matches(key))
        longer: list[CompanyRecord] = []
        if not key.isdigit() and len(key) >= 2:
            longer = [
                company
                for company in self._prefix_candidates(key)
                if normalize_name(company.corp_name) != key
            ]
        # A listed exact name is the company the user named. An unlisted exact
        # name that is also a prefix of other companies is not chosen for them.
        if len(exact) == 1 and exact[0].stock_code:
            return self._finish(mention, exact)
        if len(exact) == 1 and not longer:
            return self._finish(mention, exact)
        if exact or longer:
            return self._finish(mention, exact + longer)
        return Resolution(status="not_found", mention=mention)

    def resolve_mentions(self, mentions: list[str]) -> Resolution:
        if not mentions:
            return Resolution(status="no_mention")
        resolved: list[Resolution] = []
        for mention in mentions:
            one = self.resolve_mention(mention)
            if one.status == "ambiguous":
                return one
            if one.status in {"resolved", "unsupported"}:
                resolved.append(one)
            elif one.status == "not_found" and len(mentions) == 1:
                return one
        unique: dict[str, Resolution] = {}
        for item in resolved:
            if item.corp_code:
                unique[item.corp_code] = item
        if len(unique) == 1:
            return next(iter(unique.values()))
        if len(unique) > 1:
            # An indexed company is the one this product can search. Another
            # exact name that is not indexed does not veto it. Two indexed
            # companies stay ambiguous. Unindexed-only collisions stay ambiguous.
            indexed = [item for item in unique.values() if item.status == "resolved"]
            pool = indexed or list(unique.values())
            if len(pool) == 1:
                return pool[0]
            companies = [
                CompanyRecord(
                    corp_code=item.corp_code or "",
                    corp_name=item.company_name or "",
                    stock_code=item.stock_code or "",
                )
                for item in pool
            ]
            ordered = _sort_companies(companies)
            return Resolution(
                status="ambiguous",
                mention=" ".join(item.mention or "" for item in pool),
                candidates=tuple(ordered),
                candidate_total=len(ordered),
            )
        return Resolution(status="not_found", mention=mentions[0])

    def _exact_matches(self, key: str) -> list[CompanyRecord]:
        if key in self._by_stock:
            return list(self._by_stock[key])
        if key in self._by_corp:
            return list(self._by_corp[key])
        if key in self._by_name:
            return list(self._by_name[key])
        if key in self._by_eng:
            return list(self._by_eng[key])
        return []

    def _prefix_candidates(self, key: str) -> list[CompanyRecord]:
        if len(key) < 2:
            return []
        bucket = self._by_prefix2.get(key[:2], ())
        return [company for company in bucket if normalize_name(company.corp_name).startswith(key)]

    def _names_in_token(self, token: str) -> list[str]:
        found: list[str] = []
        if token in self._by_name or token in self._by_eng:
            found.append(token)
        seen: set[str] = set()
        for char in set(token):
            for name in self._by_first.get(char, ()):
                if name in seen:
                    continue
                seen.add(name)
                if name != token and name in token:
                    found.append(name)
        return found

    def _finish(self, mention: str, matches: list[CompanyRecord]) -> Resolution:
        unique = _unique_companies(matches)
        if not unique:
            return Resolution(status="not_found", mention=mention)
        if len(unique) > 1:
            ordered = _sort_companies(unique)
            return Resolution(
                status="ambiguous",
                mention=mention,
                candidates=tuple(ordered),
                candidate_total=len(ordered),
            )
        company = unique[0]
        status: ResolutionStatus = (
            "resolved" if company.corp_code in self.indexed_corp_codes else "unsupported"
        )
        return Resolution(
            status=status,
            company_id=company.corp_code,
            corp_code=company.corp_code,
            company_name=company.corp_name,
            stock_code=company.stock_code,
            mention=mention,
            candidates=(company,),
            candidate_total=1,
        )


def clarification_question(resolution: Resolution) -> str:
    names = [company.corp_name for company in resolution.candidates[:3]]
    total = resolution.candidate_total
    if total == 2 and len(names) == 2:
        return f"{names[0]} 또는 {names[1]} 중 어느 기업을 말씀하시는 건가요?"
    if names:
        shown = ", ".join(names)
        return (
            f"{resolution.mention or '해당 이름'}에 해당하는 기업이 {total}곳입니다. "
            f"예를 들어 {shown}입니다. 회사명을 더 정확히 알려주세요."
        )
    return "어느 기업의 공시를 볼까요? 회사명을 함께 입력해 주세요."


def _unique_companies(companies: list[CompanyRecord]) -> list[CompanyRecord]:
    found: dict[str, CompanyRecord] = {}
    for company in companies:
        found.setdefault(company.corp_code, company)
    return list(found.values())


def _sort_companies(companies: list[CompanyRecord]) -> list[CompanyRecord]:
    return sorted(
        companies,
        key=lambda company: (
            0 if company.stock_code else 1,
            company.corp_name,
            company.corp_code,
        ),
    )


def resolution_to_dict(resolution: Resolution | None) -> dict[str, Any] | None:
    if resolution is None or resolution.status == "no_mention":
        return None
    payload: dict[str, Any] = {"status": resolution.status}
    if resolution.company_id:
        payload["company_id"] = resolution.company_id
        payload["corp_code"] = resolution.corp_code
        payload["company_name"] = resolution.company_name
    return payload

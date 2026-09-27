"""Reproducible Korean BM25 baseline. No external dictionary."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Any

TOKENIZER_VERSION = "ko-mixed-v1"
TOKENIZER_RULES = [
    "Unicode NFKC",
    "Numbers keep comma groups, then commas are removed",
    "Hangul runs and ASCII words of length >= 2",
    "ASCII casefold",
    "Hangul surface form plus overlapping syllable bigrams when the run is longer than 2",
    "Percent signs are stripped from numeric tokens",
    "No stopword list and no morphological dictionary",
    "Query term frequency is saturated at 1",
    "Documents with score 0 are omitted from top-k",
]

BM25_PARAMS = {"k1": 1.2, "b": 0.75, "idf": "log(1 + (N - df + 0.5) / (df + 0.5))"}

_TOKEN_RE = re.compile(
    r"\d{1,3}(?:,\d{3})+(?:\.\d+)?%?"
    r"|\d+(?:\.\d+)?%?"
    r"|[가-힣]+"
    r"|[A-Za-z][A-Za-z0-9]{1,}"
)


def tokenize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text or "")
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(normalized):
        raw = match.group(0)
        if "가" <= raw[0] <= "힣":
            tokens.append(raw)
            if len(raw) > 2:
                tokens.extend(raw[index : index + 2] for index in range(len(raw) - 1))
            continue
        if raw[0].isascii() and raw[0].isalpha():
            tokens.append(raw.casefold())
            continue
        number = raw.replace(",", "").removesuffix("%")
        if number:
            tokens.append(number)
    return tokens


class Bm25Index:
    def __init__(self, chunks: list[dict[str, Any]], *, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunks = chunks
        self.doc_tokens: list[Counter[str]] = []
        self.doc_len: list[int] = []
        df: Counter[str] = Counter()
        for chunk in chunks:
            counts = Counter(tokenize(chunk.get("text") or ""))
            self.doc_tokens.append(counts)
            length = sum(counts.values())
            self.doc_len.append(length)
            df.update(counts.keys())
        self.df = dict(df)
        self.n = len(chunks)
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0

    def search(self, query: str, k: int) -> list[dict[str, Any]]:
        if k < 1 or self.n == 0:
            return []
        terms = sorted(set(tokenize(query)))
        ranked: list[tuple[float, int]] = []
        for index, counts in enumerate(self.doc_tokens):
            score = self._score(terms, counts, self.doc_len[index])
            if score > 0:
                ranked.append((score, index))
        ranked.sort(
            key=lambda item: (
                -item[0],
                self.chunks[item[1]]["ordinal"],
                self.chunks[item[1]]["chunk_id"],
            )
        )
        hits = []
        for rank, (score, index) in enumerate(ranked[:k], start=1):
            chunk = self.chunks[index]
            hits.append(
                {
                    "rank": rank,
                    "chunk_id": chunk["chunk_id"],
                    "ordinal": chunk["ordinal"],
                    "score": score,
                }
            )
        return hits

    def _score(self, terms: list[str], counts: Counter[str], length: int) -> float:
        if not terms or self.avgdl <= 0:
            return 0.0
        score = 0.0
        for term in terms:
            tf = counts.get(term, 0)
            if tf <= 0:
                continue
            df = self.df.get(term, 0)
            idf = math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))
            denom = tf + self.k1 * (1.0 - self.b + self.b * length / self.avgdl)
            score += idf * (tf * (self.k1 + 1.0)) / denom
        return score


def index_manifest(index: Bm25Index, *, scope: dict[str, Any]) -> dict[str, Any]:
    return {
        "tokenizer_version": TOKENIZER_VERSION,
        "tokenizer_rules": list(TOKENIZER_RULES),
        "bm25": dict(BM25_PARAMS),
        "document_count": index.n,
        "average_document_length": index.avgdl,
        "vocabulary_size": len(index.df),
        "index_scope": scope,
        "zero_score_omitted_from_topk": True,
    }

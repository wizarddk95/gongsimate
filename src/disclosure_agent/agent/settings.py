"""Run limits and answer-model settings for the product agent."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

MAX_SEARCH_CALLS = 2
MAX_LLM_CALLS = 4
# Product retrieval is the offline H10 rule, not a second fusion.
# Support is every Dense-D2a directory under data/product/indexes.
# BM25 reads the same retrieval-unit text as Dense. Filing filters run first.
# A list shorter than the candidate window does not fall back to Dense-D2a.
PRODUCT_RETRIEVAL_POLICY = "bm25_primary_dense_head"
PRODUCT_RETRIEVAL_SUPPORT = "all_product_dense_d2a_indexes"
PRODUCT_RETRIEVAL_SHORT_LIST_POLICY = "h10_unavailable"
# Dense-D2a recall was measured at 20. That depth is candidate recall.
# How many of those candidates reach the model is a context-budget decision.
DEFAULT_RETRIEVAL_TOP_K = 20
# H10 keeps BM25 ranks 1–9 and may place the Dense head at fused rank 10.
# A product disclosure search must request at least that slot. H10Search still
# slices to the explicit ``SearchDisclosuresRequest.top_k``; only the product
# agent raises a smaller planner value, and the tool log records both numbers.
# A server cap below this floor cannot show fused rank 10.
PRODUCT_DISCLOSURE_MIN_TOP_K = 10
DEFAULT_GENERATION_CONTEXT_TOKEN_BUDGET = 4096
DEFAULT_GENERATION_MAX_EVIDENCE = 20
CITED_EVIDENCE_LIMIT = 3
REQUEST_TIMEOUT_SECONDS = 180.0
EXCERPT_CHARS = 480
# One structured-output failure may be retried once. Both attempts count as LLM calls.
STRUCTURED_OUTPUT_ATTEMPTS = 2
DEFAULT_OPENAI_MODEL = "gpt-6-luna"
DEFAULT_OPENAI_TIMEOUT_SECONDS = 60.0
DEFAULT_OPENAI_MAX_RETRIES = 1


@dataclass(frozen=True)
class AgentLimits:
    max_search_calls: int = MAX_SEARCH_CALLS
    max_llm_calls: int = MAX_LLM_CALLS

    def __post_init__(self) -> None:
        if self.max_search_calls < 0 or self.max_llm_calls < 0:
            raise ValueError("run limits must be >= 0")


@dataclass(frozen=True)
class OpenAIConfig:
    """Model settings. ``api_key`` is excluded from ``repr`` and logs."""

    api_key: str
    model: str = DEFAULT_OPENAI_MODEL
    timeout_seconds: float = DEFAULT_OPENAI_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_OPENAI_MAX_RETRIES

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("OPENAI_MODEL is empty")
        if self.timeout_seconds <= 0:
            raise ValueError("OPENAI_TIMEOUT_SECONDS must be > 0")
        if self.max_retries < 0:
            raise ValueError("OPENAI_MAX_RETRIES must be >= 0")

    def __repr__(self) -> str:
        return (
            "OpenAIConfig("
            f"model={self.model!r}, "
            f"timeout_seconds={self.timeout_seconds!r}, "
            f"max_retries={self.max_retries!r}, "
            f"api_key_present={bool(self.api_key)})"
        )

    @property
    def configured(self) -> bool:
        return bool(self.api_key)


def product_search_top_k(requested: int, server_max: int) -> int:
    """Top-k the product agent sends to H10.

    Values of ``PRODUCT_DISCLOSURE_MIN_TOP_K`` or more are kept, up to
    ``server_max``. A smaller planner value is raised to that floor so the
    Dense head at fused rank 10 can enter the search response. The raise is
    recorded as ``top_k_requested`` and ``top_k_applied``. Direct callers of
    ``H10Search`` are unchanged: they still receive ``request.top_k`` hits.
    If ``server_max`` is below the floor, the server max wins and rank 10
    stays outside that call.
    """
    if server_max < 1:
        raise ValueError("server max top_k must be >= 1")
    if requested < 1:
        requested = 1
    floor = min(PRODUCT_DISCLOSURE_MIN_TOP_K, server_max)
    return min(server_max, max(requested, floor))


def load_openai_config(environ: Mapping[str, str] | None = None) -> OpenAIConfig:
    """Read answer-model settings. Missing model uses the cost-efficient default."""
    env = os.environ if environ is None else environ
    timeout_raw = env.get("OPENAI_TIMEOUT_SECONDS", "").strip()
    retries_raw = env.get("OPENAI_MAX_RETRIES", "").strip()
    try:
        timeout = float(timeout_raw) if timeout_raw else DEFAULT_OPENAI_TIMEOUT_SECONDS
        retries = int(retries_raw) if retries_raw else DEFAULT_OPENAI_MAX_RETRIES
    except ValueError as exc:
        raise ValueError("OPENAI_TIMEOUT_SECONDS and OPENAI_MAX_RETRIES must be numbers") from exc
    return OpenAIConfig(
        api_key=env.get("OPENAI_API_KEY", "").strip(),
        model=env.get("OPENAI_MODEL", "").strip() or DEFAULT_OPENAI_MODEL,
        timeout_seconds=timeout,
        max_retries=retries,
    )

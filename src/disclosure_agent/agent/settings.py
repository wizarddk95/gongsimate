"""Run limits for the product agent. These bound a single /ask call."""

from __future__ import annotations

from dataclasses import dataclass

MAX_SEARCH_CALLS = 2
MAX_LLM_CALLS = 4
DEFAULT_TOP_K = 8
CITED_EVIDENCE_LIMIT = 3
REQUEST_TIMEOUT_SECONDS = 180.0
EXCERPT_CHARS = 480


@dataclass(frozen=True)
class AgentLimits:
    max_search_calls: int = MAX_SEARCH_CALLS
    max_llm_calls: int = MAX_LLM_CALLS

    def __post_init__(self) -> None:
        if self.max_search_calls < 0 or self.max_llm_calls < 0:
            raise ValueError("run limits must be >= 0")

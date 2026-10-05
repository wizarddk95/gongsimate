"""Small provider calls, checkpoints, and a pre-run cost guard.

Prices in the run config are estimates. Token counts in the call log are the measured facts.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from disclosure_agent.agent.llm import normalize_provider_error
from disclosure_agent.chunking import canonical_dumps, sha256_text
from disclosure_agent.errors import UsageError
from disclosure_agent.evalset_v2 import PROMPT_VERSION

BLIND_FORBIDDEN = frozenset(
    {
        "rank",
        "score",
        "retriever",
        "retriever_name",
        "bm25",
        "dense",
        "baseline",
        "seed",
        "seed_evidence_ids",
        "sources",
        "judge_order",
    }
)


class CallLog:
    """Append-only JSONL. A completed input hash is not called again."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._ok: dict[str, dict[str, Any]] = {}
        self._attempts: dict[str, int] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            key = str(row.get("input_hash") or "")
            if not key:
                continue
            self._attempts[key] = self._attempts.get(key, 0) + 1
            if row.get("status") == "ok":
                self._ok[key] = row

    def get(self, input_hash: str) -> dict[str, Any] | None:
        return self._ok.get(input_hash)

    def attempts(self, input_hash: str) -> int:
        return self._attempts.get(input_hash, 0)

    def ok_count(self) -> int:
        return len(self._ok)

    def append(self, record: dict[str, Any]) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            key = str(record.get("input_hash") or "")
            self._attempts[key] = self._attempts.get(key, 0) + 1
            if record.get("status") == "ok":
                self._ok[key] = record


def input_hash(stage: str, model: str, payload: dict[str, Any]) -> str:
    body = {
        "stage": stage,
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "payload": payload,
    }
    return sha256_text(canonical_dumps(body))


def estimate_usd(
    model: str, input_tokens: int, output_tokens: int, prices: dict[str, Any]) -> float:
    rate = prices.get(model)
    if not isinstance(rate, dict):
        raise UsageError(f"no price estimate for {model}")
    return (
        input_tokens * float(rate["input"]) + output_tokens * float(rate["output"])
    ) / 1_000_000


def planned_cost(
    call_counts: dict[str, int],
    models: dict[str, str],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Upper-bound call count and estimated dollars before a stage starts."""
    tokens = config["estimated_tokens_per_call"]
    prices = config["prices_usd_per_million_tokens"]
    stage_model = {
        "generate": models["generator"],
        "decompose": models["decomposer"],
        "judge_a": models["judge_a"],
        "judge_b": models["judge_b"],
        "adjudicate": models["adjudicator"],
    }
    stage_token = {
        "generate": "generate",
        "decompose": "decompose",
        "judge_a": "judge",
        "judge_b": "judge",
        "adjudicate": "judge",
    }
    total_calls = 0
    total_usd = 0.0
    detail: dict[str, Any] = {}
    for stage, count in call_counts.items():
        token_key = stage_token[stage]
        model = stage_model[stage]
        calls = int(count)
        total_calls += calls
        usd = estimate_usd(
            model,
            calls * int(tokens[token_key]["input"]),
            calls * int(tokens[token_key]["output"]),
            prices,
        )
        total_usd += usd
        detail[stage] = {"calls": calls, "model": model, "estimated_usd": round(usd, 4)}
    return {
        "calls": total_calls,
        "estimated_usd": round(total_usd, 4),
        "price_source": config.get("price_source"),
        "stages": detail,
    }


def assert_budget(plan: dict[str, Any], config: dict[str, Any]) -> None:
    """Check the full-job estimate. Completed checkpoints are part of that job, not extra calls."""
    calls = int(plan["calls"])
    if calls > int(config["max_calls"]):
        raise UsageError(
            f"cost guard: {calls} calls exceed max_calls {config['max_calls']}"
        )
    if float(plan["estimated_usd"]) > float(config["max_estimated_usd"]):
        raise UsageError(
            f"cost guard: estimated ${plan['estimated_usd']} exceeds "
            f"max_estimated_usd {config['max_estimated_usd']}"
        )


class StructuredClient:
    """OpenAI Responses parse, or Gemini JSON. One call, one schema."""

    def __init__(self, *, provider: str, model: str, api_key: str, timeout: float) -> None:
        self.provider = provider
        self.model = model
        if not api_key:
            raise UsageError(f"{provider} API key is empty")
        self._timeout = timeout
        if provider == "openai":
            from openai import OpenAI

            self._openai = OpenAI(api_key=api_key, timeout=timeout, max_retries=0)
            self._http = None
        elif provider == "gemini":
            self._openai = None
            self._http = httpx.Client(timeout=timeout)
            self._gemini_key = api_key
        else:
            raise UsageError(f"unknown provider {provider}")

    def complete(
        self,
        *,
        instructions: str,
        payload: dict[str, Any],
        schema: type[BaseModel],
    ) -> tuple[dict[str, Any], dict[str, int] | None, float]:
        user_input = canonical_dumps(payload)
        started = time.perf_counter()
        try:
            if self.provider == "openai":
                parsed, usage = self._openai_parse(instructions, user_input, schema)
            else:
                parsed, usage = self._gemini_parse(instructions, user_input, schema)
        except UsageError:
            raise
        except Exception as exc:
            if self.provider == "openai":
                raise normalize_provider_error(exc) from None
            raise UsageError(f"gemini call failed: {type(exc).__name__}") from None
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        if not isinstance(parsed, schema):
            raise UsageError("structured output did not match the schema")
        return parsed.model_dump(), usage, latency_ms

    def _openai_parse(
        self,
        instructions: str,
        user_input: str,
        schema: type[BaseModel],
    ) -> tuple[BaseModel, dict[str, int] | None]:
        parsed = self._openai.responses.parse(
            model=self.model,
            instructions=instructions,
            input=user_input,
            text_format=schema,
            store=False,
            timeout=self._timeout,
        )
        output = getattr(parsed, "output_parsed", None)
        if not isinstance(output, schema):
            raise UsageError("model returned no structured output")
        usage = getattr(parsed, "usage", None)
        facts: dict[str, int] | None = None
        if usage is not None:
            facts = {}
            for name in ("input_tokens", "output_tokens", "total_tokens"):
                value = getattr(usage, name, None)
                if isinstance(value, int):
                    facts[name] = value
            facts = facts or None
        return output, facts

    def _gemini_parse(
        self,
        instructions: str,
        user_input: str,
        schema: type[BaseModel],
    ) -> tuple[BaseModel, dict[str, int] | None]:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        body = {
            "systemInstruction": {"parts": [{"text": instructions}]},
            "contents": [{"role": "user", "parts": [{"text": user_input}]}],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
                "responseSchema": _gemini_schema(schema),
            },
        }
        response = self._http.post(
            url,
            headers={"x-goog-api-key": self._gemini_key, "content-type": "application/json"},
            json=body,
        )
        if response.status_code != 200:
            raise UsageError(f"gemini HTTP {response.status_code}")
        data = response.json()
        candidates = data.get("candidates") or []
        if not candidates:
            raise UsageError("gemini returned no candidate")
        parts = ((candidates[0].get("content") or {}).get("parts") or [])
        text = ""
        for part in parts:
            if isinstance(part, dict) and part.get("text"):
                text = str(part["text"])
                break
        try:
            output = schema.model_validate(json.loads(_json_object(text)))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise UsageError("gemini JSON did not match the schema") from exc
        meta = data.get("usageMetadata") or {}
        facts = {}
        if isinstance(meta.get("promptTokenCount"), int):
            facts["input_tokens"] = meta["promptTokenCount"]
        if isinstance(meta.get("candidatesTokenCount"), int):
            facts["output_tokens"] = meta["candidatesTokenCount"]
        if isinstance(meta.get("totalTokenCount"), int):
            facts["total_tokens"] = meta["totalTokenCount"]
        return output, facts or None


def _gemini_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """Gemini JSON mode invents field names unless the object is pinned."""
    name = schema.__name__
    if name == "PassageJudgment":
        return {
            "type": "object",
            "properties": {
                "requirement_judgments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "requirement_id": {"type": "string"},
                            "relevance": {"type": "integer"},
                        },
                        "required": ["requirement_id", "relevance"],
                    },
                },
                "passage_relevance": {"type": "integer"},
            },
            "required": ["requirement_judgments", "passage_relevance"],
        }
    raise UsageError(f"no Gemini schema for {name}")


def _json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1]
        if stripped.endswith("```"):
            stripped = stripped[: stripped.rfind("```")]
    return stripped.strip()


def judge_payload(
    question: str,
    requirements: list[dict[str, Any]],
    candidate: dict[str, Any],
    *,
    text_limit: int = 2500,
) -> dict[str, Any]:
    """Fields a judge may see. Rank, score, and retriever identity stay out."""
    text = str(candidate.get("text") or "")
    payload = {
        "question": question,
        "requirements": [
            {
                "requirement_id": item["requirement_id"],
                "text": item["text"],
                "answerability": item["answerability"],
            }
            for item in requirements
        ],
        "evidence": {
            "document_name": candidate.get("document_name"),
            "filing_period": candidate.get("filing_period"),
            "section_path": list(candidate.get("section_path") or []),
            "receipt_date": candidate.get("receipt_date"),
            "text": text[:text_limit],
            "truncated": len(text) > text_limit,
        },
    }
    _reject_blind_keys(payload)
    return payload


def _reject_blind_keys(value: Any) -> None:
    if isinstance(value, dict):
        found = BLIND_FORBIDDEN & set(value)
        if found:
            raise UsageError(f"judge payload contains {sorted(found)}")
        for item in value.values():
            _reject_blind_keys(item)
    elif isinstance(value, list):
        for item in value:
            _reject_blind_keys(item)


def coerce_judgment(
    payload: dict[str, Any],
    requirement_ids: list[str],
) -> dict[str, Any] | None:
    """Return a normalized judgment, or None when the model output is not usable."""
    raw = payload.get("requirement_judgments")
    if not isinstance(raw, list):
        return None
    by_id: dict[str, int] = {}
    for item in raw:
        if not isinstance(item, dict):
            return None
        requirement_id = str(item.get("requirement_id") or "")
        relevance = item.get("relevance")
        if requirement_id not in requirement_ids or relevance not in {0, 1, 2}:
            return None
        if requirement_id in by_id:
            return None
        by_id[requirement_id] = int(relevance)
    if set(by_id) != set(requirement_ids):
        return None
    grades = [by_id[item] for item in requirement_ids]
    passage = max(grades) if grades else 0
    return {
        "requirement_judgments": [
            {"requirement_id": item, "relevance": by_id[item]} for item in requirement_ids
        ],
        "passage_relevance": passage,
        "passage_relevance_repaired": payload.get("passage_relevance") != passage,
    }


def judgments_agree(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if int(left["passage_relevance"]) != int(right["passage_relevance"]):
        return False
    left_map = {
        item["requirement_id"]: int(item["relevance"]) for item in left["requirement_judgments"]
    }
    right_map = {
        item["requirement_id"]: int(item["relevance"]) for item in right["requirement_judgments"]
    }
    return left_map == right_map

"""OpenAI grounded answers. The model writes prose. It does not create evidence.

Evidence ids, receipt numbers, and source locators come from the current run.
The client asks the Responses API for a schema-constrained object. Streaming
uses ``responses.stream`` and forwards claim text only after that claim's
``evidence_ids`` have closed. User-facing citation numbers are assigned
afterwards.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from disclosure_agent.agent.answer import GeneratedAnswer
from disclosure_agent.agent.citation import Citation, assign_citation_ids
from disclosure_agent.agent.context import evidence_prompt_payload
from disclosure_agent.agent.errors import (
    CitationValidationFailure,
    OpenAIAPIError,
    OpenAIAuthError,
    OpenAIRateLimitError,
    OpenAITimeoutError,
    StructuredOutputInvalid,
)
from disclosure_agent.agent.evidence import Evidence
from disclosure_agent.agent.settings import OpenAIConfig
from disclosure_agent.agent.stream_parse import PartialGrounded, parse_partial_grounded

# Keys that belong to evaluation or gold labels. They are never copied into the prompt.
FORBIDDEN_INPUT_KEYS = frozenset(
    {
        "gold",
        "gold_spans",
        "gold_answer",
        "evidence_class",
        "required_claim",
        "answer_valid_evidence",
        "question_id",
        "hard_negative",
        "dataset_id",
        "label",
        "known_conflict",
    }
)
_EVIDENCE_KEYS = (
    "evidence_id",
    "company",
    "report_name",
    "receipt_no",
    "filing_period",
    "receipt_date",
    "document_name",
    "section_path",
    "text",
)

ANSWER_INSTRUCTIONS = """당신은 금융 공시 근거에 묶인 답변만 작성합니다.

1. 입력 evidence의 text만 사실의 근거로 사용합니다.
2. evidence가 질문을 충분히 뒷받침하지 않으면 abstain을 true로 둡니다.
   부족한 점은 answer와 limitations에 적고, 없는 사실은 채우지 않습니다.
3. evidence에 없는 숫자, 원인, 날짜, 고유명사를 만들지 않습니다.
4. 공시가 직접 말한 내용과, 문장에서 제한적으로 읽을 수 있는 설명을 구분합니다.
   확실하지 않으면 단정하지 않습니다.
5. 각 claim의 evidence_ids를 text보다 먼저 적습니다.
   claim.text는 그 evidence_ids가 직접 뒷받침하는 내용만 담습니다.
   evidence_id는 입력 목록에 있는 값만 그대로 복사합니다.
6. evidence끼리 내용이 다르면 한쪽을 사실로 단정하지 않고 limitations에 적습니다.
7. 투자 조언, 매수, 매도, 목표가를 만들지 않습니다.
8. 사용자에게 보이는 문장은 claim.text입니다.
   [1], [2] 같은 인용 번호, evidence_id, URL을 문장에 넣지 않습니다.
9. 근거의 식별자는 evidence_id뿐입니다.
   company, report_name, receipt_no, filing_period, receipt_date, document_name은
   어느 공시의 어느 기간인지 구분하는 맥락이며 새 식별자로 쓰지 않습니다.
10. abstain이 true이면 확정된 사실을 주장하지 않습니다. claims의 evidence_ids는 비워도 됩니다.
    abstain이 false이면 모든 claim에 evidence_id가 하나 이상 있어야 합니다.
11. request_time은 Asia/Seoul 기준의 질문 시각입니다.
    question은 사용자가 입력한 원문이며, 시각을 넣기 위해 고치지 않습니다.
12. corpus_as_of는 색인에 들어 있는 공시의 기준 시각입니다.
    그 시각 이후의 공시가 있는 것처럼 말하지 않습니다.
"""


class GroundedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_ids: list[str] = Field(
        description="evidence_id values copied from the input. Never invent an id."
    )
    text: str = Field(description="A statement supported directly by this claim's evidence ids.")


class GroundedModelOutput(BaseModel):
    """Schema sent to the Responses API. Field order is the generation order."""

    model_config = ConfigDict(extra="forbid")

    abstain: bool = Field(description="True when the evidence does not support a firm answer.")
    claims: list[GroundedClaim]
    limitations: list[str]


@dataclass(frozen=True)
class ProviderSnapshot:
    """One ``response.output_text.delta`` accumulation."""

    snapshot: str


@dataclass(frozen=True)
class ProviderDone:
    parsed: GroundedModelOutput
    usage: dict[str, int] | None


@dataclass(frozen=True)
class AnswerDelta:
    text: str


@dataclass(frozen=True)
class AnswerCitationEvent:
    citation: Citation


@dataclass(frozen=True)
class AnswerFinished:
    answer: GeneratedAnswer


@dataclass(frozen=True)
class ToolCompletion:
    """One Responses tool-call turn. ``function_arguments`` is raw JSON text."""

    function_name: str | None
    function_arguments: str | None
    usage: dict[str, int] | None
    latency_ms: float


class AnswerClient(Protocol):
    model: str
    sdk_version: str

    def complete(self, user_input: str) -> GroundedModelOutput: ...


class OpenAIAnswerClient:
    """Responses API client. ``stream`` forwards output-text snapshots as they arrive."""

    def __init__(self, config: OpenAIConfig, *, sdk: Any | None = None) -> None:
        self.model = config.model
        self.timeout_seconds = config.timeout_seconds
        self.sdk_version = _sdk_version()
        self.last_latency_ms: float | None = None
        self.last_usage: dict[str, int] | None = None
        if sdk is not None:
            self._sdk = sdk
            return
        if not config.api_key:
            raise OpenAIAuthError("OPENAI_API_KEY is not configured")
        from openai import OpenAI

        self._sdk = OpenAI(
            api_key=config.api_key,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
        )

    def complete(self, user_input: str) -> GroundedModelOutput:
        """Parse one structured answer. Latency covers request start through parse."""
        self.last_latency_ms = None
        self.last_usage = None
        started = time.perf_counter()
        try:
            parsed = self._sdk.responses.parse(
                model=self.model,
                instructions=ANSWER_INSTRUCTIONS,
                input=user_input,
                text_format=GroundedModelOutput,
                store=False,
                timeout=self.timeout_seconds,
            )
        except Exception as exc:
            raise normalize_provider_error(exc) from None
        finally:
            self.last_latency_ms = round((time.perf_counter() - started) * 1000, 3)
        self.last_usage = _usage_facts(parsed)
        output = getattr(parsed, "output_parsed", None)
        if not isinstance(output, GroundedModelOutput):
            raise StructuredOutputInvalid("model returned no structured answer")
        return output

    def stream(self, user_input: str) -> Iterator[ProviderSnapshot | ProviderDone]:
        """Yield output-text snapshots, then the parsed object after ``response.completed``.

        Event names are the Responses streaming types exposed by the installed SDK:
        ``response.output_text.delta``, ``response.output_text.done``,
        ``response.completed``, ``response.failed``, ``response.incomplete``, and ``error``.
        """
        self.last_latency_ms = None
        self.last_usage = None
        started = time.perf_counter()
        parsed: GroundedModelOutput | None = None
        usage: dict[str, int] | None = None
        completed = False
        try:
            with self._sdk.responses.stream(
                model=self.model,
                instructions=ANSWER_INSTRUCTIONS,
                input=user_input,
                text_format=GroundedModelOutput,
                store=False,
                timeout=self.timeout_seconds,
            ) as events:
                for event in events:
                    event_type = getattr(event, "type", "")
                    if event_type == "response.output_text.delta":
                        yield ProviderSnapshot(snapshot=str(event.snapshot))
                    elif event_type == "response.output_text.done":
                        parsed = _done_output(event)
                    elif event_type == "response.completed":
                        completed = True
                        usage = _usage_facts(getattr(event, "response", None))
                        if parsed is None:
                            response = getattr(event, "response", None)
                            output = getattr(response, "output_parsed", None)
                            if isinstance(output, GroundedModelOutput):
                                parsed = output
                    elif event_type in {"response.failed", "error"}:
                        raise OpenAIAPIError(event_type)
                    elif event_type == "response.incomplete":
                        raise StructuredOutputInvalid("structured output was incomplete")
        except Exception as exc:
            raise normalize_provider_error(exc) from None
        finally:
            self.last_latency_ms = round((time.perf_counter() - started) * 1000, 3)
        if not completed or not isinstance(parsed, GroundedModelOutput):
            raise StructuredOutputInvalid("model returned no structured answer")
        self.last_usage = usage
        yield ProviderDone(parsed=parsed, usage=usage)

    def complete_tool(
        self,
        user_input: str,
        *,
        instructions: str,
        tools: list[dict[str, Any]],
        tool_name: str | None,
    ) -> ToolCompletion:
        """Ask for one function call. The answer schema is not used here."""
        self.last_latency_ms = None
        self.last_usage = None
        started = time.perf_counter()
        try:
            response = self._sdk.responses.create(
                model=self.model,
                instructions=instructions,
                input=user_input,
                tools=tools,
                tool_choice="auto",
                max_output_tokens=600,
                store=False,
                timeout=self.timeout_seconds,
            )
        except Exception as exc:
            raise normalize_provider_error(exc) from None
        finally:
            self.last_latency_ms = round((time.perf_counter() - started) * 1000, 3)
        self.last_usage = _usage_facts(response)
        name, arguments = _first_function_call(response, tool_name)
        return ToolCompletion(
            function_name=name,
            function_arguments=arguments,
            usage=self.last_usage,
            latency_ms=self.last_latency_ms or 0.0,
        )


class OpenAIAnswerGenerator:
    """Ask the configured model to write from the current evidence set only."""

    generator_type = "openai"

    def __init__(self, client: AnswerClient) -> None:
        self.client = client
        self.model = client.model
        self.sdk_version = client.sdk_version
        self.last_generation_ms: float | None = None
        self.last_usage: dict[str, int] | None = None

    def generate(
        self,
        question: str,
        evidence: list[Evidence],
        *,
        company_name: str = "",
        fiscal_year: int | None = None,
        request_time: dict[str, str] | None = None,
        corpus_as_of: str | None = None,
    ) -> GeneratedAnswer:
        payload = build_llm_input(
            question,
            evidence,
            company_name=company_name,
            fiscal_year=fiscal_year,
            request_time=request_time,
            corpus_as_of=corpus_as_of,
        )
        self.last_generation_ms = None
        self.last_usage = None
        try:
            parsed = self.client.complete(json.dumps(payload, ensure_ascii=False))
        finally:
            latency = getattr(self.client, "last_latency_ms", None)
            if isinstance(latency, (int, float)):
                self.last_generation_ms = float(latency)
            usage = getattr(self.client, "last_usage", None)
            if isinstance(usage, dict):
                facts = {
                    key: usage[key]
                    for key in ("input_tokens", "output_tokens", "total_tokens")
                    if isinstance(usage.get(key), int)
                }
                self.last_usage = facts or None
        answer = grounded_to_answer(parsed, {item.evidence_id for item in evidence})
        answer.usage = self.last_usage
        return answer

    def iter_answer(
        self,
        question: str,
        evidence: list[Evidence],
        *,
        company_name: str = "",
        fiscal_year: int | None = None,
        request_time: dict[str, str] | None = None,
        corpus_as_of: str | None = None,
    ) -> Iterator[AnswerDelta | AnswerCitationEvent | AnswerFinished]:
        """Yield claim text as the provider streams it, or one delta when it cannot.

        A client without ``stream`` has already finished the object. That path
        emits the claim text once. It does not slice the finished string.
        """
        allowed = {item.evidence_id for item in evidence}
        stream = getattr(self.client, "stream", None)
        if callable(stream):
            payload = json.dumps(
                build_llm_input(
                    question,
                    evidence,
                    company_name=company_name,
                    fiscal_year=fiscal_year,
                    request_time=request_time,
                    corpus_as_of=corpus_as_of,
                ),
                ensure_ascii=False,
            )
            yield from self._iter_stream(payload, allowed)
            return
        answer = self.generate(
            question,
            evidence,
            company_name=company_name,
            fiscal_year=fiscal_year,
            request_time=request_time,
            corpus_as_of=corpus_as_of,
        )
        if answer.text:
            yield AnswerDelta(answer.text)
        for citation in answer.citations:
            yield AnswerCitationEvent(citation)
        yield AnswerFinished(answer)

    def _iter_stream(
        self,
        payload: str,
        allowed: set[str],
    ) -> Iterator[AnswerDelta | AnswerCitationEvent | AnswerFinished]:
        self.last_generation_ms = None
        self.last_usage = None
        state = _StreamState()
        try:
            for piece in self.client.stream(payload):
                if isinstance(piece, ProviderSnapshot):
                    yield from _events_for_snapshot(
                        parse_partial_grounded(piece.snapshot),
                        allowed,
                        state,
                    )
                elif isinstance(piece, ProviderDone):
                    yield from _events_for_snapshot(
                        parse_partial_grounded(_render_grounded(piece.parsed)),
                        allowed,
                        state,
                    )
                    answer = _finish_stream(piece.parsed, state)
                    if not state.emitted and answer.text:
                        yield AnswerDelta(answer.text)
                    answer.usage = piece.usage
                    self.last_usage = piece.usage
                    yield AnswerFinished(answer)
        finally:
            latency = getattr(self.client, "last_latency_ms", None)
            if isinstance(latency, (int, float)):
                self.last_generation_ms = float(latency)


def build_llm_input(
    question: str,
    evidence: list[Evidence],
    *,
    company_name: str,
    fiscal_year: int | None,
    request_time: dict[str, str] | None = None,
    corpus_as_of: str | None = None,
) -> dict[str, Any]:
    """The only object sent as the model input. Evaluation labels are not copied.

    ``question`` is the raw user text. ``request_time`` and ``corpus_as_of``
    are separate clocks and are not written into the question.
    """
    payload: dict[str, Any] = {
        "question": question,
        "company_name": company_name,
        "fiscal_year": fiscal_year,
        "evidence": [_evidence_input(item) for item in evidence],
    }
    if request_time is not None:
        payload["request_time"] = {
            "timezone": request_time["timezone"],
            "display": request_time["display"],
            "iso": request_time["iso"],
            "weekday": request_time["weekday"],
        }
    if corpus_as_of is not None:
        payload["corpus_as_of"] = corpus_as_of
    _reject_forbidden_keys(payload)
    return payload


def grounded_to_answer(parsed: GroundedModelOutput, allowed: set[str]) -> GeneratedAnswer:
    referenced = _referenced_ids(parsed, allowed)
    texts = [claim.text.strip() for claim in parsed.claims if claim.text.strip()]
    if parsed.abstain:
        text = "\n".join(texts) or "검색된 공시에서 요청한 내용을 확인하지 못했습니다."
        return GeneratedAnswer(
            text=text,
            citations=[],
            limitations=list(parsed.limitations),
            abstain=True,
            generator_type="openai",
        )
    if not texts:
        raise StructuredOutputInvalid("answer text is empty")
    if not referenced:
        raise CitationValidationFailure("disclosure answer has no current-run citation")
    return GeneratedAnswer(
        text="\n".join(texts),
        citations=assign_citation_ids(referenced),
        limitations=list(parsed.limitations),
        abstain=False,
        generator_type="openai",
    )


def _referenced_ids(parsed: GroundedModelOutput, allowed: set[str]) -> list[str]:
    referenced: list[str] = []
    for claim in parsed.claims:
        if not claim.evidence_ids and not parsed.abstain:
            raise CitationValidationFailure("claim has no evidence_id")
        for evidence_id in claim.evidence_ids:
            token = evidence_id.strip()
            if not token or token not in allowed:
                raise CitationValidationFailure(
                    f"claim references {token or '(empty)'}, which is not in the current run"
                )
            if not parsed.abstain:
                referenced.append(token)
    return referenced


def normalize_provider_error(exc: BaseException) -> Exception:
    """Map SDK failures onto agent error codes. The original message is dropped."""
    if isinstance(exc, (StructuredOutputInvalid, CitationValidationFailure)):
        return exc
    if isinstance(exc, ValidationError):
        return StructuredOutputInvalid("structured output did not match the schema")

    import openai

    if isinstance(exc, openai.APITimeoutError):
        return OpenAITimeoutError("openai request timed out")
    if isinstance(exc, openai.AuthenticationError):
        return OpenAIAuthError("openai authentication failed")
    if isinstance(exc, openai.RateLimitError):
        return OpenAIRateLimitError("openai rate limit")
    if isinstance(exc, openai.LengthFinishReasonError):
        return StructuredOutputInvalid("structured output was incomplete")
    if isinstance(exc, openai.APIError):
        status = getattr(exc, "status_code", None)
        detail = type(exc).__name__ if status is None else f"{type(exc).__name__} status={status}"
        return OpenAIAPIError(detail)
    if isinstance(exc, openai.OpenAIError):
        return OpenAIAPIError(type(exc).__name__)
    return exc


def _evidence_input(item: Evidence) -> dict[str, Any]:
    payload = evidence_prompt_payload(item)
    return {key: payload[key] for key in _EVIDENCE_KEYS}


def _reject_forbidden_keys(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in FORBIDDEN_INPUT_KEYS or str(key).startswith("gold"):
                raise ValueError(f"refusing to send {key} to the answer model")
            _reject_forbidden_keys(item)
    elif isinstance(value, list):
        for item in value:
            _reject_forbidden_keys(item)


def _usage_facts(parsed: object) -> dict[str, int] | None:
    """Copy token counts the response actually carries. Do not derive a missing total."""
    usage = getattr(parsed, "usage", None)
    if usage is None:
        return None
    facts: dict[str, int] = {}
    for name in ("input_tokens", "output_tokens", "total_tokens"):
        value = getattr(usage, name, None)
        if isinstance(value, int):
            facts[name] = value
    return facts or None


def _first_function_call(
    response: object,
    tool_name: str | None,
) -> tuple[str | None, str | None]:
    output = getattr(response, "output", None) or []
    for item in output:
        if getattr(item, "type", None) != "function_call":
            continue
        name = getattr(item, "name", None)
        if tool_name and name != tool_name:
            continue
        arguments = getattr(item, "arguments", None)
        if arguments is None:
            return str(name), None
        return str(name), str(arguments)
    return None, None


def _sdk_version() -> str:
    try:
        return version("openai")
    except Exception:
        return "unknown"


class _StreamState:
    def __init__(self) -> None:
        self.validated: set[int] = set()
        self.sent_text: dict[int, int] = {}
        self.cited: set[int] = set()
        self.seen: dict[str, str] = {}
        self.emitted: list[str] = []
        self.any_text = False


def _events_for_snapshot(
    progress: PartialGrounded,
    allowed: set[str],
    state: _StreamState,
) -> Iterator[AnswerDelta | AnswerCitationEvent]:
    """Emit new claim text only after that claim's evidence ids have been checked."""
    if progress.abstain is None:
        return
    for index, claim in enumerate(progress.claims):
        if not claim.ids_closed or claim.evidence_ids is None:
            continue
        if index not in state.validated:
            _check_claim_ids(claim.evidence_ids, abstain=progress.abstain, allowed=allowed)
            state.validated.add(index)
        sent = state.sent_text.get(index, 0)
        if len(claim.text) < sent:
            raise StructuredOutputInvalid("claim text shrank")
        new = claim.text[sent:]
        if new:
            if sent == 0 and state.any_text:
                state.emitted.append("\n")
                yield AnswerDelta("\n")
            state.emitted.append(new)
            state.any_text = True
            state.sent_text[index] = len(claim.text)
            yield AnswerDelta(new)
        if claim.closed and index not in state.cited:
            if not claim.text.strip() and not progress.abstain:
                raise StructuredOutputInvalid("answer text is empty")
            if not progress.abstain:
                for citation in _fresh_citations(claim.evidence_ids, state.seen):
                    yield AnswerCitationEvent(citation)
            state.cited.add(index)


def _check_claim_ids(
    evidence_ids: tuple[str, ...] | list[str],
    *,
    abstain: bool,
    allowed: set[str],
) -> None:
    if not abstain and not evidence_ids:
        raise CitationValidationFailure("claim has no evidence_id")
    for evidence_id in evidence_ids:
        token = evidence_id.strip()
        if not token or token not in allowed:
            raise CitationValidationFailure(
                f"claim references {token or '(empty)'}, which is not in the current run"
            )


def _fresh_citations(
    evidence_ids: tuple[str, ...] | list[str],
    seen: dict[str, str],
) -> list[Citation]:
    fresh: list[Citation] = []
    for evidence_id in evidence_ids:
        token = evidence_id.strip()
        if not token or token in seen:
            continue
        citation_id = str(len(seen) + 1)
        seen[token] = citation_id
        fresh.append(Citation(citation_id=citation_id, evidence_id=token))
    return fresh


def _finish_stream(parsed: GroundedModelOutput, state: _StreamState) -> GeneratedAnswer:
    text = "".join(state.emitted)
    if parsed.abstain:
        return GeneratedAnswer(
            text=text or "검색된 공시에서 요청한 내용을 확인하지 못했습니다.",
            citations=[],
            limitations=list(parsed.limitations),
            abstain=True,
            generator_type="openai",
        )
    if not text:
        raise StructuredOutputInvalid("answer text is empty")
    citations = [
        Citation(citation_id=citation_id, evidence_id=evidence_id)
        for evidence_id, citation_id in state.seen.items()
    ]
    if not citations:
        raise CitationValidationFailure("disclosure answer has no current-run citation")
    return GeneratedAnswer(
        text=text,
        citations=citations,
        limitations=list(parsed.limitations),
        abstain=False,
        generator_type="openai",
    )


def _render_grounded(parsed: GroundedModelOutput) -> str:
    return parsed.model_dump_json()


def _done_output(event: object) -> GroundedModelOutput:
    parsed = getattr(event, "parsed", None)
    if isinstance(parsed, GroundedModelOutput):
        return parsed
    raw = getattr(event, "text", None)
    if isinstance(raw, str) and raw:
        try:
            return GroundedModelOutput.model_validate_json(raw)
        except ValidationError:
            raise StructuredOutputInvalid("structured output did not match the schema") from None
    raise StructuredOutputInvalid("model returned no structured answer")

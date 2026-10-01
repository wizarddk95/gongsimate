"""One-question agent run: resolve a company, optionally search, then cite."""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from disclosure_agent.agent.answer import (
    EvidenceBoundAnswerGenerator,
    GeneratedAnswer,
    answer_markers_match,
    company_followup,
    direct_answer,
)
from disclosure_agent.agent.catalog import load_dense_d2a_binding, repo_root
from disclosure_agent.agent.citation import CitationValidator
from disclosure_agent.agent.company import (
    CompanyRecord,
    CompanyRegistry,
    Resolution,
    clarification_question,
    is_disclosure_question,
    is_smalltalk,
    mentioned_years,
    resolution_to_dict,
)
from disclosure_agent.agent.errors import (
    AgentFlowError,
    CitationValidationFailure,
    LlmLimitExceeded,
    RequestTimeout,
    SearchLimitExceeded,
    user_message,
)
from disclosure_agent.agent.evidence import Evidence, EvidenceBuilder
from disclosure_agent.agent.record import RunLog
from disclosure_agent.agent.search import (
    DenseD2aSearch,
    SearchDisclosuresRequest,
    SearchDisclosuresResult,
)
from disclosure_agent.agent.settings import (
    DEFAULT_TOP_K,
    REQUEST_TIMEOUT_SECONDS,
    AgentLimits,
)
from disclosure_agent.agent.sse import StreamEvent

SearchFn = Callable[[SearchDisclosuresRequest], SearchDisclosuresResult]
DELTA_CHARS = 24


@dataclass
class ClarificationContext:
    prior_message: str | None = None
    selected_company_name: str | None = None


class AgentRuntime:
    def __init__(
        self,
        *,
        registry: CompanyRegistry,
        search: SearchFn,
        evidence_builder: EvidenceBuilder,
        answer_generator: EvidenceBoundAnswerGenerator | None = None,
        citation_validator: CitationValidator | None = None,
        limits: AgentLimits | None = None,
        run_log: RunLog | None = None,
        timeout_seconds: float = REQUEST_TIMEOUT_SECONDS,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self.registry = registry
        self.search = search
        self.evidence_builder = evidence_builder
        self.answer_generator = answer_generator or EvidenceBoundAnswerGenerator()
        self.citation_validator = citation_validator or CitationValidator()
        self.limits = limits or AgentLimits()
        self.run_log = run_log
        self.timeout_seconds = timeout_seconds
        self.top_k = top_k

    def run(
        self,
        message: str,
        *,
        conversation_id: str | None = None,
        clarification: ClarificationContext | None = None,
    ) -> dict[str, Any]:
        """Collect the stream. Useful for tests that do not need HTTP framing."""
        events = list(
            self.stream(
                message,
                conversation_id=conversation_id,
                clarification=clarification,
            )
        )
        terminal = events[-1]
        return {"events": events, "terminal": terminal}

    def stream(
        self,
        message: str,
        *,
        conversation_id: str | None = None,
        clarification: ClarificationContext | None = None,
    ) -> Iterator[StreamEvent]:
        started = time.perf_counter()
        run_id = uuid4().hex
        budget = _Budget(self.limits)
        record: dict[str, Any] = {
            "run_id": run_id,
            "question": message,
            "conversation_id": conversation_id,
            "resolved_company": None,
            "agent_action": None,
            "tool_calls": [],
            "evidence_ids": [],
            "citation_ids": [],
            "latency_ms": None,
            "status": None,
            "error": None,
        }
        try:
            yield StreamEvent("run.started", {"run_id": run_id})
            yield from self._run_body(
                message,
                clarification=clarification,
                budget=budget,
                record=record,
                started=started,
            )
        except AgentFlowError as exc:
            record["status"] = exc.code
            record["error"] = {"code": exc.code, "detail": exc.detail}
            yield StreamEvent(
                "error",
                {"run_id": run_id, "code": exc.code, "message": user_message(exc.code)},
            )
        except Exception as exc:
            record["status"] = "internal_error"
            record["error"] = {
                "code": "internal_error",
                "detail": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            yield StreamEvent(
                "error",
                {
                    "run_id": run_id,
                    "code": "internal_error",
                    "message": user_message("internal_error"),
                },
            )
        finally:
            record["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
            if self.run_log is not None:
                self.run_log.append(record)

    def _run_body(
        self,
        message: str,
        *,
        clarification: ClarificationContext | None,
        budget: _Budget,
        record: dict[str, Any],
        started: float,
    ) -> Iterator[StreamEvent]:
        self._check_timeout(started)
        yield StreamEvent("agent.status", {"status": "resolving_company"})
        question, resolution = self._resolve(message, clarification, budget)
        record["resolved_company"] = resolution_to_dict(resolution)
        record["question"] = question

        if is_smalltalk(message) and clarification is None:
            answer = self._generate_direct(message, budget)
            yield from self._emit_answer(
                answer,
                evidence=[],
                action="answer_directly",
                status="completed",
                record=record,
                started=started,
            )
            return

        if resolution.status == "ambiguous":
            yield from self._emit_clarification(
                resolution,
                status="company_ambiguous",
                record=record,
                started=started,
            )
            return

        if resolution.status == "not_found":
            text = user_message("company_not_found") + " 회사명을 다시 확인해 주세요."
            yield from self._emit_clarification(
                resolution,
                status="company_not_found",
                record=record,
                started=started,
                text=text,
            )
            return

        if resolution.status == "unsupported":
            name = resolution.company_name or "해당 기업"
            text = (
                f"{name} 기업은 목록에서 확인했지만, "
                "현재 검색할 수 있는 공시가 없습니다."
            )
            answer = GeneratedAnswer(text=text, abstain=True, limitations=[text])
            yield from self._emit_answer(
                answer,
                evidence=[],
                action="abstain",
                status="company_unsupported",
                record=record,
                started=started,
            )
            return

        if resolution.status != "resolved":
            if is_disclosure_question(question):
                yield from self._emit_clarification(
                    Resolution(status="not_found"),
                    status="company_not_found",
                    record=record,
                    started=started,
                    text="어느 기업의 공시를 볼까요? 회사명을 함께 입력해 주세요.",
                )
                return
            answer = self._generate_direct(question, budget)
            yield from self._emit_answer(
                answer,
                evidence=[],
                action="answer_directly",
                status="completed",
                record=record,
                started=started,
            )
            return

        if not is_disclosure_question(question):
            answer = company_followup(resolution.company_name or "해당 기업")
            budget.charge_llm()
            yield from self._emit_answer(
                answer,
                evidence=[],
                action="answer_directly",
                status="completed",
                record=record,
                started=started,
            )
            return

        self._check_timeout(started)
        budget.charge_search()
        yield StreamEvent("agent.status", {"status": "searching_disclosures"})
        years = mentioned_years(question)
        fiscal_year = int(years[0]) if years else None
        request = SearchDisclosuresRequest(
            query=question,
            company_id=resolution.company_id or "",
            corp_code=resolution.corp_code,
            fiscal_year=fiscal_year,
            top_k=self.top_k,
        )
        search_started = time.perf_counter()
        result = self.search(request)
        search_ms = round((time.perf_counter() - search_started) * 1000, 3)
        evidence = self.evidence_builder.build(
            result.hits,
            company_name=resolution.company_name or "",
        )
        record["tool_calls"].append(
            {
                "name": "search_disclosures",
                "query": request.query,
                "company_id": request.company_id,
                "corp_code": request.corp_code,
                "fiscal_year": request.fiscal_year,
                "top_k": request.top_k,
                "retriever_type": result.retriever_type,
                "representation_version": result.representation_version,
                "embedding_model": result.embedding_model,
                "retrieval_unit_ids": [hit.retrieval_unit_id for hit in result.hits],
                "evidence_ids": [item.evidence_id for item in evidence],
                "latency_ms": search_ms,
            }
        )
        record["evidence_ids"] = [item.evidence_id for item in evidence]
        self._check_timeout(started)
        if not evidence:
            answer = GeneratedAnswer(
                text="검색된 공시 근거가 없습니다. 확인되지 않은 내용은 답하지 않습니다.",
                limitations=["검색 결과가 비어 있습니다."],
                abstain=True,
            )
            yield from self._emit_answer(
                answer,
                evidence=[],
                action="abstain",
                status="retrieval_no_result",
                record=record,
                started=started,
            )
            return

        budget.charge_llm()
        answer = self.answer_generator.generate(question, evidence)
        self._validate_answer(answer, evidence)
        action = "abstain" if answer.abstain else "search_disclosures"
        status = "abstained" if answer.abstain else "completed"
        yield from self._emit_answer(
            answer,
            evidence=evidence,
            action=action,
            status=status,
            record=record,
            started=started,
        )

    def _resolve(
        self,
        message: str,
        clarification: ClarificationContext | None,
        budget: _Budget,
    ) -> tuple[str, Resolution]:
        if clarification and clarification.selected_company_name:
            question = clarification.prior_message or message
            budget.charge_llm()
            resolution = self.registry.resolve_mention(clarification.selected_company_name)
            return question, resolution
        if is_smalltalk(message):
            return message, Resolution(status="no_mention")
        budget.charge_llm()
        mentions = self.registry.extract_mentions(message)
        return message, self.registry.resolve_mentions(mentions)

    def _generate_direct(self, message: str, budget: _Budget) -> GeneratedAnswer:
        budget.charge_llm()
        return direct_answer(message)

    def _validate_answer(self, answer: GeneratedAnswer, evidence: list[Evidence]) -> None:
        evidence_ids = {item.evidence_id for item in evidence}
        if answer.abstain:
            if answer.citations:
                self.citation_validator.validate(answer.citations, evidence_ids)
            if "[" in answer.text and "]" in answer.text:
                raise CitationValidationFailure("abstain answer contains a citation marker")
            return
        if not answer.citations:
            raise CitationValidationFailure("disclosure answer has no current-run citation")
        if not answer_markers_match(answer):
            raise CitationValidationFailure("citation markers do not match citation ids")
        self.citation_validator.validate(answer.citations, evidence_ids)
        markers = [citation.citation_id for citation in answer.citations]
        if not markers:
            raise CitationValidationFailure("disclosure answer has no current-run citation")

    def _emit_answer(
        self,
        answer: GeneratedAnswer,
        *,
        evidence: list[Evidence],
        action: str,
        status: str,
        record: dict[str, Any],
        started: float,
    ) -> Iterator[StreamEvent]:
        self._check_timeout(started)
        shown = [] if answer.abstain else evidence
        by_id = {item.evidence_id: item for item in shown}
        record["agent_action"] = action
        record["status"] = status
        record["citation_ids"] = [item.citation_id for item in answer.citations]
        yield StreamEvent("agent.status", {"status": "generating_answer"})
        yield from _deltas(answer.text)
        for citation in answer.citations:
            item = by_id.get(citation.evidence_id)
            if item is None:
                raise CitationValidationFailure("citation is outside the current run")
            payload = item.to_dict()
            payload["citation_id"] = citation.citation_id
            yield StreamEvent("citation", payload)
        yield StreamEvent(
            "run.completed",
            _completed_payload(
                record=record,
                answer=answer,
                evidence=shown,
                action=action,
                status=status,
            ),
        )

    def _emit_clarification(
        self,
        resolution: Resolution,
        *,
        status: str,
        record: dict[str, Any],
        started: float,
        text: str | None = None,
    ) -> Iterator[StreamEvent]:
        self._check_timeout(started)
        question = text or clarification_question(resolution)
        record["agent_action"] = "ask_clarification"
        record["status"] = status
        yield from _deltas(question)
        yield StreamEvent(
            "run.completed",
            {
                "run_id": record["run_id"],
                "action": "ask_clarification",
                "status": status,
                "answer": {"text": question, "citations": [], "limitations": []},
                "evidence": [],
                "resolved_company": resolution_to_dict(resolution),
                "clarification": {
                    "question": question,
                    "candidates": resolution.public_candidates(),
                    "candidate_total": resolution.candidate_total,
                },
            },
        )

    def _check_timeout(self, started: float) -> None:
        if time.perf_counter() - started > self.timeout_seconds:
            raise RequestTimeout("agent run exceeded the request timeout")


class _Budget:
    def __init__(self, limits: AgentLimits) -> None:
        self.limits = limits
        self.search_calls = 0
        self.llm_calls = 0

    def charge_llm(self) -> None:
        if self.llm_calls >= self.limits.max_llm_calls:
            raise LlmLimitExceeded(
                f"llm calls {self.llm_calls} reached max {self.limits.max_llm_calls}"
            )
        self.llm_calls += 1

    def charge_search(self) -> None:
        if self.search_calls >= self.limits.max_search_calls:
            raise SearchLimitExceeded(
                f"search calls {self.search_calls} reached max {self.limits.max_search_calls}"
            )
        self.search_calls += 1


def _deltas(text: str) -> Iterator[StreamEvent]:
    if not text:
        return
    for offset in range(0, len(text), DELTA_CHARS):
        yield StreamEvent("answer.delta", {"text": text[offset : offset + DELTA_CHARS]})


def _completed_payload(
    *,
    record: dict[str, Any],
    answer: GeneratedAnswer,
    evidence: list[Evidence],
    action: str,
    status: str,
) -> dict[str, Any]:
    return {
        "run_id": record["run_id"],
        "action": action,
        "status": status,
        "answer": {
            "text": answer.text,
            "citations": [item.to_dict() for item in answer.citations],
            "limitations": list(answer.limitations),
        },
        "evidence": [item.to_dict() for item in evidence],
        "resolved_company": record["resolved_company"],
        "clarification": None,
    }


def build_default_runtime(
    root: Path | None = None,
    *,
    run_log_path: Path | None = None,
) -> AgentRuntime:
    """Wire the cached corp list and the locked Dense-D2a index. Does not encode documents."""
    base = root or repo_root()
    binding = load_dense_d2a_binding(base)
    registry = CompanyRegistry.from_corp_cache(
        base / "data" / "cache",
        indexed_corp_codes={binding.corp_code},
        fallback=[
            CompanyRecord(
                corp_code=binding.corp_code,
                corp_name=binding.company_name,
                stock_code="",
            )
        ],
    )
    log_path = run_log_path or (base / "data" / "product" / "runs" / "agent_runs.jsonl")
    return AgentRuntime(
        registry=registry,
        search=DenseD2aSearch(binding),
        evidence_builder=EvidenceBuilder.from_elements_path(binding.elements_path),
        run_log=RunLog(log_path),
    )

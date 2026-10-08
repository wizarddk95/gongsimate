"""One-question agent run: resolve a company, optionally search, then cite."""

from __future__ import annotations

import json
import re
import time
import traceback
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from disclosure_agent.agent.answer import (
    EvidenceBoundAnswerGenerator,
    GeneratedAnswer,
    answer_has_markers,
    answer_markers_match,
    company_followup,
    direct_answer,
)
from disclosure_agent.agent.catalog import repo_root
from disclosure_agent.agent.citation import CitationValidator
from disclosure_agent.agent.claim_class import classify_product_claim
from disclosure_agent.agent.clock import request_time_context, seoul_now
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
from disclosure_agent.agent.context import ContextBuilder
from disclosure_agent.agent.errors import (
    AgentFlowError,
    CitationValidationFailure,
    LlmLimitExceeded,
    RequestTimeout,
    RetrievalFailure,
    SearchLimitExceeded,
    StructuredOutputInvalid,
    redact_text,
    user_message,
)
from disclosure_agent.agent.evidence import Evidence, EvidenceBuilder
from disclosure_agent.agent.filing_relation import FilingRelation
from disclosure_agent.agent.llm import AnswerCitationEvent, AnswerDelta, AnswerFinished
from disclosure_agent.agent.planner import (
    OpenAISearchPlanner,
    PlannerContext,
    RawQueryPlanner,
    SearchPlanner,
)
from disclosure_agent.agent.record import RunLog
from disclosure_agent.agent.search import (
    SearchDisclosuresRequest,
    SearchDisclosuresResult,
    SearchHit,
)
from disclosure_agent.agent.search_catalog import (
    SearchCatalog,
    search_scope_record,
)
from disclosure_agent.agent.settings import (
    DEFAULT_GENERATION_CONTEXT_TOKEN_BUDGET,
    DEFAULT_GENERATION_MAX_EVIDENCE,
    DEFAULT_RETRIEVAL_TOP_K,
    REQUEST_TIMEOUT_SECONDS,
    STRUCTURED_OUTPUT_ATTEMPTS,
    AgentLimits,
)
from disclosure_agent.agent.sse import StreamEvent
from disclosure_agent.agent.temporal import FilingPeriod, period_label
from disclosure_agent.agent.tokens import CONTEXT_TOKEN_COUNTER_ID
from disclosure_agent.agent.tool_args import (
    TOOL_NAME,
    ToolResolution,
    arguments_record,
    filing_public_row,
    public_tool_observation,
    resolve_search_call,
)

SearchFn = Callable[[SearchDisclosuresRequest], SearchDisclosuresResult]


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
        retrieval_top_k: int = DEFAULT_RETRIEVAL_TOP_K,
        context_token_budget: int = DEFAULT_GENERATION_CONTEXT_TOKEN_BUDGET,
        generation_max_evidence: int = DEFAULT_GENERATION_MAX_EVIDENCE,
        filings: tuple[FilingPeriod, ...] = (),
        corpus_as_of: datetime | None = None,
        clock: Callable[[], datetime] | None = None,
        redact_secrets: tuple[str, ...] = (),
        planner: SearchPlanner | None = None,
        filing_relations: tuple[FilingRelation, ...] = (),
        catalog: SearchCatalog | None = None,
    ) -> None:
        if retrieval_top_k < 1:
            raise ValueError("retrieval_top_k must be >= 1")
        self.registry = registry
        self.search = search
        self.evidence_builder = evidence_builder
        self.answer_generator = answer_generator or EvidenceBoundAnswerGenerator()
        self.citation_validator = citation_validator or CitationValidator()
        self.limits = limits or AgentLimits()
        self.run_log = run_log
        self.timeout_seconds = timeout_seconds
        self.retrieval_top_k = retrieval_top_k
        self.context_builder = ContextBuilder(
            token_budget=context_token_budget,
            max_evidence=generation_max_evidence,
        )
        self.filings = tuple(filings)
        self.corpus_as_of = corpus_as_of
        self.clock = clock or seoul_now
        self.redact_secrets = redact_secrets
        self.planner = planner
        self.filing_relations = tuple(filing_relations)
        self.catalog = catalog

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
        stamp = request_time_context(self.clock())
        record: dict[str, Any] = {
            "run_id": run_id,
            "raw_user_query": message,
            "question": message,
            "request_time": stamp,
            "request_timezone": "Asia/Seoul",
            "corpus_as_of": (
                None
                if self.corpus_as_of is None
                else request_time_context(self.corpus_as_of)["iso"]
            ),
            "conversation_id": conversation_id,
            "resolved_company": None,
            "agent_action": None,
            "tool_calls": [],
            "evidence_ids": [],
            "citation_ids": [],
            "latency_ms": None,
            "total_ms": None,
            "company_resolution_ms": None,
            "retrieval_ms": None,
            "generation_ms": None,
            "validation_ms": None,
            "generator_type": getattr(
                self.answer_generator, "generator_type", "deterministic_fallback"
            ),
            "model": getattr(self.answer_generator, "model", None),
            "model_id": getattr(self.answer_generator, "model", None),
            "sdk_version": getattr(self.answer_generator, "sdk_version", None),
            "prompt_evidence_count": None,
            "retrieved_candidate_count": None,
            "retrieved_retrieval_unit_ids": [],
            "context_evidence_count": None,
            "context_evidence_ids": [],
            "context_token_count": None,
            "context_token_budget": self.context_builder.token_budget,
            "context_token_counter": CONTEXT_TOKEN_COUNTER_ID,
            "filing_period_end_years": [
                period.period_end_year
                for period in self.filings
                if period.period_end_year is not None
            ],
            "temporal_fact_mode": None,
            "temporal_target_year": None,
            "temporal_intent": None,
            "temporal_definition": None,
            "eligible_receipt_nos": [],
            "index_id": None,
            "search_scopes": [],
            "search_called": False,
            "planner": "llm" if self.planner is not None and self.planner.uses_llm else "raw_query",
            "retrieved_filing_counts": {},
            "context_filing_counts": {},
            "generation_start_ms": None,
            "first_answer_delta_ms": None,
            "generation_complete_ms": None,
            "partial_output_emitted": False,
            "partial_text": None,
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "status": None,
            "error": None,
            "abstain_stage": None,
            "abstain_reason": None,
        }
        try:
            yield StreamEvent("run.started", {"run_id": run_id})
            yield from self._run_body(
                message,
                clarification=clarification,
                budget=budget,
                record=record,
                started=started,
                request_time=stamp,
            )
        except AgentFlowError as exc:
            if record.get("partial_output_emitted"):
                record["status"] = "partial_generation_failure"
            else:
                record["status"] = exc.code
            record["error"] = {
                "code": exc.code,
                "detail": redact_text(exc.detail, self.redact_secrets),
            }
            yield StreamEvent(
                "error",
                {"run_id": run_id, "code": exc.code, "message": user_message(exc.code)},
            )
        except Exception as exc:
            if record.get("partial_output_emitted"):
                record["status"] = "partial_generation_failure"
            else:
                record["status"] = "internal_error"
            record["error"] = {
                "code": "internal_error",
                "detail": redact_text(f"{type(exc).__name__}: {exc}", self.redact_secrets),
                "traceback": redact_text(traceback.format_exc(), self.redact_secrets),
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
            elapsed = round((time.perf_counter() - started) * 1000, 3)
            record["latency_ms"] = elapsed
            record["total_ms"] = elapsed
            record["llm_call_count"] = budget.llm_calls
            record["deterministic_operation_count"] = budget.deterministic_calls
            record["deterministic_operations"] = list(budget.deterministic_kinds)
            record["search_call_count"] = budget.search_calls
            record["stop_reason"] = record.get("status")
            if self.run_log is not None:
                self.run_log.append(_redact_record(record, self.redact_secrets))

    def _run_body(
        self,
        message: str,
        *,
        clarification: ClarificationContext | None,
        budget: _Budget,
        record: dict[str, Any],
        started: float,
        request_time: dict[str, str],
    ) -> Iterator[StreamEvent]:
        self._check_timeout(started)
        yield StreamEvent("agent.status", {"status": "resolving_company"})
        resolve_started = time.perf_counter()
        question, resolution = self._resolve(message, clarification, budget)
        record["company_resolution_ms"] = round((time.perf_counter() - resolve_started) * 1000, 3)
        record["resolved_company"] = resolution_to_dict(resolution)
        record["question"] = question
        self._bind_company_scope(resolution, record)

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

        planner = self.planner if self.planner is not None else RawQueryPlanner()
        pending_action = None
        if (
            planner.uses_llm
            and resolution.status != "resolved"
            and clarification is None
            and self._planner_room(planner, budget, terminal=False)
        ):
            action = self._next_plan(
                planner,
                question=question,
                resolution=resolution,
                request_time=request_time,
                prior=[],
                budget=budget,
                record=record,
            )
            resolution, pending_action = self._apply_mention(action, resolution, record)
            self._bind_company_scope(resolution, record)
            direct = action.kind == "answer_directly" and bool(action.user_text)
            clarify = action.kind == "ask_clarification" and bool(action.user_text)
            if resolution.status != "resolved" and direct:
                answer = GeneratedAnswer(
                    text=action.user_text,
                    generator_type="llm_action",
                )
                yield from self._emit_answer(
                    answer,
                    evidence=[],
                    action="answer_directly",
                    status="completed",
                    record=record,
                    started=started,
                )
                return
            if resolution.status != "resolved" and clarify:
                clarify_status = (
                    "company_ambiguous" if resolution.status == "ambiguous" else "company_not_found"
                )
                yield from self._emit_clarification(
                    resolution,
                    status=clarify_status,
                    record=record,
                    started=started,
                    text=action.user_text,
                )
                return
            if resolution.status == "resolved" and direct:
                if _ungrounded_company_fact(
                    question,
                    action.user_text,
                    resolution,
                    request_time,
                ):
                    pending_action = None
                    record["evidence_guard"] = "rejected_ungrounded_direct_answer"
                else:
                    answer = GeneratedAnswer(
                        text=action.user_text,
                        generator_type="llm_action",
                    )
                    yield from self._emit_answer(
                        answer,
                        evidence=[],
                        action="answer_directly",
                        status="completed",
                        record=record,
                        started=started,
                    )
                    return
            if resolution.status == "resolved" and not action.requests_tool:
                pending_action = None

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
            yield from self._emit_unsupported_company(
                question,
                resolution=resolution,
                planner=planner,
                budget=budget,
                record=record,
                started=started,
                request_time=request_time,
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

        if not planner.uses_llm and not is_disclosure_question(question):
            answer = company_followup(resolution.company_name or "해당 기업")
            budget.note_deterministic("company_followup")
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
        hits, fiscal_year = yield from self._search_loop(
            question,
            resolution=resolution,
            planner=planner,
            budget=budget,
            record=record,
            started=started,
            request_time=request_time,
            pending_action=pending_action,
        )
        if hits is None:
            return
        retrieved = self.evidence_builder.build(
            hits,
            company_name=resolution.company_name or "",
        )
        selection = self.context_builder.build(retrieved)
        record["retrieved_filing_counts"] = dict(Counter(hit.receipt_no for hit in hits))
        record["context_filing_counts"] = dict(
            Counter(item.receipt_no for item in selection.evidence)
        )
        context_ids = [item.evidence_id for item in selection.evidence]
        record["retrieved_candidate_count"] = len(retrieved)
        record["retrieved_retrieval_unit_ids"] = [hit.retrieval_unit_id for hit in hits]
        record["context_evidence_count"] = len(selection.evidence)
        record["context_evidence_ids"] = context_ids
        record["context_token_count"] = selection.token_count
        record["prompt_evidence_count"] = len(selection.evidence)
        record["evidence_ids"] = context_ids
        for call in record["tool_calls"]:
            if call.get("status") == "success":
                call["context_evidence_ids"] = context_ids
                call["context_token_budget"] = selection.token_budget
                call["context_token_count"] = selection.token_count
                call["context_token_counter"] = selection.token_counter
        self._check_timeout(started)
        if not retrieved:
            answer = GeneratedAnswer(
                text="해당하는 공시 원문을 찾지 못했습니다. 확인되지 않은 내용은 답하지 않습니다.",
                limitations=["검색 결과가 비어 있습니다."],
                abstain=True,
                generator_type="answerability_guard",
            )
            yield from self._emit_answer(
                answer,
                evidence=[],
                action="abstain",
                status="retrieval_no_result",
                record=record,
                started=started,
                abstain_stage="pre_generation",
                abstain_reason="no_results",
            )
            return

        if not selection.evidence:
            answer = GeneratedAnswer(
                text=(
                    "검색한 공시를 답변에 넣을 수 있는 분량 안에 넣지 못했습니다. "
                    "확인되지 않은 내용은 답하지 않습니다."
                ),
                limitations=["답변에 넣을 수 있는 공시 원문이 없습니다."],
                abstain=True,
                generator_type="context_budget",
            )
            yield from self._emit_answer(
                answer,
                evidence=[],
                action="abstain",
                status="abstained",
                record=record,
                started=started,
                abstain_stage="pre_generation",
                abstain_reason="insufficient_evidence",
            )
            return

        self._check_timeout(started)
        if self._generator_uses_llm() and not budget.llm_room(reserve=0):
            yield from self._emit_generation_budget_fallback(
                question,
                selection.evidence,
                record=record,
                started=started,
                budget=budget,
            )
            return
        if hasattr(self.answer_generator, "iter_answer"):
            yield from self._stream_model_answer(
                question,
                selection.evidence,
                budget=budget,
                company_name=resolution.company_name or "",
                fiscal_year=fiscal_year,
                request_time=request_time,
                corpus_as_of=record["corpus_as_of"],
                record=record,
                started=started,
            )
            return

        if self._generator_uses_llm():
            budget.charge_llm()
        else:
            budget.note_deterministic("answer_generation")
        yield StreamEvent("agent.status", {"status": "generating_answer"})
        timings: dict[str, Any] = {
            "generation_ms": None,
            "validation_ms": None,
            "usage": None,
        }
        try:
            answer = self._generate_with_retry(
                question,
                selection.evidence,
                budget=budget,
                company_name=resolution.company_name or "",
                fiscal_year=fiscal_year,
                request_time=request_time,
                corpus_as_of=record["corpus_as_of"],
                timings=timings,
            )
        finally:
            record["generation_ms"] = timings["generation_ms"]
            record["validation_ms"] = timings["validation_ms"]
            record["generation_complete_ms"] = round((time.perf_counter() - started) * 1000, 3)
            _copy_usage(record, timings["usage"])
        action = "abstain" if answer.abstain else "search_disclosures"
        status = "abstained" if answer.abstain else "completed"
        yield from self._emit_answer(
            answer,
            evidence=selection.evidence,
            action=action,
            status=status,
            record=record,
            started=started,
            include_status=False,
            abstain_stage="generation" if answer.abstain else None,
            abstain_reason="insufficient_evidence" if answer.abstain else None,
        )

    def _search_loop(
        self,
        question: str,
        *,
        resolution: Resolution,
        planner: SearchPlanner,
        budget: _Budget,
        record: dict[str, Any],
        started: float,
        request_time: dict[str, str],
        pending_action: Any = None,
    ) -> Iterator[StreamEvent]:
        """Run at most ``max_search_calls`` tool calls. Returns hits, or None if already answered.

        The planner sees public tool observations. It does not receive evidence
        ids, stack traces, or the user-facing answer from a previous turn.
        """
        executed = 0
        hits: list[SearchHit] = []
        fiscal_year: int | None = None
        prior: list[dict[str, Any]] = []
        searched = False
        last_failure: tuple[str, str] | None = None
        rejected_direct = 0
        company_id = resolution.company_id or ""
        self._bind_company_scope(resolution, record)
        filings = self._filings_for(resolution)
        corpus_as_of = self._corpus_as_of_for(company_id)
        relations = self._relations_for(company_id)
        if record.get("evidence_guard") == "rejected_ungrounded_direct_answer":
            prior.append(_evidence_required_observation(resolution.company_name or ""))
            rejected_direct = 1
        while True:
            self._check_timeout(started)
            if executed >= self.limits.max_search_calls:
                if hits:
                    break
                if (
                    planner.uses_llm
                    and last_failure is not None
                    and self._planner_room(planner, budget, terminal=True)
                ):
                    action = self._next_plan(
                        planner,
                        question=question,
                        resolution=resolution,
                        request_time=request_time,
                        prior=prior,
                        budget=budget,
                        record=record,
                        search_calls_remaining=0,
                    )
                    if action.requests_tool:
                        record["search_suppressed"] = True
                        break
                    if action.user_text and action.kind in {"answer_directly", "ask_clarification"}:
                        if action.kind == "answer_directly" and _ungrounded_company_fact(
                            question,
                            action.user_text,
                            resolution,
                            request_time,
                        ):
                            yield from self._emit_evidence_guard_abstain(
                                record=record,
                                started=started,
                            )
                            return None, None
                        yield from self._emit_planned_speech(
                            action,
                            resolution=resolution,
                            last_failure=last_failure,
                            record=record,
                            started=started,
                        )
                        return None, None
                elif executed == 0 and pending_action is None:
                    action = self._next_plan(
                        planner,
                        question=question,
                        resolution=resolution,
                        request_time=request_time,
                        prior=prior,
                        budget=budget,
                        record=record,
                        search_calls_remaining=0,
                    )
                    if action.requests_tool:
                        budget.charge_search()
                break
            if pending_action is not None:
                action = pending_action
                pending_action = None
            else:
                if not self._planner_room(planner, budget, terminal=False):
                    break
                action = self._next_plan(
                    planner,
                    question=question,
                    resolution=resolution,
                    request_time=request_time,
                    prior=prior,
                    budget=budget,
                    record=record,
                    search_calls_remaining=self.limits.max_search_calls - budget.search_calls,
                )
            if (
                not hits
                and action.user_text
                and action.kind
                in {
                    "answer_directly",
                    "ask_clarification",
                }
            ):
                blocked = action.kind == "answer_directly" and _ungrounded_company_fact(
                    question,
                    action.user_text,
                    resolution,
                    request_time,
                )
                if blocked and rejected_direct == 0 and executed < self.limits.max_search_calls:
                    rejected_direct = 1
                    record["evidence_guard"] = "rejected_ungrounded_direct_answer"
                    prior.append(_evidence_required_observation(resolution.company_name or ""))
                    continue
                if blocked:
                    yield from self._emit_evidence_guard_abstain(
                        record=record,
                        started=started,
                    )
                    return None, None
                yield from self._emit_planned_speech(
                    action,
                    resolution=resolution,
                    last_failure=last_failure,
                    record=record,
                    started=started,
                )
                return None, None
            if not action.requests_tool:
                break
            if budget.search_calls >= self.limits.max_search_calls:
                record["search_suppressed"] = True
                break
            budget.charge_search()
            executed += 1
            tool = resolve_search_call(
                action.raw,
                company_id=company_id,
                filings=filings,
                corpus_as_of=corpus_as_of,
                max_top_k=self.retrieval_top_k,
                relations=relations,
            )
            scope = self._search_scope(company_id, tool, relations)
            if scope is not None:
                record["search_scopes"].append(scope)
            if tool.resolved is not None and tool.resolved.arguments.period is not None:
                year = tool.resolved.arguments.period.year
                if fiscal_year is None and year is not None:
                    fiscal_year = year
            if tool.status != "ok" or tool.resolved is None:
                summary = _failure_observation(
                    tool,
                    company_name=resolution.company_name or "",
                    corpus_as_of=record["corpus_as_of"],
                    filings=filings,
                )
                prior.append(summary)
                last_failure = (tool.status, str(summary["public_reason"]))
                logged = _tool_log(tool, summary, request=None, result=None, latency_ms=None)
                if scope is not None:
                    logged["search_scope"] = scope
                if logged["arguments"] is None and isinstance(action.raw, dict):
                    logged["arguments"] = action.raw
                    logged["query"] = action.raw.get("query")
                    logged["company_id"] = action.raw.get("company_id")
                record["tool_calls"].append(logged)
                continue
            yield StreamEvent("agent.status", {"status": "searching_disclosures"})
            request = SearchDisclosuresRequest(
                query=tool.resolved.arguments.query,
                company_id=company_id,
                corp_code=resolution.corp_code,
                fiscal_year=fiscal_year,
                top_k=tool.resolved.top_k,
                eligible_receipt_nos=(
                    None
                    if tool.filings is None
                    else tuple(item.receipt_no for item in tool.filings)
                ),
            )
            _remember_receipts(record, request.eligible_receipt_nos)
            search_started = time.perf_counter()
            result = self.search(request)
            search_ms = round((time.perf_counter() - search_started) * 1000, 3)
            record["retrieval_ms"] = search_ms
            record["search_called"] = True
            if getattr(result, "retrieval_policy", ""):
                record["retrieval_policy"] = result.retrieval_policy
                record["retrieval_edit"] = result.retrieval_edit
                record["retrieval_policy_reason"] = result.policy_reason
            searched = True
            call_hits = list(result.hits)
            if self.catalog is not None:
                _require_company_hits(call_hits, company_id)
            if call_hits:
                hits = _merge_hits(hits, call_hits)
                last_failure = None
                status = "success"
                message = "filings matched"
            else:
                status = "no_results"
                message = "검색된 공시 근거가 없습니다."
                last_failure = (status, message)
            built = self.evidence_builder.build(
                call_hits,
                company_name=resolution.company_name or "",
            )
            summary = _tool_summary(
                tool,
                hits=call_hits,
                status=status,
                message=message,
            )
            if status != "success":
                extra = _failure_observation(
                    tool,
                    company_name=resolution.company_name or "",
                    corpus_as_of=record["corpus_as_of"],
                    filings=filings,
                    status=status,
                )
                extra.pop("hits", None)
                extra.pop("matched_filings", None)
                summary.update(extra)
            prior.append(summary)
            log = _tool_log(tool, summary, request=request, result=result, latency_ms=search_ms)
            if scope is not None:
                log["search_scope"] = scope
            log["evidence_ids"] = [item.evidence_id for item in built]
            log["status"] = status
            record["tool_calls"].append(log)
        if hits:
            return hits, fiscal_year
        if searched:
            return [], fiscal_year
        if rejected_direct and last_failure is None:
            yield from self._emit_evidence_guard_abstain(
                record=record,
                started=started,
            )
            return None, None
        status, message = last_failure or (
            "search_not_requested",
            "공시를 검색하지 않아 기업 사실을 답하지 않습니다.",
        )
        answer = GeneratedAnswer(
            text=_abstain_text(status, message, fiscal_year),
            limitations=[message],
            abstain=True,
            generator_type="tool_validation" if status != "search_not_requested" else "planner",
        )
        yield from self._emit_answer(
            answer,
            evidence=[],
            action="abstain",
            status=status,
            record=record,
            started=started,
            abstain_stage="pre_generation",
            abstain_reason=status,
        )
        return None, None

    def _bind_company_scope(self, resolution: Resolution, record: dict[str, Any]) -> None:
        """Point the planner clock at the resolved company's index."""
        if self.catalog is None or resolution.status != "resolved":
            return
        corpus = self.catalog.corpus(resolution.company_id or "")
        if corpus is None:
            return
        record["corpus_as_of"] = request_time_context(corpus.corpus_as_of)["iso"]
        record["index_id"] = str(corpus.manifest.get("corpus_id") or "")
        record["filing_period_end_years"] = [
            period.period_end_year
            for period in corpus.filings
            if period.period_end_year is not None
        ]

    def _filings_for(self, resolution: Resolution) -> tuple[FilingPeriod, ...]:
        if resolution.status != "resolved":
            return ()
        if self.catalog is None:
            return self.filings
        corpus = self.catalog.corpus(resolution.company_id or "")
        return () if corpus is None else corpus.filings

    def _corpus_as_of_for(self, company_id: str) -> datetime | None:
        if self.catalog is None:
            return self.corpus_as_of
        corpus = self.catalog.corpus(company_id)
        return None if corpus is None else corpus.corpus_as_of

    def _relations_for(self, company_id: str) -> tuple[FilingRelation, ...]:
        if self.catalog is None:
            return self.filing_relations
        return self.catalog.relations_for(company_id)

    def _search_scope(
        self,
        company_id: str,
        tool: ToolResolution,
        relations: tuple[FilingRelation, ...],
    ) -> dict[str, Any] | None:
        if self.catalog is None:
            return None
        corpus = self.catalog.corpus(company_id)
        if corpus is None:
            return None
        return search_scope_record(self.catalog, corpus, tool, relations)

    def _next_plan(
        self,
        planner: SearchPlanner,
        *,
        question: str,
        resolution: Resolution,
        request_time: dict[str, str],
        prior: list[dict[str, Any]],
        budget: _Budget,
        record: dict[str, Any],
        search_calls_remaining: int | None = None,
    ) -> Any:
        if planner.uses_llm:
            budget.charge_llm()
        remaining = (
            self.limits.max_search_calls - budget.search_calls
            if search_calls_remaining is None
            else search_calls_remaining
        )
        context = PlannerContext(
            question=question,
            request_time=request_time,
            corpus_as_of=record["corpus_as_of"],
            company_id=resolution.company_id or "",
            company_name=resolution.company_name or "",
            filings=self._filings_for(resolution),
            prior_results=tuple(prior),
            max_top_k=self.retrieval_top_k,
            company_resolution=resolution_to_dict(resolution),
            search_calls_remaining=max(0, remaining),
        )
        action = planner.next_action(context)
        _add_planner_usage(record, action.usage)
        record.setdefault("planner_actions", []).append(_planner_observation(action))
        return action

    def _apply_mention(
        self,
        action: Any,
        resolution: Resolution,
        record: dict[str, Any],
    ) -> tuple[Resolution, Any]:
        """Resolve an explicit surface mention. Does not scan the question again."""
        mention = getattr(action, "company_mention", None)
        if not mention or resolution.status == "resolved":
            return resolution, action if getattr(action, "requests_tool", False) else None
        mentioned = self.registry.resolve_mention(mention)
        if mentioned.status not in {"resolved", "unsupported", "ambiguous"}:
            return resolution, action if getattr(action, "requests_tool", False) else None
        record["resolved_company"] = resolution_to_dict(mentioned)
        record["company_mention_source"] = "llm"
        pending = action if action.requests_tool and mentioned.status == "resolved" else None
        return mentioned, pending

    def _emit_unsupported_company(
        self,
        question: str,
        *,
        resolution: Resolution,
        planner: SearchPlanner,
        budget: _Budget,
        record: dict[str, Any],
        started: float,
        request_time: dict[str, str],
    ) -> Iterator[StreamEvent]:
        """Known company with no indexed filings. The model phrases the observation."""
        name = resolution.company_name or "해당 기업"
        observation = public_tool_observation(
            status="unsupported_company",
            company_name=name,
            corpus_as_of=record["corpus_as_of"],
            requested_period=None,
            report_types=None,
            matched_count=0,
            latest_available_period=None,
        )
        observation["public_reason"] = (
            f"{name}은 기업 목록에서 확인했지만, 현재 지원하는 공시가 없습니다."
        )
        observation["message"] = observation["public_reason"]
        record["public_observation"] = observation
        if planner.uses_llm:
            action = self._next_plan(
                planner,
                question=question,
                resolution=resolution,
                request_time=request_time,
                prior=[observation],
                budget=budget,
                record=record,
            )
            if action.user_text and action.kind == "answer_directly":
                yield from self._emit_planned_speech(
                    action,
                    resolution=resolution,
                    last_failure=("unsupported_company", observation["public_reason"]),
                    record=record,
                    started=started,
                )
                return
        answer = GeneratedAnswer(
            text=observation["public_reason"],
            abstain=True,
            limitations=[observation["public_reason"]],
            generator_type="direct",
        )
        yield from self._emit_answer(
            answer,
            evidence=[],
            action="abstain",
            status="company_unsupported",
            record=record,
            started=started,
            abstain_stage="pre_generation",
            abstain_reason="unsupported_company",
        )

    def _emit_planned_speech(
        self,
        action: Any,
        *,
        resolution: Resolution,
        last_failure: tuple[str, str] | None,
        record: dict[str, Any],
        started: float,
    ) -> Iterator[StreamEvent]:
        if action.kind == "ask_clarification":
            clarify_status = (
                "company_ambiguous" if resolution.status == "ambiguous" else "needs_clarification"
            )
            yield from self._emit_clarification(
                resolution,
                status=clarify_status,
                record=record,
                started=started,
                text=action.user_text,
            )
            return
        if last_failure is None:
            answer = GeneratedAnswer(text=action.user_text, generator_type="llm_action")
            yield from self._emit_answer(
                answer,
                evidence=[],
                action="answer_directly",
                status="completed",
                record=record,
                started=started,
            )
            return
        status, public_reason = last_failure
        answer = GeneratedAnswer(
            text=action.user_text,
            limitations=[public_reason],
            abstain=True,
            generator_type="llm_action",
        )
        yield from self._emit_answer(
            answer,
            evidence=[],
            action="answer_directly",
            status=status,
            record=record,
            started=started,
            abstain_stage="pre_generation",
            abstain_reason=status,
        )

    def _resolve(
        self,
        message: str,
        clarification: ClarificationContext | None,
        budget: _Budget,
    ) -> tuple[str, Resolution]:
        if clarification and clarification.selected_company_name:
            question = clarification.prior_message or message
            budget.note_deterministic("company_resolution")
            resolution = self.registry.resolve_mention(clarification.selected_company_name)
            return question, resolution
        if is_smalltalk(message):
            return message, Resolution(status="no_mention")
        budget.note_deterministic("company_resolution")
        mentions = self.registry.extract_mentions(message)
        return message, self.registry.resolve_mentions(mentions)

    def _generate_direct(self, message: str, budget: _Budget) -> GeneratedAnswer:
        budget.note_deterministic("direct_answer")
        return direct_answer(message)

    def _generator_uses_llm(self) -> bool:
        return bool(getattr(self.answer_generator, "uses_llm", False))

    def _planner_room(self, planner: SearchPlanner, budget: _Budget, *, terminal: bool) -> bool:
        if not planner.uses_llm:
            return True
        reserve = 0 if terminal or not self._generator_uses_llm() else 1
        return budget.llm_room(reserve=reserve)

    def _emit_evidence_guard_abstain(
        self,
        *,
        record: dict[str, Any],
        started: float,
    ) -> Iterator[StreamEvent]:
        record["evidence_guard"] = "abstain_without_evidence"
        answer = GeneratedAnswer(
            text=(
                "이 질문은 특정 기업의 공시에서 확인해야 합니다. "
                "확인된 공시 근거가 없어 그 사실을 단정하지 않습니다."
            ),
            limitations=["허용된 공시 근거가 없습니다."],
            abstain=True,
            generator_type="evidence_guard",
        )
        yield from self._emit_answer(
            answer,
            evidence=[],
            action="abstain",
            status="abstained",
            record=record,
            started=started,
            abstain_stage="pre_generation",
            abstain_reason="evidence_required",
        )

    def _emit_generation_budget_fallback(
        self,
        question: str,
        evidence: list[Evidence],
        *,
        record: dict[str, Any],
        started: float,
        budget: _Budget,
    ) -> Iterator[StreamEvent]:
        """Finish after retrieval when no API call remains for the answer model."""
        budget.note_deterministic("generation_budget_fallback")
        record["answer_stop"] = "generation_budget_fallback"
        if not evidence:
            answer = GeneratedAnswer(
                text=(
                    "검색은 마쳤지만 답변 호출 한도가 남아 있지 않습니다. "
                    "확인되지 않은 내용은 답하지 않습니다."
                ),
                limitations=["답변 모델 호출 한도가 소진되었습니다."],
                abstain=True,
                generator_type="generation_budget",
            )
            yield from self._emit_answer(
                answer,
                evidence=[],
                action="abstain",
                status="abstained",
                record=record,
                started=started,
                abstain_stage="pre_generation",
                abstain_reason="answer_budget_exhausted",
            )
            return
        answer = EvidenceBoundAnswerGenerator().generate(
            question,
            evidence,
        )
        answer.limitations = [
            *answer.limitations,
            "답변 모델 호출 한도가 남아 있지 않아 검색 발췌만 제시합니다.",
        ]
        yield from self._emit_answer(
            answer,
            evidence=evidence,
            action="search_disclosures",
            status="completed",
            record=record,
            started=started,
        )

    def _generate_with_retry(
        self,
        question: str,
        evidence: list[Evidence],
        *,
        budget: _Budget,
        company_name: str,
        fiscal_year: int | None,
        request_time: dict[str, str],
        corpus_as_of: str | None,
        timings: dict[str, Any],
    ) -> GeneratedAnswer:
        """The caller already charged the first attempt. One schema failure may retry."""
        gen_ms = 0.0
        val_ms = 0.0
        last_error: StructuredOutputInvalid | None = None
        for attempt in range(STRUCTURED_OUTPUT_ATTEMPTS):
            if attempt:
                budget.charge_llm()
            setattr(self.answer_generator, "last_generation_ms", None)
            setattr(self.answer_generator, "last_usage", None)
            started_at = time.perf_counter()
            answer: GeneratedAnswer | None = None
            failed: StructuredOutputInvalid | None = None
            try:
                answer = self.answer_generator.generate(
                    question,
                    evidence,
                    company_name=company_name,
                    fiscal_year=fiscal_year,
                    request_time=request_time,
                    corpus_as_of=corpus_as_of,
                )
            except StructuredOutputInvalid as exc:
                failed = exc
            finally:
                elapsed = (time.perf_counter() - started_at) * 1000
                reported = getattr(self.answer_generator, "last_generation_ms", None)
                if isinstance(reported, (int, float)):
                    gen_ms += float(reported)
                    val_ms += max(0.0, elapsed - float(reported))
                else:
                    gen_ms += elapsed
                timings["generation_ms"] = round(gen_ms, 3)
                timings["validation_ms"] = round(val_ms, 3)
                usage = getattr(self.answer_generator, "last_usage", None)
                if isinstance(usage, dict):
                    timings["usage"] = usage
            if failed is not None:
                last_error = failed
                continue
            assert answer is not None
            validate_started = time.perf_counter()
            try:
                self._validate_answer(answer, evidence)
            finally:
                val_ms += (time.perf_counter() - validate_started) * 1000
                timings["validation_ms"] = round(val_ms, 3)
            return answer
        assert last_error is not None
        raise last_error

    def _stream_model_answer(
        self,
        question: str,
        evidence: list[Evidence],
        *,
        budget: _Budget,
        company_name: str,
        fiscal_year: int | None,
        request_time: dict[str, str],
        corpus_as_of: str | None,
        record: dict[str, Any],
        started: float,
    ) -> Iterator[StreamEvent]:
        """Forward claim text while the provider is still generating later claims."""
        budget.charge_llm()
        yield StreamEvent("agent.status", {"status": "generating_answer"})
        record["generation_start_ms"] = round((time.perf_counter() - started) * 1000, 3)
        by_id = {item.evidence_id: item for item in evidence}
        answer: GeneratedAnswer | None = None
        last_error: StructuredOutputInvalid | None = None
        gen_ms = 0.0
        try:
            for attempt in range(STRUCTURED_OUTPUT_ATTEMPTS):
                if attempt:
                    budget.charge_llm()
                emitted = False
                try:
                    for event in self.answer_generator.iter_answer(
                        question,
                        evidence,
                        company_name=company_name,
                        fiscal_year=fiscal_year,
                        request_time=request_time,
                        corpus_as_of=corpus_as_of,
                    ):
                        if isinstance(event, AnswerDelta):
                            emitted = True
                            record["partial_output_emitted"] = True
                            record["partial_text"] = (record.get("partial_text") or "") + event.text
                            yield from _emit_text(record, started, event.text)
                        elif isinstance(event, AnswerCitationEvent):
                            item = by_id.get(event.citation.evidence_id)
                            if item is None:
                                raise CitationValidationFailure(
                                    "citation is outside the current run"
                                )
                            payload = item.to_dict()
                            payload["citation_id"] = event.citation.citation_id
                            yield StreamEvent("citation", payload)
                        elif isinstance(event, AnswerFinished):
                            validate_started = time.perf_counter()
                            try:
                                self._validate_answer(event.answer, evidence)
                            finally:
                                record["validation_ms"] = round(
                                    (time.perf_counter() - validate_started) * 1000,
                                    3,
                                )
                            answer = event.answer
                    if answer is not None:
                        break
                except StructuredOutputInvalid as exc:
                    last_error = exc
                    if emitted or attempt + 1 >= STRUCTURED_OUTPUT_ATTEMPTS:
                        raise
                    continue
                finally:
                    reported = getattr(self.answer_generator, "last_generation_ms", None)
                    if isinstance(reported, (int, float)):
                        gen_ms += float(reported)
                    usage = getattr(self.answer_generator, "last_usage", None)
                    if isinstance(usage, dict):
                        _copy_usage(record, usage)
            if answer is None:
                raise last_error or StructuredOutputInvalid("model returned no structured answer")
        finally:
            record["generation_ms"] = round(gen_ms, 3)
            record["generation_complete_ms"] = round((time.perf_counter() - started) * 1000, 3)
        record["partial_output_emitted"] = False
        record["partial_text"] = None
        action = "abstain" if answer.abstain else "search_disclosures"
        status = "abstained" if answer.abstain else "completed"
        yield from self._emit_answer(
            answer,
            evidence=evidence,
            action=action,
            status=status,
            record=record,
            started=started,
            include_status=False,
            include_deltas=False,
            include_citations=False,
            abstain_stage="generation" if answer.abstain else None,
            abstain_reason="insufficient_evidence" if answer.abstain else None,
        )

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
        if answer.generator_type == "deterministic_fallback":
            if not answer_markers_match(answer):
                raise CitationValidationFailure("citation markers do not match citation ids")
        elif answer_has_markers(answer) and not answer_markers_match(answer):
            raise CitationValidationFailure("citation markers do not match citation ids")
        self.citation_validator.validate(answer.citations, evidence_ids)

    def _emit_answer(
        self,
        answer: GeneratedAnswer,
        *,
        evidence: list[Evidence],
        action: str,
        status: str,
        record: dict[str, Any],
        started: float,
        include_status: bool = True,
        include_deltas: bool = True,
        include_citations: bool = True,
        abstain_stage: str | None = None,
        abstain_reason: str | None = None,
    ) -> Iterator[StreamEvent]:
        self._check_timeout(started)
        shown = [] if answer.abstain else evidence
        by_id = {item.evidence_id: item for item in shown}
        record["agent_action"] = action
        record["status"] = status
        record["abstain_stage"] = abstain_stage
        record["abstain_reason"] = abstain_reason
        record["generator_type"] = answer.generator_type
        record["citation_ids"] = [item.citation_id for item in answer.citations]
        record["final_answer"] = {
            "text": answer.text,
            "citations": [item.to_dict() for item in answer.citations],
            "limitations": list(answer.limitations),
            "abstain": answer.abstain,
        }
        record["generation_evidence"] = _generation_evidence_record(evidence)
        if include_status:
            yield StreamEvent("agent.status", {"status": "generating_answer"})
        if include_deltas:
            yield from _emit_text(record, started, answer.text)
        if include_citations:
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
        record["final_answer"] = {
            "text": question,
            "citations": [],
            "limitations": [],
            "abstain": False,
        }
        record["generation_evidence"] = []
        yield from _emit_text(record, started, question)
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
        self.deterministic_calls = 0
        self.deterministic_kinds: list[str] = []

    def note_deterministic(self, kind: str) -> None:
        self.deterministic_calls += 1
        self.deterministic_kinds.append(kind)

    def llm_room(self, *, reserve: int = 0) -> bool:
        """True when one more API call still fits, after ``reserve`` later calls."""
        return self.llm_calls + 1 + reserve <= self.limits.max_llm_calls

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


def _add_planner_usage(record: dict[str, Any], usage: object) -> None:
    if not isinstance(usage, dict):
        return
    current = record.get("planner_usage")
    if not isinstance(current, dict):
        current = {}
    facts: dict[str, int] = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, int):
            current[key] = int(current.get(key) or 0) + value
            facts[key] = value
    record["planner_usage"] = current
    if facts:
        record.setdefault("llm_usages", []).append({"stage": "planner", **facts})


def _planner_observation(action: Any) -> dict[str, Any]:
    """Planner choice for the run log. The user-facing answer is stored separately."""
    raw = action.raw if isinstance(getattr(action, "raw", None), dict) else None
    arguments = None
    if getattr(action, "kind", None) == TOOL_NAME and raw is not None:
        arguments = {
            key: raw.get(key)
            for key in (
                "query",
                "company_id",
                "period",
                "filing_date_from",
                "filing_date_to",
                "report_types",
                "as_of",
                "top_k",
            )
        }
    return {
        "kind": action.kind,
        "requests_tool": action.requests_tool,
        "schema_error": action.schema_error,
        "arguments": arguments,
        "latency_ms": action.latency_ms,
    }


def _generation_evidence_record(evidence: list[Evidence]) -> list[dict[str, Any]]:
    """Evidence the generator actually received, including when the answer abstains."""
    return [
        {
            "evidence_id": item.evidence_id,
            "retrieval_unit_id": item.retrieval_unit_id,
            "receipt_no": item.receipt_no,
            "report_name": item.report_name,
            "filing_period": item.filing_period,
            "receipt_date": item.receipt_date,
            "document_name": item.document_name,
            "section_path": list(item.section_path),
            "text": item.text,
        }
        for item in evidence
    ]


def _require_company_hits(hits: list[SearchHit], company_id: str) -> None:
    foreign = [hit.retrieval_unit_id for hit in hits if hit.corp_code != company_id]
    if foreign:
        raise RetrievalFailure("retrieval hit is outside the resolved company")


def _remember_receipts(record: dict[str, Any], receipt_nos: tuple[str, ...] | None) -> None:
    if not receipt_nos:
        return
    found = list(record.get("eligible_receipt_nos") or [])
    for receipt_no in receipt_nos:
        if receipt_no not in found:
            found.append(receipt_no)
    record["eligible_receipt_nos"] = found


def _merge_hits(existing: list[SearchHit], extra: list[SearchHit]) -> list[SearchHit]:
    seen = {item.retrieval_unit_id for item in existing}
    merged = list(existing)
    for item in extra:
        if item.retrieval_unit_id in seen:
            continue
        merged.append(item)
        seen.add(item.retrieval_unit_id)
    return merged


def _failure_observation(
    tool: ToolResolution,
    *,
    company_name: str,
    corpus_as_of: str | None,
    filings: tuple[FilingPeriod, ...],
    status: str | None = None,
) -> dict[str, Any]:
    """Public corpus facts. Internal schema text stays out of this object."""
    code = status or tool.status
    requested = None
    report_types = None
    if tool.resolved is not None:
        period = tool.resolved.arguments.period
        requested = None if period is None else period.model_dump()
        kinds = tool.resolved.arguments.report_types
        report_types = None if kinds is None else list(kinds)
    observation = public_tool_observation(
        status=code,
        company_name=company_name,
        corpus_as_of=corpus_as_of,
        requested_period=requested,
        report_types=report_types,
        matched_count=0 if tool.filings is None else len(tool.filings),
        latest_available_period=_latest_period_label(filings),
    )
    observation["message"] = observation["public_reason"]
    observation["matched_filings"] = []
    observation["hits"] = []
    return observation


def _latest_period_label(filings: tuple[FilingPeriod, ...]) -> str | None:
    keyed = [
        item
        for item in filings
        if item.period_end_year is not None and item.period_end_month is not None
    ]
    if not keyed:
        return None
    best = max(keyed, key=lambda item: (item.period_end_year or 0, item.period_end_month or 0))
    return period_label(best) or None


def _tool_summary(
    tool: ToolResolution,
    *,
    hits: list[SearchHit],
    status: str | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    filings = [] if tool.filings is None else [filing_public_row(item) for item in tool.filings]
    return {
        "status": status or tool.status,
        "message": message or tool.message,
        "matched_filings": filings,
        "hits": [_hit_brief(item) for item in hits[:5]],
    }


def _hit_brief(hit: SearchHit) -> dict[str, Any]:
    return {
        "receipt_no": hit.receipt_no,
        "report_name": hit.report_name,
        "filing_period": hit.filing_period,
        "receipt_date": hit.receipt_date,
        "source_sha256": hit.source_sha256,
        "retrieval_unit_id": hit.retrieval_unit_id,
        "score": hit.score,
        "excerpt": hit.text[:180],
    }


def _tool_log(
    tool: ToolResolution,
    summary: dict[str, Any],
    *,
    request: SearchDisclosuresRequest | None,
    result: SearchDisclosuresResult | None,
    latency_ms: float | None,
) -> dict[str, Any]:
    arguments = None if tool.resolved is None else arguments_record(tool.resolved)
    payload: dict[str, Any] = {
        "name": TOOL_NAME,
        "status": summary["status"],
        "message": summary["message"],
        "arguments": arguments,
        "query": None if request is None else request.query,
        "company_id": None if request is None else request.company_id,
        "corp_code": None if request is None else request.corp_code,
        "fiscal_year": None if request is None else request.fiscal_year,
        "eligible_receipt_nos": [] if request is None else list(request.eligible_receipt_nos or ()),
        "matched_filings": summary["matched_filings"],
        "top_k": None if tool.resolved is None else tool.resolved.top_k,
        "latency_ms": latency_ms,
        "evidence_ids": [],
        "retrieval_unit_ids": [],
        "retrieved_receipt_nos": [],
    }
    if request is not None and arguments is not None:
        payload["query"] = arguments["query"]
        payload["company_id"] = arguments["company_id"]
    if result is not None:
        payload["retriever_type"] = result.retriever_type
        payload["representation_version"] = result.representation_version
        payload["embedding_model"] = result.embedding_model
        payload["retrieval_unit_ids"] = [hit.retrieval_unit_id for hit in result.hits]
        payload["retrieved_receipt_nos"] = [hit.receipt_no for hit in result.hits]
        payload["retrieved_corp_codes"] = [hit.corp_code for hit in result.hits]
        if result.retrieval_policy:
            payload["retrieval_policy"] = result.retrieval_policy
            payload["retrieval_edit"] = result.retrieval_edit
            payload["retrieval_policy_reason"] = result.policy_reason
    return payload


def _evidence_required_observation(company_name: str) -> dict[str, Any]:
    observation = public_tool_observation(
        status="evidence_required",
        company_name=company_name,
        corpus_as_of=None,
        requested_period=None,
        report_types=None,
        matched_count=0,
        latest_available_period=None,
    )
    observation["hits"] = []
    observation["message"] = observation["public_reason"]
    return observation


def _ungrounded_company_fact(
    question: str,
    answer_text: str,
    resolution: Resolution,
    request_time: dict[str, str],
) -> bool:
    """True when a direct answer states a company filing fact without evidence."""
    if resolution.status != "resolved" or not answer_text.strip():
        return False
    if _future_confirmed_question(question, request_time) and not _AMOUNT_RE.search(answer_text):
        return False
    for part in [line.strip() for line in answer_text.splitlines() if line.strip()] or [
        answer_text.strip()
    ]:
        if classify_product_claim(part)["claim_type"] == "factual_claim":
            return True
    return False


_AMOUNT_RE = re.compile(r"\d{1,3}(?:,\d{3})+|\d+\s*(?:원|억|조)")


def _future_confirmed_question(question: str, request_time: dict[str, str]) -> bool:
    years = [int(year) for year in mentioned_years(question)]
    if not years:
        return False
    request_year = int(str(request_time.get("iso") or "0000")[:4])
    if max(years) <= request_year:
        return False
    if any(token in question for token in ("목표", "계획", "전망")):
        return False
    return "확정" in question


def _abstain_text(status: str, message: str, fiscal_year: int | None) -> str:
    if status == "no_matching_filing" and fiscal_year is not None:
        return (
            f"현재 보유한 공시 목록에서 {fiscal_year}년 조건에 맞는 공시를 찾지 못했습니다. "
            "확인되지 않은 수치는 답하지 않습니다."
        )
    if status == "no_matching_filing":
        return "조건에 맞는 공시를 찾지 못했습니다. 확인되지 않은 내용은 답하지 않습니다."
    if status == "corpus_not_fresh_enough":
        return "요청한 시점은 데이터 기준 시각보다 뒤입니다. 그 시점의 공시는 검색하지 않습니다."
    if status == "unsupported_company":
        return "확인된 기업과 다른 검색 대상은 조회하지 않습니다."
    if status == "search_not_requested":
        return message
    if status == "invalid_arguments":
        return "검색 조건을 확인하지 못했습니다. 확인되지 않은 내용은 답하지 않습니다."
    return "확인된 공시 원문이 없습니다. 확인되지 않은 내용은 답하지 않습니다."


def _copy_usage(record: dict[str, Any], usage: object) -> None:
    if not isinstance(usage, dict):
        return
    facts: dict[str, int] = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, int):
            record[key] = value
            facts[key] = value
    if facts:
        record.setdefault("llm_usages", []).append({"stage": "generation", **facts})


def _redact_record(record: dict[str, Any], secrets: tuple[str, ...]) -> dict[str, Any]:
    encoded = json.dumps(record, ensure_ascii=False)
    return json.loads(redact_text(encoded, secrets))


def _emit_text(record: dict[str, Any], started: float, text: str) -> Iterator[StreamEvent]:
    if not text:
        return
    if record.get("first_answer_delta_ms") is None:
        record["first_answer_delta_ms"] = round((time.perf_counter() - started) * 1000, 3)
    yield StreamEvent("answer.delta", {"text": text})


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
        "abstain_stage": record.get("abstain_stage"),
        "abstain_reason": record.get("abstain_reason"),
        "public_observation": record.get("public_observation"),
    }


def build_default_runtime(
    root: Path | None = None,
    *,
    run_log_path: Path | None = None,
) -> AgentRuntime:
    """Wire the cached corp list and every built product index. Does not encode documents."""
    base = root or repo_root()
    from disclosure_agent.agent.h10_search import H10Search
    from disclosure_agent.agent.search_catalog import (
        CatalogEvidenceBuilder,
        load_search_catalog,
    )

    catalog = load_search_catalog(base)
    if not catalog.corpora:
        raise RetrievalFailure(
            "product search catalog is empty. Build a Dense-D2a index under data/product/indexes."
        )
    registry = CompanyRegistry.from_corp_cache(
        base / "data" / "cache",
        indexed_corp_codes=catalog.indexed_corp_codes(),
        fallback=[
            CompanyRecord(
                corp_code=corpus.binding.corp_code,
                corp_name=corpus.binding.company_name,
                stock_code="",
            )
            for corpus in catalog.corpora
        ],
    )
    log_path = run_log_path or (base / "data" / "product" / "runs" / "agent_runs.jsonl")
    from disclosure_agent.agent.llm import OpenAIAnswerClient, OpenAIAnswerGenerator
    from disclosure_agent.agent.settings import load_openai_config
    from disclosure_agent.cli import load_dotenv_files

    load_dotenv_files()
    openai_config = load_openai_config()
    generator = None
    planner = None
    secrets: tuple[str, ...] = ()
    if openai_config.configured:
        client = OpenAIAnswerClient(openai_config)
        generator = OpenAIAnswerGenerator(client)
        planner = OpenAISearchPlanner(client)
        secrets = (openai_config.api_key,)
    return AgentRuntime(
        registry=registry,
        search=H10Search(catalog),
        evidence_builder=CatalogEvidenceBuilder(catalog),
        answer_generator=generator,
        run_log=RunLog(log_path),
        redact_secrets=secrets,
        planner=planner,
        filing_relations=catalog.relations,
        catalog=catalog,
    )

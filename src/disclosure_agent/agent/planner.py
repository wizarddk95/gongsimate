"""Choose search_disclosures arguments.

The language model sees the raw question, the request clock, and the filing
catalog. A fallback planner copies the raw question and does not infer a
period from wording.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from disclosure_agent.agent.temporal import FilingPeriod, filing_catalog_row
from disclosure_agent.agent.tool_args import (
    TOOL_NAME,
    SearchToolArguments,
    parse_search_arguments,
    search_tool_definition,
)

ANSWER_DIRECTLY = "answer_directly"
ASK_CLARIFICATION = "ask_clarification"

PLANNER_INSTRUCTIONS = """당신은 다음 도구 중 하나를 고릅니다. 고르지 않은 채 문장만 쓰지 않습니다.

- answer_directly: 검색 없이 답할 수 있을 때. text가 사용자에게 보입니다.
- search_disclosures: 공시에서 확인해야 할 때.
- ask_clarification: 어느 기업인지, 무엇을 묻는지 확인이 필요할 때.

입력의 question은 사용자가 친 원문입니다. 고치지 않습니다.
request_time은 Asia/Seoul 기준 질문 시각입니다. 검색 여부를 정할 때 이 시각을 사용합니다.
corpus_as_of는 이 색인에 들어 있는 공시의 기준 시각입니다.
그 뒤에 나온 공시는 색인에 없습니다.
available_filings는 검색 가능한 공시 목록입니다.
company_id는 시스템이 확정한 값입니다. 새 corp_code를 만들지 않습니다.
company_mention은 사용자가 쓴 회사 표면형만 넣습니다.

answer_directly로 답할 수 있는 경우:
- request_time보다 미래의 확정 실적. 그 기간은 아직 실현되지 않았습니다. 숫자를 만들지 않습니다.
- 인사, 감사, 날씨, 시스템 기능 설명.
- 기업 공시 사실이 필요 없는 안내.

search_disclosures가 필요한 경우:
- 이미 지난 기간의 실적, 사업 내용, 위험.
- 목표, 계획, 전망. 미래 연도라도 현재 공시에 적혀 있을 수 있으므로 연도만으로 거절하지 않습니다.

period와 report_types는 저장소 필터입니다.
연간 실적은 보통 사업보고서입니다.
분기와 반기는 그 기간만 지정합니다.
최근 실적은 mode "latest"로 가장 늦은 보고기간을 고릅니다.
이유나 배경은 해당 연도 바깥의 후속 공시에 있을 수 있습니다.
prior_tool_results가 부족하면 필터를 넓혀 한 번 더 호출할 수 있습니다.
search_calls_remaining이 0이면 search_disclosures를 고르지 않습니다.
그때는 answer_directly 또는 ask_clarification로 끝냅니다.
prior_tool_results의 status가 evidence_required이면 직전 답은 전달되지 않았습니다.
그 기업의 공시 사실을 근거 없이 단정하지 않습니다. 사용자 문장을 공시 근거로 쓰지 않습니다.
검색이 남아 있으면 search_disclosures를 고를 수 있습니다.
없으면 확인 질문이나 근거 없는 보류를 고릅니다.
company_id는 입력에 있는 값만 넣습니다.

prior_tool_results는 이 색인의 상태입니다. 외부 전체의 부재가 아닙니다.
status가 no_matching_filing이면 색인에서 그 조건을 찾지 못한 것입니다.
DART 전체나 세상 전체에 공시가 없다고 말하지 않습니다.
invalid_arguments와 tool_failure의 내부 문장은 사용자에게 옮기지 않습니다.
그 경우 answer_directly로 색인 범위 안에서만 안내합니다.
"""


@dataclass(frozen=True)
class PlannerContext:
    question: str
    request_time: Mapping[str, str]
    corpus_as_of: str | None
    company_id: str
    company_name: str
    filings: tuple[FilingPeriod, ...]
    prior_results: tuple[Mapping[str, Any], ...]
    max_top_k: int
    company_resolution: Mapping[str, Any] | None = None
    search_calls_remaining: int = 2


@dataclass(frozen=True)
class PlannerAction:
    arguments: SearchToolArguments | None
    raw: dict[str, Any] | None
    schema_error: str | None = None
    usage: dict[str, int] | None = None
    latency_ms: float | None = None
    kind: str = "search_disclosures"
    user_text: str | None = None
    company_mention: str | None = None

    @property
    def requests_tool(self) -> bool:
        return self.kind == "search_disclosures" and (
            self.raw is not None or self.arguments is not None
        )

    @classmethod
    def stop(
        cls,
        *,
        usage: dict[str, int] | None = None,
        latency_ms: float | None = None,
    ) -> PlannerAction:
        return cls(None, None, usage=usage, latency_ms=latency_ms, kind="stop")

    @classmethod
    def direct(
        cls,
        text: str,
        *,
        company_mention: str | None = None,
        usage: dict[str, int] | None = None,
        latency_ms: float | None = None,
    ) -> PlannerAction:
        mention = company_mention or None
        return cls(
            None,
            {"action": ANSWER_DIRECTLY, "text": text, "company_mention": mention},
            usage=usage,
            latency_ms=latency_ms,
            kind=ANSWER_DIRECTLY,
            user_text=text,
            company_mention=mention,
        )

    @classmethod
    def clarify(
        cls,
        text: str,
        *,
        company_mention: str | None = None,
        usage: dict[str, int] | None = None,
        latency_ms: float | None = None,
    ) -> PlannerAction:
        mention = company_mention or None
        return cls(
            None,
            {"action": ASK_CLARIFICATION, "text": text, "company_mention": mention},
            usage=usage,
            latency_ms=latency_ms,
            kind=ASK_CLARIFICATION,
            user_text=text,
            company_mention=mention,
        )


class SearchPlanner(Protocol):
    uses_llm: bool

    def next_action(self, context: PlannerContext) -> PlannerAction: ...


class RawQueryPlanner:
    """One search of the unchanged question. No period is inferred."""

    uses_llm = False

    def __init__(self) -> None:
        self._sent = False

    def next_action(self, context: PlannerContext) -> PlannerAction:
        if self._sent:
            return PlannerAction.stop()
        self._sent = True
        raw = {
            "query": context.question,
            "company_id": context.company_id,
            "period": None,
            "filing_date_from": None,
            "filing_date_to": None,
            "report_types": None,
            "as_of": None,
            "top_k": context.max_top_k,
        }
        parsed, error = parse_search_arguments(raw)
        return PlannerAction(parsed, raw, schema_error=error)


class ScriptedPlanner:
    """Test double. Each step is a tool object, or None to stop."""

    uses_llm = False

    def __init__(self, steps: list[dict[str, Any] | None], *, uses_llm: bool = False) -> None:
        self.steps = steps
        self.contexts: list[PlannerContext] = []
        self._index = 0
        self.uses_llm = uses_llm

    def next_action(self, context: PlannerContext) -> PlannerAction:
        self.contexts.append(context)
        if self._index >= len(self.steps):
            return PlannerAction.stop()
        step = self.steps[self._index]
        self._index += 1
        if step is None:
            return PlannerAction.stop()
        kind = step.get("action")
        if kind == ANSWER_DIRECTLY:
            return PlannerAction.direct(
                str(step.get("text") or ""),
                company_mention=_mention(step.get("company_mention")),
            )
        if kind == ASK_CLARIFICATION:
            return PlannerAction.clarify(
                str(step.get("text") or ""),
                company_mention=_mention(step.get("company_mention")),
            )
        parsed, error = parse_search_arguments(step)
        return PlannerAction(parsed, dict(step), schema_error=error)


class OpenAISearchPlanner:
    """Responses API function call.

    The product model rejects ``temperature``, so the client omits it.
    """

    uses_llm = True

    def __init__(self, client: Any) -> None:
        self.client = client

    def next_action(self, context: PlannerContext) -> PlannerAction:
        payload = json.dumps(planner_payload(context), ensure_ascii=False)
        completion = self.client.complete_tool(
            payload,
            instructions=PLANNER_INSTRUCTIONS,
            tools=agent_tools(),
            tool_name=None,
        )
        usage = completion.usage
        latency = completion.latency_ms
        if not completion.function_name or not completion.function_arguments:
            return PlannerAction.stop(usage=usage, latency_ms=latency)
        try:
            raw = json.loads(completion.function_arguments)
        except json.JSONDecodeError:
            return PlannerAction(
                None,
                {"_unparsed": completion.function_arguments},
                schema_error="tool arguments are not JSON",
                usage=usage,
                latency_ms=latency,
                kind="stop",
            )
        if not isinstance(raw, dict):
            return PlannerAction(
                None,
                {"_unparsed": completion.function_arguments},
                schema_error="tool arguments must be an object",
                usage=usage,
                latency_ms=latency,
                kind="stop",
            )
        if completion.function_name == ANSWER_DIRECTLY:
            text = str(raw.get("text") or "").strip()
            if not text:
                return PlannerAction.stop(usage=usage, latency_ms=latency)
            return PlannerAction.direct(
                text,
                company_mention=_mention(raw.get("company_mention")),
                usage=usage,
                latency_ms=latency,
            )
        if completion.function_name == ASK_CLARIFICATION:
            text = str(raw.get("text") or "").strip()
            if not text:
                return PlannerAction.stop(usage=usage, latency_ms=latency)
            return PlannerAction.clarify(
                text,
                company_mention=_mention(raw.get("company_mention")),
                usage=usage,
                latency_ms=latency,
            )
        if completion.function_name != TOOL_NAME:
            return PlannerAction.stop(usage=usage, latency_ms=latency)
        parsed, error = parse_search_arguments(raw)
        return PlannerAction(parsed, raw, schema_error=error, usage=usage, latency_ms=latency)


def planner_payload(context: PlannerContext) -> dict[str, Any]:
    clock = {
        "timezone": context.request_time["timezone"],
        "display": context.request_time["display"],
        "iso": context.request_time["iso"],
        "weekday": context.request_time["weekday"],
    }
    return {
        "question": context.question,
        "request_time": clock,
        "corpus_as_of": context.corpus_as_of,
        "company_id": context.company_id,
        "company_name": context.company_name,
        "available_filings": [filing_catalog_row(item) for item in context.filings],
        "prior_tool_results": [dict(item) for item in context.prior_results],
        "max_top_k": context.max_top_k,
        "max_search_calls": 2,
        "search_calls_remaining": context.search_calls_remaining,
        "company_resolution": (
            None if context.company_resolution is None else dict(context.company_resolution)
        ),
    }


def agent_tools() -> list[dict[str, Any]]:
    return [search_tool_definition(), _answer_tool(), _clarify_tool()]


def _answer_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "name": ANSWER_DIRECTLY,
        "description": (
            "검색 없이 사용자에게 답합니다. "
            "아직 오지 않은 기간의 확정 실적, 일반 대화, 시스템 설명에만 사용합니다. "
            "공시에 있어야 하는 기업 사실을 만들지 않습니다."
        ),
        "parameters": _speech_parameters(),
        "strict": True,
    }


def _clarify_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "name": ASK_CLARIFICATION,
        "description": (
            "회사나 질문이 분명하지 않을 때 사용자에게 확인을 요청합니다. "
            "company_mention은 사용자가 쓴 표면형이며 corp_code가 아닙니다."
        ),
        "parameters": _speech_parameters(),
        "strict": True,
    }


def _speech_parameters() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "text": {"type": "string"},
            "company_mention": {"type": ["string", "null"]},
        },
        "required": ["text", "company_mention"],
    }


def _mention(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None

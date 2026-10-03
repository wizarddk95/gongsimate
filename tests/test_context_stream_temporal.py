"""Context budget, incremental streaming, and filing-metadata time policy."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from test_agent_product import events_of, registry, sample_hit, sample_result
from test_grounded_answer import ANNUAL_2025, _output

from disclosure_agent.agent.catalog import repo_root
from disclosure_agent.agent.context import ContextBuilder, evidence_token_cost
from disclosure_agent.agent.errors import OpenAIAPIError
from disclosure_agent.agent.evidence import EvidenceBuilder
from disclosure_agent.agent.llm import (
    GroundedClaim,
    GroundedModelOutput,
    OpenAIAnswerClient,
    OpenAIAnswerGenerator,
    ProviderDone,
    ProviderSnapshot,
)
from disclosure_agent.agent.record import RunLog
from disclosure_agent.agent.runtime import AgentRuntime
from disclosure_agent.agent.search import SearchDisclosuresRequest, SearchDisclosuresResult
from disclosure_agent.agent.settings import (
    DEFAULT_GENERATION_CONTEXT_TOKEN_BUDGET,
    DEFAULT_GENERATION_MAX_EVIDENCE,
    DEFAULT_RETRIEVAL_TOP_K,
    load_openai_config,
)
from disclosure_agent.agent.stream_parse import parse_partial_grounded
from disclosure_agent.agent.temporal import filing_period_from_listing, load_indexed_filing
from disclosure_agent.retrieval_unit import D2A_VERSION


def _hit(index: int, text: str):
    return replace(
        sample_hit(text),
        retrieval_unit_id=f"unit-{index}",
        canonical_chunk_id=f"chunk-{index}",
        rank=index,
    )


def _joined(events: list) -> str:
    return "".join(event.data["text"] for event in events if event.event == "answer.delta")


def test_context_budget_keeps_dense_rank_and_is_not_a_fixed_eight() -> None:
    evidence = EvidenceBuilder().build(
        [_hit(index, f"서로 다른 문장 {index} 입니다.") for index in range(1, 21)],
        company_name="이수페타시스",
    )
    assert len(evidence) == DEFAULT_RETRIEVAL_TOP_K
    wide = ContextBuilder(
        token_budget=DEFAULT_GENERATION_CONTEXT_TOKEN_BUDGET,
        max_evidence=DEFAULT_GENERATION_MAX_EVIDENCE,
    ).build(evidence)
    assert len(wide.evidence) == 20
    assert [item.evidence_id for item in wide.evidence] == [item.evidence_id for item in evidence]
    assert wide.token_count <= wide.token_budget

    first = evidence_token_cost(evidence[0])
    second = evidence_token_cost(evidence[1])
    prefix = ContextBuilder(token_budget=first + second, max_evidence=20).build(evidence)
    assert [item.evidence_id for item in prefix.evidence] == [
        evidence[0].evidence_id,
        evidence[1].evidence_id,
    ]
    assert prefix.token_count <= first + second

    capped = ContextBuilder(token_budget=10**9, max_evidence=3).build(evidence)
    assert [item.evidence_id for item in capped.evidence] == [
        item.evidence_id for item in evidence[:3]
    ]


def test_context_drops_exact_duplicates_and_records_the_split(tmp_path: Path) -> None:
    evidence = EvidenceBuilder().build(
        [_hit(1, "같은 문장입니다."), _hit(2, "같은 문장입니다."), _hit(3, "다른 문장입니다.")],
        company_name="이수페타시스",
    )
    selected = ContextBuilder(token_budget=10**9, max_evidence=20).build(evidence)
    assert [item.retrieval_unit_id for item in selected.evidence] == ["unit-1", "unit-3"]

    def search(request: object) -> SearchDisclosuresResult:
        del request
        return SearchDisclosuresResult(
            hits=[_hit(1, "같은 문장입니다."), _hit(2, "다른 문장입니다.")],
            retriever_type="dense",
            representation_version=D2A_VERSION,
            embedding_model="BAAI/bge-m3",
            company_id="00107613",
            corp_code="00107613",
        )

    client = _Complete(["ev-unused"])
    log = RunLog(tmp_path / "context.jsonl")
    runtime = AgentRuntime(
        registry=registry(),
        search=search,
        evidence_builder=EvidenceBuilder(),
        answer_generator=OpenAIAnswerGenerator(client),
        run_log=log,
        context_token_budget=1,
        filings=(ANNUAL_2025,),
    )
    events = events_of(runtime, "이수페타시스의 주요 사업이 뭐야?")
    record = json.loads(log.path.read_text(encoding="utf-8"))
    assert record["retrieved_candidate_count"] == 2
    assert record["context_evidence_count"] == 0
    assert record["context_evidence_ids"] == []
    assert set(record["context_evidence_ids"]) <= set(record["tool_calls"][0]["evidence_ids"])
    assert record["context_token_budget"] == 1
    assert client.inputs == []
    assert events[-1].data["status"] == "abstained"
    assert record["generator_type"] == "context_budget"


class _Complete:
    def __init__(self, ids: list[str]) -> None:
        self.model = "gpt-custom"
        self.sdk_version = "test-sdk"
        self.inputs: list[str] = []
        self.ids = ids

    def complete(self, user_input: str) -> GroundedModelOutput:
        self.inputs.append(user_input)
        return _output(*self.ids, answer="공시에 있습니다.")


class _Scripted:
    def __init__(self, evidence_id: str) -> None:
        self.model = "gpt-custom"
        self.sdk_version = "test-sdk"
        self.evidence_id = evidence_id
        self.finished = False
        self.inputs: list[str] = []

    def stream(self, user_input: str):
        self.inputs.append(user_input)
        opened = (
            '{"abstain":false,"claims":[{"evidence_ids":["'
            + self.evidence_id
            + '"],"text":"공시에 적힌'
        )
        yield ProviderSnapshot(opened)
        closed = opened + ' 내용입니다."}],"limitations":["범위는 이 공시입니다."]}'
        yield ProviderSnapshot(closed)
        self.finished = True
        yield ProviderDone(
            parsed=GroundedModelOutput(
                abstain=False,
                claims=[
                    GroundedClaim(
                        evidence_ids=[self.evidence_id],
                        text="공시에 적힌 내용입니다.",
                    )
                ],
                limitations=["범위는 이 공시입니다."],
            ),
            usage={"input_tokens": 5, "output_tokens": 4, "total_tokens": 9},
        )


def test_first_delta_arrives_before_the_model_finishes(tmp_path: Path) -> None:
    evidence = EvidenceBuilder().build([sample_hit("본문")], company_name="이수페타시스")
    client = _Scripted(evidence[0].evidence_id)
    runtime = AgentRuntime(
        registry=registry(),
        search=lambda request: sample_result("본문"),
        evidence_builder=EvidenceBuilder(),
        answer_generator=OpenAIAnswerGenerator(client),
        run_log=RunLog(tmp_path / "stream.jsonl"),
        filings=(ANNUAL_2025,),
    )
    events = []
    saw_early_delta = False
    for event in runtime.stream("이수페타시스의 주요 사업이 뭐야?"):
        events.append(event)
        if event.event == "answer.delta" and not saw_early_delta:
            assert client.finished is False
            assert event.data["text"] == "공시에 적힌"
            assert len(event.data["text"]) != 24
            saw_early_delta = True
    assert saw_early_delta is True
    names = [event.event for event in events]
    assert names.index("answer.delta") < names.index("citation") < names.index("run.completed")
    assert _joined(events) == "공시에 적힌 내용입니다."
    assert events[-1].data["answer"]["text"] == _joined(events)
    assert events[-1].data["answer"]["citations"][0]["citation_id"] == "1"
    assert events[-1].data["answer"]["citations"][0]["evidence_id"] == evidence[0].evidence_id
    assert "[1]" not in _joined(events)
    record = json.loads((tmp_path / "stream.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert record["status"] == "completed"
    assert record["partial_output_emitted"] is False
    assert record["first_answer_delta_ms"] <= record["generation_complete_ms"]


def test_claim_text_is_hidden_until_its_evidence_ids_validate() -> None:
    class EarlyText:
        model = "gpt-custom"
        sdk_version = "test-sdk"

        def stream(self, user_input: str):
            del user_input
            first = '{"abstain":false,"claims":[{"text":"비밀'
            yield ProviderSnapshot(first)
            yield ProviderSnapshot(first + '문장","evidence_ids":["ev-nope"]}]}')

    runtime = AgentRuntime(
        registry=registry(),
        search=lambda request: sample_result(),
        evidence_builder=EvidenceBuilder(),
        answer_generator=OpenAIAnswerGenerator(EarlyText()),
        filings=(ANNUAL_2025,),
    )
    events = events_of(runtime, "이수페타시스의 주요 사업이 뭐야?")
    assert events[-1].event == "error"
    assert events[-1].data["code"] == "citation_validation_failure"
    assert not any(event.event == "answer.delta" for event in events)
    assert "비밀" not in json.dumps([event.data for event in events], ensure_ascii=False)


def test_partial_provider_failure_is_not_a_completed_answer(tmp_path: Path) -> None:
    evidence = EvidenceBuilder().build([sample_hit("본문")], company_name="이수페타시스")

    class Boom:
        model = "gpt-custom"
        sdk_version = "test-sdk"

        def __init__(self, evidence_id: str) -> None:
            self.evidence_id = evidence_id

        def stream(self, user_input: str):
            del user_input
            snapshot = (
                '{"abstain":false,"claims":[{"evidence_ids":["'
                + self.evidence_id
                + '"],"text":"앞부분'
            )
            yield ProviderSnapshot(snapshot)
            raise OpenAIAPIError("response.failed")

    log = RunLog(tmp_path / "partial.jsonl")
    runtime = AgentRuntime(
        registry=registry(),
        search=lambda request: sample_result("본문"),
        evidence_builder=EvidenceBuilder(),
        answer_generator=OpenAIAnswerGenerator(Boom(evidence[0].evidence_id)),
        run_log=log,
        filings=(ANNUAL_2025,),
    )
    events = events_of(runtime, "이수페타시스의 주요 사업이 뭐야?")
    assert any(event.event == "answer.delta" and event.data["text"] == "앞부분" for event in events)
    assert events[-1].event == "error"
    assert events[-1].data["code"] == "openai_api_error"
    assert not any(event.event == "run.completed" for event in events)
    record = json.loads(log.path.read_text(encoding="utf-8"))
    assert record["partial_output_emitted"] is True
    assert record["partial_text"] == "앞부분"
    assert record["status"] == "partial_generation_failure"
    assert record["status"] != "completed"


def test_output_text_delta_is_forwarded_before_response_completed() -> None:
    parsed = GroundedModelOutput(
        abstain=False,
        claims=[GroundedClaim(evidence_ids=["ev-1"], text="앞부분입니다.")],
        limitations=[],
    )
    partial = '{"abstain":false,"claims":[{"evidence_ids":["ev-1"],"text":"앞'
    full = partial + '부분입니다."}],"limitations":[]}'
    events = [
        SimpleNamespace(type="response.output_text.delta", snapshot=partial, delta="앞"),
        SimpleNamespace(type="response.output_text.done", text=full, parsed=parsed),
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(
                usage=SimpleNamespace(input_tokens=1, output_tokens=2, total_tokens=3),
                output_parsed=parsed,
            ),
        ),
    ]

    class SDK:
        def __init__(self) -> None:
            self.kwargs: dict[str, object] | None = None
            self.completed = False
            self.responses = self

        def stream(self, **kwargs: object) -> SDK:
            self.kwargs = kwargs
            return self

        def __enter__(self) -> SDK:
            return self

        def __exit__(self, *args: object) -> bool:
            return False

        def __iter__(self):
            for event in events:
                if event.type == "response.completed":
                    self.completed = True
                yield event

    sdk = SDK()
    config = load_openai_config({"OPENAI_API_KEY": "sk-testsecretvalue123"})
    client = OpenAIAnswerClient(config, sdk=sdk)
    first = next(client.stream("{}"))
    assert isinstance(first, ProviderSnapshot)
    assert sdk.completed is False
    assert sdk.kwargs is not None
    assert sdk.kwargs["text_format"] is GroundedModelOutput
    assert sdk.kwargs["store"] is False


def test_schema_puts_evidence_ids_before_claim_text() -> None:
    schema = GroundedModelOutput.model_json_schema()
    assert list(schema["properties"]) == ["abstain", "claims", "limitations"]
    claim = schema["$defs"]["GroundedClaim"]["properties"]
    assert list(claim) == ["evidence_ids", "text"]


def test_partial_json_emits_closed_ids_before_the_object_finishes() -> None:
    snapshot = '{"abstain":false,"claims":[{"evidence_ids":["ev-1"],"text":"앞부분'
    progress = parse_partial_grounded(snapshot)
    assert progress.abstain is False
    assert progress.claims[0].ids_closed is True
    assert progress.claims[0].evidence_ids == ("ev-1",)
    assert progress.claims[0].text == "앞부분"
    assert progress.claims[0].closed is False
    closed = parse_partial_grounded(snapshot + '입니다."}]}')
    assert closed.claims[0].closed is True
    assert closed.claims[0].text == "앞부분입니다."


def test_exact_2035_with_no_listing_abstains_before_the_answer_model(tmp_path: Path) -> None:
    from disclosure_agent.agent.planner import ScriptedPlanner

    client = _Complete(["ev-ignored"])
    log = RunLog(tmp_path / "actual.jsonl")
    events = events_of(
        _runtime(
            client,
            sample_result("중장기 수치 2035가 본문에 있다."),
            log=log,
            planner=ScriptedPlanner([_exact_year("2035년 확정 매출", 2035)]),
        ),
        "2035년 이수페타시스 확정 매출 알려줘",
    )
    assert client.inputs == []
    assert events[-1].data["status"] == "no_matching_filing"
    assert events[-1].data["answer"]["citations"] == []
    assert "999" not in _joined(events)
    record = json.loads(log.path.read_text(encoding="utf-8"))
    assert record["generator_type"] == "tool_validation"
    assert record["search_called"] is False
    assert record["input_tokens"] is None


def test_forecast_2035_is_searched_instead_of_pre_abstained(tmp_path: Path) -> None:
    seen: list[SearchDisclosuresRequest] = []

    def search(request: SearchDisclosuresRequest) -> SearchDisclosuresResult:
        seen.append(request)
        return sample_result("2035년 매출 목표는 기재되어 있지 않습니다.")

    evidence = EvidenceBuilder().build(
        [sample_hit("2035년 매출 목표는 기재되어 있지 않습니다.")],
        company_name="이수페타시스",
    )
    client = _Complete([evidence[0].evidence_id])
    log = RunLog(tmp_path / "forecast.jsonl")
    events = events_of(
        _runtime(client, log=log, search_fn=search),
        "이수페타시스가 2035년 매출 목표를 제시했어?",
    )
    assert seen
    assert seen[0].top_k == DEFAULT_RETRIEVAL_TOP_K
    assert client.inputs
    assert events[-1].data["status"] == "completed"
    record = json.loads(log.path.read_text(encoding="utf-8"))
    assert record["temporal_fact_mode"] is None
    assert record["temporal_target_year"] is None
    assert record["generator_type"] == "openai"


def test_historical_year_is_not_forced_onto_the_filing_year(tmp_path: Path) -> None:
    seen: list[SearchDisclosuresRequest] = []

    def search(request: SearchDisclosuresRequest) -> SearchDisclosuresResult:
        seen.append(request)
        return sample_result("전기에 고객사 재고 조정이 있었다고 설명합니다.")

    evidence = EvidenceBuilder().build(
        [sample_hit("전기에 고객사 재고 조정이 있었다고 설명합니다.")],
        company_name="이수페타시스",
    )
    client = _Complete([evidence[0].evidence_id])
    events = events_of(
        _runtime(client, log=RunLog(tmp_path / "history.jsonl"), search_fn=search),
        "2024년 이수페타시스 매출이 왜 줄었다고 설명했어?",
    )
    assert seen[0].fiscal_year is None
    assert seen[0].query == "2024년 이수페타시스 매출이 왜 줄었다고 설명했어?"
    assert client.inputs
    assert events[-1].data["status"] != "no_matching_filing"


def test_listing_period_is_the_temporal_source_not_the_receipt_date() -> None:
    bare = filing_period_from_listing(
        {
            "report_nm": "사업보고서",
            "rcept_no": "20260323000847",
            "rcept_dt": "20260323",
            "rm": "연",
        }
    )
    assert bare.period_end_year is None
    assert bare.receipt_date == "20260323"
    indexed = load_indexed_filing(repo_root())
    assert indexed.receipt_no == "20260323000847"
    assert indexed.report_name == "사업보고서 (2025.12)"
    assert indexed.report_type == "사업보고서"
    assert indexed.receipt_date == "20260323"
    assert indexed.period_end_year == 2025
    assert indexed.period_end_month == 12
    import disclosure_agent.agent.answer as answer_mod

    assert not hasattr(answer_mod, "mvp_year_text_guard")
    assert not hasattr(answer_mod, "year_coverage_abstain")


def _runtime(
    client: _Complete,
    result: SearchDisclosuresResult | None = None,
    *,
    log: RunLog | None = None,
    search_fn=None,
    planner=None,
) -> AgentRuntime:
    def search(request: SearchDisclosuresRequest) -> SearchDisclosuresResult:
        del request
        return result or sample_result()

    return AgentRuntime(
        registry=registry(),
        search=search_fn or search,
        evidence_builder=EvidenceBuilder(),
        answer_generator=OpenAIAnswerGenerator(client),
        run_log=log,
        filings=(ANNUAL_2025,),
        planner=planner,
    )


def _exact_year(query: str, year: int) -> dict[str, object]:
    return {
        "query": query,
        "company_id": "00107613",
        "period": {
            "year": year,
            "quarter": None,
            "half": None,
            "mode": "exact",
            "year_from": None,
            "year_to": None,
        },
        "filing_date_from": None,
        "filing_date_to": None,
        "report_types": ["annual"],
        "as_of": None,
        "top_k": 20,
    }

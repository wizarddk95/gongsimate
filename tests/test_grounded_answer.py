"""OpenAI grounded answers use a fake client. These tests do not call the API."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import httpx2
import openai
import pytest
from pydantic import ValidationError
from test_agent_product import events_of, registry, sample_hit, sample_result

from disclosure_agent.agent.answer import GeneratedAnswer
from disclosure_agent.agent.citation import Citation, assign_citation_ids
from disclosure_agent.agent.errors import (
    CitationValidationFailure,
    OpenAIAuthError,
    RetrievalFailure,
    StructuredOutputInvalid,
)
from disclosure_agent.agent.evidence import EvidenceBuilder
from disclosure_agent.agent.llm import (
    ANSWER_INSTRUCTIONS,
    FORBIDDEN_INPUT_KEYS,
    GroundedClaim,
    GroundedModelOutput,
    OpenAIAnswerClient,
    OpenAIAnswerGenerator,
    build_llm_input,
    grounded_to_answer,
    normalize_provider_error,
)
from disclosure_agent.agent.record import RunLog
from disclosure_agent.agent.runtime import AgentRuntime
from disclosure_agent.agent.search import (
    BgeM3QueryEncoder,
    SearchDisclosuresRequest,
    SearchDisclosuresResult,
    SearchHit,
)
from disclosure_agent.agent.settings import AgentLimits, load_openai_config
from disclosure_agent.agent.temporal import filing_period_from_listing
from disclosure_agent.retrieval_unit import D2A_VERSION

SECRET = "sk-testsecretvalue123"
ANNUAL_2025 = filing_period_from_listing(
    {
        "report_nm": "사업보고서 (2025.12)",
        "rcept_no": "20260323000847",
        "rcept_dt": "20260323",
        "rm": "연",
    }
)


class FakeAnswerClient:
    def __init__(
        self,
        outputs: list[object],
        *,
        model: str = "gpt-custom",
        usage: dict[str, int] | None = None,
        latency_ms: float | None = None,
    ) -> None:
        self.outputs = list(outputs)
        self.model = model
        self.sdk_version = "test-sdk"
        self.inputs: list[str] = []
        self.last_usage = usage
        self.last_latency_ms = latency_ms

    def complete(self, user_input: str) -> GroundedModelOutput:
        self.inputs.append(user_input)
        item = self.outputs.pop(0)
        if isinstance(item, Exception):
            raise item
        assert isinstance(item, GroundedModelOutput)
        return item


class RecordingSDK:
    def __init__(self, outcome: object) -> None:
        self.outcome = outcome
        self.kwargs: dict[str, object] | None = None
        self.responses = self

    def parse(self, **kwargs: object) -> object:
        self.kwargs = kwargs
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def _runtime(
    client: FakeAnswerClient,
    search_result: SearchDisclosuresResult | None = None,
    *,
    limits: AgentLimits | None = None,
    log: RunLog | None = None,
    secrets: tuple[str, ...] = (),
    planner: object | None = None,
) -> AgentRuntime:
    def search(request: object) -> SearchDisclosuresResult:
        del request
        return search_result or sample_result("2025년 인쇄회로기판 수요가 늘었습니다.")

    return AgentRuntime(
        registry=registry(),
        search=search,
        evidence_builder=EvidenceBuilder(),
        answer_generator=OpenAIAnswerGenerator(client),
        limits=limits or AgentLimits(),
        run_log=log,
        redact_secrets=secrets,
        filings=(ANNUAL_2025,),
        planner=planner,  # type: ignore[arg-type]
    )


def _output(
    *ids: str,
    answer: str = "기판 수요가 늘었습니다.",
    abstain: bool = False,
) -> GroundedModelOutput:
    if abstain:
        claims = [GroundedClaim(evidence_ids=[], text=answer)] if answer else []
    else:
        claims = [GroundedClaim(evidence_ids=list(ids), text=answer)]
    return GroundedModelOutput(abstain=abstain, claims=claims, limitations=[])


def _statuses(events: list) -> list[str]:
    return [event.data["status"] for event in events if event.event == "agent.status"]


def _joined_deltas(events: list) -> str:
    return "".join(event.data["text"] for event in events if event.event == "answer.delta")


def test_llm_input_contains_evidence_and_not_gold() -> None:
    hit = sample_hit("당기 매출은 기판 수요로 증가했습니다.")
    evidence = EvidenceBuilder().build([hit], company_name="이수페타시스")
    payload = build_llm_input(
        "이수페타시스의 매출은?",
        evidence,
        company_name="이수페타시스",
        fiscal_year=2025,
    )
    blob = json.dumps(payload, ensure_ascii=False)
    assert payload["evidence"][0]["evidence_id"] == evidence[0].evidence_id
    assert payload["evidence"][0]["text"] == hit.text
    assert payload["company_name"] == "이수페타시스"
    assert "corp_code" not in blob
    assert "company_id" not in blob
    for key in FORBIDDEN_INPUT_KEYS:
        assert key not in blob
    for key in ("gold", "evidence_class", "required_claim", "answer_valid_evidence", "question_id"):
        assert key not in ANSWER_INSTRUCTIONS


def test_valid_evidence_ids_succeed_and_runtime_numbers_citations(tmp_path: Path) -> None:
    first = sample_hit("2025년 매출은 기판 수요로 늘었습니다.")
    second = sample_hit("주요 고객향 수주가 확대되었습니다.")
    second = SearchHit(**{**second.__dict__, "retrieval_unit_id": "unit-2", "rank": 2})
    result = SearchDisclosuresResult(
        hits=[first, second],
        retriever_type="dense",
        representation_version=D2A_VERSION,
        embedding_model="BAAI/bge-m3",
        company_id="00107613",
        corp_code="00107613",
    )
    built = EvidenceBuilder().build([second, first], company_name="이수페타시스")
    client = FakeAnswerClient([_output(built[0].evidence_id, built[1].evidence_id)])
    log = RunLog(tmp_path / "runs.jsonl")
    events = events_of(
        _runtime(client, result, log=log),
        "이수페타시스의 2025년 매출 증가 이유가 뭐야?",
    )
    assert client.inputs
    sent = json.loads(client.inputs[0])
    assert [item["evidence_id"] for item in sent["evidence"]] == [
        built[1].evidence_id,
        built[0].evidence_id,
    ]
    assert events[-1].data["status"] == "completed"
    citations = events[-1].data["answer"]["citations"]
    assert [item["citation_id"] for item in citations] == ["1", "2"]
    assert citations[0]["evidence_id"] == built[0].evidence_id
    assert "[1]" not in _joined_deltas(events)
    citation_events = [event for event in events if event.event == "citation"]
    assert citation_events[0].data["citation_id"] == "1"
    assert citation_events[0].data["receipt_no"] == "20260323000847"
    assert citation_events[0].data["document_name"] == "사업보고서"
    assert citation_events[0].data["company_name"] == "이수페타시스"
    assert "section_path" in citation_events[0].data
    assert "excerpt" in citation_events[0].data
    assert "source_locator" in citation_events[0].data
    record = json.loads(log.path.read_text(encoding="utf-8"))
    assert record["generator_type"] == "openai"
    assert record["model"] == "gpt-custom"
    assert record["prompt_evidence_count"] == 2
    assert record["generation_ms"] is not None
    assert record["retrieval_ms"] is not None
    assert record["company_resolution_ms"] is not None
    assert record["total_ms"] is not None


def test_hallucinated_and_other_run_evidence_ids_are_rejected() -> None:
    evidence = EvidenceBuilder().build([sample_hit()], company_name="이수페타시스")
    current = evidence[0].evidence_id
    for bad_id in ("ev-made-up", "ev-from-another-run"):
        client = FakeAnswerClient([_output(bad_id)])
        events = events_of(_runtime(client), "이수페타시스의 주요 사업이 뭐야?")
        assert events[-1].event == "error"
        assert events[-1].data["code"] == "citation_validation_failure"
        assert not any(event.event == "answer.delta" for event in events)
        assert current != bad_id
        assert len(client.inputs) == 1


def test_invalid_structured_output_retries_once_then_succeeds() -> None:
    evidence = EvidenceBuilder().build(
        [sample_hit("2025년 기판을 생산합니다.")],
        company_name="이수페타시스",
    )
    client = FakeAnswerClient(
        [
            StructuredOutputInvalid("schema mismatch"),
            _output(evidence[0].evidence_id, answer="인쇄회로기판을 생산합니다."),
        ]
    )
    runtime = _runtime(client)
    calls_before_delta = None
    events = []
    for event in runtime.stream("이수페타시스의 주요 사업이 뭐야?"):
        events.append(event)
        if event.event == "answer.delta" and calls_before_delta is None:
            calls_before_delta = len(client.inputs)
    assert calls_before_delta == 2
    assert len(client.inputs) == 2
    assert events[-1].data["status"] == "completed"
    first_delta = next(index for index, event in enumerate(events) if event.event == "answer.delta")
    assert _statuses(events).index("generating_answer") < first_delta


def test_structured_output_retry_limit_and_llm_budget() -> None:
    client = FakeAnswerClient(
        [
            StructuredOutputInvalid("first"),
            StructuredOutputInvalid("second"),
        ]
    )
    events = events_of(_runtime(client), "이수페타시스의 주요 사업이 뭐야?")
    assert len(client.inputs) == 2
    assert events[-1].data["code"] == "structured_output_invalid"
    assert not any(event.event == "answer.delta" for event in events)

    limited = FakeAnswerClient([StructuredOutputInvalid("first"), _output("ev-unused")])
    limited_events = events_of(
        _runtime(limited, limits=AgentLimits(max_search_calls=2, max_llm_calls=1)),
        "이수페타시스의 주요 사업이 뭐야?",
    )
    assert len(limited.inputs) == 1
    assert limited_events[-1].data["code"] == "llm_limit_exceeded"
    assert not any(event.event == "answer.delta" for event in limited_events)


def test_provider_errors_map_without_keeping_the_key() -> None:
    request = httpx2.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx2.Response(401, request=request)
    auth = openai.AuthenticationError(f"bad {SECRET}", response=response, body={"error": SECRET})
    mapped = normalize_provider_error(auth)
    assert isinstance(mapped, OpenAIAuthError)
    assert SECRET not in mapped.detail

    timeout = normalize_provider_error(openai.APITimeoutError(request))
    assert timeout.code == "openai_timeout"
    rate_response = httpx2.Response(429, request=request)
    rate = normalize_provider_error(
        openai.RateLimitError(f"slow {SECRET}", response=rate_response, body=None)
    )
    assert rate.code == "openai_rate_limit"
    assert SECRET not in rate.detail
    server_response = httpx2.Response(500, request=request)
    api = normalize_provider_error(
        openai.InternalServerError("down", response=server_response, body=None)
    )
    assert api.code == "openai_api_error"
    try:
        GroundedModelOutput.model_validate({"answer": 1})
    except ValidationError as exc:
        invalid = normalize_provider_error(exc)
    assert invalid.code == "structured_output_invalid"

    sdk = RecordingSDK(auth)
    config = load_openai_config({"OPENAI_API_KEY": SECRET, "OPENAI_MODEL": "gpt-6-luna"})
    client = OpenAIAnswerClient(config, sdk=sdk)
    with pytest.raises(OpenAIAuthError) as raised:
        client.complete('{"question":"안녕"}')
    assert SECRET not in raised.value.detail
    assert sdk.kwargs is not None
    assert sdk.kwargs["model"] == "gpt-6-luna"
    assert sdk.kwargs["text_format"] is GroundedModelOutput
    assert sdk.kwargs["instructions"] == ANSWER_INSTRUCTIONS
    assert "temperature" not in sdk.kwargs
    encoded = json.dumps(sdk.kwargs, default=str)
    assert SECRET not in encoded


def test_tool_call_omits_unsupported_temperature() -> None:
    class SDK:
        def __init__(self) -> None:
            self.kwargs: dict[str, object] | None = None
            self.responses = self

        def create(self, **kwargs: object) -> SimpleNamespace:
            self.kwargs = kwargs
            return SimpleNamespace(output=[], usage=None)

    sdk = SDK()
    config = load_openai_config({"OPENAI_API_KEY": SECRET, "OPENAI_MODEL": "gpt-6-luna"})
    client = OpenAIAnswerClient(config, sdk=sdk)
    client.complete_tool(
        "{}",
        instructions="plan",
        tools=[],
        tool_name="search_disclosures",
    )
    assert sdk.kwargs is not None
    assert "temperature" not in sdk.kwargs
    assert sdk.kwargs["model"] == "gpt-6-luna"
    assert sdk.kwargs["tool_choice"] == "auto"


def test_api_key_is_absent_from_the_run_log(tmp_path: Path) -> None:
    class Leaky:
        generator_type = "openai"
        model = "gpt-6-luna"
        sdk_version = "test-sdk"

        def generate(self, question: str, evidence: list, **kwargs: object) -> GeneratedAnswer:
            del question, evidence, kwargs
            raise RuntimeError(f"provider blew up with {SECRET}")

    log = RunLog(tmp_path / "runs.jsonl")

    def search(request: object) -> SearchDisclosuresResult:
        del request
        return sample_result()

    runtime = AgentRuntime(
        registry=registry(),
        search=search,
        evidence_builder=EvidenceBuilder(),
        answer_generator=Leaky(),
        run_log=log,
        redact_secrets=(SECRET,),
    )
    events = events_of(runtime, "이수페타시스의 주요 사업이 뭐야?")
    assert events[-1].data["code"] == "internal_error"
    assert SECRET not in events[-1].data["message"]
    raw = log.path.read_text(encoding="utf-8")
    assert SECRET not in raw
    assert "[REDACTED]" in raw
    record = json.loads(raw)
    assert isinstance(record["generation_ms"], float)
    assert record["prompt_evidence_count"] == 1


def test_model_id_comes_from_config() -> None:
    default = load_openai_config({})
    assert default.model == "gpt-6-luna"
    assert default.configured is False
    custom = load_openai_config({"OPENAI_API_KEY": SECRET, "OPENAI_MODEL": "gpt-6.1-sol"})
    assert custom.model == "gpt-6.1-sol"
    assert SECRET not in repr(custom)
    sdk = RecordingSDK(SimpleNamespace(output_parsed=_output("ev-1")))
    client = OpenAIAnswerClient(custom, sdk=sdk)
    parsed = client.complete("{}")
    assert parsed.claims[0].text
    assert sdk.kwargs is not None
    assert sdk.kwargs["model"] == "gpt-6.1-sol"


def test_finished_answer_is_one_delta_not_a_fixed_chunk() -> None:
    evidence = EvidenceBuilder().build(
        [sample_hit("2025년 기판을 생산합니다.")],
        company_name="이수페타시스",
    )
    long_answer = "인쇄회로기판을 생산한다는 설명이 공시에 있습니다. " * 3
    client = FakeAnswerClient([_output(evidence[0].evidence_id, answer=long_answer)])
    events = events_of(_runtime(client), "이수페타시스의 주요 사업이 뭐야?")
    deltas = [event for event in events if event.event == "answer.delta"]
    assert len(deltas) == 1
    assert deltas[0].data["text"] == long_answer.strip()
    assert len(deltas[0].data["text"]) > 24
    names = [event.event for event in events]
    assert names.index("answer.delta") < names.index("citation")
    assert names[-1] == "run.completed"
    assert "[1]" not in long_answer


def test_abstain_has_no_fabricated_citation() -> None:
    evidence = EvidenceBuilder().build(
        [sample_hit("2025년 매출액은 사업보고서에 기재되어 있습니다.")],
        company_name="이수페타시스",
    )
    client = FakeAnswerClient(
        [
            GroundedModelOutput(
                abstain=True,
                claims=[
                    GroundedClaim(
                        evidence_ids=[],
                        text="확정 수치로 단정할 근거가 부족합니다.",
                    )
                ],
                limitations=["질문에 대한 직접 수치가 없습니다."],
            )
        ]
    )
    events = events_of(_runtime(client), "이수페타시스의 2025년 매출이 얼마야?")
    assert events[-1].data["status"] == "abstained"
    assert events[-1].data["answer"]["citations"] == []
    assert not any(event.event == "citation" for event in events)
    assert "100" not in _joined_deltas(events)
    assert evidence[0].evidence_id not in _joined_deltas(events)


def test_missing_year_does_not_call_the_model_or_invent_a_figure() -> None:
    from disclosure_agent.agent.planner import ScriptedPlanner

    client = FakeAnswerClient(
        [_output("ev-ignored", answer="2035년 확정 매출은 999조원입니다.")]
    )
    events = events_of(
        _runtime(
            client,
            planner=ScriptedPlanner(
                [
                    {
                        "query": "2035년 확정 매출",
                        "company_id": "00107613",
                        "period": {
                            "year": 2035,
                            "quarter": None,
                            "half": None,
                            "mode": "exact",
                            "year_from": None,
                            "year_to": None,
                        },
                        "report_types": ["annual"],
                        "top_k": 20,
                    }
                ]
            ),
        ),
        "2035년 이수페타시스 확정 매출 알려줘",
    )
    assert client.inputs == []
    assert events[-1].data["action"] == "abstain"
    assert events[-1].data["answer"]["citations"] == []
    assert not any(event.event == "citation" for event in events)
    text = _joined_deltas(events)
    assert "999" not in text
    assert "2035" in text
    assert events[-1].data["status"] == "no_matching_filing"
    assert "찾지 못했습니다" in events[-1].data["answer"]["text"]


def test_deterministic_fallback_is_labeled_and_api_failure_is_not_disguised(tmp_path: Path) -> None:
    from disclosure_agent.agent.runtime import AgentRuntime as Runtime

    log = RunLog(tmp_path / "fallback.jsonl")

    def search(request: object) -> SearchDisclosuresResult:
        del request
        return sample_result()

    runtime = Runtime(
        registry=registry(),
        search=search,
        evidence_builder=EvidenceBuilder(),
        run_log=log,
    )
    events_of(runtime, "이수페타시스의 주요 사업이 뭐야?")
    record = json.loads(log.path.read_text(encoding="utf-8"))
    assert record["generator_type"] == "deterministic_fallback"

    failing = FakeAnswerClient([OpenAIAuthError("openai authentication failed")])
    failed = events_of(_runtime(failing), "이수페타시스의 주요 사업이 뭐야?")
    assert failed[-1].data["code"] == "openai_auth_error"
    assert "공시에서 확인한 내용입니다" not in _joined_deltas(failed)
    assert len(failing.inputs) == 1


def test_missing_retrieval_artifact_fails_before_the_model() -> None:
    client = FakeAnswerClient([_output("ev-should-not-run")])

    def search(request: object) -> SearchDisclosuresResult:
        del request
        raise RetrievalFailure(
            "D2a embeddings are missing: evaluation/runs/isu_petasys_v0_3_dense_d2a/embeddings.f32"
        )

    runtime = AgentRuntime(
        registry=registry(),
        search=search,
        evidence_builder=EvidenceBuilder(),
        answer_generator=OpenAIAnswerGenerator(client),
    )
    events = events_of(runtime, "이수페타시스의 주요 사업이 뭐야?")
    assert events[-1].data["code"] == "retrieval_failure"
    assert client.inputs == []


def test_query_encoder_loads_the_model_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_load(device: str) -> object:
        calls.append(device)
        return {"device": device}

    def fake_encode(model: object, texts: list[str], batch_size: int) -> list[str]:
        del model, batch_size
        return texts

    monkeypatch.setattr("disclosure_agent.agent.search.load_dense_model", fake_load)
    monkeypatch.setattr("disclosure_agent.agent.search.encode_texts", fake_encode)
    monkeypatch.setattr("disclosure_agent.agent.search.resolve_device", lambda requested: "cpu")
    encoder = BgeM3QueryEncoder()
    assert encoder.encode("하나") == "하나"
    assert encoder.encode("둘") == "둘"
    assert calls == ["cpu"]


def test_retrieval_candidates_stay_distinct_from_context(tmp_path: Path) -> None:
    from dataclasses import replace

    from disclosure_agent.agent.settings import DEFAULT_RETRIEVAL_TOP_K

    hits = [
        replace(
            sample_hit(f"2025년 사업 설명 {index}번 문장입니다."),
            retrieval_unit_id=f"unit-{index}",
            canonical_chunk_id=f"chunk-{index}",
            rank=index + 1,
        )
        for index in range(DEFAULT_RETRIEVAL_TOP_K)
    ]
    result = SearchDisclosuresResult(
        hits=hits,
        retriever_type="dense",
        representation_version=D2A_VERSION,
        embedding_model="BAAI/bge-m3",
        company_id="00107613",
        corp_code="00107613",
    )
    requests: list[SearchDisclosuresRequest] = []

    def search(request: SearchDisclosuresRequest) -> SearchDisclosuresResult:
        requests.append(request)
        return result

    class EchoClient:
        def __init__(self) -> None:
            self.model = "gpt-custom"
            self.sdk_version = "test-sdk"
            self.inputs: list[str] = []

        def complete(self, user_input: str) -> GroundedModelOutput:
            self.inputs.append(user_input)
            payload = json.loads(user_input)
            evidence_id = payload["evidence"][0]["evidence_id"]
            return _output(evidence_id, answer="공시에 적힌 사업 설명입니다.")

    client = EchoClient()
    log = RunLog(tmp_path / "depth.jsonl")
    runtime = AgentRuntime(
        registry=registry(),
        search=search,
        evidence_builder=EvidenceBuilder(),
        answer_generator=OpenAIAnswerGenerator(client),
        run_log=log,
        filings=(ANNUAL_2025,),
    )
    events = events_of(runtime, "이수페타시스의 주요 사업이 뭐야?")
    assert requests[0].top_k == DEFAULT_RETRIEVAL_TOP_K
    sent = json.loads(client.inputs[0])
    sent_ids = [item["evidence_id"] for item in sent["evidence"]]
    built = EvidenceBuilder().build(hits, company_name="이수페타시스")
    assert len(sent_ids) == DEFAULT_RETRIEVAL_TOP_K
    assert len(sent_ids) != 8
    assert sent_ids == [item.evidence_id for item in built]
    record = json.loads(log.path.read_text(encoding="utf-8"))
    assert record["retrieved_candidate_count"] == DEFAULT_RETRIEVAL_TOP_K
    assert record["context_evidence_count"] == DEFAULT_RETRIEVAL_TOP_K
    assert record["context_evidence_ids"] == sent_ids
    assert record["retrieved_retrieval_unit_ids"] == [hit.retrieval_unit_id for hit in hits]
    assert set(record["context_evidence_ids"]) <= set(record["tool_calls"][0]["evidence_ids"])
    assert record["context_token_count"] <= record["context_token_budget"]
    assert record["context_token_counter"] == "utf8_bytes_ceil_div_2"
    assert "generation_evidence_limit" not in record["tool_calls"][0]
    assert record["evidence_ids"] == sent_ids
    assert record["model_id"] == "gpt-custom"
    for field in (
        "company_resolution_ms",
        "retrieval_ms",
        "generation_ms",
        "validation_ms",
        "total_ms",
        "generation_start_ms",
        "first_answer_delta_ms",
        "generation_complete_ms",
    ):
        assert isinstance(record[field], float)
    assert record["first_answer_delta_ms"] <= record["generation_complete_ms"]
    status_at = next(
        index
        for index, event in enumerate(events)
        if event.event == "agent.status" and event.data["status"] == "generating_answer"
    )
    delta_at = next(index for index, event in enumerate(events) if event.event == "answer.delta")
    assert status_at < delta_at
    assert events[-1].data["status"] == "completed"
    assert events[-1].data["answer"]["citations"][0]["evidence_id"] == sent_ids[0]


def test_usage_is_recorded_only_from_supplied_fields(tmp_path: Path) -> None:
    evidence = EvidenceBuilder().build(
        [sample_hit("2025년 기판을 생산합니다.")],
        company_name="이수페타시스",
    )
    client = FakeAnswerClient(
        [_output(evidence[0].evidence_id, answer="인쇄회로기판을 생산합니다.")],
        usage={"input_tokens": 11, "output_tokens": 7, "total_tokens": 18, "ignored": 1},
        latency_ms=12.5,
    )
    log = RunLog(tmp_path / "usage.jsonl")
    events = events_of(
        _runtime(client, log=log),
        "이수페타시스의 주요 사업이 뭐야?",
    )
    assert events[-1].data["status"] == "completed"
    record = json.loads(log.path.read_text(encoding="utf-8"))
    assert record["input_tokens"] == 11
    assert record["output_tokens"] == 7
    assert record["total_tokens"] == 18
    assert "ignored" not in record
    assert record["generation_ms"] == 12.5
    assert isinstance(record["validation_ms"], float)


def test_period_label_allows_a_year_that_the_body_does_not_spell(tmp_path: Path) -> None:
    text = "제54기 당기 매출은 전기 대비 증가했다. 보고기간말 현재 수주잔고가 있다."
    evidence = EvidenceBuilder().build([sample_hit(text)], company_name="이수페타시스")
    client = FakeAnswerClient(
        [_output(evidence[0].evidence_id, answer="당기 매출이 전기보다 늘었습니다.")]
    )
    log = RunLog(tmp_path / "year.jsonl")
    events = events_of(
        _runtime(client, sample_result(text), log=log),
        "이수페타시스의 2025년 매출이 얼마야?",
    )
    assert client.inputs
    assert "999" not in _joined_deltas(events)
    record = json.loads(log.path.read_text(encoding="utf-8"))
    assert record["generator_type"] == "openai"
    assert record["temporal_fact_mode"] is None
    assert record["temporal_target_year"] is None
    assert events[-1].data["status"] == "completed"


def test_assign_citation_ids_follow_first_appearance() -> None:
    citations = assign_citation_ids(["ev-b", "ev-a", "ev-b"])
    assert [(item.citation_id, item.evidence_id) for item in citations] == [
        ("1", "ev-b"),
        ("2", "ev-a"),
    ]


def test_grounded_answer_rejects_unknown_id_before_success() -> None:
    with pytest.raises(CitationValidationFailure):
        grounded_to_answer(_output("ev-nope"), {"ev-real"})
    answer = grounded_to_answer(_output("ev-real", answer="근거에 있습니다."), {"ev-real"})
    assert answer.citations == [Citation("1", "ev-real")]
    assert "[1]" not in answer.text

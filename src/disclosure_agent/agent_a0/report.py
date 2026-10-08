# ruff: noqa: E501
"""A0 report. Numbers come from the summary. The architecture notes are the inspected product path."""

from __future__ import annotations

from typing import Any

_HYPOTHESIS = {
    "clarification_failure": (
        "모호한 회사명에서 planner가 company_mention으로 한 회사를 고르는지 확인한다. "
        "다음 실험은 그 추측을 막는지에만 두고, 검색 정책은 바꾸지 않는다."
    ),
    "unnecessary_search": (
        "검색이 필요 없는 질문에서 search_disclosures가 호출되는지 확인한다. "
        "다음 실험은 그 결정만 줄인다."
    ),
    "missed_search": (
        "공시 사실 질문에서 검색 없이 답하거나 바로 거절하는지 확인한다. "
        "다음 실험은 그 누락만 본다."
    ),
    "tool_argument_failure": (
        "회사 또는 보고기간 인자가 질문과 어긋나는지 확인한다. "
        "사건 연도를 접수 연도 필터로 바꾸지는 않는다."
    ),
    "retrieval_failure": (
        "제품 검색은 Dense-D2a다. 근거가 비는 실패가 많으면, "
        "H10 벤치마크와 제품 경로의 차이를 다음에 따로 본다. TEST 55는 다시 열지 않는다."
    ),
    "unsupported_claim": (
        "검색 근거 밖의 공시 수치가 답에 들어가는지 확인한다. "
        "다음 실험은 그 claim만 줄이고 G0 점수는 다시 올리지 않는다."
    ),
    "over_abstention": (
        "근거가 있는데도 답변을 전부 보류하는지 확인한다. "
        "확인된 사실만 남기는 부분 답변을 다음에 본다."
    ),
    "under_abstention": (
        "색인으로 확인되지 않는 사실을 단정하는지 확인한다. 그 단정을 막는 쪽이 다음 실험이다."
    ),
    "citation_failure": ("최종 답의 evidence_id가 그 실행의 근거 밖에 있는지 확인한다."),
    "generation_failure": ("도구 선택은 맞는데 최종 문장이 requirement를 놓치는지 확인한다."),
    "runtime_failure": "실행 오류의 예외를 다음 단계에서 재현한다.",
    "timeout": "요청 시간 제한에 걸리는 질문을 다음 단계에서 재현한다.",
        "budget_exhaustion": (
            "회사 해석이 LLM 한도 1회를 먼저 쓴다. "
            "검색 두 번 뒤에 planner가 다시 호출되면 답 생성 전에 멈춘다. "
            "다음 실험은 이 한도 회계만 모델 호출과 분리하는 가설을 본다."
        ),
    "unnecessary_retry": "같은 검색 인자가 반복되는 호출을 다음 단계에서 본다.",
}


def render_report(summary: dict[str, Any]) -> str:
    task = summary["task"]
    decision = summary["search_decision"]
    arguments = summary["arguments"]
    grounding = summary["grounding"]
    latency = summary["latency_ms"]
    cost = summary["cost"]
    failures = summary["failures"]
    examples = summary["representative_failures"]
    hypothesis = _hypothesis(failures)
    lines = [
        "# GongsiMate A0 Agent Baseline",
        "",
        "## 1. Executive Summary",
        "",
        (
            f"기존 제품 Agent를 30개 DEV 질문에 그대로 실행했다. "
            f"Task 결과는 성공 {task['success']}, 부분 성공 {task['partial_success']}, "
            f"적절한 재질문 {task['appropriate_clarification']}, "
            f"적절한 보류 {task['appropriate_abstention']}, "
            f"실패 {task['failure']}, 미판정 {task['unjudged']}이다."
        ),
        (
            f"검색 필요 여부 판단은 {decision['correct']}/{decision['denominator']}이다. "
            f"필요 검색 누락 {decision['missed_required_search']}, "
            f"불필요 검색 {decision['unnecessary_search']}이다."
        ),
        "제품 검색은 H10이 아니라 Dense-D2a다. G0 점수와 이 점수는 비교하지 않는다.",
        "",
        "## 2. Existing Agent Architecture",
        "",
        "FACT: `OpenAISearchPlanner`가 Responses API `tool_choice=auto`로 "
        "`answer_directly`, `search_disclosures`, `ask_clarification` 중 하나를 고른다.",
        "FACT: 검색 구현은 `CatalogDenseSearch`이고 retriever_type은 dense다.",
        "FACT: 검색은 최대 2회, LLM 호출은 최대 4회, 요청 제한은 180초다.",
        "FACT: 회사 이름 해석은 결정적이지만 LLM 호출 한도에서 1회를 사용한다.",
        "FACT: 그 1회를 빼면 모델 호출은 3번이다. 검색 두 번 뒤에 planner가 다시 호출되면 답 생성 전에 한도가 끝난다.",
        "FACT: 답변 생성은 현재 실행의 evidence만 받는 `OpenAIAnswerGenerator`다.",
        "INTERPRETATION: 부분 답변은 별도 도구가 아니다. 검색 뒤 limitations가 있으면 평가에서 PARTIAL_ANSWER로 읽는다.",
        "",
        "## 3. Product Path vs Evaluation Path",
        "",
        "평가는 `AgentRuntime.run`을 호출한다. `/ask`와 같은 runtime이다.",
        "질문 문자열만 넘긴다. expected behavior, seed unit, requirement는 에이전트 입력에 없다.",
        "평가 시계는 2026-10-08 12:00 KST로 고정했다. 제품 코드의 기본 시계는 현재 시각이다.",
        "",
        "## 4. Reused Components",
        "",
        "- `build_default_runtime`, company resolution, tool schema, citation validator",
        "- Dense-D2a product indexes",
        "- v2.1 DEV 질문과 requirement 문장",
        "- G0 claim type 분류와 citation 필요 규칙",
        "- `gpt-6-luna` 가격 추정 2 USD / 8 USD per 1M tokens",
        "",
        "## 5. Added Components",
        "",
        "Run record에 planner 선택, 최종 답, 생성 evidence, LLM 호출 수를 남긴다.",
        "A0 dataset, 채점, grounding judge, 이 리포트가 새 코드다. 도구 선택 정책은 바꾸지 않았다.",
        "",
        "## 6. A0 Evaluation Dataset",
        "",
        f"버전 `{summary['dataset_version']}`, hash `{summary['dataset_sha256']}`.",
        f"구성: {summary['scenarios']}.",
        "재사용 질문은 v2.1 DEV만이다. TEST 55는 넣지 않았다.",
        "",
        "## 7. Tool Selection Results",
        "",
        (
            f"Search Decision Accuracy {decision['correct']}/{decision['denominator']}. "
            f"Missed Required Search {decision['missed_required_search']}. "
            f"Unnecessary Search {decision['unnecessary_search']}."
        ),
        "",
        "## 8. Tool Argument Results",
        "",
        _count_line(arguments)
        or "wrong_company, wrong_period, invalid_arguments, overly_restrictive_scope, missing_required_arguments, unauthorized_scope는 나오지 않았다.",
        "",
        "## 9. Clarification / Abstention Results",
        "",
        (
            f"Clarification Appropriateness {summary['clarification_appropriate']}/"
            f"{summary['clarification_denominator']}."
        ),
        (
            f"Abstention Appropriateness {summary['abstention_appropriate']}/"
            f"{summary['abstention_denominator']}."
        ),
        "2030년 확정 실적은 검색하지 않아도 된다. 2019년 매출은 검색 뒤 수치를 만들지 않아야 한다.",
        "requirement가 맞아도 faithfulness가 unsupported이면 자동 점수는 보류 성공으로 세지 않는다.",
        "",
        "## 10. Task Success",
        "",
        _count_line(task),
        "도구 선택이 맞다는 사실만으로 성공으로 두지 않았다.",
        "불필요 검색이 있어도 최종 과제가 맞으면 task와 efficiency flag를 나눴다.",
        "",
        "## 11. Grounding / Citation",
        "",
        (
            f"Faithfulness {grounding['faithfulness_supported']}/{grounding['faithfulness_denominator']}. "
            f"Citation Correctness {grounding['citation_correct']}/{grounding['citation_correct_denominator']}. "
            f"Citation Completeness {grounding['citation_complete']}/{grounding['citation_complete_denominator']}. "
            f"Citation Integrity Error {grounding['integrity_error_count']}."
        ),
        (
            f"Requirement Coverage {grounding['requirement_success']}/{grounding['requirement_denominator']}."
        ),
        "이 분모는 이번 실행의 evidence와 A0 답 계약이다. G0 58 requirement 점수와 같지 않다.",
        "공시 사실 claim에 citation이 없어도, 제품 계약상 보류 답이면 completeness 분모에 넣지 않았다.",
        "",
        "## 12. Failure Taxonomy",
        "",
        _count_line(failures["primary"]) or "primary cause 없음.",
        f"Secondary flags: {_count_line(failures['secondary']) or '없음.'}",
        "",
        "## 13. Representative Failures",
        "",
        _examples(examples),
        "",
        "## 14. Latency / Tokens / Cost",
        "",
        (
            f"End-to-end p50 {latency['p50']} ms, p95 {latency['p95']} ms. "
            f"Tool latency 합계 {summary['tool_latency_ms']} ms. "
            f"Generation latency 합계 {summary['generation_latency_ms']} ms."
        ),
        (
            f"LLM calls {summary['llm_calls']}, tool calls {summary['tool_calls']}. "
            f"Agent tokens in/out {cost['agent_input_tokens']}/{cost['agent_output_tokens']}. "
            f"Judge tokens in/out {cost['judge_input_tokens']}/{cost['judge_output_tokens']}."
        ),
        (
            f"Estimated agent USD {cost['agent_usd']}, judge USD {cost['judge_usd']}, "
            f"total USD {cost['total_usd']}. "
            f"성공 또는 부분 성공 1건당 agent USD {cost['usd_per_successful_task']}."
        ),
        "가격은 청구서가 아니라 `retrieval_evalset_v2_run.json`의 추정이다. G0 비용에는 Agent planner가 없다.",
        "",
        "## 15. Evaluation Integrity",
        "",
        "A0 실행 전에 도구 선택 prompt를 평가셋에 맞게 고치지 않았다.",
        "검색 정책, G0 prompt, generator 비교, TEST 55는 사용하지 않았다.",
        "Judge는 gpt-6-luna 한 모델이다. 두 번째 judge와 사람 gold는 없다.",
        "자동 평가가 문장 해석을 닫지 못한 항목은 unjudged로 남긴다.",
        "requirement judge와 faithfulness judge가 갈린 보류 답은 faithfulness를 따라 failure로 남겼다. 사람 gold는 만들지 않았다.",
        "",
        "## 16. Tests / Lint",
        "",
        summary.get("tests_note") or "테스트와 lint 결과는 실행 후 이 절에 기록한다.",
        "",
        "## 17. Artifacts",
        "",
        f"`{summary['output_dir']}`",
        "",
        "## 18. Commit",
        "",
        summary.get("commit_note") or "커밋은 리포트 작성 뒤에 만든다.",
        "",
        "## 19. Remaining Blockers",
        "",
        summary.get("blockers") or "실행을 막은 blocker는 없다.",
        "제품 경로가 H10을 쓰지 않는 점은 결함이 아니라 현재 구현이다. 이번 단계에서 검색을 다시 최적화하지 않았다.",
        "",
        "## 20. Recommended Next Hypothesis",
        "",
        hypothesis,
        "이 가설은 제안이다. 이 세션에서 Agent를 다시 고쳐 재실행하지 않는다.",
        "",
    ]
    return "\n".join(lines)


def _hypothesis(failures: dict[str, Any]) -> str:
    primary = failures.get("primary") or {}
    if not primary:
        return (
            "관측된 primary failure가 없다. "
            "검색 최적화와 G0 재개선 없이 UI와 배포로 넘어갈 수 있는지 이 리포트로 판단한다."
        )
    name = max(primary, key=primary.get)
    text = _HYPOTHESIS.get(name, "가장 많은 primary failure를 다음 실험의 단일 가설로 둔다.")
    return f"가장 많은 primary cause는 `{name}` {primary[name]}건이다. {text}"


def _examples(examples: list[dict[str, Any]]) -> str:
    if not examples:
        return "대표로 남길 primary failure가 없다."
    blocks = []
    for item in examples:
        blocks.append(
            "\n".join(
                [
                    f"### {item['query_id']}",
                    "",
                    f"질문: {item['question']}",
                    (
                        f"행동 {item['primary_behavior']}, task {item['task_outcome']}, "
                        f"primary `{item['primary_cause']}`, secondary {item['secondary_flags']}."
                    ),
                    f"답변: {item['answer_excerpt']}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _count_line(counts: dict[str, Any]) -> str:
    if not counts:
        return ""
    return ", ".join(f"{key} {value}" for key, value in counts.items())

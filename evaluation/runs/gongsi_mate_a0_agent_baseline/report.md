# GongsiMate A0 Agent Baseline

## 1. Executive Summary

기존 제품 Agent를 30개 DEV 질문에 그대로 실행했다. Task 결과는 성공 11, 부분 성공 0, 적절한 재질문 5, 적절한 보류 0, 실패 14, 미판정 0이다.
검색 필요 여부 판단은 28/30이다. 필요 검색 누락 2, 불필요 검색 0이다.
제품 검색은 H10이 아니라 Dense-D2a다. G0 점수와 이 점수는 비교하지 않는다.

## 2. Existing Agent Architecture

FACT: `OpenAISearchPlanner`가 Responses API `tool_choice=auto`로 `answer_directly`, `search_disclosures`, `ask_clarification` 중 하나를 고른다.
FACT: 검색 구현은 `CatalogDenseSearch`이고 retriever_type은 dense다.
FACT: 검색은 최대 2회, LLM 호출은 최대 4회, 요청 제한은 180초다.
FACT: 회사 이름 해석은 결정적이지만 LLM 호출 한도에서 1회를 사용한다.
FACT: 그 1회를 빼면 모델 호출은 3번이다. 검색 두 번 뒤에 planner가 다시 호출되면 답 생성 전에 한도가 끝난다.
FACT: 답변 생성은 현재 실행의 evidence만 받는 `OpenAIAnswerGenerator`다.
INTERPRETATION: 부분 답변은 별도 도구가 아니다. 검색 뒤 limitations가 있으면 평가에서 PARTIAL_ANSWER로 읽는다.

## 3. Product Path vs Evaluation Path

평가는 `AgentRuntime.run`을 호출한다. `/ask`와 같은 runtime이다.
질문 문자열만 넘긴다. expected behavior, seed unit, requirement는 에이전트 입력에 없다.
평가 시계는 2026-10-08 12:00 KST로 고정했다. 제품 코드의 기본 시계는 현재 시각이다.

## 4. Reused Components

- `build_default_runtime`, company resolution, tool schema, citation validator
- Dense-D2a product indexes
- v2.1 DEV 질문과 requirement 문장
- G0 claim type 분류와 citation 필요 규칙
- `gpt-6-luna` 가격 추정 2 USD / 8 USD per 1M tokens

## 5. Added Components

Run record에 planner 선택, 최종 답, 생성 evidence, LLM 호출 수를 남긴다.
A0 dataset, 채점, grounding judge, 이 리포트가 새 코드다. 도구 선택 정책은 바꾸지 않았다.

## 6. A0 Evaluation Dataset

버전 `gongsi_mate_a0_dev_v1`, hash `c76224ffbddf395a9d253053bc52cc261892f6e26eff46052b721ef9eadaa11b`.
구성: {'search_required': 10, 'insufficient_evidence': 5, 'additional_search': 5, 'no_search': 5, 'ambiguous': 5}.
재사용 질문은 v2.1 DEV만이다. TEST 55는 넣지 않았다.

## 7. Tool Selection Results

Search Decision Accuracy 28/30. Missed Required Search 2. Unnecessary Search 0.

## 8. Tool Argument Results

wrong_company, wrong_period, invalid_arguments, overly_restrictive_scope, missing_required_arguments, unauthorized_scope는 나오지 않았다.

## 9. Clarification / Abstention Results

Clarification Appropriateness 5/5.
Abstention Appropriateness 0/5.
2030년 확정 실적은 검색하지 않아도 된다. 2019년 매출은 검색 뒤 수치를 만들지 않아야 한다.
requirement가 맞아도 faithfulness가 unsupported이면 자동 점수는 보류 성공으로 세지 않는다.

## 10. Task Success

success 11, partial_success 0, appropriate_clarification 5, appropriate_abstention 0, failure 14, unjudged 0
도구 선택이 맞다는 사실만으로 성공으로 두지 않았다.
불필요 검색이 있어도 최종 과제가 맞으면 task와 efficiency flag를 나눴다.

## 11. Grounding / Citation

Faithfulness 9/15. Citation Correctness 10/10. Citation Completeness 6/6. Citation Integrity Error 0.
Requirement Coverage 12/27.
이 분모는 이번 실행의 evidence와 A0 답 계약이다. G0 58 requirement 점수와 같지 않다.
공시 사실 claim에 citation이 없어도, 제품 계약상 보류 답이면 completeness 분모에 넣지 않았다.

## 12. Failure Taxonomy

tool_selection_failure 2, budget_exhaustion 6, unsupported_claim 5, generation_failure 1
Secondary flags: missed_search 2, unsupported_claim 1, retrieval_failure 1, over_abstention 2, under_abstention 3

## 13. Representative Failures

### V2-00107613-business_paragraph-02

질문: 이수페타시스의 우주항공용 회로기판은 왜 극심한 온도 변화와 진동·충격을 견디고 오랫동안 믿고 쓸 수 있어야 하나요?
행동 DIRECT_ANSWER, task failure, primary `tool_selection_failure`, secondary ['missed_search', 'unsupported_claim'].
답변: 우주항공용 회로기판은 발사·비행·궤도 운용 과정에서 급격한 온도 변화와 진동·충격을 겪습니다. 이런 환경은 기판과 부품의 팽창·수축, 납땜부나 배선의 균열, 접촉 불량을 일으켜 전자장비 오작동이나 고장으로 이어질 수 있습니다. 특히 우주에서는 수리·교체가 어렵고, 통신·항법·제어처럼 임무에 중요한 장비의 고장이 안전과 임무 전체에 큰 영향을 줄 수 있습니다. 그래서 극한 환경에서도 성능을 유지하고 오랜 기간 안정적으로 작동하도

### V2-00107613-finance_table-01

질문: 이수페타시스의 2024년 상반기 연결 현금흐름표에서 투자활동현금흐름은 얼마였나요?
행동 SEARCH, task failure, primary `budget_exhaustion`, secondary [].
답변: 

### V2-00126955-finance_table-01

질문: 삼양식품의 2026년 상반기 연결 자본변동표에서 배당금 지급은 얼마로 나타났나요?
행동 ABSTAIN, task failure, primary `unsupported_claim`, secondary ['over_abstention'].
답변: 검색된 공시에서 요청한 내용을 확인하지 못했습니다.

## 14. Latency / Tokens / Cost

End-to-end p50 6685.831 ms, p95 12899.711 ms. Tool latency 합계 44314.282 ms. Generation latency 합계 32049.132 ms.
LLM calls 91, tool calls 30. Agent tokens in/out 147550/11072. Judge tokens in/out 32568/8029.
Estimated agent USD 0.3837, judge USD 0.1294, total USD 0.513. 성공 또는 부분 성공 1건당 agent USD 0.024.
가격은 청구서가 아니라 `retrieval_evalset_v2_run.json`의 추정이다. G0 비용에는 Agent planner가 없다.

## 15. Evaluation Integrity

A0 실행 전에 도구 선택 prompt를 평가셋에 맞게 고치지 않았다.
검색 정책, G0 prompt, generator 비교, TEST 55는 사용하지 않았다.
Judge는 gpt-6-luna 한 모델이다. 두 번째 judge와 사람 gold는 없다.
자동 평가가 문장 해석을 닫지 못한 항목은 unjudged로 남긴다.
requirement judge와 faithfulness judge가 갈린 보류 답은 faithfulness를 따라 failure로 남겼다. 사람 gold는 만들지 않았다.

## 16. Tests / Lint

ruff check src/disclosure_agent/agent_a0 tests/test_agent_a0.py src/disclosure_agent/agent/runtime.py passed. pytest tests/test_agent_a0.py tests/test_agent_product.py: 30 passed.

## 17. Artifacts

`evaluation/runs/gongsi_mate_a0_agent_baseline`

## 18. Commit

이 리포트와 실행 산출물을 같은 커밋에 넣는다. manifest의 git revision은 커밋 직전 HEAD다.

## 19. Remaining Blockers

실행을 막은 blocker는 없다.
제품 경로가 H10을 쓰지 않는 점은 결함이 아니라 현재 구현이다. 이번 단계에서 검색을 다시 최적화하지 않았다.

## 20. Recommended Next Hypothesis

가장 많은 primary cause는 `budget_exhaustion` 6건이다. 회사 해석이 LLM 한도 1회를 먼저 쓴다. 검색 두 번 뒤에 planner가 다시 호출되면 답 생성 전에 멈춘다. 다음 실험은 이 한도 회계만 모델 호출과 분리하는 가설을 본다.
이 가설은 제안이다. 이 세션에서 Agent를 다시 고쳐 재실행하지 않는다.

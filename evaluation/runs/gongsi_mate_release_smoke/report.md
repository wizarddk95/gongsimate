# GongsiMate release smoke

A1 `d795ba6` 이후 작업 트리의 제품 런타임을 `AgentRuntime.run`으로 확인했다.
검색 품질이나 문장 품질은 점수 내지 않았다.
A0 디렉터리 해시는 실행 전후가 같다.

## 1. 실행 결과

여덟 상황은 기존 A0 질문과 제품에서 쓰던 짧은 사업 질문이다. 새 문항은 만들지 않았다.
모델은 `gpt-6-luna`. 비용은 입력 2 USD, 출력 8 USD / 1M 토큰 추정이고 청구서가 아니다.
여덟 상황 합계는 USD 0.117904이다.

| id | 판정 | 종료 | 정책 | top_k | LLM | 검색 | ms | USD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_search | PASS | completed |  | - | 1 | 0 | 4260 | 0.003854 |
| clarify | PASS | company_ambiguous |  | - | 1 | 0 | 2567 | 0.003584 |
| cited_search | PASS | abstained | bm25_primary_dense_head | 10, 10 | 4 | 2 | 36881 | 0.028878 |
| additional_search | PASS | completed | bm25_primary_dense_head | 5, 5 | 4 | 2 | 14444 | 0.028746 |
| missing_period | PASS | no_matching_filing |  | 5, 5 | 3 | 2 | 6243 | 0.019130 |
| future_confirmed | PASS | completed |  | - | 1 | 0 | 1867 | 0.006028 |
| unsupported_company | PASS | company_ambiguous |  | - | 1 | 0 | 2863 | 0.004000 |
| primary_business | PASS | completed | bm25_primary_dense_head | 5 | 3 | 1 | 9960 | 0.023684 |

사용자 노출:

| id | 판정 | 사용자에게 보인 문장 |
| --- | --- | --- |
| h10_unavailable | KNOWN LIMITATION | 해당하는 공시 원문을 찾지 못했습니다. 확인되지 않은 내용은 답하지 않습니다. |
| evidence_required | PASS | 이 질문은 특정 기업의 공시에서 확인해야 합니다. 확인된 공시 근거가 없어 그 사실을 단정하지 않습니다. |
| answer_budget_exhausted | PASS | 검색 발췌와 인용. 한계 문장에 답변 모델 호출 한도가 남아 있지 않다고 적힌다. |
| request_timeout | PASS | 요청 처리 시간이 초과되었습니다. |

예외로 끝난 실행은 없다. timeout만 `error` 이벤트다.

## 2. 실제 Retrieval Policy와 top_k

제품 상수는 `PRODUCT_RETRIEVAL_POLICY = bm25_primary_dense_head`이다.
`build_default_runtime`의 검색 함수는 `H10Search`다.
히트가 난 실행의 `retrieval_policy`는 모두 `bm25_primary_dense_head`였다.
2019년 질문은 색인에 해당 보고서가 없어 검색기까지 가지 않았다.

Planner가 실제로 넘긴 `top_k`는 5 또는 10이다. 서버 상한은 20이다.
H10은 Dense와 BM25를 각각 20개 뽑는다. BM25 1–9위는 유지한다.
Dense 1위가 BM25 상위 10위 밖에 있을 때만 그 후보를 융합 10위에 넣는다.
이미 BM25 1–10위 안에 있으면 순서는 BM25 그대로이고 `retrieval_edit`는 `none`이다.
검색 응답은 그 목록의 앞 `top_k`개다.
따라서 융합 10위의 Dense 후보는 planner의 `top_k`가 10 이상일 때만 검색 응답에 남는다.
`top_k`가 5이면 그 자리는 응답 창 밖이다.
그 다음 답변 창은 같은 순서로 4096토큰 또는 20개까지 앞에서 채운다.
앞쪽 항목이 예산을 넘기면 10위는 모델 입력에도 들어가지 않는다.

이번 실행에서 히트가 난 세 질문의 `retrieval_edit`는 `none`이었다.
Dense 1위가 이미 BM25 상위 10위 안에 있었다.
`top_k` 10이었던 우주항공 질문은 융합 10위까지 검색 응답에 포함됐다.
`top_k` 5였던 환율 질문과 주요 사업 질문은 융합 10위가 검색 응답에서 잘렸다.
한도 소진 프로브는 `top_k` 10과 `insertion`이라 융합 10위가 검색 응답에 있었고, 중복 본문을 건너뛴 뒤 답변 창 3개 중 하나로 들어갔다.

## 3. 응답과 인용 무결성

인용 `evidence_id`가 그 실행의 evidence 밖에 있는 경우는 없다.
본문의 `[n]`이 없는 citation을 가리키는 경우도 없다.
보류와 검색 없는 답에는 인용이 없다.

- 서비스 설명과 2030년 확정 실적은 검색 없이 끝났다. 2030년 금액은 만들지 않았다.
- 삼성 실적은 회사를 되물었고 검색하지 않았다.
- 카카오 매출은 주식회사 카카오인지 되물었다. 금액은 없고 검색도 없다.
- 2019년 매출은 색인에 그 조건의 공시가 없다고 말하고 금액을 만들지 않았다.
- 환율 민감도와 주요 사업은 공시 문장과 인용이 연결된다.
- 우주항공 질문은 근거 없는 초안이 한 번 거절된 뒤 검색했고, 마지막 문장은 확인하지 못했다는 보류다. 초안은 사용자에게 나가지 않았다.

## 4. 남은 출시 차단 결함

출시 차단 결함은 없다. 제품 코드는 고치지 않았다.

Backlog:

- Planner `top_k` 5는 융합 10위 Dense 후보를 답변 창 밖에 둔다. 이번 단계에서 검색 인수나 프롬프트는 바꾸지 않는다.
- `h10_unavailable`은 로그에만 남고, 화면 문장은 빈 검색과 같다. 이번 색인에서 대표 질문은 이 경로에 들어가지 않았다.
- 답변 호출 한도가 없을 때의 본문은 검색 발췌 머리글이다. 한계 문장으로 한도 소진이 보인다.
- 우주항공 질문은 검색 뒤 보류했다. 문장 품질은 이번 범위 밖이다.

## 5. UI·UX 단계

진행할 수 있다.
대표 경로가 예외 없이 끝나고, 히트가 있으면 H10 정책을 쓰며, 인용은 그 실행의 evidence 안에 있다.
금액이 없어야 하는 질문은 금액을 만들지 않았다.
timeout은 오류 문장으로, 근거 없는 기업 사실은 보류 문장으로 사용자에게 보인다.

## 6. 테스트와 lint

`uv run ruff check scripts/release_smoke.py src/disclosure_agent/agent tests/test_agent_a1.py tests/test_agent_a0.py tests/test_agent_product.py tests/test_grounded_answer.py` passed.
`uv run pytest -q tests/test_agent_a1.py tests/test_agent_a0.py tests/test_agent_product.py tests/test_grounded_answer.py` 64 passed.

## 7. Commit

이 리포트와 `evaluation/runs/gongsi_mate_release_smoke`, `scripts/release_smoke.py`를 같은 커밋에 넣는다.
A0 산출물은 포함하지 않는다.

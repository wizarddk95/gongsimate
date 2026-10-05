# Retrieval eval set v2 — 2026-10-05 12:27 KST

## Goal

검색 알고리즘을 바꾸지 않는다. 사용자 질문에 가까운 모집단을 스펙으로 고정하고, 불완전 라벨이 Hit를 왜곡하지 않도록 재현 가능한 dataset construction pipeline을 만든 뒤, 그 위에 현재 Dense-D2a만 다시 측정한다.

`multi_company_retrieval_v1`과 그 Dense run은 수정하지 않았다. 제품 검색 코드도 바꾸지 않았다.

## Investigation

평가 스키마는 `product_retrieval_graded_v1`이다. v1은 relevance 0/1/2만 있고, 라벨되지 않은 검색 결과를 0으로 친다. 정답 단위는 element id다. preferred는 Hit와 별도다.

재사용한 것:

- 제품 경로 `CatalogDenseSearch` / `select_filings`
- Dense-D2a exact search와 `Bm25Index` (`ko-mixed-v1`)
- Responses API structured output (`openai` SDK, `responses.parse`)
- `.env`의 `OPENAI_API_KEY`, `OPENAI_MODEL`, `GEMINI_API_KEY`, `GEMINI_MODEL`, `ADJUDICATOR_MODEL`

새 retrieval 모델은 만들지 않았다. PyYAML은 의존성에 없어 population spec은 JSON이다.

색인 section path에는 위험 장이 없다. 위험은 quota로 두지 않았다. 다섯 회사 밖 산업·규모 quota도 두지 않았다.

## Population

`evaluation/specs/retrieval_population_v2.json`. 검색 결과를 보기 전에 고정했다.

회사당 fully_answerable 20개. 주제는 색인에 실제로 있는 business, overview, MD&A, finance, governance다. 형태는 paragraph, table, multi_element, cross_period다. Cartesian product는 만들지 않았다. 추가로 회사당 partial 2, unanswerable 2. 합계 120.

본 지표 분할은 stratum 안에서 `round(n * 0.6)`개를 test로 둔다. n=2는 test 1, dev 1이다. fully_answerable은 test 55, dev 45다. wording cycle이 original_term부터 시작하고 test가 각 칸의 앞 슬롯이라, test wording은 original_term 45, layperson 10이다. paraphrase는 dev에만 있다. 결과를 본 뒤 이 배정을 바꾸지 않았다.

## Pipeline

근거 단위를 먼저 고르고, 그 텍스트로 질문을 만든다. 생성은 근거 하나(또는 cross_period 두 개)당 한 호출이다. 복합 질문만 원자 요구로 분해한다. 단일 사실은 질문 자체가 R1이다.

풀은 Dense-D2a 상위 10, 기존 BM25 상위 10, Dense tail 21–50 / 51–200 / 201+에서 각 1개, 그리고 seed다. unanswerable은 상위 5+5이고 tail은 없다. seed는 검색 순위와 무관하게 판정 대상에 넣는다. Judge 입력에는 retriever, rank, score를 넣지 않는다.

Judge A `gpt-6-luna`, Judge B `gemini-3.8-flash`, 불일치만 `gpt-6.1-sol`. 호출 단위는 query × candidate다. 완료된 input hash는 다시 호출하지 않는다. 첫 Gemini 호출은 JSON 필드명을 지키지 않아 실패했고, `responseSchema`를 고정한 뒤 그 후보만 재시도했다.

relevance 2/1/0만 qrel에 쓴다. 실패한 판정은 행을 만들지 않는다. preferred는 judge 출력이 아니다. seed가 relevance 1 이상이면 그 요소가 preferred다.

## Cost

실행 전 상한 추정은 6,600 calls, 설정 단가 기준 약 $17.20였다. 단가는 청구서가 아니다.

측정된 성공 호출:

| stage | calls | input tokens | output tokens |
| --- | --- | --- | --- |
| generate | 123 | 91,477 | 31,455 |
| decompose | 45 | 31,946 | 8,148 |
| judge_a | 2,409 | 2,027,917 | 241,488 |
| judge_b | 2,409 | 1,796,821 | 132,228 |
| adjudicate | 187 | 154,685 | 20,947 |

judge_b 로그의 error 47건은 스키마 수정 전 실패다. 최종 unresolved는 0이다. 같은 단가로 다시 환산하면 약 $8.9다. 이것도 청구액이 아니다.

## Freeze

sha256 `d3635756a57c39d6d4ef8722f5f118348c00bbc034bf5963fb7720d184e6f137`.

판정 2,414쌍. 일치 2,227. agreement rate 0.923. adjudication 187. 이 일치율은 사람 정답의 증거가 아니다.

## Dense-D2a on the new set

제품 경로 그대로, 질문 원문, top 20. 본 지표는 test fully_answerable 55. 양성 qrel이 없어 제외된 문항은 0.

- Hit@1 26/55 (0.473)
- Hit@5 35/55 (0.636)
- Hit@10 40/55 (0.727)
- Recall@5 macro 0.384, micro 171/502 (0.341)
- Recall@10 macro 0.524, micro 255/502 (0.508)
- MRR 0.546
- NDCG@10 condensed 0.466
- preferred_hit@10 30/55 (0.545)
- judged_coverage@10 1.0, unjudged@10 mean 0
- judged_coverage@20 0.603, unjudged@20 mean 7.95
- requirement recall@10 46/65 (0.708)
- all-requirements-covered@10 37/55 (0.673)

회사별 test Hit@10, 각 n=11: 삼성전자 10/11, 삼양식품 9/11, 이수페타시스 7/11, 대상 7/11, 알테오젠 7/11. 알테오젠 MRR 0.292.

표 Hit@10 9/15, 문단 22/30. finance_table 5/10, governance_paragraph 2/5.

dev 45: Hit@10 28/45, MRR 0.416, NDCG@10 condensed 0.342.

unanswerable 10개는 평균 밖이다. direct relevance 2는 0건. 알테오젠 인과 질문 1건만 relevance 1이다.

v1 core Hit@10 28/31, MRR 0.543, NDCG@10 0.447과는 문항이 다르다. 개선분으로 빼지 않는다.

실행 당시 git은 `408422d`이고 dirty였다. 이 데이터셋이 아직 커밋 전이었기 때문이다.

## FACT / INTERPRETATION

FACT. test 55에서 Dense 상위 10 Hit는 40이다. 상위 10은 전부 판정됐다. 표와 finance_table, governance_paragraph가 문단보다 낮다. Recall 분모는 요소 복사이므로 Hit보다 낮다.

INTERPRETATION. v1의 0.90 Hit를 이 세트에서 재현하지 못했다. 원인 후보는 문항이 더 넓어진 것, 라벨이 단위 전체를 요소로 복사한 것, test가 원문 용어에 치우친 것이다. 이 셋 중 무엇을 검색 실패로 볼지는 dev에서 BM25와 비교하기 전에는 정하지 않는다.

## Limitations

generator bias, judge bias, 생성 모델과 Judge A의 계열 상관, 불완전 풀, 합성 질문 분포, 사람 gold 없음. test wording skew는 위 분할 규칙의 결과다.

## Next experiment

제품에 hybrid를 넣지 않는다. dev에서, 기존 BM25가 finance_table과 governance 누락을 Dense보다 위에 두는지부터 본다. test로 가중치를 맞추지 않는다.

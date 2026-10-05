# GongsiMate — Financial Disclosure Research Agent

국내 정기공시를 읽고, 근거 문단을 붙여 답하는 리서치 제품의 현재 상태다. 검색 기준선은 세 층이다. 2026-10-04의 `multi_company_retrieval_v1`은 사람이 본 31문항의 역사적 기준선이다. 2026-10-05의 `multi_company_retrieval_v2`는 모집단을 고정한 LLM-judged set이다. 같은 날의 `multi_company_retrieval_v2_1`은 그 set의 분할과 qrel 단위를 감사한 뒤 고정한 canonical benchmark다. 세 점수는 문항이나 채점 정의가 다르므로 한 계열로 이어 붙이지 않는다. 제품 검색에는 하이브리드를 넣지 않았다.

## 1. Problem

사업보고서와 반기·분기보고서는 길다. 같은 사실이 사업의 내용, 회사의 개요, 재무제표 주석에 반복되고, 정정보고서가 원본을 가린다. 표의 한 칸이 답인 경우도 있다.

개인이 묻는 질문은 공시 문장과 같지 않다. "뭘 하는 회사야", "차 쪽 사업도 하나", "매출이 왜 늘었나"처럼 짧다. 답을 만들려면 먼저 그 사실을 적은 요소를 찾아야 하고, 찾은 요소가 어느 접수번호의 어느 절인지 보여 줘야 한다.

## 2. Product

사용자는 채팅에서 회사와 질문을 넣는다. 회사는 검색 전에 정해진다. 그다음 경로는 다음과 같다.

회사 결정 → 검색 카탈로그 → 후보 공시 → 정정 반영한 유효 접수번호 → Dense-D2a → 근거 → 근거에 묶인 답 → 인용.

화면은 대화 목록과 원문 보기를 갖춘 공개 채팅이다. 이전 답은 다음 질문에 자동으로 이어지지 않는다. 이번 문서는 그 화면을 확장하지 않는다.

지금 검색되는 회사는 다섯이다. 이수페타시스, 삼성전자, 삼양식품, 알테오젠, 대상.

## 3. Why Financial Disclosure RAG Is Hard

- 파싱. 제목, 문단, 표가 한 문서 안에 섞인다. 표 셀을 평탄화하면 검색은 되지만, 화면의 표와는 다른 문자열이다.
- 기간. 같은 표가 사업·반기·분기에 다시 나오고 숫자가 바뀐다. 2025년 비중을 물었는데 2026년 반기 표가 위에 올 수 있다.
- 정정. 대상처럼 정정이 원본을 대체한다. 기준 시각이 정정 접수보다 앞이면 원본이 유효하다.
- 인용. 검색에 쓴 단위와 화면에 보여줄 요소가 다르다. 단위는 여러 요소를 묶을 수 있다.
- 검색. 같은 문장이 여러 접수번호에 있다. 주석의 회사 개요가 사업의 내용보다 위에 오르면, 답은 맞아도 선호 절은 아니다.

## 4. Architecture

Agent는 세 가지로 움직인다. 검색 없이 답하기, 공시 검색, 되묻기. 이번 baseline은 그 선택을 측정하지 않는다. 회사가 이미 정해지고 검색해야 하는 질문만 둔다.

검색 구현은 제품 카탈로그의 Dense-D2a다. 질의는 질문 원문이다. 지시문을 붙이지 않는다. 문서 벡터는 미리 계산해 둔다. 유사도는 L2 정규화 뒤 내적이고, 후보 접수번호 안을 전수 검색한다. `top_k`는 20이다. BM25, 하이브리드, RRF, reranker, 질의 재작성은 이 경로에 없다.

정정 정책은 `select_filings`다. 정정이 이미 후보에 있고 기준 시각도 만족하면 원본 접수번호는 빠진다.

## 5. Data Pipeline

OpenDART에서 정기공시 ZIP을 받아 버전을 남긴다. 파서는 제목·문단·표를 요소로 뽑는다. 제품 색인은 파싱이 성공한 사업·반기·분기 멤버로 retrieval unit을 만들고, BGE-M3로 임베딩한다. 표현 버전은 `retrieval-unit-d2a-v1`이다. 감사보고서와 부분 성공 정정본은 이 색인에 넣지 않는다.

색인과 임베딩은 `data/product/indexes` 아래라 저장소에 없다. 평가가 고정한 코퍼스는 다음이다.

| 회사 | corpus_id | corpus_as_of | units |
| --- | --- | --- | --- |
| 이수페타시스 | isu_petasys_periodic_d2a | 2026-09-25T10:26:24+09:00 | 20592 |
| 삼성전자 | generalization_00126380 | 2026-10-02T15:19:20+09:00 | 9171 |
| 삼양식품 | generalization_00126955 | 2026-10-02T15:19:22+09:00 | 8674 |
| 알테오젠 | generalization_00989619 | 2026-10-02T15:19:23+09:00 | 8986 |
| 대상 | generalization_00121941 | 2026-10-02T15:19:27+09:00 | 18261 |

모델은 `BAAI/bge-m3`, revision `5617a9f61b028005a4858fdac845db406aefb181`이다. 질의와 문서가 같은 모델이다.

## 6. Retrieval Evaluation

데이터셋 `multi_company_retrieval_v1`. sha256 `4b9795007d53820c2aca2c837bb884e010af8b2e60bd4b491fd77cfe43e65824`. 질문 32개. 본 측정은 core 31개다. MC32 한 개는 정정 시점 확인용이라 평균에 넣지 않았다.

질문은 다섯 회사를 모두 포함한다. 사업 설명, 실적 이유, 위험, 표의 숫자, 기간이 있는 질문, 공시 용어와 구어를 섞었다. 이수페타시스 v0.3의 제품 질문 두 개를 다시 썼다. 답할 수 없는 질문과 도구 선택 평가는 넣지 않았다.

정답은 청크 id가 아니라 색인된 요소 id다. 한 질문에 정답 요소가 여럿일 수 있다. relevance 2는 묻는 사실을 직접 적은 요소, 1은 나중 공시나 주석·회사의 개요에 같은 사실이 다시 적힌 요소다. preferred는 그중 답변에 쓰고 싶은 사업보고서 절이고, Hit와 Recall을 바꾸지 않는다. 원문에 없는 요소는 정답으로 두지 않았다. 같은 문단의 과거 사본을 전부 정답으로 넣지는 않았다.

| Metric | core n=31 |
| --- | --- |
| Hit@1 | 12/31 (0.387) |
| Hit@5 | 23/31 (0.742) |
| Hit@10 | 28/31 (0.903) |
| Recall@5 macro / micro | 0.538 / 29/65 (0.446) |
| Recall@10 macro / micro | 0.691 / 38/65 (0.585) |
| MRR | 0.543 |
| NDCG@10 | 0.447 |
| preferred_hit@10 | 21/31 (0.677) |

Hit@k는 relevance 1 이상이면 맞다. Recall은 라벨된 요소 중 상위 k에 들어온 비율이다. MRR은 첫 유효 근거의 역순위이고, 20위 안에 없으면 0이다. NDCG 이득은 relevance 2가 3, 1이 1이다. 회사별 문항은 5–7개라 회사 순위로 읽지 않는다.

회사별 Hit@10은 이수페타시스 5/7, 삼성전자 7/7, 삼양식품 6/6, 알테오젠 5/6, 대상 5/5다. 유형별 Hit@10은 사업 11/13, 사실 8/9, 실적 이유 4/4 (n=4), 위험 5/5다. 실적 이유 4개는 전부 맞았지만 수가 작다. 기간이 있는 질문 8개는 Hit@10과 preferred@10이 모두 8/8이다.

상위 10 누락은 세 질문이다. MC02 제품 용도 표, MC07 MLB 생산 법인, MC21 "플랫폼 기술".

## 6.1 LLM-judged set v2

데이터셋 `multi_company_retrieval_v2`. sha256 `d3635756a57c39d6d4ef8722f5f118348c00bbc034bf5963fb7720d184e6f137`. 라벨 출처는 LLM-judged evaluation set이다. 사람이 relevance를 검수하지 않았다.

질문 120개. fully_answerable 100, partially_answerable 10, unanswerable 10. 다섯 회사의 색인된 정기공시에서 근거를 먼저 고르고 질문을 만들었다. 회사 규모나 산업 quota는 없다. 이 다섯 회사가 한국 상장사 전체를 대표하지는 않는다. dev 45 / test 55는 fully_answerable 안에서의 분할이고, n=2 칸은 1:1이라 전체 dev 비율은 0.4보다 크다. 테스트 문항의 wording은 칸의 첫 슬롯이 test로 가면서 original_term 45, layperson 10이다. paraphrase는 dev에만 있다. 검색 결과를 본 뒤 이 비율을 다시 맞추지 않았다.

본 지표는 test fully_answerable 55개다. 양성 판정이 없어 빠진 문항은 0이다. unanswerable 10개는 Hit, Recall, MRR, NDCG에 넣지 않았다. unjudged는 0으로 저장하지 않는다. Hit는 원 순위 상위 k에 판정된 relevance 1 이상이 있으면 1이다. NDCG@10 본 지표는 unjudged를 뺀 condensed list다. 상위 10의 judged_coverage는 1.0이다. 11–20위의 coverage는 0.603이다.

| Metric | test n=55 |
| --- | --- |
| Hit@1 | 26/55 (0.473) |
| Hit@5 | 35/55 (0.636) |
| Hit@10 | 40/55 (0.727) |
| Recall@5 macro / micro | 0.384 / 171/502 (0.341) |
| Recall@10 macro / micro | 0.524 / 255/502 (0.508) |
| MRR | 0.546 |
| NDCG@10 condensed | 0.466 |
| preferred_hit@10 | 30/55 (0.545) |
| requirement recall@10 | 46/65 (0.708) |
| all-requirements-covered@10 | 37/55 (0.673) |

회사별 Hit@10은 각 11문항이다. 삼성전자 10/11, 삼양식품 9/11, 이수페타시스 7/11, 대상 7/11, 알테오젠 7/11. 알테오젠 MRR은 0.292다. 표 문항 Hit@10은 9/15, 문단은 22/30이다. finance_table은 5/10, governance_paragraph는 2/5다. n=5 칸은 small_n 기준(5 미만)에는 걸리지 않지만 순위를 만들 폭은 아니다.

Judge A는 gpt-6-luna, Judge B는 gemini-3.8-flash, 불일치는 gpt-6.1-sol이다. 비교 2,414건 중 일치 2,227건, agreement rate 0.923, adjudication 187, unresolved 0이다. 일치율이 높다는 것은 사람 정답의 증거가 아니다. unanswerable 10개 중 direct evidence로 판정된 것은 0이다. 알테오젠 인과 질문 1개는 supporting relevance 1만 받았다.

Recall 분모 502는 요소 수다. 판정은 retrieval unit 단위로 하고, 그 단위의 source element에 같은 등급을 복사했다. Hit는 단위가 relevant element를 포함하면 성공이다. 그래서 Hit@10과 Recall@10은 같은 사실을 다른 단위로 센다.

dev fully_answerable 45개의 Hit@10은 28/45, MRR 0.416, NDCG@10 condensed 0.342다. 이 숫자는 v2 정의의 기록이다. 이후 비교의 기준은 아래 v2.1이다.

## 6.2 Benchmark audit, v2.1

v2를 검색 개선의 출발점으로 쓰기 전에 평가 정의부터 확인했다. 검색기, 질문 문장, judge 호출은 그대로 뒀다. v2 artifact도 수정하지 않았다.

분할은 회사·stratum 안에서 wording cycle의 앞 슬롯이 test로 가도록 되어 있었다. fully_answerable test 55개의 wording은 original_term 45, layperson 10, paraphrase 0이었다. 같은 구조로 query type도 한쪽으로 몰렸다. test에는 composition_ratio, calculation, cross_filing_comparison이 없고, dev에는 causal_explanation, table_lookup, single_numeric, time_series_comparison이 없었다. 검색 점수를 보고 옮긴 결과는 아니다.

판정 입력은 retrieval unit 텍스트 하나였다. `judge_payload`의 evidence는 그 unit의 본문, 문서명, 기간, 섹션이다. 요소 id는 judge에게 나가지 않는다. qrel을 만들 때는 그 등급을 unit 안의 source element마다 복사했다. v2 test의 Recall 분모 502는 그렇게 복사된 요소 수다. 같은 문항을 unit으로 세면 양성 unit은 271개다.

NDCG@10 condensed 0.466과 unjudged-as-zero 0.517은 상위 10의 judged coverage가 1.0인데도 달랐다. coverage는 요소가 qrel에 있는지로 계산됐다. 다른 unit과 요소를 공유하면, 그 unit 자신의 등급이 0이어도 요소 최댓값으로 relevance가 올라갔다. condensed 경로는 이미 본 요소 집합이면 gain을 0으로 지웠고, zero-filled 경로는 같은 gain을 다시 더했다. 그 결과 zero-filled NDCG가 1을 넘는 문항이 있었다. 이것은 검색 품질의 차이가 아니라 채점 단위가 어긋난 결과다.

v2.1은 fully_answerable 100개를 seed 20261005로 다시 나눈다. 회사마다 test 11, dev 9다. wording과 query type은 55/100 Hamilton 목표와 같다. test wording은 original_term 25, layperson 25, paraphrase 5다. query type 아홉 가지는 dev와 test 양쪽에 있다. 표 문항의 독립 Hamilton 목표는 test 14개인데, query type이 문단과 표를 가로지르지 않아 표는 test 13개다. finance_table은 test 7로, 주변 stratum 목표 8에서 1개 다르다. 입력에는 검색 점수와 Hit를 넣지 않았다. 질문 문장 변경은 0이다. 기존 judgment 2,414건을 재사용했고 새 judge 호출은 0이다.

qrel 키는 `(query_id, requirement_id, retrieval_unit_id)`다. 행은 3,112개다. source element는 provenance다. relevance 0은 읽고 무관하다고 본 것이고, 아직 안 본 unit은 qrel에 없다. Recall 분모는 양성 retrieval unit이다. requirement recall의 분모는 requirement 수다. preferred는 relevance와 별개이고, seed unit이 여전히 양성이면 그 unit이 preferred다. NDCG gain은 `2^relevance - 1`이고 unit당 한 번이다. 상위 k가 모두 판정됐으면 condensed와 unjudged-as-zero는 같다.

데이터셋 sha256 `2b1c372b7adaca29af831986bf5e6807c2ae2bd7835a52735690e227c1b03137`. 부모는 v2 `d3635756a57c39d6d4ef8722f5f118348c00bbc034bf5963fb7720d184e6f137`이다. 분할이 바뀐 문항은 50개다.

같은 Dense-D2a를 이 정의로 다시 측정했다. 본 지표는 v2.1 test fully_answerable 55개다. 양성 unit이 없어 빠진 문항은 0이다. unanswerable 10개는 평균 밖에 있다.

| Metric | v2.1 test n=55 |
| --- | --- |
| Hit@1 | 19/55 (0.345) |
| Hit@5 | 27/55 (0.491) |
| Hit@10 | 32/55 (0.582) |
| Recall@5 macro / micro | 0.246 / 69/282 (0.245) |
| Recall@10 macro / micro | 0.392 / 116/282 (0.411) |
| MRR | 0.411 |
| NDCG@10 condensed | 0.325 |
| NDCG@10 unjudged-as-zero | 0.325 |
| preferred_hit@1 | 4/55 (0.073) |
| preferred_hit@10 | 21/55 (0.382) |
| judged_coverage@10 / @20 | 1.000 / 0.536 |
| requirement recall@10 | 37/66 (0.561) |
| all-requirements-covered@10 | 29/55 (0.527) |

상위 10은 모두 그 unit 자신이 판정된 상태라 두 NDCG가 같다. 1을 넘는 NDCG는 없다. 11–20위의 coverage는 0.536이고, 문항당 unjudged는 평균 9.27개다.

test wording별 Hit@10은 original_term 15/25, layperson 14/25, paraphrase 3/5다. paraphrase가 test에 있다. 회사별 Hit@10은 각 11문항이라 회사 순위로 읽지 않는다. n이 5 미만인 query type은 calculation, cross_filing_comparison, single_numeric, time_series_comparison이다.

v2 test 55문항을 unit 정의로만 다시 세면 Hit@10은 37/55, Recall@10 micro는 130/271, NDCG@10은 0.434다. v2에 공개된 40/55, 255/502, 0.466은 그 문항 집합에 요소 복사를 적용한 값이다. v2.1 본 지표 32/55는 여기에 분할 변경이 더해진 값이다. 검색 코드는 같다. 이 차이를 성능 향상이나 하락으로 읽지 않는다.

dev fully_answerable 45개의 Hit@10은 33/45, MRR 0.525, NDCG@10 0.478이다. 다음 Dense와 BM25 비교는 이 dev에서 한다. test 점수를 보며 가중치를 맞추지 않는다.

## 7. Failure Analysis

아래 세 누락은 v1 core 31개에 대한 기록이다. v2 점수와 섞지 않는다.

세 누락의 공통점은 가까운 다른 문장이 위를 차지한다는 점이다.

MC02는 구체적용도 표를 찾지 못했다. 상위는 여러 보고서의 "PCB를 전문으로 생산" 개요 문장이다. 제품 종류는 보이고 용도 열은 안 보인다.

MC07은 "어느 법인에서 만드나"인데, 상위는 초고다층 MLB가 네트워크 장비에 쓰인다는 용도 문장이다. MLB라는 단어가 다른 사실을 끌어올렸다.

MC21은 Hybrozyme 문단이 상위 10 밖에 있다. 상위는 경영진단 서두와 "유전자 개발과 형질전환" 주석이다. 같은 사실을 "정맥을 피하로"라고 묻거나 Hybrozyme이라고 물으면 2–4위에 있다.

맞는 것처럼 보여도 선호 절이 아닌 경우가 있다. MC01은 주석의 "PCB 제조 및 판매"가 1위이고, 사업의 내용 개요는 상위 20에 없다. MC29는 반기 제품 설명이 1위로 정답이고, 선호하는 2025 사업보고서 개요는 상위 20에 없다. 이것은 검색 실패가 아니다.

MC14는 2025년 DX 56.3%, DS 39.0% 표가 8위다. 그 위에 다른 기간의 39.3%, 33.0% 표가 있다. MC27은 1–3위가 대상홀딩스 주주 설명이고, 식품·소재라는 답은 6위의 2024 경영진단에 있다.

같은 주석이 분기·반기·과거 사업보고서에 반복되며 목록을 채우는 경우도 있다. MC01과 MC05가 그렇다. 라벨된 요소의 중복 횟수는 거의 0이다. 중복은 라벨하지 않은 사본에서 난다.

## 8. Retrieval Improvement

canonical benchmark는 v2.1이다. 이번 감사에서 제품 검색에 BM25, hybrid, RRF, reranker를 넣지 않았다. Dense-D2a도 바꾸지 않았다.

다음 실험은 v2.1 dev에서 현재 Dense-D2a와 이미 있는 BM25를 같은 질문, 같은 unit qrel로 비교하는 것이다. test는 그 비교의 설정을 고른 뒤에 한 번 확인하는 자리다.

v1에서 적었던 표 헤더, 주석 반복, reranker 후보는 그 31문항에 묶인 가설로 남긴다. v2 숫자와 이어 붙여 개선분으로 말하지 않는다.

## 9. Agent Architecture

직접 답, 공시 검색, 되묻기를 구분한다. 회사가 없거나 질문이 공시로 답할 수 없으면 검색으로 넘기지 않는 것이 목표다. 그 선택은 별도 평가다. v1 검색 세트에는 답 없는 질문을 넣지 않았다. v2의 unanswerable 10개는 검색 Hit 평균 밖에 두고, 근거가 충분한지 따로 센다.

검색이 끝나면 근거 요소에 묶인 답을 만든다. 검색 점수가 높다고 그 문단만으로 답이 완성되는 것은 아니다.

## 10. Citation / Source UX

검색 단위의 텍스트와 화면에 그리는 원문은 분리한다. 표는 검색용으로 평탄화한 문자열과, 셀을 가진 원문 표시가 다르다. 인용은 접수번호와 요소로 돌아간다. 이 분리는 이번 측정에서 바꾸지 않았다.

## 11. Product / Demo

공개 채팅에서 다섯 회사를 고르고 질문할 수 있다. 대화는 그 브라우저에 남고, 답에 쓰인 공시를 원문 보기에서 연다. 파싱된 표는 한 줄 문자열이 아니라 표로 본다.

## 12. Trade-offs

- 정답을 사업의 내용으로만 두면 Hit가 낮아진다. 주석의 같은 문장도 답은 된다. relevance 1로 나누고 preferred는 진단으로 뒀다.
- 과거 공시의 동일 문단을 모두 정답으로 두면, 그 사본을 못 찾은 것이 Recall 감점이 된다. 그렇게 하지 않았다. 대신 사본이 상위 10을 채우는 현상은 실패 기록에 남긴다.
- 단일 접수번호 평가(이수페타시스 v0.3)는 그대로 둔다. 제품 경로는 여러 공시를 동시에 본다. 두 점수를 같은 숫자로 비교하지 않는다.
- 문항 31개다. 유형이 갈라지면 칸이 4–13개다. 평균은 이 세트의 기준선이지, 다섯 회사 공시 전체의 성능이 아니다.

## 13. Remaining Limitations

상위 10 Hit가 0.90이어도 NDCG@10은 0.45다. 첫 칸에 선호 근거가 오는 비율은 6/31이다.

표의 특정 열, 생산 법인, 일반어로 물은 기술명은 아직 놓친다. 회사 이름 충돌도 한 건 확인됐다.

정정 시점 문항은 하나다. 그 원본이 8위에 있었다는 관찰만 있고, 정정 정책의 성공률로 쓰지 않는다.

하이브리드와 reranker는 제품 경로에 없다. v2.1은 그 비교를 아직 하지 않았다.

v2와 v2.1의 라벨은 LLM judge 두 명과 불일치 때의 세 번째 모델이다. 생성 모델과 Judge A가 같은 계열이다. 풀은 Dense-D2a와 기존 BM25의 상위와 일부 tail이다. 상위 20의 약 절반은 아직 판정되지 않았다. 질문 분포는 실제 사용자 로그가 아니라 색인 근거에서 만든 합성 질문이다. 사람 gold adjudication은 없다. judge agreement를 사람 정확도로 읽지 않는다. 요소를 공유하는 양성 unit은 합치지 않는다. v2.1 test에서 그런 unit은 34개다.

# v2.1 DEV Dense-D2a + BM25 RRF

## 1. Executive summary

판정은 C이다. RRF Hit@10은 36/45이고 BM25는 42/45이다. BM25-only 10건 중 2건을 유지했다. Dense-only 복구는 1/1이다. MRR 0.742 → 0.686, NDCG@10 0.708 → 0.654, Recall@5 macro 0.572 → 0.468.

## 2. Previous experiment commit

이 실행의 HEAD는 `3ab8dfe478463d6a4201e6f2189db9d34fe98491` (eval: compare dense and bm25 complementarity on v2.1 dev)이다. 작업 트리의 RRF 코드는 아직 그 커밋에 들어 있지 않다. 입력 순위는 complementarity 산출물의 top-20과 같아야 한다.

## 3. Experiment invariants

dataset `multi_company_retrieval_v2_1` sha256 `2b1c372b7adaca29af831986bf5e6807c2ae2bd7835a52735690e227c1b03137`. 모집단은 DEV fully_answerable 45문항이다. TEST 검색 수는 0이다. k sweep, weighted fusion, alpha, query routing, reranker, query expansion, embedding 교체, chunking 변경은 없다.

## 4. RRF implementation audit

`disclosure_agent.hybrid.reciprocal_rank_fusion` k=60. 식은 `score(d) = 1/(k + rank_bm25(d)) + 1/(k + rank_dense(d))`이다. 한쪽에만 있는 문서에 contribution 0을 주지 않는다. 후보 집합이 다르면 UsageError다. 이번 실험은 filing scope 안의 eligible D2a unit 전체를 양쪽이 순위화한 뒤 합친다. BM25 점수가 0인 문서도 `1/(k+rank)`를 받는다. `reciprocal_rank_fusion_zero_tie`는 쓰지 않았다. 공식 대조 519354건, rank가 하나뿐인 문서 0건, BM25 0점 문서 307603건, 0점 spot check에서 contribution이 0보다 큰 횟수 45.

filing scope 밖 unit은 후보가 아니다. 0점으로 합쳐지지 않는다.

## 5. Dense / BM25 / RRF configuration

Dense는 제품 경로 `CatalogDenseSearch`와 `exact_normalized_search`다. 모델은 `BAAI/bge-m3`이고 질의 원문을 지시문 없이 인코딩한다. BM25는 `Bm25Index` k1=1.2 b=0.75 tokenizer `ko-mixed-v1`이다. headline BM25는 양수 점수 top-20이다. RRF 입력 BM25는 같은 순서의 `rank_all`이다. 재계산한 top-20 prefix 45/45가 저장된 Dense, BM25와 같았다.

## 6. Headline comparison

| metric | Dense | BM25 | RRF |
| --- | --- | --- | --- |
| Hit@1 | 20/45 | 29/45 | 28/45 |
| Hit@5 | 28/45 | 40/45 | 33/45 |
| Hit@10 | 33/45 | 42/45 | 36/45 |
| Recall@5 macro | 0.387 | 0.572 | 0.468 |
| Recall@5 micro | 78/231 | 113/231 | 95/231 |
| Recall@10 macro | 0.561 | 0.767 | 0.629 |
| Recall@10 micro | 129/231 | 174/231 | 141/231 |
| MRR | 0.525 | 0.742 | 0.686 |
| NDCG@10 | 0.478 | 0.708 | 0.654 |
| preferred_hit@10 | 25/45 | 37/45 | 31/45 |
| requirement_recall@10 | 41/58 | 51/58 | 46/58 |
| all_requirements_covered@10 | 32/45 | 41/45 | 35/45 |
| judged_coverage@10 | 1.000 | 1.000 | 0.787 |
| judged_coverage@20 | 0.536 | 0.557 | 0.597 |

RRF judged_coverage@10은 0.787이고 상위 10의 unjudged 평균은 2.133이다. condensed NDCG@10은 0.654, unjudged-as-zero NDCG@10은 0.615이다. 민감도 지표도 BM25 condensed NDCG보다 낮다.

## 7. Oracle union ceiling

이전 top-10 합집합 oracle Hit@10은 43/45이다. BM25는 42/45, RRF는 36/45이다. BM25 대비 RRF 차이는 -6이고, oracle 대비 차이는 -7이다. top-10 합집합 기준으로 BM25 위에 남은 이론적 여지는 1문항이다. 전체 순위 RRF는 양쪽 top-10 밖에 있던 양성 unit을 끌어올릴 수 있으므로 이 oracle은 수학적 상한이 아니다.

## 8. Both-hit retention

이전 both-hit 32 중 RRF Hit@10은 32이다. 잃은 질의는 0건.

## 9. BM25-only retention

이전 BM25-only 10 중 RRF가 유지한 Hit@10은 2이다. 잃은 질의는 8건: V2-00121941-business_paragraph-02, V2-00121941-business_paragraph-03, V2-00121941-finance_table-02, V2-00121941-overview_paragraph-01, V2-00126380-governance_paragraph-01, V2-00126955-finance_paragraph-02, V2-00989619-governance_paragraph-02, V2-00989619-overview_paragraph-01.

- `V2-00121941-business_paragraph-02` 대상: Dense None, BM25 1, RRF 17, regression. 그 unit의 전체 순위는 BM25 1, Dense 303. 대상의 2026년 1분기 식품·소재 부문 매출을 더한 금액보다 연결 매출액이 낮은 이유는 무엇인가요?
- `V2-00121941-business_paragraph-03` 대상: Dense None, BM25 8, RRF 19, regression. 그 unit의 전체 순위는 BM25 8, Dense 774. 대상이 종속·관계·공동기업을 제외하고 보유한 지분증권 투자액은 2024년 말 장부상 얼마였나요?
- `V2-00121941-finance_table-02` 대상: Dense None, BM25 7, RRF 24, regression. 그 unit의 전체 순위는 BM25 7, Dense 75. 대상은 2026년 1분기 말 특수관계자 거래로 받을 돈과 갚을 돈이 각각 얼마였어?
- `V2-00121941-overview_paragraph-01` 대상: Dense None, BM25 1, RRF 23, regression. 그 unit의 전체 순위는 BM25 1, Dense 283. 대상의 2026년 6월 30일 현재 총 유통주식수는 몇 주인가요?
- `V2-00126380-governance_paragraph-01` 삼성전자: Dense None, BM25 1, RRF 27, regression. 그 unit의 전체 순위는 BM25 1, Dense 705. 삼성전자는 당기에 주식매수선택권 부여로 인식한 보상비용이 있나요?
- `V2-00126955-cross_period-02` 삼양식품: Dense None, BM25 4, RRF 10, regression. 그 unit의 전체 순위는 BM25 4, Dense 230. 삼양식품의 2026년 1분기 말 단기·유동성장기 차입금 잔액과 반기보고서에 나온 전기말 대출 약정 한도는 각각 얼마인가요?
- `V2-00126955-finance_paragraph-02` 삼양식품: Dense None, BM25 1, RRF 17, regression. 그 unit의 전체 순위는 BM25 1, Dense 301. 삼양식품의 현행 기준 영업이익은 매출액에서 어떤 비용을 빼서 계산하나요?
- `V2-00126955-finance_table-01` 삼양식품: Dense None, BM25 4, RRF 2, improvement. 그 unit의 전체 순위는 BM25 4, Dense 61. 삼양식품의 2026년 상반기 연결 자본변동표에서 배당금 지급은 얼마로 나타났나요?
- `V2-00989619-governance_paragraph-02` 알테오젠: Dense None, BM25 4, RRF 18, regression. 그 unit의 전체 순위는 BM25 4, Dense 125. 알테오젠은 이번 분기에 우선주 몇 주를 보통주로 바꿨나요?
- `V2-00989619-overview_paragraph-01` 알테오젠: Dense None, BM25 1, RRF 16, regression. 그 unit의 전체 순위는 BM25 1, Dense 413. 알테오젠의 종류주식 발행현황에 기재된 주식은 어떤 종류인가요?

## 10. Dense-only recovery

- `V2-00126380-finance_table-02` 삼성전자: Dense 1, BM25 None, RRF 6, recovered. 그 unit의 전체 순위는 BM25 75, Dense 1. 삼성전자는 전분기에 삼성에스디에스와 구매 관련 거래를 얼마만큼 했나요?

## 11. Both-miss result

이 두 질의는 period, 연결/별도 scope, cross-filing failure family로 둔다. RRF가 Hit@10에 넣어도 그 family가 해결된 것으로 보지 않고, 못 넣어도 chunking 실패로 단정하지 않는다.

- `V2-00107613-cross_period-02` 이수페타시스: Dense None, BM25 None, RRF 135, still_missing. 그 unit의 전체 순위는 BM25 544, Dense 80. 이수페타시스의 2024년 반기보고서와 3분기보고서에는 투자조합의 지분 정보와 받을 돈의 연체 현황이 각각 어떻게 나와 있나요?
- `V2-00107613-finance_table-03` 이수페타시스: Dense 20, BM25 None, RRF 10, recovered. 그 unit의 전체 순위는 BM25 82, Dense 20. 이수페타시스의 2026년 상반기 투자활동에서 무형자산 취득과 투자부동산 취득은 각각 어느 정도를 차지했나요?

## 12. Rank movement

movement 분포: `{"improvement": 7, "neutral": 24, "recovered": 2, "regression": 11, "still_missing": 1}`. BM25 대비 first-relevant delta 평균 -3.214, 중앙값 0.000. 양수 delta는 RRF가 첫 양성 unit을 더 앞에 둔 것이다. 같은 unit이 뒤로 밀린 질의는 18건이다.

## 13. Requirement-level analysis

requirement recall@10 Dense 41/58, BM25 51/58, RRF 46/58. all requirements covered@10 Dense 32/45, BM25 41/45, RRF 35/45.

- both prior 40: `{"retained": 40}`
- dense_only prior 1: `{"recovered": 1}`
- bm25_only prior 11: `{"lost": 8, "retained": 3}`
- neither prior 6: `{"recovered": 2, "still_missing": 4}`

- `V2-00126380-finance_table-02` R1 dense_only → recovered. 삼성전자는 전분기에 삼성에스디에스와 구매 관련 거래를 얼마만큼 했나요?
- `V2-00126955-cross_period-01` R1 both → retained. 제62기부터 제64기까지 매출액은 어떻게 변했는가?
- `V2-00126955-cross_period-01` R2 bm25_only → retained. 제62기부터 제64기까지 영업이익은 어떻게 변했는가?
- `V2-00126955-cross_period-02` R1 neither → still_missing. 2026년 1분기 말 단기차입금 잔액은 얼마인가?
- `V2-00126955-cross_period-02` R2 bm25_only → retained. 2026년 1분기 말 유동성장기부채 잔액은 얼마인가?
- `V2-00126955-cross_period-02` R3 neither → still_missing. 반기보고서에 기재된 전기말 대출 약정 한도는 약정 유형별로 얼마인가?

## 14. Representative improvements

### V2-00989619-finance_table-02

알테오젠 분기보고서에서 기술 지원과 위탁생산 용역 기간은 언제부터 언제까지로 나와 있어?

회사 알테오젠. 이전 셀 dense_hit_bm25_hit. movement improvement. Dense 첫 양성 8, BM25 첫 양성 5, RRF 첫 양성 2. delta vs BM25 3, delta vs Dense 6. 같은 BM25 unit의 RRF 순위 2 (delta 3). RRF 첫 양성 unit의 Dense 순위 8, BM25 순위 5. 상위 5 양성 1, 상위 10 양성 1.

RRF 1위 relevance 0: [section] 목 차 > III. 재무에 관한 사항 > 5. 재무제표 주석 > 38. 주요기술이전 계약 및 수익 인식 내역 주요기술이전계약에 대한 세부 공시 당기 (단위 : 원) | 계약의 유형 | ALT-B4 | 고객 | A사 | B사 | C사 | D사 | E사 | F사 | G사 | 재화나 용역의 이전 시기 | 한…

첫 양성 발췌: [section] 목 차 > III. 재무에 관한 사항 > 5. 재무제표 주석 > 38. 주요기술이전 계약 및 수익 인식 내역 주요기술이전계약에 대한 세부 공시 당기 (단위 : 원) | 계약의 유형 | ALT-B4 | 고객 | A사 | B사 | C사 | D사 | E사 | F사 | G사 | 재화나 용역의 이전 시기 | 한…

### V2-00107613-finance_table-01

이수페타시스의 2024년 상반기 연결 현금흐름표에서 투자활동현금흐름은 얼마였나요?

회사 이수페타시스. 이전 셀 dense_hit_bm25_hit. movement improvement. Dense 첫 양성 9, BM25 첫 양성 3, RRF 첫 양성 1. delta vs BM25 2, delta vs Dense 8. 같은 BM25 unit의 RRF 순위 1 (delta 2). RRF 첫 양성 unit의 Dense 순위 9, BM25 순위 3. 상위 5 양성 1, 상위 10 양성 1.

RRF 1위 relevance 2: [section] 목 차 > III. 재무에 관한 사항 > 2. 연결재무제표 > 2-4. 연결 현금흐름표 연결 현금흐름표 제 53 기 반기 2024.01.01 부터 2024.06.30 까지 제 52 기 반기 2023.01.01 부터 2023.06.30 까지 (단위 : 원) | 제 53 기 반기 | 제 52 기 반기 법인…

첫 양성 발췌: [section] 목 차 > III. 재무에 관한 사항 > 2. 연결재무제표 > 2-4. 연결 현금흐름표 연결 현금흐름표 제 53 기 반기 2024.01.01 부터 2024.06.30 까지 제 52 기 반기 2023.01.01 부터 2023.06.30 까지 (단위 : 원) | 제 53 기 반기 | 제 52 기 반기 법인…

### V2-00126955-finance_table-01

삼양식품의 2026년 상반기 연결 자본변동표에서 배당금 지급은 얼마로 나타났나요?

회사 삼양식품. 이전 셀 dense_miss_bm25_hit. movement improvement. Dense 첫 양성 None, BM25 첫 양성 4, RRF 첫 양성 2. delta vs BM25 2, delta vs Dense None. 같은 BM25 unit의 RRF 순위 2 (delta 2). RRF 첫 양성 unit의 Dense 순위 61, BM25 순위 4. 상위 5 양성 1, 상위 10 양성 1.

RRF 1위 relevance 0: [section] 목 차 > III. 재무에 관한 사항 > 4. 재무제표 > 4-3. 자본변동표 자본변동표 제 66 기 반기 2026.01.01 부터 2026.06.30 까지 제 65 기 반기 2025.01.01 부터 2025.06.30 까지 (단위 : 원) | 자본의 구성요소 자본금 | 기타자본구성요소 | 기타포괄손익…

첫 양성 발췌: [section] 목 차 > III. 재무에 관한 사항 > 2. 연결재무제표 > 2-3. 연결 자본변동표 연결 자본변동표 제 66 기 반기 2026.01.01 부터 2026.06.30 까지 제 65 기 반기 2025.01.01 부터 2025.06.30 까지 (단위 : 원) | 자본의 구성요소 지배기업의 소유주에게 귀속되…

## 15. Representative regressions

### V2-00126380-governance_paragraph-01

삼성전자는 당기에 주식매수선택권 부여로 인식한 보상비용이 있나요?

회사 삼성전자. 이전 셀 dense_miss_bm25_hit. movement regression. Dense 첫 양성 None, BM25 첫 양성 1, RRF 첫 양성 27. delta vs BM25 -26, delta vs Dense None. 같은 BM25 unit의 RRF 순위 27 (delta -26). RRF 첫 양성 unit의 Dense 순위 705, BM25 순위 1. 상위 5 양성 0, 상위 10 양성 0.

RRF 1위 relevance 0: [section] 목 차 > III. 재무에 관한 사항 > 5. 재무제표 주석 > 20. 기타자본항목 성과연동 주식보상에 대한 공시 (단위 : 백만원) | 주식기준보상약정 | 성과연동 주식보상 | 삼성전자주식회사 주식기준보상약정에 대한 기술 | 회사는 임직원을 대상으로 성과연동주식보상제도(PSU)를 도입하였습니다. 가중…

첫 양성 발췌: [section] 목 차 > VIII. 임원 및 직원 등에 관한 사항 > 2. 임원의 보수 등 ※「자본시장과 금융투자업에 관한 법률」제159조, 동법 시행령 제168조에 따라 당해 사업연도에 재임한 등기이사ㆍ독립이사ㆍ감사위원회 위원에게 지급된 금액 기준입니다.※ 인원수는 작성기준일 현재 기준입니다.※ 보수총액은 당해 사…

### V2-00121941-overview_paragraph-01

대상의 2026년 6월 30일 현재 총 유통주식수는 몇 주인가요?

회사 대상. 이전 셀 dense_miss_bm25_hit. movement regression. Dense 첫 양성 None, BM25 첫 양성 1, RRF 첫 양성 23. delta vs BM25 -22, delta vs Dense None. 같은 BM25 unit의 RRF 순위 23 (delta -22). RRF 첫 양성 unit의 Dense 순위 283, BM25 순위 1. 상위 5 양성 0, 상위 10 양성 0.

RRF 1위 relevance 0: [section] VII. 주주에 관한 사항 ※ 2005년 11월 14일 최대주주는 임상민에서 대상홀딩스로 변경되었으며 상기소유주식수 및 지분율은 보통주 기준으로 특수관계인인 임창욱, 임세령, 대상문화재단 등을 포함한 주식수입니다. 2009년 3월 20일 제55기 정기주주총회 결과에 따른 특수관계인인 임동인 (前)대표이…

첫 양성 발췌: [section] I. 회사의 개요 > 4. 주식의 총수 등 당사는 보고서 작성기준일 현재, 발행할 주식의 총수는 보통주 115,000,000주, 우선주 10,000,000주이며 발행한 주식의 총수는 보통주 51,534,789주, 우선주 4,177,287주 입니다.(*1) 2011년 4월 14일 2우선주가 보통주로 전환…

### V2-00121941-finance_table-02

대상은 2026년 1분기 말 특수관계자 거래로 받을 돈과 갚을 돈이 각각 얼마였어?

회사 대상. 이전 셀 dense_miss_bm25_hit. movement regression. Dense 첫 양성 None, BM25 첫 양성 7, RRF 첫 양성 24. delta vs BM25 -17, delta vs Dense None. 같은 BM25 unit의 RRF 순위 24 (delta -17). RRF 첫 양성 unit의 Dense 순위 75, BM25 순위 7. 상위 5 양성 0, 상위 10 양성 0.

RRF 1위 relevance 0: [section] III. 재무에 관한 사항 > 5. 재무제표 주석 > 31. 특수관계자 전기말 (단위 : 백만원) | | | | 특수관계자거래의 채권·채무 잔액 | | | | 매출채권, 특수관계자거래 | 대여금, 특수관계자거래 | 기타채권, 특수관계자거래 | 매입채무, 특수관계자거래 | 기타채무, 특수관계자거래 전체 …

첫 양성 발췌: [section] III. 재무에 관한 사항 > 3. 연결재무제표 주석 > 31. 특수관계자 (연결) 특수관계자거래의 채권·채무 잔액에 대한 공시 당분기말 (단위 : 백만원) | | | | 특수관계자거래의 채권·채무 잔액 | | | | 매출채권, 특수관계자거래 | 기타채권, 특수관계자거래 | 매입채무, 특수관계자거래 |…

점수 대조 `V2-00126380-governance_paragraph-01`: 첫 양성 unit은 BM25 1, Dense 705, RRF 점수 0.01770. RRF 1위는 relevance 0, BM25 32, Dense 1, RRF 점수 0.02726.

## 16. FACT

RRF Hit@1 28/45, Hit@5 33/45, Hit@10 36/45. both-hit 유지 32/32. BM25-only 유지 2/10. Dense-only 복구 1/1. both-miss 복구 1/2. requirement recall 46/58, all requirements covered 35/45.

## 17. INTERPRETATION

both-hit Hit@10은 32/32로 남았다. BM25-only는 2/10만 남았다. 손실은 BM25만 맞히던 lexical anchor에 몰린다. Dense-only 복구는 1/1이고, both-miss 복구는 1/2이다. both-miss 유입은 성공 조건이 아니다. MRR, NDCG@10, Recall@5 macro는 BM25보다 낮다. 전체 순위 RRF는 Dense 순위가 깊어도 0이 아닌 reciprocal rank를 더한다. BM25 1위 양성 unit의 Dense 순위가 수백이면, Dense 1위 irrelevant unit이 앞선다. 이 고정 k의 단순 RRF는 DEV에서 기각한다. weighted fusion과 query routing은 구현하지 않는다.

## 18. Limitations

DEV 45문항만 본다. k=60은 고정이고 sweep하지 않았다. TEST로 k, weight, routing을 고르지 않았다. 재사용한 함수는 반환 목록에 없다는 이유로 contribution을 0으로 두지 않는다. 이번 기각은 그 전체 순위 RRF에 대한 것이다. top-20 밖 contribution을 0으로 두는 변형은 실행하지 않았다. BM25 0점 문서도 깊은 순위의 reciprocal rank를 더한다. both-miss를 chunking 실패로 바꾸지 않는다. 순위 동점은 ordinal, unit_index, chunk_id로 가른다.

## 19. RRF hypothesis verdict

판정 C. C if Hit@10 drops by 2 or more, or 2 or more BM25-only Hit@10 questions are lost. Otherwise A if coverage is almost or fully retained and either the Dense-only question is recovered or MRR, NDCG@10, and Recall@5 macro improve with none of them falling. Otherwise B.

Hit@10 drop 6, BM25-only lost 8, Dense-only recovered True, ranking mrr down, ndcg@10 down, recall@5_macro down.

## 20. Recommended next experiment

전체 순위 단순 RRF를 기각한다. 반환 밖 contribution을 0으로 두는 변형은 이번 숫자가 판단하지 않는다. weighted fusion 또는 query routing은 새 DEV 가설로만 남기고 이번 결과에서 그 규칙을 만들지 않는다. both-miss 2건은 period-aware ranking, 연결/별도 scope, cross-filing retrieval 후보로 남긴다. chunking 실패로 단정하지 않는다. 다음에도 TEST로 k나 weight를 고르지 않는다.

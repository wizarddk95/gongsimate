# v2.1 DEV candidate-truncated RRF

## 1. Executive summary

판정은 B (부분 지지)이다. truncated RRF Hit@10은 40/45, full-RRF는 36/45, BM25는 42/45이다. full-RRF가 잃은 BM25-only 8건 중 Hit@10 회복은 5건이다. Dense-only Hit@10은 1/1이다. MRR 0.742 → 0.737, NDCG@10 0.708 → 0.684, Recall@5 macro 0.572 → 0.505.

## 2. Experiment commit / base

기준 commit은 `6c3fec85523545e3552a71930b56409b8be8e754` (True)이다. 실행 시점 HEAD는 `6c3fec85523545e3552a71930b56409b8be8e754` (eval: reject fixed-k full-ranking rrf on v2.1 dev), dirty=True이다. 이 실행은 그 기준 위에서 stored ranking만 읽었다.

## 3. Hypothesis

각 retriever의 deep-tail 전체 순위를 fusion signal로 쓴 것이 full-ranking RRF의 문제였다. 실제 반환 Top-20에만 RRF contribution을 주면 BM25의 lexical evidence를 덜 훼손하고 Dense의 상위 semantic signal을 보존할 수 있다.

## 4. Experiment invariants

dataset `multi_company_retrieval_v2_1` sha256 `2b1c372b7adaca29af831986bf5e6807c2ae2bd7835a52735690e227c1b03137`. 모집단은 DEV fully_answerable 45문항이다. TEST 검색 수는 0이다. N=20, k=60, Dense weight 1, BM25 weight 1이다. N sweep, k sweep, weight, alpha, query routing, reranker, query expansion, embedding, chunking, v2, v2.1 질문, qrel, split, judgment는 바꾸지 않았다. 바꾼 변수는 full-ranking RRF에서 candidate-truncated RRF로의 후보 집합이다.

## 5. Truncated-RRF semantics

후보 집합은 Dense top-20과 BM25 top-20의 union이다. 같은 retrieval_unit_id는 한 행이다. 양쪽에 있으면 contribution을 더한다. 한쪽에만 있으면 그 retriever의 contribution만 가진다. 양쪽 top-20 밖은 후보가 아니다. score(d) = I_dense(d)/(60+dense_rank(d)) + I_bm25(d)/(60+bm25_rank(d)). I는 그 retriever의 top-20에 있으면 1이고 없으면 0이다. 75, 283, 705 같은 full rank는 이 식의 입력이 아니다. 이 설정에서 교집합 최소 점수는 0.02500이고 단일 리스트 최대 점수는 0.01639이다. 교집합 점수가 더 크다. 그래서 양쪽 top-20에 있는 unit은 한쪽에만 있는 unit보다 항상 앞에 선다. 동점 정렬은 score 내림차순, retrieval_unit_id 오름차순이다.

## 6. Implementation audit

함수 `disclosure_agent.evalset_v2_1.truncated_rrf.candidate_truncated_rrf`. `disclosure_agent.hybrid.reciprocal_rank_fusion`는 호출하지 않았다. N=20 k=60 weights 1:1. 식 `score(d) = I_dense(d)/(k + dense_rank(d)) + I_bm25(d)/(k + bm25_rank(d))`. tie-break: score descending, retrieval_unit_id ascending. Stored top-20 hits have no ordinal or unit_index. retrieval_unit_id is a total order, so equal scores do not depend on dict order. 공식 대조 1555건. 점수에 들어간 최대 rank 20. deep-tail rank 사용 0. Dense top-20 일치 45/45, BM25 top-20 일치 45/45. 불일치가 있으면 fusion 점수를 쓰지 않는다.

## 7. Dense / BM25 / Full-RRF / Truncated-RRF

A. Dense-D2a는 저장된 제품 경로 순위다. 모델 `BAAI/bge-m3`, 질의 원문, 지시문 없음. B. BM25는 저장된 `Bm25Index.search` top-20이다. k1=1.2 b=0.75 tokenizer `ko-mixed-v1`. C. full-ranking RRF는 직전 `rrf_dev.json`이다. k=60이고 eligible unit 전체 순위를 썼다. D. truncated RRF는 A와 B의 top-20 union만 다시 점수한 것이다. retrieval은 재실행하지 않았다. complementarity 파일과 full-RRF 사본의 top-20은 45/45 같았다.

## 8. Headline metrics

| metric | Dense | BM25 | Full RRF | Truncated RRF |
| --- | --- | --- | --- | --- |
| Hit@1 | 20/45 | 29/45 | 28/45 | 29/45 |
| Hit@5 | 28/45 | 40/45 | 33/45 | 37/45 |
| Hit@10 | 33/45 | 42/45 | 36/45 | 40/45 |
| Recall@5 macro | 0.387 | 0.572 | 0.468 | 0.505 |
| Recall@5 micro | 78/231 | 113/231 | 95/231 | 99/231 |
| Recall@10 macro | 0.561 | 0.767 | 0.629 | 0.725 |
| Recall@10 micro | 129/231 | 174/231 | 141/231 | 166/231 |
| MRR | 0.525 | 0.742 | 0.686 | 0.737 |
| NDCG@10 condensed | 0.478 | 0.708 | 0.654 | 0.684 |
| NDCG@10 zero-filled | 0.478 | 0.708 | 0.615 | 0.675 |
| preferred_hit@10 | 25/45 | 37/45 | 31/45 | 34/45 |
| requirement_recall@10 | 41/58 | 51/58 | 46/58 | 49/58 |
| all_requirements_covered@10 | 32/45 | 41/45 | 35/45 | 39/45 |
| judged_coverage@10 | 1.000 | 1.000 | 0.787 | 0.969 |
| judged_coverage@20 | 0.536 | 0.557 | 0.597 | 0.837 |

## 9. BM25-only regression recovery

full-RRF BM25-only 손실 8건 중 truncated Hit@10 회복은 5건, 잔존은 3건이다. 교집합이 10개 이상 앞에 있어 Hit@10 밖으로 남은 질의는 0건이다. full-RRF가 이미 유지하던 BM25-only 2건 중 truncated Hit@10은 2건이다. 각 행의 Dense contribution은 top-20 밖 full rank를 쓰지 않은 값이다.

| query | BM25 rank | Dense top-20 | full RRF | truncated | recovered | intersection |
| --- | --- | --- | --- | --- | --- | --- |
| `V2-00121941-business_paragraph-02` | 1 | False | 17 | 3 | True | 1 |
| `V2-00121941-business_paragraph-03` | 8 | False | 19 | 15 | False | 0 |
| `V2-00121941-finance_table-02` | 7 | False | 24 | 14 | False | 8 |
| `V2-00121941-overview_paragraph-01` | 1 | False | 23 | 7 | True | 6 |
| `V2-00126380-governance_paragraph-01` | 1 | False | 27 | 1 | True | 0 |
| `V2-00126955-finance_paragraph-02` | 1 | False | 17 | 1 | True | 0 |
| `V2-00989619-governance_paragraph-02` | 4 | False | 18 | 11 | False | 8 |
| `V2-00989619-overview_paragraph-01` | 1 | False | 16 | 1 | True | 0 |

### V2-00126380-governance_paragraph-01

삼성전자: 삼성전자는 당기에 주식매수선택권 부여로 인식한 보상비용이 있나요?

BM25 relevant rank 1. 그 unit은 Dense top-20에 없다. 이전 full ranking의 Dense rank 705는 truncated 점수에 넣지 않았다. Dense contribution 0.0, BM25 contribution 0.016393, truncated score 0.016393. 그 unit의 truncated rank는 1이고 full-RRF rank는 27이다. 첫 양성 truncated rank는 1. 교집합 0개, 그 앞에 있는 교집합 unit 0개. Hit@10 회복 True.

이전 full-RRF 1위 relevance 0는 BM25 full rank 32, Dense full rank 1였다. 그 full rank 쌍은 truncated 점수의 입력이 아니다. truncated top3 안에서는 rank 2, Dense rank 1, BM25 rank None, score 0.016393이다. 발췌: [section] 목 차 > III. 재무에 관한 사항 > 5. 재무제표 주석 > 20. 기타자본항목 성과연동 주식보상에 대한 공시 (단위 : 백만원) | 주식기준보상약정 | 성과연동 주식보상 | 삼성전자주식회사 주식기준보상약정에 대한 기술 | 회사는 임직원을 대상으로 성과연동주식보상제도(PSU)를 도입하였습니다. 가중…

### V2-00121941-overview_paragraph-01

대상: 대상의 2026년 6월 30일 현재 총 유통주식수는 몇 주인가요?

BM25 relevant rank 1. 그 unit은 Dense top-20에 없다. 이전 full ranking의 Dense rank 283는 truncated 점수에 넣지 않았다. Dense contribution 0.0, BM25 contribution 0.016393, truncated score 0.016393. 그 unit의 truncated rank는 7이고 full-RRF rank는 23이다. 첫 양성 truncated rank는 7. 교집합 6개, 그 앞에 있는 교집합 unit 6개. Hit@10 회복 True.

이전 full-RRF 1위 relevance 0는 BM25 full rank 5, Dense full rank 12였다. 그 full rank 쌍은 truncated 점수의 입력이 아니다. truncated top3 안에서는 rank 1, Dense rank 12, BM25 rank 5, score 0.029274이다. 발췌: [section] VII. 주주에 관한 사항 ※ 2005년 11월 14일 최대주주는 임상민에서 대상홀딩스로 변경되었으며 상기소유주식수 및 지분율은 보통주 기준으로 특수관계인인 임창욱, 임세령, 대상문화재단 등을 포함한 주식수입니다. 2009년 3월 20일 제55기 정기주주총회 결과에 따른 특수관계인인 임동인 (前)대표이…

### V2-00121941-finance_table-02

대상: 대상은 2026년 1분기 말 특수관계자 거래로 받을 돈과 갚을 돈이 각각 얼마였어?

BM25 relevant rank 7. 그 unit은 Dense top-20에 없다. 이전 full ranking의 Dense rank 75는 truncated 점수에 넣지 않았다. Dense contribution 0.0, BM25 contribution 0.014925, truncated score 0.014925. 그 unit의 truncated rank는 14이고 full-RRF rank는 24이다. 첫 양성 truncated rank는 14. 교집합 8개, 그 앞에 있는 교집합 unit 8개. Hit@10 회복 False.

이전 full-RRF 1위 relevance 0는 BM25 full rank 2, Dense full rank 6였다. 그 full rank 쌍은 truncated 점수의 입력이 아니다. truncated top3 안에서는 rank 1, Dense rank 6, BM25 rank 2, score 0.031281이다. 발췌: [section] III. 재무에 관한 사항 > 5. 재무제표 주석 > 31. 특수관계자 전기말 (단위 : 백만원) | | | | 특수관계자거래의 채권·채무 잔액 | | | | 매출채권, 특수관계자거래 | 대여금, 특수관계자거래 | 기타채권, 특수관계자거래 | 매입채무, 특수관계자거래 | 기타채무, 특수관계자거래 전체 …

## 10. Dense-only semantic retention

`V2-00126380-finance_table-02` 삼성전자: 삼성전자는 전분기에 삼성에스디에스와 구매 관련 거래를 얼마만큼 했나요? Dense 첫 양성 rank 1. BM25 top-20에는 그 unit이 없다. 이전 full ranking BM25 rank 75는 점수에 넣지 않았다. full-RRF rank 6, truncated rank 2. 이 문항은 Hit@10에 남았다. 교집합은 1개다. BM25 lexical coverage를 지키는 동안 Dense rank 1 signal이 남는지가 이 사례다. 이 한 문항의 성패로 Dense 전체를 지우거나 hybrid 전체를 확정하지 않는다.

## 11. Both-hit retention

이전 both-hit 32 중 truncated Hit@10은 32이다. 새로 잃은 질의는 0건. full-RRF에서는 32/32가 유지됐다.

## 12. Both-miss

이수페타시스 cross-filing과 현금흐름 period-scope 문항은 truncated RRF의 성공 조건이 아니다. Hit@10에 들어와도 period, scope, cross-filing 문제가 해결된 것이 아니다.

- `V2-00107613-cross_period-02` 이수페타시스: Dense None, BM25 None, full-RRF 135, truncated None, truncated Hit@10 False. 이수페타시스의 2024년 반기보고서와 3분기보고서에는 투자조합의 지분 정보와 받을 돈의 연체 현황이 각각 어떻게 나와 있나요?
- `V2-00107613-finance_table-03` 이수페타시스: Dense 20, BM25 None, full-RRF 10, truncated 39, truncated Hit@10 False. 이수페타시스의 2026년 상반기 투자활동에서 무형자산 취득과 투자부동산 취득은 각각 어느 정도를 차지했나요?

## 13. Rank movement

BM25 대비 movement `{"improvement": 5, "neutral": 27, "recovered": 1, "regression": 10, "still_missing": 2}`. first-relevant delta 평균 -0.786, 중앙값 0.000. top-20 교집합 크기는 최소 0, 중앙값 6.000, 최대 16, 평균 5.444이다. 교집합이 10 이상인 질의는 8개다.

## 14. Requirement transition

requirement recall@10 Dense 41/58, BM25 51/58, full-RRF 46/58, truncated 49/58.
all requirements covered@10 Dense 32/45, BM25 41/45, full-RRF 35/45, truncated 39/45.

full-RRF transition:
- both prior 40: `{"retained": 40}`
- dense_only prior 1: `{"recovered": 1}`
- bm25_only prior 11: `{"lost": 8, "retained": 3}`
- neither prior 6: `{"recovered": 2, "still_missing": 4}`

truncated RRF transition:
- both prior 40: `{"retained": 40}`
- dense_only prior 1: `{"recovered": 1}`
- bm25_only prior 11: `{"lost": 3, "retained": 8}`
- neither prior 6: `{"still_missing": 6}`

## 15. Judgment coverage

unjudged는 nonrelevant가 아니다. zero-filled NDCG는 민감도 지표다.

| system | judged@10 | judged@20 | mean unjudged@10 | NDCG condensed | NDCG zero-filled |
| --- | --- | --- | --- | --- | --- |
| dense | 1.000 | 0.536 | 0.000 | 0.478 | 0.478 |
| bm25 | 1.000 | 0.557 | 0.000 | 0.708 | 0.708 |
| full_rrf | 0.787 | 0.597 | 2.133 | 0.654 | 0.615 |
| truncated_rrf | 0.969 | 0.837 | 0.311 | 0.684 | 0.675 |

truncated judged_coverage@10은 0.969이고 mean unjudged@10은 0.311이다. full-RRF는 0.787, 2.133이다. 후보가 top-20 union으로 줄면 full-RRF가 끌어올리던 deep-tail unjudged가 상위 10에 덜 섞인다. 그 변화는 평가 확실성을 올린다. 남은 unjudged를 오답으로 세지 않는다.

## 16. Representative improvements

### V2-00989619-finance_table-02

알테오젠 분기보고서에서 기술 지원과 위탁생산 용역 기간은 언제부터 언제까지로 나와 있어?

회사 알테오젠. 이전 셀 dense_hit_bm25_hit. movement improvement. Dense 첫 양성 8, BM25 첫 양성 5, full-RRF 2, truncated 2. 교집합 4.

truncated 1위 relevance 0, Dense rank 9, BM25 rank 4, score 0.030118.

### V2-00107613-finance_table-01

이수페타시스의 2024년 상반기 연결 현금흐름표에서 투자활동현금흐름은 얼마였나요?

회사 이수페타시스. 이전 셀 dense_hit_bm25_hit. movement improvement. Dense 첫 양성 9, BM25 첫 양성 3, full-RRF 1, truncated 1. 교집합 10.

truncated 1위 relevance 2, Dense rank 9, BM25 rank 3, score 0.030366.

### V2-00989619-mdna_paragraph-02

알테오젠은 사업보고서에 적은 향후 예상이 실제 결과와 다를 수 있다고 어떤 점을 주의시키나요?

회사 알테오젠. 이전 셀 dense_hit_bm25_hit. movement improvement. Dense 첫 양성 1, BM25 첫 양성 3, full-RRF 1, truncated 1. 교집합 2.

truncated 1위 relevance 2, Dense rank 7, BM25 rank 3, score 0.030798.

## 17. Representative regressions

### V2-00121941-business_paragraph-03

대상이 종속·관계·공동기업을 제외하고 보유한 지분증권 투자액은 2024년 말 장부상 얼마였나요?

회사 대상. 이전 셀 dense_miss_bm25_hit. movement regression. Dense 첫 양성 None, BM25 첫 양성 8, full-RRF 19, truncated 15. 교집합 0.

truncated 1위 relevance 0, Dense rank None, BM25 rank 1, score 0.016393.

### V2-00121941-finance_table-02

대상은 2026년 1분기 말 특수관계자 거래로 받을 돈과 갚을 돈이 각각 얼마였어?

회사 대상. 이전 셀 dense_miss_bm25_hit. movement regression. Dense 첫 양성 None, BM25 첫 양성 7, full-RRF 24, truncated 14. 교집합 8.

truncated 1위 relevance 0, Dense rank 6, BM25 rank 2, score 0.031281.

### V2-00989619-governance_paragraph-02

알테오젠은 이번 분기에 우선주 몇 주를 보통주로 바꿨나요?

회사 알테오젠. 이전 셀 dense_miss_bm25_hit. movement regression. Dense 첫 양성 None, BM25 첫 양성 4, full-RRF 18, truncated 11. 교집합 8.

truncated 1위 relevance 0, Dense rank 2, BM25 rank 6, score 0.031281.

## 18. FACT

truncated Hit@1 29/45, Hit@5 37/45, Hit@10 40/45. BM25-only 8건 회복 5/8. both-hit 32/32. Dense-only 1/1. both-miss Hit@10 0/2. requirement recall 49/58, all requirements 39/45. judged_coverage@10 0.969.

## 19. INTERPRETATION

full-RRF Hit@10 36/45보다 truncated는 분명 높고, BM25-only 8건 중 5건을 되돌렸다. BM25보다는 아직 열세이고 lexical 보존과 semantic 추가 사이의 trade-off가 남아 있다. 삼성전자 구매/매입 truncated rank는 2이다. 이 문장은 Top-20 candidate-truncated, equal-weight, k=60 RRF가 현재 v2.1 DEV에서 보인 결과다. Hybrid Search 일반으로 확장하지 않는다. Dense-only semantic case는 이미 관측된 사실이고, 이번 성패로 Dense가 필요 없다는 결론을 만들지 않는다. 교집합 중앙값은 6.000이고, 10 이상인 질의는 8개다. 양쪽 top-20에 있는 unit은 한쪽에만 있는 unit보다 항상 앞에 선다.

## 20. Limitations

DEV 45문항만 본다. TEST로 N, k, weight를 고르지 않았다. 저장된 top-20을 재사용했고 retrieval은 다시 돌리지 않았다. 동점은 retrieval_unit_id로 가른다. ordinal은 저장 순위에 없다. unjudged를 nonrelevant로 두지 않는다. both-miss 2건으로 period, scope, cross-filing 해결을 말하지 않는다. 다음 실험으로 weighted RRF, score fusion, routing, reranker, period-aware ranking을 구현하지 않는다.

## 21. Hypothesis verdict

판정 B (부분 지지). A when Hit@10 is at least 41/45, Dense-only Hit@10 remains, at least 6 of the 8 full-RRF BM25-only losses return to Hit@10, both-hit losses are 0, previously kept BM25-only hits remain, MRR, NDCG@10, and Recall@5 macro each drop by at most 0.05, and requirement recall is at least 49/58 with all-requirements at least 39/45. Otherwise B when Hit@10 is at least 3 above full-RRF and at least 4 of the 8 are recovered. Otherwise C.

Hit@10 40/45, full-RRF 36/45, BM25-only 8건 회복 5, Dense-only retained True, both-hit lost 0, mrr drop 0.005, ndcg@10 drop 0.025, recall@5_macro drop 0.067.

## 22. Recommended next hypothesis

미회복 문항은 BM25 단독 순위가 교집합과 Dense 단독의 더 앞선 등위 뒤에 놓인 BM25 8위가 truncated 15위, BM25 7위가 truncated 14위, BM25 4위가 truncated 11위이다. 이 등위 결합이 Hit@10 잔여 손실의 원인인지가 다음 가설이다.

## 23. Portfolio Summary


### Hypothesis

Top-20 밖 deep-tail 순위를 빼면 BM25 lexical Hit@10을 지키면서 Dense의 상위 semantic signal을 더할 수 있다.

### Success Condition

- BM25 Hit@10 42/45를 유지하거나 41/45 이상.
- 삼성전자 구매/매입 Dense-only 문항이 Hit@10에 남는다.
- full-RRF가 잃은 BM25-only 8건의 대부분을 Hit@10으로 되돌린다.
- MRR, NDCG@10, Recall@5 macro가 BM25보다 0.05를 넘겨 떨어지지 않는다.
- requirement recall과 all-requirements covered가 BM25 근처에 남는다.

### Result

Hit@10 40/45, BM25-only 8건 중 회복 5, Dense-only Hit@10 1/1, MRR 0.737.

### Why It Worked / Failed

deep-tail rank를 빼자 BM25-only 8건 중 5건이 Hit@10으로 돌아왔고 Dense-only는 truncated rank 2에 남았다. 남은 손실은 BM25 단독 rank 4, 7, 8이 교집합과 상대 리스트의 더 높은 등위 뒤에 선 것이다. k=60, N=20에서 교집합 최소 점수는 0.02500이고 단일 리스트 최대 점수는 0.01639이다.

### Decision

부분 지지

### Next Hypothesis

미회복 문항은 BM25 단독 순위가 교집합과 Dense 단독의 더 앞선 등위 뒤에 놓인 BM25 8위가 truncated 15위, BM25 7위가 truncated 14위, BM25 4위가 truncated 11위이다. 이 등위 결합이 Hit@10 잔여 손실의 원인인지가 다음 가설이다.

| Experiment | Hypothesis | Success condition | Result | Verdict | Next |
| --- | --- | --- | --- | --- | --- |
| Dense-D2a baseline | v2.1 DEV에서 Dense-D2a 단독 순위를 측정한다. | DEV fully_answerable 45의 Hit, Recall, MRR, NDCG를 고정 산출물로 남긴다. | Hit@10 33/45, MRR 0.525, NDCG@10 0.478, requirement recall 41/58. | 기준선 확정 | BM25와 실패가 겹치는지 본다. |
| Dense vs BM25 complementarity | Dense와 BM25의 Hit@10 실패가 서로 다른가. | BM25가 Dense 실패를 복구하고 both-miss가 비대칭 셀보다 작다. | both 32, Dense only 1, BM25 only 10, both miss 2. BM25 Hit@10 42/45. | 상보성 관측 | 전체 순위 RRF가 그 차이를 순위로 옮기는지 본다. |
| Full-ranking RRF | k=60 전체 순위 RRF가 BM25 coverage를 유지한 채 Dense-only를 복구하는가. | Hit@10 감소 1 이하, BM25-only 손실 1 이하, Dense-only 복구 또는 순위 품질 개선. | Hit@10 36/45, BM25-only 2/10 유지, Dense-only 1/1, MRR 0.686. | 기각 | Top-20 밖 순위를 fusion에서 뺀다. |
| Candidate-truncated RRF | Top-20 밖 deep-tail 순위를 빼면 BM25 lexical Hit@10을 지키면서 Dense의 상위 semantic signal을 더할 수 있다. | BM25 Hit@10 41/45 이상, Dense-only 유지, full-RRF가 잃은 BM25-only 8건의 대부분 회복, MRR/NDCG/Recall@5 하락 0.05 이내, requirement가 BM25 근처. | Hit@10 40/45, BM25-only 8건 중 회복 5, Dense-only Hit@10 1/1, MRR 0.737. | 부분 지지 | 미회복 문항은 BM25 단독 순위가 교집합과 Dense 단독의 더 앞선 등위 뒤에 놓인 BM25 8위가 truncated 15위, BM25 7위가 truncated 14위, BM25 4위가 truncated 11위이다. 이 등위 결합이 Hit@10 잔여 손실의 원인인지가 다음 가설이다. |


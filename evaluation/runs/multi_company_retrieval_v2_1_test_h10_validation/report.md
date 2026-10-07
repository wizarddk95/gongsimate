# v2.1 TEST one-shot validation: bm25_primary_dense_head

이 실행은 `multi_company_retrieval_v2_1` fully_answerable TEST의 최초이자 유일한 frozen validation이다. retrieval 실행 횟수는 1이다.

## 1. Executive Summary

DEV에서 failure mechanism 기반으로 선택한 BM25-primary asymmetric policy가 독립 TEST에서도 BM25 baseline을 보존하면서 semantic miss를 추가 복구했다.

판정은 `Generalization Support`이고 reason은 `bm25_preserved_and_semantic_recovery`이다. TEST Hit@10은 BM25 47/55, H10 48/55, Dense 32/55이다.
질의 전이는 both hit 47, BM25 only 0, H10 only 1, both miss 7이다. requirement 유입 1, 상실 0이다.
Recall@10 micro는 BM25 199/282에서 H10 197/282이다. NDCG@10 condensed 하락은 +0.005889이다. 새 Hit@10은 1건이다.
DEV에서 H10이 Strong Support였던 증분은 저장된 gained 목록 기준 Hit@10 1건, requirement 1건이다. 그 크기를 큰 성능 향상으로 쓰지 않는다. TEST 숫자도 같은 방식으로 적는다.

## 2. Frozen Configuration

frozen commit `f9112833f83967863353c87d30fe34fc6bb08cf0`.
rule `bm25_primary_dense_head`. 함수 `disclosure_agent.evalset_v2_1.asymmetric_fusion.bm25_primary_dense_head`.
BM25 rank 1..9는 유지한다. Dense rank 1이 BM25 top-10 안에 있으면 순서를 유지한다. 밖에 있으면 그 head를 rank 10에 두고 기존 BM25 rank 10을 뒤로 민다. Dense rank 2..20, intersection bonus, weight, score normalization, RRF는 쓰지 않는다.
정책 blob `917ff1375e21aa355e6438c8947207657af2f138`. semantics blob `3fc59d1465261482a9f449cdd5d02d3c843d4f5c`.
이 blob은 commit f911283의 파일과 같다. TEST 점수 이후에 정책을 고치지 않았다.

## 3. Evaluation Integrity

- frozen commit: `f9112833f83967863353c87d30fe34fc6bb08cf0`
- rule: `bm25_primary_dense_head`
- policy blob: `917ff1375e21aa355e6438c8947207657af2f138`
- semantics sha256: `71b87fd2a0569cf2d95552e1e47783f66cc092d8a938db2d8c5ab52ea9081f8e`
- benchmark: `multi_company_retrieval_v2_1` sha256 `2b1c372b7adaca29af831986bf5e6807c2ae2bd7835a52735690e227c1b03137`
- split: seed 20261005, DEV 45, TEST 55, overlap 0
- TEST query id sha256: `a05d3c6cb63efc73b8107877c7c4511a0beb8a2a2f0919ffb064c9af13a67ce0`
- DEV query id sha256: `9e580beaee9237bd6a4f9cbd4eb74f8341331ebe612f58553771b7ea06445234`
- TEST retrieval 실행 횟수: 1
- headline 문항: 55
- positive qrel이 없어 headline에서 빠진 문항: 0
- Dense ranking sha256: `74ce429c7859987f40078a3ab936275f1cd45bd3914572862f754b98baa6050c`
- BM25 ranking sha256: `58fcd07480d882e77c5591b3f1aa93e3cc8dde4e6738451b0f239f33d9e5494d`
- qrels sha256: `2ef0d1d179897034bf891bdad0454a8dc16004d751036b34e6517a337f7b63e3`
- qrel count: 3112
- judgments sha256: `c952fd4740bc971e8c0a8e2c742712c2acb9a615ddf35fa926d0a4d524f6af51`
- judgments reused: 2414
- new judge calls: 0
- prompt version: `retrieval-evalset-v2.1`
- label source: v2 LLM judgments reused at retrieval-unit granularity
- 검색 시점 git: `f9112833f83967863353c87d30fe34fc6bb08cf0` dirty True
- 채점 시점 git: `f9112833f83967863353c87d30fe34fc6bb08cf0` dirty True
- Full-RRF와 Candidate-Truncated RRF는 이번 TEST configuration이 아니다.
- DEV는 다시 튜닝하지 않았다.

## 4. TEST Metrics

| metric | Dense-D2a | BM25 | H10 |
| --- | --- | --- | --- |
| Hit@1 | 19/55 | 36/55 | 36/55 |
| Hit@5 | 27/55 | 45/55 | 45/55 |
| Hit@10 | 32/55 | 47/55 | 48/55 |
| Recall@5 macro | 0.246 | 0.495 | 0.495 |
| Recall@5 micro | 69/282 | 127/282 | 127/282 |
| Recall@10 macro | 0.392 | 0.701 | 0.693 |
| Recall@10 micro | 116/282 | 199/282 | 197/282 |
| MRR | 0.411 | 0.717 | 0.718 |
| NDCG@10 condensed | 0.325 | 0.640 | 0.635 |
| NDCG@10 zero-filled | 0.325 | 0.640 | 0.635 |
| requirement_recall@10 | 37/66 | 56/66 | 57/66 |
| all_requirements_covered@10 | 29/55 | 45/55 | 46/55 |
| judged_coverage@10 | 1.000 | 1.000 | 1.000 |
| preferred_hit@10 (diagnostic) | 21/55 | 38/55 | 39/55 |

NDCG headline은 condensed다. zero-filled는 sensitivity다. preferred_hit@10은 diagnostic이다. 하락은 BM25 - H10이고 양수면 H10이 낮다. BM25 대비 하락: recall@5_macro +0.000000, recall@10_macro +0.007944, mrr -0.001818, ndcg@10_condensed +0.005889.

## 5. Query-level BM25 ↔ H10 Transition

both hit 47, BM25 only 0, H10 only 1, both miss 7.

BM25 only: 없음.

H10 only: `V2-00107613-overview_paragraph-02`.

both miss: `V2-00107613-business_table-02`, `V2-00107613-finance_table-02`, `V2-00121941-finance_table-01`, `V2-00126955-governance_paragraph-02`, `V2-00126955-mdna_paragraph-02`, `V2-00126955-overview_paragraph-02`, `V2-00989619-overview_paragraph-02`.

## 6. Requirement-level Transition

requirement 66. both 56, BM25 only 0, H10 only 1, neither 9.

H10 only: `V2-00107613-overview_paragraph-02/R1`.

BM25 only: 없음.

requirement_recall@10 57/66. all_requirements_covered@10 46/55. BM25 requirement_recall@10 56/66. BM25 all_requirements_covered@10 45/55.

## 7. Dense-head Insertions

질의 55. edit none 16, insertion 37, promotion 2. insertion/promotion 합계 39.
삽입된 Dense head 판정: {"relevant": 8, "irrelevant": 31, "unjudged": 0}.
밀려난 BM25 rank 10 판정: {"relevant": 10, "irrelevant": 29, "unjudged": 0}.
새 Hit@10 1: `V2-00107613-overview_paragraph-02`. 기존 Hit@10 상실 0: 없음.
편집 질의의 requirement 유입 1: `V2-00107613-overview_paragraph-02/R1`.
편집 질의의 requirement 상실 0: 없음.

| query | company | edit | head BM25 rank | head judgment | displaced judgment | BM25 Hit@10 | H10 Hit@10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| V2-00107613-business_paragraph-01 | 이수페타시스 | insertion | None | irrelevant | relevant | True | True |
| V2-00107613-business_paragraph-03 | 이수페타시스 | insertion | None | irrelevant | irrelevant | True | True |
| V2-00107613-business_table-02 | 이수페타시스 | insertion | None | irrelevant | irrelevant | False | False |
| V2-00107613-cross_period-01 | 이수페타시스 | insertion | None | irrelevant | irrelevant | True | True |
| V2-00107613-finance_paragraph-01 | 이수페타시스 | insertion | None | irrelevant | irrelevant | True | True |
| V2-00107613-finance_paragraph-02 | 이수페타시스 | insertion | None | irrelevant | irrelevant | True | True |
| V2-00107613-finance_table-02 | 이수페타시스 | insertion | None | irrelevant | irrelevant | False | False |
| V2-00107613-governance_paragraph-01 | 이수페타시스 | insertion | None | irrelevant | irrelevant | True | True |
| V2-00107613-multi_element-02 | 이수페타시스 | insertion | None | relevant | relevant | True | True |
| V2-00107613-overview_paragraph-02 | 이수페타시스 | insertion | None | relevant | irrelevant | False | True |
| V2-00121941-cross_period-02 | 대상 | insertion | None | irrelevant | irrelevant | True | True |
| V2-00121941-finance_table-01 | 대상 | insertion | None | irrelevant | irrelevant | False | False |
| V2-00121941-finance_table-03 | 대상 | insertion | None | irrelevant | irrelevant | True | True |
| V2-00121941-governance_paragraph-01 | 대상 | insertion | None | irrelevant | irrelevant | True | True |
| V2-00121941-governance_paragraph-02 | 대상 | insertion | None | irrelevant | relevant | True | True |
| V2-00121941-mdna_paragraph-02 | 대상 | insertion | None | relevant | irrelevant | True | True |
| V2-00121941-multi_element-02 | 대상 | insertion | None | irrelevant | irrelevant | True | True |
| V2-00121941-overview_paragraph-02 | 대상 | insertion | None | relevant | irrelevant | True | True |
| V2-00126380-business_paragraph-03 | 삼성전자 | insertion | None | relevant | irrelevant | True | True |
| V2-00126380-business_table-01 | 삼성전자 | insertion | None | irrelevant | irrelevant | True | True |
| V2-00126380-cross_period-02 | 삼성전자 | insertion | None | relevant | irrelevant | True | True |
| V2-00126380-overview_paragraph-01 | 삼성전자 | promotion | 17 | irrelevant | irrelevant | True | True |
| V2-00126380-overview_paragraph-02 | 삼성전자 | insertion | None | irrelevant | irrelevant | True | True |
| V2-00126955-business_table-02 | 삼양식품 | insertion | None | irrelevant | irrelevant | True | True |
| V2-00126955-finance_table-02 | 삼양식품 | insertion | None | irrelevant | relevant | True | True |
| V2-00126955-finance_table-03 | 삼양식품 | insertion | None | irrelevant | relevant | True | True |
| V2-00126955-governance_paragraph-01 | 삼양식품 | insertion | None | irrelevant | irrelevant | True | True |
| V2-00126955-governance_paragraph-02 | 삼양식품 | promotion | 19 | irrelevant | irrelevant | False | False |
| V2-00126955-mdna_paragraph-02 | 삼양식품 | insertion | None | irrelevant | irrelevant | False | False |
| V2-00126955-multi_element-01 | 삼양식품 | insertion | None | relevant | relevant | True | True |
| V2-00989619-business_paragraph-02 | 알테오젠 | insertion | None | irrelevant | relevant | True | True |
| V2-00989619-business_paragraph-03 | 알테오젠 | insertion | None | relevant | relevant | True | True |
| V2-00989619-business_table-01 | 알테오젠 | insertion | None | irrelevant | irrelevant | True | True |
| V2-00989619-cross_period-01 | 알테오젠 | insertion | None | irrelevant | irrelevant | True | True |
| V2-00989619-cross_period-02 | 알테오젠 | insertion | None | irrelevant | relevant | True | True |
| V2-00989619-finance_paragraph-02 | 알테오젠 | insertion | None | irrelevant | irrelevant | True | True |
| V2-00989619-finance_table-01 | 알테오젠 | insertion | None | irrelevant | irrelevant | True | True |
| V2-00989619-governance_paragraph-01 | 알테오젠 | insertion | None | irrelevant | relevant | True | True |
| V2-00989619-mdna_paragraph-01 | 알테오젠 | insertion | None | irrelevant | irrelevant | True | True |

## 8. Representative Gain Cases

유입 사례 1. Hit@10이 바뀌거나 requirement가 바뀐 질의 전부다.

### `V2-00107613-overview_paragraph-02` 이수페타시스

이수페타시스는 주로 어떤 제품을 만들어 판매하나요?

edit insertion. transition h10_only. BM25 첫 양성 None, Dense 첫 양성 1, H10 첫 양성 10. head relevant, displaced irrelevant. Hit@10 BM25 False, H10 True. requirement 유입 R1. requirement 상실 없음.

| rank | bm25 rank | dense head | judgment | unit |
| --- | --- | --- | --- | --- |
| 1 | 1 | False | irrelevant | `ab28c2e7ed81f6d5ea5c929e4f9c49c8689ecaca5221960f45138963ea69da74` |
| 2 | 2 | False | irrelevant | `880611f232ae5ef9c8a8c3d64345938d3b71f1869229aa86d28593bb62d47b40` |
| 3 | 3 | False | irrelevant | `d85ef80c375693ce4b41723bb13e2e32c249d9ed5db2201df1d0f9da18c9f408` |
| 4 | 4 | False | irrelevant | `517b47399136a158f6a2a858f793a2c6bb8105dd8ed9f5f6771dfd8de1ecac06` |
| 5 | 5 | False | irrelevant | `65b93f887b1d101d1fd0587072cf9f0f46f66260a7b86ef5d9b814d78d1c85e4` |
| 6 | 6 | False | irrelevant | `9a5c4f9094e54d7d23aec1ae8bcfc30e7d8d88f38e591d12a5c6450d7638e3f6` |
| 7 | 7 | False | irrelevant | `9b56049e7781f1feff4e7a04699202c292f0ca564991975f4f19993a6686243d` |
| 8 | 8 | False | irrelevant | `c2fa0087f4c160dc237a01751c7b2fbf0f1c104c7c3fd12d6e26037763faaba2` |
| 9 | 9 | False | irrelevant | `f43c8a265337b2de6579d13b11a6e973118e6daf50fe3740010635cd3a21adc7` |
| 10 | None | True | relevant | `c20dda19329ed1d205e59d9c9f07753365e346ef069454310c3aab754032cffe` |

## 9. Representative Regression Cases

회귀 사례가 없다.

## 10. Judgment Coverage

unjudged는 irrelevant로 넣지 않았다. zero-filled NDCG는 headline이 아니다.

condensed NDCG와 zero-filled NDCG가 다른 system이 없다.

| system | judged_coverage@10 | unjudged mean@10 | NDCG condensed | NDCG zero-filled |
| --- | --- | --- | --- | --- |
| dense | 1.000 | 0.000 | 0.325 | 0.325 |
| bm25 | 1.000 | 0.000 | 0.640 | 0.640 |
| h10 | 1.000 | 0.000 | 0.635 | 0.635 |

## 11. FACT

검색 문항 55. headline 문항 55. 제외 0. 실행 횟수 1.
Hit@10 Dense 32/55, BM25 47/55, H10 48/55.
Hit@1과 Hit@5와 Recall@5는 BM25와 H10이 같다. Hit@1 36/55, Hit@5 45/55, Recall@5 macro 0.495.
질의 전이 both hit 47, BM25 only 0, H10 only 1, both miss 7.
requirement both 56, BM25 only 0, H10 only 1, neither 9.
edit none 16, insertion 37, promotion 2.
편집 질의의 Dense head 판정 {"relevant": 8, "irrelevant": 31, "unjudged": 0}. 밀려난 BM25 rank 10 판정 {"relevant": 10, "irrelevant": 29, "unjudged": 0}.
새 Hit@10 1. 상실 Hit@10 0. requirement 유입 1. requirement 상실 0.
하락: recall@5_macro +0.000000, recall@10_macro +0.007944, mrr -0.001818, ndcg@10_condensed +0.005889.
judged_coverage@10 Dense 1.000, BM25 1.000, H10 1.000.

## 12. INTERPRETATION

TEST에서 BM25 Hit@10 질의 상실이 없고, H10만 맞춘 질의가 있으며, requirement는 손실이 없거나 순증이고, ranking 하락이 사전 한계 0.05 안에 있다. DEV에서 고른 asymmetric policy의 방향이 독립 TEST에서 다시 나타났다.

체크: {"hit10_maintained_or_improved": true, "no_query_level_bm25_hit10_loss": true, "additional_recovery": true, "requirement_no_loss_or_net_gain": true, "ranking_drop_within_frozen_limit": true}.

DEV Strong Support의 절대 분자 하한은 이번 판정에 넣지 않았다. 그 하한은 DEV 45문항에 고정된 값이다. 큰 ranking regression의 경계는 그때 고정한 0.05를 그대로 썼다.
삽입된 head가 relevant여도 BM25 rank 1..9에 이미 양성이 있으면 Hit@10은 늘지 않는다. 그 경우는 redundant coverage다. relevant BM25 rank 10이 밀려도 rank 1..9에 양성이 남으면 Hit@10은 유지된다.
이번 TEST에서 relevant Dense head는 8개, relevant로 밀려난 BM25 rank 10은 10개다. Recall@10 micro가 줄어든 크기는 그 교환의 순효과다. 그 순효과는 Hit@10 질의 상실이나 requirement 상실로 이어지지 않았다.

## 13. DEV vs TEST comparison

DEV는 historical reference다. verdict `Strong Support`, reason `all_strong_floors`, 문항 45. 저장된 Hit@10 유입 1, requirement 유입 1, requirement 상실 0, BM25-only Hit@10 상실 0.

DEV 증분은 대부분 그 Hit@10 유입 건수다. 큰 성능 향상으로 적지 않는다.

| metric | DEV BM25 | DEV H10 | TEST Dense | TEST BM25 | TEST H10 |
| --- | --- | --- | --- | --- | --- |
| Hit@1 | 29/45 | 29/45 | 19/55 | 36/55 | 36/55 |
| Hit@5 | 40/45 | 40/45 | 27/55 | 45/55 | 45/55 |
| Hit@10 | 42/45 | 43/45 | 32/55 | 47/55 | 48/55 |
| Recall@5 macro | 0.572 | 0.572 | 0.246 | 0.495 | 0.495 |
| Recall@10 macro | 0.767 | 0.787 | 0.392 | 0.701 | 0.693 |
| MRR | 0.742 | 0.744 | 0.411 | 0.717 | 0.718 |
| NDCG@10 condensed | 0.708 | 0.713 | 0.325 | 0.640 | 0.635 |
| requirement_recall@10 | 51/58 | 52/58 | 37/66 | 56/66 | 57/66 |
| all_requirements_covered@10 | 41/45 | 42/45 | 29/55 | 45/55 | 46/55 |
| judged_coverage@10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Full-RRF와 Candidate-Truncated RRF는 DEV에서 이미 끝난 실험이다. TEST 후보로 돌리지 않았다.

## 14. Generalization Verdict

Generalization Support.

DEV에서 선택한 asymmetric policy가 독립 TEST에서도 방향적으로 재현됐다.

reason `bm25_preserved_and_semantic_recovery`. Support when Hit@10 is maintained or improved, no BM25 Hit@10 query is lost, at least one H10-only Hit@10 recovery exists, requirement loss is zero or the gain is strictly larger, and no ranking drop exceeds 0.05. Failure when Hit@10 falls, a ranking drop exceeds that frozen limit, or requirement losses exceed gains. Otherwise Neutral / Mixed. DEV numerator floors are not reused.

## 15. Final Retrieval Decision

DEV에서 failure mechanism 기반으로 선택한 BM25-primary asymmetric policy가 독립 TEST에서도 BM25 baseline을 보존하면서 semantic miss를 추가 복구했다.

선택 configuration: `bm25_primary_dense_head`.
retune: false. 제품 검색 경로를 바꾸는 변경은 없다.
TEST 결과를 보고 rule, rank slot, Dense rank threshold, weight, fallback, query routing, reranker, embedding, chunking을 바꾸지 않는다.

## 16. Limitations

모집단은 fully_answerable TEST 55문항 한 split이다.
retrieval은 한 번이다. 이 점수로 두 번째 TEST를 열지 않는다.
unjudged unit은 미판정이다. relevance 0으로 채우지 않는다.
preferred_hit@10은 diagnostic이다.
NDCG headline은 condensed다.
DEV의 41/45, 49/58, 39/45 하한은 TEST 분자에 다시 적용하지 않았다.
ranking 하락 0.05는 DEV에서 고정한 경계이고, TEST 점수에 맞춰 다시 고르지 않았다.
Full-RRF와 truncated RRF는 이번 비교의 팔이 아니다.
작은 유입 건수를 큰 향상으로 쓰지 않는다.
qrel과 judgment는 수정하지 않았다. 새 judge call은 0이다.

## 17. Portfolio-ready Summary

## Portfolio Summary

### Hypothesis

DEV에서 고정한 BM25-primary Dense-head policy가 독립 TEST에서도 BM25 Hit@10을 손상시키지 않고 Dense semantic miss를 추가 복구하는가.

### Result

TEST fully_answerable 55문항, 실행 1회. BM25 Hit@10 47/55, H10 Hit@10 48/55. H10-only 1, BM25-only 0. requirement 유입 1, 상실 0.

### Verdict

Generalization Support. reason `bm25_preserved_and_semantic_recovery`.

### Decision

DEV에서 failure mechanism 기반으로 선택한 BM25-primary asymmetric policy가 독립 TEST에서도 BM25 baseline을 보존하면서 semantic miss를 추가 복구했다.

최종 configuration은 `bm25_primary_dense_head`이다. TEST를 본 뒤 rule, slot, weight, fallback, routing, reranker, embedding, chunking은 바꾸지 않았다.

## 18. pytest / ruff

`uv run pytest -q`는 326 passed, warning 1이다. `uv run ruff check .`는 통과했다.

## 19. artifacts

`evaluation/runs/multi_company_retrieval_v2_1_test_h10_validation/`

- `evaluation/runs/multi_company_retrieval_v2_1_test_h10_validation/execution_lock.json`
- `evaluation/runs/multi_company_retrieval_v2_1_test_h10_validation/search_started.json`
- `evaluation/runs/multi_company_retrieval_v2_1_test_h10_validation/experiment_config.json`
- `evaluation/runs/multi_company_retrieval_v2_1_test_h10_validation/evaluation_integrity.json`
- `evaluation/runs/multi_company_retrieval_v2_1_test_h10_validation/dense_test.json`
- `evaluation/runs/multi_company_retrieval_v2_1_test_h10_validation/bm25_test.json`
- `evaluation/runs/multi_company_retrieval_v2_1_test_h10_validation/dense_scored.json`
- `evaluation/runs/multi_company_retrieval_v2_1_test_h10_validation/bm25_scored.json`
- `evaluation/runs/multi_company_retrieval_v2_1_test_h10_validation/h10_test.json`
- `evaluation/runs/multi_company_retrieval_v2_1_test_h10_validation/headline_comparison.json`
- `evaluation/runs/multi_company_retrieval_v2_1_test_h10_validation/per_query_transition.json`
- `evaluation/runs/multi_company_retrieval_v2_1_test_h10_validation/per_query_transition.csv`
- `evaluation/runs/multi_company_retrieval_v2_1_test_h10_validation/requirement_transition.json`
- `evaluation/runs/multi_company_retrieval_v2_1_test_h10_validation/dense_head_insertions.json`
- `evaluation/runs/multi_company_retrieval_v2_1_test_h10_validation/judgment_coverage.json`
- `evaluation/runs/multi_company_retrieval_v2_1_test_h10_validation/representative_cases.json`
- `evaluation/runs/multi_company_retrieval_v2_1_test_h10_validation/dev_vs_test.json`
- `evaluation/runs/multi_company_retrieval_v2_1_test_h10_validation/verdict.json`
- `evaluation/runs/multi_company_retrieval_v2_1_test_h10_validation/report.md`
- `evaluation/runs/multi_company_retrieval_v2_1_test_h10_validation/portfolio_summary.md`

`docs/worklogs/2026-10-07_v21-test-h10-frozen-validation.md`

## 20. commit

정책 원본 commit `f9112833f83967863353c87d30fe34fc6bb08cf0`.
TEST 검색 시점 HEAD `f9112833f83967863353c87d30fe34fc6bb08cf0`, dirty True.
이 보고서 채점 시점 HEAD `f9112833f83967863353c87d30fe34fc6bb08cf0`, dirty True.
검증 산출물의 commit은 채점 뒤에 남긴다. 그 commit은 정책 blob을 f911283에서 바꾸지 않는다.

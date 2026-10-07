# v2.1 DEV H11 ranking failure versus candidate generation

## 1. Executive Summary

판정은 `Partial Support`이고 reason은 `candidate_generation_comparable_or_larger`이다. DEV H10 Hit@10 failure 2건 중 ranking failure는 1건, candidate-generation failure는 1건이다.
C20 Oracle Hit@10은 44/45이다. H10 Hit@10은 43/45이다. 차이는 1이다.
Requirement recall은 H10 52/58, C20 oracle 54/58이다. 차이는 2이다.
이 차이는 reranker 성능이 아니다. C20 relevance를 완벽히 알고 Top-10을 고르는 upper bound다.

## 2. H11 Hypothesis

남은 retrieval failure 중 유의미한 부분이 candidate generation failure가 아니라 ranking failure인가?

## 3. Preregistered Semantics

문서 `docs/experiments/2026-10-07_h11_ranking_failure_semantics.md` sha256 `e900ea26a1da69ae417d35ab303202a128539394edf38f8101c0d50390412119`.
POOL_ID = C20. CANDIDATE_N = 20. HIT_K = 10. FAMILY_KEY = stratum_id.
Strong은 Hit@10 ranking failure가 candidate-generation failure보다 많고, Hit@10 oracle과 requirement oracle이 모두 올라가며, ranking failure가 두 개 이상의 질의와 두 개 이상의 stratum_id에 있을 때다.
Partial은 Strong이 아니고 oracle 분자가 올라가며 ranking failure 또는 requirement ranking miss가 있을 때다. requirement-only headroom is Partial.
그 밖은 Reject다. 숫자 컷은 점수 이후에 더하지 않았다.

## 4. Evaluation Integrity

dataset `multi_company_retrieval_v2_1` sha256 `2b1c372b7adaca29af831986bf5e6807c2ae2bd7835a52735690e227c1b03137`. 모집단은 DEV fully_answerable 45이다. benchmark는 LLM-judged / LLM-adjudicated benchmark다. human gold가 아니다.
qrel primary key는 (query_id, requirement_id, retrieval_unit_id)다. source element는 provenance다.
재계산한 BM25, Dense, H10 headline 분자는 저장된 DEV headline과 같다. H10 top-10은 C20의 부분집합이다. Oracle Hit@10 gap은 ranking failure 수와 같다.
retrieval 재실행 0. TEST 검색 0. TEST 파일 0. reranker 구현 없음. depth sweep 없음. qrel 수정 없음.

## 5. Candidate Pool

C20 = 저장된 Dense top-20 ∪ 저장된 BM25 top-20. 중복 제거는 retrieval_unit_id다.
저장된 full-ranking RRF의 첫 relevant unit rank는 C20 밖 사례의 참고다. best rank sweep이 아니고 pool을 바꾸지 않는다.

## 6. Current H10 Metrics

| metric | BM25 | H10 |
| --- | --- | --- |
| Hit@10 | 42/45 | 43/45 |
| Recall@10 macro | 0.767398 | 0.787186 |
| Recall@10 micro | 174/231 | 173/231 |
| MRR | 0.741878 | 0.744101 |
| NDCG@10 condensed | 0.708399 | 0.712533 |
| requirement_recall@10 | 51/58 | 52/58 |
| all_requirements_covered@10 | 41/45 | 42/45 |

H10의 DEV 증분은 저장된 headline 기준 Hit@10 1건, requirement 1건이다. 대규모 성능 개선이라고 적지 않는다.

## 7. Candidate Oracle Metrics

| metric | H10 | C20 oracle |
| --- | --- | --- |
| Hit@10 | 43/45 | 44/45 |
| Recall@10 macro | 0.787186 | 0.891138 |
| Recall@10 micro | 173/231 | 200/231 |
| requirement_recall@10 | 52/58 | 54/58 |
| all_requirements_covered@10 | 42/45 | 43/45 |

Requirement oracle이 unit-first prefix보다 많은 requirement를 덮는 질의는 0건이다. cover budget가 availability보다 적게 덮는 질의는 0건이다.
Oracle role은 upper_bound_not_a_system이다.

## 8. Query-level Failure Taxonomy

H10 hit 43/45. ranking failure 1. candidate-generation failure 1.
ranking failure queries: V2-00107613-finance_table-03.
candidate-generation failure queries: V2-00107613-cross_period-02.
ranking failure strata: finance_table.
candidate-generation strata: cross_period.

## 9. Requirement-level Failure Taxonomy

requirement 58. covered 52. ranking missed 2. candidate missing 4.
multi-requirement 질의 10. 그 중 H10 Hit@10 8. Hit@10 ranking failure 1. Hit@10 candidate-generation failure 1.
multi-requirement requirement 23. ranking missed 2. candidate missing 4. H10 all-covered 7/10. oracle all-covered 8/10.
requirement ranking miss가 있는 multi-requirement 질의: V2-00107613-finance_table-03.

## 10. Ranking Failure Cases

### V2-00107613-finance_table-03

이수페타시스 / finance_table / composition_ratio / table / paraphrase / period_sensitive=True
이수페타시스의 2026년 상반기 투자활동에서 무형자산 취득과 투자부동산 취득은 각각 어느 정도를 차지했나요?
BM25 Hit@10 False, Dense Hit@10 False, H10 Hit@10 False, oracle Hit@10 True.
relevant units 1, C20 안 1, H10 top-10 안 0. BM25 best rank None, Dense best rank 20.
저장된 full-RRF 첫 relevant unit은 BM25 rank 82, Dense rank 20, RRF rank 10이다. 이 쌍은 모든 relevant unit의 best rank가 아니다.
놓친 requirement: R1, R2.

relevant units:
- b05a34f024f4 rel 2 req R1, R2 BM25 None Dense 20 C20 True H10 False 반기보고서 2026.06 canonical_text
  - [section] 목     차 > III. 재무에 관한 사항 > 4. 재무제표 > 4-4. 현금흐름표 현금흐름표 제 55 기 반기 2026.01.01 부터 2026.06.30 까지 제 54 기 반기 2025.01.01 부터 2025.06.30 까지 (단위 : 원)  | 제 55 기 반기 | 제 54 기 반기 영업권 이외의 무형자산의 취득 | (24,500,000) | (769,792,470) 투자부동산의 취득 | (7,193,380) | 0 재무활동현금흐름 | 14,652,569,881 | 240,299,630,580 단기차입금의 증가 | 20,000,000,000 | 11,000,000,000

## 11. Candidate Generation Failure Cases

### V2-00107613-cross_period-02

이수페타시스 / cross_period / cross_filing_comparison / multi_element / layperson / period_sensitive=True
이수페타시스의 2024년 반기보고서와 3분기보고서에는 투자조합의 지분 정보와 받을 돈의 연체 현황이 각각 어떻게 나와 있나요?
BM25 Hit@10 False, Dense Hit@10 False, H10 Hit@10 False, oracle Hit@10 False.
relevant units 2, C20 안 0, H10 top-10 안 0. BM25 best rank None, Dense best rank None.
저장된 full-RRF 첫 relevant unit은 BM25 rank 544, Dense rank 80, RRF rank 135이다. 이 쌍은 모든 relevant unit의 best rank가 아니다.
놓친 requirement: R1, R2.

relevant units:
- 298e0ed60bd1 rel 2 req R1 BM25 None Dense None C20 False H10 False 반기보고서 2024.06 canonical_text
  - [section] 목     차 > III. 재무에 관한 사항 > 5. 재무제표 주석 ISU Petasys Asia Ltd.는 전기말 이전에 완전자본잠식으로 인하여 지분법적용이 중단되었습니다.  ③ ISU-세컨더리1호투자조합 (단위: 천원) 구분 | 기말 자기자본(a) | 지분율(b) | 순자산지분금액(a*b) | 장부금액 당반기말 | 679,960 | 49.16% | 334,297 | 334,297 전기말 | 677,527 | 49.16% | 333,101 | 333,101
- 571929a0fa91 rel 1 req R2 BM25 None Dense None C20 False H10 False 분기보고서 2024.09 canonical_text
  - [section] 목     차 > III. 재무에 관한 사항 > 5. 재무제표 주석 ② 전기말 (단위: 천원) 구분 | 연체되지않은 채권 | 연체되었으나손상되지 아니한 채권 | 손상채권 | 합계 7~12개월 | 1년~3년 매출채권 | 123,233,004 | 1,579 | - | - | 123,234,583 단기대여금 | 17,250 | - | - | - | 17,250 미수금 | 2,400,700 | - | - | 141,112 | 2,541,812 미수수익 | 115,272 | - | - | - | 115,272 합계 | 125,766,226 | 1,579 | - | 141,112 | 125,908,917

## 12. Scope / Representation / Benchmark Issues

판정 라벨은 stratum_id 비교만 사용한다. 아래 family 이름은 저장 unit 텍스트를 읽은 뒤의 관찰이다.
`V2-00107613-finance_table-03`은 ranking failure다. 양성 unit 하나는 2026년 반기 별도 현금흐름표의 '영업권 이외의 무형자산의 취득'과 '투자부동산의 취득' 행이고, R1과 R2를 함께 덮는다. Dense rank 20이라 C20 안에 있다. BM25 top-20에는 없다. 저장된 full-RRF 첫 relevant unit이 그 unit이고 BM25 rank는 82다. H10 top-10은 2024–2025년 투자부동산 주석과 담보 주석이다. 같은 단어가 다른 표, 다른 기간에 있다. 현금흐름표 unit은 만들어져 있고 검색 텍스트에 행 이름이 있다. chunking failure로 적지 않는다.
`V2-00107613-cross_period-02`는 candidate-generation failure다. R1 양성 unit은 2024년 반기 주석의 투자조합 지분율 표다. R2 양성 unit은 2024년 3분기 주석의 연체 채권 표다. 둘 다 C20 밖에 있다. 저장된 full-RRF 첫 relevant unit은 R1 unit이고 BM25 rank 544, Dense rank 80이다. R2 unit의 best rank는 저장돼 있지 않다. H10 top-10은 '투자조합의 경우 법인이 아닌 단체'라는 계열회사 제외 문장과, 같은 기간 주석의 표지 문장이다. 두 filing의 표는 corpus에 있다. eligible ranking 안에 깊은 순위가 있을 뿐, 이번 pool에는 없다.
`V2-00126955-cross_period-02`는 H10 Hit@10이 참이다. R2는 BM25 rank 4의 차입금 주석이 덮고, 그 주석은 유동성장기부채 차감 행을 포함한다. R1의 양성 unit은 2026년 1분기 연결 재무상태표의 단기차입금·유동성장기부채 행이고 C20 밖에 있다. R3의 양성 unit은 전기말 대출약정 주석이고 C20 밖에 있다. 이 두 requirement는 C20 재정렬로 복구되지 않는다. 질의 Hit가 requirement 충족을 뜻하지 않는 사례다.
세 질의에서 양성 unit의 canonical text에는 요구한 행이나 표가 있다. qrel을 고칠 근거로 본 불일치는 없다. filing eligibility 밖으로 빠진 것으로 보지도 않는다. full ranking이 그 unit을 순위에 올렸기 때문이다. 관찰된 family는 period, cross-filing, table, lexical이다. finance_table-03의 Dense rank 20은 semantic signal이 pool 경계에 한 건 있다는 관찰이다.

아래는 Hit@10 miss와 requirement miss 질의의 저장 메타다. 주제 라벨을 판정에 넣지 않는다.
chunking failure라고 단정하지 않는다. qrel은 수정하지 않는다.

- `V2-00107613-cross_period-02` candidate_generation_failure. stratum `cross_period`, evidence `multi_element`, period_sensitive True. R1=candidate_missing R2=candidate_missing
- `V2-00107613-finance_table-03` ranking_failure. stratum `finance_table`, evidence `table`, period_sensitive True. R1=ranking_missed R2=ranking_missed
- `V2-00126955-cross_period-02` h10_hit. stratum `cross_period`, evidence `multi_element`, period_sensitive True. R1=candidate_missing R3=candidate_missing

## 13. Oracle Headroom

Hit@10 gap 1. Requirement recall gap 2.
이미 Hit@10인 질의를 포함한 unit recall headroom 질의는 16건이고, 분자 합은 27이다. Recall@10 macro headroom은 0.103952이다.
이 recall headroom은 Hit@10 failure 수가 아니다. 판정 트리에 넣지 않았다.
C20 안에 있으면서 H10 top-10 밖에 있는 relevant unit은 44개다. Dense top-20에만 있는 것은 26개, BM25 top-20에만 있는 것은 7개, 두 리스트에 모두 있는 것은 11개다. Recall@10은 이 가운데 질의당 10개까지만 센다.

## 14. Reranker Opportunity Rate

현재 C20 후보군을 완벽하게 재정렬할 수 있다는 oracle 가정에서 이 정도 failure가 theoretically recoverable하다. 이 비율은 reranker가 그 수만큼 개선한다는 뜻이 아니다.
Query-level: 1/2 (0.500000).
Requirement-level: 2/6 (0.333333).

## 15. FACT

ranking failure 1, candidate-generation failure 1, oracle Hit@10 44/45, H10 Hit@10 43/45, oracle requirement 54/58, H10 requirement 52/58.

## 16. INTERPRETATION

C20 재정렬로 이론상 복구되는 failure가 있다. 다만 Hit@10 failure에서 candidate-generation failure가 같거나 더 크거나, 복구 가능성이 소수 질의 또는 한 stratum에 모이거나, requirement 쪽에서만 headroom이 있다. 한 모델의 예상 개선치로 바꾸지 않는다.

## 17. H11 Verdict

`Partial Support`
reason `candidate_generation_comparable_or_larger`.
Strong when Hit@10 ranking failures strictly outnumber candidate-generation failures, both oracle numerators rise, and those queries cover at least two queries and two stratum_id values. Partial when an oracle numerator rises and a ranking failure or requirement ranking miss exists, including requirement-only headroom. Otherwise Reject.

## 18. Whether Re-ranker Experiment Is Justified

이론적 headroom은 있다. 그 크기가 한 조각이거나 candidate-generation failure와 비슷하면, 모델 비교를 시작할 근거로는 약하다. 실험을 연다면 복구 대상을 이번 DEV에서 보인 ranking failure로 한정하고, C20 밖에 있는 failure를 그 실험의 성공으로 세지 않는다.

## 19. If justified: Next Hypothesis

다음 가설은 넓게 'reranker를 넣으면 좋아진다'가 되면 안 된다. 가설은 DEV에서 확인된 ranking failure 또는 requirement ranking miss만 복구 대상으로 두고, candidate-generation failure는 성공 기준 밖에 둔다. 예상 상한은 이번 C20 oracle이다. 특정 reranker 모델을 이 진단이 선택하지는 않는다.

## 20. Limitations

C20은 저장된 top-20 두 리스트의 합이다. 그 밖의 rank는 후보가 아니다.
저장된 full ranking 전체는 남아 있지 않다. full-RRF 첫 relevant unit의 rank만 참고한다. 그 값은 모든 relevant unit의 best rank가 아니다.
Oracle은 judged qrel 안의 relevance를 안다고 가정한다. unjudged unit을 relevant로 승격하지 않는다.
DEV 45의 판정이다. 같은 숫자로 TEST를 다시 맞추지 않는다.
과거 H10 TEST validation은 Hit@10이 BM25 47/55, H10 48/55이고 requirement가 56/66에서 57/66이었다. 증분은 질의 1건, requirement 1건이다. 그 결과는 역사적 final validation이다. 이번 진단의 입력이 아니다.

## 21. Portfolio-ready Summary

# H11 portfolio summary

DEV 45에서 H10 Hit@10 failure 중 ranking failure는 1건, C20에 relevant unit이 없는 candidate-generation failure는 1건이다.
C20 oracle Hit@10은 44/45이고 H10은 43/45이다. Requirement recall oracle은 54/58이고 H10은 52/58이다.
판정은 `Partial Support` (candidate_generation_comparable_or_larger)이다.
Oracle은 C20을 완벽히 재정렬할 때의 upper bound다. reranker가 그 차이를 달성한다는 뜻이 아니다.
TEST는 이 진단의 development data로 쓰지 않았다.

## 22. pytest / ruff

`uv run pytest -q`는 335 passed, warning 1이다.
warning은 기존 zipfile duplicate name이다.
`uv run ruff check .`는 통과했다.

## 23. artifacts

`evaluation/runs/multi_company_retrieval_v2_1_dev_h11_ranking_failure/`

## 24. commit

진단 코드, semantics, DEV artifact, worklog를 커밋한다. TEST 파일은 읽지 않았다.


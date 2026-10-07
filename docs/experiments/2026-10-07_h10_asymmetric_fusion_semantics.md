# H10 asymmetric fusion semantics

상태: 실험 점수를 내기 전에 고정한다. 이 문서의 rule과 상수를 결과를 본 뒤에 바꾸지 않는다. 다른 asymmetric rule은 이 실험에서 구현하지 않는다.

## 검증 질문

현재 v2.1 DEV처럼 BM25가 dominant retriever인 조건에서, Dense와 BM25를 대칭적으로 경쟁시키는 fusion보다, BM25 ranking을 primary로 보호하면서 Dense의 high-confidence semantic signal만 제한적으로 보조하는 asymmetric fusion이 BM25 lexical coverage를 유지하면서 Dense-only semantic recovery를 보존할 수 있는가?

이 질문이 묻는 것은 BM25보다 높은 headline 숫자가 아니다. 묻는 것은 BM25-primary asymmetric semantics가 H8/H9에서 관측된 symmetric-fusion failure를 줄이는가이다.

## Rule

`bm25_primary_dense_head`

한 정책만 둔다. weight, k, N, alpha를 훑지 않는다.

## 왜 이 rule인가

H9의 잔여 3건은 같은 mechanism이 아니다.

- 대상 지분증권: symmetric rank competition. Dense-only 등위가 BM25-only 양성과 단일 리스트 점수로 경쟁한다.
- 대상 특수관계자: intersection dominance와 symmetric competition이 함께 있다.
- 알테오젠 우선주: intersection dominance. 교집합 가산이 BM25 순서를 건너뛴다.

H9는 한 weight가 이 셋을 동시에 지운다고 점수화하지 않았다. Dense rank 1을 넘는 weight ratio와 최악 intersection을 넘는 weight ratio는 크기가 다르다. 그래서 H10은 weight를 고르지 않는다.

대칭 truncated RRF(N=20, k=60, 1:1)에서는 최악 교집합 점수 `1/80 + 1/80 = 0.025`가 최선 단일 리스트 점수 `1/61 = 0.016393`보다 크다. 교집합 unit은 단일 리스트 unit보다 항상 앞선다. 이 가산과 단일 리스트끼리의 등위 경쟁을 없애면 세 mechanism의 공통 통로가 닫힌다. 공통 통로는 “두 리스트의 등위를 같은 종류의 점수로 다시 매긴다”는 것이다.

Dense의 보조는 그 통로를 한 칸만 연다. high-confidence는 score threshold가 아니다. 저장된 Dense 리스트가 threshold 없이 주는 최상위 신호는 Dense rank 1 하나다. rank 2 이하는 정렬 입력이 아니다.

삽입 칸도 sweep하지 않는다. BM25 rank 1..9는 그대로 둔다. Dense head가 BM25 top-10에 없을 때만 fused rank 10을 그 head로 바꾼다. 이 칸은 기존 Hit@10 컷오프 안에 들어가면서 BM25의 비마진 prefix를 지나치지 않는 유일한 자리다. 더 앞이면 BM25 순서를 바꾸고, 더 뒤면 알려진 Dense-only 사례가 Hit@10에 남지 않는다. `MARGINAL_SLOT = 10`은 이 벤치마크에 이미 있는 Hit@10이지, H10에서 맞춘 hyperparameter가 아니다. 5, 8, 9, 15는 시도하지 않는다.

## Deterministic semantics

입력은 저장된 Dense top-20과 BM25 top-20이다. `h`는 Dense rank 1의 `retrieval_unit_id`이다. `B[1..20]`은 BM25 순서이다.

1. `h`가 `B[1..10]`에 있으면 fused list는 `B`이다. 어떤 unit의 순서도 바뀌지 않는다. edit는 `none`이다.
2. `h`가 `B[1..10]`에 없으면 edit가 발생한다. `h`가 `B[11..20]`에 있으면 `promotion`, `B`에 없으면 `insertion`이다.
3. 그 경우 fused list는 `B[1..9] + [h] + [u in B | u not in B[1..9] and u != h]`이다.
4. `insertion`이면 길이는 21이다. `B[20]`은 rank 21로 남고, MRR cutoff 20과 Hit@10 밖이다. BM25 unit을 삭제하지 않는다.
5. `promotion`이면 길이는 20이다. `h`보다 뒤의 BM25 unit은 상대 순서를 유지한다. `B[10]`부터 `h` 직전까지는 한 칸 뒤로 밀린다.
6. Dense rank 2..20은 후보로 추가되지 않는다. 그 unit이 이미 BM25 리스트에 있으면 BM25 자리에 남는다.
7. 교집합 membership은 가산점이 아니다. 교집합 unit은 BM25 자리를 유지하고, Dense head인 경우에만 2번의 예외를 받는다.
8. 동점 점수는 없다. 이 rule은 점수를 만들지 않는다. 순서는 위의 배치가 전부다.
9. 입력 top-20의 rank는 1부터 연속이어야 하고 `retrieval_unit_id`는 리스트 안에서 유일해야 한다. 아니면 점수를 내지 않는다.
10. 같은 unit의 source element id가 두 리스트에 있고 서로 다르면 점수를 내지 않는다.

입력 검증은 top-20의 연속성과 중복을 확인한다. 정렬 함수에 넘어가는 Dense 정보는 rank 1의 id뿐이다.

## BM25 unit ordering이 바뀌는 조건

바뀌는 조건은 하나다. Dense rank 1이 BM25 top-10 밖에 있는 것이다.

그때 바뀌는 범위도 고정이다.

- `B[1..9]`의 unit과 상대 순서는 모든 질의에서 그대로다.
- rank 10 이후만 바뀐다.
- BM25 top-10 안에 있던 unit 가운데 자리를 옮기는 것은 `B[10]`뿐이다. 그 unit은 rank 11로 밀리거나, 그 unit 자신이 Dense head라서 rank 10에 남는 promotion의 경계에 있다. `B[10]`이 Dense head이면 `h`는 top-10 안이므로 edit는 `none`이고 자리는 그대로다.
- `B[11..20]`은 Dense head보다 앞선 unit이 한 칸 밀릴 수 있다. 그 외의 상대 순서는 BM25 그대로다.

`h`가 BM25 top-10 안에 있으면 BM25 ordering은 한 unit도 바뀌지 않는다.

## 이 rule에서 움직일 수 있는 지표

실행 전에 구조를 고정한다. 아래는 구현이 맞을 때의 귀결이고, 효과 추정치가 아니다.

- fused rank 1..9는 BM25 rank 1..9와 같다. Hit@1, Hit@5, Recall@5 macro는 BM25와 같다.
- MRR은 첫 양성 rank가 10 이상이거나, 삽입된 head가 새 첫 양성이 될 때만 변한다. 첫 양성이 10에서 11로 밀리면 그 질의의 MRR 변화는 `1/10 - 1/11`이다.
- Hit@10, Recall@10, NDCG@10, requirement@10은 marginal slot에서만 변한다.
- NDCG@10의 DCG 차이는 position 10 한 칸이다. discount는 `1/log2(11)`이다.

따라서 Strong Support의 Recall@5와 MRR 하한은 손상이 없는지를 확인하는 잠금이다. 이 rule의 경험적 내용은 marginal slot 교체가 Hit@10, Recall@10, NDCG@10, requirement coverage, judgment에 남기는 순효과다.

## 실행 전 예측의 논리 지위

H9에 적힌 BM25 rank는 대상 지분증권 8, 대상 특수관계자 7, 알테오젠 우선주 4다. 셋 다 protected prefix 안이다. 구현이 이 문서를 따르면 그 세 질의의 첫 양성 rank는 BM25 rank와 같다. 그 복원을 독립적인 treatment effect로 읽지 않는다.

알려진 Dense-only 사례 `V2-00126380-finance_table-02`는 Dense rank 1이고 BM25 top-20 안에 양성이 없다. 구현이 이 문서를 따르면 그 unit은 fused rank 10이고 Hit@10에 남는다. truncated RRF의 rank 2보다 뒤다. 성공 기준은 Hit@10이다. rank 2 유지는 기준이 아니다.

경험적으로 열린 부분은 Dense head가 BM25 top-10 밖에 있는 질의다. 그 교체가 BM25 rank 10의 양성, requirement, unjudged head를 어떻게 바꾸는지가 판정의 내용이다.

## 사전 상수

```
RULE_ID = bm25_primary_dense_head
CANDIDATE_N = 20
PROTECTED_PREFIX = 9
MARGINAL_SLOT = 10
STRONG_HIT10_FLOOR = 41
STRONG_BM25_ONLY_LOST_MAX = 1
RANKING_DROP_LIMIT = 0.05
STRONG_REQUIREMENT_RECALL_FLOOR = 49
STRONG_ALL_REQUIREMENTS_FLOOR = 39
TRUNCATED_HIT10 = 40
TRUNCATED_BM25_ONLY_LOSSES = 3
```

`RANKING_DROP_LIMIT = 0.05`는 H8이 둔 하락 경계다. H10 점수에 맞춰 다시 고르지 않는다. requirement 하한 49와 39는 BM25의 51/58, 41/45에서 2 이내다. “근처”의 정의는 이 2이다.

BM25-only regression은 prior cell `dense_miss_bm25_hit`에서 fused Hit@10을 잃는 질의 수다. both-hit 상실은 그 칸에 넣지 않고 따로 센다. both-hit 상실은 Hit@10 하한을 통해서만 판정에 들어간다.

비교 population은 DEV fully_answerable 45다. qrel과 metric 정의는 바꾸지 않는다. unjudged는 irrelevant가 아니다. NDCG headline은 condensed다. zero-filled NDCG는 sensitivity다.

## 판정 트리

Strong Support는 아래가 모두 참일 때다.

- fused Hit@10 numerator가 `STRONG_HIT10_FLOOR` 이상이다.
- BM25-only Hit@10 상실이 `STRONG_BM25_ONLY_LOST_MAX` 이하이다.
- `V2-00126380-finance_table-02`의 fused Hit@10이 참이다.
- Recall@5 macro, Recall@10 macro, MRR, NDCG@10 condensed의 BM25 대비 하락이 `RANKING_DROP_LIMIT` 이하이다. 하락은 BM25 값에서 fused 값을 뺀 것이다. 개선은 음수이며 하한을 통과한다.
- requirement_recall@10 numerator가 `STRONG_REQUIREMENT_RECALL_FLOOR` 이상이다.
- all_requirements_covered@10 numerator가 `STRONG_ALL_REQUIREMENTS_FLOOR` 이상이다.

Partial Support는 Strong이 아니고 아래가 모두 참일 때다.

- Dense-only Hit@10이 유지된다.
- fused Hit@10 numerator가 `TRUNCATED_HIT10`보다 크다.
- BM25-only Hit@10 상실이 `TRUNCATED_BM25_ONLY_LOSSES`보다 적다.

Partial의 reason은 BM25-only 상실이 1 이상이면 `lexical_regression_remains`, 0이면 `strong_floor_missed`이다.

그 밖은 Reject다. Dense-only Hit@10을 잃으면 reason은 `dense_only_lost`이다. 그 외 Reject reason은 `not_clearly_better_than_truncated`이다.

“명확히 개선”은 Hit@10이 저장된 truncated RRF 40보다 크고, BM25-only 상실이 truncated의 잔여 3보다 적은 것이다. 이 두 수를 점수 이후에 바꾸지 않는다.

Strong이어도 TEST를 열지 않는다. Reject여도 TEST를 열지 않는다. Reject이면 BM25가 strongest baseline으로 남고, 이 fusion 실험을 여기서 끝낼 수 있다.

## 대표 사례 선택

점수를 본 뒤 고르지 않는다. 보고에 항상 넣는 질의는 다음이다.

- `V2-00126380-finance_table-02`
- `V2-00121941-business_paragraph-03`
- `V2-00121941-finance_table-02`
- `V2-00989619-governance_paragraph-02`
- fused Hit@10이 BM25 Hit@10과 다른 모든 질의
- edit가 `none`이 아니고, 밀려난 BM25 rank 10 unit의 판정이 relevant인 모든 질의

## 금지

weight sweep, k sweep, N sweep, alpha sweep, 여러 asymmetric rule의 동시 비교, query routing, reranker, query expansion, embedding 변경, chunking 변경, TEST 검색, evaluation definition 변경.

## 산출 시 읽을 순서

FACT는 배치, 편집 수, 지표, 전이, coverage다. 세 residual의 복원은 FACT로 적되 귀결이라고 적는다. INTERPRETATION은 그 순효과가 사전 기준의 어느 칸에 들어가는지다. 판정 라벨은 위의 트리만 사용한다.

# v2.1 DEV H9 rank-pressure diagnosis

Candidate-truncated RRF의 기존 순위를 분해한다. 새 랭커는 없다. TEST 검색은 0이다.

## DEFINITIONS

후보 집합은 Dense top-20과 BM25 top-20의 union이다. score = I_dense/(60+dense_rank) + I_bm25/(60+bm25_rank). 교집합 최소 점수는 0.025000이고 단일 리스트 최대 점수는 0.016393이다.
intersection은 양쪽 top-20에 있는 unit, dense_only는 Dense top-20에만 있는 unit, bm25_only는 BM25 top-20에만 있는 unit이다.
primary positive는 BM25 top-20 안에서 판정 relevance가 1 이상인 첫 unit이다. unjudged는 relevance 0이 아니다.
BM25 predecessor는 BM25 rank가 positive보다 앞인 unit이다. queue jump는 positive보다 BM25 rank가 나쁜데 truncated 순서에서는 앞에 있는 intersection unit이다. symmetric competition pressure는 positive 앞에 있는 Dense-only unit 수다.
counterfactual A는 positive 앞의 intersection unit을 순서에서 지운 자리다. BM25가 원래 앞에 둔 intersection predecessor도 지워진다. queue jump만 지운 자리는 따로 적는다. B는 Dense-only ahead 수다. C는 BM25 원래 rank다. D의 최소 displacement는 BM25 rank를 11위로 만드는 추가 unit 수인 `11 - BM25 rank`다. Top-10으로 되돌리는 최소 제거 수는 `truncated rank - 10`이다. 이 자리는 새 랭킹이 아니다.

## FACT

### Structural property

세 residual의 deep-tail rank 사용은 0이고 점수에 들어간 최대 rank는 20이다.

- 대상 지분증권: holds=True, intersection 0, worst intersection score , best single-list score 0.016393, vacuous_intersection=True.
- 대상 특수관계자: holds=True, intersection 8, worst intersection score 0.026743, best single-list score 0.016393, vacuous_intersection=False.
- 알테오젠 우선주: holds=True, intersection 8, worst intersection score 0.027588, best single-list score 0.016129, vacuous_intersection=False.

### Residual decompositions

| query | BM25 | truncated | ahead I | ahead D | ahead B | queue jump | dense displacement | A | A queue-only | dense-removed | min exit | removals to restore |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 대상 지분증권 | 8 | 15 | 0 | 7 | 7 | 0 | 7 | 15 | 15 | 8 | 3 | 5 |
| 대상 특수관계자 | 7 | 14 | 8 | 3 | 2 | 4 | 3 | 6 | 10 | 11 | 4 | 4 |
| 알테오젠 우선주 | 4 | 11 | 8 | 0 | 2 | 7 | 0 | 3 | 4 | 11 | 7 | 1 |

#### 대상 지분증권

`V2-00121941-business_paragraph-03` 대상: 대상이 종속·관계·공동기업을 제외하고 보유한 지분증권 투자액은 2024년 말 장부상 얼마였나요?

primary `6c3f2984ba262a6a78e6d1db1be108f7602d0999cdfad8177de0d12b9d9c840a` category bm25_only. Dense contribution 0.000000, BM25 contribution 0.014706, fused score 0.014706. mechanism `symmetric_rank_competition`.

predecessor 7 `{"intersection": 0, "dense_only": 0, "bm25_only": 7}`. ahead judgments `{"intersection": {"relevant": 0, "irrelevant": 0, "unjudged": 0}, "dense_only": {"relevant": 0, "irrelevant": 7, "unjudged": 0}, "bm25_only": {"relevant": 0, "irrelevant": 7, "unjudged": 0}}`. displacement judgments `{"intersection": {"relevant": 0, "irrelevant": 0, "unjudged": 0}, "dense_only": {"relevant": 0, "irrelevant": 7, "unjudged": 0}, "bm25_only": {"relevant": 0, "irrelevant": 0, "unjudged": 0}}`.

removal restores Hit@10: `{"intersection": false, "queue_jumps": false, "dense_only": true, "bm25_only": true}`. displacement category can supply the exit alone: `{"intersection": false, "dense_only": true, "bm25_only": false}`.

Displacement units:

- rank 2 `9ae4ddd7f9cd06b41a80e95c484f2efbbda879801bbeb18b7d49db7085c31b13` dense_only dense 1 bm25 None relevance 0 requirements -
- rank 4 `1f60c34f226cd8610f37c62a93c9484cb022ba42d2fc48559caed195ef781527` dense_only dense 2 bm25 None relevance 0 requirements -
- rank 6 `f20ac7a0d4a8c420159db97df8bbfa409f8281c4c99e7ff17aa57f46dd6d7933` dense_only dense 3 bm25 None relevance 0 requirements -
- rank 7 `b4b2c0034056b806f4dcea48d1793f658a1f05c750fa4146d78cbf40b3e2cbbc` dense_only dense 4 bm25 None relevance 0 requirements -
- rank 10 `d6465bb272972a426f872612f9e74b1390ca235a9b0da2bb850a67ef87ec9323` dense_only dense 5 bm25 None relevance 0 requirements -
- rank 11 `9d54ae07a9a8f4efaae340c9d2d30ca167845c5b0130bafa7eed638146e911bc` dense_only dense 6 bm25 None relevance 0 requirements -
- rank 14 `d5555de61f4a36fc3ad5147933d14b6507aabbbd323f341aaca210bc6ea28cfb` dense_only dense 7 bm25 None relevance 0 requirements -

BM25 relevant units in the union: 3.

- BM25 8 → truncated 15 `6c3f2984ba262a6a78e6d1db1be108f7602d0999cdfad8177de0d12b9d9c840a` bm25_only requirements R1 ahead `{"intersection": 0, "dense_only": 7, "bm25_only": 7}`.
- BM25 9 → truncated 18 `cc42b6ee74bc79a5f0bf1b8513eb58df5bf027acdd19ebbb4f7d80d5da5345b6` bm25_only requirements R1 ahead `{"intersection": 0, "dense_only": 9, "bm25_only": 8}`.
- BM25 10 → truncated 19 `2fdf19c16308af2733dc9134726f13f1c7d069218487841a915effc8343f140a` bm25_only requirements R1 ahead `{"intersection": 0, "dense_only": 9, "bm25_only": 9}`.

#### 대상 특수관계자

`V2-00121941-finance_table-02` 대상: 대상은 2026년 1분기 말 특수관계자 거래로 받을 돈과 갚을 돈이 각각 얼마였어?

primary `62c8564da1be79a9d8b5bad6f32b3786894d1015029ad67477f292ac03b00a86` category bm25_only. Dense contribution 0.000000, BM25 contribution 0.014925, fused score 0.014925. mechanism `intersection_dominance_and_symmetric_competition`.

predecessor 6 `{"intersection": 4, "dense_only": 0, "bm25_only": 2}`. ahead judgments `{"intersection": {"relevant": 0, "irrelevant": 7, "unjudged": 1}, "dense_only": {"relevant": 0, "irrelevant": 3, "unjudged": 0}, "bm25_only": {"relevant": 0, "irrelevant": 2, "unjudged": 0}}`. displacement judgments `{"intersection": {"relevant": 0, "irrelevant": 3, "unjudged": 1}, "dense_only": {"relevant": 0, "irrelevant": 3, "unjudged": 0}, "bm25_only": {"relevant": 0, "irrelevant": 0, "unjudged": 0}}`.

removal restores Hit@10: `{"intersection": true, "queue_jumps": true, "dense_only": false, "bm25_only": false}`. displacement category can supply the exit alone: `{"intersection": true, "dense_only": false, "bm25_only": false}`.

Displacement units:

- rank 2 `bab2a7632a01b7fffd03e46f82382b3c0f0d5cfb5d98bab0abd2d122cc0b3b75` intersection dense 2 bm25 8 relevance 0 requirements -
- rank 3 `a90898b290d2038bf39dbb9fef42ae7aa35edeae6699051ba018616098620a2a` intersection dense 1 bm25 13 relevance 0 requirements -
- rank 6 `da623d2e6d34296bf75d50bcff2de78519ce5bcfb38aa78862b7b8ba05b5ff20` intersection dense 3 bm25 20 relevance 0 requirements -
- rank 8 `f5bc08e94a198d3d335a203bd51db2f0b13d2e49dd247db3eaa22b3e8e56dde9` intersection dense 19 bm25 11 relevance None requirements -
- rank 10 `67e2831447ff4e9a3845cdd444e4cbc836b8ba93e1e601eceb8a7034fe87cd62` dense_only dense 4 bm25 None relevance 0 requirements -
- rank 11 `113f34e64c281db3a245b96adf54cb82f2b40b75eda20857235392b3deac04d9` dense_only dense 5 bm25 None relevance 0 requirements -
- rank 13 `285f3a30a7d982ee39a6ba61f00e7e67fff4440cbda9503f53c769b25cac0e89` dense_only dense 7 bm25 None relevance 0 requirements -

BM25 relevant units in the union: 1.

- BM25 7 → truncated 14 `62c8564da1be79a9d8b5bad6f32b3786894d1015029ad67477f292ac03b00a86` bm25_only requirements R1 ahead `{"intersection": 8, "dense_only": 3, "bm25_only": 2}`.

#### 알테오젠 우선주

`V2-00989619-governance_paragraph-02` 알테오젠: 알테오젠은 이번 분기에 우선주 몇 주를 보통주로 바꿨나요?

primary `ea7512ea104a8b27f90c11ac016a9e18e87550993881959f7426797781fc075f` category bm25_only. Dense contribution 0.000000, BM25 contribution 0.015625, fused score 0.015625. mechanism `intersection_dominance`.

predecessor 3 `{"intersection": 1, "dense_only": 0, "bm25_only": 2}`. ahead judgments `{"intersection": {"relevant": 0, "irrelevant": 7, "unjudged": 1}, "dense_only": {"relevant": 0, "irrelevant": 0, "unjudged": 0}, "bm25_only": {"relevant": 0, "irrelevant": 2, "unjudged": 0}}`. displacement judgments `{"intersection": {"relevant": 0, "irrelevant": 6, "unjudged": 1}, "dense_only": {"relevant": 0, "irrelevant": 0, "unjudged": 0}, "bm25_only": {"relevant": 0, "irrelevant": 0, "unjudged": 0}}`.

removal restores Hit@10: `{"intersection": true, "queue_jumps": true, "dense_only": false, "bm25_only": true}`. displacement category can supply the exit alone: `{"intersection": true, "dense_only": false, "bm25_only": false}`.

Displacement units:

- rank 1 `fe34f16c22a1f6f7f87190e5149a4d2094a3709bdd3bc54d3a7e23ac75623e9e` intersection dense 2 bm25 6 relevance 0 requirements -
- rank 3 `fabd7f06134a801dd16e219ca0998b9102eafe61013fb5ee6ca9fcdc01b5bc21` intersection dense 1 bm25 11 relevance 0 requirements -
- rank 4 `f6f97e0b8376f423c6b5d5f5529e218629054d10666aa284cab8cec569482f20` intersection dense 3 bm25 12 relevance 0 requirements -
- rank 5 `7c04651449c88759ade20cf68325c79ee9abc28f58eae6e5d9d8574bc1a8bc9a` intersection dense 4 bm25 14 relevance 0 requirements -
- rank 6 `261caa209e5993054c7b832aacdb38fae971166e6078816974abe67432fa42fa` intersection dense 5 bm25 15 relevance 0 requirements -
- rank 7 `7acb8b6c4d17504c52ecec748a944da65c0c710fa44c075c2bf200c331e0022b` intersection dense 6 bm25 16 relevance 0 requirements -
- rank 8 `98f728dce06c85349ac4d82141cd5e38689448e03f989881305090159580d144` intersection dense 12 bm25 13 relevance None requirements -

BM25 relevant units in the union: 2.

- BM25 4 → truncated 11 `ea7512ea104a8b27f90c11ac016a9e18e87550993881959f7426797781fc075f` bm25_only requirements R1 ahead `{"intersection": 8, "dense_only": 0, "bm25_only": 2}`.
- BM25 9 → truncated 17 `59bf1caeda932865f4ecd39c6dd7237fdd9aa475d85499d2ad4b2b1005912c4d` bm25_only requirements R1 ahead `{"intersection": 8, "dense_only": 2, "bm25_only": 6}`.


### DEV aggregate

질의 45. 교집합 prefix 성립 45/45. BM25 positive가 top-20 안에 있는 pair 42. same-unit rank change 평균 1.381. 앞당겨짐 5, 동일 19, 밀림 18. truncated 첫 양성과 BM25 positive unit이 다른 질의 8. movement 라벨은 첫 양성 rank의 비교이고, same-unit change는 BM25 positive unit의 자리다. 두 unit이 다르면 라벨과 same-unit change가 어긋난다. mean dense_only ahead는 intersection positive도 포함한다. 그 positive는 prefix 안에 있어 Dense-only ahead가 0인 경우가 많다.

Intersection count와 same-unit rank change. 칸의 n이 5 미만이면 small_n이다.

| intersection | n | mean rank change | median | Hit@10 losses | small_n |
| --- | --- | --- | --- | --- | --- |
| 0 | 7 | 2.286 | 1.000 | 1 | false |
| 1 | 3 | 0.667 | 0.000 | 0 | true |
| 2 | 1 | -2.000 | -2.000 | 0 | true |
| 3 | 3 | 1.333 | 1.000 | 0 | true |
| 4 | 2 | -1.500 | -1.500 | 0 | true |
| 5 | 4 | 0.000 | 0.000 | 0 | true |
| 6 | 6 | 2.000 | 0.500 | 0 | false |
| 7 | 3 | 4.000 | 4.000 | 0 | true |
| 8 | 3 | 5.000 | 7.000 | 2 | true |
| 9 | 2 | 0.000 | 0.000 | 0 | true |
| 10 | 3 | -0.667 | 0.000 | 0 | true |
| 11 | 1 | 3.000 | 3.000 | 0 | true |
| 12 | 1 | 0.000 | 0.000 | 0 | true |
| 13 | 1 | 2.000 | 2.000 | 0 | true |
| 15 | 1 | 0.000 | 0.000 | 0 | true |
| 16 | 1 | -1.000 | -1.000 | 0 | true |

| movement | n | mean intersection | median intersection | mean same-unit change | mean dense_only ahead | small_n |
| --- | --- | --- | --- | --- | --- | --- |
| improvement | 5 | 7.600 | 6.000 | -1.800 | 0.000 | false |
| neutral | 27 | 5.889 | 6.000 | 0.926 | 0.000 | false |
| regression | 10 | 4.100 | 3.500 | 4.200 | 2.000 | false |
| recovered | 1 | 1.000 | 1.000 |  |  | true |
| still_missing | 2 | 3.000 | 3.000 |  |  | true |

| BM25 rank band | n | mean BM25 rank | mean truncated rank | mean change | small_n |
| --- | --- | --- | --- | --- | --- |
| 1 | 29 | 1.000 | 2.207 | 1.207 | false |
| 2-5 | 11 | 3.000 | 3.818 | 0.818 | false |
| 6-10 | 2 | 7.500 | 14.500 | 7.000 | true |

BM25-only identity regression 11건. queue jump 7, Dense-only displacement 6, 둘 다 2.

기술 상관계수. 이 계수는 이 DEV 조각의 기술 통계다. 표본이 작고 질의가 한 벤치마크에 묶여 있다. 인과 추정치로 읽지 않는다.

| x | y | n | spearman | population |
| --- | --- | --- | --- | --- |
| intersection_count | same_unit_rank_change | 42 | -0.058 | queries whose BM25 positive is inside the stored top-20 |
| dense_only_count | same_unit_rank_change | 42 | 0.058 | queries whose BM25 positive is inside the stored top-20 |
| dense_only_ahead | same_unit_rank_change | 14 | 0.248 | BM25-only positives whose displacement identity holds |
| bm25_positive_rank | bm25_positive_truncated_rank | 42 | 0.404 | queries whose BM25 positive is inside the stored top-20 |


## INTERPRETATION

판정은 Partial Support이다. 세 질의 positive Dense contribution은 0이고 사용된 rank는 20 이하다. 대상 지분증권의 mechanism은 symmetric_rank_competition이다. queue jump 0, Dense-only displacement 7. queue jump를 빼면 15위, Dense-only를 빼면 8위다. 대상 특수관계자의 mechanism은 intersection_dominance_and_symmetric_competition이다. queue jump 4, Dense-only displacement 3. queue jump를 빼면 10위, Dense-only를 빼면 11위다. 알테오젠 우선주의 mechanism은 intersection_dominance이다. queue jump 7, Dense-only displacement 0. queue jump를 빼면 4위, Dense-only를 빼면 11위다. 세 사례의 mechanism 라벨은 서로 다르다. DEV 계수 표는 그 구성의 분포다. 그 분포를 세 건의 원인으로 읽지 않는다. 관측이 이름을 준 다음 후보는 BM25-primary asymmetric fusion, weighted RRF이다. 후보는 구현하지 않았고 weight도 점수화하지 않았다. lexical confidence 점수는 이 산출물에 없다.

## Answers

### A. positive 앞을 무엇이 차지했는가?

FACT. 대상 지분증권(`V2-00121941-business_paragraph-03`) positive 앞에는 intersection 0개, dense_only 7개, bm25_only 7개가 있다. 그 앞 판정은 relevant 0, irrelevant 14, unjudged 0이다. BM25 양성 unit의 자리는 8위→15위, 9위→18위, 10위→19위이다. 대상 특수관계자(`V2-00121941-finance_table-02`) positive 앞에는 intersection 8개, dense_only 3개, bm25_only 2개가 있다. 그 앞 판정은 relevant 0, irrelevant 12, unjudged 1이다. 알테오젠 우선주(`V2-00989619-governance_paragraph-02`) positive 앞에는 intersection 8개, dense_only 0개, bm25_only 2개가 있다. 그 앞 판정은 relevant 0, irrelevant 9, unjudged 1이다. BM25 양성 unit의 자리는 4위→11위, 9위→17위이다.

### B. intersection dominance의 rank pressure는 얼마인가?

FACT. 대상 지분증권의 positional intersection pressure는 0칸이고, BM25 순서 대비 추가로 앞으로 온 queue jump는 0개다. intersection을 앞에서 뺀 자리는 15위, queue jump만 뺀 자리는 15위다. A가 함께 빼는 BM25 predecessor intersection은 0개다. Top-10을 벗어나는 최소 displacement는 3이고 관측 displacement는 7이다. 대상 특수관계자의 positional intersection pressure는 8칸이고, BM25 순서 대비 추가로 앞으로 온 queue jump는 4개다. intersection을 앞에서 뺀 자리는 6위, queue jump만 뺀 자리는 10위다. A가 함께 빼는 BM25 predecessor intersection은 4개다. Top-10을 벗어나는 최소 displacement는 4이고 관측 displacement는 7이다. 알테오젠 우선주의 positional intersection pressure는 8칸이고, BM25 순서 대비 추가로 앞으로 온 queue jump는 7개다. intersection을 앞에서 뺀 자리는 3위, queue jump만 뺀 자리는 4위다. A가 함께 빼는 BM25 predecessor intersection은 1개다. Top-10을 벗어나는 최소 displacement는 7이고 관측 displacement는 7이다.

### C. Dense-only competition은 얼마를 기여했는가?

FACT. 대상 지분증권의 Dense-only ahead는 7개이고 그 중 displacement는 7개다. Dense-only를 빼면 자리는 8위가 되고 Hit@10 복원 여부는 True이다. BM25 원래 자리는 8위다. 대상 특수관계자의 Dense-only ahead는 3개이고 그 중 displacement는 3개다. Dense-only를 빼면 자리는 11위가 되고 Hit@10 복원 여부는 False이다. BM25 원래 자리는 7위다. 알테오젠 우선주의 Dense-only ahead는 0개이고 그 중 displacement는 0개다. Dense-only를 빼면 자리는 11위가 되고 Hit@10 복원 여부는 False이다. BM25 원래 자리는 4위다.

### D. 세 사례의 failure mechanism이 동일한가?

FACT. 대상 지분증권=symmetric_rank_competition, 대상 특수관계자=intersection_dominance_and_symmetric_competition, 알테오젠 우선주=intersection_dominance. 세 mechanism은 같지 않다.

### E. DEV 전체에서도 같은 pattern이 관찰되는가?

FACT. 교집합 prefix 성질은 45/45에서 성립한다. BM25-only identity가 성립하는 same-unit regression 11건 중 queue jump가 있는 질의는 7건, Dense-only displacement가 있는 질의는 6건, 둘 다 있는 질의는 2건이다. Spearman은 intersection_count 대 same_unit_rank_change -0.058 (n=42), dense_only_count 대 same_unit_rank_change 0.058 (n=42), dense_only_ahead 대 same_unit_rank_change 0.248 (n=14), bm25_positive_rank 대 bm25_positive_truncated_rank 0.404 (n=42)이다. 이 계수는 이 DEV 조각의 기술 통계다. 표본이 작고 질의가 한 벤치마크에 묶여 있다. 인과 추정치로 읽지 않는다.

### F. H9 판정은 무엇인가?

VERDICT. H9 판정은 Partial Support이다. 세 mechanism은 symmetric_rank_competition, intersection_dominance_and_symmetric_competition, intersection_dominance이다.

### G. 이 결과만으로 asymmetric fusion을 정당화할 수 있는가?

INTERPRETATION. 이 진단은 asymmetric fusion의 순위를 만들지 않았다. 관측된 pressure만으로 그 규칙을 채택할 근거는 되지 않는다. Dense rank 1을 넘는 weight ratio와 최악 intersection을 넘는 weight ratio는 크기가 다르다. 한 weight가 세 pressure를 함께 지우는지 이 진단은 점수화하지 않았다. `V2-00121941-business_paragraph-03` BM25 8위에서 Dense rank 1을 넘기려면 weight ratio 1.115, 최악 intersection을 넘기려면 5.667이다. 이 비율은 식의 경계이고 점수화하지 않았다. `V2-00121941-finance_table-02` BM25 7위에서 Dense rank 1을 넘기려면 weight ratio 1.098, 최악 intersection을 넘기려면 5.154이다. 이 비율은 식의 경계이고 점수화하지 않았다. `V2-00989619-governance_paragraph-02` BM25 4위에서 Dense rank 1을 넘기려면 weight ratio 1.049, 최악 intersection을 넘기려면 4.000이다. 이 비율은 식의 경계이고 점수화하지 않았다.

### H. 다음 controlled experiment를 시작할 근거가 있는가?

INTERPRETATION. Top-20 안의 등위 경쟁이 세 Hit@10 회귀의 displacement를 설명하므로, 기전 하나를 미리 못 박은 DEV 실험을 설계할 근거는 있다. 세 사례의 mechanism은 symmetric_rank_competition, intersection_dominance_and_symmetric_competition, intersection_dominance이다. 하나의 weight sweep이 세 건을 동시에 설명한다는 실험은 이 분해의 다음 단계가 아니다. 이번 작업은 그 실험을 구현하지 않는다.

## Verdict

Partial Support. Strong Support when deep-tail contribution is 0, the intersection prefix property holds, the BM25-only displacement identity holds, and every residual case is intersection queue jumps plus Dense-only competition. Partial Support when those audits hold and every case is one or both of those mechanisms, but the three cases are not the same pair. Reject when a deep-tail rank is used, the identity fails, or a displacement is unexplained.

## Not done

weighted RRF, weight sweep, N sweep, k sweep, score fusion, alpha tuning, query routing, reranker, query expansion, embedding 변경, chunking 변경, period-aware ranking은 구현하지 않았다. TEST는 실행하지 않았다. counterfactual 자리는 진단 수치이고 시스템 점수가 아니다.

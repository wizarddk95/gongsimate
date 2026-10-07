# H11 ranking-failure semantics

상태: DEV 진단을 돌리기 전에 고정한다. 이 문서의 pool, oracle, 판정 트리를 결과를 본 뒤에 바꾸지 않는다. 숫자 threshold는 두지 않는다. 결과를 본 뒤 “몇 퍼센트부터 Strong” 같은 컷을 추가하지 않는다.

## 검증 질문

현재 GongsiMate에 남아 있는 retrieval failure 중 유의미한 부분이 candidate generation failure가 아니라 ranking failure인가?

이 질문이 묻는 것은 Re-ranker를 넣으면 좋아질지 느낌으로 답하는 것이 아니다. 묻는 것은 현재 저장된 결과에서, Re-ranker가 원리적으로 개입할 수 있는 ranking failure가 얼마나 존재하는가이다.

oracle is not a reranker. Oracle 숫자는 시스템 점수가 아니다.

## 모집단

dataset `multi_company_retrieval_v2_1`.

benchmark sha256 `2b1c372b7adaca29af831986bf5e6807c2ae2bd7835a52735690e227c1b03137`.

비교 모집단은 DEV fully_answerable 45문항이다. H11 판정은 DEV에서만 만든다.

canonical qrel primary key는 `(query_id, requirement_id, retrieval_unit_id)`이다. source element는 provenance다. retrieval metric의 relevance 단위가 아니다.

이 benchmark는 LLM-judged / LLM-adjudicated benchmark다. human gold라고 적지 않는다.

qrel은 수정하지 않는다. unjudged는 irrelevant가 아니다. relevance 0은 읽히고 irrelevant로 판정된 unit이다. relevant는 그 unit의 judged relevance가 1 이상인 것이다. unit relevance는 requirement 등급의 최댓값이다.

TEST is not development data. TEST 저장 결과로 reranker 선택, 모델 이름, candidate depth, threshold, parameter를 고르지 않는다. TEST miss를 하나씩 읽어 다음 모델을 설계하지 않는다. 기존 TEST는 H10 final validation의 역사적 결과로만 언급할 수 있다.

## Pool

```
POOL_ID = C20
CANDIDATE_N = 20
HIT_K = 10
FAMILY_KEY = stratum_id
```

C20은 저장된 Dense top-20과 BM25 top-20의 합집합이다. 중복 제거 키는 `retrieval_unit_id`이다.

같은 unit의 source element id가 두 리스트에 있고 서로 다르면 진단을 멈추고 점수를 내지 않는다.

no depth sweep. Top-50, Top-100, 그 밖의 N을 훑어 가장 좋은 candidate depth를 찾지 않는다. depth 선택은 이번 가설의 결과가 아니다.

저장된 full ranking이 파일로 남아 있으면, C20 밖에 있는 relevant unit에 대해 그 파일에 적힌 rank를 추가 진단으로만 읽는다. 그 rank로 pool 정의를 바꾸지 않는다. full ranking 파일이 단위 전체의 best rank를 주지 않으면, 없는 rank를 추정하지 않는다.

현재 최종 순위는 저장된 H10 rule `bm25_primary_dense_head`를 저장된 top-20에 다시 적용한 리스트다. 새 fusion, score fusion, query expansion, routing, embedding, chunking, parser, reranker는 만들지 않는다.

## 실패 유형

Hit@10 failure는 H10 top-10에 relevant unit이 하나도 없는 질의다.

- `ranking_failure`: relevant unit이 C20 안에는 있으나 H10 top-10에는 없다. 이 경우 C20 위의 완벽한 재정렬은 이론적으로 Hit@10을 복구할 수 있다.
- `candidate_generation_failure`: C20 안에 relevant unit이 없다. 이 경우 그 pool 위의 Re-ranker로는 복구할 수 없다.

한 Hit@10 failure는 둘 중 하나다. H10 top-10은 C20의 부분집합이어야 한다. 따라서 Oracle Hit@10과 H10 Hit@10의 차이는 ranking failure 질의 수와 같아야 한다. 다르면 진단을 멈추고 점수를 내지 않는다.

Eligibility, representation, benchmark issue는 C20 밖 failure를 읽은 뒤의 관찰 메모다. 자동 판정 라벨이 아니다. 근거 없이 chunking failure라고 적지 않는다. qrel은 고치지 않는다.

Requirement 상태도 세 값으로 나눈다.

- covered: H10 top-10이 그 requirement의 positive unit을 포함한다.
- `ranking_missed`: positive unit은 C20에 있으나 H10 top-10은 그 requirement를 놓친다.
- `candidate_missing`: 그 requirement의 positive unit이 C20에 없다.

## Oracle

Oracle은 candidate pool 안의 relevance를 완벽하게 알고 Top-K를 최적으로 골랐을 때의 upper bound다.

Unit oracle:

- Oracle Hit@10은 C20에 relevant unit이 하나라도 있으면 1이다.
- Oracle Recall@10의 분자는 C20 안 relevant unit 수와 10 중 작은 값이다. 분모는 질의의 relevant unit 수다. 등급 순으로 relevant unit을 앞에 둔다고 가정한다.

Requirement oracle은 unit oracle과 다른 선택일 수 있다. requirement를 많이 덮는 10개 unit이, relevant unit을 단순히 앞에서부터 10개 채우는 선택과 다를 수 있다. 둘을 하나의 순서로 섞지 않는다. 갈리면 갈린다고 적는다.

Requirement oracle:

- 각 unit은 자신이 relevance 1 이상인 requirement만 덮는다. 다른 requirement의 양성은 복사되지 않는다.
- Oracle Requirement Recall@10의 분자는, C20 unit을 최대 10개 골라 동시에 덮을 수 있는 requirement 수의 최댓값이다.
- Oracle All Requirements Covered@10은 그 질의의 fully_answerable requirement를 모두 덮는 10개 이하 선택이 있을 때 참이다.
- 계산은 requirement 집합의 bitmask DP다. requirement가 16개를 넘으면 진단을 멈춘다. greedy로 대체하지 않는다.
- 같은 coverage mask를 가진 unit은 하나면 충분하다. mask를 두 번 골라도 덮는 requirement는 늘지 않는다.
- 예산 10이 묶이면, C20에 unit이 있는 requirement 수보다 oracle 분자가 작을 수 있다. 그 차이를 숨기지 않는다.

이미 Hit@10인 질의에서 C20 안의 relevant unit이 top-10 밖에 있는 것은 unit recall headroom이다. 그것은 Hit@10 failure가 아니다. 판정 트리의 입력이 아니다. 사실로만 보고한다.

## 판정 트리

FAMILY_KEY는 데이터셋에 이미 있는 `stratum_id`다. 점수를 본 뒤 만든 주제 라벨은 판정에 넣지 않는다.

Strong Support는 아래가 모두 참일 때다.

- Hit@10 failure 안에서 ranking failure 수가 candidate-generation failure 수를 strictly outnumber 한다.
- oracle Hit@10 numerator가 H10 Hit@10 numerator보다 크다.
- oracle requirement recall numerator가 H10 requirement recall numerator보다 크다.
- ranking failure 질의가 at least two queries 이고, 그 질의의 stratum_id가 at least two stratum_id values 이다.

한 질의, 또는 한 stratum에만 있는 headroom은 Strong이 아니다.

Partial Support는 Strong이 아니고, oracle Hit@10 또는 oracle requirement recall이 H10보다 크며, Hit@10 ranking failure 또는 requirement ranking miss가 하나 이상일 때다.

이 안에는 다음이 들어간다.

- candidate-generation failure가 ranking failure와 같거나 더 많다.
- ranking failure가 한 질의이거나 한 stratum이다.
- requirement-only headroom is Partial. Hit@10 failure가 전부 candidate generation이어도, requirement ranking miss가 oracle requirement recall을 올리면 Partial이다. Strong은 아니다.

Reject는 그 밖이다. Oracle 분자가 H10보다 크지 않으면 Reject다. 남은 실패가 C20 안에 relevant unit이 없는 경우이고, Top-10 재정렬의 upper bound가 현재 H10을 넘지 않는 상태다.

oracle 분자가 H10보다 작으면 upper bound가 깨진 것이다. 라벨을 주지 않고 진단을 멈춘다.

이 트리는 Hit@10 ranking failure의 반복을 Strong의 조건으로 둔다. unit recall headroom만으로 Strong을 주지 않는다.

## 금지

Cross-encoder 구현, LLM reranker 구현, reranker 모델 비교, score fusion, query expansion, routing, embedding 변경, chunking 변경, parser 변경, qrel 수정, depth sweep, TEST 검색, TEST를 development data로 사용.

## 대표 사례

점수를 본 뒤 고르지 않는다. H10 Hit@10 miss는 전부 보고한다. H10이 놓친 requirement는 전부 보고한다.

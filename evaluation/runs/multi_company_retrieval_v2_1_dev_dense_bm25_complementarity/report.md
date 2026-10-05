# v2.1 DEV Dense-D2a vs BM25 complementarity

Hybrid는 구현하지 않았다. 비교 모집단은 fully_answerable dev 45문항이다. test는 검색하지 않았고 튜닝에 쓰지 않았다.

- dataset_sha256: `2b1c372b7adaca29af831986bf5e6807c2ae2bd7835a52735690e227c1b03137`
- git_commit: `e119c2c5cdd0a7fe55f6649690cfd0f57f6c763a`
- git_dirty: True

## Headline

| metric | Dense | BM25 |
| --- | --- | --- |
| Hit@1 | 20/45 (0.4444) | 29/45 (0.6444) |
| Hit@5 | 28/45 (0.6222) | 40/45 (0.8889) |
| Hit@10 | 33/45 (0.7333) | 42/45 (0.9333) |
| Recall@5 macro | 0.3872 | 0.5724 |
| Recall@5 micro | 78/231 (0.3377) | 113/231 (0.4892) |
| Recall@10 macro | 0.5612 | 0.7674 |
| Recall@10 micro | 129/231 (0.5584) | 174/231 (0.7532) |
| MRR | 0.5246 | 0.7419 |
| NDCG@10 condensed | 0.4785 | 0.7084 |
| preferred_hit@10 (diagnostic) | 25/45 (0.5556) | 37/45 (0.8222) |
| requirement_recall@10 | 41/58 (0.7069) | 51/58 (0.8793) |
| all_requirements_covered@10 | 32/45 (0.7111) | 41/45 (0.9111) |
| judged_coverage@10 | 1.0000 | 1.0000 |
| judged_coverage@20 | 0.5356 | 0.5567 |

## Query Hit@10 cells

- both hit: 32
- Dense only: 1
- BM25 only: 10
- both miss: 2

## Requirement cells

- both: 40
- Dense only: 1
- BM25 only: 11
- neither: 6
- crossed queries: none

## Pre-registered hybrid label

Primary label: A.

Primary label uses DEV Hit@10 and requirement@10 only. B when BM25 recovers nothing Dense missed: BM25-only query hits are 0, BM25-only requirement recoveries are 0, and no query has crossed requirement coverage. C when that is not true and both-miss queries are at least as many as the two asymmetric query cells combined and at least one third of DEV. A otherwise, which requires BM25 to add a query hit, a requirement, or a crossed multi-requirement case without shared misses dominating. B takes precedence over C so a nested BM25 does not become a hybrid proposal. Shared-miss dominance still selects the non-hybrid follow-up.

## Subgroup Hit@10

small_n은 문항 수가 5 미만이다. 이 표는 우열 순위가 아니다.

### company_name

| group | n | small_n | Dense Hit@10 | BM25 Hit@10 | both | dense only | bm25 only | both miss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 대상 | 9 | no | 5/9 | 9/9 | 5 | 0 | 4 | 0 |
| 삼성전자 | 9 | no | 8/9 | 8/9 | 7 | 1 | 1 | 0 |
| 삼양식품 | 9 | no | 6/9 | 9/9 | 6 | 0 | 3 | 0 |
| 알테오젠 | 9 | no | 7/9 | 9/9 | 7 | 0 | 2 | 0 |
| 이수페타시스 | 9 | no | 7/9 | 7/9 | 7 | 0 | 0 | 2 |

### wording_style

| group | n | small_n | Dense Hit@10 | BM25 Hit@10 | both | dense only | bm25 only | both miss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| layperson | 20 | no | 14/20 | 18/20 | 13 | 1 | 5 | 1 |
| original_term | 20 | no | 16/20 | 20/20 | 16 | 0 | 4 | 0 |
| paraphrase | 5 | no | 3/5 | 4/5 | 3 | 0 | 1 | 1 |

### evidence_form

| group | n | small_n | Dense Hit@10 | BM25 Hit@10 | both | dense only | bm25 only | both miss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| multi_element | 9 | no | 7/9 | 8/9 | 7 | 0 | 1 | 1 |
| paragraph | 24 | no | 17/24 | 24/24 | 17 | 0 | 7 | 0 |
| table | 12 | no | 9/12 | 10/12 | 8 | 1 | 2 | 1 |

### query_type

| group | n | small_n | Dense Hit@10 | BM25 Hit@10 | both | dense only | bm25 only | both miss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| calculation | 2 | yes | 1/2 | 2/2 | 1 | 0 | 1 | 0 |
| causal_explanation | 4 | yes | 3/4 | 4/4 | 3 | 0 | 1 | 0 |
| composition_ratio | 5 | no | 4/5 | 4/5 | 4 | 0 | 0 | 1 |
| cross_filing_comparison | 2 | yes | 0/2 | 1/2 | 0 | 0 | 1 | 1 |
| multi_evidence | 5 | no | 5/5 | 5/5 | 5 | 0 | 0 | 0 |
| single_fact | 18 | no | 13/18 | 18/18 | 13 | 0 | 5 | 0 |
| single_numeric | 2 | yes | 1/2 | 2/2 | 1 | 0 | 1 | 0 |
| table_lookup | 5 | no | 4/5 | 4/5 | 3 | 1 | 1 | 0 |
| time_series_comparison | 2 | yes | 2/2 | 2/2 | 2 | 0 | 0 | 0 |

### stratum_id

| group | n | small_n | Dense Hit@10 | BM25 Hit@10 | both | dense only | bm25 only | both miss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| business_paragraph | 7 | no | 5/7 | 7/7 | 5 | 0 | 2 | 0 |
| business_table | 4 | yes | 4/4 | 4/4 | 4 | 0 | 0 | 0 |
| cross_period | 4 | yes | 2/4 | 3/4 | 2 | 0 | 1 | 1 |
| finance_paragraph | 4 | yes | 3/4 | 4/4 | 3 | 0 | 1 | 0 |
| finance_table | 8 | no | 5/8 | 6/8 | 4 | 1 | 2 | 1 |
| governance_paragraph | 4 | yes | 2/4 | 4/4 | 2 | 0 | 2 | 0 |
| mdna_paragraph | 5 | no | 5/5 | 5/5 | 5 | 0 | 0 | 0 |
| multi_element | 5 | no | 5/5 | 5/5 | 5 | 0 | 0 | 0 |
| overview_paragraph | 4 | yes | 2/4 | 4/4 | 2 | 0 | 2 | 0 |

### period_sensitive

| group | n | small_n | Dense Hit@10 | BM25 Hit@10 | both | dense only | bm25 only | both miss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| False | 21 | no | 18/21 | 21/21 | 18 | 0 | 3 | 0 |
| True | 24 | no | 15/24 | 21/24 | 14 | 1 | 7 | 2 |

### requirement_band

| group | n | small_n | Dense Hit@10 | BM25 Hit@10 | both | dense only | bm25 only | both miss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| multi | 10 | no | 7/10 | 8/10 | 7 | 0 | 1 | 2 |
| single | 35 | no | 26/35 | 34/35 | 25 | 1 | 9 | 0 |

## Coverage and duplicates

- judged_coverage@10 dense 1.0, bm25 1.0, gap 0.0
- coverage gap flag: False
- dense ranking duplicates: 0
- bm25 ranking duplicates: 0
- BM25 rankings shorter than 10: 0
- questions with a positive unit outside the candidate set: 0

## Failure taxonomy

범주는 질문, 양성 unit 본문, 두 retriever의 상위 5개를 읽은 뒤 붙였다. 상세는 `failure_taxonomy.json`이다.

BM25만 Hit@10인 10건에서 반복된 관찰은 질문과 본문이 같은 표면 문자열을 가진다는 점이다. 유통주식수, 주식매수선택권, 종류주식, 지분증권(종속·관계·공동기업 제외), 배당금 지급, 우선주·보통주·전환, 특수관계자, 식품·소재, 유동성장기부채, 현행 기준이 그 문자열이다. wording이 layperson 또는 paraphrase인 문항도 복구 근거는 그 공유 문자열이었다. 받을 돈/갚을 돈, 바꿨나요, 빼서 계산은 양성 본문의 복구 근거로 확인되지 않았다.

Dense만 Hit@10인 1건은 삼성전자 전분기 삼성에스디에스 구매 거래다. 양성 표의 행 이름은 매입이고, BM25 상위는 같은 회사명이 반복되는 계열회사 지분율 표다.

both-miss 2건의 양성 unit은 후보 집합 안에 있다. 이수페타시스 교차 공시는 반기 투자조합 지분과 3분기 채권 연체라는 두 사실이다. BM25는 투자조합을 제외한다고 쓴 문장을 올렸다. 현금흐름 취득 문항은 Dense 20위에 2026년 행이 있고 4위는 2025년의 같은 행 이름이다. BM25는 투자부동산 장부 변동 주석에 머물렀다. 상위 10 합집합에도 그 양성 unit은 없다.

query Hit가 양쪽인 삼양식품 제62–64기 문항은 R1 매출액은 양쪽이 복구하고 R2 영업이익은 BM25만 복구한다. Dense의 상위 10 양성 unit은 면스낵 품목 매출 표다. 같은 질의에서 requirement가 서로 엇갈리지는 않았다. crossed query는 0이다.

## FACT

- 모집단은 v2.1 fully_answerable dev 45이다. test 검색 횟수는 0이다.
- Dense와 BM25는 회사별 동일 receipt 집합의 D2a unit을 검색했다. universe 중복 id는 0이다.
- 순위 안 중복 retrieval_unit_id는 Dense 0, BM25 0이다.
- Hit@10 judged coverage는 양쪽 1.0이다. unjudged와 relevance 0은 분리되어 있고, 상위 10의 unjudged 평균은 0이다.
- 기존 v2.1 Dense-D2a dev 순위와 이번 Dense 순위의 mismatch는 0/45이다.
- BM25 k1=1.2, b=0.75는 커밋 236c2a3에서 고정된 값이다. 이번 실험은 바꾸지 않았다.
- 사전 규칙의 primary label은 A다. BM25-only query 10, requirement BM25-only 11, both-miss 2다.

## INTERPRETATION

두 retriever는 같은 실패를 반복하지 않는다. DEV Hit@10에서 BM25가 Dense의 실패 10건을 복구하고, Dense가 BM25의 실패 1건을 복구한다. both-miss는 2건이다. BM25 성공의 내용은 정확한 용어, 계정 구문, 표 행 이름, 날짜다. Dense의 유일한 단독 성공은 구매와 매입처럼 표면이 어긋나고, 같은 고유명사가 다른 표에 더 자주 나오는 경우다.

query Hit는 보완의 크기를 크게 보이게 한다. 삼양식품 차입금 문항은 BM25 Hit이지만 requirement 3개 중 1개만 복구한다. 제62–64기 문항은 둘 다 Hit이지만 영업이익 행은 BM25만 갖는다. 서로 다른 requirement를 한 질문 안에서 나눠 갖는 crossed 사례는 없다.

## Limitations

- k1과 b는 v2.1 test로 고른 값은 아니다. v2.1 dev에서 고른 값도 아니다. 다른 값이 보완 구조를 바꾸는지 이 실험은 말하지 않는다.
- 질의 전처리는 다르다. Dense는 BGE-M3 질의 인코더이고 BM25는 ko-mixed-v1이다. 둘 다 원문 질문을 받는다.
- Hit@10의 judged coverage가 1.0인 것은 풀이 Dense와 BM25 상위로 만들어졌기 때문이다. 상위 10 비교는 unjudged에 덜 민감하고, 20위 coverage는 0.54와 0.56이다.
- small_n subgroup으로 회사나 query type의 우열을 일반화하지 않는다.
- both-miss 중 현금흐름 문항의 양성 unit은 Dense 20위다. 상위 10 fusion은 그 unit을 넣지 못한다.

## Hybrid hypothesis

사전 규칙의 라벨은 A다. 근거는 BM25가 Dense가 놓친 unit을 실제로 올리고, 그 실패의 내용이 Dense의 단독 성공과 다르다는 점이다. both-miss가 DEV의 주된 실패는 아니다.

다음 실험은 이번 DEV 순위 두 개를 고정 규칙으로 합치는 것이다. 후보는 저장소에 이미 있는 RRF k=60이다. 가중치와 k는 이번 DEV에서 새로 고르지 않고, test로는 고르지 않는다. 볼 기준은 삼성전자 구매 거래 Hit을 유지한 채 BM25-only 10건 중 얼마나 복구되는지, 그리고 both-hit 32건이 무너지는지다. 차입금 문항은 requirement 단위로 다시 본다.

상위 10 합집합으로 기대하지 않을 실패는 이수페타시스 두 건이다. 하나는 두 공시의 두 사실이고, 하나는 양성 unit이 Dense 20위다. 그 둘은 기간·명세 구분과 복수 사실 질의의 문제로 따로 둔다.

## Tests / lint

`uv run pytest -q` 272 passed. `uv run ruff check .` passed. 추가 테스트는 dev 모집단, 네 칸, requirement cross, hybrid 규칙, 중복 순위, unit id 채점이다.

## Changed files

- `src/disclosure_agent/evalset_v2_1/complementarity.py`
- `src/disclosure_agent/evalset_v2_1/__main__.py`
- `tests/test_evalset_v2_1_complementarity.py`
- `evaluation/runs/multi_company_retrieval_v2_1_dev_dense_bm25_complementarity/`
- `docs/worklogs/2026-10-05_2045_v21-dev-dense-bm25-complementarity.md`

v2와 v2.1 dataset 파일은 쓰지 않았다.

## Git commit

숫자는 부모 `e119c2c5cdd0a7fe55f6649690cfd0f57f6c763a` 위에서 dirty tree로 계산했다. 그 기록이 이 커밋이다. 이후 RRF는 이 커밋을 baseline으로 둔다.

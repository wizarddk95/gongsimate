# v2.1 DEV Dense-D2a vs BM25 complementarity — 2026-10-05 20:45 KST

## Goal

Hybrid를 만들기 전에, v2.1 DEV fully_answerable 45문항에서 Dense-D2a와 기존 BM25의 실패가 서로 다른지 확인한다. v2와 v2.1은 수정하지 않는다. test로 파라미터를 고르지 않는다.

## Setup

같은 질문, `select_filings`의 빈 period/report/as_of, 그 receipt 안의 D2a retrieval unit, v2.1 unit qrel, K=20, `evalset_v2_1.metrics`를 쓴다. Dense는 제품 경로 `CatalogDenseSearch`다. BM25는 `Bm25Index.search`이고 k1=1.2, b=0.75, tokenizer는 ko-mixed-v1이다. 새 embedding, query expansion, reranker, hybrid, RRF는 없다.

dataset sha256 `2b1c372b7adaca29af831986bf5e6807c2ae2bd7835a52735690e227c1b03137`. 기준 git `e119c2c5cdd0a7fe55f6649690cfd0f57f6c763a`.

## Audit

다섯 회사의 BM25 문서 id는 Dense가 검색한 eligible unit id와 같았다. universe 중복 id 0, 순위 내 중복 0, 양성 unit이 후보 밖에 있는 문항 0, BM25 반환이 10 미만인 문항 0. Hit@10 judged coverage는 양쪽 1.0이고 gap flag는 꺼져 있다. 상위 10 unjudged 평균은 0이다. relevance 0과 unjudged는 metric이 구분한다.

저장된 v2.1 Dense-D2a dev 45문항의 unit 순위와 이번 Dense 순위는 45/45가 같았다. test 순위는 결정에 넣지 않았다.

BM25 파라미터는 커밋 236c2a3에서 고정된 뒤 바뀌지 않았다. 저장소에 k1/b sweep는 없다. v2.1 test로 고른 값이 아니다.

질의 전처리 차이는 원문 질문을 공유한 뒤 Dense만 BGE-M3, BM25만 NFKC와 ko-mixed-v1을 쓰는 것이다.

## DEV headline

| metric | Dense | BM25 |
| --- | --- | --- |
| Hit@1 | 20/45 | 29/45 |
| Hit@5 | 28/45 | 40/45 |
| Hit@10 | 33/45 | 42/45 |
| Recall@5 macro | 0.387 | 0.572 |
| Recall@5 micro | 78/231 | 113/231 |
| Recall@10 macro | 0.561 | 0.767 |
| Recall@10 micro | 129/231 | 174/231 |
| MRR | 0.525 | 0.742 |
| NDCG@10 condensed | 0.478 | 0.708 |
| preferred_hit@10 | 25/45 | 37/45 |
| requirement_recall@10 | 41/58 | 51/58 |
| all_requirements_covered@10 | 32/45 | 41/45 |
| judged_coverage@10 | 1.000 | 1.000 |
| judged_coverage@20 | 0.536 | 0.557 |

상위 10이 전부 judged라 condensed NDCG와 unjudged-as-zero NDCG는 같다.

## Complementarity

Hit@10: both 32, Dense only 1, BM25 only 10, both miss 2.

requirement@10: both 40, Dense only 1, BM25 only 11, neither 6. crossed query 0.

사전 규칙 primary label은 A다. BM25가 Dense가 놓친 것을 복구하고, both-miss는 비대칭 셀의 합보다 작고 DEV의 3분의 1보다 작다.

BM25-only 본문은 유통주식수, 주식매수선택권, 종류주식, 지분증권 구문, 배당금 지급, 우선주 전환, 특수관계자 표, 식품·소재와 부문 제거, 유동성장기부채 한 requirement, 현행 기준 회계정책 문단이다. Dense-only 1건은 삼성전자 구매/매입이다. both-miss 2건은 두 공시의 두 사실, 그리고 2026 현금흐름 행이 Dense 20위에 있는 기간 혼동이다.

삼양식품 제62–64기는 둘 다 query Hit이고 영업이익 requirement는 BM25만 복구한다.

## Tests

`uv run pytest -q`는 272 passed, warning 1이다. `uv run ruff check .`는 통과했다. 추가 테스트는 dev 모집단이 test를 빼는지, Hit@10 네 칸, requirement cross, hybrid 규칙 A/B/C, 순위 중복을 metric이 숨기지 않는지, BM25 hit가 element id가 아니라 retrieval unit으로 채점되는지다.

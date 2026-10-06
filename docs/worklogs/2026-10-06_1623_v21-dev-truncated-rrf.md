# v2.1 DEV candidate-truncated RRF — 2026-10-06 16:23 KST

## Goal

full-ranking RRF가 기각된 뒤, 한 변수만 바꿔 본다. Dense top-20과 BM25 top-20의 union만 fusion 후보로 두고, top-20 밖 순위는 contribution 0으로 둔다. weighted fusion, routing, reranker, query expansion, embedding, chunking은 건드리지 않는다. TEST는 검색하지 않는다.

## Setup

기준 commit은 `6c3fec85523545e3552a71930b56409b8be8e754`이다. 입력은 complementarity DEV top-20과 full-RRF 사본이다. 두 사본의 Dense/BM25 top-20이 45/45 같은 뒤에만 점수를 냈다. retrieval은 다시 돌리지 않았다. dataset sha256 `2b1c372b7adaca29af831986bf5e6807c2ae2bd7835a52735690e227c1b03137`.

식은 `I_dense/(60+dense_rank) + I_bm25/(60+bm25_rank)`이다. I는 그 retriever의 top-20 안에서만 1이다. 705, 283, 75 같은 full rank는 점수 입력이 아니다. 동점은 score 내림차순, retrieval_unit_id 오름차순이다. 저장 top-20에는 ordinal이 없다. 공식 대조 1555건, 점수에 들어간 최대 rank 20, deep-tail rank 사용 0.

k=60, N=20에서 교집합 최소 점수는 0.02500이고 단일 리스트 최대 점수는 0.01639이다. 양쪽 top-20에 있는 unit이 한쪽에만 있는 unit보다 항상 앞에 선다.

## DEV headline

| metric | Dense | BM25 | Full RRF | Truncated RRF |
| --- | --- | --- | --- | --- |
| Hit@1 | 20/45 | 29/45 | 28/45 | 29/45 |
| Hit@5 | 28/45 | 40/45 | 33/45 | 37/45 |
| Hit@10 | 33/45 | 42/45 | 36/45 | 40/45 |
| Recall@5 macro | 0.387 | 0.572 | 0.468 | 0.505 |
| Recall@10 macro | 0.561 | 0.767 | 0.629 | 0.725 |
| MRR | 0.525 | 0.742 | 0.686 | 0.737 |
| NDCG@10 condensed | 0.478 | 0.708 | 0.654 | 0.684 |
| NDCG@10 zero-filled | 0.478 | 0.708 | 0.615 | 0.675 |
| requirement recall@10 | 41/58 | 51/58 | 46/58 | 49/58 |
| all requirements covered@10 | 32/45 | 41/45 | 35/45 | 39/45 |
| judged_coverage@10 | 1.000 | 1.000 | 0.787 | 0.969 |

truncated mean unjudged@10은 0.311이다. full-RRF는 2.133이다. unjudged는 nonrelevant가 아니다.

## Recovery

both-hit 32/32는 Hit@10을 유지했다. full-RRF가 잃은 BM25-only 8건 중 5건이 Hit@10으로 돌아왔다. 주식매수선택권은 BM25 1, Dense full rank 705 미사용, truncated 1이다. 이전 Dense 1위 irrelevant는 BM25 full rank 32가 top-20 밖이라 contribution 0이고, 같은 점수 0.016393에서 unit id 순으로 2위다. 유통주식수는 BM25 1, Dense 283 미사용, 교집합 6 뒤의 truncated 7이다. 특수관계자 채권·채무는 BM25 7, Dense 75 미사용, 교집합 8 뒤의 truncated 14로 Hit@10 밖이다.

미회복 3건은 BM25 8→15, BM25 7→14, BM25 4→11이다. Dense-only 구매/매입은 BM25 full rank 75를 쓰지 않고 truncated 2위로 Hit@10에 남았다. both-miss 2건은 Hit@10 밖이다. 현금흐름 문항은 full-RRF 10위에서 truncated 39위로 나갔고, 이것을 period/scope 해결이나 악화의 결론으로 쓰지 않는다.

requirement both 40 유지, dense-only 1 복구, bm25-only 11 중 8 유지 3 상실, neither 6은 전부 잔존이다.

## Verdict

판정은 B, 부분 지지다. Hit@10 40/45는 full-RRF 36/45보다 높고 BM25 42/45보다 낮다. 8건 중 5건 회복은 강지지 기준 6건에 못 미친다. Recall@5 macro 하락은 0.067로 0.05를 넘는다. MRR 하락은 0.005, NDCG@10 하락은 0.025이다.

다음 가설은 미회복 3건의 등위 결합이다. weighted RRF, routing, reranker, period-aware ranking은 구현하지 않는다.

## Tests

`uv run pytest -q`는 293 passed, warning 1이다. `uv run ruff check .`는 통과했다.

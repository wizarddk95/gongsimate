# v2.1 DEV H10 BM25-primary Dense-head insertion — 2026-10-07

## Goal

BM25가 dominant인 v2.1 DEV에서, 대칭 fusion 대신 BM25 순서를 primary로 두고 Dense rank 1만 제한적으로 보조하는 단일 rule이 lexical Hit@10을 지키면서 Dense-only semantic recovery를 남기는지 본다. weight, k, N, alpha sweep과 다른 rule의 비교는 하지 않는다. TEST는 검색하지 않는다.

## Rule

`bm25_primary_dense_head`. semantics는 점수 전에 `docs/experiments/2026-10-07_h10_asymmetric_fusion_semantics.md`에 고정했다.

Dense rank 1이 BM25 top-10 안에 있으면 fused list는 BM25와 같다. 밖에 있으면 fused list는 BM25 rank 1..9 뒤에 그 head를 rank 10으로 두고, 나머지 BM25 tail을 잇는다. Dense rank 2..20은 정렬 입력이 아니다. 교집합 가산은 없다. 점수는 만들지 않는다.

BM25 ordering이 바뀌는 조건은 그 한 경우뿐이다. rank 1..9는 모든 질의에서 그대로다.

## Setup

dataset `multi_company_retrieval_v2_1` sha256 `2b1c372b7adaca29af831986bf5e6807c2ae2bd7835a52735690e227c1b03137`. 입력은 저장된 DEV Dense/BM25 top-20과 truncated RRF 산출물이다. retrieval 재실행 0. TEST 검색 0. qrel과 metric 정의는 바꾸지 않았다. unjudged는 irrelevant가 아니다. NDCG headline은 condensed다.

사전 판정: Strong은 Hit@10 >= 41/45, BM25-only 상실 <= 1, Dense-only Hit@10 유지, Recall@5 / Recall@10 / MRR / NDCG@10 하락 <= 0.05, requirement_recall >= 49/58, all_requirements >= 39/45. Partial은 Strong이 아니면서 Dense-only 유지, Hit@10 > 40, BM25-only 상실 < 3. 그 밖은 Reject.

## FACT

| metric | Dense | BM25 | Truncated | H10 |
| --- | --- | --- | --- | --- |
| Hit@1 | 20/45 | 29/45 | 29/45 | 29/45 |
| Hit@5 | 28/45 | 40/45 | 37/45 | 40/45 |
| Hit@10 | 33/45 | 42/45 | 40/45 | 43/45 |
| Recall@5 macro | 0.387 | 0.572 | 0.505 | 0.572 |
| Recall@10 macro | 0.561 | 0.767 | 0.725 | 0.787 |
| MRR | 0.525 | 0.742 | 0.737 | 0.744 |
| NDCG@10 condensed | 0.478 | 0.708 | 0.684 | 0.713 |
| requirement recall@10 | 41/58 | 51/58 | 49/58 | 52/58 |
| all requirements covered@10 | 32/45 | 41/45 | 39/45 | 42/45 |
| judged_coverage@10 | 1.000 | 1.000 | 0.969 | 1.000 |
| unjudged mean@10 | 0.000 | 0.000 | 0.311 | 0.000 |

BM25 대비 하락(BM25 - H10)은 Recall@5 0, Recall@10 -0.019788, MRR -0.002222, NDCG@10 -0.004135이다. 음수는 H10이 높다.

edit는 none 16, insertion 20, promotion 9이다. 편집된 29건의 Dense head는 relevant 6, irrelevant 23이고 unjudged는 0이다. 밀려난 BM25 rank 10은 relevant 7, irrelevant 22이고 unjudged는 0이다. 이 DEV에서 Dense rank 1은 전부 judged라 judged_coverage@10이 1에 남았다. rule이 unjudged head를 judged로 바꾸지는 않는다.

Hit@10 전이는 BM25 hit→H10 hit 42, BM25 miss→H10 hit 1, BM25 miss→H10 miss 2이다. BM25 hit→H10 miss는 0이다. truncated에는 BM25 hit→miss 3이 있었다. 첫 양성 movement는 neutral 42, recovered 1, still_missing 2이다.

BM25-only Hit@10 상실 0. both-hit Hit@10 상실 0. Hit@10 유입은 삼성전자 구매/매입 1건이다.

세 residual의 첫 양성은 BM25 8, 7, 4에서 H10도 8, 7, 4다. truncated는 15, 14, 11이었다. 이 일치는 protected prefix 귀결이다. 효과 크기로 읽지 않는다.

대상 지분증권은 그 귀결과 별개로 BM25 rank 10 relevant를 창 밖으로 밀었다. rank 8과 9의 relevant가 남아 Hit@10은 유지된다. relevant rank 10이 밀린 질의는 7건이고, 어느 것도 Hit@10이나 requirement를 잃지 않았다.

삼성전자 `V2-00126380-finance_table-02`는 Dense rank 1, BM25 top-20 양성 없음, truncated rank 2, H10 rank 10, head relevant, Hit@10 유지다. 기준은 Hit@10이지 rank 2 유지가 아니다.

requirement는 BM25 대비 유입 1(`V2-00126380-finance_table-02/R1`), 상실 0이다. prior cell은 both 40 retained, dense_only 1 recovered, bm25_only 11 retained, neither 6 still missing이다.

## INTERPRETATION

판정은 Strong Support다. reason은 `all_strong_floors`다.

대칭 truncated RRF가 만든 세 Hit@10 회귀는 이 semantics 안에서는 같은 방식으로 생기지 않는다. 교집합이 BM25 순서를 건너뛰지 않고, Dense rank 2..20이 BM25-only 양성과 점수 경쟁하지 않는다. 그 세 건이 8, 7, 4위에 남은 것은 그 구조의 확인이다.

경험적 내용은 marginal slot 29건이다. relevant head 6개 중 Hit@10을 새로 만든 것은 사전 사례인 삼성전자 1건이다. relevant BM25 rank 10을 민 7건은 lexical unit을 top-10 밖으로 내보내지만, 이 DEV에서는 rank 1..9의 redundant coverage 때문에 Hit@10과 requirement 상실이 되지 않았다. Recall@10과 NDCG@10은 BM25보다 조금 높고, MRR 증분 0.002222는 삼성전자 한 질의의 첫 양성이 rank 10에 들어온 크기와 맞다.

이 지지는 TEST를 열지 않는다. DEV headline은 이 rule이 BM25와 truncated RRF보다 사전 기준 안에 있다는 뜻이다. 채택을 벤치마크 봉인 해제와 묶지 않는다.

## Tests

`uv run pytest -q`는 311 passed, warning 1이다. `uv run ruff check .`는 통과했다.

산출물: `evaluation/runs/multi_company_retrieval_v2_1_dev_h10_asymmetric/`.

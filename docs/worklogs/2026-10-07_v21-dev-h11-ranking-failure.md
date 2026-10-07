# v2.1 DEV H11 ranking failure versus candidate generation — 2026-10-07

## Goal

남은 retrieval failure가 C20 안에서의 ranking failure인지, C20 자체에 relevant unit이 없는 candidate-generation failure인지 DEV 45에서 가른다. Re-ranker는 구현하지 않는다.

## Semantics

`docs/experiments/2026-10-07_h11_ranking_failure_semantics.md`를 진단 전에 고정했다. sha256 `e900ea26a1da69ae417d35ab303202a128539394edf38f8101c0d50390412119`.
C20은 Dense top-20과 BM25 top-20의 합이다. Oracle은 그 pool의 upper bound다. Strong / Partial / Reject 트리는 숫자 컷 없이 비교만 사용한다.

## Setup

dataset `multi_company_retrieval_v2_1` sha256 `2b1c372b7adaca29af831986bf5e6807c2ae2bd7835a52735690e227c1b03137`. DEV fully_answerable 45. retrieval 재실행 0. TEST 검색 0.
입력은 저장된 complementarity top-20과 H10 rule의 재적용이다. full-RRF 파일은 첫 relevant unit의 저장 rank를 읽는 데만 쓴다.

## FACT

H10 Hit@10 43/45. Oracle Hit@10 44/45. ranking failure 1. candidate-generation failure 1.
H10 requirement 52/58. Oracle requirement 54/58.
판정 `Partial Support`, reason `candidate_generation_comparable_or_larger`.

## INTERPRETATION

C20 재정렬로 이론상 복구되는 failure가 있다. 다만 Hit@10 failure에서 candidate-generation failure가 같거나 더 크거나, 복구 가능성이 소수 질의 또는 한 stratum에 모이거나, requirement 쪽에서만 headroom이 있다. 한 모델의 예상 개선치로 바꾸지 않는다.

## Tests

`uv run pytest -q`는 335 passed, warning 1이다. `uv run ruff check .`는 통과했다.

산출물: `evaluation/runs/multi_company_retrieval_v2_1_dev_h11_ranking_failure/`.

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

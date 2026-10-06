## Portfolio Summary

### Hypothesis

Top-20 밖 deep-tail 순위를 빼면 BM25 lexical Hit@10을 지키면서 Dense의 상위 semantic signal을 더할 수 있다.

### Success Condition

- BM25 Hit@10 42/45를 유지하거나 41/45 이상.
- 삼성전자 구매/매입 Dense-only 문항이 Hit@10에 남는다.
- full-RRF가 잃은 BM25-only 8건의 대부분을 Hit@10으로 되돌린다.
- MRR, NDCG@10, Recall@5 macro가 BM25보다 0.05를 넘겨 떨어지지 않는다.
- requirement recall과 all-requirements covered가 BM25 근처에 남는다.

### Result

Hit@10 40/45, BM25-only 8건 중 회복 5, Dense-only Hit@10 1/1, MRR 0.737.

### Why It Worked / Failed

deep-tail rank를 빼자 BM25-only 8건 중 5건이 Hit@10으로 돌아왔고 Dense-only는 truncated rank 2에 남았다. 남은 손실은 BM25 단독 rank 4, 7, 8이 교집합과 상대 리스트의 더 높은 등위 뒤에 선 것이다. k=60, N=20에서 교집합 최소 점수는 0.02500이고 단일 리스트 최대 점수는 0.01639이다.

### Decision

부분 지지

### Next Hypothesis

미회복 문항은 BM25 단독 순위가 교집합과 Dense 단독의 더 앞선 등위 뒤에 놓인 BM25 8위가 truncated 15위, BM25 7위가 truncated 14위, BM25 4위가 truncated 11위이다. 이 등위 결합이 Hit@10 잔여 손실의 원인인지가 다음 가설이다.

| Experiment | Hypothesis | Success condition | Result | Verdict | Next |
| --- | --- | --- | --- | --- | --- |
| Dense-D2a baseline | v2.1 DEV에서 Dense-D2a 단독 순위를 측정한다. | DEV fully_answerable 45의 Hit, Recall, MRR, NDCG를 고정 산출물로 남긴다. | Hit@10 33/45, MRR 0.525, NDCG@10 0.478, requirement recall 41/58. | 기준선 확정 | BM25와 실패가 겹치는지 본다. |
| Dense vs BM25 complementarity | Dense와 BM25의 Hit@10 실패가 서로 다른가. | BM25가 Dense 실패를 복구하고 both-miss가 비대칭 셀보다 작다. | both 32, Dense only 1, BM25 only 10, both miss 2. BM25 Hit@10 42/45. | 상보성 관측 | 전체 순위 RRF가 그 차이를 순위로 옮기는지 본다. |
| Full-ranking RRF | k=60 전체 순위 RRF가 BM25 coverage를 유지한 채 Dense-only를 복구하는가. | Hit@10 감소 1 이하, BM25-only 손실 1 이하, Dense-only 복구 또는 순위 품질 개선. | Hit@10 36/45, BM25-only 2/10 유지, Dense-only 1/1, MRR 0.686. | 기각 | Top-20 밖 순위를 fusion에서 뺀다. |
| Candidate-truncated RRF | Top-20 밖 deep-tail 순위를 빼면 BM25 lexical Hit@10을 지키면서 Dense의 상위 semantic signal을 더할 수 있다. | BM25 Hit@10 41/45 이상, Dense-only 유지, full-RRF가 잃은 BM25-only 8건의 대부분 회복, MRR/NDCG/Recall@5 하락 0.05 이내, requirement가 BM25 근처. | Hit@10 40/45, BM25-only 8건 중 회복 5, Dense-only Hit@10 1/1, MRR 0.737. | 부분 지지 | 미회복 문항은 BM25 단독 순위가 교집합과 Dense 단독의 더 앞선 등위 뒤에 놓인 BM25 8위가 truncated 15위, BM25 7위가 truncated 14위, BM25 4위가 truncated 11위이다. 이 등위 결합이 Hit@10 잔여 손실의 원인인지가 다음 가설이다. |

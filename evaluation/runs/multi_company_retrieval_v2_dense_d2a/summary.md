# multi_company_retrieval_v2 Dense-D2a

LLM-judged evaluation set. 사람 검수는 없다. v1 31문항 결과와 같은 계열이 아니다.

- dataset_sha256: `d3635756a57c39d6d4ef8722f5f118348c00bbc034bf5963fb7720d184e6f137`
- test fully_answerable headline: 55. 양성 qrel이 없어 빠진 문항은 0.
- Hit@1 26/55, Hit@5 35/55, Hit@10 40/55
- Recall@5 macro 0.384, Recall@10 macro 0.524, micro 255/502
- MRR 0.546 (cutoff 20)
- NDCG@10 condensed 0.466
- preferred_hit@10 30/55 (diagnostic)
- judged_coverage@10 1.0, unjudged_count@10 mean 0
- judged_coverage@20 0.603, unjudged_count@20 mean 7.95
- requirement recall@10 46/65, all-requirements-covered@10 37/55

상위 10은 이번 Dense 실행에서 모두 판정됐다. 11–20위는 풀에 일부만 있어 coverage@20이 낮다. unjudged는 relevance 0으로 저장하지 않았다.

테스트 분할은 이후 비교용이다. 이 분할을 반복해서 보며 검색기를 고르지 않는다.

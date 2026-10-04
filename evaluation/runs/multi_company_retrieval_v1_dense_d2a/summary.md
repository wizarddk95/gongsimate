# Multi-company Dense-D2a baseline

Dataset `multi_company_retrieval_v1`, sha256 `4b9795007d53820c2aca2c837bb884e010af8b2e60bd4b491fd77cfe43e65824`.

Run at 2026-10-04T04:17:52Z. Git parent `b34a9d5da0f16aba4cd1b4996ece6b0a29632984`. The tree was dirty because this dataset and runner were not committed yet.

This file records the measurement. It does not change retrieval.

## What was measured

Product path with the company already chosen: `select_filings` then exact Dense-D2a. Query text is the raw question. `top_k` is 20. No period filter, no report-type filter, no rewrite, no BM25, no hybrid, no reranker.

Model `BAAI/bge-m3` revision `5617a9f61b028005a4858fdac845db406aefb181`. Representation `retrieval-unit-d2a-v1`. L2 normalized dot product, exhaustive search.

Headline population is the core subset, 31 questions. MC32 is a filing-policy check and is not in the headline.

## Metric definitions

Hit@k is 1 when any evidence with relevance at least 1 appears in the top k. Relevance 1 counts. A preferred filing is not required.

Recall@k macro is the unweighted mean of per-question relevant elements found over relevant elements labeled. Micro pools those counts. One element counts once even if several units contain it.

MRR is the mean of 1/rank of the first relevance >= 1 unit. Missing inside the cutoff 20 is 0.

NDCG@10 uses gain `2^relevance - 1`. Relevance 2 gains 3, relevance 1 gains 1. A repeated element adds gain once. The ideal list is that question's grades, sorted, cut at 10.

preferred_hit@k is diagnostic. A miss there is not a retrieval failure when other valid evidence was retrieved.

`small_n` in the JSON is true when a group has fewer than 5 questions. Company groups are 5 to 7 questions. Those counts are descriptive, not a ranking of companies.

## Headline (n=31)

| Metric | Count | Value |
| --- | --- | --- |
| Hit@1 | 12/31 | 0.387 |
| Hit@5 | 23/31 | 0.742 |
| Hit@10 | 28/31 | 0.903 |
| Recall@5 macro |  | 0.538 |
| Recall@5 micro | 29/65 | 0.446 |
| Recall@10 macro |  | 0.691 |
| Recall@10 micro | 38/65 | 0.585 |
| MRR |  | 0.543 |
| NDCG@10 |  | 0.447 |
| preferred_hit@10 | 21/31 | 0.677 |

Hit@10 is high because a note or a company-overview restatement often states the same fact. preferred_hit@10 is 21/31, and NDCG stays near 0.45, because the direct annual section is often lower or absent and because many labeled elements are never reached.

## Filing policy (n=1, not a rate)

MC32, as_of `2026-03-20T00:00:00+09:00`. The correction `20260327000936` is after that instant, so the gold is the original `20260318000733`. That element is rank 8. Hit@10 is 1, Hit@5 is 0, MRR is 0.125. One question.

## Company (core)

| Company | n | Hit@1 | Hit@5 | Hit@10 | Recall@10 macro | MRR | preferred@10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 이수페타시스 | 7 | 4 | 5 | 5 | 0.417 | 0.607 | 2 |
| 삼성전자 | 7 | 3 | 5 | 7 | 0.738 | 0.607 | 6 |
| 삼양식품 | 6 | 2 | 4 | 6 | 0.875 | 0.507 | 5 |
| 알테오젠 | 6 | 2 | 5 | 5 | 0.833 | 0.583 | 5 |
| 대상 | 5 | 1 | 4 | 5 | 0.617 | 0.359 | 3 |

이수페타시스 is the only company with two top-10 misses (MC02, MC07). Its preferred@10 is 2/7. The other companies each have at most one top-10 miss.

## Query type

| Type | n | Hit@10 | Recall@10 macro | MRR | preferred@10 |
| --- | --- | --- | --- | --- | --- |
| business | 13 | 11 | 0.532 | 0.473 | 6 |
| fact | 9 | 8 | 0.833 | 0.502 | 8 |
| performance | 4 | 4 | 1.000 | 0.667 | 4 |
| risk | 5 | 5 | 0.600 | 0.700 | 3 |

Performance is four questions, all hits, and is too small to treat as a stable strength. Business questions are the weak preferred slice: 6/13 preferred inside 10, while Hit@10 is 11/13. Risk Hit@10 is 5/5 only after note tables that state the management sentence were labeled. The business-section risk narrative is often outside 10.

## Other slices

Wording: colloquial 14/14 Hit@10, exact 10/13, paraphrase 4/4 (n=4). The three misses are all exact wording (MC02, MC07, MC21). That does not mean colloquial is easier in general. MC21 fails on the generic phrase "플랫폼 기술", while the colloquial description and the proper name hit.

Evidence form: paragraph 23/24 Hit@10, table 4/6, mixed 1/1 at rank 8 (MC16, n=1). The two table misses are MC02 and MC07.

Period-sensitive: 8/8 Hit@10 and 8/8 preferred@10. Time-insensitive: 20/23 Hit@10 and 13/23 preferred@10. Asking for a year or a half did not make retrieval worse on this set. Wrong-period tables still rank above the asked year in MC14.

## What the ranks show

Strong when the question carries a distinctive number or name: 10.9%, 98.7%, HBM, 95.09% at rank 4, ALT-B4 78.5%, MSD, Hybrozyme, USD risk, 삼양 매출 이유, 공장·원재료.

Weak in four observed ways.

1. The same fact in a note or in I. 회사의 개요 outranks II. 사업의 내용. MC01, MC03, MC05, MC13, MC15, MC16. Hit can be true while the preferred section is missing.
2. Copies of one paragraph across filings fill the top 10. MC01 and MC02. The automatic duplicate counter rarely fires, because those copies are unlabeled. The list is still redundant.
3. A nearby but different fact wins the lexical overlap. MC07 retrieves MLB use, not the producing entity. MC21 retrieves a generic biotech purpose, not Hybrozyme. MC27 ranks 대상홀딩스 shareholder text first.
4. The right table is in the top 10 but below another period or a sales amount. MC14 rank 8, with 39.3% and 33.0% above it. MC09 rank 8. MC31 rank 7.

MC29 is the preferred-versus-correctness split: half-year product text is rank 1 and valid, the annual overview never appears in the top 20.

MC30 looked like a miss until the measurement paragraph in the same risk section was labeled. The heading element and the paragraph element are different. Granularity of the label, not a missing section.

## Hypotheses for the next experiment

Not implemented here.

- BM25 beside Dense, for MC02 and MC07. These two top-10 misses are table rows. Distinctive headers such as 구체적용도 and the producing entity never enter the dense list, while a nearby PCB or MLB sentence does. The next run should check whether BM25 top hits include those table elements.
- A filing or section prior, for MC01, MC05, MC13, and the repeated-filing lists. Notes and older copies outrank the latest annual business section. Collapsing near-duplicate filings, or preferring II. 사업의 내용 for a business question, is the hypothesis. MC27 adds a name collision with 대상홀딩스 in 주주에 관한 사항.
- A reranker, for late but present evidence: MC09 rank 8, MC14 rank 8, MC16 rank 8, MC20 rank 6, MC31 rank 7, MC32 rank 8. The question is whether the preferred passage moves above the note or the other period.
- MC21 is a poor BM25 target by itself. The query words are generic, and the answer uses names the query does not contain. The paired questions that say the mechanism or the name already hit. Query expansion is the hypothesis to test, not a lexical index alone.
- Chunking is not the first change. MC20's split paragraph is still rank 6.

## Files

- `metrics.json` — configuration, corpora, headline, breakdowns, per-question scores
- `per_question.json` — same questions with hit text
- `inspection.jsonl` — expected evidence and top 10 for review
- `failure_review.json` — human category and note for every question

# Retrieval eval set v2.1 audit — 2026-10-05 16:36 KST

## Goal

검색기를 바꾸기 전에 benchmark 정의를 검증한다. 대상은 dev/test 대표성, relevance 단위, Recall 분모, NDCG 의미, metric 구현이다. Dense, BM25, hybrid, reranker, 질문 생성, judge 재호출은 하지 않는다.

`multi_company_retrieval_v1`과 `multi_company_retrieval_v2`는 수정하지 않았다. 수정은 `multi_company_retrieval_v2_1`에만 있다.

## FACT: v2가 실제로 하던 일

분할 코드는 `corpus_sample._slot`이다. stratum 안에서 index가 1부터 시작하고, `test_n = round(n * 0.6)`까지가 test다. wording은 `wording_cycle`의 `(index - 1) % 3`이고 cycle은 original_term, layperson, paraphrase다. query type도 `allowed_query_types`를 같은 index로 돈다.

그 결과 fully_answerable 100개의 test 55 / dev 45는 회사마다 11 / 9로 맞다. wording은 test가 original_term 45, layperson 10, paraphrase 0이다. paraphrase 10개는 전부 dev다. query type도 같은 이유로 한쪽에만 있다. test에 없는 타입은 composition_ratio 10, calculation 5, cross_filing_comparison 5다. dev에 없는 타입은 causal_explanation 10, table_lookup 10, single_numeric 5, time_series_comparison 5다.

Judge가 본 대상은 retrieval unit 전체다. `judge._visible_candidate`와 `judge_payload`는 text, document_name, filing_period, section_path, receipt_date만 넣는다. rank, score, retriever, element id는 빠진다. 저장된 judgment 2,414건은 모두 `final`이 있고 unresolved는 0이다. agreement는 2,227 / 2,414 = 0.923이다.

`build_qrels`는 그 final을 `element_ids`마다 복사한다. v2 qrel은 5,751행이고 키에 element_id가 있다. test fully_answerable에서 양성 요소는 502개, 양성 retrieval unit은 271개다. 요소 등급이 unit 안에서 어긋난 경우는 0이다. 양성 요소 60개는 둘 이상의 unit에 등장한다.

Recall 분모는 `_recall_counts`의 요소 집합이다. Hit와 등급 부여는 요소 등급의 최댓값이다. unit 자신의 등급이 0이어도 공유 요소가 양성이면 hit relevance가 올라간다. v2 test 상위 10의 relevant hit 156개 중 21개가 이 경우였다.

NDCG headline은 condensed다. unjudged를 빼고, 이미 등장한 source element만 남은 hit는 gain 0이다. unjudged-as-zero는 raw 순위의 gain을 요소 중복 제거 없이 더한다. IDCG는 unit당 하나의 등급이다. test 상위 10 judged coverage는 1.0인데 NDCG@10은 0.466과 0.517로 달랐다. 55문항 중 10문항이 달랐고, 10문항 모두 상위 10이 전부 judged였다. 차이는 unjudged가 아니라 요소 집합 중복 제거와, 공유 요소로 물려받은 relevance다. zero-filled가 1을 넘는 문항이 4개 있었다. 같은 unit이 순위에 두 번 나온 경우와, 자신의 등급은 0인데 공유 요소 때문에 gain을 받은 경우가 있다.

dataset hash는 freeze body의 canonical JSON sha256이다. v2 값은 `d3635756a57c39d6d4ef8722f5f118348c00bbc034bf5963fb7720d184e6f137`이다. 이 세션 이후에도 같다.

## v2.1 split

fully_answerable 100개만 다시 배정한다. partial과 unanswerable의 split은 v2 그대로다. seed는 20261005다. 회사 캡은 test 11, dev 9다. wording과 query type은 55/100 Hamilton 목표와 정확히 같다. 작은 query type을 먼저 두고, 동점은 sha256(seed, query_id, split)으로 고른다. 그다음 회사 안에서 swap해 section, period, multi_evidence, evidence form을 맞춘다. swap은 wording과 query type의 L1을 바꾸지 않을 때만 한다.

검색 필드와 기존 split 값은 읽지 않는다. 같은 입력은 같은 배정이다.

결과:

- test wording: original_term 25, layperson 25, paraphrase 5
- dev wording: 20, 20, 5
- query type 아홉 가지는 양쪽 모두에 있다
- 표는 test 13, dev 12다. query type 목표를 더하면 표는 13이고, 표만 따로 Hamilton하면 14다. query type이 더 가는 stratum이라 13을 유지했다
- finance_table test 7, overview_paragraph test 6이다. 독립 목표는 8과 5다. 더 옮기면 wording 또는 query type이 움직인다
- period_sensitive test는 True 30, False 25로 Hamilton과 같다
- multi_evidence requirement가 둘 이상인 문항은 test 8, dev 10이다
- 배정이 바뀐 문항은 50개다. 질문 문장 변경은 0이다

difficulty_tags는 생성기가 붙인 주석이다. 표본 quota가 아니라 분할 축으로 쓰지 않았다. stratum_id가 문서·섹션 범주다.

## v2.1 qrel과 metric

judgment 2,414건을 재사용했다. 새 judge 호출은 0이다. judgments sha256 `c952fd4740bc971e8c0a8e2c742712c2acb9a615ddf35fa926d0a4d524f6af51`.

qrel은 requirement × retrieval unit이다. 3,112행. provenance 2,414행이 source element 목록을 가진다. 요소 수만큼 relevance를 복사하지 않는다. relevance 0은 저장한다. final이 없는 판정은 행을 만들지 않는다.

Recall 분모는 양성 unit이다. v2.1 test는 282다. macro는 문항 recall의 평균, micro는 unit을 모두 합친 비율이다. requirement recall의 분모는 fully_answerable requirement 수다. 한 requirement는 그 requirement로 양성인 unit이 창 안에 있으면 충족이다.

preferred는 seed unit이 relevance 1 이상이면 그 unit이다. seed가 모두 0이면 사업보고서, 그다음 최신 기간, 그다음 unit id로 하나를 고른다. preferred가 아니라고 무관한 것은 아니다.

NDCG gain은 `2^relevance - 1`이다. unit의 첫 등장만 gain을 받고 나중 복사는 그 순위에서 0이다. condensed는 unjudged를 건너뛰고 k개의 judged unit에서 멈춘다. unjudged-as-zero는 raw 1..k에서 unjudged를 0으로 둔다. raw 상위 k가 모두 judged면 두 값은 같다.

요소를 공유하는 양성 unit은 합치지 않는다. 각 unit을 judge가 따로 읽었기 때문이다. v2.1 test에서 그런 unit은 34개다.

dataset sha256 `2b1c372b7adaca29af831986bf5e6807c2ae2bd7835a52735690e227c1b03137`.

## Dense-D2a 재측정

제품 경로 `SearchCatalog` → `select_filings` → Dense-D2a를 그대로 실행했다. 검색 코드는 바꾸지 않았다. 본 지표는 v2.1 test 55개다. 양성 unit이 없어 빠진 문항은 0이다.

- Hit@1 19/55 (0.345), Hit@5 27/55 (0.491), Hit@10 32/55 (0.582)
- Recall@5 macro 0.246, micro 69/282 (0.245)
- Recall@10 macro 0.392, micro 116/282 (0.411)
- MRR 0.411
- NDCG@10 condensed 0.325, unjudged-as-zero 0.325
- preferred_hit@1 4/55, preferred_hit@10 21/55
- judged_coverage@10 1.0, @20 0.536
- unjudged mean @10 0, @20 9.27
- requirement recall@10 37/66, all-requirements-covered@10 29/55

test 55문항에서 두 NDCG가 다른 문항은 0이고, 1을 넘는 문항은 0이다.

wording Hit@10은 original_term 15/25, layperson 14/25, paraphrase 3/5다. n<5인 query type은 calculation, cross_filing_comparison, single_numeric, time_series_comparison이다.

unanswerable 10개 중 direct relevance 2는 0이다. 알테오젠 인과 문항 1개만 supporting 1이다. partial 10개는 본 지표 밖이다.

v2 test 문항을 unit metric으로만 다시 세면 Hit@10 37/55, Recall@10 micro 130/271, NDCG@10 0.434다. v2 공개값 40/55, 255/502, 0.466 / 0.517과 v2.1 본 지표 32/55의 차이는 분할, qrel 단위, metric 정의가 바뀐 결과다. 검색 성능의 향상이나 하락이 아니다.

dev Hit@10은 33/45, MRR 0.525, NDCG@10 0.478이다. 다음 BM25 비교는 이 dev만 쓴다.

## Tests

`uv run pytest -q` 264 passed. `uv run ruff check .` passed. 추가 테스트는 결정적 분할, 검색 결과 무시, wording 대표성, unit qrel, Recall macro/micro, requirement, unjudged와 irrelevant의 구분, NDCG 예제, preferred, dataset hash, v2 hash 불변이다.

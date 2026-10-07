# v2.1 DEV H10 BM25-primary Dense-head insertion

## 1. Rule

Rule `bm25_primary_dense_head`. 함수 `disclosure_agent.evalset_v2_1.asymmetric_fusion.bm25_primary_dense_head`.
문서 `docs/experiments/2026-10-07_h10_asymmetric_fusion_semantics.md` sha256 `71b87fd2a0569cf2d95552e1e47783f66cc092d8a938db2d8c5ab52ea9081f8e`.
if dense_head in BM25[1..10]: fused = BM25; else fused = BM25[1..9] + [dense_head] + BM25 tail without the head.
Dense rank 2..20은 정렬 입력이 아니다. 교집합 가산은 없다. 점수는 만들지 않는다.
BM25 순서가 바뀌는 조건은 Dense rank 1이 BM25 top-10 밖에 있는 것이다.
그때도 rank 1..9는 바뀌지 않고 rank 10만 head로 바뀐다.
dataset `multi_company_retrieval_v2_1` sha256 `2b1c372b7adaca29af831986bf5e6807c2ae2bd7835a52735690e227c1b03137`. 저장 DEV top-20을 재사용했다. retrieval 재실행 0. TEST 검색 0.

## 2. Pre-registered criteria

판정 트리는 점수 전에 고정했다.
Strong: Hit@10 >= 41, BM25-only 상실 <= 1, Dense-only Hit@10 유지, Recall@5 / Recall@10 / MRR / NDCG@10 하락 <= 0.05, requirement_recall >= 49, all_requirements >= 39.
Partial: Strong이 아니고, Dense-only 유지, Hit@10 > 40, BM25-only 상실 < 3.
그 밖은 Reject. unjudged는 irrelevant가 아니다.

## 3. Comparison

| metric | Dense | BM25 | Truncated | H10 |
| --- | --- | --- | --- | --- |
| Hit@1 | 20/45 | 29/45 | 29/45 | 29/45 |
| Hit@5 | 28/45 | 40/45 | 37/45 | 40/45 |
| Hit@10 | 33/45 | 42/45 | 40/45 | 43/45 |
| Recall@5 macro | 0.387 | 0.572 | 0.505 | 0.572 |
| Recall@10 macro | 0.561 | 0.767 | 0.725 | 0.787 |
| MRR | 0.525 | 0.742 | 0.737 | 0.744 |
| NDCG@10 condensed | 0.478 | 0.708 | 0.684 | 0.713 |
| NDCG@10 zero-filled | 0.478 | 0.708 | 0.675 | 0.713 |
| requirement_recall@10 | 41/58 | 51/58 | 49/58 | 52/58 |
| all_requirements_covered@10 | 32/45 | 41/45 | 39/45 | 42/45 |
| judged_coverage@10 | 1.000 | 1.000 | 0.969 | 1.000 |
| unjudged_count_mean@10 | 0.000 | 0.000 | 0.311 | 0.000 |

NDCG@10 zero-filled는 sensitivity다. headline NDCG는 condensed다.

BM25 대비 하락(BM25 - H10, 양수면 H10이 낮다): recall@5_macro +0.000000, recall@10_macro -0.019788, mrr -0.002222, ndcg@10_condensed -0.004135.

## 4. Edit census

질의 45. edit none 16, insertion 20, promotion 9.
편집된 질의의 Dense head 판정: {"irrelevant": 23, "relevant": 6}.
밀려난 BM25 rank 10 판정: {"irrelevant": 22, "relevant": 7}.

| query | company | edit | head BM25 rank | head judgment | displaced judgment | BM25 Hit@10 | H10 Hit@10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| V2-00107613-business_paragraph-02 | 이수페타시스 | insertion | None | relevant | irrelevant | True | True |
| V2-00107613-cross_period-02 | 이수페타시스 | promotion | 18 | irrelevant | irrelevant | False | False |
| V2-00107613-finance_table-01 | 이수페타시스 | promotion | 17 | irrelevant | irrelevant | True | True |
| V2-00107613-finance_table-03 | 이수페타시스 | insertion | None | irrelevant | irrelevant | False | False |
| V2-00107613-governance_paragraph-02 | 이수페타시스 | promotion | 17 | irrelevant | irrelevant | True | True |
| V2-00107613-mdna_paragraph-02 | 이수페타시스 | insertion | None | irrelevant | irrelevant | True | True |
| V2-00121941-business_paragraph-02 | 대상 | insertion | None | irrelevant | irrelevant | True | True |
| V2-00121941-business_paragraph-03 | 대상 | insertion | None | irrelevant | relevant | True | True |
| V2-00121941-cross_period-01 | 대상 | promotion | 14 | irrelevant | irrelevant | True | True |
| V2-00121941-finance_paragraph-02 | 대상 | promotion | 14 | relevant | relevant | True | True |
| V2-00121941-finance_table-02 | 대상 | promotion | 13 | irrelevant | irrelevant | True | True |
| V2-00121941-overview_paragraph-01 | 대상 | insertion | None | irrelevant | irrelevant | True | True |
| V2-00126380-finance_table-02 | 삼성전자 | insertion | None | relevant | irrelevant | False | True |
| V2-00126380-finance_table-03 | 삼성전자 | insertion | None | irrelevant | relevant | True | True |
| V2-00126380-governance_paragraph-01 | 삼성전자 | insertion | None | irrelevant | irrelevant | True | True |
| V2-00126380-mdna_paragraph-01 | 삼성전자 | insertion | None | irrelevant | irrelevant | True | True |
| V2-00126380-mdna_paragraph-02 | 삼성전자 | insertion | None | irrelevant | irrelevant | True | True |
| V2-00126955-business_paragraph-03 | 삼양식품 | insertion | None | relevant | irrelevant | True | True |
| V2-00126955-cross_period-01 | 삼양식품 | insertion | None | irrelevant | irrelevant | True | True |
| V2-00126955-cross_period-02 | 삼양식품 | insertion | None | irrelevant | irrelevant | True | True |
| V2-00126955-finance_paragraph-02 | 삼양식품 | insertion | None | irrelevant | irrelevant | True | True |
| V2-00126955-finance_table-01 | 삼양식품 | insertion | None | irrelevant | irrelevant | True | True |
| V2-00126955-mdna_paragraph-01 | 삼양식품 | insertion | None | relevant | relevant | True | True |
| V2-00989619-finance_paragraph-01 | 알테오젠 | promotion | 18 | irrelevant | relevant | True | True |
| V2-00989619-finance_table-02 | 알테오젠 | insertion | None | irrelevant | irrelevant | True | True |
| V2-00989619-finance_table-03 | 알테오젠 | insertion | None | irrelevant | relevant | True | True |
| V2-00989619-governance_paragraph-02 | 알테오젠 | promotion | 11 | irrelevant | irrelevant | True | True |
| V2-00989619-mdna_paragraph-02 | 알테오젠 | promotion | 18 | relevant | irrelevant | True | True |
| V2-00989619-overview_paragraph-01 | 알테오젠 | insertion | None | irrelevant | relevant | True | True |

## 5. Query-level transition

Hit@10 전이. other는 비교 대상이다.

BM25 대 truncated: {"bm25_hit__other_hit": 39, "bm25_hit__other_miss": 3, "bm25_miss__other_hit": 1, "bm25_miss__other_miss": 2}

BM25 대 H10: {"bm25_hit__other_hit": 42, "bm25_miss__other_hit": 1, "bm25_miss__other_miss": 2}

첫 양성 rank의 movement 대 BM25: {"neutral": 42, "recovered": 1, "still_missing": 2}.
rank change는 fused first-relevant rank에서 BM25 rank를 뺀 값이다. 양수면 뒤로 밀렸다.
질의별 행은 `per_query_transition.csv`에 있다.

## 6. BM25-only regression

BM25-only Hit@10 상실 0. both-hit Hit@10 상실 0. BM25 Hit@10 밖이었다가 H10에서 들어온 질의 1.

H9 잔여 3건. 복원 칸이 참이면 protected prefix 귀결이다.

| name | mechanism | BM25 | truncated | H10 | H10 Hit@10 | prefix entailment |
| --- | --- | --- | --- | --- | --- | --- |
| 대상 지분증권 | symmetric_rank_competition | 8 | 15 | 8 | True | True |
| 대상 특수관계자 | intersection_dominance_and_symmetric_competition | 7 | 14 | 7 | True | True |
| 알테오젠 우선주 | intersection_dominance | 4 | 11 | 4 | True | True |

BM25-only 상실: 0.

both-hit 상실: 0.

Hit@10 유입: 1.
- `V2-00126380-finance_table-02` 삼성전자: 삼성전자는 전분기에 삼성에스디에스와 구매 관련 거래를 얼마만큼 했나요? edit insertion, BM25 None, truncated 2, H10 10, head relevant, displaced irrelevant.

## 7. Dense-only semantic retention

`V2-00126380-finance_table-02` 삼성전자: 삼성전자는 전분기에 삼성에스디에스와 구매 관련 거래를 얼마만큼 했나요?
Dense 첫 양성 rank 1. BM25 첫 양성 rank None. truncated 첫 양성 rank 2. H10 Dense head rank 10. H10 첫 양성 rank 10.
edit insertion. head `85a98a4bddadb6d49eab49875653c980a2ce353fc30a8cda1fc63c8e1d09020e` 판정 relevant. BM25 Hit@10 False. truncated Hit@10 True. H10 Hit@10 True.
이 Hit@10은 head를 rank 10에 두는 배치의 귀결이다. truncated rank 2를 유지하는 기준이 아니다.

## 8. Requirement-level metrics

requirement_recall@10 52/58. all_requirements_covered@10 42/45.
BM25 대비 requirement 유입 1, 상실 0.
유입: V2-00126380-finance_table-02/R1.
상실: 없음.

| prior cell | prior | transitions | lost or still missing |
| --- | --- | --- | --- |
| both | 40 | `{"retained": 40}` | - |
| dense_only | 1 | `{"recovered": 1}` | - |
| bm25_only | 11 | `{"retained": 11}` | - |
| neither | 6 | `{"still_missing": 6}` | V2-00107613-cross_period-02/R1, V2-00107613-cross_period-02/R2, V2-00107613-finance_table-03/R1, V2-00107613-finance_table-03/R2, V2-00126955-cross_period-02/R1, V2-00126955-cross_period-02/R3 |

## 9. Judgment coverage

unjudged는 irrelevant로 넣지 않았다. zero-filled NDCG는 headline이 아니다.

| system | judged_coverage@10 | unjudged mean@10 | NDCG condensed | NDCG zero-filled |
| --- | --- | --- | --- | --- |
| dense | 1.000 | 0.000 | 0.478 | 0.478 |
| bm25 | 1.000 | 0.000 | 0.708 | 0.708 |
| truncated_rrf | 0.969 | 0.311 | 0.684 | 0.675 |
| h10 | 1.000 | 0.000 | 0.713 | 0.713 |

## 10. Representative cases

선택 규칙은 semantics 문서에 있다. pinned 4건, Hit@10이 BM25와 다른 질의, relevant BM25 rank 10을 민 질의다.

### `V2-00121941-business_paragraph-03` 대상

대상이 종속·관계·공동기업을 제외하고 보유한 지분증권 투자액은 2024년 말 장부상 얼마였나요?

reasons: pinned_residual, displaced_relevant_bm25_rank10. edit insertion. BM25 8 / truncated 15 / H10 8. head irrelevant. displaced relevant. Hit@10 BM25 True, truncated False, H10 True.

| rank | bm25 rank | dense head | judgment | unit |
| --- | --- | --- | --- | --- |
| 1 | 1 | False | irrelevant | `6a1ea80afb3ae5563db902d9a938a1297ef9193659117957fd435830d766e19d` |
| 2 | 2 | False | irrelevant | `17f9f373eb3c0bb32ce1f97d0011f10a2ad10eabbb7fc59ce18b5f5947db84f8` |
| 3 | 3 | False | irrelevant | `57742f45bcd93b5dae34e3b3ac7d69efbb1d40533f1a55644ac1f0dfc365932d` |
| 4 | 4 | False | irrelevant | `c06a074215c233b5e6316ca238470dc2b571491af559f4010feeccc8cf2f35da` |
| 5 | 5 | False | irrelevant | `80e7ab5b2e78d0ae9c6a2c7a10cc3e5c8fa2bcfa0351808b8786484a1b8b20d4` |
| 6 | 6 | False | irrelevant | `ec7acc4d54cfbeffb9b836dfbcf2ecca8f7577c70332c378daa1de94bec1c4b8` |
| 7 | 7 | False | irrelevant | `875fac4ae4d52552f3bfa558be2703f435e7bed97998e4d14e0b425a0421627e` |
| 8 | 8 | False | relevant | `6c3f2984ba262a6a78e6d1db1be108f7602d0999cdfad8177de0d12b9d9c840a` |
| 9 | 9 | False | relevant | `cc42b6ee74bc79a5f0bf1b8513eb58df5bf027acdd19ebbb4f7d80d5da5345b6` |
| 10 | None | True | irrelevant | `9ae4ddd7f9cd06b41a80e95c484f2efbbda879801bbeb18b7d49db7085c31b13` |

### `V2-00121941-finance_paragraph-02` 대상

대상이 대상네트웍스의 실수요영업부문을 인수하기로 정한 뒤 실제로 인수할 때까지 며칠이 걸렸나요?

reasons: displaced_relevant_bm25_rank10. edit promotion. BM25 1 / truncated 1 / H10 1. head relevant. displaced relevant. Hit@10 BM25 True, truncated True, H10 True.

| rank | bm25 rank | dense head | judgment | unit |
| --- | --- | --- | --- | --- |
| 1 | 1 | False | relevant | `2495d7fe7c0390a9305bec173bdfe548706a51ed978a3fe8e4754cdda5ec8e77` |
| 2 | 2 | False | relevant | `3d104d1cd7e9a6b5e208e6620696b71b17e6a87d863174197d018a19cf1ff4ea` |
| 3 | 3 | False | relevant | `44863c7e87042de87c9f232e03dd780d7b1e6762e3c3f90a25a9189b185f4745` |
| 4 | 4 | False | relevant | `fc42bff6c2b27a461d09c961a914698f1986e6bbf15be77256d170dcb4e31979` |
| 5 | 5 | False | relevant | `2baeb7fd6d1d6a0d4747bc2f8307f7618e4baa0df716b4237e94c0206a004eb9` |
| 6 | 6 | False | relevant | `10582cec8ff103adc93475dcba0e08e39b99660fb9eb839d9db73e87b88a0e76` |
| 7 | 7 | False | relevant | `c6dfa64449be9077c9d0124f4e7c7fc9d1a51fd37304858abada8bd8f6e80f6d` |
| 8 | 8 | False | relevant | `363f1346b5f66430688a4388a91fa752a71e3ddd0392d3afd97025f09ee47cfa` |
| 9 | 9 | False | relevant | `e3df0b35f8cdcbba20cceb5bb98b9481ba5cfc5bb69904ad5c2dc9d9242f71d9` |
| 10 | 14 | True | relevant | `b4b2c0034056b806f4dcea48d1793f658a1f05c750fa4146d78cbf40b3e2cbbc` |

### `V2-00121941-finance_table-02` 대상

대상은 2026년 1분기 말 특수관계자 거래로 받을 돈과 갚을 돈이 각각 얼마였어?

reasons: pinned_residual. edit promotion. BM25 7 / truncated 14 / H10 7. head irrelevant. displaced irrelevant. Hit@10 BM25 True, truncated False, H10 True.

| rank | bm25 rank | dense head | judgment | unit |
| --- | --- | --- | --- | --- |
| 1 | 1 | False | irrelevant | `ce4aa1a2ecfef29a73ca6e7711d45213615b28dc0b64d54886a17ac2d4ec5b6e` |
| 2 | 2 | False | irrelevant | `bdf51f0ec4f8e892fb94d0e92c2dfaaad5c9f92110966f47d357f9b2e1d73f30` |
| 3 | 3 | False | irrelevant | `5359dc99efca0728ce3c8092d17f217c3050df2696f81473445442cdfb678072` |
| 4 | 4 | False | irrelevant | `b4c9e1d9609513d31593780f0223413957a39d9c157f47ac67c3177db7580808` |
| 5 | 5 | False | irrelevant | `6ee9668f0560d30b1a2c560c0f2bc8f9cbc796aff89062453d8717e00e31b99d` |
| 6 | 6 | False | irrelevant | `bb322b934e9f006aa29cc7a3fc12ace9bf309508cd8b9307ab577cf185cd65c7` |
| 7 | 7 | False | relevant | `62c8564da1be79a9d8b5bad6f32b3786894d1015029ad67477f292ac03b00a86` |
| 8 | 8 | False | irrelevant | `bab2a7632a01b7fffd03e46f82382b3c0f0d5cfb5d98bab0abd2d122cc0b3b75` |
| 9 | 9 | False | irrelevant | `50f41b922e563b44f29430b07363eaa1a4cf108f61be8cffbbd4c2752c83e9c5` |
| 10 | 13 | True | irrelevant | `a90898b290d2038bf39dbb9fef42ae7aa35edeae6699051ba018616098620a2a` |

### `V2-00126380-finance_table-02` 삼성전자

삼성전자는 전분기에 삼성에스디에스와 구매 관련 거래를 얼마만큼 했나요?

reasons: pinned_dense_only, hit10_changed_vs_bm25. edit insertion. BM25 None / truncated 2 / H10 10. head relevant. displaced irrelevant. Hit@10 BM25 False, truncated True, H10 True.

| rank | bm25 rank | dense head | judgment | unit |
| --- | --- | --- | --- | --- |
| 1 | 1 | False | irrelevant | `976ae60e70fdda6d21d17e641a31b0b3c1d25c0c9c83b4271e209d4971760817` |
| 2 | 2 | False | irrelevant | `65d03cbcba25bdbef5965d8fbfcde90e71b85df290117f4ea75e0c8ad69d02c9` |
| 3 | 3 | False | irrelevant | `17de3d61309677df80c10fbca09093cef729927094836d47b975e73fe5c01503` |
| 4 | 4 | False | irrelevant | `eba319a27df04c65bffddc9b4bcc73be80474833452e19a37985cc01871b211c` |
| 5 | 5 | False | irrelevant | `f6949865b201306d18ab0b87384ad8ea5871d25440ad9e7b7365dc24fa42a561` |
| 6 | 6 | False | irrelevant | `0755e0d3a753de8e756d73063aa1dc162b13671028c7a13745956201b4dae9df` |
| 7 | 7 | False | irrelevant | `9fb4087b1abb706f4c38d61dee0ffab0924904e158c745ef2c3388362f72402a` |
| 8 | 8 | False | irrelevant | `4d991b46afc537f32ca053b6dde5c7d53bc6c5db378eaccc32c2b3a8b22323f9` |
| 9 | 9 | False | irrelevant | `f655f8bace5156950441a0b9a0598a7db70769064e5b881482f53ebcaca16a80` |
| 10 | None | True | relevant | `85a98a4bddadb6d49eab49875653c980a2ce353fc30a8cda1fc63c8e1d09020e` |

### `V2-00126380-finance_table-03` 삼성전자

삼성전자의 2025년 말 자본금은 우선주와 보통주가 각각 어느 정도 비중을 차지하나요?

reasons: displaced_relevant_bm25_rank10. edit insertion. BM25 1 / truncated 2 / H10 1. head irrelevant. displaced relevant. Hit@10 BM25 True, truncated True, H10 True.

| rank | bm25 rank | dense head | judgment | unit |
| --- | --- | --- | --- | --- |
| 1 | 1 | False | relevant | `78cde9af4a9c40ba09d5f7e4d03134fbce88f433d0b1fcca82b49fee14b33881` |
| 2 | 2 | False | irrelevant | `7843d53d8aaa84e2c634a3916f047e9077646bd4b5d676a900925a7c81ba571c` |
| 3 | 3 | False | irrelevant | `1943af6d7e64ab521354e5648bb54e254230215c460120c18c688a65efa1d6a9` |
| 4 | 4 | False | irrelevant | `00d91fee9e11e8a154c552c9033c3c053a28b7cdc588cf871512663e82e61303` |
| 5 | 5 | False | relevant | `777475fc47a37a5862baa33be44b0f46cf17df7a721c06f211fb464c9aa87d79` |
| 6 | 6 | False | irrelevant | `f264b87e9d22876754c8576855d434b2962ffc294c4ec2a0760186ff6b7ed046` |
| 7 | 7 | False | relevant | `a2dc7664ad4f6b3cfab19e7ab326b4ff9d25f71f4fbc435ea080b7bf2549c2bc` |
| 8 | 8 | False | irrelevant | `ccb8f12f3992ace7dad34c0b7f072b199e056b19a0d8a24c836e225f12bc11de` |
| 9 | 9 | False | irrelevant | `e2a22dc5bf8600522163df4509d0ca79329f925e34e6e5306ea014e4e37a9d64` |
| 10 | None | True | irrelevant | `751d82665dd52a7f7a2b63dc44ede4b9e2873146ab445a4f510a86d3fe966da9` |

### `V2-00126955-mdna_paragraph-01` 삼양식품

삼양식품은 왜 환율변동위험, 특히 달러화 및 엔화 관련 위험에 노출되어 있나요?

reasons: displaced_relevant_bm25_rank10. edit insertion. BM25 1 / truncated 1 / H10 1. head relevant. displaced relevant. Hit@10 BM25 True, truncated True, H10 True.

| rank | bm25 rank | dense head | judgment | unit |
| --- | --- | --- | --- | --- |
| 1 | 1 | False | relevant | `683ee73e6d735446065ae1f511c8eb2df027809c3dc502e607a34974bd6acff1` |
| 2 | 2 | False | relevant | `cba78735880069e6b5c27608ac3e3a539b4608069a70768dad8eee3e118fc2dc` |
| 3 | 3 | False | relevant | `791ae2bbdd4c7be7f2b0efa80c9a2a92f67aa4421e0774dd8f07ae2d51a0960d` |
| 4 | 4 | False | relevant | `a0e5f17cbd066d796ac787d29566a0c21c95d1ec4243f38f8eb846f65a51bebb` |
| 5 | 5 | False | relevant | `9af42a0030d74e5405a94cac550323300b8b132ba8b35f327f3d11ce8b9d0204` |
| 6 | 6 | False | relevant | `cb0d33390e932ba8e9ffc5376abb443f92ab05aaffbda32d02888d10275a4043` |
| 7 | 7 | False | relevant | `01e27ccf65c0d11311bf585ad0347c0b4be198749f78b6d8d6e4671b3461a4fe` |
| 8 | 8 | False | irrelevant | `1189072f56534072cf4440320c85a615e90a69322b96ed7f64e5099b2a49453a` |
| 9 | 9 | False | relevant | `7d358d1d9e37fbd4a1bd2dd357487e72a8e54e75cc3dfe0e4872e9d9406b7ee5` |
| 10 | None | True | relevant | `acfe52a397168fa8460d5b03cadd51fdd460f360889c087b456366d7191b8450` |

### `V2-00989619-finance_paragraph-01` 알테오젠

알테오젠은 제1118호를 언제부터 의무 적용해야 하나요?

reasons: displaced_relevant_bm25_rank10. edit promotion. BM25 1 / truncated 1 / H10 1. head irrelevant. displaced relevant. Hit@10 BM25 True, truncated True, H10 True.

| rank | bm25 rank | dense head | judgment | unit |
| --- | --- | --- | --- | --- |
| 1 | 1 | False | relevant | `3ce2b3b972407efa7e5d7b3c0c5ab0f250957ef41032be35f414a4daec11c0dc` |
| 2 | 2 | False | relevant | `38b9be90301287d6f51ae19d88680083c8b66beff20065291d55957828c15a13` |
| 3 | 3 | False | irrelevant | `6206bbf5fec9cf9a4a9d3d42c3e4bafcb6211779558b317a08d4968640c06ba2` |
| 4 | 4 | False | relevant | `2a6f84cc231fe7634d34851cabd3dacccc26a6d3f76d2cd2bb1feccacc6eeab5` |
| 5 | 5 | False | relevant | `aa4c64a447701c63dbcd5e71091f414abe26aefcdd5574aa0e67675b20dd140e` |
| 6 | 6 | False | irrelevant | `fcd65bc51d6beff0f08c417b23d28301f74099902703dc39422b36d7fa2a2f14` |
| 7 | 7 | False | irrelevant | `d00a0a093a5370dfa5fc725be031a6fc2d4b5e1f68a450d6830166ec3785b8b3` |
| 8 | 8 | False | irrelevant | `d43534c5d4d0205cac39f3d44bd02e7d12ef9cfdf8e977999a07490744e989f5` |
| 9 | 9 | False | relevant | `02f3f749fc906c52aa5f0560f7b01dc004a505d21a074bc5830494dc3df60c7d` |
| 10 | 18 | True | irrelevant | `9fbe0a32b38c9d704d63eb8e7f684dd3dd24cd0d3418ef3570f4211c3553177c` |

### `V2-00989619-finance_table-03` 알테오젠

알테오젠의 전기 외화환산이익 중 금융자산과 금융부채에서 발생한 금액은 각각 전체의 몇 퍼센트인가요?

reasons: displaced_relevant_bm25_rank10. edit insertion. BM25 1 / truncated 1 / H10 1. head irrelevant. displaced relevant. Hit@10 BM25 True, truncated True, H10 True.

| rank | bm25 rank | dense head | judgment | unit |
| --- | --- | --- | --- | --- |
| 1 | 1 | False | relevant | `899ede79b6083f2651305070d0a9a3afb7f0e2a1df91c5cda434b9024cf6f898` |
| 2 | 2 | False | relevant | `c1d8b08e73206354e6d986c9dd02c08ce12528cb2163d188d25039ad029c2931` |
| 3 | 3 | False | relevant | `9c4fd363e1f4f5aa2947ed45cc6f0707c7c6f6ef1a394d4a904fb16073f1582f` |
| 4 | 4 | False | irrelevant | `b6811c47ca7985f0465d1d75090be9d889d2e8df5f58ea1b5c5725978d92cfd2` |
| 5 | 5 | False | relevant | `658427dc40803fa8164c60073f34ee0e591894f10c9a50cde815b712d1810752` |
| 6 | 6 | False | irrelevant | `786b82f35e4c94e720a751e8336c4b77399559cb7445bbf08482992d93d25a60` |
| 7 | 7 | False | irrelevant | `632f66d613bc3a431e89c62c3854b4624ff22027692a75e7f4c403e0e7093b87` |
| 8 | 8 | False | irrelevant | `55de3ee14b68195846301d87c8bb6acf8df31d7c6f14919048d4c3c7eafb3a6d` |
| 9 | 9 | False | irrelevant | `323bbef1e54725061873e7f154167b27e4dad76812d2d61ae82c822d8364661f` |
| 10 | None | True | irrelevant | `7a96097584f6cca2a6e615e421b18d0d53c07b639a0756ac4baf4ea0c4a4a16f` |

### `V2-00989619-governance_paragraph-02` 알테오젠

알테오젠은 이번 분기에 우선주 몇 주를 보통주로 바꿨나요?

reasons: pinned_residual. edit promotion. BM25 4 / truncated 11 / H10 4. head irrelevant. displaced irrelevant. Hit@10 BM25 True, truncated False, H10 True.

| rank | bm25 rank | dense head | judgment | unit |
| --- | --- | --- | --- | --- |
| 1 | 1 | False | irrelevant | `0bb7635affe4315f5c455ba6945253bdd9075a11f45a92777a2df96bdf2dafb3` |
| 2 | 2 | False | irrelevant | `e15cd5f307cf5cec0e0b13e226fd12fe26e286277601e2b08c584cf095853d79` |
| 3 | 3 | False | irrelevant | `f14f61af96edcf2ff164372d2f455d14974a21fac76d84428bb92b658c57d0ee` |
| 4 | 4 | False | relevant | `ea7512ea104a8b27f90c11ac016a9e18e87550993881959f7426797781fc075f` |
| 5 | 5 | False | irrelevant | `fa4fd0c27a8055bae133f50aabd0f670c995486a5e7babda7b2e6ad7ffe2016d` |
| 6 | 6 | False | irrelevant | `fe34f16c22a1f6f7f87190e5149a4d2094a3709bdd3bc54d3a7e23ac75623e9e` |
| 7 | 7 | False | irrelevant | `75248bd40542d8bc36bced4ad9e7b21cd06f3ee1f0d6e6867f31dc559af50ed0` |
| 8 | 8 | False | irrelevant | `213c4a193e419b9b7e4ce1bd974ded7dde9794f154a666649715c140c444c1bd` |
| 9 | 9 | False | relevant | `59bf1caeda932865f4ecd39c6dd7237fdd9aa475d85499d2ad4b2b1005912c4d` |
| 10 | 11 | True | irrelevant | `fabd7f06134a801dd16e219ca0998b9102eafe61013fb5ee6ca9fcdc01b5bc21` |

### `V2-00989619-overview_paragraph-01` 알테오젠

알테오젠의 종류주식 발행현황에 기재된 주식은 어떤 종류인가요?

reasons: displaced_relevant_bm25_rank10. edit insertion. BM25 1 / truncated 1 / H10 1. head irrelevant. displaced relevant. Hit@10 BM25 True, truncated True, H10 True.

| rank | bm25 rank | dense head | judgment | unit |
| --- | --- | --- | --- | --- |
| 1 | 1 | False | relevant | `2c5406010146ca68f03acf81022a9e6806cba3660ed48ed9a5a25b15536d4bb3` |
| 2 | 2 | False | relevant | `530d9d1a7fa205d252ec7187452b4e050a0c89333fc02083965f0bfcff311829` |
| 3 | 3 | False | relevant | `14a71e25cf9c71f56b1668c539e3277cf1b2e36a54545a2355cc496125b38feb` |
| 4 | 4 | False | relevant | `bc667cdd22d8f0dc9e608e684b8cacf3231dd085ee9b6675f6c82f47733c9f78` |
| 5 | 5 | False | relevant | `cd7d4fe7a66cfe28a437e36ed837305ff96b086e77e9936c9d66b05dae0d7a71` |
| 6 | 6 | False | relevant | `3e686a33203af4c18ea214d6d6a7add68b3779945b1da53ff2fe44dc9241f690` |
| 7 | 7 | False | relevant | `c7c651f98bb1da5aae9866af37ace282461bdcad62f267914f28fcd40fd9e45e` |
| 8 | 8 | False | relevant | `6d392f4d3a5fe70d19bd440053060ff71934366065dd2edc7215deebbceb9570` |
| 9 | 9 | False | relevant | `a115ebf3a10cbed8bbd381a41c60c56658bc17877b5e187c78b3fe6d7c52a23f` |
| 10 | None | True | irrelevant | `59083ba9e15c8fba848583695ab3abf22ccb7fe65ae548b2a163d1caac8bcbe2` |

## FACT

H10 Hit@10은 43/45이다. BM25는 42/45, truncated는 40/45, Dense는 33/45이다.
edit none 16, insertion 20, promotion 9.
BM25-only 상실 0. both-hit 상실 0. Hit@10 유입 1.
편집된 질의의 Dense head 판정 {"irrelevant": 23, "relevant": 6}. 밀려난 BM25 rank 10 판정 {"irrelevant": 22, "relevant": 7}. relevant rank 10이 밀려도 Hit@10이 유지된 질의는 rank 1..9에 다른 양성이 남은 경우다.
Dense-only `V2-00126380-finance_table-02` H10 Hit@10은 True이고 head rank는 10이다.
세 residual의 H10 첫 양성 rank는 BM25 rank와 같고, 그 일치는 protected prefix 귀결로 검증했다.
requirement_recall@10 52/58, all_requirements_covered@10 42/45.
하락: recall@5_macro +0.000000, recall@10_macro -0.019788, mrr -0.002222, ndcg@10_condensed -0.004135.

## INTERPRETATION

세 residual이 Hit@10으로 돌아온 것은 이 rule이 대칭 점수 경쟁을 하지 않기 때문이다. 그 복원 자체는 DEV에서 새로 추정한 효과 크기가 아니다. BM25 rank 8, 7, 4는 prefix 안에 있고 prefix는 동결된다.
열린 검증은 marginal slot이다. Dense head가 BM25 top-10 밖인 질의에서 rank 10 교체가 lexical Hit@10과 requirement를 얼마나 남기는지다. relevant BM25 rank 10이 창 밖으로 나간 질의는 Hit@10 실패가 아니라 redundant coverage의 비용이다.
판정은 `Strong Support`이고 reason은 `all_strong_floors`이다. Strong 체크: {"hit10": true, "bm25_only_lost": true, "dense_only": true, "recall5": true, "recall10": true, "mrr": true, "ndcg": true, "requirement_recall": true, "all_requirements": true}. Partial 체크: {"dense_only": true, "hit10_above_truncated": true, "fewer_bm25_only_losses_than_truncated": true}.
이 결과는 TEST를 열지 않는다. Reject이면 BM25가 strongest baseline으로 남고 fusion 실험을 여기서 멈출 수 있다. Strong이어도 같은 봉인을 유지한다.

## Verdict

Strong Support. reason `all_strong_floors`. Strong when every pre-registered floor holds. Partial when Dense-only Hit@10 remains, Hit@10 exceeds truncated 40, and BM25-only losses are fewer than 3. Otherwise Reject.

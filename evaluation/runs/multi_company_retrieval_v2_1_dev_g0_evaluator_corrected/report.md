# G0 evaluator-corrected

## 범위

기존 G0 generation 45건과 기존 judge 출력을 재사용했다. Generator prompt, retrieval, qrel, TEST는 그대로 두었다. 모델은 다시 호출하지 않았다.
원본 run `multi_company_retrieval_v2_1_dev_g0_grounded_generation`. 수정본 `multi_company_retrieval_v2_1_dev_g0_evaluator_corrected`.

## Raw 와 corrected

| metric | raw | corrected |
| --- | ---: | ---: |
| Faithfulness | 73/75 | 66/67 |
| Requirement coverage | 50/58 | 51/58 |
| Citation correctness | 71/73 | 70/72 |
| Citation completeness | 68/75 | 67/67 |
| Requirement success | 49/58 | 50/58 |

Corrected faithfulness 66/67. Coverage 51/58. Citation correctness 70/72. Citation completeness 67/67.
Integrity errors 0.

Faithfulness 분모는 검증 가능한 factual claim이다. Abstention 7건, limitation 1건, meta 0건을 그 분모 밖에 두었다. Claim type unjudged 0건.
Citation completeness 분모는 citation이 필요한 factual claim이다.
Citation correctness의 기존 pair verdict는 유지했다. Factual claim이 아닌 pair만 분모에서 뺐다.

## Retrieval-limited 와 generation-limited

Requirement primary: success 50, retrieval-limited 6, generation-limited 2, unjudged 0.
관련 qrel unit이 Top-10에 없으면 retrieval-limited다. 있으면 답변의 충족과 factual claim의 지지로 generation을 본다.

## 판정이 바뀐 claim

- `V2-00107613-finance_table-03` K1 abstention raw=supported appropriateness=appropriate: 제공된 공시 근거만으로는 이수페타시스의 2026년 상반기 영업권 이외 무형자산 취득 금액을 확인할 수 없습니다.
- `V2-00107613-finance_table-03` K2 abstention raw=supported appropriateness=appropriate: 제공된 공시 근거만으로는 이수페타시스의 2026년 상반기 투자부동산 취득 금액을 확인할 수 없습니다.
- `V2-00107613-cross_period-02` K1 abstention raw=supported appropriateness=appropriate: 제공된 근거만으로는 2024년 반기보고서의 투자조합 지분율, 순자산지분금액 및 장부금액을 확인할 수 없습니다.
- `V2-00107613-cross_period-02` K2 abstention raw=supported appropriateness=appropriate: 제공된 근거만으로는 2024년 3분기보고서의 받을 돈 연체 현황과 금액을 확인할 수 없습니다.
- `V2-00126955-cross_period-02` K2 abstention raw=supported appropriateness=appropriate: 제공된 근거로는 2026년 1분기 말 단기차입금 잔액을 확인할 수 없다.
- `V2-00126955-cross_period-02` K3 abstention raw=supported appropriateness=appropriate: 제공된 근거로는 반기보고서에 기재된 전기말 대출 약정 한도를 약정 유형별로 확인할 수 없다.
- `V2-00989619-finance_table-02` K1 abstention raw=unsupported appropriateness=unjudged: 제공된 분기보고서 근거에는 기술지원·위탁생산 용역기간이 확인되지 않습니다.
- `V2-00989619-finance_table-02` K3 limitation raw=supported appropriateness=-: 사업보고서에 기재된 기간을 분기보고서의 기간으로 볼 수는 없습니다.

## 판정이 바뀐 requirement

- `V2-00107613-finance_table-03` R1 coverage abstained -> abstained, support supported -> not_factual, primary retrieval_limited -> retrieval_limited, flags retrieval_failure,citation_missing -> retrieval_failure
- `V2-00107613-finance_table-03` R2 coverage abstained -> abstained, support supported -> not_factual, primary retrieval_limited -> retrieval_limited, flags retrieval_failure,citation_missing -> retrieval_failure
- `V2-00107613-cross_period-02` R1 coverage abstained -> abstained, support supported -> not_factual, primary retrieval_limited -> retrieval_limited, flags retrieval_failure,citation_missing -> retrieval_failure
- `V2-00107613-cross_period-02` R2 coverage abstained -> abstained, support supported -> not_factual, primary retrieval_limited -> retrieval_limited, flags retrieval_failure,citation_missing -> retrieval_failure
- `V2-00126380-finance_table-03` R1 coverage abstained -> covered, support supported -> supported, primary generation_limited -> success, flags over_abstention -> -
- `V2-00126955-cross_period-02` R1 coverage abstained -> abstained, support supported -> not_factual, primary retrieval_limited -> retrieval_limited, flags retrieval_failure,period_scope_confusion,citation_missing -> retrieval_failure
- `V2-00126955-cross_period-02` R3 coverage abstained -> abstained, support supported -> not_factual, primary retrieval_limited -> retrieval_limited, flags retrieval_failure,period_scope_confusion,citation_missing -> retrieval_failure
- `V2-00989619-finance_table-02` R1 coverage abstained -> abstained, support unsupported -> not_factual, primary generation_limited -> generation_limited, flags over_abstention,period_scope_confusion,citation_missing -> over_abstention

## Coverage 재검토

### `V2-00126380-finance_table-03` R1

답변은 2025년 말 발행주식 액면총액 기준 보통주 약 87.9%, 우선주 약 12.1%를 말하고 C5를 붙였다. K1과 K2는 supported이고 citation도 supports다. K3는 납입자본금 897,514백만원이 종류별로 나뉘어 있지 않아 비중 기준이 액면총액이라는 한정이다. Raw judge는 K3만 연결하고, 한정이 있다는 이유로 abstained로 두었다. 요구한 비중은 답변에 있다. 값의 참과 근거의 지지는 이미 supported다. Coverage를 covered로 고친다.

Requirement 충족. Judge 오류. Coverage abstained -> covered.

### `V2-00989619-finance_table-02` R1

요구는 분기보고서에 나온 기술지원·위탁생산 용역기간이다. 답변은 제공된 분기보고서 근거에서 그 기간이 확인되지 않는다고 하고, 2025년 사업보고서의 기간을 적은 뒤 그 기간을 분기보고서 기간으로 보지 않는다. 선호 분기보고서 unit은 저장된 Top-10에 없다. 관련 사업보고서 unit은 C5로 Top-10에 있다. 요구한 분기보고서 기간은 답으로 제시되지 않았다. abstained는 그 기권과 같다. Coverage는 유지한다.

Requirement 미충족. Judge 판정 유지. Coverage abstained -> abstained.

## 남아 있는 generation failure

- `V2-00126380-finance_table-02` R1 generation_limited coverage=covered support=misinterpretation flags=evidence_misinterpretation,period_scope_confusion,citation_mismatch: 삼성전자는 전분기에 삼성에스디에스와 구매 관련 거래를 얼마만큼 했나요?
- `V2-00989619-finance_table-02` R1 generation_limited coverage=abstained support=not_factual flags=over_abstention: 알테오젠 분기보고서에서 기술 지원과 위탁생산 용역 기간은 언제부터 언제까지로 나와 있어?

Citation mismatch로 남은 factual pair는 아래다. 기존 primary 규칙은 이 pair만으로 success를 generation-limited로 바꾸지 않는다.

- `V2-00107613-business_paragraph-02` K2 C1 does_not_support: 엄격한 품질 기준과 장기 신뢰성 요구는 언급하지만, 오작동이 치명적인 문제나 큰 사고로 이어진다는 내용은 없습니다.
- `V2-00126380-finance_table-02` K1 C10 does_not_support: 532,755백만원은 전체 특수관계자 항목의 금액이며, 삼성에스디에스㈜ 열의 매입 등 금액은 313,853백만원입니다.

## Abstention 적절성

| name | count |
| --- | ---: |
| appropriate | 6 |
| inappropriate | 0 |
| unjudged | 1 |
| total | 7 |

V2-00989619-finance_table-02 K1은 제공된 분기보고서 근거에 대한 기권이다. Raw judge는 그 문장을 문서 부재 주장으로 읽고 unsupported로 두었다. 그 판정을 적절한 기권으로 바꾸지 않았고, faithfulness 분모에도 다시 넣지 않았다. 적절성은 unjudged로 남아 있다.

## 평가 한계

유형은 문장 규칙으로 정했다. 새 judge는 호출하지 않았다. K1 알테오젠 기권의 적절성은 unjudged다. Citation correctness의 pair verdict는 다시 판단하지 않았다. Coverage 수정은 두 사례의 기록된 재검토다. Qrel과 retrieval policy는 바꾸지 않았다.

## Failure taxonomy

| name | count |
| --- | ---: |
| retrieval_failure | 6 |
| evidence_coverage_failure | 1 |
| generation_omission | 0 |
| unsupported_claim | 0 |
| evidence_misinterpretation | 1 |
| period_scope_confusion | 1 |
| citation_missing | 0 |
| citation_mismatch | 2 |
| citation_hallucination | 0 |
| over_abstention | 1 |
| under_abstention | 0 |

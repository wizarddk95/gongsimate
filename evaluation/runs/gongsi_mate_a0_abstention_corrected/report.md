# A0 abstention contract correction

Source `gongsi_mate_a0_agent_baseline`. Judge recalled 0.
Generator factual flags were not read.

## Task

| outcome | raw | corrected |
| --- | ---: | ---: |
| success | 11 | 11 |
| partial_success | 0 | 1 |
| appropriate_clarification | 5 | 5 |
| appropriate_abstention | 0 | 2 |
| failure | 14 | 11 |
| unjudged | 0 | 0 |

Faithfulness denominator 15 -> 10. Numerator 9.
Abstention appropriateness 3/5.

## Outcome changes

- `V2-00126955-finance_table-01`: failure/unsupported_claim -> failure/over_abstention.
- `V2-00121941-partial-02`: failure/unsupported_claim -> partial_success/None.
- `V2-00107613-cross_period-02`: failure/unsupported_claim -> failure/over_abstention.
- `A0-abstain-01`: failure/unsupported_claim -> appropriate_abstention/None.
- `A0-abstain-02`: failure/unsupported_claim -> appropriate_abstention/None.

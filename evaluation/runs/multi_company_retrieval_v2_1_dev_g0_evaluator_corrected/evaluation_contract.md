# G0 evaluation contract v1

이 계약은 저장된 G0 generation과 저장된 judge 출력을 다시 집계한다.
Generator prompt, generation model, retrieval, qrel, TEST는 이 계약의 입력이 아니다.

## 발견한 오류

Raw faithfulness와 citation completeness는 judge verdict가
`supported`, `unsupported`, `misinterpretation`이면 그 claim을 factual claim으로 세었다.
Generator가 `factual: false`로 둔 확인 불가 문장도 judge가 `supported`로 두면 분모에 들어갔다.
Citation이 없는 그 문장이 citation completeness의 미충족으로 집계됐다.

Requirement coverage는 요구한 수치가 답변에 있어도, 기준을 한정하는 문장이 함께 있으면
`abstained`가 될 수 있었다.

## Claim type

한 claim은 다음 중 하나다.

- `factual_claim`: evidence로 확인할 수 있는 사실 주장. 문서 전체에 대한 부재 단정도 여기 둔다.
- `abstention`: 제공된 evidence 범위에서 확인할 수 없다는 진술.
- `limitation`: 다른 사실의 해석 범위를 제한하는 문장.
- `meta`: 답변 절차를 안내하는 문장.
- `unjudged`: 위 유형이 한 문장에서 겹치거나 분류가 닫히지 않은 문장.

Generator의 `factual` 값은 기록만 하고 유형을 정하지 않는다.

분류 규칙:

1. `제공된 근거` 또는 `입력된 근거`가 확인 불가와 함께 있고,
   문서 전체 부재나 범위 제한이 없으면 `abstention`.
2. `A를 B로 볼 수 없다`가 주된 행위이면 `limitation`.
3. `분기보고서에는 해당 정보가 없다`처럼 문서 자체의 부재를 말하면 `factual_claim`.
4. 1과 2, 또는 1과 3이 한 문장에 있으면 `unjudged`.
5. 나머지 검증 가능한 주장은 `factual_claim`.

`unjudged`는 faithfulness와 citation completeness의 점수 분모에 넣지 않고, 건수를 별도로 남긴다.

## Faithfulness

분모는 `factual_claim`이면서 raw verdict가
`supported`, `unsupported`, `misinterpretation`인 claim이다.
Abstention, limitation, meta는 이 분모 밖에 둔다.

Abstention의 적절성은 별도다. Raw verdict가 `supported`이면 `appropriate`다.
그 외 verdict는 `unjudged`로 남긴다. 문서 부재로 읽힌 판정을 적절한 기권으로 바꾸지 않고,
faithfulness 오답으로 다시 넣지도 않는다.

## Citation completeness

분모는 citation이 필요한 `factual_claim`이다.
이 데이터에서 그 claim은 공시에서 확인하는 사실이라 citation이 필요하다.
Abstention, limitation, meta에는 citation을 요구하지 않는다.
`unjudged` claim의 citation 필요는 `unjudged`로 남긴다.

## Citation correctness

저장된 pair verdict는 유지한다.
`factual_claim`이 아닌 claim의 pair는 수정 분모에서 빼고, verdict와 제외 이유를 남긴다.

## Requirement coverage

Coverage는 답변이 요구한 내용을 말하는지를 본다.
수치의 참과 evidence 충실성은 faithfulness와 citation이 본다.
요구한 내용이 있고 한정이 함께 있다는 이유만으로 `abstained`로 두지 않는다.

적용한 재검토:

- `V2-00126380-finance_table-03` R1은 보통주 약 87.9%, 우선주 약 12.1%를 말했고 C5가 있다.
  Raw judge는 납입자본금이 종류별로 없다는 한정만 보고 `abstained`로 두었다.
  요구한 비중은 답변에 있다. Coverage를 `covered`로 고친다. 이 requirement는 충족되었다.
- `V2-00989619-finance_table-02` R1은 분기보고서 기간을 말하지 않았다.
  사업보고서 기간을 제시한 뒤 그 기간을 분기보고서 기간으로 보지 않았다.
  선호 분기보고서 unit은 Top-10에 없고, 관련 사업보고서 unit은 있다.
  `abstained`는 답변의 기권과 같다. Coverage는 유지한다. 이 requirement는 충족되지 않았다.

## 유지한 규칙

Retrieval-limited와 generation-limited의 분리는 유지한다.
관련 qrel unit이 Top-10에 없으면 retrieval-limited다.
Citation mismatch가 success requirement의 primary를 바꾸던 규칙은 유지한다.
Qrel, H10 policy, generator prompt는 유지한다.

# A0 evaluation contract

이 계약은 제품 Agent의 이번 실행만 집계한다.
G0 generator 출력, H10 ranking, TEST 55는 입력이 아니다.

검색 필요 여부는 tool call로 본다.
최종 과제 성공은 허용된 행동과 requirement 판정으로 본다.
불필요 검색은 task가 맞아도 secondary flag로 남긴다.

Faithfulness는 이번 실행에서 생성기에 들어간 evidence만 본다.
Requirement 문장은 faithfulness 입력에 넣지 않는다.
Citation integrity는 citation의 evidence_id가 그 evidence 안에 있는지로 본다.
보류 답변과 검색 없는 답변에는 citation을 요구하지 않는다.

Claim type은 문장으로 정한다. Generator의 factual 표시는 쓰지 않는다.
G0의 factual claim과 abstention 구분을 먼저 적용한다.
그 다음에도 사실 주장으로 남은 문장만 다음으로 가른다.

- 제공된 검색 결과에서 확인하지 못했다는 문장은 abstention이다.
- 지원하거나 수집한 범위 밖이라는 문장은 limitation이다.
- 도구가 실행되지 않았다는 문장은 abstention이다.
- 아직 오지 않은 확정 사실을 거절하는 문장은 abstention이다.
- 문서 자체에 그 정보가 없다는 단정은 factual claim으로 남긴다.

Abstention, limitation, meta는 Faithfulness 분모 밖이다.
적절한 보류를 Faithfulness 오류로 세지 않는다.

점수는 G0 requirement 58개 점수와 비교하지 않는다.

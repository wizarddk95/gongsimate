# isu_petasys_v0_3

v0.1과 v0.2의 평가셋 파일, 그리고 그 실행 결과는 수정하지 않는다.

## 왜 dataset version을 올렸는가

정답으로 인정하는 gold span은 v0.2와 같다. `questions.json`과 `claims.json`은 v0.2를 그대로 복사했다. chunker도 바꾸지 않는다.

바뀐 것은 hard negative의 평가 의미다. `dataset_file_hash`가 `hard_negatives.json`을 포함하므로, 이 파일을 v0.2 안에서 고치면 잠긴 v0.2 dataset SHA가 파일과 어긋난다. 그래서 스키마 확장본을 v0.3으로 분리했다. evaluation schema version은 2다.

## v0.3

- E186은 positive answer evidence가 아니다. `evidence_class=relevant_conflicting`이다.
- 주제 관련성, 정답 가능 근거, 충돌 근거를 분리한다. E186은 주제 관련이고, 정답을 만들 수 없고, canonical 연결 매출액과 충돌한다.
- 이 분류는 `classified_for`의 Q03, Q06, Q07에 적용한다. 원소 id를 평가 코드에 하드코딩하지 않는다.
- `relevant_conflict_hit@k`는 진단 지표다. 높거나 낮다고 검색 품질 점수로 읽지 않는다.
- Q07 Path A의 연결 scope 문장 E2357은 chunk text에 주입하지 않는다. 그 문장이 E2362까지 구조적으로 열린 scope라는 점이 parser tree에 없다.

## Q07 scope를 chunk에 넣지 않은 이유

E2357과 E2362는 같은 `section_path`에 있다. 그 section은 `3. 재무상태 및 영업실적` 전체라서, 요약 연결손익, 주요경영지표 1)~3), `다.` 이후의 환율 표까지 한 구간에 들어 있다. E2357은 heading이 아니라 문장이고, `1)` `2)`도 heading이 아니다. 현재 chunker는 E2362 바로 앞의 `2) 수익성지표`만 preface로 붙인다. E2357과 E2362 사이에 scope를 `별도기준`으로 바꾸는 문장은 없다. 다만 `아래는 주요경영지표`가 번호 매긴 표 묶음에만 걸리고 `다.`에서 끝난다는 판단은 문서 위계가 아니라 읽기 해석이다. section 끝까지 연결기준을 전파하면 환율 표까지 같은 scope가 붙는다.

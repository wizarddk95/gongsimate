# 대표 corpus / benchmark sampling plan

작성일: 2026-10-01

이 문서는 향후 수집 계획이다. 이번 세션에서 KOSPI/KOSDAQ 전체를 받지 않았고, 100문항 benchmark도 만들지 않았다. 이수페타시스에 이미 저장된 정기공시는 development corpus다. 그 회사로 쓴 기존 질문 세트는 검색 결과를 보고 만든 문항이 섞여 있으므로 representative benchmark 점수로 읽지 않는다.

세 층을 섞지 않는다.

| 층 | 역할 | 이번 상태 |
| --- | --- | --- |
| development corpus | 파서, 색인, temporal policy를 고치는 동안 반복 실행 | 이수페타시스 정기공시. 제품 색인 대상 |
| regression set | 정책이 깨졌는지 보는 소수 질문. 정답 문장보다 eligible filing, retrieved filing, context filing, generation 여부를 본다 | `tests/test_multifiling_temporal.py`의 A–G |
| representative benchmark | 아직 검색해 보지 않은 기업의 원문에서 만든 문항. 시스템 점수는 문항이 고정된 뒤에만 계산 | 이번 세션에서 수집하지 않음 |

## Sampling axes

기업은 시장(KOSPI / KOSDAQ), 시가총액 구간, 업종을 같이 본다. 문서 유형은 사업보고서, 반기보고서, 분기보고서, 정정공시다. 구조 난이도는 문서 길이, 표 비중, 병합·중첩 표, ZIP 첨부 멤버 수, 이미지가 본문을 대신하는 정도다. 시간 축은 여러 회계연도, 당기·전기 비교 칼럼, 정정, 과거 사건을 후속 공시가 설명하는 경우다.

질문 축은 실적, 실적 변동 원인, 주요 사업, CAPEX, 신규사업, 위험요인, 배당, 자금조달이다. 한 기업에서 이 축을 모두 채우지 않아도 된다. 층별로 빈 칸을 남기고, 어느 칸이 비었는지 적는다.

파서 난이도는 둘로 나눈다. 현재 contract 안쪽은 DART XML의 제목, 문단, 표다. contract 밖은 OCR, 이미지 본문, HTML 전용 첨부, `unsupported_tag`로 `partial`이 되는 구조다. 밖은 검색 점수 분모에 넣지 않고 unsupported 로 따로 센다. 이수페타시스 `[기재정정]사업보고서 (2024.12)`는 이 바깥 사례다. `unsupported_tag:CORRECTION` 때문에 `overall_status=partial`이고 제품 색인에 없다.

## 금융업은 별도 track

은행, 증권, 보험의 사업보고서 목차와 계정 체계는 제조·서비스 정기공시와 다르다. 초기 representative benchmark의 같은 stratum에 넣으면 업종 효과가 파서·검색 효과와 섞인다. 초기 일반 stratum은 비금융으로 두고, 금융업은 challenge track으로 나중에 같은 질문 축을 따로 측정한다.

## 규모

초기 generalization smoke는 이수페타시스 밖에 기업 4곳을 더하는 안이 맞다. 3곳이면 시장·규모 축이 한 칸씩밖에 없고, 5곳을 넘기면 이번 파서 contract의 실패를 기업 수와 헷갈리기 쉽다. 네 칸은 다음이다.

1. KOSPI 대형 제조 또는 서비스. 금융 제외.
2. KOSPI 중형. 이수페타시스와 크기가 비슷한 비금융.
3. KOSDAQ 성장 기업. 분기 보고서가 짧고 첨부 멤버가 적은 쪽.
4. 정정이 실제로 이어진 비금융 기업. 원본과 정정본이 모두 `success`로 파싱되는 곳. 정정 태그 때문에 `partial`이면 그 기업은 정책 사례이지 검색 stratum이 아니다.

각 기업에서 우선 사업보고서 2개 회계연도, 그 사이 반기 1개, 분기 1개, 정정이 있으면 그 접수번호까지 보관한다. 감사보고서 멤버는 현재 제품 색인과 같이 검색 대상에서 뺀다.

최종 benchmark는 그 smoke에서 파서 `success` 비율과 temporal regression이 유지된 뒤에 6~10곳으로 늘린다. 10곳을 채우기 위해 금융업을 일반 stratum에 끼워 넣지 않는다.

회사 이름은 이 문서에서 확정하지 않는다. 이름을 고르려면 상장 목록과 접수 이력을 봐야 하고, 이번 세션은 그 수집을 하지 않았다.

## 문서 분포

smoke 4곳 × (사업보고서 2 + 반기 1 + 분기 1)이면 정기공시 16건이다. 정정본은 해당 기업에 있을 때만 추가하고, 건수 목표를 맞추려고 정정을 만들지 않는다. development corpus인 이수페타시스 건수에 이 16건을 더해 평균을 내지 않는다.

## Dataset construction protocol

문항과 시스템 실행을 시간상으로 분리한다.

1. 원문 ZIP과 parse manifest만 연다. 검색 순위, retrieval unit 텍스트, 모델 답은 보지 않는다.
2. 질문마다 질문 문장, 기대 사실의 성격(확정, 설명, 전망, 기간 없음), 근거가 있어야 하는 보고기간, 근거 element의 `document_id` / `element_id` / 문자 또는 셀 범위를 적는다. 근거 id는 청크 id가 아니다.
3. 그 파일을 잠근다. 이후 검색이 틀려도 질문을 고치지 않는다. 질문이 틀린 경우에만 별도 changelog로 고친다.
4. 시스템 평가는 잠긴 문항으로 한 번 돌린다. 실패를 보고 질문을 다시 쓰면 leakage다.
5. development corpus에서 이미 검색 결과를 보고 만든 문항은 regression 또는 개발 기록으로만 남긴다. representative 점수 표에 넣지 않는다.

이수페타시스 v0.1–v0.3 질문 세트는 2번 이전에 같은 사업보고서를 보고 만들어졌다. 그 점수는 그 스냅샷의 회귀이고 회사 대표 성능이 아니다.

## 이번 세션에서 하지 않은 것

시장 전체 다운로드, 추가 기업 수집, 100문항 작성, embedding 교체, BM25/RRF 재도입, reranker, query rewrite는 이 계획의 바깥이다.

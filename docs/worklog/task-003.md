# TASK-003 worklog — 고정된 DART 원본의 문서 요소 파싱

## 관측한 원본

세 건 모두 이수페타시스(007660) 정기공시다. 기업은 같고 보고서 종류·기간이 다르다. 다른 기업의 원본은 저장소에 없다.

| receipt_no | source_sha256 | 목록상 보고서 | ZIP 멤버 |
| --- | --- | --- | --- |
| 20260323000847 | `ff2ee7ca28c18ac6487ea4349cfde9d8dca4d03771c2a8a1bd37c9d81cfd7bb6` | 사업보고서 (2025.12) | `20260323000847.xml` (사업보고서), `_00761.xml` (연결감사보고서), `_00760.xml` (감사보고서). 이름 중복 없음 |
| 20260515002006 | `1609c4d4db119563e98c507ea939fc91c1e1e0b32f72f3eb0f102479fd7acb8c` | 분기보고서 (2026.03) | XML 1개 |
| 20260814002283 | `c58fbe472b2cafe15e0865b64774be69a482c4f76e89a5469de3f50ddc70a218` | 반기보고서 (2026.06) | XML 1개 |

공통 구조: UTF-8 선언의 dart4 XML. 루트 `DOCUMENT`, 본문은 `BODY` 아래 `COVER` / `SECTION-1`~`3`. 제목은 `COVER-TITLE`·`TITLE`, 문단은 `P`(`SPAN` 인라인), 표는 `TABLE`/`TR`/`TD`·`TH`·`TU`·`TE`와 `ROWSPAN`·`COLSPAN`. 이미지는 `IMAGE`/`IMG` 파일명만 있고 ZIP 안에 바이트는 없다. `PGBRK`는 쪽 나눔 표시일 뿐 쪽 번호가 아니다.

본문 XML은 그대로는 well-formed가 아니다. `<신설>` 같은 비교용 꺾쇠와 속성 안의 bare `&`가 있다. 파서는 알려진 태그 밖의 `<`와 bare `&`를 이스케이프한 뒤 엄격 파싱한다. 이 전처리 후에도 실패하면 복구 파싱으로 성공 처리하지 않는다.

멤버 role은 모두 `unknown`이다. `DOCUMENT-NAME`은 관측값으로만 남긴다. `report_fiscal_year`는 만들지 않았다.

## 지원 범위

- 지원: dart4 XML의 제목, 문단, 표(병합 포함). 표 안 문단은 셀 텍스트로만 남고 본문 문단으로 다시 뽑지 않는다.
- 중첩 표: 셀이 가리키는 자식 표로 분리한다. 값을 여러 칸에 복사하지 않는다. 관측된 중첩은 한 단계다. 그보다 깊으면 warning과 partial.
- 헤더: `TH` 또는 `THEAD` 안이면 `header`. 그 외는 `unknown`.
- 공백: `\r\n`을 `\n`으로 바꾸고 요소 앞뒤만 잘라 낸다. 내부 공백·줄바꿈·부호·괄호·단위는 유지한다.
- 제외: 이미지 OCR, HTML(이번 표본에 없음), 쪽 번호, 계정 매핑. 비텍스트 ZIP 멤버는 `excluded`로 적고 success를 깨지 않는다. 깨진 텍스트 멤버는 success로 숨기지 않는다.
- `element_id`는 source SHA, parser `1`, config hash, 문서, 순서, locator의 해시다. 문장 해시가 아니다.

locator는 `tree=dart4-xml-elementtree`와 루트부터의 `{tag, index}` 경로다. 같은 전처리 뒤 ElementTree에서 자식 순번으로 다시 찾는다. 바이트 오프셋이 아니다.

## 요소 예시

문단 (`20260323000847.xml`, order 2627): text `- 해당사항 없음`. 경로는 루트 `DOCUMENT`에서 자식 index를 따라 `P`에 도달한다.

표 (같은 멤버, order 2, 표지 사업연도): 셀 `사업연도`는 `ROWSPAN=2`, `2025년 01월 01일` / `부터` / `2025년 12월 31일` / `까지`. 병합으로 덮인 칸에는 값을 반복하지 않아 미리보기 두 번째 행 첫 칸은 비어 있다.

## 검증

| 구분 | 결과 |
| --- | --- |
| 자동 | `uv run pytest` 34 passed. `uv run ruff check src tests` 통과. fixture로 인라인·줄바꿈, 반복 문장, 병합 표, 표 텍스트 비중복, `<신설>`, 중첩 표, locator, 재실행 ID, current 고정, 손상 ZIP, DOCTYPE 거부, 인코딩, 중단 시 COMPLETE 없음 |
| 실원본 대조 | 위 3건을 `dart-parse`로 파싱(success, 요소 3664 / 1992 / 2180). 멤버마다 앞·중간·끝과 병합 표를 골라 25개 locator를 원문 트리와 다시 비교. 불일치 0. 빈 표 838개는 셀이 비어 있는 레이아웃 표였다 |

이 표본 통과를 다른 기업·공시의 품질 보증으로 쓰지 않는다.

## 누락·다음 단계

- HTML 멤버는 아직 없다. 나타나면 `unsupported`/`partial`이며 이번 성공에 포함되지 않는다.
- 이미지 파일명·캡션만 기록한다. OCR 없음.
- 표 단위 `(단위: 원)`은 셀로 남긴다. 가까운 문단이라고 표에 붙이지 않는다.
- 수집 backlog (이번 파서가 고치지 않음): 목록 중간 실패의 부분 manifest 없음, 버전/current/checks가 단일 트랜잭션이 아님, `.tmp-*`·lock 잔여물 자동 복구 없음, 중단된 개별 목록 재개 없음. 파서는 `.tmp-*` 원본을 읽지 않는다.

## CLI·변경·커밋 초안

```powershell
uv run dart-parse --receipt-no 20260323000847 --raw-dir ./data/raw --output-dir ./data/parsed
uv run dart-parse --receipt-no 20260323000847 --source-sha256 ff2ee7ca28c18ac6487ea4349cfde9d8dca4d03771c2a8a1bd37c9d81cfd7bb6
```

산출 경로: `data/parsed/dart/<receipt>/<sha>/p1/<config_hash 16자>/`. 완료 표식은 `COMPLETE`.

변경 파일: `src/disclosure_agent/parsing.py`, `src/disclosure_agent/cli.py`, `pyproject.toml`, `tests/test_parsing.py`, `README.md`, `docs/worklog/task-003.md`.

커밋 메시지 초안:

```
Add a snapshot parser for stored DART XML.

Parse a pinned receipt ZIP into headings, paragraphs, and tables without calling the API, and keep each source hash and parser config in its own result.
```

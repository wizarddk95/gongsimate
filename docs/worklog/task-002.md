# TASK-002 worklog — 공시 발견과 원본 재확인 동기화

## 정책 변경 이유

TASK-001의 “로컬이 정상이면 네트워크 없이 skipped”는 원격 원본 변경·손상 복구를 덮어 가렸다. 수집 계층의 기본값을 **온라인 재확인**으로 바꿨다. 로컬 검증만으로 ‘DART 최신 확인 완료’라고 표시하지 않는다.

“최신”은 해당 실행에서 검증에 성공한 DART ZIP 바이트와 현재 포인터가 일치한다는 뜻이다. 주기 스케줄·전 시장 스냅샷은 범위 밖이다.

## 선택 요약

- 저장: `versions/<sha256>/` + `current.json` + `checks.jsonl`. DB 없음.
- 발견: `corpCode.xml` 캐시 → `list.json` 페이지네이션(`pblntf_ty=A`, `last_reprt_at=N`) → receipt별 sync.
- CLI: `dart-ingest`(단건), `dart-sync`(기업·기간), `dart-corp-codes`(매핑 갱신).
- 재시도: 일시적 HTTP/API만. `020`·인증은 배치 중단.
- 단일 작성자: `data/raw/dart/.sync.lock`.

## 오프라인 검증

```
uv sync --locked
uv run pytest
uv run ruff check .
```

결과:

- pytest: **21 passed**
- ruff: **All checks passed**

## 실 API 검증

수행함 (키는 기록하지 않음).

| 실행 | 결과 |
| --- | --- |
| `dart-ingest --receipt-no 20190401004781` | TASK-001 레거시 → 버전 레이아웃 이전 후 `unchanged`, sha256 `ec52802b…`, 564569 bytes |
| `dart-sync --stock-code 005930 --bgn-de 20240301 --end-de 20240331 --max-filings 1` | list_complete, discovered=1, `created` rcept_no `20240312000736`, sha256 `3e452c1a…`, 596351 bytes |
| 동일 sync 재실행 | `unchanged`, 버전 수 1 유지, 확인 시각 갱신 |

원본 바이트가 원격에서 바뀐 사례는 실환경에서 관측하지 못함. `updated`/`repaired`/원격 실패 보존은 **모킹 테스트로 검증**.

## 코드 리뷰 우선 확인

1. `current.json` 전환 전에 version 디렉터리 staging이 깨지면 포인터가 안 바뀌는지.
2. 레거시 → 버전 이관 시 이전 ZIP이 `remote_verified: false`로 남는 경우.
3. 배치 `batch_ok`가 deferred/failed를 성공으로 숨기지 않는지.
4. 로그/checks에 키가 없는지.

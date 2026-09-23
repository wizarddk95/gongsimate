# TASK-001 worklog — DART 원본 한 건 수집

> **정책 변경 (TASK-002):** 아래 “로컬 정상이면 네트워크 없이 skipped / 키 없이 재사용”은 **폐기**되었다.
> 현재 기본 동작은 온라인 재확인이다. 현재 사용법은 README와 `docs/worklog/task-002.md`를 본다.
> 이 문서는 TASK-001 당시 구현·검증 이력으로만 남긴다.

## 선택 이유

- 한 건·동기 수집이므로 argparse + httpx 동기 클라이언트만 사용했다.
- `.env`는 CLI가 직접 읽는다. `OPENDART_API_KEY` 우선, `DART_API_KEY` 별칭.
- ZIP은 디스크에 풀지 않고 무결성만 확인했다.

## 오프라인 검증 (TASK-001 당시)

- `uv run pytest`: 11 passed
- `uv run ruff check .`: All checks passed

## 실 API 검증 (TASK-001 당시)

| 항목 | 값 |
| --- | --- |
| 접수번호 | `20190401004781` |
| 1차 | `fetched` |
| 2차 | `skipped` (당시 정책: 로컬 재사용) |
| size_bytes | `564569` |
| sha256 | `ec52802b8566843d943ce1277c8da38b4b946e6c968429d7430c480d8d333456` |

## 후속

TASK-002에서 온라인 재확인·목록 발견·버전 보존으로 정책을 교체했다.

# disclosure-agent

개인 투자자용 국내 공시 리서치 에이전트의 수집 계층입니다. 접수번호 단건 원격 확인과, 기업·접수일 범위로 정기공시를 발견해 원본 ZIP을 버전 보존하며 동기화합니다.

## 요구 사항

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

## 설치

```powershell
uv sync --locked
```

엔트리포인트: `dart-ingest`, `dart-sync`, `dart-corp-codes`.

## 인증키

OpenDART 인증키는 환경변수로만 받습니다.

- 기본: `OPENDART_API_KEY`
- 별칭: `DART_API_KEY` (`OPENDART_API_KEY`가 없을 때만)

실행 시 cwd와 프로젝트 루트의 `.env`를 **명시적으로** 읽습니다. 이미 설정된 프로세스 환경변수는 덮어쓰지 않습니다.

```powershell
$env:OPENDART_API_KEY = "your-opendart-api-key"
```

**원격 확인이 기본입니다.** 로컬에 정상 파일이 있어도 API 키 없이 성공 처리하지 않습니다. 키가 없거나 원격 조회가 실패하면 원격 확인 실패입니다.

## 사용

### 접수번호 한 건 원격 확인

```powershell
uv run dart-ingest --receipt-no 20190401004781 --output-dir ./data/raw
```

항상 DART에서 ZIP을 다시 받아 비교합니다. 결과는 `created` / `unchanged` / `updated` / `repaired` / `failed` 중 하나입니다.

### 기업·기간 발견 후 동기화

```powershell
uv run dart-sync --stock-code 005930 --bgn-de 20240301 --end-de 20240331 --max-filings 2 --output-dir ./data/raw
uv run dart-sync --corp-code 00126380 --bgn-de 20240301 --end-de 20240331 --max-filings 2
```

- `--stock-code`와 `--corp-code` 중 하나만 지정합니다.
- 기간은 **접수일**(`bgn_de`/`end_de`, YYYYMMDD)입니다. 사업연도와 같지 않을 수 있습니다.
- 기본 공시유형은 정기공시(`pblntf_ty=A`), `last_reprt_at=N`(정정 포함).
- `--max-filings`로 처리 상한을 두면 나머지는 `deferred`이며 실패와 구분합니다.

### 기업 고유번호 매핑 갱신

```powershell
uv run dart-corp-codes --force --cache-dir ./data/cache
```

매핑의 `fetched_at`이 `data/cache/dart/corp_codes.json`에 기록됩니다. `dart-sync`는 캐시가 없으면 자동으로 한 번 받습니다.

### 검증

```powershell
uv run pytest
uv run ruff check .
```

## 저장 구조

```
data/raw/dart/<receipt_no>/
  current.json              # 현재 원본 포인터 + 성공/시도 시각
  versions/<sha256>/document.zip
  versions/<sha256>/version.json
  checks.jsonl              # 확인 이력
  quarantine/               # 손상 로컬 격리
data/raw/dart/runs/<run_id>-manifest.json
data/raw/dart/runs/<run_id>-summary.json
data/cache/dart/corp_codes.json
```

TASK-001의 `document.zip` + `metadata.json` 레이아웃은 읽습니다. 원격 확인 성공 전에는 `remote_verified`로 승격하지 않으며, 확인 후 `versions/`로 이전합니다.

## 결과 상태

| 상태 | 의미 |
| --- | --- |
| created | 이 접수번호의 첫 정상 관측 |
| unchanged | 현재 버전과 원격 ZIP 바이트가 같음 (원본 중복 저장 없음, 확인 시각 갱신) |
| updated | 바이트가 다름 (이전 버전 보존, 현재 포인터 전환). 본문 정정 확정이 아님 |
| repaired | 로컬 손상 후 원격으로 복구 |
| failed | 원격/검증 실패 (이전 정상 원본·성공 확인 시각 유지) |

`unchanged`/`updated`의 “최신”은 **해당 실행에서 검증에 성공한 DART 응답과 일치**한다는 뜻입니다. 서버의 지속적 최신성이나 배치 동일 시점 스냅샷을 보장하지 않습니다.

별도 접수번호의 정정공시는 별도 filing으로 수집합니다.

## 실패·배치

- 종료 코드 `2`: 인자/키 오류 (HTTP 전)
- 종료 코드 `1`: 원격 실패, 또는 배치가 완전 성공이 아님 (`failed`/`deferred`/`aborted`/목록 미완료)
- 일시적 네트워크·서버 오류만 제한 재시도(기본 3회, exponential backoff). 인증·입력 오류는 재시도하지 않음
- 요청 제한(`020`)·치명적 인증 오류는 배치 중단
- 개별 문서 실패는 가능한 범위에서 다음 문서로 진행
- `data/raw/dart/.sync.lock`으로 단일 작성자만 가정 (분산 동시성 미지원)

## 한계

- 강제 종료 후 `.tmp-*` / lock 잔여물은 자동 복구하지 않습니다
- ETag/HEAD만으로 동일성을 주장하지 않습니다 (항상 ZIP 다운로드 후 SHA-256)
- 본문 파싱·정정 diff·스케줄러·전체 시장 일괄은 범위 밖입니다

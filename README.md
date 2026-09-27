# disclosure-agent

개인 투자자용 국내 공시 리서치 에이전트의 수집·파싱 계층입니다. 접수번호 단건 원격 확인과, 기업·접수일 범위로 정기공시를 발견해 원본 ZIP을 버전 보존하며 동기화합니다. 저장된 ZIP의 특정 버전에서는 제목·문단·표를 문서 요소로 뽑습니다.

## 요구 사항

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

## 설치

```powershell
uv sync --locked
```

엔트리포인트: `dart-ingest`, `dart-sync`, `dart-corp-codes`, `dart-parse`, `dart-chunk`, `dart-eval-bm25`.

## 인증키

OpenDART 인증키는 환경변수로만 받습니다.

- 기본: `OPENDART_API_KEY`
- 별칭: `DART_API_KEY` (`OPENDART_API_KEY`가 없을 때만)

실행 시 cwd와 프로젝트 루트의 `.env`를 **명시적으로** 읽습니다. 이미 설정된 프로세스 환경변수는 덮어쓰지 않습니다.

```powershell
$env:OPENDART_API_KEY = "your-opendart-api-key"
```

**원격 확인이 `dart-ingest`와 `dart-sync`의 기본입니다.** 로컬에 정상 파일이 있어도 API 키 없이 성공 처리하지 않습니다. 키가 없거나 원격 조회가 실패하면 원격 확인 실패입니다. `dart-parse`는 저장된 ZIP만 읽으므로 키가 필요 없습니다.

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

### 저장된 원본 파싱

API 키 없이, 이미 받은 ZIP만 읽습니다. `--source-sha256`을 생략하면 `current.json`을 시작할 때 한 번만 읽고 그 해시에 고정합니다.

```powershell
uv run dart-parse --receipt-no 20260323000847 --raw-dir ./data/raw --output-dir ./data/parsed
uv run dart-parse --receipt-no 20260323000847 --source-sha256 <64-hex>
```

결과는 `data/parsed/dart/<receipt_no>/<sha256>/p2/<config_hash 앞 16자>/`에 `parse_manifest.json`, `elements.jsonl`, `preview.md`, `preprocess_report.json`으로 남습니다. 파서 버전은 `2`, 스키마는 `2`, 전처리 버전은 `2`입니다. `p1/` 결과는 그대로 둡니다. `COMPLETE`는 산출물 쓰기가 끝났다는 뜻입니다. 파싱 품질은 manifest의 `overall_status`입니다. `partial`도 종료 코드 0일 수 있지만, 그건 완전 성공이 아닙니다.

`dart-parse` 출력 앞부분은 `status`와 `search_eligible`을 구분합니다. `partial`이면 `처리 완료, 검색 입력 제외`라고 적습니다. 이어서 원본 SHA, 파서·전처리 버전, 설정 해시, 재사용 여부, manifest 경로, 멤버 상태와 경고 코드를 보입니다. 경고가 길면 개수와 생략 여부, 상세 파일(manifest) 경로를 적습니다. 검색 자격은 이 문장을 파싱해서 정하지 않습니다. 기준은 manifest의 `overall_status`입니다.

전처리는 디코딩된 멤버 텍스트의 유니코드 코드포인트 오프셋으로 기록합니다. ZIP 바이트 위치나 locator가 아닙니다. 확인된 리터럴(`<신설>`, `<연결기준>`, `<별도기준>`, `연결기준_`, `별도기준_`, `<중단영업손실>`, `<표1>`, `<표2>`)과 태그 이름이 아닌 `<`, 그리고 기존 엔티티가 아닌 `&`만 안전하게 고칩니다. 모르는 ASCII 태그(예: `<CORRECTION>`)는 텍스트로 바꾸지 않고 트리에 남겨 `partial`로 둡니다. 래퍼 안의 `TITLE`은 그 래퍼 뒤의 문단·표에만 `section_path`로 이어지고, 래퍼 밖 형제나 다음 문서에는 붙지 않습니다. 그 경로를 고쳤다고 태그가 지원되거나 `partial`이 `success`가 되지는 않습니다. 안전한 보정이 있었다는 이유만으로 `partial`이 되지 않습니다. 전처리 버전은 설정 해시에 포함됩니다.

locator는 멤버 트리별 컨텍스트입니다. 프로세스 전역 경로 맵을 쓰지 않습니다. 각 locator에는 트리 이름, 원본 SHA, 멤버 순번, 멤버 이름, 자식 순번 경로가 있습니다. 같은 파일명의 멤버는 순번으로 구분합니다. `resolve_locator`와 `element_text_at`의 인자는 이전과 같고, 호출마다 넘긴 멤버 텍스트로 트리를 새로 만듭니다. 왕복은 전처리 뒤 같은 ElementTree에서 자식 순번을 다시 찾는 검증입니다. 전처리 전 원문과 출력의 독립 대조는 그 왕복과 별개입니다.

같은 결과 디렉터리는 로컬 파일 잠금으로 한 번에 하나만 씁니다. 잠금 파일은 그 디렉터리 옆 `.<config_hash 앞 16자>.lock`이고, 수집용 `data/raw/dart/.sync.lock`과는 별개입니다. 잠금 안에서 기존 결과 검증, 손상본 격리, 재사용 판단, 공개를 합니다. 대기 중 다른 실행이 정상 완료하면 그 결과를 재사용합니다. 기본 대기는 300초이고, 넘기면 종료 코드 `3`입니다. 이 시간 초과는 원본 손상이나 파싱 실패가 아니며 실패 manifest를 쓰지 않습니다. 잠금은 `O_CREAT|O_EXCL`입니다. 이 환경(Windows NTFS, 로컬 디스크)에서 확인했습니다. 분산 잠금이 아니고, Linux·macOS·네트워크 파일시스템은 이 변경에서 검증하지 않았습니다.

작성 중 디렉터리는 `.tmp-<pid>-<uuid>`입니다. 실패한 실행은 자기가 만든 임시 디렉터리만 지웁니다. 다른 실행의 `.tmp-*`, lock, `COMPLETE` 결과는 지우지 않습니다. 강제 종료로 남은 임시 디렉터리와 lock은 자동으로 지우지 않습니다. lock은 그 파일을 연 프로세스가 없을 때만 사람이 지웁니다. 오래됐다는 이유만으로 지우지 않습니다.

재사용 전에는 `COMPLETE`와 필수 파일이 읽히는지, manifest·전처리 보고서가 JSON인지, 스키마·원본 SHA·파서 버전·전처리 버전·**전체** `config_hash`가 맞는지, `preview.md`가 비어 있지 않은지, `elements.jsonl`의 각 줄이 JSON이고 줄 수가 `element_count`와 같은지를 확인합니다. 파일 바이트의 내용 해시는 비교하지 않습니다. 줄 수가 같고 JSON으로 남는 텍스트 변조는 여기서 걸리지 않습니다. 경로 키는 해시 앞 16자이므로 전체 해시가 다르면 재사용하거나 덮어쓰지 않습니다. 손상된 완료 결과는 격리한 뒤 다시 만듭니다. 완료된 `partial`은 재사용해도 `partial`과 경고를 유지합니다. `COMPLETE`가 없는 `.tmp-*`는 완료 결과가 아닙니다.

괄호로 된 단위 라벨(`(단위 : 백만원)`, `(단위: 백만원, %)`)은 셀·문단 원문 그대로 남습니다. 숫자로 바꾸거나 표 전체에 적용하지 않습니다. 라벨이 보이면 `unit_mentions`에 `raw_text`, 그 요소·셀의 locator, `applies_to: null`을 둡니다. 적용 범위는 확정하지 않습니다(`unknown`). 한 라벨에 단위가 여럿이면 `unresolved`이고 하나의 배수가 아닙니다. 표 앞 문단의 단위는 그 문단의 기록으로 남고 다음 표에 붙이지 않습니다. 라벨이 없으면 `unit_mentions` 키 자체가 없습니다. 이것은 부재이고, 라벨은 있으나 범위가 `unknown`/`unresolved`인 것과 다릅니다. 이 저장소의 이수페타시스 원본에서는 셀 안, 표로 보이는 레이아웃 칸, 표 밖 문단, 쉼표로 섞인 라벨을 봤습니다. 계정 매핑과 단위 환산은 하지 않습니다.

청킹을 만들 때는 요소 텍스트와 함께 이 `unit_mentions`를 문맥으로 실어야 합니다. `unknown`과 `unresolved`는 확정된 재무 사실이 아닙니다. `dart-chunk`가 그 라벨을 청크에 복사하되 `unit_applied`는 항상 false입니다. `TD`/`TE`의 `header_role=unknown`은 헤더 텍스트가 없다는 뜻이 아닙니다.

```powershell
uv run dart-chunk --manifest data/parsed/dart/<receipt>/<sha>/p2/<prefix>/parse_manifest.json --elements data/parsed/dart/<receipt>/<sha>/p2/<prefix>/elements.jsonl --document-id m0:<member>.xml --output evaluation/runs/chunks.jsonl
uv run dart-eval-bm25 --dataset evaluation/datasets/isu_petasys_v0_1 --manifest <parse_manifest.json> --elements <elements.jsonl> --output-dir evaluation/runs/isu_petasys_v0_1
```

검색 자격은 manifest의 `overall_status == success`입니다. 첫 BM25 기준선의 인덱스는 관측된 `document_name=사업보고서`인 멤버 0만 포함합니다. 평가 정답은 청크 ID가 아니라 `evaluation/datasets/isu_petasys_v0_1/`의 원문 근거 범위입니다. 이 10문항 점수는 해당 스냅샷의 회귀 결과이며 전체 기업 성능이 아닙니다.

후속 검색은 `COMPLETE`나 종료 코드, CLI 문장이 아니라 manifest의 `overall_status == success`만 입력으로 받습니다. 초기에는 `partial`을 자동으로 받지 않습니다. 영향 없는 멤버만 받는 품질 정책은 검색을 만들 때 따로 정합니다.

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
data/parsed/dart/<receipt_no>/<sha256>/p<parser_version>/<config_hash 앞 16자>/
  parse_manifest.json       # 입력 계보, 멤버 처리, 상태, 요소 수
  elements.jsonl            # 제목·문단·표 요소 (읽기 순서)
  preview.md                # 원문 대조용 미리보기
  preprocess_report.json    # 보정 규칙·건수·코드포인트 위치·전후 텍스트
  COMPLETE                  # 쓰기 완료. 품질은 parse_manifest.json의 overall_status
  quarantine/               # 재사용 불가였던 손상 결과
```

TASK-001의 `document.zip` + `metadata.json` 레이아웃은 읽습니다. 원격 확인 성공 전에는 `remote_verified`로 승격하지 않으며, 확인 후 `versions/`로 이전합니다.

- `versions/<sha256>/version.json`: ZIP 버전 메타
- `current.json`: "지금 어떤 버전이 현재인가" 포인터
- `checks.jsonl`: 이번 확인 시도 이력 1줄 append

ingest로 받은 경우 `listing`은 비어있고, sync로 받은 경우에는 `corp_code` / `stock_code` / `report_nm` 등이 들어 있습니다. 다운로드되는 본문 파일 자체는 같으며 차이는 발견·배치 메타와 listing 보강입니다.

파싱 결과는 원본 `versions/`와 분리됩니다. 경로의 SHA-256은 파싱 시작 때 고정한 원본이고, 파서 버전과 설정 해시가 다르면 다른 디렉터리에 공존합니다. 현재 파서 버전은 `2`입니다. 전체 설정 해시는 `parse_manifest.json`의 `config_hash`에 있습니다. 작성 중 디렉터리는 `.tmp-<pid>-<uuid>`이며, `COMPLETE` 없이 남은 결과는 성공이 아닙니다. 같은 원본·파서·설정으로 다시 실행하면 완료된 `success`/`partial` 결과를 재사용합니다.

## 결과 상태

아래 표는 `dart-ingest`와 `dart-sync`입니다.

| 상태 | 의미 |
| --- | --- |
| created | 이 접수번호로 처음 정상 ZIP을 저장함. `versions/<sha>/` 생성, `current`가 그걸 가리킴 |
| unchanged | 원격 ZIP 바이트가 현재 버전과 같음. ZIP 재저장 없음. 확인 시각(`last_success_verified_at`)·check 이력만 갱신 |
| updated | 원격 바이트가 현재와 다름. 이전 version은 유지하고, 새 sha version을 만든 뒤 `current` 포인터 전환. (본문 정정 확정은 아님 ― ZIP 포장 차이일 수도 있음) |
| repaired | 로컬이 손상으로 보였고, 원격 ZIP으로 복구함. 손상분은 quarantine 등으로 격리. `unchange`로 숨기지 않음 |
| failed | 원격 다운로드·검증·저장 실패. 이전 정상 원본과 `last_success_verified_at`은 유지. `last_attempt_at`·`last_result=failed`·check 한 줄만 실패로 기록 |

`unchanged`/`updated`의 “최신”은 **해당 실행에서 검증에 성공한 DART 응답과 일치**한다는 뜻입니다. 서버의 지속적 최신성이나 배치 동일 시점 스냅샷을 보장하지 않습니다.

별도 접수번호의 정정공시는 별도 filing으로 수집합니다.

`dart-corp-codes`의 `status: refreshed` / `cached` ― corp 매핑 캐시를 새로 받았는지, 기존 캐시를 재사용했는지를 뜻함.

`batch_ok` / summary의 created·failed 카운트 ― 배치 집계일 뿐, 단건 enum이 아님

OpenDART API `status` (`013` 없음, `020` 제한 등) ― HTTP API 응답 코드. 로컬 `created`/`failed`와는 별개

`dart-parse`의 패키지 상태:

| 상태 | 의미 |
| --- | --- |
| success | 지원하는 dart4 XML 멤버에서 제목·문단·표를 생성함. 의미 정확성이나 첨부 해석 완료를 뜻하지 않음. 이미지 등 의도적 제외는 manifest에 남음 |
| partial | 일부 텍스트 멤버는 읽었으나 실패·미지원 텍스트·복구가 필요한 구조가 있음 |
| failed | 유효한 입력 또는 출력을 만들지 못함 |
| unsupported | 지원 포맷에 해당하는 텍스트 멤버가 없음 |

멤버 상태는 `parsed` / `partial` / `failed` / `unsupported` / `excluded`(이미지·바이너리 등 비텍스트)로 집계됩니다. `excluded`만으로는 패키지를 `success`에서 내리지 않습니다. 읽지 못한 텍스트 멤버는 `excluded`로 숨기지 않습니다.

## 실패·배치

- 종료 코드 `2`: 인자/키 오류 (HTTP 전). `dart-parse`도 접수번호·해시 형식 오류는 `2`
- 종료 코드 `1`: 원격 실패, 또는 배치가 성공이 아님 (`failed`/`aborted`/목록 미완료). `--max-filings`로 남은 `deferred`는 의도적 상한이라 실패로 보지 않음. `dart-parse`는 패키지 상태가 `failed` 또는 `unsupported`일 때 `1`, `success`/`partial`일 때 `0`
- 종료 코드 `3`: `dart-parse`가 로컬 결과 잠금을 기다리는 동안 시간이 초과됨. 파싱 실패가 아님
- 일시적 네트워크·서버 오류만 제한 재시도(기본 3회, exponential backoff). 인증·입력 오류는 재시도하지 않음
- 요청 제한(`020`)·치명적 인증 오류는 배치 중단 (목록 단계 실패도 `runs/*-summary.json`에 `aborted`로 기록)
- 개별 문서 실패는 가능한 범위에서 다음 문서로 진행
- `data/raw/dart/.sync.lock`으로 수집은 단일 작성자만 가정 (분산 동시성 미지원). 파서 결과 잠금은 결과 디렉터리마다 따로 둠

## 한계

- 강제 종료 후 수집 `.tmp-*` / lock 잔여물과 파서 `.tmp-*`·`.lock`은 자동으로 지우지 않습니다. 파서 결과는 결과 디렉터리에 `COMPLETE`가 있을 때만 완료입니다. 다른 실행의 임시 디렉터리를 완료로 보지 않습니다
- ETag/HEAD만으로 동일성을 주장하지 않습니다 (항상 ZIP 다운로드 후 SHA-256)
- 파싱은 고정된 ZIP 스냅샷만 처리합니다. 파싱 도중 DART를 다시 조회하거나 `current.json`을 따라가지 않습니다
- 지원 문서는 dart4 XML의 제목·문단·표입니다. HTML, 이미지 OCR, 쪽 번호, 계정 매핑, 단위 환산, 정정 diff, 스케줄러, 전체 시장 일괄은 범위 밖입니다. `<CORRECTION>`은 미지원이며 해당 문서는 `partial`입니다

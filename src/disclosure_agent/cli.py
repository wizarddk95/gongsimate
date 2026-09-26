"""CLI: single-receipt remote sync, company discovery sync, corp-code refresh."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from disclosure_agent.dart import fetch_document
from disclosure_agent.discovery import (
    discover_filings,
    load_corp_cache,
    refresh_corp_cache,
    resolve_corp,
    validate_ymd,
    write_manifest,
)
from disclosure_agent.errors import (
    FatalBatchError,
    IngestError,
    RateLimitError,
    UsageError,
)
from disclosure_agent.lock import output_run_lock
from disclosure_agent.parsing import parse_receipt
from disclosure_agent.sync import sync_receipt

RECEIPT_NO_RE = re.compile(r"^\d{14}$")
STOCK_CODE_RE = re.compile(r"^\d{6}$")
CORP_CODE_RE = re.compile(r"^\d{8}$")


@dataclass
class BatchSummary:
    run_id: str
    list_complete: bool
    selected: int
    processed: int
    created: int = 0
    unchanged: int = 0
    updated: int = 0
    repaired: int = 0
    failed: int = 0
    deferred: int = 0
    aborted: str | None = None

    def bump(self, status: str) -> None:
        if status == "created":
            self.created += 1
        elif status == "unchanged":
            self.unchanged += 1
        elif status == "updated":
            self.updated += 1
        elif status == "repaired":
            self.repaired += 1
        elif status == "failed":
            self.failed += 1
        else:
            raise ValueError(f"unknown sync status: {status!r}")

    @property
    def ok(self) -> bool:
        # deferred는 --max-filings로 의도적으로 남긴 건수라 실패가 아니다.
        return (
            self.list_complete
            and self.aborted is None
            and self.failed == 0
        )


# project.toml의 [project.scripts]가 여기에 연결됨.
# dart-ingest 명령어를 실행하면 실행되는 함수
def main(argv: Sequence[str] | None = None) -> None:
    raise SystemExit(run_cli(argv))

# dart-corp-codes 명령어를 실행하면 실행되는 함수
def corp_codes_main(argv: Sequence[str] | None = None) -> None:
    raise SystemExit(run_corp_codes_cli(argv))

# dart-sync 명령어를 실행하면 실행되는 함수
def sync_main(argv: Sequence[str] | None = None) -> None:
    raise SystemExit(run_sync_cli(argv))


def parse_main(argv: Sequence[str] | None = None) -> None:
    raise SystemExit(run_parse_cli(argv))


def run_cli(argv: Sequence[str] | None = None) -> int:
    parser = _build_receipt_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
        load_dotenv_files()
        receipt_no = validate_receipt_no(args.receipt_no)
        api_key = require_api_key()
        with output_run_lock(args.output_dir):
            result = sync_receipt(receipt_no, args.output_dir, api_key)
    except IngestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.exit_code
    _print_sync_outcome(result.outcome)
    return 0 if result.outcome.status != "failed" else 1


def run_sync_cli(argv: Sequence[str] | None = None) -> int:
    # sync 파서를 생성
    parser = _build_sync_parser()
    try:
        # 입력 검증
        # 파서를 통해 인자를 파싱
        args = parser.parse_args(list(argv) if argv is not None else None)
        # .env 파일을 로드
        load_dotenv_files()
        # 회사 동기화 실행
        summary = run_company_sync(args)
    except IngestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.exit_code
    _print_batch_summary(summary)
    return 0 if summary.ok else 1


def run_parse_cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dart-parse",
        description="Parse a stored DART original ZIP into document elements. No API key.",
    )
    parser.add_argument("--receipt-no", required=True, help="14-digit DART receipt number")
    parser.add_argument(
        "--source-sha256",
        default=None,
        help="Pinned original SHA-256. If omitted, current.json is read once.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("./data/raw"),
        help="Raw storage root (default: ./data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./data/parsed"),
        help="Parsed output root (default: ./data/parsed)",
    )
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
        result = parse_receipt(
            args.receipt_no,
            args.raw_dir,
            args.output_dir,
            source_sha256=args.source_sha256,
        )
    except IngestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.exit_code
    _print_parse_result(result)
    if result["status"] in {"failed", "unsupported"}:
        return 1
    return 0


_CLI_CODE_LIMIT = 12


def _print_parse_result(result: dict[str, Any]) -> None:
    """Print diagnostics. Search eligibility is read from the manifest, not from this text."""
    print(f"status: {result['status']}")
    print(f"search_eligible: {result['search_eligible']}")
    if result["status"] == "partial":
        print("quality_note: 처리 완료, 검색 입력 제외")
    print(f"source_sha256: {result['source_sha256']}")
    print(f"parser_version: {result['parser_version']}")
    print(f"preprocess_version: {result['preprocess_version']}")
    print(f"config_hash: {result['config_hash']}")
    print(f"reused: {result['reused']}")
    print(f"manifest: {result['manifest']}")
    print(f"element_count: {result['element_count']}")
    print(f"dest: {result['dest']}")
    omitted = _print_codes("package_warnings", list(result.get("warnings") or []))
    print("members:")
    for member in result.get("members") or []:
        shown, member_omitted = _clip_codes(list(member.get("warnings") or []))
        omitted += member_omitted
        reason = member.get("reason")
        reason_text = f" reason={reason}" if reason else ""
        print(
            f"  {member.get('member_index')} {member.get('member_name')} "
            f"{member.get('status')} {shown}{reason_text}"
        )
    if omitted:
        print(f"warnings_omitted: {omitted}")
        print(f"warnings_detail: {result['manifest']}")


def _print_codes(label: str, codes: list[str]) -> int:
    shown, omitted = _clip_codes(codes)
    print(f"{label}: {shown}")
    return omitted


def _clip_codes(codes: list[str]) -> tuple[str, int]:
    if not codes:
        return "-", 0
    if len(codes) <= _CLI_CODE_LIMIT:
        return ", ".join(codes), 0
    shown = ", ".join(codes[:_CLI_CODE_LIMIT])
    omitted = len(codes) - _CLI_CODE_LIMIT
    return f"{shown} (count={len(codes)}, omitted={omitted})", omitted


def run_corp_codes_cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dart-corp-codes",
        description="Download and cache OpenDART corpCode mapping.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("./data/cache"),
        help="Cache root (default: ./data/cache)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refresh even if a cache file already exists",
    )
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
        load_dotenv_files()
        api_key = require_api_key()
        had_cache = load_corp_cache(args.cache_dir) is not None
        cache = refresh_corp_cache(args.cache_dir, api_key, force=args.force)
    except IngestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.exit_code
    reused = had_cache and not args.force
    print(f"status: {'cached' if reused else 'refreshed'}")
    print(f"fetched_at: {cache.get('fetched_at')}")
    print(f"count: {cache.get('count')}")
    print(f"path: {args.cache_dir / 'dart' / 'corp_codes.json'}")
    return 0


def run_company_sync(args: argparse.Namespace) -> BatchSummary:
    # API 키 검증
    api_key = require_api_key()
    # 시작 날짜 검증
    bgn_de = validate_ymd(args.bgn_de, name="--bgn-de")
    # 종료 날짜 검증
    end_de = validate_ymd(args.end_de, name="--end-de")
    # 시작 날짜가 종료 날짜보다 이후면 오류
    if bgn_de > end_de:
        raise UsageError("--bgn-de must be on or before --end-de")
    # 인자에서 종목 코드 또는 회사 코드를 가져옴
    stock_code = args.stock_code
    corp_code = args.corp_code
    # 종목 코드 또는 회사 코드 검증
    if stock_code:
        stock_code = validate_stock_code(stock_code)
    if corp_code:
        corp_code = validate_corp_code(corp_code)

    # 실행 ID 생성
    run_id = uuid4().hex 
    # 배치 요약 생성 (목록 전이라 list_complete는 False로 시작)
    summary = BatchSummary(
        run_id=run_id,
        list_complete=False,
        selected=0,
        processed=0,
        deferred=0,
    )

    # 락 + 공유 HTTP 클라이언트
    # 락: 같은 output_dir에 대해 ingest/sync 동시 실행 방지
    with output_run_lock(args.output_dir):
        # 하나의 httpx.Client를 사용해 corp resolve → list → document까지 재사용 
        with httpx.Client(timeout=30.0, follow_redirects=False) as client:
            # 기업 해석 (캐시가 없으면 여기서 corp ZIP 다운로드)
            corp, cache = resolve_corp(
                args.cache_dir,
                stock_code=stock_code,
                corp_code=corp_code,
                api_key=api_key,
                client=client,
                refresh_if_missing=True,
            )
            try:
                # 공시 목록 검색
                manifest = discover_filings(
                    api_key,
                    corp=corp,
                    bgn_de=bgn_de,
                    end_de=end_de,
                    cache_meta=cache,
                    max_filings=args.max_filings,
                    client=client,
                    run_id=run_id,
                )
            except IngestError as exc:
                # 목록 실패도 summary에 aborted를 남기고 반환 (raise하지 않음)
                summary.aborted = str(exc)
                summary.list_complete = False
            else:
                # manifest 기록·콘솔
                manifest_path = write_manifest(args.output_dir, manifest)
                summary.list_complete = manifest.list_complete
                summary.selected = len(manifest.selected_receipt_nos)
                summary.deferred = len(manifest.deferred_receipt_nos)
                print(f"manifest: {manifest_path}")
                print(f"list_complete: {manifest.list_complete}")
                print(f"discovered: {len(manifest.items)}")
                print(f"selected: {summary.selected}")
                print(f"deferred: {summary.deferred}")

                listing_by_receipt = {
                    item["rcept_no"]: item
                    for item in manifest.items
                    if "rcept_no" in item
                }

                def fetch(r: str, k: str, _client: httpx.Client = client) -> bytes:
                    return fetch_document(r, k, client=_client)

                for receipt_no in manifest.selected_receipt_nos:
                    listing = listing_by_receipt.get(receipt_no)
                    try:
                        # 접수번호 1건의 "원격 확인 + 로컬 버전 반영"
                        result = sync_receipt(
                            receipt_no,
                            args.output_dir,
                            api_key,
                            run_id=run_id,
                            listing=listing,
                            fetch=fetch,
                        )
                    except RateLimitError as exc:
                        summary.aborted = str(exc)
                        break
                    except FatalBatchError as exc:
                        summary.aborted = str(exc)
                        break
                    summary.processed += 1
                    summary.bump(result.outcome.status)
                    _print_sync_outcome(result.outcome, compact=True)
                    if result.outcome.error_type in {
                        "RateLimitError",
                        "FatalBatchError",
                    }:
                        summary.aborted = result.outcome.error_message
                        break

    _write_run_summary(args.output_dir, summary)
    return summary


def validate_receipt_no(receipt_no: str) -> str:
    value = receipt_no.strip()
    if not RECEIPT_NO_RE.fullmatch(value):
        raise UsageError("receipt number must be exactly 14 digits")
    return value


def validate_stock_code(value: str) -> str:
    text = value.strip()
    if not STOCK_CODE_RE.fullmatch(text):
        raise UsageError("stock code must be exactly 6 digits")
    return text


def validate_corp_code(value: str) -> str:
    text = value.strip()
    if not CORP_CODE_RE.fullmatch(text):
        raise UsageError("corp code must be exactly 8 digits")
    return text


def require_api_key() -> str:
    key = (
        os.environ.get("OPENDART_API_KEY") or os.environ.get("DART_API_KEY") or ""
    ).strip()
    if not key:
        raise UsageError(
            "OPENDART_API_KEY environment variable is required for remote verification"
        )
    return key


def load_dotenv_files() -> None:
    """Load KEY=VALUE pairs from .env. Existing process environment wins."""
    candidates: list[Path] = [Path.cwd() / ".env"]
    project_root = Path(__file__).resolve().parents[2]
    candidates.append(project_root / ".env")
    seen: set[Path] = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen or not path.is_file():
            continue
        seen.add(resolved)
        _apply_dotenv(path)


def _apply_dotenv(path: Path) -> None:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, value = stripped.partition("=")
        name = name.strip()
        if name.lower().startswith("export "):
            name = name[7:].strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if name and name not in os.environ:
            os.environ[name] = value


def _build_receipt_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dart-ingest",
        description=(
            "Remotely verify one DART original disclosure ZIP and store a versioned snapshot."
        ),
    )
    parser.add_argument(
        "--receipt-no",
        required=True,
        help="14-digit DART receipt number (rcept_no)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./data/raw"),
        help="Root directory for raw artifacts (default: ./data/raw)",
    )
    return parser


def _build_sync_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dart-sync",
        description="Discover periodic filings for one company and remotely sync originals.",
    )
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--stock-code", help="6-digit stock code")
    identity.add_argument("--corp-code", help="8-digit OpenDART corp_code")
    parser.add_argument("--bgn-de", required=True, help="Receipt start date YYYYMMDD")
    parser.add_argument("--end-de", required=True, help="Receipt end date YYYYMMDD")
    parser.add_argument(
        "--max-filings",
        type=int,
        default=None,
        help="Process at most N discovered filings; remainder is deferred (not failed)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./data/raw"),
        help="Root directory for raw artifacts (default: ./data/raw)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("./data/cache"),
        help="Cache root for corp code mapping (default: ./data/cache)",
    )
    return parser


def _print_sync_outcome(outcome: Any, *, compact: bool = False) -> None:
    if compact:
        print(
            f"{outcome.status}\t{outcome.receipt_no}\t"
            f"{outcome.sha256 or '-'}\t{outcome.size_bytes or '-'}"
        )
        return
    print(f"status: {outcome.status}")
    print(f"receipt_no: {outcome.receipt_no}")
    print(f"dest: {outcome.dest}")
    print(f"sha256: {outcome.sha256}")
    print(f"size_bytes: {outcome.size_bytes}")
    print(f"version_count: {outcome.version_count}")
    if outcome.current:
        print(f"last_success_verified_at: {outcome.current.get('last_success_verified_at')}")
        print(f"last_attempt_at: {outcome.current.get('last_attempt_at')}")
    if outcome.status == "failed":
        print(f"error_type: {outcome.error_type}")
        print(f"error_message: {outcome.error_message}")


def _print_batch_summary(summary: BatchSummary) -> None:
    print("---")
    print(f"run_id: {summary.run_id}")
    print(f"list_complete: {summary.list_complete}")
    print(f"selected: {summary.selected}")
    print(f"processed: {summary.processed}")
    print(f"created: {summary.created}")
    print(f"unchanged: {summary.unchanged}")
    print(f"updated: {summary.updated}")
    print(f"repaired: {summary.repaired}")
    print(f"failed: {summary.failed}")
    print(f"deferred: {summary.deferred}")
    print(f"aborted: {summary.aborted}")
    print(f"batch_ok: {summary.ok}")


def _write_run_summary(output_dir: Path, summary: BatchSummary) -> None:
    runs = output_dir / "dart" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    path = runs / f"{summary.run_id}-summary.json"
    path.write_text(
        json.dumps(
            {
                "run_id": summary.run_id,
                "list_complete": summary.list_complete,
                "selected": summary.selected,
                "processed": summary.processed,
                "created": summary.created,
                "unchanged": summary.unchanged,
                "updated": summary.updated,
                "repaired": summary.repaired,
                "failed": summary.failed,
                "deferred": summary.deferred,
                "aborted": summary.aborted,
                "batch_ok": summary.ok,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

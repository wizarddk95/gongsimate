"""Versioned sync contracts. External HTTP is mocked."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import httpx
import pytest

from disclosure_agent.cli import load_dotenv_files, run_cli
from disclosure_agent.dart import fetch_document
from disclosure_agent.errors import (
    DartHttpError,
    DartZipError,
    SizeLimitError,
)
from disclosure_agent.storage import (
    CURRENT_FILENAME,
    ORIGINAL_FILENAME,
    apply_remote_success,
    artifact_dir,
    inspect_local,
    list_version_ids,
    sha256_bytes,
)
from disclosure_agent.sync import sync_receipt

RECEIPT = "20190401004781"
SECRET = "super-secret-key-abc123"


@pytest.fixture(autouse=True)
def isolate_secrets(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.name != "test_dotenv_does_not_override_existing_env":
        monkeypatch.setattr("disclosure_agent.cli.load_dotenv_files", lambda: None)
    monkeypatch.delenv("OPENDART_API_KEY", raising=False)
    monkeypatch.delenv("DART_API_KEY", raising=False)


def make_zip(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def api_error_xml(status: str = "013", message: str = "조회된 데이타가 없습니다.") -> bytes:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f"<result><status>{status}</status><message>{message}</message></result>"
    ).encode()


def write_legacy(dest: Path, receipt_no: str, zip_bytes: bytes) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    dest.joinpath(ORIGINAL_FILENAME).write_bytes(zip_bytes)
    meta = {
        "schema_version": 1,
        "source": "dart",
        "receipt_no": receipt_no,
        "artifact_type": "disclosure_original_zip",
        "source_endpoint": "https://opendart.fss.or.kr/api/document.xml",
        "fetched_at": "2026-01-01T00:00:00+00:00",
        "sha256": sha256_bytes(zip_bytes),
        "size_bytes": len(zip_bytes),
        "original_filename": ORIGINAL_FILENAME,
    }
    dest.joinpath("metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def test_created_stores_bytes_and_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    zip_bytes = make_zip({"a.txt": b"disclosure-body"})
    monkeypatch.setenv("OPENDART_API_KEY", "test-key")
    calls = {"n": 0}

    def fetch(receipt_no: str, api_key: str) -> bytes:
        calls["n"] += 1
        return zip_bytes

    result = sync_receipt(RECEIPT, tmp_path, "test-key", fetch=fetch)
    assert result.outcome.status == "created"
    assert calls["n"] == 1
    dest = artifact_dir(tmp_path, RECEIPT)
    version = dest / "versions" / sha256_bytes(zip_bytes) / ORIGINAL_FILENAME
    assert version.read_bytes() == zip_bytes
    current = json.loads((dest / CURRENT_FILENAME).read_text(encoding="utf-8"))
    assert current["current_sha256"] == sha256_bytes(zip_bytes)
    assert current["last_success_verified_at"]


def test_local_ok_still_hits_remote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    zip_bytes = make_zip({"a.txt": b"same"})
    monkeypatch.setenv("OPENDART_API_KEY", "test-key")
    first = sync_receipt(RECEIPT, tmp_path, "test-key", fetch=lambda *_: zip_bytes)
    first_verified = first.outcome.current["last_success_verified_at"]
    versions_before = list_version_ids(artifact_dir(tmp_path, RECEIPT))

    calls = {"n": 0}

    def fetch(receipt_no: str, api_key: str) -> bytes:
        calls["n"] += 1
        return zip_bytes

    second = sync_receipt(RECEIPT, tmp_path, "test-key", fetch=fetch)
    assert calls["n"] == 1
    assert second.outcome.status == "unchanged"
    assert list_version_ids(artifact_dir(tmp_path, RECEIPT)) == versions_before
    assert second.outcome.current["last_success_verified_at"] != first_verified
    # first_fetched_at preserved on version
    version_meta = json.loads(
        (
            artifact_dir(tmp_path, RECEIPT)
            / "versions"
            / sha256_bytes(zip_bytes)
            / "version.json"
        ).read_text(encoding="utf-8")
    )
    assert version_meta["first_fetched_at"]


def test_updated_keeps_previous_version(tmp_path: Path) -> None:
    zip_a = make_zip({"a.txt": b"v1"})
    zip_b = make_zip({"a.txt": b"v2-changed"})
    sync_receipt(RECEIPT, tmp_path, "k", fetch=lambda *_: zip_a)
    result = sync_receipt(RECEIPT, tmp_path, "k", fetch=lambda *_: zip_b)
    assert result.outcome.status == "updated"
    dest = artifact_dir(tmp_path, RECEIPT)
    assert set(list_version_ids(dest)) == {sha256_bytes(zip_a), sha256_bytes(zip_b)}
    current = json.loads((dest / CURRENT_FILENAME).read_text(encoding="utf-8"))
    assert current["current_sha256"] == sha256_bytes(zip_b)
    assert (dest / "versions" / sha256_bytes(zip_a) / ORIGINAL_FILENAME).is_file()


def test_remote_failure_keeps_current_and_success_time(tmp_path: Path) -> None:
    zip_bytes = make_zip({"a.txt": b"keep"})
    ok = sync_receipt(RECEIPT, tmp_path, "k", fetch=lambda *_: zip_bytes)
    success_at = ok.outcome.current["last_success_verified_at"]

    def boom(*_a: object, **_k: object) -> bytes:
        raise DartHttpError("OpenDART request timed out")

    failed = sync_receipt(RECEIPT, tmp_path, "k", fetch=boom)
    assert failed.outcome.status == "failed"
    current = json.loads(
        (artifact_dir(tmp_path, RECEIPT) / CURRENT_FILENAME).read_text(encoding="utf-8")
    )
    assert current["current_sha256"] == sha256_bytes(zip_bytes)
    assert current["last_success_verified_at"] == success_at
    assert current["last_attempt_at"] != success_at
    assert current["last_result"] == "failed"


def test_http_200_api_error_xml_not_saved(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=api_error_xml())

    client = httpx.Client(transport=httpx.MockTransport(handler))

    def fetch(receipt_no: str, api_key: str) -> bytes:
        return fetch_document(receipt_no, api_key, client=client, max_retries=0)

    result = sync_receipt(RECEIPT, tmp_path, "k", fetch=fetch)
    assert result.outcome.status == "failed"
    assert result.outcome.error_type == "DartApiError"
    dest = artifact_dir(tmp_path, RECEIPT)
    assert list_version_ids(dest) == []


def test_damaged_local_then_remote_ok_is_repaired(tmp_path: Path) -> None:
    dest = artifact_dir(tmp_path, RECEIPT)
    dest.mkdir(parents=True)
    dest.joinpath(ORIGINAL_FILENAME).write_bytes(make_zip({"a.txt": b"partial"}))
    # missing metadata → damaged
    assert inspect_local(dest, RECEIPT).kind == "damaged"
    zip_bytes = make_zip({"a.txt": b"fixed"})
    result = sync_receipt(RECEIPT, tmp_path, "k", fetch=lambda *_: zip_bytes)
    assert result.outcome.status == "repaired"
    current = json.loads((dest / CURRENT_FILENAME).read_text(encoding="utf-8"))
    assert current["current_sha256"] == sha256_bytes(zip_bytes)
    assert any((dest / "quarantine").glob("*"))


def test_save_interrupt_does_not_publish_partial_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = artifact_dir(tmp_path, RECEIPT)
    original = Path.write_text

    def fail_version(self: Path, *args: object, **kwargs: object) -> int:
        if self.name == "version.json":
            raise OSError("disk full")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_version)
    with pytest.raises(OSError, match="disk full"):
        apply_remote_success(
            dest, RECEIPT, make_zip({"a.txt": b"payload"}), run_id="run1"
        )
    assert not (dest / CURRENT_FILENAME).exists()
    assert list_version_ids(dest) == []


def test_missing_key_and_bad_receipt_fail_before_http(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_a: object, **_k: object) -> bytes:
        raise AssertionError("HTTP must not be called")

    monkeypatch.setattr("disclosure_agent.cli.sync_receipt", boom)
    assert run_cli(["--receipt-no", "abc", "--output-dir", str(tmp_path)]) == 2
    assert run_cli(["--receipt-no", RECEIPT, "--output-dir", str(tmp_path)]) == 2


def test_secret_not_leaked_on_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"failed connecting to {request.url}", request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(DartHttpError) as caught:
        fetch_document(RECEIPT, SECRET, client=client, max_retries=0)
    text = str(caught.value)
    assert SECRET not in text
    assert "crtfc_key" not in text


def test_cli_failed_remote_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("OPENDART_API_KEY", SECRET)

    def boom(receipt_no: str, output_dir: Path, api_key: str, **kwargs: object):
        from disclosure_agent.storage import SyncOutcome
        from disclosure_agent.sync import SyncReceiptResult

        return SyncReceiptResult(
            outcome=SyncOutcome(
                status="failed",
                receipt_no=receipt_no,
                dest=artifact_dir(output_dir, receipt_no),
                sha256=None,
                size_bytes=None,
                current=None,
                error_type="DartHttpError",
                error_message="OpenDART request timed out",
            ),
            run_id="x",
        )

    monkeypatch.setattr("disclosure_agent.cli.sync_receipt", boom)
    code = run_cli(["--receipt-no", RECEIPT, "--output-dir", str(tmp_path)])
    captured = capsys.readouterr()
    assert code == 1
    assert SECRET not in captured.out + captured.err


def test_legacy_layout_not_treated_as_remote_verified(tmp_path: Path) -> None:
    zip_bytes = make_zip({"a.txt": b"legacy"})
    dest = artifact_dir(tmp_path, RECEIPT)
    write_legacy(dest, RECEIPT, zip_bytes)
    snap = inspect_local(dest, RECEIPT)
    assert snap.kind == "legacy_ok"
    assert snap.remote_verified is False
    # After real remote check with same bytes → unchanged + migrated
    result = sync_receipt(RECEIPT, tmp_path, "k", fetch=lambda *_: zip_bytes)
    assert result.outcome.status == "unchanged"
    assert not (dest / ORIGINAL_FILENAME).exists()
    assert (dest / CURRENT_FILENAME).is_file()
    current = json.loads((dest / CURRENT_FILENAME).read_text(encoding="utf-8"))
    assert current["legacy_unverified"] is False
    assert current["last_success_verified_at"]


def test_correction_receipt_is_separate_filing(tmp_path: Path) -> None:
    zip_a = make_zip({"a.txt": b"original-filing"})
    zip_b = make_zip({"a.txt": b"correction-filing"})
    sync_receipt("20190401004781", tmp_path, "k", fetch=lambda *_: zip_a)
    sync_receipt("20190401004782", tmp_path, "k", fetch=lambda *_: zip_b)
    assert list_version_ids(artifact_dir(tmp_path, "20190401004781")) == [
        sha256_bytes(zip_a)
    ]
    assert list_version_ids(artifact_dir(tmp_path, "20190401004782")) == [
        sha256_bytes(zip_b)
    ]


def test_size_limit_and_corrupt_zip() -> None:
    zip_bytes = make_zip({"a.txt": b"hello-world"})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=zip_bytes)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(SizeLimitError):
        fetch_document(
            RECEIPT, "k", client=client, max_compressed_bytes=4, max_retries=0
        )
    with pytest.raises(DartZipError):
        from disclosure_agent.dart import validate_zip

        validate_zip(b"PK\x03\x04not-a-real-zip")


def test_dotenv_does_not_override_existing_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    env_file = tmp_path / ".env"
    env_file.write_text("OPENDART_API_KEY=from-file\nOTHER=1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENDART_API_KEY", "already-set")
    monkeypatch.delenv("OTHER", raising=False)
    load_dotenv_files()
    assert os.environ.get("OPENDART_API_KEY") == "already-set"
    assert os.environ.get("OTHER") == "1"


def test_bytes_returning_to_previous_version_reuses_dir(tmp_path: Path) -> None:
    zip_a = make_zip({"a.txt": b"a"})
    zip_b = make_zip({"a.txt": b"b"})
    sync_receipt(RECEIPT, tmp_path, "k", fetch=lambda *_: zip_a)
    sync_receipt(RECEIPT, tmp_path, "k", fetch=lambda *_: zip_b)
    again = sync_receipt(RECEIPT, tmp_path, "k", fetch=lambda *_: zip_a)
    assert again.outcome.status == "updated"
    assert again.outcome.sha256 == sha256_bytes(zip_a)
    assert set(list_version_ids(artifact_dir(tmp_path, RECEIPT))) == {
        sha256_bytes(zip_a),
        sha256_bytes(zip_b),
    }


def test_batch_summary_ok_ignores_deferred_and_bump_rejects_unknown() -> None:
    from disclosure_agent.cli import BatchSummary

    summary = BatchSummary(
        run_id="r",
        list_complete=True,
        selected=1,
        processed=1,
        deferred=3,
    )
    summary.bump("created")
    assert summary.ok is True
    summary.failed = 1
    assert summary.ok is False
    with pytest.raises(ValueError, match="unknown sync status"):
        summary.bump("skipped")


def test_list_stage_failure_still_writes_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import argparse

    from disclosure_agent.cli import run_company_sync
    from disclosure_agent.discovery import CorpRecord
    from disclosure_agent.errors import RateLimitError

    monkeypatch.setenv("OPENDART_API_KEY", "test-key")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "dart").mkdir()
    (cache_dir / "dart" / "corp_codes.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fetched_at": "2026-01-01T00:00:00+00:00",
                "count": 1,
                "companies": [
                    {
                        "corp_code": "00126380",
                        "corp_name": "삼성전자",
                        "corp_eng_name": "",
                        "stock_code": "005930",
                        "modify_date": "",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def boom(*_args: object, **_kwargs: object) -> None:
        raise RateLimitError("limited")

    monkeypatch.setattr("disclosure_agent.cli.discover_filings", boom)
    monkeypatch.setattr(
        "disclosure_agent.cli.resolve_corp",
        lambda *a, **k: (
            CorpRecord("00126380", "삼성전자", "", "005930", ""),
            {"fetched_at": "2026-01-01T00:00:00+00:00"},
        ),
    )

    args = argparse.Namespace(
        stock_code="005930",
        corp_code=None,
        bgn_de="20240301",
        end_de="20240331",
        max_filings=None,
        output_dir=tmp_path / "raw",
        cache_dir=cache_dir,
    )
    summary = run_company_sync(args)
    assert summary.aborted == "limited"
    assert summary.list_complete is False
    assert summary.ok is False
    path = tmp_path / "raw" / "dart" / "runs" / f"{summary.run_id}-summary.json"
    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["aborted"] == "limited"
    assert payload["batch_ok"] is False

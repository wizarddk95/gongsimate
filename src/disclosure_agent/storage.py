"""Versioned filing storage, check history, and TASK-001 legacy migration."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from disclosure_agent.dart import DOCUMENT_ENDPOINT, validate_zip

SCHEMA_VERSION = 2
LEGACY_SCHEMA_VERSION = 1
SOURCE = "dart"
ARTIFACT_TYPE = "disclosure_original_zip"
ORIGINAL_FILENAME = "document.zip"
CURRENT_FILENAME = "current.json"
CHECKS_FILENAME = "checks.jsonl"

LEGACY_REQUIRED_FIELDS = (
    "schema_version",
    "source",
    "receipt_no",
    "artifact_type",
    "source_endpoint",
    "fetched_at",
    "sha256",
    "size_bytes",
    "original_filename",
)


@dataclass(frozen=True)
class LocalSnapshot:
    """Readable local state before a remote check."""

    kind: Literal["missing", "ok", "legacy_ok", "damaged"]
    sha256: str | None = None
    size_bytes: int | None = None
    first_fetched_at: str | None = None
    remote_verified: bool = False
    message: str | None = None
    zip_bytes: bytes | None = None


@dataclass(frozen=True)
class SyncOutcome:
    status: Literal["created", "unchanged", "updated", "repaired", "failed"]
    receipt_no: str
    dest: Path
    sha256: str | None
    size_bytes: int | None
    current: dict[str, Any] | None
    error_type: str | None = None
    error_message: str | None = None
    version_count: int = 0


def artifact_dir(output_dir: Path, receipt_no: str) -> Path:
    return output_dir / "dart" / receipt_no


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def inspect_local(dest: Path, receipt_no: str) -> LocalSnapshot:
    """Classify local state. Never claims remote freshness."""
    if not dest.exists():
        return LocalSnapshot(kind="missing")
    if not dest.is_dir():
        return LocalSnapshot(kind="damaged", message="result path is not a directory")

    current_path = dest / CURRENT_FILENAME
    if current_path.is_file():
        return _inspect_versioned(dest, receipt_no)

    legacy_zip = dest / ORIGINAL_FILENAME
    legacy_meta = dest / "metadata.json"
    if legacy_zip.is_file() or legacy_meta.is_file():
        return _inspect_legacy(dest, receipt_no)

    versions = dest / "versions"
    if versions.exists():
        return LocalSnapshot(
            kind="damaged",
            message="versioned layout is missing current.json",
        )
    # Empty / unknown directory.
    if any(dest.iterdir()):
        return LocalSnapshot(kind="damaged", message="incomplete result directory")
    return LocalSnapshot(kind="missing")


def list_version_ids(dest: Path) -> list[str]:
    versions = dest / "versions"
    if not versions.is_dir():
        return []
    return sorted(path.name for path in versions.iterdir() if path.is_dir())


def apply_remote_success(
    dest: Path,
    receipt_no: str,
    zip_bytes: bytes,
    *,
    run_id: str,
    observed_at: datetime | None = None,
    listing: dict[str, Any] | None = None,
) -> SyncOutcome:
    """Stage remote ZIP, then update current pointer. Preserves prior good versions."""
    now = _aware(observed_at or datetime.now(UTC))
    digest = sha256_bytes(zip_bytes)
    size = len(zip_bytes)
    local = inspect_local(dest, receipt_no)

    if local.kind == "damaged":
        _quarantine_damaged(dest, run_id=run_id, reason=local.message or "damaged")
        status: Literal["created", "unchanged", "updated", "repaired"] = "repaired"
    elif local.kind == "missing":
        status = "created"
    elif local.sha256 == digest:
        status = "unchanged"
    else:
        status = "updated"

    # Preserve a readable TASK-001 legacy ZIP as its own version before replacing.
    if (
        local.kind == "legacy_ok"
        and local.zip_bytes is not None
        and local.sha256
        and local.sha256 != digest
    ):
        legacy_version = {
            "schema_version": SCHEMA_VERSION,
            "source": SOURCE,
            "receipt_no": receipt_no,
            "artifact_type": ARTIFACT_TYPE,
            "source_endpoint": DOCUMENT_ENDPOINT,
            "sha256": local.sha256,
            "size_bytes": local.size_bytes,
            "original_filename": ORIGINAL_FILENAME,
            "first_fetched_at": local.first_fetched_at or now.isoformat(),
            "remote_verified": False,
            "listing": {},
        }
        if not (dest / "versions" / local.sha256).is_dir():
            _write_version_dir(dest, local.sha256, local.zip_bytes, legacy_version)

    # Reuse an existing identical version directory when possible.
    version_dir = dest / "versions" / digest
    first_fetched_at = now.isoformat()
    if version_dir.is_dir():
        existing_meta = _read_json(version_dir / "version.json")
        if isinstance(existing_meta, dict) and existing_meta.get("first_fetched_at"):
            first_fetched_at = str(existing_meta["first_fetched_at"])
        elif local.first_fetched_at and local.sha256 == digest:
            first_fetched_at = local.first_fetched_at
    elif local.kind in {"ok", "legacy_ok"} and local.sha256 == digest and local.first_fetched_at:
        first_fetched_at = local.first_fetched_at

    version_meta = {
        "schema_version": SCHEMA_VERSION,
        "source": SOURCE,
        "receipt_no": receipt_no,
        "artifact_type": ARTIFACT_TYPE,
        "source_endpoint": DOCUMENT_ENDPOINT,
        "sha256": digest,
        "size_bytes": size,
        "original_filename": ORIGINAL_FILENAME,
        "first_fetched_at": first_fetched_at,
        "remote_verified": True,
        "listing": listing or {},
    }

    dest.mkdir(parents=True, exist_ok=True)
    if not version_dir.is_dir():
        _write_version_dir(dest, digest, zip_bytes, version_meta)

    # Migrate away from legacy root files after a successful remote check.
    _remove_legacy_root_files(dest)

    previous_sha = None
    previous_success = None
    current_existing = _read_json(dest / CURRENT_FILENAME)
    if isinstance(current_existing, dict):
        previous_sha = current_existing.get("current_sha256")
        previous_success = current_existing.get("last_success_verified_at")

    current = {
        "schema_version": SCHEMA_VERSION,
        "source": SOURCE,
        "receipt_no": receipt_no,
        "current_sha256": digest,
        "current_size_bytes": size,
        "last_success_verified_at": now.isoformat(),
        "last_attempt_at": now.isoformat(),
        "last_result": status,
        "legacy_unverified": False,
    }
    # unchanged must not rewrite first_fetched_at (already preserved in version meta).
    if status == "unchanged" and previous_success and previous_sha == digest:
        # Keep prior success timestamp? Task: update remote verification time on unchanged.
        # So last_success_verified_at SHOULD update. first_fetched_at must not.
        pass

    _atomic_write_json(dest / CURRENT_FILENAME, current)
    _append_check(
        dest,
        {
            "run_id": run_id,
            "receipt_no": receipt_no,
            "started_at": now.isoformat(),
            "finished_at": now.isoformat(),
            "result": status,
            "observed_sha256": digest,
            "observed_size_bytes": size,
            "error_type": None,
            "error_message": None,
        },
    )
    return SyncOutcome(
        status=status,
        receipt_no=receipt_no,
        dest=dest,
        sha256=digest,
        size_bytes=size,
        current=current,
        version_count=len(list_version_ids(dest)),
    )


def record_remote_failure(
    dest: Path,
    receipt_no: str,
    *,
    run_id: str,
    error: Exception,
    observed_at: datetime | None = None,
) -> SyncOutcome:
    """Record a failed remote check without changing the current good pointer."""
    now = _aware(observed_at or datetime.now(UTC))
    dest.mkdir(parents=True, exist_ok=True)
    current = _read_json(dest / CURRENT_FILENAME)
    if not isinstance(current, dict):
        local = inspect_local(dest, receipt_no)
        current = {
            "schema_version": SCHEMA_VERSION,
            "source": SOURCE,
            "receipt_no": receipt_no,
            "current_sha256": local.sha256,
            "current_size_bytes": local.size_bytes,
            "last_success_verified_at": None,
            "last_attempt_at": now.isoformat(),
            "last_result": "failed",
            "legacy_unverified": local.kind == "legacy_ok",
        }
    else:
        current = dict(current)
        current["last_attempt_at"] = now.isoformat()
        current["last_result"] = "failed"
        # Do not touch last_success_verified_at.

    _atomic_write_json(dest / CURRENT_FILENAME, current)
    err_type = type(error).__name__
    err_msg = str(error)
    _append_check(
        dest,
        {
            "run_id": run_id,
            "receipt_no": receipt_no,
            "started_at": now.isoformat(),
            "finished_at": now.isoformat(),
            "result": "failed",
            "observed_sha256": None,
            "observed_size_bytes": None,
            "error_type": err_type,
            "error_message": err_msg,
        },
    )
    return SyncOutcome(
        status="failed",
        receipt_no=receipt_no,
        dest=dest,
        sha256=current.get("current_sha256") if isinstance(current, dict) else None,
        size_bytes=current.get("current_size_bytes") if isinstance(current, dict) else None,
        current=current if isinstance(current, dict) else None,
        error_type=err_type,
        error_message=err_msg,
        version_count=len(list_version_ids(dest)),
    )


# --- compatibility helpers used by older tests / callers ---


@dataclass(frozen=True)
class ExistingResult:
    kind: Literal["missing", "reuse", "integrity"]
    metadata: dict[str, Any] | None = None
    message: str | None = None


def inspect_existing(dest: Path, receipt_no: str) -> ExistingResult:
    """Backward-compatible classifier. 'reuse' means locally readable, not remote-fresh."""
    snap = inspect_local(dest, receipt_no)
    if snap.kind == "missing":
        return ExistingResult(kind="missing")
    if snap.kind in {"ok", "legacy_ok"}:
        meta = {
            "schema_version": SCHEMA_VERSION if snap.kind == "ok" else LEGACY_SCHEMA_VERSION,
            "source": SOURCE,
            "receipt_no": receipt_no,
            "artifact_type": ARTIFACT_TYPE,
            "source_endpoint": DOCUMENT_ENDPOINT,
            "fetched_at": snap.first_fetched_at,
            "sha256": snap.sha256,
            "size_bytes": snap.size_bytes,
            "original_filename": ORIGINAL_FILENAME,
            "remote_verified": snap.remote_verified,
        }
        return ExistingResult(kind="reuse", metadata=meta)
    return ExistingResult(kind="integrity", message=snap.message or "damaged local result")


def build_metadata(receipt_no: str, zip_bytes: bytes, fetched_at: datetime) -> dict[str, Any]:
    fetched_at = _aware(fetched_at)
    return {
        "schema_version": SCHEMA_VERSION,
        "source": SOURCE,
        "receipt_no": receipt_no,
        "artifact_type": ARTIFACT_TYPE,
        "source_endpoint": DOCUMENT_ENDPOINT,
        "fetched_at": fetched_at.isoformat(),
        "sha256": sha256_bytes(zip_bytes),
        "size_bytes": len(zip_bytes),
        "original_filename": ORIGINAL_FILENAME,
    }


def save_document(dest: Path, receipt_no: str, zip_bytes: bytes) -> dict[str, Any]:
    """Write a new remote-verified version (used by simple callers)."""
    outcome = apply_remote_success(
        dest,
        receipt_no,
        zip_bytes,
        run_id=f"save-{uuid4().hex}",
    )
    assert outcome.current is not None
    return {
        **build_metadata(receipt_no, zip_bytes, datetime.now(UTC)),
        "sha256": outcome.sha256,
        "size_bytes": outcome.size_bytes,
        "fetched_at": outcome.current.get("last_success_verified_at"),
        "status": outcome.status,
    }


def _inspect_versioned(dest: Path, receipt_no: str) -> LocalSnapshot:
    current = _read_json(dest / CURRENT_FILENAME)
    if not isinstance(current, dict):
        return LocalSnapshot(kind="damaged", message="current.json is invalid")
    digest = current.get("current_sha256")
    if not isinstance(digest, str) or not digest:
        return LocalSnapshot(kind="damaged", message="current.json missing current_sha256")
    version_dir = dest / "versions" / digest
    zip_path = version_dir / ORIGINAL_FILENAME
    meta_path = version_dir / "version.json"
    if not zip_path.is_file() or not meta_path.is_file():
        return LocalSnapshot(kind="damaged", message="current version files are missing")
    try:
        zip_bytes = zip_path.read_bytes()
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return LocalSnapshot(kind="damaged", message="current version is unreadable")
    if not isinstance(meta, dict):
        return LocalSnapshot(kind="damaged", message="version.json must be an object")
    if meta.get("receipt_no") != receipt_no:
        return LocalSnapshot(kind="damaged", message="version receipt_no mismatch")
    if meta.get("sha256") != digest or sha256_bytes(zip_bytes) != digest:
        return LocalSnapshot(kind="damaged", message="version hash mismatch")
    if meta.get("size_bytes") != len(zip_bytes):
        return LocalSnapshot(kind="damaged", message="version size mismatch")
    try:
        validate_zip(zip_bytes)
    except Exception as exc:
        return LocalSnapshot(kind="damaged", message=f"stored ZIP failed validation: {exc}")
    return LocalSnapshot(
        kind="ok",
        sha256=digest,
        size_bytes=len(zip_bytes),
        first_fetched_at=str(meta.get("first_fetched_at") or "") or None,
        remote_verified=bool(meta.get("remote_verified")),
        zip_bytes=zip_bytes,
    )


def _inspect_legacy(dest: Path, receipt_no: str) -> LocalSnapshot:
    zip_path = dest / ORIGINAL_FILENAME
    meta_path = dest / "metadata.json"
    if not zip_path.is_file() or not meta_path.is_file():
        missing = "metadata.json" if zip_path.is_file() else ORIGINAL_FILENAME
        if not zip_path.is_file() and not meta_path.is_file():
            return LocalSnapshot(kind="damaged", message="incomplete result directory")
        return LocalSnapshot(kind="damaged", message=f"partial result is missing {missing}")
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        zip_bytes = zip_path.read_bytes()
    except (OSError, UnicodeError, json.JSONDecodeError):
        return LocalSnapshot(kind="damaged", message="legacy metadata/ZIP unreadable")
    if not isinstance(metadata, dict):
        return LocalSnapshot(kind="damaged", message="metadata.json must contain a JSON object")
    problem = _legacy_mismatch(metadata, receipt_no, zip_bytes)
    if problem:
        return LocalSnapshot(kind="damaged", message=problem)
    try:
        validate_zip(zip_bytes)
    except Exception as exc:
        return LocalSnapshot(kind="damaged", message=f"stored ZIP failed validation: {exc}")
    return LocalSnapshot(
        kind="legacy_ok",
        sha256=str(metadata["sha256"]),
        size_bytes=int(metadata["size_bytes"]),
        first_fetched_at=str(metadata.get("fetched_at") or "") or None,
        remote_verified=False,
        zip_bytes=zip_bytes,
    )


def _legacy_mismatch(
    metadata: dict[str, Any],
    receipt_no: str,
    zip_bytes: bytes,
) -> str | None:
    missing = [field for field in LEGACY_REQUIRED_FIELDS if field not in metadata]
    if missing:
        return f"metadata.json missing fields: {', '.join(missing)}"
    if metadata.get("schema_version") not in {LEGACY_SCHEMA_VERSION, SCHEMA_VERSION}:
        return "metadata.json schema_version is unsupported"
    if metadata.get("source") != SOURCE:
        return "metadata.json source does not match dart"
    if metadata.get("receipt_no") != receipt_no:
        return "metadata.json receipt_no does not match the request"
    if metadata.get("artifact_type") != ARTIFACT_TYPE:
        return "metadata.json artifact_type is invalid"
    if metadata.get("source_endpoint") != DOCUMENT_ENDPOINT:
        return "metadata.json source_endpoint is invalid"
    if metadata.get("original_filename") != ORIGINAL_FILENAME:
        return "metadata.json original_filename is invalid"
    if metadata.get("size_bytes") != len(zip_bytes):
        return "stored ZIP size does not match metadata.json"
    if metadata.get("sha256") != sha256_bytes(zip_bytes):
        return "stored ZIP hash does not match metadata.json"
    return None


def _write_version_dir(
    dest: Path,
    digest: str,
    zip_bytes: bytes,
    version_meta: dict[str, Any],
) -> None:
    parent = dest / "versions"
    parent.mkdir(parents=True, exist_ok=True)
    tmp = parent / f".tmp-{digest}-{uuid4().hex}"
    final = parent / digest
    try:
        tmp.mkdir()
        (tmp / ORIGINAL_FILENAME).write_bytes(zip_bytes)
        (tmp / "version.json").write_text(
            json.dumps(version_meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.rename(tmp, final)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def _quarantine_damaged(dest: Path, *, run_id: str, reason: str) -> None:
    quarantine = dest / "quarantine" / f"{run_id}-{uuid4().hex[:8]}"
    quarantine.mkdir(parents=True, exist_ok=True)
    note = {"reason": reason, "run_id": run_id}
    (quarantine / "note.json").write_text(
        json.dumps(note, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for name in (ORIGINAL_FILENAME, "metadata.json", CURRENT_FILENAME):
        src = dest / name
        if src.exists():
            shutil.move(str(src), str(quarantine / name))


def _remove_legacy_root_files(dest: Path) -> None:
    for name in (ORIGINAL_FILENAME, "metadata.json"):
        path = dest / name
        if path.exists():
            path.unlink()


def _append_check(dest: Path, record: dict[str, Any]) -> None:
    path = dest / CHECKS_FILENAME
    line = json.dumps(record, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp-{uuid4().hex}")
    try:
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value

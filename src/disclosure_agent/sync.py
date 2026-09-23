"""Single-receipt remote verification / sync."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from disclosure_agent.dart import fetch_document
from disclosure_agent.errors import IngestError, UsageError
from disclosure_agent.storage import (
    SyncOutcome,
    apply_remote_success,
    artifact_dir,
    record_remote_failure,
)

FetchFn = Callable[[str, str], bytes]


@dataclass(frozen=True)
class SyncReceiptResult:
    outcome: SyncOutcome
    run_id: str


def sync_receipt(
    receipt_no: str,
    output_dir: Path,
    api_key: str,
    *,
    fetch: FetchFn | None = None,
    run_id: str | None = None,
    listing: dict[str, Any] | None = None,
) -> SyncReceiptResult:
    """Always perform a remote check. Local-only success is not allowed."""
    if not api_key.strip():
        raise UsageError("OPENDART_API_KEY is required for remote verification")
    rid = run_id or uuid4().hex
    dest = artifact_dir(output_dir, receipt_no)
    fetch_fn = fetch or (lambda r, k: fetch_document(r, k))
    started = datetime.now(UTC)
    try:
        zip_bytes = fetch_fn(receipt_no, api_key)
        outcome = apply_remote_success(
            dest,
            receipt_no,
            zip_bytes,
            run_id=rid,
            observed_at=started,
            listing=listing,
        )
        return SyncReceiptResult(outcome=outcome, run_id=rid)
    except IngestError as exc:
        outcome = record_remote_failure(
            dest,
            receipt_no,
            run_id=rid,
            error=exc,
            observed_at=started,
        )
        return SyncReceiptResult(outcome=outcome, run_id=rid)

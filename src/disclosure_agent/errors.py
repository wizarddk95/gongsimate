"""User-facing ingest/sync errors that must never include API keys or signed URLs."""


class IngestError(Exception):
    """Base error for dart CLI tools. Safe to print."""

    exit_code = 1


class UsageError(IngestError):
    """Invalid arguments or missing configuration. No HTTP call is made."""

    exit_code = 2


class IntegrityError(IngestError):
    """Existing result is incomplete or does not match stored metadata."""


class DartApiError(IngestError):
    """OpenDART returned an API-level error payload instead of expected data."""

    def __init__(self, message: str, *, status: str | None = None) -> None:
        super().__init__(message)
        self.status = status


class DartHttpError(IngestError):
    """HTTP transport or non-success HTTP status from OpenDART."""


class DartZipError(IngestError):
    """Response was not a usable ZIP archive."""


class SizeLimitError(IngestError):
    """Compressed or uncompressed payload exceeded the local safety limit."""


class RateLimitError(IngestError):
    """OpenDART request limit exceeded; stop further batch calls."""

    exit_code = 1


class CorpMappingError(IngestError):
    """stock_code ↔ corp_code mapping failed or was ambiguous."""


class FatalBatchError(IngestError):
    """Authentication or other fatal error that should abort a batch."""

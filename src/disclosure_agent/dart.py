"""OpenDART HTTP helpers: document ZIP, corp codes, disclosure list, retries."""

from __future__ import annotations

import io
import json
import time
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final, TypeVar
from urllib.parse import quote
from xml.etree.ElementTree import ParseError, fromstring

import httpx

from disclosure_agent.errors import (
    DartApiError,
    DartHttpError,
    DartZipError,
    FatalBatchError,
    RateLimitError,
    SizeLimitError,
)

DOCUMENT_ENDPOINT: Final = "https://opendart.fss.or.kr/api/document.xml"
CORP_CODE_ENDPOINT: Final = "https://opendart.fss.or.kr/api/corpCode.xml"
LIST_ENDPOINT: Final = "https://opendart.fss.or.kr/api/list.json"
HTTP_TIMEOUT_SECONDS: Final = 30.0
MAX_COMPRESSED_BYTES: Final = 50 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES: Final = 200 * 1024 * 1024
MAX_CORP_CODE_BYTES: Final = 50 * 1024 * 1024
DEFAULT_MAX_RETRIES: Final = 3
DEFAULT_BACKOFF_SECONDS: Final = 1.0

_ZIP_MAGIC = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
_XML_BOM = b"\xef\xbb\xbf"

# Auth / client / input errors — do not retry.
_FATAL_API_STATUS = frozenset({"010", "011", "012", "100", "101", "901"})
_RATE_LIMIT_STATUS = frozenset({"020"})
# Transient OpenDART / transport issues that may clear on retry.
_RETRYABLE_API_STATUS = frozenset({"800", "900"})

T = TypeVar("T")


@dataclass(frozen=True)
class DisclosureListItem:
    corp_code: str
    corp_name: str
    stock_code: str
    report_nm: str
    rcept_no: str
    flr_nm: str
    rcept_dt: str
    rm: str
    corp_cls: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class DisclosureListPage:
    status: str
    message: str
    page_no: int
    page_count: int
    total_count: int
    total_page: int
    items: list[DisclosureListItem]


def redact_secret(text: str, secret: str) -> str:
    """Remove a secret and its URL-encoded form from text."""
    if not secret:
        return text
    redacted = text.replace(secret, "<redacted>")
    encoded = quote(secret, safe="")
    if encoded and encoded != secret:
        redacted = redacted.replace(encoded, "<redacted>")
    return redacted


def with_retries[T](
    operation: Callable[[], T],
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Retry transient HTTP/API failures. Never retries auth, usage, or rate limits."""
    attempt = 0
    while True:
        try:
            return operation()
        except RateLimitError:
            raise
        except FatalBatchError:
            raise
        except (DartHttpError, DartApiError) as exc:
            if isinstance(exc, DartApiError) and not _is_retryable_api(exc):
                raise
            if attempt >= max_retries:
                raise
            sleep(backoff_seconds * (2**attempt))
            attempt += 1


def fetch_document(
    receipt_no: str,
    api_key: str,
    *,
    client: httpx.Client | None = None,
    timeout: float = HTTP_TIMEOUT_SECONDS,
    max_compressed_bytes: int = MAX_COMPRESSED_BYTES,
    max_uncompressed_bytes: int = MAX_UNCOMPRESSED_BYTES,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> bytes:
    """Download one original disclosure ZIP. Never returns API error XML as ZIP bytes."""

    def once() -> bytes:
        return _fetch_document_once(
            receipt_no,
            api_key,
            client=client,
            timeout=timeout,
            max_compressed_bytes=max_compressed_bytes,
            max_uncompressed_bytes=max_uncompressed_bytes,
        )

    return with_retries(once, max_retries=max_retries)


def fetch_corp_code_zip(
    api_key: str,
    *,
    client: httpx.Client | None = None,
    timeout: float = HTTP_TIMEOUT_SECONDS,
    max_compressed_bytes: int = MAX_CORP_CODE_BYTES,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> bytes:
    """Download the official corpCode.xml ZIP mapping archive."""

    def once() -> bytes:
        return _get_binary(
            CORP_CODE_ENDPOINT,
            {"crtfc_key": api_key},
            api_key,
            client=client,
            timeout=timeout,
            max_compressed_bytes=max_compressed_bytes,
        )

    return with_retries(once, max_retries=max_retries)


def fetch_disclosure_list_page(
    api_key: str,
    *,
    corp_code: str,
    bgn_de: str,
    end_de: str,
    page_no: int = 1,
    page_count: int = 100,
    last_reprt_at: str = "N",
    pblntf_ty: str = "A",
    client: httpx.Client | None = None,
    timeout: float = HTTP_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> DisclosureListPage:
    """Fetch one page of the disclosure list JSON API."""

    def once() -> DisclosureListPage:
        return _fetch_list_page_once(
            api_key,
            corp_code=corp_code,
            bgn_de=bgn_de,
            end_de=end_de,
            page_no=page_no,
            page_count=page_count,
            last_reprt_at=last_reprt_at,
            pblntf_ty=pblntf_ty,
            client=client,
            timeout=timeout,
        )

    return with_retries(once, max_retries=max_retries)


def interpret_document_body(
    body: bytes,
    *,
    max_uncompressed_bytes: int = MAX_UNCOMPRESSED_BYTES,
) -> bytes:
    """Classify an HTTP 200 body as ZIP, OpenDART XML error, or unexpected data."""
    if not body:
        raise DartZipError("OpenDART returned an empty response")
    if body.startswith(_ZIP_MAGIC):
        validate_zip(body, max_uncompressed_bytes=max_uncompressed_bytes)
        return body
    xml_candidate = body[len(_XML_BOM) :] if body.startswith(_XML_BOM) else body
    stripped = xml_candidate.lstrip()
    if stripped.startswith(b"<?xml") or stripped.startswith(b"<"):
        raise _api_error_from_xml(stripped)
    raise DartZipError("OpenDART returned an unexpected non-ZIP response")


def validate_zip(
    data: bytes,
    *,
    max_uncompressed_bytes: int = MAX_UNCOMPRESSED_BYTES,
) -> None:
    """Open the ZIP and check integrity without extracting files to disk."""
    if not data:
        raise DartZipError("ZIP archive is empty")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            infos = archive.infolist()
            if not infos:
                raise DartZipError("ZIP archive contains no files")
            total_uncompressed = 0
            for info in infos:
                if info.file_size < 0:
                    raise DartZipError("ZIP metadata is invalid")
                total_uncompressed += info.file_size
                if total_uncompressed > max_uncompressed_bytes:
                    raise SizeLimitError(
                        "Uncompressed ZIP contents exceed "
                        f"{max_uncompressed_bytes} bytes"
                    )
            bad_name = archive.testzip()
            if bad_name is not None:
                raise DartZipError("ZIP archive is corrupted")
    except SizeLimitError:
        raise
    except DartZipError:
        raise
    except zipfile.BadZipFile:
        raise DartZipError("Response is not a valid ZIP archive") from None


def parse_corp_code_xml(xml_bytes: bytes) -> list[dict[str, str]]:
    """Parse CORPCODE.xml (or similar) bytes into corp mapping rows."""
    try:
        root = fromstring(xml_bytes)
    except ParseError as exc:
        raise DartZipError("corp code XML is not well-formed") from exc
    rows: list[dict[str, str]] = []
    for node in root.iter():
        tag = _local(node.tag)
        if tag not in {"list", "item"}:
            continue
        corp_code = _child_text(node, "corp_code")
        if not corp_code:
            continue
        rows.append(
            {
                "corp_code": corp_code,
                "corp_name": _child_text(node, "corp_name"),
                "corp_eng_name": _child_text(node, "corp_eng_name"),
                "stock_code": _child_text(node, "stock_code").strip(),
                "modify_date": _child_text(node, "modify_date"),
            }
        )
    if not rows:
        raise DartZipError("corp code XML contained no company rows")
    return rows


def extract_corp_code_xml_from_zip(zip_bytes: bytes) -> bytes:
    """Return the first XML member from a corpCode ZIP."""
    validate_zip(zip_bytes, max_uncompressed_bytes=MAX_UNCOMPRESSED_BYTES)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".xml")]
        if not names:
            raise DartZipError("corp code ZIP contains no XML file")
        return archive.read(names[0])


def _fetch_document_once(
    receipt_no: str,
    api_key: str,
    *,
    client: httpx.Client | None,
    timeout: float,
    max_compressed_bytes: int,
    max_uncompressed_bytes: int,
) -> bytes:
    body = _get_binary(
        DOCUMENT_ENDPOINT,
        {"crtfc_key": api_key, "rcept_no": receipt_no},
        api_key,
        client=client,
        timeout=timeout,
        max_compressed_bytes=max_compressed_bytes,
    )
    return interpret_document_body(body, max_uncompressed_bytes=max_uncompressed_bytes)


def _get_binary(
    url: str,
    params: Mapping[str, str],
    api_key: str,
    *,
    client: httpx.Client | None,
    timeout: float,
    max_compressed_bytes: int,
) -> bytes:
    owns_client = client is None
    if owns_client:
        client = httpx.Client(timeout=timeout, follow_redirects=False)
    assert client is not None
    try:
        try:
            with client.stream("GET", url, params=dict(params)) as response:
                body = _read_limited(response, max_compressed_bytes)
                status_code = response.status_code
        except SizeLimitError:
            raise
        except httpx.TimeoutException:
            raise DartHttpError("OpenDART request timed out") from None
        except httpx.HTTPError as exc:
            kind = type(exc).__name__
            raise DartHttpError(
                redact_secret(
                    f"OpenDART request failed due to an HTTP error ({kind})",
                    api_key,
                )
            ) from None
        if status_code != 200:
            raise DartHttpError(f"OpenDART HTTP status {status_code}")
        return body
    finally:
        if owns_client:
            client.close()


def _fetch_list_page_once(
    api_key: str,
    *,
    corp_code: str,
    bgn_de: str,
    end_de: str,
    page_no: int,
    page_count: int,
    last_reprt_at: str,
    pblntf_ty: str,
    client: httpx.Client | None,
    timeout: float,
) -> DisclosureListPage:
    owns_client = client is None
    if owns_client:
        client = httpx.Client(timeout=timeout, follow_redirects=False)
    assert client is not None
    params = {
        "crtfc_key": api_key,
        "corp_code": corp_code,
        "bgn_de": bgn_de,
        "end_de": end_de,
        "last_reprt_at": last_reprt_at,
        "pblntf_ty": pblntf_ty,
        "page_no": str(page_no),
        "page_count": str(page_count),
    }
    try:
        try:
            response = client.get(LIST_ENDPOINT, params=params)
        except httpx.TimeoutException:
            raise DartHttpError("OpenDART list request timed out") from None
        except httpx.HTTPError as exc:
            kind = type(exc).__name__
            raise DartHttpError(
                redact_secret(
                    f"OpenDART list request failed due to an HTTP error ({kind})",
                    api_key,
                )
            ) from None
        if response.status_code != 200:
            raise DartHttpError(f"OpenDART HTTP status {response.status_code}")
        try:
            payload = response.json()
        except json.JSONDecodeError:
            raise DartApiError("OpenDART list response was not valid JSON") from None
        return _parse_list_payload(payload)
    finally:
        if owns_client:
            client.close()


def _parse_list_payload(payload: Any) -> DisclosureListPage:
    if not isinstance(payload, dict):
        raise DartApiError("OpenDART list response was not a JSON object")
    # Some responses nest under "result".
    data = payload.get("result") if isinstance(payload.get("result"), dict) else payload
    if not isinstance(data, dict):
        raise DartApiError("OpenDART list response was unexpected")
    status = str(data.get("status") or "").strip()
    message = str(data.get("message") or "").strip()
    if status in _RATE_LIMIT_STATUS:
        raise RateLimitError(f"OpenDART API error {status}: {message or 'rate limited'}")
    if status in _FATAL_API_STATUS:
        raise FatalBatchError(f"OpenDART API error {status}: {message or 'fatal'}")
    if status and status not in {"000", "013"} and status not in _RETRYABLE_API_STATUS:
        raise DartApiError(
            f"OpenDART API error {status}: {message or 'no message'}",
            status=status,
        )
    if status in _RETRYABLE_API_STATUS:
        raise DartApiError(
            f"OpenDART API error {status}: {message or 'transient'}",
            status=status,
        )
    # 013 = no data — treat as empty success page.
    raw_list = data.get("list") or []
    if status == "013":
        raw_list = []
    if not isinstance(raw_list, list):
        raise DartApiError("OpenDART list payload list field is invalid")
    items: list[DisclosureListItem] = []
    for entry in raw_list:
        if not isinstance(entry, dict):
            continue
        rcept_no = str(entry.get("rcept_no") or "").strip()
        if not rcept_no:
            continue
        items.append(
            DisclosureListItem(
                corp_code=str(entry.get("corp_code") or "").strip(),
                corp_name=str(entry.get("corp_name") or "").strip(),
                stock_code=str(entry.get("stock_code") or "").strip(),
                report_nm=str(entry.get("report_nm") or "").strip(),
                rcept_no=rcept_no,
                flr_nm=str(entry.get("flr_nm") or "").strip(),
                rcept_dt=str(entry.get("rcept_dt") or "").strip(),
                rm=str(entry.get("rm") or "").strip(),
                corp_cls=str(entry.get("corp_cls") or "").strip(),
                raw=dict(entry),
            )
        )
    return DisclosureListPage(
        status=status or "000",
        message=message,
        page_no=_as_int(data.get("page_no"), 1),
        page_count=_as_int(data.get("page_count"), len(items)),
        total_count=_as_int(data.get("total_count"), len(items)),
        total_page=_as_int(data.get("total_page"), 1),
        items=items,
    )


def _read_limited(response: httpx.Response, max_compressed_bytes: int) -> bytes:
    content_length = response.headers.get("content-length")
    if content_length is not None:
        try:
            declared = int(content_length)
        except ValueError:
            declared = None
        else:
            if declared > max_compressed_bytes:
                raise SizeLimitError(
                    f"Compressed response exceeded {max_compressed_bytes} bytes"
                )
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_bytes():
        total += len(chunk)
        if total > max_compressed_bytes:
            raise SizeLimitError(
                f"Compressed response exceeded {max_compressed_bytes} bytes"
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _api_error_from_xml(payload: bytes) -> DartApiError | RateLimitError | FatalBatchError:
    try:
        root = fromstring(payload)
    except ParseError:
        return DartApiError("OpenDART returned an unexpected XML response")
    status = (root.findtext("status") or root.findtext(".//status") or "").strip()
    message = (root.findtext("message") or root.findtext(".//message") or "").strip()
    if not status and not message:
        return DartApiError("OpenDART returned an unexpected XML response")
    detail = message or "no message"
    code = status or "unknown"
    text = f"OpenDART API error {code}: {detail}"
    if status in _RATE_LIMIT_STATUS:
        return RateLimitError(text)
    if status in _FATAL_API_STATUS:
        return FatalBatchError(text)
    return DartApiError(text, status=status or None)


def _is_retryable_api(exc: DartApiError) -> bool:
    return exc.status in _RETRYABLE_API_STATUS if exc.status else False


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _local(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _child_text(node: Any, name: str) -> str:
    for child in list(node):
        if _local(child.tag) == name:
            return (child.text or "").strip()
    return ""

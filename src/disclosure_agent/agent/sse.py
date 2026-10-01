"""SSE framing. Event payloads are JSON objects on a single data line."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StreamEvent:
    event: str
    data: dict[str, Any]


def format_sse(event: StreamEvent) -> str:
    payload = json.dumps(event.data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event.event}\ndata: {payload}\n\n"

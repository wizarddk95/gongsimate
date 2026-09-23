"""Single-writer run lock (local filesystem only; not distributed)."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

from disclosure_agent.errors import UsageError


@dataclass
class RunLock:
    path: Path
    fd: int | None = None

    def acquire(self, *, timeout_seconds: float = 0.0) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self.fd, f"pid={os.getpid()}\n".encode())
                return
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise UsageError(
                        f"another dart sync appears to be running (lock: {self.path}). "
                        "Single-writer only; remove the lock if no process holds it."
                    ) from None
                time.sleep(0.2)

    def release(self) -> None:
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass

    def __enter__(self) -> RunLock:
        self.acquire()
        return self

    def __exit__(self, *args: object) -> None:
        self.release()


def output_run_lock(output_dir: Path) -> RunLock:
    return RunLock(path=output_dir / "dart" / ".sync.lock")

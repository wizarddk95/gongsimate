"""User-facing agent runtime. Retrieval experiments stay in their own modules."""

from __future__ import annotations

__all__ = ["build_default_runtime"]


def __getattr__(name: str) -> object:
    if name == "build_default_runtime":
        from disclosure_agent.agent.runtime import build_default_runtime

        return build_default_runtime
    raise AttributeError(name)

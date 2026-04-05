"""Console helpers for Windows-friendly script output."""

from __future__ import annotations

import os
import sys


def configure_console_output() -> None:
    """Prefer UTF-8 output so scripts do not crash on emoji/unicode in Windows terminals."""
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

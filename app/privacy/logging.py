"""Logging setup with a redaction filter so personal data never reaches log output."""
from __future__ import annotations

import logging

from app.privacy.pii import redact_for_log


class PIIRedactionFilter(logging.Filter):
    """Rewrites every record's message with names, emails and phone numbers redacted."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - malformed format args
            message = str(record.msg)
        record.msg = redact_for_log(message)
        record.args = None
        return True


def setup_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    root.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    handler.addFilter(PIIRedactionFilter())
    root.handlers = [handler]
    # uvicorn installs its own handlers; route them through ours.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True

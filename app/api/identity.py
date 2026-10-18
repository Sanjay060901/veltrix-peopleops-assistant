"""Optional verification that X-Employee-Id was set by a trusted login layer.

The assignment treats X-Employee-Id as already authenticated, so by default the header is
trusted as-is (header mode). In production the service sits behind a login gateway; setting
IDENTITY_SECRET (shared only with that gateway) turns on signed mode:

    X-Employee-Signature: t=<unix seconds>,v1=<hex HMAC-SHA256(secret, "<employee_id>.<t>")>

A browser user who types someone else's ID cannot produce the signature. The timestamp bounds
replay of a captured header to IDENTITY_MAX_AGE_SECONDS.
"""
from __future__ import annotations

import hashlib
import hmac
import re
import time

MIN_SECRET_LENGTH = 32
_HEADER = re.compile(r"^t=(\d{1,12}),v1=([0-9a-f]{64})$")


def sign(secret: str, employee_id: str, timestamp: int | None = None) -> str:
    """Produce the header value. Used by the login layer (and by tests and scripts)."""
    ts = int(time.time()) if timestamp is None else int(timestamp)
    mac = hmac.new(secret.encode(), f"{employee_id}.{ts}".encode(), hashlib.sha256).hexdigest()
    return f"t={ts},v1={mac}"


def verify(secret: str, employee_id: str, header: str | None, max_age: int, now: float | None = None) -> bool:
    if not header:
        return False
    match = _HEADER.match(header.strip())
    if not match:
        return False
    ts, mac = int(match.group(1)), match.group(2)
    current = time.time() if now is None else now
    if abs(current - ts) > max_age:
        return False
    expected = hmac.new(secret.encode(), f"{employee_id}.{ts}".encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, mac)


def check_secret(secret: str) -> None:
    """Fail closed at startup on a weak secret rather than running with false security."""
    if secret and len(secret) < MIN_SECRET_LENGTH:
        raise ValueError(f"IDENTITY_SECRET must be at least {MIN_SECRET_LENGTH} characters")

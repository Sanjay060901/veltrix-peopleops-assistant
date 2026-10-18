"""Password hashing with scrypt (stdlib). Stored format: scrypt$n$r$p$salt_hex$hash_hex."""
from __future__ import annotations

import hashlib
import hmac
import secrets

N, R, P, DKLEN = 2**14, 8, 1, 32
MIN_LENGTH = 10


class WeakPassword(ValueError):
    pass


def check_policy(password: str) -> None:
    if len(password) < MIN_LENGTH:
        raise WeakPassword(f"password must be at least {MIN_LENGTH} characters")
    if password.lower() == password or password.upper() == password or not any(c.isdigit() for c in password):
        raise WeakPassword("password needs upper- and lower-case letters and a digit")


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=N, r=R, p=P, dklen=DKLEN)
    return f"scrypt${N}${R}${P}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt_hex, hash_hex = stored.split("$")
        if scheme != "scrypt":
            return False
        digest = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex), n=int(n), r=int(r), p=int(p), dklen=len(hash_hex) // 2)
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(digest.hex(), hash_hex)


# Verified against when the employee has no credential, so unknown IDs take as long as wrong passwords.
_DUMMY = hash_password(secrets.token_urlsafe(16))


def burn_time(password: str) -> None:
    verify_password(password, _DUMMY)

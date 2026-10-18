"""Print a signed identity header, as a login gateway would send it in signed mode.

Usage: IDENTITY_SECRET=... python scripts/sign_identity.py E2002
       curl -H "X-Employee-Id: E2002" -H "X-Employee-Signature: $(python scripts/sign_identity.py E2002)" ...
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.api.identity import sign  # noqa: E402


def main() -> int:
    secret = os.environ.get("IDENTITY_SECRET", "")
    if len(sys.argv) != 2 or not secret:
        print("usage: IDENTITY_SECRET=... python scripts/sign_identity.py <employee_id>", file=sys.stderr)
        return 2
    print(sign(secret, sys.argv[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

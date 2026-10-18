"""Admin command for login credentials (AUTH_MODE=login). There is no self sign-up.

    python -m app.auth.manage set-password E2002            # prompts twice, input hidden
    python -m app.auth.manage set-password E2002 --generate # prints a random password once
    python -m app.auth.manage remove E2002
    python -m app.auth.manage list
    docker compose exec assistant python -m app.auth.manage set-password E2002 --generate
"""
from __future__ import annotations

import argparse
import getpass
import secrets
import string
import sys

from app.auth import store
from app.auth.passwords import WeakPassword, check_policy
from app.config import load_settings
from app.data import repositories as repo
from app.data.db import Database


def generate() -> str:
    alphabet = string.ascii_letters + string.digits
    while True:
        candidate = "".join(secrets.choice(alphabet) for _ in range(16))
        try:
            check_policy(candidate)
            return candidate
        except WeakPassword:
            continue


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.auth.manage")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("set-password")
    sp.add_argument("employee_id")
    sp.add_argument("--generate", action="store_true", help="create a random password and print it once")
    rp = sub.add_parser("remove")
    rp.add_argument("employee_id")
    sub.add_parser("list")
    args = parser.parse_args(argv)

    db = Database(load_settings().db_path)
    db.ensure_schema()
    if args.cmd == "list":
        with db.read() as conn:
            ids = store.employees_with_credentials(conn)
        print("\n".join(ids) if ids else "(no credentials set)")
        return 0

    employee_id = args.employee_id.strip().upper()
    with db.read() as conn:
        if repo.get_employee(conn, employee_id) is None:
            print(f"unknown employee {employee_id}", file=sys.stderr)
            return 1
    if args.cmd == "remove":
        with db.transaction() as conn:
            removed = store.remove_password(conn, employee_id)
        print("removed" if removed else "no credential to remove")
        return 0

    if args.generate:
        password = generate()
    else:
        password = getpass.getpass("New password: ")
        if password != getpass.getpass("Repeat: "):
            print("passwords do not match", file=sys.stderr)
            return 1
    try:
        check_policy(password)
    except WeakPassword as exc:
        print(str(exc), file=sys.stderr)
        return 1
    with db.transaction() as conn:
        store.set_password(conn, employee_id, password)
    print(f"password set for {employee_id}" + (f": {password}" if args.generate else ""))
    print("existing sign-ins of this employee are ended on their next request")
    return 0


if __name__ == "__main__":
    sys.exit(main())

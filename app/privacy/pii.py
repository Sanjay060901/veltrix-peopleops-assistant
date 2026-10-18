"""Personal-data helpers: email masking and scrubbing of free text.

Personal data in this service is full_name, email and phone (TASK §7.4).
"""
from __future__ import annotations

import re
import threading
from typing import Iterable

EMAIL_RE = re.compile(r"\b([A-Za-z0-9._%+-])([A-Za-z0-9._%+-]*)@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")
# Phone-like sequences: optional +, then 8+ digits possibly separated by spaces, dots, dashes or brackets.
PHONE_RE = re.compile(r"(?<![\w-])\+?\(?\d[\d\s().-]{6,}\d(?![\w-])")


def mask_email(email: str) -> str:
    """'riya@veltrix.example' -> 'r***@veltrix.example'."""
    local, _, domain = email.partition("@")
    if not local or not domain:
        return "***"
    return f"{local[0]}***@{domain}"


class PersonalDataRegistry:
    """Holds the personal data present in the employee table so it can be scrubbed.

    Loaded at startup from the DB. Only used for redaction, never logged.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.emails: set[str] = set()
        self.phones: set[str] = set()
        self.names: set[str] = set()

    def load(self, rows: Iterable[tuple[str, str, str]]) -> None:
        emails, phones, names = set(), set(), set()
        for full_name, email, phone in rows:
            if email:
                emails.add(email.lower())
            if phone:
                phones.add(_digits(phone))
            if full_name:
                names.add(full_name)
        with self._lock:
            self.emails, self.phones, self.names = emails, phones, names


registry = PersonalDataRegistry()


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def mask_employee_emails(text: str, reg: PersonalDataRegistry = registry) -> str:
    """Mask any email address that belongs to an employee. Generic mailboxes from the corpus stay."""

    def repl(m: re.Match) -> str:
        email = m.group(0)
        return mask_email(email) if email.lower() in reg.emails else email

    return EMAIL_RE.sub(repl, text)


def strip_phones(text: str, reg: PersonalDataRegistry = registry) -> str:
    """Remove phone numbers that belong to employees (and anything that looks like one of them)."""

    def repl(m: re.Match) -> str:
        digits = _digits(m.group(0))
        if len(digits) >= 8 and any(digits.endswith(p[-8:]) or p.endswith(digits[-8:]) for p in reg.phones):
            return "[phone removed]"
        return m.group(0)

    return PHONE_RE.sub(repl, text)


def scrub_answer(text: str, reg: PersonalDataRegistry = registry) -> str:
    """Last line of defence on user-facing text: no phones, employee emails masked."""
    return mask_employee_emails(strip_phones(text, reg), reg)


def redact_for_log(text: str, reg: PersonalDataRegistry = registry) -> str:
    """Logs carry no names, emails or phone numbers at all."""
    text = EMAIL_RE.sub("[email]", text)
    text = PHONE_RE.sub(lambda m: "[phone]" if len(_digits(m.group(0))) >= 8 else m.group(0), text)
    for name in reg.names:
        if name and name in text:
            text = text.replace(name, "[name]")
    return text

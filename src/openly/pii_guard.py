from __future__ import annotations

import re
from typing import Iterable

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[ -]?)?(?:\d[ -]?){9,12}\b")


DEFAULT_PII_KEYS = {"name", "child_name", "parent_name", "phone", "email", "address"}


def redact_text(text: str) -> str:
    text = _EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = _PHONE_RE.sub("[REDACTED_PHONE]", text)
    return text


def redact_payload(payload: dict, pii_keys: Iterable[str] = DEFAULT_PII_KEYS) -> dict:
    redacted = {}
    pii_keys = set(pii_keys)
    for key, value in payload.items():
        if key in pii_keys:
            redacted[key] = "[REDACTED]"
        elif isinstance(value, str):
            redacted[key] = redact_text(value)
        else:
            redacted[key] = value
    return redacted

"""Defence in depth against instructions embedded in documents.

Retrieved text is always treated as information only (the prompts say so), but paragraphs that
address an AI system directly carry no HR information and are dropped at ingestion so they never
reach the LLM at all. Patterns are about the *form* of an injection, not about corpus content.
"""
from __future__ import annotations

import re

_AI = r"(?:ai|a\.i\.|artificial intelligence|assistant|chat\s*bot|chatbot|llm|language model|bot|copilot|gpt)"
_PATTERNS = [
    re.compile(rf"\b(?:note|message|instructions?|reminder|attention)\s+(?:for|to)\s+(?:the\s+|any\s+|all\s+)?{_AI}", re.I),
    re.compile(rf"\b(?:dear|hey|hi)\s+{_AI}\b", re.I),
    re.compile(rf"\b{_AI}s?\s+(?:reading|processing|summari[sz]ing)\s+this", re.I),
    re.compile(r"\bignore\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above|earlier|other)\s+(?:instructions|rules|guidance)", re.I),
    re.compile(r"\b(?:system|developer)\s+prompt\b", re.I),
    re.compile(rf"\bif you are an? {_AI}\b", re.I),
]


def is_injection(text: str) -> bool:
    return any(p.search(text) for p in _PATTERNS)

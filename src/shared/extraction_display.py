"""Presentation helpers shared by document-extraction UIs."""

from __future__ import annotations

import json
from typing import Any


def format_extraction_value(value: Any) -> str:
    """Produce a compact table-safe representation for typed extraction values."""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)

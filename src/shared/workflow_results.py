"""Normalize workflow SDK responses for Streamlit result rendering."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel


def workflow_result_mapping(result: Any) -> dict[str, Any]:
    """Return a result payload as a dictionary across SDK response shapes."""
    if isinstance(result, Mapping):
        return dict(result)
    if isinstance(result, BaseModel):
        return result.model_dump(mode="json")
    if isinstance(result, (str, bytes, bytearray)):
        try:
            decoded = json.loads(result)
        except (TypeError, ValueError, UnicodeDecodeError):
            return {}
        return dict(decoded) if isinstance(decoded, Mapping) else {}
    return {}


def workflow_status_name(status: Any) -> str:
    """Return a stable uppercase status for strings and SDK enum values."""
    value = getattr(status, "value", status)
    return str(value).rsplit(".", maxsplit=1)[-1].upper()

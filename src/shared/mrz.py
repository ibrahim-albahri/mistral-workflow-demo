"""Small, dependency-free ICAO 9303 MRZ parser for TD1 cards and TD3 passports."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

_WEIGHTS = (7, 3, 1)
_VALUE_MAP = {str(number): number for number in range(10)}
_VALUE_MAP.update({chr(ord("A") + index): 10 + index for index in range(26)})
_VALUE_MAP["<"] = 0


def _clean(value: str) -> str:
    return "".join(character for character in value.upper() if not character.isspace())


def normalise_mrz(raw_mrz: str | Iterable[str] | None) -> list[str]:
    """Return normalized MRZ lines, including compact OCR output when possible."""
    if raw_mrz is None:
        return []
    if isinstance(raw_mrz, str):
        lines = [_clean(line) for line in raw_mrz.splitlines() if _clean(line)]
    else:
        lines = [_clean(str(line)) for line in raw_mrz if _clean(str(line))]

    if len(lines) == 1:
        compact = lines[0]
        if len(compact) == 88:
            return [compact[:44], compact[44:]]
        if len(compact) == 90:
            return [compact[:30], compact[30:60], compact[60:]]
    return lines


def _check_digit(value: str) -> str:
    total = sum(
        _VALUE_MAP.get(character, 0) * _WEIGHTS[index % 3]
        for index, character in enumerate(value)
    )
    return str(total % 10)


def _valid(value: str, check_digit: str, label: str, errors: list[str]) -> bool:
    if not check_digit.isdigit():
        errors.append(f"{label} check digit is missing or invalid.")
        return False
    if _check_digit(value) != check_digit:
        errors.append(f"{label} check digit does not match.")
        return False
    return True


def _date(value: str) -> str | None:
    return None if not value or "<" in value else value


def _name(value: str) -> str | None:
    parts = value.rstrip("<").split("<<", maxsplit=1)
    if not parts or not parts[0]:
        return None
    return (
        " ".join(piece.replace("<", " ").strip() for piece in parts if piece).strip()
        or None
    )


def parse_mrz(raw_mrz: str | Iterable[str] | None) -> dict[str, Any]:
    """Parse TD1/TD3 data without raising for incomplete or OCR-corrupted input."""
    lines = normalise_mrz(raw_mrz)
    result: dict[str, Any] = {
        "raw_lines": lines,
        "format": None,
        "parsed": {},
        "checksum_valid": None,
        "validation_errors": [],
        "disagreements": [],
    }
    errors: list[str] = result["validation_errors"]

    if len(lines) == 2 and all(len(line) == 44 for line in lines):
        result["format"] = "TD3"
        first, second = lines
        parsed = {
            "document_number": second[0:9].replace("<", "") or None,
            "nationality": second[10:13].replace("<", "") or None,
            "date_of_birth": _date(second[13:19]),
            "sex": second[20].replace("<", "") or None,
            "expiry_date": _date(second[21:27]),
            "country_of_issue": first[2:5].replace("<", "") or None,
            "full_name": _name(first[5:44]),
        }
        result["parsed"] = {
            key: value for key, value in parsed.items() if value is not None
        }
        checks = [
            _valid(second[0:9], second[9], "Document number", errors),
            _valid(second[13:19], second[19], "Birth date", errors),
            _valid(second[21:27], second[27], "Expiry date", errors),
            _valid(second[28:42], second[42], "Personal number", errors),
            _valid(
                second[0:10] + second[13:20] + second[21:43],
                second[43],
                "Composite",
                errors,
            ),
        ]
        result["checksum_valid"] = all(checks)
        return result

    if len(lines) == 3 and all(len(line) == 30 for line in lines):
        result["format"] = "TD1"
        first, second, third = lines
        parsed = {
            "document_number": first[5:14].replace("<", "") or None,
            "nationality": second[15:18].replace("<", "") or None,
            "date_of_birth": _date(second[0:6]),
            "sex": second[7].replace("<", "") or None,
            "expiry_date": _date(second[8:14]),
            "country_of_issue": first[2:5].replace("<", "") or None,
            "full_name": _name(third),
        }
        result["parsed"] = {
            key: value for key, value in parsed.items() if value is not None
        }
        checks = [
            _valid(first[5:14], first[14], "Document number", errors),
            _valid(second[0:6], second[6], "Birth date", errors),
            _valid(second[8:14], second[14], "Expiry date", errors),
            _valid(
                first[5:15] + first[15:30] + second[0:7] + second[8:15] + second[18:29],
                second[29],
                "Composite",
                errors,
            ),
        ]
        result["checksum_valid"] = all(checks)
        return result

    errors.append(
        "MRZ must contain either two 44-character TD3 lines or three 30-character TD1 lines."
    )
    return result

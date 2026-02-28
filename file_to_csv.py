#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

COLUMNS = [
    "gender",
    "first_name",
    "last_name",
    "phone",
    "address",
    "postal_code",
    "city",
    "iban",
    "bic",
]

SYNONYMS = {
    "gender": {"gender", "sex"},
    "first_name": {"first_name", "firstname", "first name", "name", "prenom", "given_name"},
    "last_name": {"last_name", "lastname", "last name", "surname", "family_name", "nom"},
    "phone": {"phone", "telephone", "mobile", "tel", "gsm"},
    "address": {"address", "street", "line1", "adresse"},
    "postal_code": {"postal_code", "postal", "zip", "zipcode", "post code", "cp"},
    "city": {"city", "town", "ville"},
    "iban": {"iban"},
    "bic": {"bic", "swift", "swift_code"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert almost any text/json file to normalized CSV")
    parser.add_argument("input_file", type=Path)
    parser.add_argument("output_csv", type=Path)
    return parser.parse_args()


def decode_bytes(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace")


def normalize_key(key: str) -> str:
    return re.sub(r"\s+", "_", key.strip().lower())


def blank_row() -> dict[str, str]:
    return {col: "" for col in COLUMNS}


def normalize_row(row: dict[str, Any]) -> dict[str, str]:
    normalized = blank_row()
    for target, keys in SYNONYMS.items():
        for key, value in row.items():
            nkey = normalize_key(str(key))
            if nkey in keys or nkey.replace("_", " ") in keys:
                normalized[target] = "" if value is None else str(value).strip()
                break

    # heuristic from name field
    if not normalized["first_name"] and not normalized["last_name"]:
        name = str(row.get("name", "")).strip()
        if name:
            parts = name.split()
            normalized["first_name"] = parts[0]
            normalized["last_name"] = " ".join(parts[1:])

    return normalized


def parse_json_text(text: str) -> list[dict[str, str]]:
    payload = json.loads(text)
    if isinstance(payload, dict):
        if isinstance(payload.get("records"), list):
            payload = payload["records"]
        else:
            payload = [payload]

    rows: list[dict[str, str]] = []
    if not isinstance(payload, list):
        return rows

    for item in payload:
        if isinstance(item, dict):
            rows.append(normalize_row(item))
    return rows


def looks_like_iban(value: str) -> bool:
    compact = value.replace(" ", "")
    return bool(re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{10,30}", compact, flags=re.IGNORECASE))


def looks_like_bic(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?", value.strip(), flags=re.IGNORECASE))


def split_line(line: str) -> list[str]:
    for delim in ("|", ";", "\t", ","):
        parts = [p.strip() for p in line.split(delim)]
        if len(parts) >= 6:
            return parts
    return [line.strip()]


def parse_delimited_text(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = split_line(line)
        if len(parts) < 2:
            continue

        row = blank_row()

        # format like sample:
        # first|last|dob|address|postal|city|phone|email|iban|bic
        if len(parts) >= 10 and looks_like_iban(parts[8]) and looks_like_bic(parts[9]):
            row["first_name"] = parts[0]
            row["last_name"] = parts[1]
            row["address"] = parts[3]
            row["postal_code"] = parts[4]
            row["city"] = parts[5]
            row["phone"] = parts[6]
            row["iban"] = parts[8].replace(" ", "")
            row["bic"] = parts[9]
            rows.append(row)
            continue

        # generic mapping fallback by index
        if len(parts) >= 8:
            row["first_name"] = parts[0]
            row["last_name"] = parts[1]
            row["address"] = parts[2]
            row["postal_code"] = parts[3]
            row["city"] = parts[4]
            row["phone"] = parts[5]
            row["iban"] = parts[-2].replace(" ", "") if looks_like_iban(parts[-2]) else ""
            row["bic"] = parts[-1] if looks_like_bic(parts[-1]) else ""
            rows.append(row)
            continue

        # key-value single line fallback (e.g. phone:xxx)
        kv = {}
        for frag in re.split(r"[|;,]", line):
            if ":" in frag:
                k, v = frag.split(":", 1)
                kv[k.strip()] = v.strip()
        if kv:
            rows.append(normalize_row(kv))

    return rows


def parse_any(path: Path) -> list[dict[str, str]]:
    raw = path.read_bytes()
    text = decode_bytes(raw)

    # Try JSON first (for .json or json-looking files)
    if path.suffix.lower() == ".json" or text.lstrip().startswith(("{", "[")):
        try:
            rows = parse_json_text(text)
            if rows:
                return rows
        except json.JSONDecodeError:
            pass

    rows = parse_delimited_text(text)
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    rows = parse_any(args.input_file)
    write_csv(args.output_csv, rows)
    print(f"Wrote {len(rows)} row(s) to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

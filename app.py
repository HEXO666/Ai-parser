#!/usr/bin/env python3
from __future__ import annotations

import csv
import io
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any
from urllib import error, request

from flask import Flask, Response, flash, redirect, render_template, request as flask_request

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")

TARGET_COLUMNS = [
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

SYSTEM_PROMPT = (
    "You are a data extraction assistant. "
    "Extract people/customer records from the provided content and normalize each record "
    "to this exact schema: gender, first_name, last_name, phone, address, postal_code, city, iban, bic. "
    "If data is missing, use an empty string. "
    "Respond with valid JSON only: either an array of objects or {\"records\": [...]}"
)


def load_uploaded_content(file_storage) -> str:
    filename = file_storage.filename or "uploaded_file"
    suffix = Path(filename).suffix.lower()
    mime, _ = mimetypes.guess_type(filename)
    raw = file_storage.read()

    if suffix == ".json":
        obj = json.loads(raw.decode("utf-8", errors="replace"))
        return json.dumps(obj, ensure_ascii=False, indent=2)

    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        text = raw.decode("utf-8", errors="replace")
        rows = list(csv.DictReader(io.StringIO(text), delimiter=delimiter))
        return json.dumps(rows, ensure_ascii=False, indent=2)

    is_text = mime and (mime.startswith("text/") or "json" in mime or "xml" in mime)
    if is_text or suffix in {".txt", ".log", ".xml", ".yaml", ".yml", ".md"}:
        return raw.decode("utf-8", errors="replace")

    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace")


def call_deepseek(content: str, api_key: str, model: str, base_url: str) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Extract all valid records from the following file content. "
                    "Return JSON only.\n\n"
                    f"FILE CONTENT:\n{content}"
                ),
            },
        ],
        "temperature": 0,
    }

    req = request.Request(
        base_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=120) as resp:
            response_body = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DeepSeek API HTTP error {exc.code}: {details}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"DeepSeek API connection error: {exc}") from exc

    try:
        response_json = json.loads(response_body)
        return response_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unexpected DeepSeek response: {response_body[:800]}") from exc


def extract_json_block(text: str) -> Any:
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9]*\n", "", cleaned)
        cleaned = re.sub(r"\n```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"(\{.*\}|\[.*\])", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError("Could not parse JSON from DeepSeek response")
    return json.loads(match.group(1))


def normalize_records(data: Any) -> list[dict[str, str]]:
    if isinstance(data, dict) and "records" in data:
        data = data["records"]
    elif isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        raise ValueError("DeepSeek result must be a list of records or {'records': [...]}.")

    output: list[dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        row = {column: "" for column in TARGET_COLUMNS}
        for column in TARGET_COLUMNS:
            value = item.get(column, "")
            row[column] = "" if value is None else str(value).strip()
        output.append(row)

    return output


def to_csv_bytes(records: list[dict[str, str]]) -> bytes:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=TARGET_COLUMNS)
    writer.writeheader()
    writer.writerows(records)
    return stream.getvalue().encode("utf-8")


@app.get("/")
def index():
    return render_template(
        "index.html",
        default_model="deepseek-chat",
        default_base_url="https://api.deepseek.com/chat/completions",
    )


@app.post("/convert")
def convert():
    uploaded_file = flask_request.files.get("input_file")
    if not uploaded_file or uploaded_file.filename == "":
        flash("Please upload a file first.")
        return redirect("/")

    api_key = flask_request.form.get("api_key", "").strip() or os.getenv("DEEPSEEK_API_KEY", "").strip()
    model = flask_request.form.get("model", "deepseek-chat").strip() or "deepseek-chat"
    base_url = flask_request.form.get("base_url", "https://api.deepseek.com/chat/completions").strip()

    if not api_key:
        flash("Missing DeepSeek API key. Add it in the form or DEEPSEEK_API_KEY env var.")
        return redirect("/")

    try:
        max_chars = int(flask_request.form.get("max_chars", "50000"))
    except ValueError:
        max_chars = 50000

    try:
        content = load_uploaded_content(uploaded_file)
        if len(content) > max_chars:
            content = content[:max_chars]

        llm_response = call_deepseek(content, api_key=api_key, model=model, base_url=base_url)
        parsed = extract_json_block(llm_response)
        records = normalize_records(parsed)
        csv_data = to_csv_bytes(records)
    except Exception as exc:
        flash(f"Conversion failed: {exc}")
        return redirect("/")

    filename_stem = Path(uploaded_file.filename).stem or "converted"
    output_name = f"{filename_stem}_normalized.csv"
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{output_name}"'},
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

#!/usr/bin/env python3
from __future__ import annotations

import base64
import csv
import io
import json
import mimetypes
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any
from urllib import error, request

from flask import Flask, Response, jsonify, redirect, render_template, request as flask_request

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

JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()


def set_job(job_id: str, **updates: Any) -> None:
    with JOBS_LOCK:
        if job_id not in JOBS:
            JOBS[job_id] = {}
        JOBS[job_id].update(updates)


def append_job_log(job_id: str, message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    with JOBS_LOCK:
        JOBS.setdefault(job_id, {"logs": []})
        JOBS[job_id].setdefault("logs", [])
        JOBS[job_id]["logs"].append(f"[{timestamp}] {message}")


def load_uploaded_content(file_name: str, raw: bytes) -> tuple[str, list[str]]:
    warnings: list[str] = []
    suffix = Path(file_name).suffix.lower()
    mime, _ = mimetypes.guess_type(file_name)
    decoded = raw.decode("utf-8", errors="replace")

    if suffix == ".json":
        try:
            obj = json.loads(decoded)
            return json.dumps(obj, ensure_ascii=False, indent=2), warnings
        except json.JSONDecodeError as exc:
            warnings.append(f"Input JSON is not strictly valid ({exc}); sending raw text to DeepSeek.")
            return decoded, warnings

    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        rows = list(csv.DictReader(io.StringIO(decoded), delimiter=delimiter))
        return json.dumps(rows, ensure_ascii=False, indent=2), warnings

    is_text = mime and (mime.startswith("text/") or "json" in mime or "xml" in mime)
    if is_text or suffix in {".txt", ".log", ".xml", ".yaml", ".yml", ".md"}:
        return decoded, warnings

    try:
        return raw.decode("utf-8"), warnings
    except UnicodeDecodeError:
        warnings.append("Binary/unknown file decoded with latin-1 fallback.")
        decoded_fallback = raw.decode("latin-1", errors="replace")
        b64_preview = base64.b64encode(raw[:2048]).decode("ascii")
        packaged = (
            f"FILE_NAME: {file_name}\n"
            f"MIME_TYPE: {mime or 'unknown'}\n"
            "NOTE: Binary-like content, text may be noisy.\n"
            "LATIN1_PREVIEW:\n"
            f"{decoded_fallback[:4000]}\n\n"
            "BASE64_PREVIEW_FIRST_2048_BYTES:\n"
            f"{b64_preview}"
        )
        return packaged, warnings


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
        with request.urlopen(req, timeout=300) as resp:
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


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9]*\n", "", cleaned)
        cleaned = re.sub(r"\n```$", "", cleaned)
    return cleaned


def _json_candidates(text: str) -> list[str]:
    cleaned = _strip_code_fences(text)
    candidates = [cleaned]

    match = re.search(r"(\{.*\}|\[.*\])", cleaned, flags=re.DOTALL)
    if match:
        candidates.append(match.group(1))

    repaired = cleaned
    repaired = repaired.replace("“", '"').replace("”", '"').replace("’", "'")
    repaired = re.sub(r"//.*", "", repaired)
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    candidates.append(repaired)

    if match:
        inner_repaired = match.group(1)
        inner_repaired = re.sub(r",\s*([}\]])", r"\1", inner_repaired)
        candidates.append(inner_repaired)

    # de-dup preserve order
    return list(dict.fromkeys(candidates))


def extract_json_block(text: str) -> Any:
    last_error: Exception | None = None
    for candidate in _json_candidates(text):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue

    error_msg = f"Could not parse JSON from DeepSeek response. Last parser error: {last_error}"
    raise ValueError(error_msg)


SYNONYM_MAP = {
    "gender": ["gender", "sex"],
    "first_name": ["first_name", "firstname", "first name", "name"],
    "last_name": ["last_name", "lastname", "last name", "surname", "family_name"],
    "phone": ["phone", "telephone", "mobile", "tel"],
    "address": ["address", "street", "line1"],
    "postal_code": ["postal_code", "postal", "zip", "zipcode", "post code"],
    "city": ["city", "town"],
    "iban": ["iban"],
    "bic": ["bic", "swift", "swift_code"],
}


def heuristic_records_from_text(text: str) -> list[dict[str, str]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    row = {column: "" for column in TARGET_COLUMNS}

    for line in lines:
        for target, keys in SYNONYM_MAP.items():
            for key in keys:
                pattern = rf"^{re.escape(key)}\s*[:=\-]\s*(.+)$"
                match = re.match(pattern, line, flags=re.IGNORECASE)
                if match and not row[target]:
                    row[target] = match.group(1).strip()

    # additional global regex lookups
    if not row["iban"]:
        iban_match = re.search(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b", text.replace(" ", ""), flags=re.IGNORECASE)
        if iban_match:
            row["iban"] = iban_match.group(0).upper()
    if not row["bic"]:
        bic_match = re.search(r"\b[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?\b", text, flags=re.IGNORECASE)
        if bic_match:
            row["bic"] = bic_match.group(0).upper()
    if not row["phone"]:
        phone_match = re.search(r"\+?[0-9][0-9\s().-]{7,}[0-9]", text)
        if phone_match:
            row["phone"] = phone_match.group(0).strip()

    if any(value for value in row.values()):
        return [row]
    return []


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


def run_conversion_job(
    job_id: str,
    *,
    file_name: str,
    file_bytes: bytes,
    api_key: str,
    model: str,
    base_url: str,
    max_chars: int,
) -> None:
    try:
        append_job_log(job_id, f"Starting conversion for '{file_name}'.")
        append_job_log(job_id, "Reading uploaded file (any format).")
        append_job_log(job_id, f"Uploaded file: {file_name} ({len(file_bytes)} bytes)")
        content, warnings = load_uploaded_content(file_name, file_bytes)
        for warning in warnings:
            append_job_log(job_id, f"Warning: {warning}")

        original_size = len(content)
        if len(content) > max_chars:
            content = content[:max_chars]
            append_job_log(job_id, f"Input truncated from {original_size} to {max_chars} chars.")
        else:
            append_job_log(job_id, f"Input size: {original_size} chars.")

        append_job_log(job_id, f"Calling DeepSeek model '{model}'.")
        llm_response = call_deepseek(content, api_key=api_key, model=model, base_url=base_url)

        append_job_log(job_id, "Parsing model response JSON.")
        records: list[dict[str, str]] = []
        try:
            parsed = extract_json_block(llm_response)
            append_job_log(job_id, "Normalizing records to target columns.")
            records = normalize_records(parsed)
        except ValueError as parse_exc:
            append_job_log(job_id, f"Warning: model JSON parse failed ({parse_exc}). Trying text heuristic.")
            records = heuristic_records_from_text(llm_response)
            if records:
                append_job_log(job_id, "Heuristic extraction from model text succeeded.")
            else:
                append_job_log(job_id, "Trying heuristic extraction directly from uploaded content.")
                records = heuristic_records_from_text(content)

        if not records:
            append_job_log(job_id, "No structured records found. Creating CSV with header only.")

        append_job_log(job_id, f"Generating CSV file with {len(records)} record(s).")
        csv_data = to_csv_bytes(records)

        output_name = f"{Path(file_name).stem or 'converted'}_normalized.csv"
        set_job(
            job_id,
            status="done",
            csv_data=csv_data,
            output_name=output_name,
            total_records=len(records),
        )
        append_job_log(job_id, "Done. CSV is ready to download.")
    except Exception as exc:
        set_job(job_id, status="error", error=str(exc))
        append_job_log(job_id, f"Failed: {exc}")


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
        return jsonify({"ok": False, "error": "Please upload a file first."}), 400

    api_key = flask_request.form.get("api_key", "").strip() or os.getenv("DEEPSEEK_API_KEY", "").strip()
    model = flask_request.form.get("model", "deepseek-chat").strip() or "deepseek-chat"
    base_url = flask_request.form.get("base_url", "https://api.deepseek.com/chat/completions").strip()

    if not api_key:
        return jsonify({"ok": False, "error": "Missing DeepSeek API key."}), 400

    try:
        max_chars = max(1000, int(flask_request.form.get("max_chars", "50000")))
    except ValueError:
        max_chars = 50000

    file_bytes = uploaded_file.read()
    file_name = uploaded_file.filename or "uploaded_file"

    job_id = uuid.uuid4().hex
    set_job(
        job_id,
        status="running",
        created_at=time.time(),
        logs=[],
        csv_data=None,
        output_name=None,
        error=None,
        total_records=0,
    )

    worker = threading.Thread(
        target=run_conversion_job,
        kwargs={
            "job_id": job_id,
            "file_name": file_name,
            "file_bytes": file_bytes,
            "api_key": api_key,
            "model": model,
            "base_url": base_url,
            "max_chars": max_chars,
        },
        daemon=True,
    )
    worker.start()

    return jsonify({"ok": True, "job_id": job_id})


@app.get("/status/<job_id>")
def status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"ok": False, "error": "Job not found."}), 404

        return jsonify(
            {
                "ok": True,
                "status": job.get("status", "unknown"),
                "logs": job.get("logs", []),
                "error": job.get("error"),
                "total_records": job.get("total_records", 0),
                "ready": bool(job.get("csv_data")),
            }
        )


@app.get("/download/<job_id>")
def download(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return redirect("/")
        csv_data = job.get("csv_data")
        output_name = job.get("output_name") or "converted_normalized.csv"

    if not csv_data:
        return Response("CSV not ready yet.", status=400)

    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{output_name}"'},
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

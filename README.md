# DeepSeek File-to-CSV Flask App

A Flask web app with a modern UI that:

1. Accepts an uploaded file (`.json`, `.csv`, `.tsv`, text-like files, and best-effort unknown formats).
2. Sends file content to DeepSeek Chat Completions API.
3. Returns a downloadable CSV with this fixed header:

`gender, first_name, last_name, phone, address, postal_code, city, iban, bic`

## Features

- **Live progress tracking** with per-step logs for long conversions.
- **Background conversion job** so the page stays responsive for large files.
- **Settings panel (⚙️)** to configure DeepSeek API key, model, endpoint, and max input chars.
- **One-click CSV download** once conversion finishes.
- **Any-file fallback behavior**: unknown/binary formats are still accepted and sent with safe previews/metadata.
- **Heuristic fallback**: if model JSON is malformed, app tries text-based extraction and still generates a CSV file (at least header).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
export DEEPSEEK_API_KEY="your_api_key"   # optional if you will paste key in UI
python3 app.py
```

Then open: `http://localhost:5000`

## Notes

- You can provide the DeepSeek key either in the settings panel or via `DEEPSEEK_API_KEY` env var.
- Default model: `deepseek-chat`
- Default endpoint: `https://api.deepseek.com/chat/completions`
- `max_chars` limits payload size sent to DeepSeek.

- Malformed `.json` uploads are now handled gracefully (fallback to raw text instead of hard-failing).
- Model responses with minor JSON issues (code fences/trailing commas/smart quotes) are automatically repaired when possible.
- If no valid JSON is returned, the app attempts key-value/regex extraction from text and still creates a CSV file.

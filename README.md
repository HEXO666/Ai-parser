# DeepSeek File-to-CSV Flask App

A small Flask web app with friendly UI that:

1. Accepts an uploaded file (`.json`, `.csv`, `.tsv`, text-like files, and best-effort unknown formats).
2. Sends file content to DeepSeek Chat Completions API.
3. Returns a downloadable CSV with this fixed header:

`gender, first_name, last_name, phone, address, postal_code, city, iban, bic`

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

- You can provide the DeepSeek key either in the UI form or via `DEEPSEEK_API_KEY` env var.
- Default model: `deepseek-chat`
- Default endpoint: `https://api.deepseek.com/chat/completions`
- `max_chars` limits payload size sent to DeepSeek.

# File to CSV Parser (No AI)

This project now includes a direct Python script that does **not** call any AI API.

It reads an input file (TXT / JSON / CSV / TSV / other delimited text) and creates a normalized CSV with this schema:

`gender, first_name, last_name, phone, address, postal_code, city, iban, bic`

## Script usage

```bash
python3 file_to_csv.py input.txt output.csv
```

## Supported formats

- Pipe-delimited (`|`) text (like your example)
- CSV / TSV / semicolon-delimited text
- JSON objects or JSON arrays (including `{ "records": [...] }`)
- Fallback key-value lines (`first_name: John | last_name: Doe | ...`)

## Example (your sample format)

Input line:

```text
CATHERINE|COUROUZIAN|21/09/1964|65 ROUTE DU CIMETIERE|38200|SERPAIZE|0662646061|kathyisere@gmail.com|FR7610278089280002057070178|CMCIFR2AXXX
```

Output columns mapping:

- `first_name` = `CATHERINE`
- `last_name` = `COUROUZIAN`
- `address` = `65 ROUTE DU CIMETIERE`
- `postal_code` = `38200`
- `city` = `SERPAIZE`
- `phone` = `0662646061`
- `iban` = `FR7610278089280002057070178`
- `bic` = `CMCIFR2AXXX`
- `gender` = empty (unless provided in source)

## Notes

- The Flask UI files are still in the repo, but this script is fully standalone and does not require DeepSeek.
- If the parser cannot infer values, it leaves fields empty.

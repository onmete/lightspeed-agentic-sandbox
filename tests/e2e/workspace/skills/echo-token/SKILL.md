# echo-token

Generate a verification token and write it to disk.

## Usage

Run the tool script:

```bash
bash echo-token/tools/echo-token.sh
```

## Output

The script prints a JSON object:

```json
{"token": "<random-hex>", "status": "ok"}
```

Include the **token** value in your response summary verbatim.

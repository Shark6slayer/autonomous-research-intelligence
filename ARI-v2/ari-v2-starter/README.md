# ARI v2 — Sprint 1

Research paper discovery + semantic ranking. This is **not yet** autonomous research or evidence verification.

## Install (Python 3.11 or 3.12 recommended)

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Optional: set `SEMANTIC_SCHOLAR_API_KEY` in your environment for authenticated access. Do not commit keys.

## Run

```bash
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs and try POST `/research`:

```json
{"question":"transformers for RF modulation classification at low SNR", "limit":10}
```

First ranking request downloads the embedding model. Internet access is needed for paper search and initial model download. Semantic Scholar rate limits apply.

## Test

```bash
pytest -q
```

## Next sprint

Add durable storage, deduplication, source retrieval, claim-level evidence with provenance, then a bounded planning loop and sandboxed experiments.

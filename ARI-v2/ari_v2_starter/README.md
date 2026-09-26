# ARI v2 — Phase 1 Starter

A working, transparent **literature-discovery MVP**. It is not yet an autonomous multi-agent system. The report preserves paper URLs and abstracts and clearly labels unverified research questions.

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
# Optional: set SEMANTIC_SCHOLAR_API_KEY to improve API access
uvicorn ari.main:app --reload
```

Visit http://127.0.0.1:8000/docs and use `POST /research`:

```json
{"question":"Can transformers improve RF signal classification at low SNR?","max_papers":10}
```

The response contains a project ID, retrieved papers and a Markdown report. You can retrieve it later with `GET /research/{project_id}`. Check `/health` to verify the service is running.

## Limitations and next steps

Semantic Scholar may rate-limit requests or return zero results for long queries. Requests run synchronously in this MVP; Phase 2 should add a durable job queue, query expansion, reranking, evidence extraction with passage citations, human review, and evaluation on a labeled paper set. Never execute generated experiment code outside a restricted sandbox.

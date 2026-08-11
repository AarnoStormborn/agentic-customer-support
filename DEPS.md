# DEPS — dependency reconciliation log (per integration-contract rule 3)

## retrieval-core (track 1)

- **New npm packages added: none.** All runtime deps (`pg`, `openai`, `pdf-parse`,
  `dotenv`) were already in `package.json` from Phase 1 scaffolding. `@types/pg`
  already present.
- Added npm scripts only: `db:migrate`, `query` (existing `ingest` reused).
- **Non-npm tool requirements (PATH):**
  - `python3` + `pyarrow` (suraj520 parquet → CSV; `scripts/convert-suraj520.py`).
    Installed on this machine (python3 3.11 + pyarrow 24.0.0). Orchestrator note:
    ingest auto-converts only when the CSV is missing — a committed CSV would
    remove the python requirement entirely.
- Reconciliation: merge the three tracks' `package.json` carefully; this track's
  additions are scripts only, no version conflicts expected.

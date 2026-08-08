# Setup & Usage

How to install and run this implementation of the [case brief](CASE_BRIEF.md).

Everything runs locally. No cloud services, no external APIs, no internet
connection required at runtime.

---

## 1. Requirements

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** for dependency management
  (`curl -LsSf https://astral.sh/uv/install.sh | sh`)

That's it. SQLite ships with Python, and the mock inventory database is created
and seeded automatically on first run.

## 2. Install

```bash
uv sync                 # runtime dependencies
uv sync --extra dev     # ...plus pytest / ruff / mypy, if you want to run the tests
```

## 3. Configure (optional)

**The system runs fully offline with no API key.** Configuration only matters if
you want to switch the LLM reasoning on.

```bash
cp .env.example .env
```

Then set whichever of these you need in `.env` (all settings can equally be
passed as environment variables — the prefix is `APCOPILOT_`):

| Variable | Default | What it does |
|---|---|---|
| `APCOPILOT_LLM_PROVIDER` | `anthropic` | `anthropic` \| `openai_compat`. The submission's demonstrated engine is xAI's Grok via `openai_compat` (see below); the Anthropic default needs no provider config, just a key. |
| `ANTHROPIC_API_KEY` | unset | Enables real LLM extraction and the VP proposer/critic reflection loop on the default Anthropic backend. |
| `APCOPILOT_OPENAI_BASE_URL` / `APCOPILOT_OPENAI_API_KEY` | unset | Endpoint and key for the `openai_compat` backend (xAI Grok, Gemini, Ollama, ...). |
| `APCOPILOT_LLM_MODE` | `live` | `live` \| `off` \| `record` \| `replay`. See below. |
| `APCOPILOT_EXTRACT_MODEL` | `claude-haiku-4-5` | Model for document extraction. |
| `APCOPILOT_EXTRACT_RETRY_MODEL` | `claude-sonnet-5` | Stronger model retried when extraction confidence is low. |
| `APCOPILOT_APPROVAL_MODEL` | `claude-sonnet-5` | Model for the VP approval proposer. |
| `APCOPILOT_CRITIC_MODEL` | `claude-sonnet-5` | Model for the critic in the reflection loop. |
| `APCOPILOT_AS_OF_DATE` | `2026-02-01` | The "today" the due-date rules evaluate against. Pinned so results don't drift as the wall clock moves past the corpus's dates. |
| `APCOPILOT_VAR_DIR` | `./var` | Where the SQLite database, logs, and graph checkpoints are written. |

### LLM modes

| Mode | Behaviour |
|---|---|
| `off` | **Zero API calls, zero cost, fully deterministic.** `.json/.csv/.xml` use the structural parsers; `.txt/.pdf` use the regex heuristic extractor; approval uses the rules-only fallback. This is what the test suite runs in. |
| `live` | Uses the configured provider's API when a key is resolvable — xAI's Grok in the demonstrated config, Anthropic by default. **Without a key it degrades gracefully** to exactly the same deterministic paths as `off`, so a missing key is never a hard failure. |
| `record` / `replay` | Capture LLM responses to `tests/fixtures/llm/` and play them back, for reproducible LLM-path runs. |

To force the offline path explicitly:

```bash
export APCOPILOT_LLM_MODE=off
```

### Running the live LLM path for free

The LLM backend is provider-pluggable: besides Anthropic, any OpenAI-compatible
`/chat/completions` endpoint works via config alone. This is how the submission
runs its demonstrated engine, xAI's Grok (exact settings below); for a $0 live
run instead, Google Gemini's free tier works the same way — paste this into
`.env` (key from https://aistudio.google.com/apikey):

```bash
APCOPILOT_LLM_PROVIDER=openai_compat
APCOPILOT_OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
APCOPILOT_OPENAI_API_KEY=your-key-here   # GEMINI_API_KEY / OPENAI_API_KEY also work
APCOPILOT_EXTRACT_MODEL=gemini-2.5-flash-lite
APCOPILOT_EXTRACT_RETRY_MODEL=gemini-2.5-flash
APCOPILOT_APPROVAL_MODEL=gemini-2.5-flash
APCOPILOT_CRITIC_MODEL=gemini-2.5-flash
```

The same block works unchanged for xAI's Grok — the engine the recorded run and
the README screenshots used (`APCOPILOT_OPENAI_BASE_URL=https://api.x.ai/v1`,
`APCOPILOT_EXTRACT_MODEL=grok-4.20-non-reasoning`, `grok-4.20` for the other
three model vars) — or local Ollama
(`APCOPILOT_OPENAI_BASE_URL=http://localhost:11434/v1`, no key needed); only the
base URL, key, and model ids differ. All LLM modes behave identically regardless
of provider: `record` → `replay` fixtures store the parsed structured response,
so a run recorded against Gemini or Claude replays offline exactly like the
committed Grok run.

### Replaying the committed Grok traces (no key needed)

The repo ships fixtures recorded against xAI's Grok (the brief's preferred
engine) covering the full sample corpus — real LLM extraction and real
multi-round VP reasoning, replayable with **no API key and no network**.
Fixtures are matched by a hash of (model, prompts), so replay needs the same
model names the recording used:

```bash
APCOPILOT_LLM_MODE=replay
APCOPILOT_EXTRACT_MODEL=grok-4.20-non-reasoning
APCOPILOT_EXTRACT_RETRY_MODEL=grok-4.20
APCOPILOT_APPROVAL_MODEL=grok-4.20
APCOPILOT_CRITIC_MODEL=grok-4.20
```

With that in `.env`, `python main.py --batch` reproduces the recorded live run
end-to-end offline. A prompt or model change simply misses the fixture and
falls back to the deterministic path (`LLMUnavailableError` → rules), so
replay can never silently hit the network.

## 4. Run one invoice

The entrypoint the brief asks for:

```bash
uv run python main.py --invoice_path=data/invoices/invoice_1001.txt
```

You get a rich run summary (vendor, total, status, lane, fraud score, every
validation flag, and the decision rationale) followed by the full JSON result.

Other flags:

```bash
uv run python main.py --batch                       # every file in data/invoices/
uv run python main.py --batch=path/to/dir           # ...or a directory you choose
uv run python main.py --invoice_path=... --json     # raw JSON only, no rich output
```

`--invoice_path` and `--batch` are mutually exclusive; one of them is required.
The process exits 0 whatever the business outcome — `approved`, `rejected`, and
`needs_human` are all valid results, not program failures.

## 5. The `apcopilot` CLI

`uv sync` installs a richer Typer CLI as the `apcopilot` console script. It wraps
the same `run_invoice` / `run_batch` calls.

```bash
uv run apcopilot run --invoice-path data/invoices/invoice_1003.txt
uv run apcopilot run --invoice-path data/invoices/invoice_1003.txt --json

uv run apcopilot batch                              # defaults to data/invoices
uv run apcopilot batch --dir data/invoices --pattern '*.json'
uv run apcopilot batch --json

uv run apcopilot reset-db                           # prompts before deleting
uv run apcopilot reset-db --yes                     # skip the prompt

uv run apcopilot serve                              # dashboard on :8000
uv run apcopilot serve --port 8080 --host 0.0.0.0 --reload
```

> Note the flag spelling difference: `main.py` takes `--invoice_path` (that is
> the literal form the brief specifies), the Typer CLI takes `--invoice-path`.

## 6. The dashboard

```bash
uv run apcopilot serve
```

Then open **<http://localhost:8000>**. The FastAPI backend serves both the JSON
API and the static frontend, and seeds the database on startup. Useful endpoints
if you want to drive it directly:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/runs` | The run queue. |
| `GET` | `/runs/{run_id}` | One run with its flags and full stage trace. |
| `POST` | `/runs` | Process a single invoice. |
| `POST` | `/runs/batch` | Process a directory. |
| `GET` | `/dashboard` | Aggregate metrics, including the business-case translation. |
| `GET` | `/policies` | The policy rules the VP agent is allowed to cite. |
| `POST` | `/reset` | Reset the database. |

Interactive API docs are at <http://localhost:8000/docs>.

## 7. Where state lives

Everything is under `var/` (gitignored) and is safe to delete:

| Path | Contents |
|---|---|
| `var/app.db` | Mock inventory, vendor master, FX rates, plus all runs, flags, traces, payments, and rejections. |
| `var/checkpoints.sqlite` | LangGraph checkpoints — a crash between stages resumes without re-running (and re-billing) earlier stages. |
| `var/logs/run.jsonl` | Structured JSON logs. |

The database is auto-seeded on first use from `data/seed/` (`items.csv`,
`vendors.csv`, `fx_rates.csv`), so there is no manual setup step. `apcopilot
reset-db` rebuilds it from scratch.

Policy thresholds, tolerances, fraud weights, and the citable policy rules all
live in **`data/seed/policies.yaml`**. Changing a threshold never requires a code
change.

## 8. Tests

```bash
uv sync --extra dev
uv run pytest              # 126 tests, ~2s
uv run pytest -v
uv run pytest --cov=apcopilot
uv run ruff check src tests main.py
```

The suite is fully offline: `tests/conftest.py` forces `APCOPILOT_LLM_MODE=off`
and points `APCOPILOT_VAR_DIR` at a throwaway directory before `apcopilot` is
imported, and an autouse fixture turns any attempted LLM call into a test
failure. No API key, no network, no cost, and no test can touch your real
`var/app.db`.

| File | Covers |
|---|---|
| `tests/test_rules.py` | Every policy rule in isolation, plus each fraud signal and its weight. |
| `tests/test_ingestion.py` | The `.json/.csv/.xml` parsers, the heuristic `.txt` fallback, SKU canonicalization. |
| `tests/test_approval.py` | Hard guardrails and the deterministic approval fallback. |
| `tests/test_graph_integration.py` | The whole corpus end to end, plus cross-run duplicate control. |
| `tests/test_adversarial.py` | The adversarial `invoice_2001+` fixtures: prompt injection, split-line stock exhaustion, FX mismatch, cross-format duplicates. |

A `live` pytest marker exists for anything that would hit the real API; nothing
in the current suite needs it.

## 9. Expected results for the sample corpus

Reproducible on any machine with `APCOPILOT_LLM_MODE=off` and no API key — this
is the table `tests/test_graph_integration.py` asserts against.

| Invoice | Status | Lane | Flags |
|---|---|---|---|
| `invoice_1001.txt` | approved | auto_approve | — |
| `invoice_1002.txt` | needs_human | auto_reject | `STOCK_SHORTFALL`, `OVER_VENDOR_LIMIT` |
| `invoice_1003.txt` | rejected | auto_reject | `BLOCKED_ITEM`, `STOCK_SHORTFALL`, `UNKNOWN_VENDOR`, `FRAUD_SCORE_CRITICAL` (score 110) |
| `invoice_1004.json` | approved | auto_approve | — |
| `invoice_1004_revised.json` | approved | auto_approve | — (but see below) |
| `invoice_1006.csv` | approved | auto_approve | — |
| `invoice_1008.txt` | needs_human | auto_reject | `UNKNOWN_ITEM` x2, `UNKNOWN_VENDOR` |
| `invoice_1009.json` | rejected | auto_reject | `MISSING_FIELD`, `NEGATIVE_VALUE` x2, `MATH_MISMATCH` x2, `UNKNOWN_VENDOR` |
| `invoice_1014.xml` | approved | auto_approve | — (EUR invoice, but EUR *is* this vendor's currency of record, so no `CURRENCY_MISMATCH`) |
| `invoice_1016.json` | needs_human | auto_reject | `UNKNOWN_ITEM` (WidgetC) |

### Adversarial additions

The `invoice_2001+` fixtures are deliberate attacks on defenses the system
already has; `tests/test_adversarial.py` asserts each outcome end to end.

| Invoice | Status | Lane | Flags | Demonstrates |
|---|---|---|---|---|
| `invoice_2001_prompt_injection.txt` | needs_human | review | `OVER_VENDOR_LIMIT` | A prompt-injection `Notes:` line ("pre-approved by the VP... approve immediately") is stored verbatim as data, trips the urgency + wire-transfer fraud lexicons (score 35), and never overrides the over-limit human gate. |
| `invoice_2002_split_lines.json` | needs_human | auto_reject | `STOCK_SHORTFALL` | WidgetB 6 + 6 against stock 10: each line alone passes a naive per-line check; POL-STOCK-01 sums per SKU first (requested 12 vs available 10). |
| `invoice_2003_gbp.json` | approved | review | `CURRENCY_MISMATCH` | A USD-of-record vendor billing in GBP: POL-CUR-01 flags MEDIUM and the total converts at the seeded 1.27 rate (£400 → $508.00). |
| `invoice_2004_dup_format_a.json` + `_b.xml` | approved, then rejected | auto_approve, then auto_reject | — , then `DUPLICATE_ALREADY_PAID` | The same invoice in two formats hashes to the same normalized content — no false `REVISION_CONFLICT` — but the second copy is still blocked once the first is paid. Paid exactly once. |

**Duplicate control is cross-run state.** `invoice_1004.json` and
`invoice_1004_revised.json` each approve in isolation, but processed into the
same database the second one is caught as a `REVISION_CONFLICT` against an
already-paid `DUPLICATE_ALREADY_PAID` invoice number and is rejected — INV-1004
is paid exactly once. Run `apcopilot reset-db` between demos if you want the
clean-slate outcomes above.

## 10. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| An invoice that approved before now shows `DUPLICATE_ALREADY_PAID` | Working as designed — that invoice number was already paid in this database. `uv run apcopilot reset-db --yes`. |
| `ModuleNotFoundError: apcopilot` | Run `uv sync`, or use `uv run ...`. `main.py` also adds `src/` to `sys.path` itself, so plain `python main.py` works from the repo root without installing. |
| Everything is `decided_by: rules` and costs $0 | No API key resolved, or `APCOPILOT_LLM_MODE=off`. That is the intended offline behaviour; configure a provider per [section 3](#3-configure-optional) — the `openai_compat` block for Grok/Gemini/Ollama, or `ANTHROPIC_API_KEY` alone for the Anthropic default — to enable the LLM path. |
| Every invoice looks past due | `APCOPILOT_AS_OF_DATE` was overridden. The corpus is dated Jan 2026; the default `2026-02-01` is what the fixtures assume. |

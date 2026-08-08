# Design Decisions

The tradeoffs behind the implementation, one per section: what was chosen, what it was
chosen over, and why. File paths point at the code that embodies each decision. For the
system map, see the [Architecture section of the README](README.md#architecture).

## 1. Hybrid ingestion: deterministic where possible, LLM only where necessary

`src/apcopilot/ingestion/__init__.py` routes by format: JSON/CSV/XML go through
deterministic structural parsers; only messy TXT/PDF documents hit the LLM
(`llm_extract.py`: a cheap model first, escalating to a stronger one on low
confidence — `grok-4.20-non-reasoning` then `grok-4.20` in the recorded run); and a regex
heuristic (`heuristic.py`) catches the LLM path whenever the model is off or
unavailable. The rejected alternative — LLM-extract everything — is strictly worse on
three axes: cost (paying tokens to re-derive fields a `json.loads` gets exactly),
latency (a network round trip per invoice where a parse is microseconds), and
determinism (an LLM can misread a well-formed document; a structural parser cannot).
The fallback chain also means the failure mode of "no API key" is degraded extraction
quality, not a dead system — ingestion never raises on bad data; malformed fields
become `None` plus warnings that validation then flags.

## 2. Deterministic rules engine; the LLM reasons only where judgment is needed

Validation (`src/apcopilot/agents/validation.py` plus one module per rule family in
`agents/rules/`) makes zero LLM calls: data integrity, arithmetic, inventory, vendor
matching, duplicates, and fraud scoring are pure policy code against SQLite. That makes
every flag auditable (evidence attached), unit-testable, free, and identical on every
run — properties a controller needs and an LLM cannot promise. The LLM enters only at
approval, where there is genuine judgment: weighing non-blocking flags against vendor
history and policy. Every threshold — the $10K scrutiny line, tolerances, fraud
weights, confidence floors — lives in `data/seed/policies.yaml`, never in code, so
tuning policy is a config change for finance, not an engineering ticket.

## 3. A reflection loop with teeth

The approval stage (`src/apcopilot/agents/approval.py`) is propose/critique, but the
check that matters is not the second LLM call — it is `_verify_decision()`, which runs
in Python on every draft: each cited policy ID must resolve via `get_policy()`
(`tools/policy.py` returns `None` for invented IDs, so a hallucinated citation *fails
verification* instead of sounding authoritative), every HIGH/CRITICAL flag must be
explicitly addressed, and an APPROVE above the high-value threshold or the vendor's
auto-approve limit is rejected in code regardless of what the critic thinks. Numeric
policy is never delegated to a model's judgment. If the loop can't converge within
`max_approval_rounds`, the forced outcome is NEEDS_HUMAN with the outstanding issues
attached — the system escalates rather than shrugs.

## 4. Money is Decimal end-to-end

Amounts are `Decimal` in Python, TEXT in SQLite (a `sqlite3.register_adapter(Decimal,
str)` in `src/apcopilot/db/connection.py`), and strings over JSON. The trap this avoids
is real and specific: FastAPI's `jsonable_encoder` silently turns `Decimal` into
`float`, which is exactly the class of rounding bug the POL-MATH-01 rules exist to
catch — so API responses in `src/apcopilot/api/routes.py` round-trip through
`json.dumps(default=str)` to force string serialization. An AP system that introduces
float drift while checking invoices for arithmetic errors would be an embarrassment.

## 5. Stock is never decremented during validation

Checking an invoice must not mutate inventory: if it did, results would depend on the
order documents happen to be processed in, re-running a batch would give different
answers, and the eval would be unreproducible. Stock stays a read-only snapshot during
validation; aggregate demand pressure is surfaced instead through
`committed_vs_available()` in `src/apcopilot/tools/ledger.py`, which sums requested
quantities on flagged invoices against stock per SKU. In production, reservation
belongs at settlement, transactionally — not as a side effect of a validation read.

## 6. Idempotent payments and a checkpointed graph

The pipeline is a real LangGraph `StateGraph` with per-node SQLite checkpointing
(`src/apcopilot/graph/build.py`), and payments carry a unique idempotency key
(`invoice_number|amount|run_id` in `tools/ledger.py::record_payment`). The two
decisions compose: replaying a checkpoint past the pay node re-executes it, and the
unique key is what makes that a suppressed no-op rather than a double payment — the
single worst outcome an AP system can produce. Checkpointing also means a crash
between validate and approve resumes where it stopped instead of re-running ingestion
and re-billing the extraction LLM call. A linear four-step pipeline didn't strictly
need a graph framework; crash-safe resume and per-stage state transitions are why it
got one.

## 7. Duplicates and revisions by content hash

Every extraction is hashed (SHA-256 of the normalized invoice,
`ingestion/__init__.py::_content_hash`), and `agents/rules/duplicate.py` compares
against prior runs: same invoice number + same hash is the same document arriving in
another format (harmless); same number + different hash is a revision conflict that a
human must resolve; and if the earlier version was already paid, the later one is
blocked with a CRITICAL flag. This is exercised live by the corpus —
`data/invoices/invoice_1004.json` vs `invoice_1004_revised.json` — not a hypothetical.
The alternative (dedupe on invoice number alone) either blocks legitimate
re-transmissions or silently pays revised totals; content identity distinguishes the
two cases.

## 8. Fraud scoring is deterministic and auditable

`agents/rules/fraud.py` computes a weighted sum of named signals — unknown vendor,
blocked item, urgency language, payment-method-change requests, implausible dates,
suspiciously round high totals, amounts far above vendor history — with every weight in
`policies.yaml` and every fired signal recorded in the flag's evidence, so a controller
can reconstruct any score by hand. An LLM "fraud vibe check" was rejected: a fraud
determination that can't be explained can't be defended. Critically, adversarial
content is *data to score, never instructions to follow* — the extraction system
prompt (`ingestion/llm_extract.py`) explicitly instructs the model to treat "urgent",
"wire transfer", "new bank account" and similar as adversarial, copy them verbatim
into `notes` for the fraud lexicons to score, and never let them influence extracted
values. That closes the prompt-injection path where an invoice talks its way to
payment.

## 9. Four LLM modes: live / record / replay / off

`src/apcopilot/llm/client.py` honors `APCOPILOT_LLM_MODE`: `off` raises immediately so
every deterministic fallback path is exercised; `record` captures real responses as
fixtures keyed by hash of (model, system, user); `replay` serves those fixtures with
zero network access and never falls back to a live call. The practical payoff: a
grader can run the full system with no API key at all, and can also watch genuine
LLM reasoning offline by replaying the committed fixtures from the live Grok corpus
run. It's the same mechanism that makes the test suite deterministic and free.

## 10. Pinned as-of date

`Settings.as_of_date` is `2026-02-01` (`src/apcopilot/config.py`), not
`datetime.now()`. The corpus is dated January 2026 with February–March due dates; a
live clock would eventually mark every invoice past-due, shifting fraud scores and
decisions each time the repo is evaluated. Pinning the clock keeps every run
reproducible against the shipped data. The date is one config value away from being
live in production.

## 11. Engine-agnostic by design; Grok as the demonstrated engine

The brief names Grok as the preferred engine, and Grok is what this submission
demonstrates. The system was built engine-agnostic from the start: every LLM
interaction goes through one structured-call interface — a forced tool call returning
a schema-validated payload, a technique any tool-calling provider supports — and model
names are four fields in `Settings`. Initial development ran on Anthropic. Adopting
Grok as the demonstrated backend then cost exactly what the isolation claim predicted:
one new client (`llm/openai_compat.py`, the same forced-tool structured call against
any OpenAI-compatible `/chat/completions` endpoint) plus env vars
(`APCOPILOT_LLM_PROVIDER=openai_compat`, base URL `https://api.x.ai/v1`, Grok model
ids). No agent, rule, or graph code changed — the shallowness of that swap is the
evidence that nothing architectural ever depended on the vendor.

The Grok backend is live-verified, not hypothetical: the full sample corpus ran
against `grok-4.20-non-reasoning` (extraction) and `grok-4.20` (extraction retry, VP
approval, critic), exercising real multi-round reflection (invoice_1007 took two
critique rounds) and real injection resistance, and the fixtures recorded from that
run are committed under `tests/fixtures/llm/`, replayable with no key. Anthropic
remains a first-class backend and the code default — `ANTHROPIC_API_KEY` alone
enables it — and Gemini's free tier or local Ollama are each a `.env` change; see
"Running the live LLM path for free" in SETUP.md. Modes, fixtures, per-attempt
logging, and cost accounting stay in the shared wrapper; only the single live-call
step branches on the provider.

## What I'd do next with more time

- **SSE instead of polling** — the dashboard currently polls the run store on a timer
  (`api/static/app.js`); server-sent events off the trace table would make batch
  progress feel live and cut request noise.
- **File upload** — drag a new invoice into the dashboard instead of only processing
  the shipped corpus.
- **Vendor onboarding flow** — an unknown vendor currently dead-ends in
  rejection/escalation; a review queue that promotes vetted vendors into the master
  (with an auto-approve limit) closes the loop.
- **Extraction eval harness** — the corpus is effectively labeled; a per-field
  precision/recall harness would turn prompt and model changes from vibes into
  measured regressions, and justify the cheap-first/stronger-retry model split with
  numbers.
- **Partial approval for stock shortfalls** — POL-STOCK-01 already contemplates
  paying up to available stock with human confirmation; the workflow isn't built.
- **Prompt caching** — the catalog and policy digests are stable per run; caching the
  system prefix would cut extraction and approval token cost further.

-- The case brief's starter schema was:
--   CREATE TABLE inventory (item TEXT PRIMARY KEY, stock INTEGER)
--   INSERT INTO inventory VALUES ('WidgetA',15),('WidgetB',10),('GadgetX',5),('FakeItem',0)
-- `items` below is a strict superset: same four rows, same stock levels, plus the
-- columns richer validation needs (unit_price for variance checks, status so
-- FakeItem can be actively blocked rather than merely zero-stock).

CREATE TABLE IF NOT EXISTS items (
    sku         TEXT PRIMARY KEY,   -- canonicalized: uppercase, no spaces/punctuation
    name        TEXT NOT NULL,      -- display name as it appears in the catalog
    unit_price  NUMERIC,            -- catalog price in USD; NULL when not purchasable
    stock       INTEGER NOT NULL,
    category    TEXT,
    status      TEXT NOT NULL DEFAULT 'active'  -- active | discontinued | blocked
);

CREATE TABLE IF NOT EXISTS vendors (
    name               TEXT PRIMARY KEY,
    aka                TEXT,               -- '|'-separated former/alternate names
    status             TEXT NOT NULL DEFAULT 'active',  -- active | watchlist | blocked
    risk_tier          TEXT NOT NULL DEFAULT 'medium',  -- low | medium | high
    auto_approve_limit NUMERIC NOT NULL DEFAULT 0,
    currency           TEXT NOT NULL DEFAULT 'USD',
    first_seen         DATE,
    name_changed_at    DATE
);

CREATE TABLE IF NOT EXISTS fx_rates (
    base   TEXT NOT NULL,
    quote  TEXT NOT NULL,
    rate   NUMERIC NOT NULL,
    as_of  DATE NOT NULL,
    PRIMARY KEY (base, quote)
);

-- Operational tables. These are what the web UI reads; the queue survives
-- restarts because it is just a table.

CREATE TABLE IF NOT EXISTS invoice_runs (
    run_id            TEXT PRIMARY KEY,
    batch_id          TEXT,
    document_path     TEXT NOT NULL,
    source_format     TEXT NOT NULL,
    content_hash      TEXT,            -- sha256 of the normalized extraction
    invoice_number    TEXT,
    revision          TEXT,
    vendor_name       TEXT,
    currency          TEXT,
    total             NUMERIC,
    total_usd         NUMERIC,
    due_date          TEXT,
    status            TEXT NOT NULL,   -- queued|running|approved|rejected|needs_human|failed
    lane              TEXT,            -- auto_approve | auto_reject | review
    confidence        NUMERIC,
    extraction_attempts INTEGER DEFAULT 0,
    approval_rounds   INTEGER DEFAULT 0,
    fraud_score       INTEGER DEFAULT 0,
    input_tokens      INTEGER DEFAULT 0,
    output_tokens     INTEGER DEFAULT 0,
    cost_usd          NUMERIC DEFAULT 0,
    duration_ms       INTEGER,
    extraction_json   TEXT,            -- full ExtractedInvoice
    decision_json     TEXT,            -- full ApprovalDecision
    error             TEXT,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_invoice   ON invoice_runs(invoice_number);
CREATE INDEX IF NOT EXISTS idx_runs_status    ON invoice_runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_batch     ON invoice_runs(batch_id);
CREATE INDEX IF NOT EXISTS idx_runs_hash      ON invoice_runs(content_hash);

CREATE TABLE IF NOT EXISTS invoice_flags (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT NOT NULL REFERENCES invoice_runs(run_id),
    rule_id      TEXT NOT NULL,
    code         TEXT NOT NULL,
    severity     TEXT NOT NULL,
    message      TEXT NOT NULL,
    sku          TEXT,
    evidence_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_flags_run ON invoice_flags(run_id);

CREATE TABLE IF NOT EXISTS trace_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL REFERENCES invoice_runs(run_id),
    seq         INTEGER NOT NULL,
    node        TEXT NOT NULL,
    status      TEXT NOT NULL,
    summary     TEXT,
    duration_ms INTEGER,
    detail_json TEXT,
    started_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trace_run ON trace_events(run_id, seq);

CREATE TABLE IF NOT EXISTS llm_calls (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                TEXT,
    node                  TEXT NOT NULL,
    model                 TEXT NOT NULL,
    prompt_name           TEXT,
    prompt_hash           TEXT,
    attempt               INTEGER DEFAULT 1,
    input_tokens          INTEGER DEFAULT 0,
    output_tokens         INTEGER DEFAULT 0,
    cache_read_tokens     INTEGER DEFAULT 0,
    cache_creation_tokens INTEGER DEFAULT 0,
    cost_usd              NUMERIC DEFAULT 0,
    latency_ms            INTEGER,
    stop_reason           TEXT,
    request_preview       TEXT,
    response_preview      TEXT,
    created_at            TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_llm_run ON llm_calls(run_id);

CREATE TABLE IF NOT EXISTS payments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    idempotency_key TEXT NOT NULL UNIQUE,   -- invoice_number|amount|run_id
    run_id          TEXT NOT NULL REFERENCES invoice_runs(run_id),
    invoice_number  TEXT NOT NULL,
    vendor          TEXT NOT NULL,
    amount          NUMERIC NOT NULL,
    currency        TEXT NOT NULL DEFAULT 'USD',
    status          TEXT NOT NULL,
    paid_at         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_payments_invoice ON payments(invoice_number);

CREATE TABLE IF NOT EXISTS rejections (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         TEXT NOT NULL REFERENCES invoice_runs(run_id),
    invoice_number TEXT,
    vendor         TEXT,
    amount         NUMERIC,
    reason         TEXT NOT NULL,
    decided_by     TEXT NOT NULL,     -- rules | vp_agent | human
    detail_json    TEXT,
    rejected_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS human_actions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT NOT NULL REFERENCES invoice_runs(run_id),
    actor      TEXT NOT NULL,
    outcome    TEXT NOT NULL,
    note       TEXT,
    acted_at   TEXT NOT NULL
);

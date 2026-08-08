// AP Copilot dashboard — vanilla JS, no build step, same-origin fetch against
// the FastAPI backend's /api/* JSON endpoints. Defensive about field names on
// /api/dashboard (built in parallel, contract explicitly says it may drift);
// the documented /api/runs and /api/runs/{id} shapes are trusted more directly.

(() => {
  "use strict";

  const POLL_MS = 3000;

  const state = {
    runs: [],
    policies: null,
    policyRuleMap: {},
    dashboard: null,
    statusFilter: "",
    selectedRunId: null,
    pollTimer: null,
    inFlight: false, // true while a manual run/batch/reset action is in progress
  };

  // ---------------------------------------------------------------------
  // small utilities
  // ---------------------------------------------------------------------

  const $ = (sel) => document.querySelector(sel);

  function escapeHtml(str) {
    if (str === null || str === undefined) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function get(obj, path, fallback) {
    // dot-path getter used only for the loosely-specified /api/dashboard
    // payload, so the UI doesn't break if field names drift slightly.
    try {
      const parts = path.split(".");
      let cur = obj;
      for (const p of parts) {
        if (cur === null || cur === undefined) return fallback;
        cur = cur[p];
      }
      return cur === undefined || cur === null ? fallback : cur;
    } catch (_e) {
      return fallback;
    }
  }

  function firstDefined(obj, paths, fallback) {
    for (const p of paths) {
      const v = get(obj, p, undefined);
      if (v !== undefined) return v;
    }
    return fallback;
  }

  function toNumber(v) {
    if (v === null || v === undefined || v === "") return null;
    const n = typeof v === "number" ? v : parseFloat(v);
    return Number.isFinite(n) ? n : null;
  }

  function fmtMoney(value, currency) {
    const n = toNumber(value);
    if (n === null) return "—";
    try {
      return n.toLocaleString(undefined, {
        style: "currency",
        currency: currency || "USD",
        maximumFractionDigits: 2,
      });
    } catch (_e) {
      return `${n.toFixed(2)} ${currency || ""}`.trim();
    }
  }

  function fmtNumber(value, opts) {
    const n = toNumber(value);
    if (n === null) return "—";
    return n.toLocaleString(undefined, opts || {});
  }

  function fmtPercent(value, opts) {
    const n = toNumber(value);
    if (n === null) return "—";
    // accept either 0..1 fractions or already-percent 0..100 values
    const pct = n <= 1 ? n * 100 : n;
    return `${pct.toLocaleString(undefined, { maximumFractionDigits: 1, ...opts })}%`;
  }

  function fmtDuration(ms) {
    const n = toNumber(ms);
    if (n === null) return "—";
    if (n < 1000) return `${Math.round(n)}ms`;
    return `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}s`;
  }

  function fmtDateShort(iso) {
    if (!iso) return "—";
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return String(iso);
    return d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
  }

  function timeAgo(iso) {
    if (!iso) return "—";
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return String(iso);
    const diffMs = Date.now() - d.getTime();
    const s = Math.round(diffMs / 1000);
    if (s < 5) return "just now";
    if (s < 60) return `${s}s ago`;
    const m = Math.round(s / 60);
    if (m < 60) return `${m}m ago`;
    const h = Math.round(m / 60);
    if (h < 24) return `${h}h ago`;
    const dd = Math.round(h / 24);
    return `${dd}d ago`;
  }

  // ---------------------------------------------------------------------
  // API helpers
  // ---------------------------------------------------------------------

  async function apiGet(path) {
    const res = await fetch(path, { headers: { Accept: "application/json" } });
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw new Error(`GET ${path} failed: ${res.status} ${body.slice(0, 200)}`);
    }
    return res.json();
  }

  async function apiPost(path, body) {
    const res = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`POST ${path} failed: ${res.status} ${text.slice(0, 300)}`);
    }
    return res.json();
  }

  // ---------------------------------------------------------------------
  // toasts
  // ---------------------------------------------------------------------

  function toast(message, type) {
    const container = $("#toast-container");
    const el = document.createElement("div");
    el.className = `toast${type ? ` toast-${type}` : ""}`;
    el.textContent = message;
    container.appendChild(el);
    setTimeout(() => {
      el.style.transition = "opacity 0.25s ease";
      el.style.opacity = "0";
      setTimeout(() => el.remove(), 260);
    }, type === "error" ? 6000 : 3800);
  }

  // ---------------------------------------------------------------------
  // impact strip (business case translation)
  // ---------------------------------------------------------------------

  function computeImpact(dashboard, policies, runs) {
    const d = dashboard || {};

    const statusCounts = firstDefined(
      d,
      ["status_counts", "counts_by_status", "counts", "by_status"],
      null
    );

    const totalRuns = firstDefined(
      d,
      ["total_runs", "run_count", "counts.total", "total"],
      runs.length
    );

    const avgDurationMs = firstDefined(
      d,
      ["avg_duration_ms", "average_duration_ms", "duration.avg_ms", "avg_processing_ms"],
      null
    );

    const totalCostUsd = firstDefined(
      d,
      ["total_cost_usd", "cost.total_usd", "total_llm_cost_usd"],
      null
    );

    const avgCostUsd = firstDefined(
      d,
      ["avg_cost_usd", "average_cost_usd", "cost.avg_usd", "avg_llm_cost_usd"],
      null
    );

    // Business case inputs: prefer whatever /api/dashboard computed itself;
    // fall back to the raw policy config's business_case block (guaranteed
    // present there per the seed policies.yaml) and derive locally so the
    // hero number never just goes blank.
    const bc = get(policies, "business_case", {}) || {};

    const manualMinutes = firstDefined(
      d,
      ["baseline.manual_minutes_per_invoice", "savings.manual_minutes_per_invoice"],
      toNumber(bc.manual_minutes_per_invoice) ?? 22
    );
    const hourlyRate = firstDefined(
      d,
      ["baseline.fully_loaded_hourly_rate", "savings.fully_loaded_hourly_rate"],
      toNumber(bc.fully_loaded_hourly_rate) ?? 48
    );
    const baselineErrorRate = firstDefined(
      d,
      ["error_rate_caught", "baseline_error_rate", "baseline.error_rate"],
      toNumber(bc.baseline_error_rate) ?? 0.3
    );
    const baselineDays = firstDefined(
      d,
      ["baseline_days_to_process", "baseline.days_to_process"],
      toNumber(bc.baseline_days_to_process) ?? 5
    );

    const baselineCostPerInvoice = firstDefined(
      d,
      [
        "savings.baseline_cost_per_invoice_usd",
        "baseline_cost_per_invoice_usd",
        "baseline_manual_cost_per_invoice_usd",
      ],
      (manualMinutes / 60) * hourlyRate
    );

    // actual cost per invoice: prefer explicit field, else avg LLM cost from
    // dashboard, else compute from the runs currently loaded in the table.
    const runsWithCost = runs.filter((r) => toNumber(r.cost_usd) !== null);
    const localAvgCost =
      runsWithCost.length > 0
        ? runsWithCost.reduce((sum, r) => sum + toNumber(r.cost_usd), 0) / runsWithCost.length
        : null;

    const actualCostPerInvoice = firstDefined(
      d,
      ["savings.actual_cost_per_invoice_usd", "actual_cost_per_invoice_usd"],
      toNumber(avgCostUsd) ?? localAvgCost ?? 0
    );

    const assumedAnnualVolume = firstDefined(
      d,
      ["savings.assumed_annual_volume", "assumed_annual_volume", "annual_volume"],
      50000
    );

    const explicitAnnualSavings = firstDefined(
      d,
      [
        "estimated_annual_savings_usd",
        "annual_savings_usd",
        "savings.annual_usd",
        "savings.estimated_annual_usd",
        "savings.estimated_annual_savings_usd",
      ],
      null
    );

    const computedAnnualSavings =
      (toNumber(baselineCostPerInvoice) ?? 0) - (toNumber(actualCostPerInvoice) ?? 0);
    const annualSavings =
      toNumber(explicitAnnualSavings) ??
      Math.max(0, computedAnnualSavings) * (toNumber(assumedAnnualVolume) ?? 50000);
    const annualSavingsIsEstimate = toNumber(explicitAnnualSavings) === null;

    // avg processing time: prefer dashboard's own number, else derive from
    // loaded runs, else fall back to the baseline days figure inverted.
    const runsWithDuration = runs.filter((r) => toNumber(r.duration_ms) !== null);
    const localAvgDuration =
      runsWithDuration.length > 0
        ? runsWithDuration.reduce((sum, r) => sum + toNumber(r.duration_ms), 0) /
          runsWithDuration.length
        : null;
    const processingTimeMs = toNumber(avgDurationMs) ?? localAvgDuration;

    return {
      statusCounts,
      totalRuns,
      avgDurationMs: processingTimeMs,
      totalCostUsd: toNumber(totalCostUsd),
      avgCostUsd: toNumber(avgCostUsd) ?? localAvgCost,
      baselineErrorRate,
      baselineDays,
      baselineCostPerInvoice: toNumber(baselineCostPerInvoice),
      actualCostPerInvoice: toNumber(actualCostPerInvoice),
      assumedAnnualVolume: toNumber(assumedAnnualVolume),
      annualSavings,
      annualSavingsIsEstimate,
    };
  }

  function renderImpactStrip() {
    const el = $("#impact-strip");
    const impact = computeImpact(state.dashboard, state.policies, state.runs);

    if (!state.dashboard && state.runs.length === 0) {
      el.innerHTML = `<div class="impact-skeleton">Loading business impact summary&hellip;</div>`;
      return;
    }

    const savingsSub = `vs. ~${fmtMoney(impact.baselineCostPerInvoice)}/invoice manual baseline &rarr; ~${fmtMoney(
      impact.actualCostPerInvoice
    )}/invoice actual, at an assumed ${fmtNumber(impact.assumedAnnualVolume)} invoices/yr${
      impact.annualSavingsIsEstimate ? " (estimated client-side)" : ""
    }`;

    const tiles = [
      {
        hero: true,
        label: "Est. annual savings",
        value: fmtMoney(impact.annualSavings),
        sub: savingsSub,
      },
      {
        label: "Baseline error rate caught",
        value: fmtPercent(impact.baselineErrorRate),
        sub: `manual process baseline &bull; was ${fmtNumber(impact.baselineDays)}-day turnaround`,
      },
      {
        label: "Avg. processing time",
        value: fmtDuration(impact.avgDurationMs),
        sub: "ingest through decision, per invoice",
      },
      {
        label: "Runs processed",
        value: fmtNumber(impact.totalRuns),
        sub: impact.statusCounts
          ? Object.entries(impact.statusCounts)
              .map(([k, v]) => `${escapeHtml(k)}: ${fmtNumber(v)}`)
              .join(" &bull; ")
          : "since last reset",
      },
      {
        label: "Avg. LLM cost / invoice",
        value: impact.avgCostUsd !== null ? fmtMoney(impact.avgCostUsd) : "—",
        sub: impact.totalCostUsd !== null ? `${fmtMoney(impact.totalCostUsd)} total spend` : " ",
      },
    ];

    el.innerHTML = tiles
      .map(
        (t) => `
        <div class="impact-tile${t.hero ? " impact-hero" : ""}">
          <div class="impact-label">${escapeHtml(t.label)}</div>
          <div class="impact-value">${t.value}</div>
          <div class="impact-sub">${t.sub}</div>
        </div>`
      )
      .join("");
  }

  // ---------------------------------------------------------------------
  // rulebook panel
  // ---------------------------------------------------------------------

  function renderRulebook() {
    const el = $("#rulebook-content");
    const p = state.policies;
    if (!p) {
      el.innerHTML = `<p class="muted">Could not load policy configuration.</p>`;
      return;
    }

    const thresholds = p.thresholds || {};
    const tolerances = p.tolerances || {};
    const fraud = p.fraud || {};
    const rules = p.policy_rules || {};

    const statPairs = [
      ["High-value threshold", fmtMoney(thresholds.high_value)],
      ["Medium-flag value floor", fmtMoney(thresholds.medium_flag_value)],
      ["Auto-approve confidence", fmtPercent(thresholds.auto_approve_confidence)],
      ["Human-gate confidence", fmtPercent(thresholds.human_gate_confidence)],
      ["Max extraction attempts", fmtNumber(thresholds.max_extraction_attempts)],
      ["Max approval rounds", fmtNumber(thresholds.max_approval_rounds)],
      ["Money tolerance (abs)", fmtMoney(tolerances.money_abs)],
      ["Money tolerance (rel)", fmtPercent(tolerances.money_rel)],
      ["Fraud: high threshold", fmtNumber(fraud.high_threshold)],
      ["Fraud: critical threshold", fmtNumber(fraud.critical_threshold)],
    ].filter(([, v]) => v !== "—");

    const statsHtml = statPairs
      .map(
        ([label, value]) => `
        <div class="rulebook-stat">
          <div class="rb-label">${escapeHtml(label)}</div>
          <div class="rb-value">${escapeHtml(value)}</div>
        </div>`
      )
      .join("");

    const ruleIds = Object.keys(rules).sort();
    const rulesHtml = ruleIds.length
      ? ruleIds
          .map((id) => {
            const r = rules[id] || {};
            return `
            <div class="rule-card">
              <div class="rule-id">${escapeHtml(id)}</div>
              <div class="rule-title">${escapeHtml(r.title || "")}</div>
              <div class="rule-text">${escapeHtml(r.text || "")}</div>
            </div>`;
          })
          .join("")
      : `<p class="muted">No policy rules found.</p>`;

    el.innerHTML = `
      <div class="rulebook-grid">${statsHtml}</div>
      <div class="rulebook-section-title">Policy rules (cited by the approval agent)</div>
      ${rulesHtml}
    `;
  }

  // ---------------------------------------------------------------------
  // run queue table
  // ---------------------------------------------------------------------

  function statusPillHtml(status) {
    const cls = `pill-${status || "queued"}`;
    const label = (status || "unknown").replace(/_/g, " ");
    return `<span class="pill ${cls}">${escapeHtml(label)}</span>`;
  }

  function laneTagHtml(lane) {
    if (!lane) return `<span class="faint">—</span>`;
    return `<span class="lane-tag">${escapeHtml(lane.replace(/_/g, " "))}</span>`;
  }

  function fraudCellHtml(score, policies) {
    const n = toNumber(score);
    if (n === null) return `<span class="faint">—</span>`;
    const highT = toNumber(get(policies, "fraud.critical_threshold", null)) ?? 80;
    const medT = toNumber(get(policies, "fraud.high_threshold", null)) ?? 60;
    let cls = "";
    if (n >= highT) cls = "fraud-hi";
    else if (n >= medT) cls = "fraud-med";
    return `<span class="fraud-score ${cls}">${n}</span>`;
  }

  function flagsCountForRow(row) {
    // /api/runs rows are not documented to include a flag count; check a few
    // plausible extra field names before giving up, since the backend is
    // being built in parallel and may include convenience fields.
    return firstDefined(row, ["flags_count", "flag_count", "num_flags", "flags.length"], null);
  }

  function renderRunsTable() {
    const table = $("#runs-table");
    const emptyEl = $("#queue-empty");
    const loadingEl = $("#queue-loading");
    const tbody = $("#runs-tbody");

    loadingEl.hidden = true;

    if (!state.runs.length) {
      table.hidden = true;
      emptyEl.hidden = false;
      return;
    }

    emptyEl.hidden = true;
    table.hidden = false;

    tbody.innerHTML = state.runs
      .map((r) => {
        const flagsCount = flagsCountForRow(r);
        return `
        <tr data-run-id="${escapeHtml(r.run_id)}">
          <td class="cell-invoice">
            <span class="mono">${escapeHtml(r.invoice_number || "—")}</span>
            <span class="run-doc">${escapeHtml((r.document_path || "").split("/").pop() || "")}</span>
          </td>
          <td class="cell-vendor">${escapeHtml(r.vendor_name || "—")}</td>
          <td class="num mono">${fmtMoney(r.total, r.currency)}</td>
          <td>${statusPillHtml(r.status)}</td>
          <td>${laneTagHtml(r.lane)}</td>
          <td class="num">${flagsCount === null ? '<span class="faint">—</span>' : fmtNumber(flagsCount)}</td>
          <td class="num">${fraudCellHtml(r.fraud_score, state.policies)}</td>
          <td class="faint">${escapeHtml(timeAgo(r.updated_at || r.created_at))}</td>
        </tr>`;
      })
      .join("");

    tbody.querySelectorAll("tr[data-run-id]").forEach((tr) => {
      tr.addEventListener("click", () => openDetail(tr.getAttribute("data-run-id")));
    });
  }

  // ---------------------------------------------------------------------
  // detail panel
  // ---------------------------------------------------------------------

  function policyChipHtml(ruleId) {
    const rule = state.policyRuleMap[ruleId];
    if (!rule) {
      return `<button type="button" class="policy-chip unknown-rule" data-rule-id="${escapeHtml(
        ruleId
      )}" title="Not found in policy config">${escapeHtml(ruleId)} ⚠</button>`;
    }
    return `<button type="button" class="policy-chip" data-rule-id="${escapeHtml(
      ruleId
    )}" title="${escapeHtml(rule.title || "")}">${escapeHtml(ruleId)}</button>`;
  }

  function renderKvGrid(pairs) {
    return `<div class="kv-grid">${pairs
      .map(
        ([label, value]) => `
        <div class="kv-item">
          <div class="kv-label">${escapeHtml(label)}</div>
          <div class="kv-value">${value === null || value === undefined || value === "" ? "—" : value}</div>
        </div>`
      )
      .join("")}</div>`;
  }

  function renderLineItems(items) {
    if (!Array.isArray(items) || items.length === 0) {
      return `<p class="muted">No line items extracted.</p>`;
    }
    const rows = items
      .map(
        (li) => `
        <tr>
          <td>${escapeHtml(li.description || "—")}</td>
          <td class="mono">${escapeHtml(li.sku || "—")}</td>
          <td class="mono">${fmtNumber(li.quantity)}</td>
          <td class="mono">${li.unit_price !== undefined && li.unit_price !== null ? fmtMoney(li.unit_price) : "—"}</td>
          <td class="mono">${li.line_total !== undefined && li.line_total !== null ? fmtMoney(li.line_total) : "—"}</td>
        </tr>`
      )
      .join("");
    return `
      <table class="mini-table">
        <thead>
          <tr><th>Description</th><th>SKU</th><th>Qty</th><th>Unit price</th><th>Line total</th></tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>`;
  }

  function renderFlags(flags, addressedIds) {
    if (!Array.isArray(flags) || flags.length === 0) {
      return `<p class="muted">No validation flags raised.</p>`;
    }
    const order = { CRITICAL: 0, HIGH: 1, MEDIUM: 2, INFO: 3 };
    const sorted = [...flags].sort(
      (a, b) => (order[a.severity] ?? 9) - (order[b.severity] ?? 9)
    );
    const addressed = new Set(addressedIds || []);
    return sorted
      .map((f) => {
        const isAddressed = addressed.has(f.rule_id);
        const evidence =
          f.evidence && Object.keys(f.evidence).length
            ? `<details class="flag-evidence"><summary>Evidence</summary><pre>${escapeHtml(
                JSON.stringify(f.evidence, null, 2)
              )}</pre></details>`
            : "";
        return `
        <div class="flag-card sev-border-${escapeHtml(f.severity || "")}">
          <div class="flag-card-top">
            <span class="sev-badge sev-${escapeHtml(f.severity || "")}">${escapeHtml(f.severity || "")}</span>
            <span class="flag-rule-id">${escapeHtml(f.rule_id || "")} &middot; ${escapeHtml(f.code || "")}</span>
            ${f.sku ? `<span class="flag-rule-id">sku: ${escapeHtml(f.sku)}</span>` : ""}
            ${isAddressed ? `<span class="flag-addressed">✓ addressed in decision</span>` : ""}
          </div>
          <div class="flag-message">${escapeHtml(f.message || "")}</div>
          ${evidence}
        </div>`;
      })
      .join("");
  }

  function renderDecision(decision, flags) {
    if (!decision) {
      return `<p class="muted">No approval decision recorded yet.</p>`;
    }
    const citedIds = Array.isArray(decision.cited_policy_ids) ? decision.cited_policy_ids : [];
    const chips = citedIds.length
      ? `<div class="policy-chips">${citedIds.map(policyChipHtml).join("")}</div>
         <div class="policy-chip-details"></div>`
      : `<p class="muted" style="margin-top:6px;">No policy rules cited.</p>`;

    const critiques =
      Array.isArray(decision.critique_notes) && decision.critique_notes.length
        ? `<div class="detail-section" style="margin-top:14px;">
             <h3>Critique / reflection notes (round-trip ${escapeHtml(decision.rounds ?? "?")})</h3>
             <ol class="critique-notes">${decision.critique_notes
               .map((n) => `<li>${escapeHtml(n)}</li>`)
               .join("")}</ol>
           </div>`
        : "";

    return `
      <div class="decision-block">
        <div class="stats-row">
          <span class="stat-chip">Decision${statusPillHtml(decision.decision)}</span>
          <span class="stat-chip">Decided by <span class="stat-chip-value">${escapeHtml(
            decision.decided_by || "—"
          )}</span></span>
          <span class="stat-chip">Confidence <span class="stat-chip-value">${fmtPercent(
            decision.confidence
          )}</span></span>
          <span class="stat-chip">Rounds <span class="stat-chip-value">${fmtNumber(decision.rounds)}</span></span>
        </div>
        <div class="decision-rationale">${escapeHtml(decision.rationale || "")}</div>
        <div class="detail-section" style="margin-top:0;">
          <h3>Cited policy rules</h3>
          ${chips}
        </div>
        ${critiques}
      </div>`;
  }

  function traceStatusClass(status) {
    const s = (status || "").toLowerCase();
    if (s.includes("error") || s.includes("fail")) return "tl-error";
    if (s.includes("warn") || s.includes("retry")) return "tl-warn";
    if (s.includes("ok") || s.includes("success") || s.includes("done") || s.includes("complete")) return "tl-ok";
    return "";
  }

  function renderTrace(trace) {
    if (!Array.isArray(trace) || trace.length === 0) {
      return `<p class="muted">No trace events recorded.</p>`;
    }
    const sorted = [...trace].sort((a, b) => (a.seq ?? 0) - (b.seq ?? 0));
    const items = sorted
      .map((t) => {
        const detail =
          t.detail && (typeof t.detail !== "object" || Object.keys(t.detail).length)
            ? `<details><summary>Detail</summary><pre>${escapeHtml(
                typeof t.detail === "string" ? t.detail : JSON.stringify(t.detail, null, 2)
              )}</pre></details>`
            : "";
        return `
        <div class="timeline-item ${traceStatusClass(t.status)}">
          <div class="timeline-head">
            <span class="timeline-node">${escapeHtml(t.node || "")}</span>
            <span class="timeline-status">${escapeHtml(t.status || "")}</span>
            <span class="timeline-duration">${fmtDuration(t.duration_ms)}</span>
          </div>
          ${t.summary ? `<div class="timeline-summary">${escapeHtml(t.summary)}</div>` : ""}
          ${detail}
        </div>`;
      })
      .join("");
    return `<div class="timeline">${items}</div>`;
  }

  function humanOutcomePillHtml(outcome) {
    // reuse the status-pill palette: approve → approved green, reject → rejected red
    const cls =
      outcome === "approve" ? "pill-approved" : outcome === "reject" ? "pill-rejected" : "pill-queued";
    return `<span class="pill ${cls}">${escapeHtml(outcome ?? "unknown")}</span>`;
  }

  function renderHumanActionBar() {
    return `
      <div class="detail-section">
        <h3>Human review required</h3>
        <div class="action-bar">
          <p class="action-bar-hint">This invoice needs VP review — the approval agent could not auto-decide.</p>
          <div class="action-bar-controls">
            <input
              type="text"
              id="action-note-input"
              placeholder="Optional note (kept in the audit trail)"
              autocomplete="off"
              spellcheck="false"
            />
            <button type="button" class="btn btn-approve" id="action-approve-btn">Approve &amp; pay</button>
            <button type="button" class="btn btn-reject" id="action-reject-btn">Reject</button>
          </div>
          <div class="error-banner" id="action-error" hidden></div>
        </div>
      </div>`;
  }

  function renderHumanActions(actions) {
    if (!Array.isArray(actions) || actions.length === 0) return "";
    const items = actions
      .map(
        (a) => `
        <div class="human-action-item">
          <div class="human-action-head">
            ${humanOutcomePillHtml(a?.outcome)}
            <span class="human-action-actor mono">${escapeHtml(a?.actor ?? "unknown")}</span>
            <span class="human-action-time faint" title="${escapeHtml(a?.acted_at ?? "")}">${escapeHtml(
              timeAgo(a?.acted_at ?? null)
            )}</span>
          </div>
          ${a?.note ? `<div class="human-action-note">${escapeHtml(a.note)}</div>` : ""}
        </div>`
      )
      .join("");
    return `
      <div class="detail-section">
        <h3>Human review</h3>
        <div class="human-actions">${items}</div>
      </div>`;
  }

  function renderDetailContent(data) {
    const run = data.run || {};
    const flags = data.flags || [];
    const trace = data.trace || [];
    const humanActions = data.human_actions || [];
    const extraction = run.extraction_json || {};
    const decision = run.decision_json || null;

    const warnings = Array.isArray(extraction.extraction_warnings)
      ? extraction.extraction_warnings
      : [];

    $("#detail-content").innerHTML = `
      <div class="detail-header">
        <div>
          <h2>${escapeHtml(run.invoice_number || "Unknown invoice")}</h2>
          <div class="run-id mono">${escapeHtml(run.run_id || "")}</div>
        </div>
        <div class="detail-badges">
          ${statusPillHtml(run.status)}
        </div>
      </div>
      <div class="detail-meta">
        ${escapeHtml(run.vendor_name || "Unknown vendor")} &bull;
        ${escapeHtml(run.document_path || "")} (${escapeHtml(run.source_format || "?")}) &bull;
        lane: ${escapeHtml(run.lane || "—")}
      </div>

      ${
        run.error
          ? `<div class="error-banner"><strong>Error:</strong> ${escapeHtml(run.error)}</div>`
          : ""
      }

      <div class="detail-section">
        <div class="stats-row">
          <span class="stat-chip">Total <span class="stat-chip-value">${fmtMoney(
            run.total,
            run.currency
          )}</span></span>
          <span class="stat-chip">Total (USD) <span class="stat-chip-value">${fmtMoney(
            run.total_usd
          )}</span></span>
          <span class="stat-chip">Due <span class="stat-chip-value">${fmtDateShort(run.due_date)}</span></span>
          <span class="stat-chip">Confidence <span class="stat-chip-value">${fmtPercent(
            run.confidence
          )}</span></span>
          <span class="stat-chip">Fraud score ${fraudCellHtml(run.fraud_score, state.policies)}</span>
          <span class="stat-chip">Extraction attempts <span class="stat-chip-value">${fmtNumber(
            run.extraction_attempts
          )}</span></span>
          <span class="stat-chip">Approval rounds <span class="stat-chip-value">${fmtNumber(
            run.approval_rounds
          )}</span></span>
          <span class="stat-chip">Cost <span class="stat-chip-value">${fmtMoney(run.cost_usd)}</span></span>
          <span class="stat-chip">Duration <span class="stat-chip-value">${fmtDuration(
            run.duration_ms
          )}</span></span>
        </div>
      </div>

      <div class="detail-section">
        <h3>Extracted invoice</h3>
        ${renderKvGrid([
          ["Currency", escapeHtml(extraction.currency || run.currency || "—")],
          ["Subtotal", extraction.subtotal !== undefined && extraction.subtotal !== null ? fmtMoney(extraction.subtotal) : null],
          ["Tax", extraction.tax !== undefined && extraction.tax !== null ? fmtMoney(extraction.tax) : null],
          ["Invoice date", extraction.invoice_date ? fmtDateShort(extraction.invoice_date) : null],
          ["Payment terms", escapeHtml(extraction.payment_terms || "")],
          ["Revision", escapeHtml(extraction.revision || "")],
        ])}
        ${extraction.notes ? `<p class="muted" style="margin-top:8px;">${escapeHtml(extraction.notes)}</p>` : ""}
        ${
          warnings.length
            ? `<ul class="warn-list">${warnings.map((w) => `<li>${escapeHtml(w)}</li>`).join("")}</ul>`
            : ""
        }
      </div>

      <div class="detail-section">
        <h3>Line items</h3>
        ${renderLineItems(extraction.line_items)}
      </div>

      <div class="detail-section">
        <h3>Validation flags (${flags.length})</h3>
        ${renderFlags(flags, decision ? decision.addressed_flag_rule_ids : [])}
      </div>

      <div class="detail-section">
        <h3>Approval decision</h3>
        ${renderDecision(decision, flags)}
      </div>

      ${run.status === "needs_human" ? renderHumanActionBar() : ""}
      ${renderHumanActions(humanActions)}

      <div class="detail-section">
        <h3>Trace timeline</h3>
        ${renderTrace(trace)}
      </div>
    `;

    const approveBtn = $("#action-approve-btn");
    const rejectBtn = $("#action-reject-btn");
    if (approveBtn && rejectBtn) {
      approveBtn.addEventListener("click", () => submitHumanAction(run.run_id, "approve"));
      rejectBtn.addEventListener("click", () => submitHumanAction(run.run_id, "reject"));
    }

    // policy citation chips: click-to-expand rule title + text (also has a
    // native title-attribute tooltip for hover). Unknown rule ids (a
    // hallucinated citation that failed verification) render with a warning
    // style and an explicit "not found" message instead of silently citing
    // something that doesn't exist in the policy config.
    $("#detail-content").querySelectorAll(".policy-chip").forEach((chip) => {
      chip.addEventListener("click", () => {
        const ruleId = chip.getAttribute("data-rule-id");
        const rule = state.policyRuleMap[ruleId];
        const container = chip.parentElement.nextElementSibling;
        if (!container) return;
        const existing = container.querySelector(`[data-for="${CSS.escape(ruleId)}"]`);
        if (existing) {
          existing.remove();
          return;
        }
        const box = document.createElement("div");
        box.className = `policy-chip-detail${rule ? "" : " pcd-unknown"}`;
        box.setAttribute("data-for", ruleId);
        box.innerHTML = rule
          ? `<div class="pcd-title">${escapeHtml(ruleId)} &mdash; ${escapeHtml(rule.title || "")}</div><div>${escapeHtml(
              rule.text || ""
            )}</div>`
          : `<div class="pcd-title">${escapeHtml(ruleId)} &mdash; unknown rule id</div><div>This id was cited by the decision agent but does not exist in the loaded policy config. Citation verification failed — treat this rationale with suspicion.</div>`;
        container.appendChild(box);
      });
    });
  }

  async function openDetail(runId) {
    state.selectedRunId = runId;
    const overlay = $("#detail-overlay");
    overlay.hidden = false;
    $("#detail-content").innerHTML = `<div class="loading-state">Loading run detail&hellip;</div>`;
    try {
      const data = await apiGet(`/api/runs/${encodeURIComponent(runId)}`);
      if (state.selectedRunId !== runId) return; // panel closed/changed while in flight
      renderDetailContent(data);
    } catch (err) {
      $("#detail-content").innerHTML = `<div class="error-banner">Failed to load run detail: ${escapeHtml(
        err.message
      )}</div>`;
    }
  }

  function closeDetail() {
    state.selectedRunId = null;
    $("#detail-overlay").hidden = true;
  }

  // ---------------------------------------------------------------------
  // data loading + polling
  // ---------------------------------------------------------------------

  async function loadRuns({ silent } = {}) {
    try {
      const qs = state.statusFilter ? `?status=${encodeURIComponent(state.statusFilter)}` : "";
      const runs = await apiGet(`/api/runs${qs}`);
      state.runs = Array.isArray(runs) ? runs : [];
      renderRunsTable();
      renderImpactStrip();
    } catch (err) {
      if (!silent) toast(`Could not load runs: ${err.message}`, "error");
      $("#queue-loading").hidden = true;
    }
  }

  async function loadDashboard() {
    try {
      state.dashboard = await apiGet("/api/dashboard");
    } catch (_err) {
      state.dashboard = state.dashboard || {};
    }
    renderImpactStrip();
  }

  async function loadPolicies() {
    try {
      state.policies = await apiGet("/api/policies");
      state.policyRuleMap = (state.policies && state.policies.policy_rules) || {};
    } catch (err) {
      state.policies = state.policies || {};
      toast(`Could not load policy config: ${err.message}`, "error");
    }
    renderRulebook();
    renderImpactStrip();
    renderRunsTable();
  }

  function startPolling() {
    if (state.pollTimer) clearInterval(state.pollTimer);
    state.pollTimer = setInterval(() => {
      loadRuns({ silent: true });
      loadDashboard();
    }, POLL_MS);
  }

  // ---------------------------------------------------------------------
  // actions
  // ---------------------------------------------------------------------

  function setBusy(button, busy, busyLabel) {
    if (!button) return;
    if (busy) {
      button.dataset.originalLabel = button.dataset.originalLabel || button.textContent;
      button.disabled = true;
      button.textContent = busyLabel || "Working…";
    } else {
      button.disabled = false;
      if (button.dataset.originalLabel) button.textContent = button.dataset.originalLabel;
    }
  }

  async function submitHumanAction(runId, outcome) {
    if (!runId) return;
    if (outcome === "reject") {
      const confirmed = window.confirm("Reject this invoice? The vendor will not be paid.");
      if (!confirmed) return;
    }
    const approveBtn = $("#action-approve-btn");
    const rejectBtn = $("#action-reject-btn");
    const errEl = $("#action-error");
    const note = ($("#action-note-input")?.value ?? "").trim() || null;
    if (errEl) errEl.hidden = true;
    const activeBtn = outcome === "approve" ? approveBtn : rejectBtn;
    const otherBtn = outcome === "approve" ? rejectBtn : approveBtn;
    setBusy(activeBtn, true, outcome === "approve" ? "Approving…" : "Rejecting…");
    if (otherBtn) otherBtn.disabled = true;
    try {
      const res = await fetch(`/api/runs/${encodeURIComponent(runId)}/action`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({ outcome, note }),
      });
      let data = null;
      try {
        data = await res.json();
      } catch (_e) {
        data = null;
      }
      if (!res.ok) {
        // FastAPI error bodies are {"detail": "..."} — surface that inline.
        const detail = data?.detail ?? data?.message ?? `HTTP ${res.status}`;
        throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      }
      toast(`Invoice ${outcome === "approve" ? "approved" : "rejected"}.`, "success");
      // Re-render the panel straight from the write's response (it has the
      // same shape as GET /api/runs/{id}), then refresh the queue behind it.
      if (state.selectedRunId === runId) renderDetailContent(data);
      loadRuns({ silent: true });
      loadDashboard();
    } catch (err) {
      if (errEl) {
        errEl.textContent = err?.message ?? String(err);
        errEl.hidden = false;
      }
      setBusy(activeBtn, false);
      if (otherBtn) otherBtn.disabled = false;
    }
  }

  async function handleRunSingle(evt) {
    evt.preventDefault();
    const input = $("#single-path-input");
    const btn = $("#run-single-btn");
    const path = input.value.trim();
    if (!path) {
      toast("Enter a document path first.", "error");
      return;
    }
    setBusy(btn, true, "Running…");
    try {
      const data = await apiPost("/api/runs", { document_path: path });
      toast(`Run finished: ${get(data, "run.status", "done")} — ${get(data, "run.invoice_number", path)}`, "success");
      await Promise.all([loadRuns(), loadDashboard()]);
      const runId = get(data, "run.run_id", null);
      if (runId) openDetail(runId);
    } catch (err) {
      toast(`Run failed: ${err.message}`, "error");
    } finally {
      setBusy(btn, false);
    }
  }

  async function handleRunBatch(evt) {
    evt.preventDefault();
    const dirInput = $("#batch-dir-input");
    const patternInput = $("#batch-pattern-input");
    const btn = $("#run-batch-btn");
    const dir = dirInput.value.trim() || "data/invoices";
    const pattern = patternInput.value.trim() || "*";

    setBusy(btn, true, "Processing batch…");
    toast(`Batch started over ${dir} (${pattern}) — this can take a while.`);
    // Keep the queue polling running while the batch call is in flight so
    // rows fill in progressively as the backend writes them.
    try {
      const results = await apiPost("/api/runs/batch", { dir, pattern });
      const list = Array.isArray(results) ? results : [];
      const counts = {};
      list.forEach((r) => {
        const s = get(r, "run.status", "unknown");
        counts[s] = (counts[s] || 0) + 1;
      });
      const summary = Object.entries(counts)
        .map(([k, v]) => `${v} ${k}`)
        .join(", ");
      toast(`Batch complete: ${list.length} processed (${summary || "no results"}).`, "success");
      await Promise.all([loadRuns(), loadDashboard()]);
    } catch (err) {
      toast(`Batch failed: ${err.message}`, "error");
    } finally {
      setBusy(btn, false);
    }
  }

  async function handleResetDb() {
    const confirmed = window.confirm(
      "Reset the demo database? This clears every run, flag, and trace event. This cannot be undone."
    );
    if (!confirmed) return;
    const btn = $("#reset-db-btn");
    setBusy(btn, true, "Resetting…");
    try {
      await apiPost("/api/reset");
      toast("Demo database reset.", "success");
      closeDetail();
      state.runs = [];
      renderRunsTable();
      await Promise.all([loadRuns(), loadDashboard(), loadPolicies()]);
    } catch (err) {
      toast(`Reset failed: ${err.message}`, "error");
    } finally {
      setBusy(btn, false);
    }
  }

  function updatePollIndicator() {
    const el = $("#poll-indicator");
    const now = new Date();
    el.textContent = `auto-refreshing · ${now.toLocaleTimeString()}`;
  }

  // ---------------------------------------------------------------------
  // wire up
  // ---------------------------------------------------------------------

  function init() {
    $("#run-single-form").addEventListener("submit", handleRunSingle);
    $("#run-batch-form").addEventListener("submit", handleRunBatch);
    $("#reset-db-btn").addEventListener("click", handleResetDb);
    $("#detail-close").addEventListener("click", closeDetail);
    $("#detail-overlay").addEventListener("click", (evt) => {
      if (evt.target === evt.currentTarget) closeDetail();
    });
    document.addEventListener("keydown", (evt) => {
      if (evt.key === "Escape" && !$("#detail-overlay").hidden) closeDetail();
    });
    $("#status-filter").addEventListener("change", (evt) => {
      state.statusFilter = evt.target.value;
      loadRuns();
    });
    $("#refresh-btn").addEventListener("click", () => {
      loadRuns();
      loadDashboard();
    });

    loadPolicies();
    loadDashboard();
    loadRuns().then(() => {
      updatePollIndicator();
    });

    startPolling();
    setInterval(updatePollIndicator, POLL_MS);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();

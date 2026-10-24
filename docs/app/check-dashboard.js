// Presentation only: keep source artifacts and calibration decisions untouched.
"use strict";
(() => {
  const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
  const number = (n) => typeof n === "number" && Number.isFinite(n) ? Number(n.toPrecision(4)).toString() : "—";
  const tones = new Set(["pass", "warn", "fail", "failed", "inconclusive", "skipped", "observed", "running"]);
  const tone = (s) => tones.has(s) ? s : "unknown";
  class LiveRun {
    constructor() { this.reset(); }
    reset() { this.workers = new Map(); }
    consume(event) {
      if (["run_started", "run_finished"].includes(event.event)) return;
      const key = event.worker || 1;
      const w = this.workers.get(key) || { label: "Preparing", stage: "Starting Python", status: "running", points: [], history: [] };
      if (event.event === "pair_started") {
        w.label = event.label; w.stage = "Starting"; w.status = "running";
        w.done = w.total = null; w.points = []; w.history = [];
      } else if (event.event === "stage") {
        w.stage = event.text; w.history.push(event.text); w.history = w.history.slice(-4);
        w.done = w.total = null; w.points = [];
      } else if (event.event === "tick") {
        if (w.done != null && event.done < w.done) w.points = [];
        w.done = event.done; w.total = event.total; w.elapsed = event.elapsed_s;
        if (Number.isFinite(event.done)) w.points.push(event.done);
        w.points = w.points.slice(-80);
      } else if (event.event === "pair_finished") { w.status = event.status; w.stage = "Pair finished";
      } else if (event.event === "pair_failed" || event.event === "pair_skipped") {
        w.label = event.label; w.status = event.event === "pair_failed" ? "failed" : "skipped";
        w.stage = event.error || event.reason;
      }
      this.workers.set(key, w);
    }
    stop(status) {
      for (const w of this.workers.values()) if (w.status === "running") { w.status = status; w.stage = status; }
    }
    html() {
      if (!this.workers.size) return '<div class="activity-card"><p class="muted">Preparing your calibration run…</p></div>';
      return [...this.workers].map(([key, w]) => {
        const max = Math.max(1, ...w.points);
        const points = w.points.map((v, i) => `${10 + i * 280 / Math.max(1, w.points.length - 1)},${85 - 65 * v / max}`).join(" ");
        return `<article class="activity-card"><div class="dashboard-heading"><span class="eyebrow">WORKER ${key}</span><span class="pill ${tone(w.status)}">${esc(w.status)}</span></div>
          <h4>${esc(w.label)}</h4><p class="activity-stage">${esc(w.stage)}</p>
          <div class="activity-count">${number(w.done)} <small>${w.total > 0 ? `/ ${number(w.total)}` : ""} items processed</small></div>
          <svg class="activity-chart" viewBox="0 0 300 100" role="img" aria-label="Recent processed item counts in the current stage">
          <path d="M10 20H290 M10 52H290 M10 85H290" class="chart-grid"/>
          ${w.points.length > 1 ? `<polyline points="${points}"/>` : '<text x="10" y="55">Waiting for processing updates</text>'}</svg>
          <ol class="stage-trail">${w.history.map((s) => `<li>${esc(s)}</li>`).join("")}</ol></article>`;
      }).join("");
    }
  }
  function resultHTML(artifact, mode) {
    const check = mode === "check";
    const cards = (artifact.pairs || []).map((pair) => {
      const axes = [...(pair.axes || []), ...(check ? (pair.unchecked_axes || []).map((a) => ({ ...a, status: "unchecked" })) : [])];
      return `<article class="axis-card"><div class="dashboard-heading"><h4>${esc(pair.pair)}</h4><span class="pill ${tone(pair.status)}">${esc(pair.status)}</span></div>
        <p class="note">${(pair.sensors || pair.frames || []).map(esc).join(" ↔ ")}</p>
        ${axes.length ? axes.map((a) => {
          const measured = check ? a.status !== "unchecked" : a.status === "observed";
          const value = check ? `${number(a.candidate_error)} ${esc(a.unit)} / ${number(a.tolerance)} ${esc(a.unit)}` : `${number(a.value)} ± ${number(a.std)} ${esc(a.unit)}`;
          const ratio = Number.isFinite(a.ratio) ? Math.max(0, a.ratio) : 0;
          return `<div class="axis-row ${measured ? tone(a.status) : "unknown"}"><div class="axis-label"><b>${esc(a.name)}</b><span>${measured ? value : (check ? "UNCHECKED" : "NOT MEASURED")}</span></div>
            ${check && measured ? `<div class="axis-track" role="img" aria-label="${esc(a.name)}: ${number(ratio)} times tolerance, ${esc(a.status)}"><div style="width:${Math.min(100, ratio / 3 * 100)}%"></div><i class="limit-one"></i><i class="limit-two"></i></div>` : ""}
            <small>${measured ? `${esc(a.status)}${check ? ` · ${number(ratio)}× tolerance` : ""}` : esc(a.reason || a.status)}</small></div>`;
        }).join("") : `<p class="muted">${esc(pair.reason || "No axis measurements available.")}</p>`}</article>`;
    });
    const pairs = artifact.pairs || [];
    const active = cards.filter((_, i) => pairs[i].status !== "skipped").join("");
    const skipped = cards.filter((_, i) => pairs[i].status === "skipped");
    return active + (skipped.length ? `<details class="skipped-results"><summary>${skipped.length} skipped pair(s) · show reasons</summary><div class="dashboard-grid">${skipped.join("")}</div></details>` : "");
  }
  globalThis.CalibrexDashboard = { LiveRun, resultHTML };
})();

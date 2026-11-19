const $ = (s) => document.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const state = { branch: "exp", n: "n1000", dev: false };
let M = null, D = null, C = null, K = null;

const WHAT = {
  main: "baseline: v16 manifest (2,356 rows), 60k EMA, w=1.0 — 14/15 @1000/class, finalized student audit (precision 0.965).",
  exp: "+1,089 recovered dog rows (3,445 total), 60k EMA retrained, w=0.75 — 15/15 @1000 and @2000, 13/15 @250; +1 mark over main at every protocol."
};

function headInfo() {
  const h = state.branch === "main" ? C.main_head : C.branch_head;
  $("#head-info").textContent = `${h.hash} · ${h.date} · ${h.subject.slice(0, 70)}`;
  $("#what-changed").textContent = WHAT[state.branch];
}

function metricRows(a, b) {
  const rows = [
    ["accuracy", a.accuracy, b.accuracy, (v) => v.toFixed(4)],
    ["cat acc", a.cat, b.cat, (v) => v.toFixed(4)],
    ["dog acc", a.dog, b.dog, (v) => v.toFixed(4)],
    ["FID", a.fid, b.fid, (v) => v.toFixed(2)],
    ["KID", a.kid, b.kid, (v) => v.toFixed(6)],
    ["recall", a.recall, b.recall, (v) => v.toFixed(4)],
    ["precision", a.precision, b.precision, (v) => v.toFixed(4)],
    ["memorization", a.memorization, b.memorization, (v) => v.toFixed(3)],
    ["marks /15", a.marks, b.marks, (v) => String(v)]
  ];
  return rows.map(([name, va, vb, f]) => {
    const win = name === "marks /15" ? (vb > va ? "win" : "") : "";
    return `<tr><td>${name}</td><td class="num">${f(va)}</td><td class="num ${win}">${f(vb)}</td></tr>`;
  }).join("");
}

function renderMetrics() {
  const a = M.main[state.n], b = M.branch_w075[state.n];
  $("#metric-table").innerHTML =
    `<tr><th>metric</th><th>main 60k w=1.0</th><th>branch 60k w=0.75</th></tr>` + metricRows(a, b);
  $("#delta-banner").textContent =
    `At ${state.n.replace("n", "")}/class: main ${a.marks}/15 vs branch ${b.marks}/15 → +${b.marks - a.marks} mark. ` +
    (state.n === "n250" ? "FID is in the Pass tier for both at this sample size." :
      "Both FID and recall tiers favour the branch here.");
  const e = M.equal_steps_40k;
  $("#equal-table").innerHTML =
    `<tr><th>metric (40k, 1000/class)</th><th>main v16</th><th>branch 3,445</th></tr>` +
    metricRows(e.main, e.branch);
}

function renderData() {
  const f = D;
  $("#flow").innerHTML = [
    ["raw pool", f.pool], ["v16 frozen", f.v16.rows], ["recovery candidates", f.recovery_candidates],
    ["recheck compliant", f.recheck_compliant], ["ruling 1", -f.ruling1_dropped],
    ["ruling 2", -f.ruling2_dropped], ["full scan", -f.ruling3_full_scan_dropped],
    ["added dog", f.added_dog], ["branch", f.branch.rows]
  ].map(([label, v], i) =>
    `<span class="step"><b>${typeof v === "number" && v < 0 ? v : v.toLocaleString()}</b>${label}</span>` +
    (i < 8 ? '<span class="arrow">→</span>' : "")).join("");
  const c = f.coverage_dog;
  $("#cov-table").innerHTML =
    `<tr><th>dog set</th><th>n</th><th>dev modes /24</th><th>participation ratio</th><th>cov@0.9</th></tr>` +
    ["v16", "branch", "v5", "dev"].map((k) => {
      const r = c[k];
      const modes = r["mode_coverage24@0.8"] !== undefined ? (r["mode_coverage24@0.8"] * 24).toFixed(0) + "/24" : "—";
      const cov = r["cov@0.9"] !== undefined ? r["cov@0.9"].toFixed(4) : "—";
      return `<tr><td>${k}${k === "branch" ? " (submitted)" : ""}</td><td class="num">${r.n}</td><td class="num">${modes}</td><td class="num">${r.participation_ratio}</td><td class="num">${cov}</td></tr>`;
    }).join("");
  $("#ruling-note").textContent =
    `Full-scan ruling reason: ${f.ruling_reasons}. Cat side unchanged (${f.coverage_cat_branch.n} rows). ` +
    `Known residual (accepted, quantified by the audit): 6 rows with watermark fragments / human hands / soft heads.`;
}

function renderCommits() {
  const list = state.branch === "main" ? C.main_commits : C.commits;
  $("#commit-title").textContent = state.branch === "main" ? "Main line (recent)" : "exp/dog-diversity commits (main..HEAD)";
  $("#commit-table").innerHTML =
    `<tr><th>hash</th><th>date</th><th>subject</th></tr>` +
    list.map((c) => `<tr><td class="num">${esc(c.hash)}</td><td class="num">${esc(c.date)}</td><td>${esc(c.subject)}</td></tr>`).join("");
  $("#diffstat").textContent = state.branch === "main" ? "" : `main..HEAD total: ${C.diffstat}`;
}

function renderCaptions() {
  const groups = [["main w=1.0", K.main_w1], ["branch w=1.0", K.branch_w1],
    ["branch w=0.75", K.branch_w075], ["dev", K.dev]];
  $("#captions").innerHTML = groups.map(([g, items]) =>
    items.map((t) => `<li><b>${esc(g)}</b> — ${esc(t)}</li>`).join("")).join("") +
    `<li><b>trade-off</b> — ${esc(K.tradeoff)}</li>`;
}

function setBranch(b) {
  state.branch = b;
  document.querySelectorAll(".branch-btn").forEach((x) => x.classList.toggle("active", x.dataset.branch === b));
  headInfo(); renderCommits();
}
function setView(v) {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x.dataset.view === v));
  document.querySelectorAll(".view").forEach((x) => x.classList.toggle("active", x.id === "view-" + v));
}
function setN(n) {
  state.n = n;
  document.querySelectorAll(".proto[data-n]").forEach((x) => x.classList.toggle("active", x.dataset.n === n));
  renderMetrics();
}

Promise.all(["metrics", "datadiff", "commits", "captions"].map((n) =>
  fetch(`assets/${n}.json`).then((r) => r.json())))
  .then(([m, d, c, k]) => {
    M = m; D = d; C = c; K = k;
    headInfo(); renderMetrics(); renderData(); renderCommits(); renderCaptions();
    document.querySelectorAll(".branch-btn").forEach((x) => x.onclick = () => setBranch(x.dataset.branch));
    document.querySelectorAll(".tab").forEach((x) => x.onclick = () => setView(x.dataset.view));
    document.querySelectorAll(".proto[data-n]").forEach((x) => x.onclick = () => setN(x.dataset.n));
    $("#dev-on").onchange = (e) => {
      $("#dev-fig").hidden = !e.target.checked;
      $("#sample-cols").classList.toggle("with-dev", e.target.checked);
    };
  })
  .catch((e) => { document.body.insertAdjacentHTML("afterbegin",
    `<div class="banner warn">asset load failed: ${e}. Serve this folder over http (e.g. python -m http.server).</div>`); });
const DEMO_TICK_MS = 1500;
const CAMERA_INTERVAL_MS = 10000;

const $ = (id) => document.getElementById(id);
let locations = [];
let domains = {};
let domainKey = null;
let locationId = null;
let chart = null;
let demoTimer = null;
let cameraTimer = null;
let cameraStream = null;

async function api(url, options = {}) {
  const res = await fetch(url, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}

function currentLocation() {
  return locations.find((l) => l.id === locationId);
}

function domain() { return domains[domainKey]; }
function plural(word, n) { return n === 1 ? word : `${word}s`; }

/* ---------- domains and locations ---------- */
async function loadLocations() {
  const list = await api("/api/domains");
  domains = Object.fromEntries(list.map((d) => [d.key, d]));
  locations = await api("/api/locations");
  const wanted = Number(new URLSearchParams(window.location.search).get("location"));
  const first = locations.find((l) => l.id === wanted) || locations.find((l) => l.is_demo) || locations[0];
  selectDomain(first.domain, first.id);
}

function selectDomain(key, preferredLocation) {
  stopDemo();
  domainKey = key;
  document.querySelectorAll(".domain-tab").forEach((b) => b.classList.toggle("active", b.dataset.domain === key));
  const here = locations.filter((l) => l.domain === key);
  $("location").innerHTML = here.map((l) => `<option value="${l.id}">${l.name}</option>`).join("");
  locationId = (here.find((l) => l.id === preferredLocation) || here.find((l) => l.is_demo) || here[0]).id;
  $("location").value = locationId;
  applyDomainLabels();
  onLocationChange();
}

function applyDomainLabels() {
  const d = domain();
  document.querySelectorAll(".item").forEach((e) => (e.textContent = d.item));
  document.querySelectorAll(".item-one").forEach((e) => (e.textContent = d.item.replace(/s$/, "")));
  document.querySelectorAll(".server").forEach((e) => (e.textContent = plural(d.server, 2)));
  $("tagline").textContent = d.tagline;
  $("thresholds").textContent =
    `Risk thresholds: MEDIUM at ${d.medium_wait}+ min wait, HIGH at ${d.high_wait}+ min wait ` +
    "or a high predicted probability of critical congestion.";
  $("counters").max = d.max_servers;
  $("service").min = d.service_sec_limits[0];
  $("service").max = d.service_sec_limits[1];
  ["upload-panel", "roi-panel", "camera-block"].forEach((id) => ($(id).hidden = !d.camera));
  renderMetrics();
}

function onLocationChange() {
  stopDemo();
  const loc = currentLocation();
  $("counters").value = loc.counters;
  $("service").value = loc.service_time_sec;
  $("demo-block").hidden = !loc.is_demo;
  roi.frame = null;
  roi.draft = loc.roi;
  $("roi-box").hidden = true;
  $("roi-save").disabled = true;
  showStatus("roi-status", loc.roi ? "A queue area is set for this location." : "Counting the whole frame.");
  showStatus("demo-status", "");
  refresh();
}

function setText(id, text) { $(id).textContent = text; }

/* ---------- rendering ---------- */
const STATUS_TEXT = { SAFE: "SAFE", AT_RISK: "AT RISK", IMMINENT: "COLLAPSE IMMINENT", COLLAPSED: "COLLAPSED" };

function fmtWait(minutes) {
  if (minutes < 1) return `${Math.round(minutes * 60)} s`;
  return minutes >= 120 ? `${(minutes / 60).toFixed(1)} h` : `${minutes} min`;
}

function render(status) {
  const p = status.prediction;
  const d = status.domain;
  const riskCard = $("risk-card");
  riskCard.className = "card risk";
  const advice = $("advice");

  if (!p) {
    ["current", "predicted", "wait", "risk"].forEach((id) => setText(id, "–"));
    setText("asof", "No data yet – start a live source" + (d.camera ? " or upload footage" : ""));
    ["trend", "wait-future", "probability"].forEach((id) => setText(id, ""));
    advice.hidden = true;
    $("radar").hidden = true;
    $("whatif-panel").hidden = true;
  } else {
    setText("current", `${p.current_queue}`);
    setText("asof", `${d.item} · as of ${p.as_of.slice(11)}`);
    setText("horizon", p.horizon_min);
    setText("predicted", `${p.predicted_queue}`);
    const delta = p.predicted_queue - p.current_queue;
    setText("trend", delta === 0 ? "steady" : `${delta > 0 ? "▲ +" : "▼ "}${delta} ${d.item}`);
    setText("wait", fmtWait(p.current_wait_min));
    setText("wait-future", `≈ ${fmtWait(p.predicted_wait_min)} in ${p.horizon_min} min · ${p.counters} ${plural(d.server, p.counters)}`);
    setText("risk", p.risk);
    setText("probability", `${Math.round(p.critical_probability * 100)}% chance of critical congestion`);
    riskCard.classList.add(p.risk);
    advice.hidden = false;
    advice.className = `advice ${p.risk}`;
    advice.textContent = p.recommendation;
    renderRadar(p, d);
  }

  renderChart(status);
  renderAlerts(status.alerts || [], d);
}

function renderRadar(p, d) {
  const r = p.radar;
  $("radar").hidden = false;
  $("radar").className = `radar ${r.status}`;
  setText("radar-status", STATUS_TEXT[r.status]);
  setText("radar-horizon", r.horizon_min);
  const ttc = r.time_to_collapse;
  if (r.status === "COLLAPSED") {
    setText("radar-ttc", "NOW");
    setText("radar-ttc-range", `wait is already above ${d.high_wait} min`);
  } else if (ttc) {
    setText("radar-ttc", `~${ttc.median} min`);
    setText("radar-ttc-range", `80% range: ${ttc.p10}–${ttc.p90} min`);
  } else {
    setText("radar-ttc", "none");
    setText("radar-ttc-range", `no collapse expected in ${r.horizon_min} min`);
  }
  setText("radar-prob", `${Math.round(r.collapse_probability * 100)}%`);
  setText("radar-load", `${Math.round(r.peak_load * 100)}%`);
  if (r.safe_servers == null) {
    setText("radar-safe", `>${r.what_if[r.what_if.length - 1].servers}`);
  } else {
    setText("radar-safe", `${r.safe_servers} ${plural(d.server, r.safe_servers)}`);
  }
  renderWhatIf(r, d);
}

function renderWhatIf(r, d) {
  $("whatif-panel").hidden = false;
  const rows = r.what_if.map((row) => {
    const pct = Math.round(row.collapse_probability * 100);
    const level = pct >= 50 ? "HIGH" : pct >= 10 ? "MEDIUM" : "LOW";
    const tags = [];
    if (row.servers === r.current_servers) tags.push('<span class="tag">now</span>');
    if (row.servers === r.safe_servers) tags.push('<span class="tag safe">safe</span>');
    const apply = row.servers === r.current_servers ? "" :
      `<button class="secondary small-btn" data-servers="${row.servers}">Apply</button>`;
    return `<tr class="${row.servers === r.current_servers ? "current" : ""}">` +
      `<td>${row.servers} ${plural(d.server, row.servers)} ${tags.join(" ")}</td>` +
      `<td><div class="bar"><span class="${level}" style="width:${pct}%"></span></div></td>` +
      `<td>${pct}%</td><td>${fmtWait(row.median_peak_wait)}</td><td>${fmtWait(row.p90_peak_wait)}</td>` +
      `<td>${apply}</td></tr>`;
  });
  $("whatif").innerHTML =
    `<tr><th>Staffing</th><th>Chance of collapse in ${r.horizon_min} min</th><th></th>` +
    "<th>Typical worst wait</th><th>Bad-case worst wait</th><th></th></tr>" + rows.join("");
}

$("whatif").addEventListener("click", (e) => {
  const button = e.target.closest("button[data-servers]");
  if (!button) return;
  $("counters").value = button.dataset.servers;
  $("settings-form").requestSubmit();
});

function renderMetrics() {
  const m = (window.METRICS || {})[domainKey];
  const d = domain();
  if (!m) {
    $("metrics").innerHTML = '<p class="muted">Run <code>python -m queue_predictor.train</code> to see metrics.</p>';
    $("study").innerHTML = "";
    return;
  }
  const rows = Object.entries(m.horizons).map(([h, v]) =>
    `<tr><td>+${h} min</td><td>${v.r2}</td><td>${v.mae}</td><td><strong>${v.busy_mae}</strong></td><td>${v.busy_baseline_mae}</td></tr>`).join("");
  $("metrics").innerHTML =
    `<table class="metrics"><tr><th>Horizon</th><th>R²</th><th>MAE (all day)</th><th>MAE (busy)</th><th>Naive (busy)</th></tr>${rows}</table>` +
    `<p class="muted small">Critical-congestion classifier AUC: ${m.critical.auc} (${m.critical.auc_before_collapse ?? "�"} when the queue is not critical yet, i.e. foreseeing the collapse). MAE = average error in ${d.item}. ` +
    `"Busy" = ${m.busy_definition}. "Naive" assumes the backlog stays the same.</p>`;

  const e = m.early_warning;
  const names = { reactive: "Reactive alarm (today's practice)", classifier: "XGBoost classifier", radar: "Collapse Radar", system: "Full system (classifier + radar)" };
  const pct = (v) => (v == null ? "–" : `${Math.round(v * 100)}%`);
  const order = ["reactive", "classifier", "radar", "system"];
  const body = order.filter((k) => e.policies[k]).map((k) => [k, e.policies[k]]).map(([k, v]) =>
    `<tr class="${k === "system" ? "current" : ""}"><td>${names[k]}</td><td>${pct(v.early_rate)}</td>` +
    `<td><strong>${v.mean_lead_min} min</strong></td><td>${v.false_alarms_per_day}</td></tr>`).join("");
  $("study").innerHTML =
    '<table class="metrics"><tr><th>Alarm</th><th>Collapses warned early</th><th>Mean warning time</th><th>False alarms / day</th></tr>' +
    `${body}</table><p class="muted small">${e.policies.reactive.events} collapses on ${e.days} unseen days. ` +
    `"Early" = at least 2 min before the wait reached ${d.high_wait} min, looking back ${e.window_min} min.</p>`;
}

function renderChart(status) {
  const p = status.prediction;
  const d = status.domain;
  const observed = status.history.map((h) => ({ x: h.t, y: h.count }));
  const forecast = p ? [{ x: 0, y: p.current_queue }, ...p.forecast.map((f) => ({ x: f.minutes, y: f.queue }))] : [];
  const past = status.past_predictions.filter((q) => q.t <= 0).map((q) => ({ x: q.t, y: q.queue }));
  const fan = p ? p.radar.fan : [];
  const maxX = p ? p.radar.horizon_min + 1 : 16;
  const minX = Math.min(-15, ...observed.map((o) => o.x));
  const critical = p ? [{ x: minX, y: p.critical_queue }, { x: maxX, y: p.critical_queue }] : [];

  const datasets = [
    { label: "Observed", data: observed, borderColor: "#2563eb", backgroundColor: "rgba(37,99,235,.12)",
      fill: true, tension: 0.25, pointRadius: observed.length > 40 ? 0 : 3 },
    { label: "Forecast", data: forecast, borderColor: "#7c3aed", borderDash: [6, 4], pointRadius: 4, tension: 0.2 },
    { label: "Best case (10%)", data: fan.map((f) => ({ x: f.t, y: f.p10 })), borderColor: "transparent",
      backgroundColor: "transparent", pointRadius: 0 },
    { label: "Likely range (80%)", data: fan.map((f) => ({ x: f.t, y: f.p90 })), borderColor: "transparent",
      backgroundColor: "rgba(124,58,237,.16)", fill: "-1", pointRadius: 0 },
    { label: `Earlier ${p ? p.horizon_min : 10}-min predictions`, data: past, borderColor: "transparent",
      backgroundColor: "#a78bfa", showLine: false, pointRadius: 3 },
    { label: "Critical level", data: critical, borderColor: "#dc2626", borderWidth: 1.5, borderDash: [3, 3], pointRadius: 0 },
  ];

  if (!chart) {
    chart = new Chart($("chart"), {
      type: "line",
      data: { datasets },
      options: {
        animation: false,
        maintainAspectRatio: false,
        parsing: false,
        interaction: { mode: "nearest", intersect: false },
        scales: {
          x: { type: "linear", max: maxX,
               title: { display: true, text: "minutes (0 = latest reading)" },
               ticks: { callback: (v) => (v === 0 ? "now" : v > 0 ? `+${v}` : v) } },
          y: { beginAtZero: true, title: { display: true, text: `${d.item} waiting` } },
        },
        plugins: { legend: { position: "bottom", labels: { filter: (item) => item.text !== "Best case (10%)" } } },
      },
    });
  } else {
    chart.data.datasets.forEach((ds, i) => {
      ds.data = datasets[i].data;
      ds.pointRadius = datasets[i].pointRadius;
      ds.label = datasets[i].label;
    });
    chart.options.scales.x.max = maxX;
    chart.options.scales.y.title.text = `${d.item} waiting`;
    chart.update();
  }
}

function renderAlerts(alerts, d) {
  const list = $("alerts");
  if (!alerts.length) {
    list.innerHTML = '<li class="muted">No warnings yet.</li>';
    return;
  }
  list.innerHTML = alerts.map((a) =>
    `<li class="${a.risk}"><strong>${a.time} · ${a.risk}</strong> – ${a.queue} now, ` +
    `${a.predicted} expected in ${d.main_horizon} min (${Math.round(a.probability * 100)}% critical risk)</li>`).join("");
}

async function refresh() {
  try {
    render(await api(`/api/status/${locationId}`));
  } catch (err) {
    showStatus("upload-status", err.message, true);
  }
}

/* ---------- uploads ---------- */
function showStatus(id, message, isError = false) {
  const el = $(id);
  el.textContent = message;
  el.classList.toggle("error", isError);
}

async function uploadFile(file, source) {
  const form = new FormData();
  form.append("file", file);
  if (source) form.append("source", source);
  return api(`/api/upload/${locationId}`, { method: "POST", body: form });
}

$("upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = $("file").files[0];
  if (!file) return showStatus("upload-status", "Choose an image or video first.", true);
  $("upload-btn").disabled = true;
  showStatus("upload-status", file.type.startsWith("video")
    ? "Analysing video frames… this can take a while for long clips."
    : "Detecting people… (the first run loads the YOLO model)");
  try {
    const res = await uploadFile(file);
    const extra = res.samples.length > 1
      ? ` in the last of ${res.samples.length} sampled frames (busiest frame shown)` : "";
    showStatus("upload-status", `${res.detector} counted ${res.count} people in the queue${extra}.`);
    showResult(res);
    await refresh();
  } catch (err) {
    showStatus("upload-status", err.message, true);
  } finally {
    $("upload-btn").disabled = false;
  }
});

const drop = $("drop");
$("file").addEventListener("change", () => {
  setText("drop-text", $("file").files[0]?.name || "Click or drop an image / video of the queue");
});
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => {
  if (e.dataTransfer.files.length) {
    $("file").files = e.dataTransfer.files;
    setText("drop-text", e.dataTransfer.files[0].name);
  }
});

/* ---------- simulated live feed ---------- */
async function demoTick() {
  try {
    await api(`/api/demo/${locationId}/tick`, { method: "POST" });
    await refresh();
  } catch (err) {
    stopDemo();
    showStatus("demo-status", err.message, true);
  }
}

function stopDemo() {
  clearInterval(demoTimer);
  demoTimer = null;
  $("demo-start").disabled = false;
  $("demo-stop").disabled = true;
  $("demo-surge").disabled = true;
}

$("demo-start").addEventListener("click", () => {
  $("demo-start").disabled = true;
  $("demo-stop").disabled = false;
  $("demo-surge").disabled = false;
  demoTick();
  demoTimer = setInterval(demoTick, DEMO_TICK_MS);
});
$("demo-stop").addEventListener("click", stopDemo);
$("demo-surge").addEventListener("click", async () => {
  try {
    await api(`/api/demo/${locationId}/surge`, { method: "POST" });
    showStatus("demo-status", "Surge injected: watch the Collapse Radar react within a few ticks.");
  } catch (err) {
    showStatus("demo-status", err.message, true);
  }
});
$("demo-reset").addEventListener("click", async () => {
  stopDemo();
  await api(`/api/demo/${locationId}/reset`, { method: "POST" });
  showStatus("demo-status", "");
  await refresh();
});

/* ---------- webcam ---------- */
async function captureFrame() {
  const video = $("video");
  if (!video.videoWidth) return;
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.85));
  try {
    const res = await uploadFile(new File([blob], "camera.jpg", { type: "image/jpeg" }), "camera");
    showStatus("upload-status", `Camera: ${res.detector} counted ${res.count} people in the queue.`);
    showResult(res);
    await refresh();
  } catch (err) {
    showStatus("upload-status", err.message, true);
  }
}

$("cam-start").addEventListener("click", async () => {
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: true });
  } catch (err) {
    return showStatus("upload-status", `Camera unavailable: ${err.message}`, true);
  }
  $("video").srcObject = cameraStream;
  $("video").hidden = false;
  $("cam-start").disabled = true;
  $("cam-stop").disabled = false;
  setTimeout(captureFrame, 1500);
  cameraTimer = setInterval(captureFrame, CAMERA_INTERVAL_MS);
});

$("cam-stop").addEventListener("click", () => {
  clearInterval(cameraTimer);
  cameraStream?.getTracks().forEach((t) => t.stop());
  $("video").hidden = true;
  $("cam-start").disabled = false;
  $("cam-stop").disabled = true;
});
$("cam-interval").textContent = CAMERA_INTERVAL_MS / 1000;

/* ---------- queue area (ROI) ---------- */
const roi = { frame: null, draft: null, start: null };

function showResult(res) {
  $("result-img").src = `${res.image}?t=${Date.now()}`;
  $("result-img").hidden = false;
  if (res.frame) {
    roi.frame = res.frame;
    $("roi-img").src = `/results/${res.frame}`;
    $("roi-box").hidden = false;
    $("roi-help").textContent = "Drag on the frame to mark where the queue stands, then press Save area.";
  }
}

function drawRoi() {
  const canvas = $("roi-canvas");
  const img = $("roi-img");
  canvas.width = img.clientWidth;
  canvas.height = img.clientHeight;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!roi.draft) return;
  const [x1, y1, x2, y2] = roi.draft;
  const w = canvas.width, h = canvas.height;
  ctx.fillStyle = "rgba(0,0,0,.45)";            // dim everything outside the area
  ctx.fillRect(0, 0, w, h);
  ctx.clearRect(x1 * w, y1 * h, (x2 - x1) * w, (y2 - y1) * h);
  ctx.strokeStyle = "#00aaff";
  ctx.lineWidth = 3;
  ctx.strokeRect(x1 * w, y1 * h, (x2 - x1) * w, (y2 - y1) * h);
}

function pointerFraction(e) {
  const rect = $("roi-canvas").getBoundingClientRect();
  return [
    Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width)),
    Math.min(1, Math.max(0, (e.clientY - rect.top) / rect.height)),
  ];
}

$("roi-img").addEventListener("load", drawRoi);
window.addEventListener("resize", drawRoi);

$("roi-canvas").addEventListener("pointerdown", (e) => {
  roi.start = pointerFraction(e);
  $("roi-canvas").setPointerCapture(e.pointerId);
});
$("roi-canvas").addEventListener("pointermove", (e) => {
  if (!roi.start) return;
  const [x, y] = pointerFraction(e);
  const [sx, sy] = roi.start;
  roi.draft = [Math.min(sx, x), Math.min(sy, y), Math.max(sx, x), Math.max(sy, y)];
  drawRoi();
});
$("roi-canvas").addEventListener("pointerup", () => {
  roi.start = null;
  $("roi-save").disabled = !roi.draft;
});

async function saveRoi(value) {
  try {
    const updated = await api(`/api/locations/${locationId}/roi`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ roi: value }),
    });
    Object.assign(currentLocation(), updated);
    roi.draft = updated.roi;
    drawRoi();
    $("roi-save").disabled = true;
    let message = value ? "Queue area saved." : "Counting the whole frame again.";
    if (roi.frame) {
      const res = await api(`/api/locations/${locationId}/preview`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ frame: roi.frame }),
      });
      $("result-img").src = `${res.image}?t=${Date.now()}`;
      message += ` This frame now counts ${res.count} people. New uploads and camera frames use this area.`;
    }
    showStatus("roi-status", message);
  } catch (err) {
    showStatus("roi-status", err.message, true);
  }
}

$("roi-save").addEventListener("click", () => saveRoi(roi.draft));
$("roi-clear").addEventListener("click", () => saveRoi(null));

/* ---------- settings ---------- */
$("settings-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  try {
    const updated = await api(`/api/locations/${locationId}/settings`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ counters: +$("counters").value, service_time_sec: +$("service").value }),
    });
    Object.assign(currentLocation(), updated);
    showStatus("settings-status", "Saved – forecasts and the what-if table now use the new settings.");
    await refresh();
  } catch (err) {
    showStatus("settings-status", err.message, true);
  }
});

$("location").addEventListener("change", (e) => {
  locationId = +e.target.value;
  onLocationChange();
});
document.querySelectorAll(".domain-tab").forEach((b) =>
  b.addEventListener("click", () => selectDomain(b.dataset.domain)));

loadLocations();

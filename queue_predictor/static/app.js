const DEMO_TICK_MS = 1500;
const CAMERA_INTERVAL_MS = 10000;

const $ = (id) => document.getElementById(id);
let locations = [];
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

async function loadLocations() {
  locations = await api("/api/locations");
  const select = $("location");
  select.innerHTML = locations.map((l) => `<option value="${l.id}">${l.name}</option>`).join("");
  const wanted = Number(new URLSearchParams(window.location.search).get("location"));
  locationId = (locations.find((l) => l.id === wanted) || locations[0]).id;
  select.value = locationId;
  onLocationChange();
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
  refresh();
}

function setText(id, text) { $(id).textContent = text; }

function render(status) {
  const p = status.prediction;
  const riskCard = $("risk-card");
  riskCard.className = "card risk";
  const advice = $("advice");

  if (!p) {
    ["current", "predicted", "wait", "risk"].forEach((id) => setText(id, "–"));
    setText("asof", "No data yet – upload footage or start a live source");
    ["trend", "wait-future", "probability"].forEach((id) => setText(id, ""));
    advice.hidden = true;
  } else {
    setText("current", `${p.current_queue}`);
    setText("asof", `people · as of ${p.as_of.slice(11)}`);
    setText("horizon", p.horizon_min);
    setText("predicted", `${p.predicted_queue}`);
    const delta = p.predicted_queue - p.current_queue;
    setText("trend", delta === 0 ? "steady" : `${delta > 0 ? "▲ +" : "▼ "}${delta} people`);
    setText("wait", `${p.current_wait_min} min`);
    setText("wait-future", `≈ ${p.predicted_wait_min} min in ${p.horizon_min} min · ${p.counters} counter(s)`);
    setText("risk", p.risk);
    setText("probability", `${Math.round(p.critical_probability * 100)}% chance of critical congestion`);
    riskCard.classList.add(p.risk);
    advice.hidden = false;
    advice.className = `advice ${p.risk}`;
    advice.textContent = p.recommendation;
  }

  renderChart(status);
  renderAlerts(status.alerts || []);
}

function renderChart(status) {
  const p = status.prediction;
  const observed = status.history.map((h) => ({ x: h.t, y: h.count }));
  const forecast = p ? [{ x: 0, y: p.current_queue }, ...p.forecast.map((f) => ({ x: f.minutes, y: f.queue }))] : [];
  const past = status.past_predictions.filter((q) => q.t <= 0).map((q) => ({ x: q.t, y: q.queue }));
  const minX = Math.min(-15, ...observed.map((o) => o.x));
  const critical = p ? [{ x: minX, y: p.critical_queue }, { x: 15, y: p.critical_queue }] : [];

  const datasets = [
    { label: "Observed queue", data: observed, borderColor: "#2563eb", backgroundColor: "rgba(37,99,235,.12)",
      fill: true, tension: 0.25, pointRadius: observed.length > 40 ? 0 : 3 },
    { label: "Forecast", data: forecast, borderColor: "#7c3aed", borderDash: [6, 4], pointRadius: 4, tension: 0.2 },
    { label: "Earlier 10-min predictions", data: past, borderColor: "transparent", backgroundColor: "#a78bfa",
      showLine: false, pointRadius: 3 },
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
          x: { type: "linear", max: 16,
               title: { display: true, text: "minutes (0 = latest reading)" },
               ticks: { callback: (v) => (v === 0 ? "now" : v > 0 ? `+${v}` : v) } },
          y: { beginAtZero: true, title: { display: true, text: "people in queue" } },
        },
        plugins: { legend: { position: "bottom" } },
      },
    });
  } else {
    chart.data.datasets.forEach((ds, i) => { ds.data = datasets[i].data; ds.pointRadius = datasets[i].pointRadius; });
    chart.update();
  }
}

function renderAlerts(alerts) {
  const list = $("alerts");
  if (!alerts.length) {
    list.innerHTML = '<li class="muted">No warnings yet.</li>';
    return;
  }
  list.innerHTML = alerts.map((a) =>
    `<li class="${a.risk}"><strong>${a.time} · ${a.risk}</strong> – ${a.queue} now, ` +
    `${a.predicted} expected in 10 min (${Math.round(a.probability * 100)}% critical risk)</li>`).join("");
}

async function refresh() {
  try {
    render(await api(`/api/status/${locationId}`));
  } catch (err) {
    setText("upload-status", err.message);
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
    showStatus("upload-status", err.message, true);
  }
}

function stopDemo() {
  clearInterval(demoTimer);
  demoTimer = null;
  $("demo-start").disabled = false;
  $("demo-stop").disabled = true;
}

$("demo-start").addEventListener("click", () => {
  $("demo-start").disabled = true;
  $("demo-stop").disabled = false;
  demoTick();
  demoTimer = setInterval(demoTick, DEMO_TICK_MS);
});
$("demo-stop").addEventListener("click", stopDemo);
$("demo-reset").addEventListener("click", async () => {
  stopDemo();
  await api(`/api/demo/${locationId}/reset`, { method: "POST" });
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
    showStatus("settings-status", "Saved – forecasts now use the new settings.");
    await refresh();
  } catch (err) {
    showStatus("settings-status", err.message, true);
  }
});

$("location").addEventListener("change", (e) => {
  locationId = +e.target.value;
  onLocationChange();
});

loadLocations();

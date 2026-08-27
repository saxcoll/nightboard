/**
 * Pizza-station camera: sim kitchen-cam by default, optional getUserMedia + naive hue vision.
 * Parent mounts via mountPizza(rootEl, pizzaConfig). Root class: pizza-root.
 */

const CSS_HREF = new URL("../css/pizza.css", import.meta.url).href;
const VISION_MS = 800;
const SAMPLE_W = 160;
const SAMPLE_H = 90;
const SAMPLE_STEP = 2;
const MIN_SAT = 18;
const MIN_LIGHT = 10;
const MAX_LIGHT = 88;
const FIRST_FRAME_HITS = 28;

/** Hawaiian empties first; others linger so the board stays interesting. */
const DECAY_MS = {
  hawaiian: 8000,
  veggie: 17000,
  pepperoni: 24000,
  cheese: 32000,
};

/** Demo restock: 4s delay, labeled as 4 min. Refill toward max, +8 per order. */
const ORDER_MS = 4000;
const ORDER_ETA = "4 min";
const ORDER_BATCH = 8;

const mounts = new WeakMap();

function ensureStyles() {
  if (document.querySelector("link[data-pizza-css]")) return;
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = CSS_HREF;
  link.dataset.pizzaCss = "1";
  document.head.appendChild(link);
}

function rgbToHsl(r, g, b) {
  r /= 255;
  g /= 255;
  b /= 255;
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const l = (max + min) / 2;
  if (max === min) return { h: 0, s: 0, l: l * 100 };
  const d = max - min;
  const s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
  let h;
  if (max === r) h = (g - b) / d + (g < b ? 6 : 0);
  else if (max === g) h = (b - r) / d + 2;
  else h = (r - g) / d + 4;
  return { h: h * 60, s: s * 100, l: l * 100 };
}

function hueInRange(h, range) {
  const [a, b] = range;
  if (a <= b) return h >= a && h <= b;
  return h >= a || h <= b;
}

function hueMid(range) {
  const [a, b] = range;
  if (a <= b) return (a + b) / 2;
  const span = 360 - a + b;
  return (a + span / 2) % 360;
}

function hueDist(h, mid) {
  const d = Math.abs(h - mid);
  return Math.min(d, 360 - d);
}

function clamp(n, lo, hi) {
  return Math.max(lo, Math.min(hi, n));
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

function cameraReason(err) {
  const name = err && err.name;
  if (name === "NotAllowedError" || name === "PermissionDeniedError") {
    return "Camera permission denied — staying in sim.";
  }
  if (name === "NotFoundError" || name === "DevicesNotFoundError") {
    return "No camera found — staying in sim.";
  }
  if (name === "NotReadableError" || name === "TrackStartError") {
    return "Camera is in use — staying in sim.";
  }
  if (name === "SecurityError") {
    return "Camera blocked on this origin — staying in sim.";
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    return "No camera API — staying in sim.";
  }
  return "Camera unavailable — staying in sim.";
}

function botLine(trays, lowAt, pending) {
  const alerts = [];
  for (const t of trays) {
    const inbound = Boolean(pending && pending[t.id]);
    if (t.slices === 0) {
      alerts.push(inbound ? `${t.name} is out. More is on the way.` : `${t.name} is out.`);
    } else if (t.slices <= lowAt) {
      alerts.push(`${t.name} is down to ${t.slices}.`);
    }
  }
  if (alerts.length === 0) {
    const best = trays.reduce((a, b) => (b.slices / b.max >= a.slices / a.max ? b : a));
    return `All trays holding. ${best.name} is fine.`;
  }
  const ok = trays.find((t) => t.slices > lowAt);
  if (ok) alerts.push(`${ok.name} is fine.`);
  return alerts.join(" ");
}

export function mountPizza(rootEl, pizzaConfig) {
  if (!rootEl) throw new Error("mountPizza: rootEl is required");
  if (!pizzaConfig || !Array.isArray(pizzaConfig.trays)) {
    throw new Error("mountPizza: pizzaConfig.trays is required");
  }

  const prev = mounts.get(rootEl);
  if (prev) prev.destroy();

  ensureStyles();

  const lowAt = pizzaConfig.lowAt;
  const trays = pizzaConfig.trays.map((t) => ({
    id: t.id,
    name: t.name,
    hue: Array.isArray(t.hue) ? t.hue.slice() : [0, 0],
    color: t.color,
    slices: t.slices,
    max: t.max,
  }));

  const state = {
    mode: "sim",
    source: "sim",
    reason: "",
    stream: null,
    manualHold: false,
    baseline: Object.fromEntries(trays.map((t) => [t.id, 0])),
    calibrated: false,
    confidence: 0,
    lastMatches: Object.fromEntries(trays.map((t) => [t.id, 0])),
    lastPoints: [],
    decayAt: Object.fromEntries(trays.map((t) => [t.id, performance.now()])),
    pending: {},
    visionTimer: 0,
    decayTimer: 0,
    overlayRaf: 0,
    destroyed: false,
  };

  rootEl.classList.add("pizza-root");
  rootEl.replaceChildren();

  const head = el("header", "pizza-head");
  const title = el("h2", "pizza-title", "Pizza cam");
  const badge = el("span", "pizza-badge pizza-badge-sim", "SIM");
  badge.setAttribute("data-mode", "sim");
  const reasonEl = el("p", "pizza-reason");
  reasonEl.hidden = true;
  const armBtn = el("button", "pizza-btn pizza-btn-arm", "Arm camera");
  armBtn.type = "button";
  const calBtn = el("button", "pizza-btn pizza-btn-cal", "Calibrate");
  calBtn.type = "button";
  calBtn.setAttribute("aria-label", "Calibrate full trays");
  const restockBtn = el("button", "pizza-btn pizza-btn-restock", "Restock kitchen");
  restockBtn.type = "button";
  restockBtn.setAttribute("aria-label", "Restock every empty or low tray");
  const actions = el("div", "pizza-actions");
  actions.append(armBtn, calBtn, restockBtn);
  head.append(title, badge, actions);

  const viewport = el("div", "pizza-viewport");
  const sim = el("div", "pizza-sim");
  sim.setAttribute("aria-hidden", "true");
  const simStage = el("div", "pizza-sim-stage");
  const simMeta = el("div", "pizza-sim-meta");
  const rec = el("span", "pizza-rec", "REC");
  const camClock = el("span", "pizza-cam-clock");
  simMeta.append(rec, el("span", "pizza-cam-label", "KITCHEN · TRAY CAM"), camClock);
  const scan = el("div", "pizza-scan");
  sim.append(simStage, simMeta, scan);

  const pies = new Map();
  for (const t of trays) {
    const pie = el("div", "pizza-sim-pie");
    pie.dataset.id = t.id;
    pie.style.setProperty("--tray", t.color);
    simStage.append(pie);
    pies.set(t.id, pie);
  }

  const video = el("video", "pizza-video");
  video.setAttribute("playsinline", "");
  video.setAttribute("muted", "");
  video.muted = true;
  video.autoplay = true;
  video.playsInline = true;

  const overlay = el("canvas", "pizza-overlay");
  overlay.setAttribute("aria-hidden", "true");

  const hud = el("div", "pizza-hud");
  const hudBoxes = new Map();
  for (const t of trays) {
    const box = el("div", "pizza-hud-box");
    box.dataset.id = t.id;
    box.style.setProperty("--tray", t.color);
    const name = el("span", "pizza-hud-name", t.name);
    const count = el("span", "pizza-hud-count");
    box.append(name, count);
    hud.append(box);
    hudBoxes.set(t.id, { box, count });
  }

  const visionTag = el("div", "pizza-vision-tag", "vision estimate");
  visionTag.hidden = true;

  viewport.append(sim, video, overlay, hud, visionTag);

  const trayRow = el("div", "pizza-trays");
  const trayUi = new Map();
  for (const t of trays) {
    const card = el("article", "pizza-tray");
    card.dataset.id = t.id;
    const chip = el("span", "pizza-chip");
    chip.style.background = t.color;
    chip.setAttribute("aria-hidden", "true");
    const meta = el("div", "pizza-tray-meta");
    const nameRow = el("div", "pizza-tray-name");
    nameRow.append(el("strong", "", t.name));
    const lowMark = el("span", "pizza-low", "LOW");
    lowMark.hidden = true;
    nameRow.append(lowMark);
    const nums = el("div", "pizza-tray-count");
    const bar = el("div", "pizza-bar");
    const fill = el("i", "pizza-bar-fill");
    fill.style.background = t.color;
    bar.append(fill);
    bar.setAttribute("role", "progressbar");
    bar.setAttribute("aria-label", `${t.name} remaining`);
    meta.append(nameRow, nums, bar);

    const stepper = el("div", "pizza-stepper");
    const minus = el("button", "pizza-step", "−");
    minus.type = "button";
    minus.setAttribute("aria-label", `Remove one ${t.name} slice`);
    const plus = el("button", "pizza-step", "+");
    plus.type = "button";
    plus.setAttribute("aria-label", `Add one ${t.name} slice`);
    stepper.append(minus, plus);

    minus.addEventListener("click", () => bump(t.id, -1));
    plus.addEventListener("click", () => bump(t.id, 1));

    const orderBtn = el("button", "pizza-order", "Order more");
    orderBtn.type = "button";
    orderBtn.setAttribute("aria-label", `Order more ${t.name}`);
    orderBtn.hidden = true;
    orderBtn.addEventListener("click", () => orderTray(t.id));
    const orderEta = el("span", "pizza-order-eta");
    orderEta.hidden = true;
    meta.append(orderBtn, orderEta);

    card.append(chip, meta, stepper);
    trayRow.append(card);
    trayUi.set(t.id, { card, nums, fill, bar, lowMark, minus, plus, orderBtn, orderEta });
  }

  const kitchenStatus = el("p", "pizza-kitchen-status");
  kitchenStatus.setAttribute("aria-live", "polite");
  kitchenStatus.hidden = true;

  const voice = el("p", "pizza-voice");
  voice.setAttribute("aria-live", "polite");

  rootEl.append(head, reasonEl, viewport, trayRow, kitchenStatus, voice);

  function snapshot() {
    return trays.map((t) => ({
      id: t.id,
      name: t.name,
      color: t.color,
      slices: t.slices,
      max: t.max,
    }));
  }

  function emit(source) {
    state.source = source;
    const traysSnap = snapshot();
    const low = traysSnap.filter((t) => t.slices <= lowAt);
    window.dispatchEvent(
      new CustomEvent("nightboard:pizza", {
        detail: { trays: traysSnap, low, source },
      }),
    );
  }

  function paint() {
    const live = state.mode === "live";
    rootEl.classList.toggle("pizza-mode-live", live);
    rootEl.classList.toggle("pizza-mode-sim", !live);
    badge.textContent = live ? "LIVE" : "SIM";
    badge.classList.toggle("pizza-badge-live", live);
    badge.classList.toggle("pizza-badge-sim", !live);
    badge.dataset.mode = live ? "live" : "sim";
    armBtn.disabled = live;
    armBtn.textContent = live ? "Camera on" : "Arm camera";

    if (state.reason) {
      reasonEl.textContent = state.reason;
      reasonEl.hidden = false;
    } else {
      reasonEl.textContent = "";
      reasonEl.hidden = true;
    }

    if (live) {
      visionTag.hidden = false;
      visionTag.textContent = `vision estimate · ${state.confidence}%`;
    } else {
      visionTag.hidden = true;
    }

    camClock.textContent = new Intl.DateTimeFormat("en-CA", {
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    }).format(new Date());

    for (const t of trays) {
      const low = t.slices <= lowAt;
      const pct = t.max ? Math.round((t.slices / t.max) * 100) : 0;
      const hudBox = hudBoxes.get(t.id);
      hudBox.count.textContent = `${t.slices}/${t.max}`;
      hudBox.box.classList.toggle("pizza-hud-low", low);

      const pie = pies.get(t.id);
      pie.style.setProperty("--pct", `${pct}%`);
      pie.classList.toggle("pizza-pie-low", low);
      pie.classList.toggle("pizza-pie-empty", t.slices === 0);

      const ui = trayUi.get(t.id);
      ui.card.classList.toggle("pizza-tray-low", low);
      ui.nums.textContent = `${t.slices} / ${t.max}`;
      ui.fill.style.width = `${pct}%`;
      ui.bar.setAttribute("aria-valuenow", String(t.slices));
      ui.bar.setAttribute("aria-valuemin", "0");
      ui.bar.setAttribute("aria-valuemax", String(t.max));
      ui.lowMark.hidden = !low;
      ui.minus.disabled = t.slices <= 0;
      ui.plus.disabled = t.slices >= t.max;

      const inbound = Boolean(state.pending[t.id]);
      const needsOrder = low || inbound;
      ui.orderBtn.hidden = !needsOrder;
      ui.orderBtn.disabled = inbound;
      ui.card.classList.toggle("pizza-tray-ordering", inbound);
      if (inbound) {
        ui.orderEta.hidden = false;
        ui.orderEta.textContent = `${t.name} ordered · ${ORDER_ETA}`;
      } else {
        ui.orderEta.hidden = true;
        ui.orderEta.textContent = "";
      }
    }

    const pendingNames = trays.filter((t) => state.pending[t.id]).map((t) => t.name);
    if (pendingNames.length) {
      kitchenStatus.hidden = false;
      kitchenStatus.textContent =
        pendingNames.length === 1
          ? `${pendingNames[0]} ordered · ${ORDER_ETA}`
          : `${pendingNames.join(", ")} ordered · ${ORDER_ETA}`;
    } else {
      kitchenStatus.hidden = true;
      kitchenStatus.textContent = "";
    }

    const canRestock = trays.some((t) => t.slices <= lowAt && !state.pending[t.id]);
    restockBtn.disabled = !canRestock;

    voice.textContent = botLine(trays, lowAt, state.pending);
  }

  function bump(id, delta) {
    const t = trays.find((x) => x.id === id);
    if (!t) return;
    const next = clamp(t.slices + delta, 0, t.max);
    if (next === t.slices) return;
    t.slices = next;
    state.manualHold = true;
    paint();
    emit("manual");
  }

  function cancelPending(id) {
    if (!state.pending[id]) return;
    clearTimeout(state.pending[id]);
    delete state.pending[id];
  }

  function cancelAllPending() {
    for (const id of Object.keys(state.pending)) cancelPending(id);
  }

  function fulfill(id) {
    if (state.destroyed) return;
    delete state.pending[id];
    const t = trays.find((x) => x.id === id);
    if (!t) return;
    t.slices = clamp(t.slices + ORDER_BATCH, 0, t.max);
    state.decayAt[id] = performance.now();
    state.manualHold = true;
    paint();
    emit("manual");
  }

  function orderTray(id) {
    const t = trays.find((x) => x.id === id);
    if (!t || state.pending[id] || t.slices > lowAt) return;
    state.pending[id] = window.setTimeout(() => fulfill(id), ORDER_MS);
    paint();
  }

  function restockKitchen() {
    let started = false;
    for (const t of trays) {
      if (t.slices > lowAt || state.pending[t.id]) continue;
      state.pending[t.id] = window.setTimeout(() => fulfill(t.id), ORDER_MS);
      started = true;
    }
    if (started) paint();
  }

  function decayTick() {
    if (state.destroyed || state.mode !== "sim") return;
    const now = performance.now();
    let changed = false;
    for (const t of trays) {
      const every = DECAY_MS[t.id] || 20000;
      if (t.slices <= 0) continue;
      if (now - state.decayAt[t.id] < every) continue;
      t.slices -= 1;
      state.decayAt[t.id] = now;
      changed = true;
    }
    if (changed) {
      paint();
      emit("sim");
    } else {
      camClock.textContent = new Intl.DateTimeFormat("en-CA", {
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        hour12: false,
      }).format(new Date());
    }
  }

  function matchPixel(h, s, l) {
    if (s < MIN_SAT || l < MIN_LIGHT || l > MAX_LIGHT) return null;
    let best = null;
    let bestD = Infinity;
    for (const t of trays) {
      if (!hueInRange(h, t.hue)) continue;
      const d = hueDist(h, hueMid(t.hue));
      if (d < bestD) {
        bestD = d;
        best = t.id;
      }
    }
    return best;
  }

  const analysis = document.createElement("canvas");
  analysis.width = SAMPLE_W;
  analysis.height = SAMPLE_H;
  const actx = analysis.getContext("2d", { willReadFrequently: true });

  function sampleFrame() {
    if (state.destroyed || state.mode !== "live") return;
    if (video.readyState < 2 || video.videoWidth === 0) return;

    actx.drawImage(video, 0, 0, SAMPLE_W, SAMPLE_H);
    const img = actx.getImageData(0, 0, SAMPLE_W, SAMPLE_H);
    const data = img.data;
    const matches = Object.fromEntries(trays.map((t) => [t.id, 0]));
    const points = [];
    let gated = 0;

    for (let y = 0; y < SAMPLE_H; y += SAMPLE_STEP) {
      for (let x = 0; x < SAMPLE_W; x += SAMPLE_STEP) {
        const i = (y * SAMPLE_W + x) * 4;
        const { h, s, l } = rgbToHsl(data[i], data[i + 1], data[i + 2]);
        if (s < MIN_SAT || l < MIN_LIGHT || l > MAX_LIGHT) continue;
        gated += 1;
        const id = matchPixel(h, s, l);
        if (id) {
          matches[id] += 1;
          if (points.length < 220) points.push({ x, y, id });
        }
      }
    }

    state.lastMatches = matches;
    state.lastPoints = points;

    const classified = trays.reduce((n, t) => n + matches[t.id], 0);
    const ratio = gated ? classified / gated : 0;
    state.confidence = gated === 0 ? 0 : Math.round(clamp(28 + ratio * 67, 0, 95));

    if (!state.calibrated && classified >= FIRST_FRAME_HITS) {
      for (const t of trays) {
        if (t.slices > 0 && matches[t.id] > 0) {
          state.baseline[t.id] = matches[t.id] * (t.max / t.slices);
        }
      }
      state.calibrated = trays.some((t) => state.baseline[t.id] > 0);
    }

    if (!state.manualHold && state.calibrated) {
      let changed = false;
      for (const t of trays) {
        const base = state.baseline[t.id];
        if (!(base > 0)) continue;
        const estimate = clamp(Math.round((matches[t.id] / base) * t.max), 0, t.max);
        if (estimate !== t.slices) {
          t.slices = estimate;
          changed = true;
        }
      }
      paint();
      if (changed) emit("vision");
    } else {
      paint();
    }
  }

  function syncOverlay() {
    const r = viewport.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(r.width * dpr));
    const h = Math.max(1, Math.floor(r.height * dpr));
    if (overlay.width !== w || overlay.height !== h) {
      overlay.width = w;
      overlay.height = h;
    }
  }

  function drawOverlay() {
    if (state.destroyed) return;
    syncOverlay();
    const ctx = overlay.getContext("2d");
    const w = overlay.width;
    const h = overlay.height;
    ctx.clearRect(0, 0, w, h);

    if (state.mode === "live" && state.lastPoints.length) {
      const sx = w / SAMPLE_W;
      const sy = h / SAMPLE_H;
      for (const p of state.lastPoints) {
        const tray = trays.find((t) => t.id === p.id);
        ctx.fillStyle = tray ? tray.color : "#efe6d4";
        ctx.globalAlpha = 0.55;
        ctx.fillRect(p.x * sx, p.y * sy, 2.2 * sx, 2.2 * sy);
      }
      ctx.globalAlpha = 1;
    }

    state.overlayRaf = requestAnimationFrame(drawOverlay);
  }

  function calibrate() {
    if (state.mode === "live") {
      state.manualHold = true;
      sampleFrame();
      for (const t of trays) {
        const m = state.lastMatches[t.id] || 0;
        if (m > 0) state.baseline[t.id] = m;
        t.slices = t.max;
      }
      state.calibrated = trays.some((t) => state.baseline[t.id] > 0);
      state.manualHold = false;
      cancelAllPending();
      paint();
      emit("vision");
      return;
    }
    cancelAllPending();
    for (const t of trays) t.slices = t.max;
    for (const t of trays) state.decayAt[t.id] = performance.now();
    state.manualHold = false;
    paint();
    emit("sim");
  }

  async function armCamera() {
    if (state.mode === "live") return;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      state.reason = cameraReason({ name: "NoAPI" });
      paint();
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (state.destroyed) {
        stream.getTracks().forEach((tr) => tr.stop());
        return;
      }
      state.stream = stream;
      video.srcObject = stream;
      await video.play();
      state.mode = "live";
      state.reason = "";
      state.calibrated = false;
      clearInterval(state.decayTimer);
      state.visionTimer = window.setInterval(sampleFrame, VISION_MS);
      paint();
    } catch (err) {
      state.reason = cameraReason(err);
      state.mode = "sim";
      paint();
    }
  }

  function destroy() {
    state.destroyed = true;
    cancelAllPending();
    clearInterval(state.decayTimer);
    clearInterval(state.visionTimer);
    cancelAnimationFrame(state.overlayRaf);
    resizeObs.disconnect();
    if (state.stream) {
      state.stream.getTracks().forEach((tr) => tr.stop());
      state.stream = null;
    }
    video.srcObject = null;
    mounts.delete(rootEl);
  }

  armBtn.addEventListener("click", () => {
    armCamera();
  });
  calBtn.addEventListener("click", () => {
    calibrate();
  });
  restockBtn.addEventListener("click", () => {
    restockKitchen();
  });

  const resizeObs = new ResizeObserver(() => syncOverlay());
  resizeObs.observe(viewport);

  state.decayTimer = window.setInterval(decayTick, 1000);
  drawOverlay();
  paint();
  emit("sim");

  const api = { destroy };
  mounts.set(rootEl, api);
  return api;
}

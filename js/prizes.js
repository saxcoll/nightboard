/**
 * Organizer prize payouts — Interac e-Transfer *deposit requests* to winners.
 * Persists to localStorage (`nightboard-payouts`). Does not move money,
 * call a bank, or collect account / SIN / card numbers.
 *
 * Events: nightboard:payouts after send / status / confirm.
 */

export const STORAGE_KEY = "nightboard-payouts";

const DEFAULT_SLOTS = [
  { id: "first", label: "1st", amount: 1500 },
  { id: "second", label: "2nd", amount: 1000 },
  { id: "third", label: "3rd", amount: 500 },
  { id: "audience", label: "Audience", amount: 250 },
];

const DEMO_WINNERS = {
  first: "spaceflex",
  second: "transit-x",
  third: "promptforge",
  audience: "cursor-barely",
};

const METHOD = "Interac e-Transfer deposit request";
const CURRENCY = "CAD";

let ctx = {
  teams: [],
  youAre: "",
  slots: [],
  requests: [],
};

const timers = new Map();

function ensureStyles() {
  if (document.querySelector("link[data-prizes-css]")) return;
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = new URL("../css/prizes.css", import.meta.url).href;
  link.dataset.prizesCss = "1";
  document.head.appendChild(link);
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

function stopCardClick(node) {
  node.addEventListener("click", (ev) => ev.stopPropagation());
}

function uid(prefix) {
  return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`;
}

function loadRaw() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return null;
    return parsed;
  } catch {
    return null;
  }
}

function save() {
  const payload = {
    v: 1,
    slots: ctx.slots.map((s) => ({
      id: s.id,
      label: s.label,
      amount: Number(s.amount) || 0,
      teamId: s.teamId || "",
    })),
    requests: ctx.requests,
  };
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
  } catch {
    /* ignore quota */
  }
}

function emit() {
  window.dispatchEvent(new CustomEvent("nightboard:payouts", { detail: { requests: ctx.requests } }));
}

function teamById(id) {
  return ctx.teams.find((t) => t.id === id) || null;
}

function memberName(m) {
  return typeof m === "string" ? m : m?.name || "";
}

function memberRole(m) {
  return typeof m === "string" ? "member" : m?.role || "member";
}

function leaderOf(team) {
  if (!team?.members?.length) return { name: team?.name || "Team", role: "leader" };
  const star = team.members.find((m) => memberRole(m) === "leader");
  return star || team.members[0];
}

/** Invent a plausible Interac inbox from a display name. Shown in the UI. */
export function emailFromName(name, teamId) {
  const base = String(name || teamId || "leader")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ".")
    .replace(/^\.+|\.+$/g, "")
    .slice(0, 48);
  return `${base || "leader"}@luma.local`;
}

function leaderEmail(team) {
  const leader = leaderOf(team);
  return emailFromName(memberName(leader), team?.id);
}

function fmtCAD(n) {
  const v = Number(n) || 0;
  return `CAD $${v.toLocaleString("en-CA", { maximumFractionDigits: 0 })}`;
}

function defaultSlots() {
  return DEFAULT_SLOTS.map((s) => ({ ...s, teamId: DEMO_WINNERS[s.id] || "" }));
}

function latestForSlot(slotId) {
  for (let i = ctx.requests.length - 1; i >= 0; i--) {
    if (ctx.requests[i].slotId === slotId) return ctx.requests[i];
  }
  return null;
}

function latestForTeam(teamId) {
  for (let i = ctx.requests.length - 1; i >= 0; i--) {
    const req = ctx.requests[i];
    if (req.teamId === teamId && req.status !== "cancelled") return req;
  }
  return null;
}

function statusLabel(status) {
  if (status === "queued") return "Queued";
  if (status === "sent") return "Sent";
  if (status === "awaiting") return "Awaiting winner confirm";
  if (status === "confirmed") return "Winner confirmed email";
  return status || "";
}

function clearTimer(id) {
  const t = timers.get(id);
  if (t) {
    clearTimeout(t);
    timers.delete(id);
  }
}

function findReq(id) {
  return ctx.requests.find((r) => r.id === id) || null;
}

function advanceSent(id) {
  const req = findReq(id);
  if (!req || req.status !== "queued") return;
  req.status = "sent";
  req.sentAt = new Date().toISOString();
  save();
  emit();
  clearTimer(id);
  const wait = window.setTimeout(() => advanceAwaiting(id), 400);
  timers.set(id, wait);
}

function advanceAwaiting(id) {
  const req = findReq(id);
  if (!req || req.status !== "sent") return;
  req.status = "awaiting";
  save();
  emit();
  clearTimer(id);
}

function scheduleAdvances(req) {
  if (!req) return;
  const created = Date.parse(req.createdAt) || Date.now();
  const age = Date.now() - created;
  if (req.status === "queued") {
    const delay = Math.max(0, 1000 - age);
    clearTimer(req.id);
    const t = window.setTimeout(() => advanceSent(req.id), delay);
    timers.set(req.id, t);
  } else if (req.status === "sent") {
    clearTimer(req.id);
    const t = window.setTimeout(() => advanceAwaiting(req.id), 400);
    timers.set(req.id, t);
  }
}

function catchUp() {
  const now = Date.now();
  for (const req of ctx.requests) {
    if (req.status !== "queued" && req.status !== "sent") continue;
    const created = Date.parse(req.createdAt) || now;
    const age = now - created;
    if (req.status === "queued" && age >= 1400) {
      req.status = "awaiting";
      req.sentAt = req.sentAt || new Date(created + 1000).toISOString();
    } else if (req.status === "queued" && age >= 1000) {
      req.status = "sent";
      req.sentAt = req.sentAt || new Date(created + 1000).toISOString();
      scheduleAdvances(req);
    } else if (req.status === "queued") {
      scheduleAdvances(req);
    } else if (req.status === "sent") {
      if (age >= 1400) req.status = "awaiting";
      else scheduleAdvances(req);
    }
  }
}

function buildRequest(slot) {
  const team = teamById(slot.teamId);
  if (!team) return null;
  const leader = leaderOf(team);
  const leaderName = memberName(leader);
  return {
    id: uid("payout"),
    slotId: slot.id,
    slotLabel: slot.label,
    teamId: team.id,
    teamName: team.name,
    leaderName,
    leaderEmail: emailFromName(leaderName, team.id),
    amount: Number(slot.amount) || 0,
    currency: CURRENCY,
    method: METHOD,
    status: "queued",
    createdAt: new Date().toISOString(),
    sentAt: null,
    confirmedAt: null,
    confirmedEmail: null,
    note: "Organizer prize payout. Winner replies with the email to receive the Interac e-Transfer. No bank login.",
  };
}

/**
 * Queue an Interac deposit request for a prize slot (does not move money).
 * @param {string} slotId
 * @returns {object|null}
 */
export function sendDepositRequest(slotId, opts = {}) {
  const slot = ctx.slots.find((s) => s.id === slotId);
  if (!slot?.teamId) return null;
  const existing = latestForSlot(slotId);
  if (existing && existing.status !== "confirmed") return existing;
  const req = buildRequest(slot);
  if (!req) return null;
  ctx.requests.push(req);
  save();
  if (!opts.silent) emit();
  scheduleAdvances(req);
  return req;
}

/** Queue deposit requests for every slot that has a winner and no open request. */
export function sendAllDepositRequests() {
  const out = [];
  for (const slot of ctx.slots) {
    if (!slot.teamId) continue;
    const existing = latestForSlot(slot.id);
    if (existing && existing.status !== "confirmed") continue;
    const req = sendDepositRequest(slot.id, { silent: true });
    if (req) out.push(req);
  }
  if (out.length) emit();
  return out;
}

export function confirmPayout(requestId, email) {
  const req = findReq(requestId);
  if (!req) return null;
  if (req.status !== "sent" && req.status !== "awaiting") return req;
  const next = String(email || req.leaderEmail || "").trim();
  if (!next || !next.includes("@")) return req;
  req.status = "confirmed";
  req.confirmedAt = new Date().toISOString();
  req.confirmedEmail = next;
  clearTimer(req.id);
  save();
  emit();
  return req;
}

export function seedDemoWinners() {
  for (const slot of ctx.slots) {
    if (DEMO_WINNERS[slot.id]) slot.teamId = DEMO_WINNERS[slot.id];
  }
  save();
  emit();
}

function csvEscape(v) {
  const s = v == null ? "" : String(v);
  if (/[",\n]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
  return s;
}

function exportRows() {
  return ctx.requests.map((r) => ({
    slot: r.slotLabel,
    team: r.teamName,
    leader: r.leaderName,
    email: r.confirmedEmail || r.leaderEmail,
    amount: r.amount,
    currency: r.currency,
    method: r.method,
    status: r.status,
    createdAt: r.createdAt,
    sentAt: r.sentAt || "",
    confirmedAt: r.confirmedAt || "",
  }));
}

function download(filename, mime, text) {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.rel = "noopener";
  document.body.append(a);
  a.click();
  a.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 500);
}

export function exportPayoutsJSON() {
  const payload = {
    event: "Cursor Calgary Meetup — prize payouts",
    disclaimer:
      "Organizer Interac e-Transfer deposit requests. Nightboard does not log in to a bank or move money.",
    exportedAt: new Date().toISOString(),
    requests: ctx.requests,
  };
  download("nightboard-payouts.json", "application/json", JSON.stringify(payload, null, 2));
}

export function exportPayoutsCSV() {
  const rows = exportRows();
  const headers = [
    "slot",
    "team",
    "leader",
    "email",
    "amount",
    "currency",
    "method",
    "status",
    "createdAt",
    "sentAt",
    "confirmedAt",
  ];
  const lines = [headers.join(",")];
  for (const row of rows) {
    lines.push(headers.map((h) => csvEscape(row[h])).join(","));
  }
  download("nightboard-payouts.csv", "text/csv", lines.join("\n"));
}

function teamSelect(slot) {
  const sel = document.createElement("select");
  sel.className = "prize-select";
  sel.setAttribute("aria-label", `Winning team for ${slot.label}`);
  const blank = document.createElement("option");
  blank.value = "";
  blank.textContent = "Pick a team";
  sel.append(blank);
  for (const team of ctx.teams) {
    const opt = document.createElement("option");
    opt.value = team.id;
    opt.textContent = team.name;
    if (team.id === slot.teamId) opt.selected = true;
    sel.append(opt);
  }
  sel.addEventListener("change", () => {
    slot.teamId = sel.value;
    save();
    paintPrizePanel.lastRoot && paintPrizePanel(paintPrizePanel.lastRoot);
  });
  return sel;
}

function amountInput(slot) {
  const input = document.createElement("input");
  input.type = "number";
  input.min = "0";
  input.step = "50";
  input.className = "prize-amount";
  input.autocomplete = "off";
  input.value = String(slot.amount ?? 0);
  input.setAttribute("aria-label", `${slot.label} prize amount in CAD`);
  input.addEventListener("change", () => {
    const n = Math.max(0, Math.round(Number(input.value) || 0));
    slot.amount = n;
    input.value = String(n);
    save();
  });
  return input;
}

function canSend(slot) {
  if (!slot.teamId) return false;
  const req = latestForSlot(slot.id);
  if (!req) return true;
  return req.status === "confirmed";
}

function paintSlot(slot) {
  const team = teamById(slot.teamId);
  const req = latestForSlot(slot.id);
  const row = el("div", "prize-slot");
  row.dataset.slotId = slot.id;

  const rank = el("div", "prize-rank");
  rank.append(el("p", "prize-kicker", slot.label));
  rank.append(amountInput(slot));
  rank.append(el("span", "prize-cad", "CAD"));

  const who = el("div", "prize-who");
  who.append(teamSelect(slot));
  if (team) {
    const leader = leaderOf(team);
    const mail = el("p", "prize-mail", emailFromName(memberName(leader), team.id));
    mail.title = "Interac inbox we will request a deposit to — not a bank login";
    who.append(mail);
  } else {
    who.append(el("p", "prize-mail", "Choose a winning team from tonight’s roster."));
  }

  const actions = el("div", "prize-slot-actions");
  const send = el("button", "ops-btn", "Send deposit request");
  send.type = "button";
  send.disabled = !canSend(slot);
  send.addEventListener("click", () => sendDepositRequest(slot.id));
  actions.append(send);

  if (req) {
    const badge = el("span", `prize-status is-${req.status}`, statusLabel(req.status));
    actions.append(badge);
  }

  row.append(rank, who, actions);
  return row;
}

/**
 * Organizer control — nest under Presentations.
 * @param {HTMLElement} root
 */
export function paintPrizePanel(root) {
  if (!root) return;
  paintPrizePanel.lastRoot = root;
  ensureStyles();

  let mount = root.querySelector("[data-prize-panel]");
  if (!mount) {
    mount = el("section", "prize-panel");
    mount.dataset.prizePanel = "1";
    mount.setAttribute("aria-label", "Prize payouts");
    mount.setAttribute("data-hide-on-projector", "");
    root.append(mount);
  }
  mount.replaceChildren();

  mount.append(el("p", "prize-kicker", "Prizes"));
  mount.append(el("h3", "prize-title", "Winner payouts"));
  const blurb = el(
    "p",
    "prize-blurb",
    "Organizer Interac e-Transfer deposit requests for tonight’s CAD prizes. Nightboard never logs in to a bank, never collects account numbers, SIN, or cards, and never moves money. Each winner is asked to confirm the email that should receive the prize.",
  );
  mount.append(blurb);

  const slots = el("div", "prize-slots");
  for (const slot of ctx.slots) slots.append(paintSlot(slot));
  mount.append(slots);

  const bar = el("div", "prize-bar");
  const sendAll = el("button", "ops-btn ops-btn-ink", "Send all");
  sendAll.type = "button";
  sendAll.addEventListener("click", () => sendAllDepositRequests());
  const csv = el("button", "ops-btn", "Export CSV");
  csv.type = "button";
  csv.addEventListener("click", () => exportPayoutsCSV());
  const json = el("button", "ops-btn", "Export JSON");
  json.type = "button";
  json.addEventListener("click", () => exportPayoutsJSON());
  const seed = el("button", "ops-btn", "Seed demo winners");
  seed.type = "button";
  seed.addEventListener("click", () => seedDemoWinners());
  bar.append(sendAll, csv, json, seed);
  mount.append(bar);

  const logKicker = el("p", "prize-kicker");
  logKicker.style.marginTop = "16px";
  logKicker.textContent = "Payout log";
  mount.append(logKicker);

  if (!ctx.requests.length) {
    mount.append(
      el("p", "prize-mail", "No deposit requests yet. Pick winners, then send."),
    );
  } else {
    const log = el("div", "prize-log");
    for (const req of [...ctx.requests].reverse()) {
      const line = el("div", "prize-log-row");
      const left = el("div");
      left.append(
        el(
          "strong",
          null,
          `${req.slotLabel} · ${req.teamName} · ${fmtCAD(req.amount)}`,
        ),
      );
      left.append(
        el(
          "p",
          "prize-mail",
          `${req.leaderName} · ${req.confirmedEmail || req.leaderEmail} · ${METHOD}`,
        ),
      );
      line.append(left, el("span", `prize-status is-${req.status}`, statusLabel(req.status)));
      log.append(line);
    }
    mount.append(log);
  }
}

/**
 * Winner-side banner on a team card: confirm the Interac receiving email.
 * @param {HTMLElement} card
 * @param {{id:string}} team
 */
export function paintPrizeBanner(card, team) {
  if (!card || !team) return;
  ensureStyles();
  card.querySelectorAll("[data-prize-banner]").forEach((n) => n.remove());

  const req = latestForTeam(team.id);
  if (!req) return;
  if (req.status !== "sent" && req.status !== "awaiting" && req.status !== "confirmed") return;

  const banner = el("div", "prize-banner");
  banner.dataset.prizeBanner = "1";
  stopCardClick(banner);

  banner.append(el("p", "prize-kicker", "Prize deposit"));

  if (req.status === "confirmed") {
    banner.append(el("strong", null, "Receiving email confirmed"));
    banner.append(
      el(
        "p",
        "prize-mail",
        `Organizers will send a ${fmtCAD(req.amount)} Interac e-Transfer deposit request to ${req.confirmedEmail}.`,
      ),
    );
    card.prepend(banner);
    return;
  }

  banner.append(el("strong", null, "Prize deposit — confirm receiving email"));
  banner.append(
    el(
      "p",
      "prize-mail",
      `${req.slotLabel} · ${fmtCAD(req.amount)}. Cursor Calgary organizers will send an Interac e-Transfer deposit request. Confirm the email that should receive the prize — not a bank login.`,
    ),
  );

  const row = el("div", "prize-confirm-row");
  const input = document.createElement("input");
  input.type = "email";
  input.className = "prize-email";
  input.autocomplete = "off";
  input.value = req.leaderEmail;
  input.setAttribute("aria-label", "Email to receive the Interac prize");
  const btn = el("button", "ops-btn ops-btn-ink", "Confirm");
  btn.type = "button";
  btn.addEventListener("click", (ev) => {
    ev.stopPropagation();
    confirmPayout(req.id, input.value);
  });
  row.append(input, btn);
  banner.append(row);
  card.prepend(banner);
}

/**
 * @param {{ teams: object[], youAre?: string }} options
 */
export function initPrizes({ teams, youAre }) {
  ctx.teams = teams;
  ctx.youAre = youAre || "";
  ensureStyles();

  const stored = loadRaw();
  if (stored?.slots?.length) {
    ctx.slots = DEFAULT_SLOTS.map((base) => {
      const hit = stored.slots.find((s) => s.id === base.id) || {};
      return {
        ...base,
        amount: hit.amount != null ? Number(hit.amount) : base.amount,
        teamId: hit.teamId || "",
        label: hit.label || base.label,
      };
    });
    ctx.requests = Array.isArray(stored.requests) ? stored.requests : [];
  } else {
    ctx.slots = defaultSlots();
    ctx.requests = [];
    save();
  }
  catchUp();
  save();
}

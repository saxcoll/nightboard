/**
 * Nightboard chat — local-only, two channels.
 *
 * Channels: `team` (your table only) · `general` (whole room)
 * Storage:  localStorage key `nightboard-chat`
 *
 * Not a network. Not encrypted. Survives refresh in this browser.
 */

export const STORAGE_KEY = "nightboard-chat";
export const CHANNELS = Object.freeze(["team", "general"]);

const CSS_HREF = new URL("../css/chat.css", import.meta.url).href;
const SIM_INCOMING_ID = "nb-sim-general-incoming";
const SIM_INCOMING_MS = 8000;
const STATE_VERSION = 1;

const SEED_TEAM_LINES = [
  {
    author: "Daniel Collins",
    offsetMin: 0,
    text: "Nightboard is up on the table. Team chat is just us — refresh keeps the thread.",
  },
  {
    author: "Sadhvi Sharma",
    offsetMin: 2,
    text: "I’ll stay on the floor grid. Don’t let Chat fight the packer.",
  },
  {
    author: "Nathan Nguyen",
    offsetMin: 5,
    text: "Pizza cam still empties Hawaiian first. Flag hits at 3 slices.",
  },
  {
    author: "Samuel Collins",
    offsetMin: 8,
    text: "Wiring this tab now. General is the whole room; this channel is Spaceflex only.",
  },
];

const SEED_GENERAL_LINES = [
  {
    id: "nb-seed-general-1",
    author: "Check-in",
    teamName: "Room",
    offsetMin: -32,
    text: "Welcome in. Suite 1900 — pizza by the windows, stage is the far wall.",
  },
  {
    id: "nb-seed-general-2",
    author: "Nightboard ops",
    teamName: "Room",
    offsetMin: -4,
    text: "Build is live. Volunteer for demos on Presentations — three teams, one click each.",
  },
];

let root = null;
let tz = "America/Edmonton";
let you = { name: "You", teamId: "", teamName: "Team" };
let state = emptyState();
let chatVisible = false;
let incomingTimer = 0;

function emptyState() {
  return {
    v: STATE_VERSION,
    active: "team",
    unread: { team: 0, general: 0 },
    messages: { team: [], general: [] },
  };
}

function memberName(m) {
  return typeof m === "string" ? m : m?.name || "";
}

function resolveYou(teams, youAre) {
  const list = Array.isArray(teams) ? teams : [];
  const team =
    list.find((t) => t.id === youAre) ||
    list.find((t) => t.mine) ||
    list.find((t) => /spaceflex/i.test(`${t.id || ""} ${t.name || ""}`)) ||
    list[0];
  const names = (team?.members || []).map(memberName).filter(Boolean);
  const name = names.find((n) => /samuel collins/i.test(n)) || names[0] || "You";
  return {
    name,
    teamId: team?.id || "",
    teamName: team?.name || "Team",
    members: names,
  };
}

function seedStart() {
  return new Date("2026-08-26T18:34:00-06:00").getTime();
}

function makeId(channel, suffix) {
  return `nb-${channel}-${suffix}`;
}

function seedMessages(youCtx) {
  const base = seedStart();
  const members = new Set(youCtx.members.map((n) => n.toLowerCase()));
  const team = [];
  SEED_TEAM_LINES.forEach((line, i) => {
    if (!members.has(line.author.toLowerCase())) return;
    team.push({
      id: makeId("team", `seed-${i + 1}`),
      channel: "team",
      author: line.author,
      teamId: youCtx.teamId,
      teamName: youCtx.teamName,
      text: line.text,
      ts: base + line.offsetMin * 60000,
    });
  });

  const general = SEED_GENERAL_LINES.map((line) => ({
    id: line.id,
    channel: "general",
    author: line.author,
    teamId: null,
    teamName: line.teamName,
    text: line.text,
    ts: base + line.offsetMin * 60000,
  }));

  return { team, general };
}

function loadState(youCtx) {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed || parsed.v !== STATE_VERSION) return null;
    const team = Array.isArray(parsed.messages?.team) ? parsed.messages.team : [];
    const general = Array.isArray(parsed.messages?.general) ? parsed.messages.general : [];
    return {
      v: STATE_VERSION,
      active: parsed.active === "general" ? "general" : "team",
      unread: {
        team: Number(parsed.unread?.team) || 0,
        general: Number(parsed.unread?.general) || 0,
      },
      messages: { team, general },
    };
  } catch {
    return null;
  }
}

function saveState() {
  try {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        v: state.v,
        active: state.active,
        unread: state.unread,
        messages: state.messages,
      }),
    );
  } catch {
    /* quota / private mode — chat still works this session */
  }
}

function ensureStyles() {
  if (document.getElementById("nb-chat-css")) return;
  const link = document.createElement("link");
  link.id = "nb-chat-css";
  link.rel = "stylesheet";
  link.href = CSS_HREF;
  document.head.appendChild(link);
}

function fmtMsgTime(ts) {
  return new Intl.DateTimeFormat("en-CA", {
    hour: "numeric",
    minute: "2-digit",
    timeZone: tz,
  }).format(new Date(ts));
}

function unreadTotal() {
  return (state.unread.team || 0) + (state.unread.general || 0);
}

function paintTabBadge() {
  const tab = document.querySelector('.tabs [data-view="chat"]');
  if (!tab) return;
  let badge = tab.querySelector(".chat-tab-badge");
  if (!badge) {
    badge = document.createElement("span");
    badge.className = "chat-tab-badge";
    tab.append(" ", badge);
  }
  const n = unreadTotal();
  badge.hidden = n === 0;
  badge.textContent = n > 9 ? "9+" : String(n);
  tab.setAttribute("aria-label", n ? `Chat, ${n} unread` : "Chat");
}

function markRead(channel) {
  if (!CHANNELS.includes(channel)) return;
  if (state.unread[channel] === 0) return;
  state.unread[channel] = 0;
  saveState();
}

function pushMessage(msg, { simulated = false } = {}) {
  const channel = msg.channel === "general" ? "general" : "team";
  const list = state.messages[channel];
  if (list.some((m) => m.id === msg.id)) return false;
  list.push(msg);
  const viewing = chatVisible && state.active === channel;
  if (simulated && !viewing) {
    state.unread[channel] += 1;
  }
  saveState();
  return true;
}

function sendFromComposer(text) {
  const trimmed = text.trim();
  if (!trimmed) return;
  const channel = state.active;
  pushMessage({
    id: makeId(channel, `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`),
    channel,
    author: you.name,
    teamId: you.teamId,
    teamName: you.teamName,
    text: trimmed,
    ts: Date.now(),
  });
  paint();
}

function setChannel(channel) {
  if (!CHANNELS.includes(channel) || state.active === channel) {
    if (CHANNELS.includes(channel)) {
      markRead(channel);
      paint();
    }
    return;
  }
  state.active = channel;
  markRead(channel);
  saveState();
  paint();
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

function renderShell() {
  root.replaceChildren();
  root.classList.add("chat-root");

  const head = el("div", "chat-head");
  const title = el("strong", "chat-title", "Chat");
  const sub = el("span", "chat-sub", `${you.teamName} · this browser only`);
  head.append(title, sub);

  const channels = el("div", "chat-channels");
  channels.setAttribute("role", "group");
  channels.setAttribute("aria-label", "Channel");
  for (const id of CHANNELS) {
    const btn = el("button", "chat-channel");
    btn.type = "button";
    btn.dataset.channel = id;
    btn.setAttribute("aria-pressed", "false");
    const label = el("span", null, id === "team" ? "Team" : "General");
    const pip = el("span", "chat-pip");
    pip.hidden = true;
    btn.append(label, pip);
    btn.addEventListener("click", () => setChannel(id));
    channels.append(btn);
  }

  const log = el("div", "chat-log");
  log.id = "chat-log";
  log.setAttribute("role", "log");
  log.setAttribute("aria-live", "polite");
  log.setAttribute("aria-relevant", "additions");

  const form = el("form", "chat-composer");
  const input = document.createElement("input");
  input.type = "text";
  input.id = "chat-input";
  input.autocomplete = "off";
  input.maxLength = 500;
  const send = el("button", "chat-send", "Send");
  send.type = "submit";
  form.append(input, send);
  form.addEventListener("submit", (ev) => {
    ev.preventDefault();
    sendFromComposer(input.value);
    input.value = "";
    input.focus();
  });
  input.addEventListener("keydown", (ev) => {
    if (ev.key !== "Enter" || ev.shiftKey) return;
    ev.preventDefault();
    form.requestSubmit();
  });

  root.append(head, channels, log, form);
}

function paintLog() {
  const log = root.querySelector("#chat-log");
  if (!log) return;
  const channel = state.active;
  const messages = state.messages[channel] || [];
  log.replaceChildren();

  if (!messages.length) {
    const empty = el("p", "chat-empty", channel === "team" ? "No team messages yet." : "No room messages yet.");
    log.append(empty);
    return;
  }

  for (const msg of messages) {
    const row = el("article", "chat-msg");
    const mine = msg.author === you.name && msg.teamId === you.teamId;
    if (mine) row.classList.add("is-mine");
    const meta = el("div", "chat-msg-meta");
    const author = el("span", "chat-msg-author", mine ? `${msg.author} (you)` : msg.author);
    meta.append(author);
    if (channel === "general" && msg.teamName) {
      meta.append(el("span", "chat-msg-team", msg.teamName));
    }
    meta.append(el("time", "chat-msg-time", fmtMsgTime(msg.ts)));
    const body = el("p", "chat-msg-text", msg.text);
    row.append(meta, body);
    log.append(row);
  }

  log.scrollTop = log.scrollHeight;
}

function paintChrome() {
  const input = root.querySelector("#chat-input");
  if (input) {
    input.placeholder = state.active === "team" ? `Message ${you.teamName}` : "Message the room";
    input.setAttribute("aria-label", state.active === "team" ? `Message ${you.teamName}` : "Message general chat");
  }

  root.querySelectorAll(".chat-channel").forEach((btn) => {
    const id = btn.dataset.channel;
    const on = id === state.active;
    btn.setAttribute("aria-pressed", on ? "true" : "false");
    const pip = btn.querySelector(".chat-pip");
    const n = state.unread[id] || 0;
    if (pip) {
      pip.hidden = n === 0;
      pip.textContent = n > 9 ? "9+" : String(n);
    }
  });

  paintTabBadge();
}

function paint() {
  if (!root) {
    paintTabBadge();
    return;
  }
  paintChrome();
  paintLog();
}

function scheduleIncoming() {
  if (incomingTimer) return;
  const already = (state.messages.general || []).some((m) => m.id === SIM_INCOMING_ID);
  if (already) return;
  incomingTimer = window.setTimeout(() => {
    incomingTimer = 0;
    const added = pushMessage(
      {
        id: SIM_INCOMING_ID,
        channel: "general",
        author: "Pizza station",
        teamId: null,
        teamName: "Room",
        text: "Hawaiian is down to a few slices. Come grab some before demos.",
        ts: Date.now(),
      },
      { simulated: true },
    );
    if (added) paint();
  }, SIM_INCOMING_MS);
}

export function mountChat(rootEl, { teams, youAre, timezone } = {}) {
  if (!rootEl) return;
  root = rootEl;
  tz = timezone || "America/Edmonton";
  you = resolveYou(teams, youAre);
  ensureStyles();
  state = loadState(you) || { ...emptyState(), messages: seedMessages(you) };
  if (!loadState(you)) saveState();
  renderShell();
  if (chatVisible) markRead(state.active);
  paint();
  scheduleIncoming();
}

export function setChatActive(on) {
  chatVisible = Boolean(on);
  if (chatVisible) {
    markRead(state.active);
    paint();
  } else {
    paintTabBadge();
  }
}

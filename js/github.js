const POLL_MS = 90_000;
const REQUEST_BUDGET = 10;
const GH = "https://api.github.com";
const SHIP_HINT = /\b(ship|demo|readme|ui)\b/i;
const FAKE_COMMIT = "wire pizza cam";

const STATUS_RANK = {
  stalled: 0,
  "demo-ready": 1,
  shipping: 2,
  warming: 3,
  quiet: 4,
};

let root = null;
let teams = [];
let eventInfo = null;
let timer = null;
let lastPollAt = null;
let fetchCursor = 0;

/** @type {Record<string, object>} */
const pulses = Object.create(null);
/** @type {Map<string, object>} */
const repoCache = new Map();
/** @type {Map<string, object[]>} */
const commitCache = new Map();

export function mountGithubBot(rootEl, { teams: teamList, event }) {
  root = rootEl;
  teams = Array.isArray(teamList) ? teamList : [];
  eventInfo = event;
  ensureStyles();
  renderShell();
  if (timer) clearInterval(timer);
  void poll();
  timer = setInterval(() => void poll(), POLL_MS);
}

export function getTeamPulse(teamId) {
  const pulse = pulses[teamId];
  return pulse ? publicPulse(pulse) : null;
}

function publicPulse(pulse) {
  return {
    percent: pulse.percent,
    status: pulse.status,
    lastMessage: pulse.lastMessage,
    lastAt: pulse.lastAt,
    commits: pulse.commits,
    source: pulse.source,
  };
}

function ensureStyles() {
  if (document.getElementById("gh-bot-css")) return;
  const link = document.createElement("link");
  link.id = "gh-bot-css";
  link.rel = "stylesheet";
  link.href = new URL("../css/github.css", import.meta.url).href;
  document.head.appendChild(link);
}

function renderShell() {
  if (!root) return;
  root.innerHTML = "";
  const bot = document.createElement("section");
  bot.className = "gh-bot";
  bot.innerHTML = `
    <header class="gh-head">
      <strong class="gh-title">Repo bot</strong>
      <span class="gh-poll" data-gh="poll">—</span>
      <span class="gh-live"><i class="gh-dot" aria-hidden="true"></i>Polling GitHub</span>
    </header>
    <ul class="gh-list" data-gh="list"></ul>
  `;
  root.appendChild(bot);
  paintList();
}

async function poll() {
  if (!eventInfo) return;
  const now = Date.now();
  let budget = REQUEST_BUDGET;
  const live = teams.filter((t) => t.repo);
  const simulated = teams.filter((t) => !t.repo);

  const order = rotate(live, fetchCursor);
  fetchCursor = live.length ? (fetchCursor + 1) % live.length : 0;

  for (const team of order) {
    const needMeta = !repoCache.has(team.id);
    if (budget < 1) {
      settleTeam(team, now, "budget");
      continue;
    }

    try {
      if (needMeta && budget > 1) {
        budget -= 1;
        repoCache.set(team.id, await ghRepo(team.repo));
      }
      budget -= 1;
      commitCache.set(team.id, await ghCommits(team.repo, eventInfo.start));
      pulses[team.id] = estimateTeam(team, now, "github");
    } catch {
      settleTeam(team, now, "error");
    }
  }

  for (const team of simulated) {
    pulses[team.id] = simPulse(team, now, { house: Boolean(team.mine) || team.id === "spaceflex" });
  }

  lastPollAt = now;
  paintList();
  dispatchPulses();
}

function settleTeam(team, now, reason) {
  if (commitCache.has(team.id) || repoCache.has(team.id)) {
    const pulse = estimateTeam(team, now, "sim");
    pulse.comment =
      reason === "budget"
        ? `${pulse.comment} Holding last GitHub read.`
        : `GitHub went quiet. Holding last read — ${pulse.status}.`;
    pulses[team.id] = pulse;
    return;
  }
  pulses[team.id] = simPulse(team, now, { house: false, waiting: true });
}

function rotate(list, start) {
  if (!list.length) return list;
  const i = start % list.length;
  return list.slice(i).concat(list.slice(0, i));
}

async function ghRepo(repo) {
  return ghGet(`/repos/${repo}`);
}

async function ghCommits(repo, since) {
  const q = new URLSearchParams({ since, per_page: "20" });
  const rows = await ghGet(`/repos/${repo}/commits?${q}`);
  if (!Array.isArray(rows)) return [];
  return rows.map((row) => ({
    message: String(row?.commit?.message || "").split("\n")[0].trim(),
    date: row?.commit?.committer?.date || row?.commit?.author?.date || null,
  }));
}

async function ghGet(path) {
  let res;
  try {
    res = await fetch(`${GH}${path}`, {
      headers: { Accept: "application/vnd.github+json" },
    });
  } catch {
    const err = new Error("network");
    err.code = "network";
    throw err;
  }
  if (res.status === 403 || res.status === 429) {
    const err = new Error("rate");
    err.code = "rate";
    throw err;
  }
  if (!res.ok) {
    const err = new Error("http");
    err.code = "http";
    err.status = res.status;
    throw err;
  }
  return res.json();
}

function estimateTeam(team, now, source) {
  const commits = commitCache.get(team.id) || [];
  const meta = repoCache.get(team.id) || null;
  const last = commits[0] || null;
  const count = commits.length;
  const minutesAgo = last?.date ? (now - Date.parse(last.date)) / 60000 : Infinity;

  const commitScore = Math.min(58, (count / 8) * 58);

  let recency = 0;
  if (Number.isFinite(minutesAgo)) {
    if (minutesAgo < 8) recency = 28;
    else if (minutesAgo < 20) recency = 16;
    else if (minutesAgo < 40) recency = 6;
    else recency = -20;
  }

  let signals = 0;
  if (meta) {
    if (meta.has_pages) signals += 8;
    if (typeof meta.size === "number" && meta.size > 0) {
      signals += Math.min(7, Math.log10(Math.max(10, meta.size)) * 1.6);
    }
    if (meta.pushed_at) {
      const pushedMin = (now - Date.parse(meta.pushed_at)) / 60000;
      if (pushedMin < 60) signals += 6;
      else if (pushedMin < 24 * 60) signals += 3;
    }
  }

  const percent = clamp(0, 100, Math.round(commitScore + recency + signals));
  const lastMessage = truncate(last?.message || (count ? "" : "no commits since doors"));
  const lastAt = last?.date || null;

  let status = "quiet";
  if (count > 0 && minutesAgo > 40) status = "stalled";
  else if (count >= 8 || (count >= 3 && minutesAgo < 8) || (count >= 5 && minutesAgo < 20)) {
    status = "shipping";
  } else if (count >= 1) status = "warming";

  if (status === "shipping" && SHIP_HINT.test(lastMessage)) status = "demo-ready";

  return {
    teamId: team.id,
    name: team.name,
    repo: team.repo,
    percent,
    status,
    lastMessage,
    lastAt,
    commits: count,
    source,
    comment: botLine({ commits, now, status, minutesAgo }),
  };
}

function simPulse(team, now, { house = false, waiting = false } = {}) {
  const start = Date.parse(eventInfo.start);
  const end = Date.parse(eventInfo.end);
  const span = Math.max(1, end - start);
  const u = clamp(0, 1, (now - start) / span);
  const eased = u * u * (3 - 2 * u);

  let percent = Math.round(6 + eased * 86);
  if (!house) percent = Math.round(4 + eased * 48);
  if (waiting) percent = Math.min(percent, 22);

  const commits = Math.min(12, Math.floor(eased * (house ? 10 : 5)));
  const lagMin = house ? 4 + (1 - u) * 10 : 12 + (1 - u) * 28;
  const lastAt = new Date(now - lagMin * 60000).toISOString();
  const lastMessage = waiting ? "waiting on GitHub" : house ? FAKE_COMMIT : "sim pulse — no live repo";

  let status = "quiet";
  if (u < 0.12) status = "quiet";
  else if (u < 0.38) status = "warming";
  else if (house && u > 0.88) status = "demo-ready";
  else if (u > 0.48) status = "shipping";
  else status = "warming";
  if (!house && status === "shipping" && percent < 40) status = "warming";

  const minutesAgo = lagMin;
  const hourPushes = Math.max(0, Math.min(commits, Math.round(eased * 4)));
  const comment = waiting
    ? `Sim · GitHub unreachable. Climbing with the clock — ${status}.`
    : house
      ? `Sim · ${hourPushes} fake ${hourPushes === 1 ? "push" : "pushes"} this hour. Last commit ${fmtMins(minutesAgo)} — ${status}.`
      : `Sim · no live repo. Last commit ${fmtMins(minutesAgo)} — ${status}.`;

  return {
    teamId: team.id,
    name: team.name,
    repo: team.repo,
    percent: clamp(0, 100, percent),
    status,
    lastMessage,
    lastAt,
    commits,
    source: "sim",
    comment,
  };
}

function botLine({ commits, now, status, minutesAgo }) {
  const hourAgo = now - 60 * 60 * 1000;
  const inHour = commits.filter((c) => c.date && Date.parse(c.date) >= hourAgo).length;
  const pushBit =
    inHour === 0
      ? "No pushes in the last hour."
      : inHour === 1
        ? "1 push in the last hour."
        : `${inHour} pushes in the last hour.`;
  const lastBit = !Number.isFinite(minutesAgo)
    ? "No commits since doors"
    : minutesAgo < 1
      ? "Last commit just now"
      : `Last commit ${fmtMins(minutesAgo)}`;
  return `${pushBit} ${lastBit} — ${status}.`;
}

function dispatchPulses() {
  const detail = { pulses: {} };
  for (const team of teams) {
    const pulse = pulses[team.id];
    if (pulse) detail.pulses[team.id] = publicPulse(pulse);
  }
  window.dispatchEvent(new CustomEvent("nightboard:github", { detail }));
}

function paintList() {
  if (!root) return;
  const pollEl = root.querySelector("[data-gh=poll]");
  const list = root.querySelector("[data-gh=list]");
  if (pollEl) pollEl.textContent = lastPollAt ? fmtPoll(lastPollAt) : "waiting";
  if (!list) return;

  const rows = teams
    .map((t) => pulses[t.id] || pendingRow(t))
    .sort((a, b) => {
      const ra = STATUS_RANK[a.status] ?? 9;
      const rb = STATUS_RANK[b.status] ?? 9;
      if (ra !== rb) return ra - rb;
      return b.percent - a.percent;
    });

  list.innerHTML = "";
  for (const pulse of rows) list.appendChild(rowEl(pulse));
}

function pendingRow(team) {
  return {
    teamId: team.id,
    name: team.name,
    repo: team.repo,
    percent: 0,
    status: "quiet",
    lastMessage: team.repo ? "asking GitHub…" : FAKE_COMMIT,
    lastAt: null,
    commits: 0,
    source: team.repo ? "github" : "sim",
    comment: team.repo ? "First poll in flight." : "Sim pulse warming up with the room.",
  };
}

function rowEl(pulse) {
  const li = document.createElement("li");
  li.className = `gh-row gh-${pulse.status}`;
  li.dataset.teamId = pulse.teamId;

  const top = document.createElement("div");
  top.className = "gh-row-top";

  const name = document.createElement("span");
  name.className = "gh-name";
  name.textContent = pulse.name;
  top.appendChild(name);

  if (pulse.repo) {
    const link = document.createElement("a");
    link.className = "gh-repo";
    link.href = `https://github.com/${pulse.repo}`;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = pulse.repo;
    top.appendChild(link);
  } else {
    const none = document.createElement("span");
    none.className = "gh-repo gh-repo-none";
    none.textContent = "no remote";
    top.appendChild(none);
  }

  if (pulse.source === "sim") {
    const badge = document.createElement("span");
    badge.className = "gh-sim";
    badge.textContent = "sim";
    top.appendChild(badge);
  }

  const pct = document.createElement("span");
  pct.className = "gh-pct";
  pct.textContent = `${pulse.percent}%`;
  top.appendChild(pct);

  const bar = document.createElement("div");
  bar.className = "gh-bar";
  bar.setAttribute("role", "progressbar");
  bar.setAttribute("aria-valuemin", "0");
  bar.setAttribute("aria-valuemax", "100");
  bar.setAttribute("aria-valuenow", String(pulse.percent));
  const fill = document.createElement("i");
  fill.style.width = `${pulse.percent}%`;
  bar.appendChild(fill);

  const meta = document.createElement("div");
  meta.className = "gh-meta";
  const status = document.createElement("span");
  status.className = `gh-status gh-status-${pulse.status}`;
  status.textContent = pulse.status;
  const commit = document.createElement("span");
  commit.className = "gh-commit";
  commit.textContent = pulse.lastMessage || "—";
  commit.title = pulse.lastMessage || "";
  meta.append(status, commit);

  const comment = document.createElement("p");
  comment.className = "gh-comment";
  comment.textContent = pulse.comment;

  li.append(top, bar, meta, comment);
  return li;
}

function fmtPoll(ms) {
  return new Intl.DateTimeFormat("en-CA", {
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
  }).format(new Date(ms));
}

function fmtMins(minutes) {
  const m = Math.max(0, Math.round(minutes));
  if (m === 1) return "1 min ago";
  return `${m} min ago`;
}

function truncate(text, n = 72) {
  const s = String(text || "").replace(/\s+/g, " ").trim();
  if (s.length <= n) return s;
  return `${s.slice(0, n - 1)}…`;
}

function clamp(min, max, n) {
  return Math.min(max, Math.max(min, n));
}

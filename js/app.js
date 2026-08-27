import { announcements, demoQueue, event, hub, people, pizza, rooms, rules, schedule, teams, youAre } from "./data.js";
import { mountChat, setChatActive } from "./chat.js";
import { getTeamPulse, mountGithubBot } from "./github.js";
import { initJoins, paintJoinPanel, paintInbound, paintTeamActions } from "./joins.js";
import { mountOps } from "./ops.js";
import { mountPizza } from "./pizza.js";

const tz = event.timezone;
let selectedId = youAre || teams[0]?.id;
const volunteers = new Set(teams.filter((t) => t.volunteered).map((t) => t.id));

function fmtTime(iso) {
  return new Intl.DateTimeFormat("en-CA", {
    hour: "numeric",
    minute: "2-digit",
    timeZone: tz,
  }).format(new Date(iso));
}

function fmtClock(date) {
  return new Intl.DateTimeFormat("en-CA", {
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
    timeZone: tz,
  }).format(date);
}

function minutesBetween(a, b) {
  return Math.max(0, Math.round((b - a) / 60000));
}

function currentPhase(now) {
  const t = now.getTime();
  let current = null;
  let next = null;
  for (const block of schedule) {
    const start = new Date(block.start).getTime();
    const end = new Date(block.end).getTime();
    if (t >= start && t < end) current = block;
    if (t < start && !next) next = block;
  }
  return { current, next };
}

function renderNow(now) {
  const { current, next } = currentPhase(now);
  const nowTitle = document.getElementById("now-title");
  const nowMeta = document.getElementById("now-meta");
  const nextTitle = document.getElementById("next-title");
  const nextMeta = document.getElementById("next-meta");

  if (!current && now.getTime() < new Date(event.start).getTime()) {
    nowTitle.textContent = "Doors soon";
    nowMeta.textContent = `Opens ${fmtTime(event.start)}`;
  } else if (!current && now.getTime() >= new Date(event.end).getTime()) {
    nowTitle.textContent = "That’s a wrap";
    nowMeta.textContent = "Gallery stays up. Go talk to people.";
  } else if (current) {
    nowTitle.textContent = current.title;
    nowMeta.textContent = `${minutesBetween(now.getTime(), new Date(current.end).getTime())} min left · ${fmtTime(current.start)}–${fmtTime(current.end)}`;
  }

  if (next) {
    nextTitle.textContent = next.title;
    nextMeta.textContent = `Starts ${fmtTime(next.start)}`;
  } else {
    nextTitle.textContent = "After";
    nextMeta.textContent = "Demos, then the hallway conversations.";
  }
}

function teamById(id) {
  return teams.find((t) => t.id === id);
}

function memberName(m) {
  return typeof m === "string" ? m : m.name;
}

function memberRole(m) {
  return typeof m === "string" ? "" : m.role || "";
}

function pulseLabel(pulse) {
  if (!pulse) return "";
  return `${pulse.percent}% ${pulse.status}`;
}

function isYourTable(team) {
  if (!team) return false;
  if (team.mine) return true;
  if (youAre && team.id === youAre) return true;
  return /spaceflex/i.test(`${team.id} ${team.name}`);
}

/** Columns from floor width and team count. Cell never narrower than the table. */
function packColumns(count, widthPx) {
  const n = Math.max(1, count);
  const width = !widthPx || widthPx < 80 ? 720 : widthPx;
  const compact = width < 520;
  const cellW = compact ? 128 : 156;
  const gapX = compact ? 20 : 28;
  const pad = compact ? 16 : 24;
  const usable = Math.max(cellW, width - pad);
  const maxByWidth = Math.max(1, Math.floor((usable + gapX) / (cellW + gapX)));
  const target = Math.ceil(Math.sqrt(n * 1.4));
  return Math.max(1, Math.min(n, maxByWidth, Math.max(2, target), 7));
}

function packedSeat(i) {
  const cols = 7;
  return `${String.fromCharCode(65 + (Math.floor(i / cols) % 26))}${(i % cols) + 1}`;
}

let floorPackerObs;

function bindFloorPacker(floor, grid) {
  const apply = () => {
    grid.style.setProperty("--floor-cols", String(packColumns(teams.length, floor.clientWidth)));
  };
  apply();
  floorPackerObs?.disconnect();
  floorPackerObs = new ResizeObserver(apply);
  floorPackerObs.observe(floor);
}

function renderFloor() {
  const floor = document.getElementById("floor");
  floor.innerHTML = "";

  const zones = document.createElement("div");
  zones.className = "floor-zones";
  for (const zone of Object.values(rooms)) {
    const el = document.createElement("div");
    el.className = "zone";
    el.textContent = zone.label;
    zones.appendChild(el);
  }
  floor.appendChild(zones);

  const grid = document.createElement("div");
  grid.className = "floor-tables";
  grid.id = "floor-tables";

  teams.forEach((team, i) => {
    const pulse = getTeamPulse(team.id);
    const mine = isYourTable(team);
    const label = team.table || packedSeat(i);
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = mine ? "table table-mine" : "table";
    btn.dataset.teamId = team.id;
    const n = team.members.length;
    btn.title = mine
      ? `${team.name} — your table · ${n} ${n === 1 ? "member" : "members"}`
      : `${team.name} · ${n} ${n === 1 ? "member" : "members"}`;
    btn.innerHTML = `
      <span class="table-top">
        <div class="meta">
          <span>${label}</span>
          ${mine ? `<span class="you-flag">you</span>` : ""}
          <span class="table-n">${n}</span>
          <span class="status ${team.status}">${team.status}</span>
        </div>
        <strong>${team.name}</strong>
        <div class="table-pulse" data-status="${pulse?.status || ""}">${pulse ? pulseLabel(pulse) : "repo…"}</div>
        ${team.need ? `<div class="need-flag" title="Need: ${team.need}">Need: ${team.need}</div>` : ""}
      </span>
    `;
    btn.addEventListener("click", () => selectTeam(team.id));
    grid.appendChild(btn);
  });

  floor.appendChild(grid);
  bindFloorPacker(floor, grid);
}

function selectTeam(id) {
  selectedId = id;
  document.querySelectorAll(".table").forEach((el) => {
    el.setAttribute("aria-pressed", el.dataset.teamId === id ? "true" : "false");
  });
  const team = teamById(id);
  const pulse = getTeamPulse(id);
  document.getElementById("panel-title").textContent = team.name;
  document.getElementById("panel-sub").textContent = team.mine
    ? "Your team"
    : `${team.members.length} ${team.members.length === 1 ? "member" : "members"}`;
  const repoLine = team.repo
    ? `<p class="time-left"><a href="https://github.com/${team.repo}" target="_blank" rel="noreferrer">${team.repo}</a></p>`
    : `<p class="time-left">No public repo yet — bot is simulating this table.</p>`;
  const pulseLine = pulse
    ? `<p><span class="status">${pulseLabel(pulse)}</span></p>
       <p class="time-left">${pulse.lastMessage || "No commit message yet."}</p>`
    : `<p class="time-left">Waiting on the first GitHub poll.</p>`;
  const chips = team.members
    .map((m) => `<span>${memberName(m)}${memberRole(m) === "leader" ? " ★" : ""}</span>`)
    .join("");
  document.getElementById("panel-body").innerHTML = `
    <div class="detail">
      <p>${team.oneLiner}</p>
      <p><span class="status ${team.status}">${team.status}</span></p>
      ${repoLine}
      ${pulseLine}
      <div class="chips">${chips}</div>
      ${team.need ? `<p class="need-flag">Open seat: ${team.need}</p>` : `<p>Team is full enough to ship tonight.</p>`}
    </div>
  `;
  paintTeamActions(document.getElementById("panel-body"), team);
}

function renderHackathon() {
  const body = document.getElementById("hackathon-body");
  if (!body) return;
  body.replaceChildren();

  const mine = teams.find((t) => t.id === youAre);
  if (mine) {
    const card = document.createElement("article");
    card.className = "team-card team-card-mine";
    const kicker = document.createElement("p");
    kicker.className = "kicker";
    kicker.textContent = "Your team";
    const title = document.createElement("h2");
    title.textContent = mine.name;
    const meta = document.createElement("p");
    meta.className = "time-left";
    meta.textContent = `${mine.members.length} members · ${volunteers.has(mine.id) ? "volunteered" : "not presenting"}`;
    const chips = document.createElement("div");
    chips.className = "chips";
    for (const mem of mine.members) {
      const span = document.createElement("span");
      span.textContent = `${memberName(mem)}${memberRole(mem) === "leader" ? " ★" : ""}`;
      chips.append(span);
    }
    const line = document.createElement("p");
    line.textContent = mine.oneLiner;
    card.append(kicker, title, meta, line, chips);
    paintInbound(card, mine.id);
    body.append(card);
  }

  const tools = document.createElement("div");
  tools.className = "team-tools";
  const search = document.createElement("input");
  search.type = "search";
  search.placeholder = "Filter teams";
  search.setAttribute("aria-label", "Filter teams");
  search.id = "team-filter";
  tools.append(search);
  body.append(tools);

  const list = document.createElement("div");
  list.id = "team-list";
  body.append(list);

  const paint = () => {
    const q = search.value.trim().toLowerCase();
    list.replaceChildren();
    const shown = teams.filter((team) => {
      if (!q) return true;
      const names = team.members.map(memberName).join(" ");
      return `${team.name} ${names}`.toLowerCase().includes(q);
    });
    document.getElementById("team-count").textContent = `${shown.length} teams`;
    for (const team of shown) {
      list.append(teamCard(team));
    }
  };

  search.addEventListener("input", paint);
  paint();
}

function teamCard(team) {
  const card = document.createElement("article");
  card.className = "team-card";
  if (team.mine) card.classList.add("is-mine");
  const head = document.createElement("div");
  head.className = "team-card-head";
  const title = document.createElement("h3");
  title.textContent = team.name;
  const count = document.createElement("span");
  count.className = "time-left";
  count.textContent = `${team.members.length} ${team.members.length === 1 ? "member" : "members"}`;
  head.append(title, count);
  const chips = document.createElement("div");
  chips.className = "chips";
  for (const mem of team.members) {
    const span = document.createElement("span");
    span.textContent = `${memberName(mem)}${memberRole(mem) === "leader" ? " ★" : ""}`;
    chips.append(span);
  }
  card.append(head, chips);
  paintTeamActions(card, team);
  card.addEventListener("click", () => selectTeam(team.id));
  return card;
}

function renderPresentations() {
  const body = document.getElementById("presentations-body");
  if (!body) return;
  body.replaceChildren();

  const intro = document.createElement("p");
  intro.textContent = "Your team volunteered to present. Three teams will be picked from volunteers tonight. One click per team.";
  body.append(intro);

  const mine = teams.find((t) => t.id === youAre);
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "ops-btn ops-btn-ink";
  const volunteered = volunteers.has(youAre);
  btn.textContent = volunteered ? "Volunteered" : "Volunteer to present";
  btn.disabled = volunteered;
  btn.addEventListener("click", () => {
    if (volunteers.has(youAre)) return;
    volunteers.add(youAre);
    renderPresentations();
    renderHackathon();
  });
  body.append(btn);

  const kicker = document.createElement("p");
  kicker.className = "kicker";
  kicker.style.marginTop = "18px";
  kicker.textContent = "Volunteered";
  body.append(kicker);

  const pool = teams.filter((t) => volunteers.has(t.id));
  if (!pool.length) {
    const empty = document.createElement("p");
    empty.className = "time-left";
    empty.textContent = "No volunteers yet.";
    body.append(empty);
  } else {
    for (const team of pool) {
      body.append(teamCard(team));
    }
  }

  if (mine) {
    const note = document.createElement("p");
    note.className = "time-left";
    note.textContent = `${mine.name} is in the volunteer pool.`;
    body.append(note);
  }
}

function renderRules() {
  const body = document.getElementById("rules-body");
  if (!body) return;
  body.replaceChildren();
  for (const line of rules) {
    const p = document.createElement("p");
    p.className = "rule-line";
    p.textContent = line;
    body.append(p);
  }
}

function setView(view) {
  document.querySelectorAll('.tabs [role="tab"]').forEach((tab) => {
    tab.setAttribute("aria-selected", tab.dataset.view === view ? "true" : "false");
  });

  const show = {
    hackathon: view === "hackathon",
    presentations: view === "presentations",
    rules: view === "rules",
    floor: view === "floor",
    pizza: view === "pizza",
    github: view === "github",
    chat: view === "chat",
  };

  const apply = (id, on) => {
    const node = document.getElementById(id);
    if (!node) return;
    node.classList.toggle("hidden", !on);
    node.hidden = !on;
  };

  apply("view-hackathon", show.hackathon);
  apply("view-joins", show.hackathon);
  apply("view-presentations", show.presentations);
  apply("view-rules", show.rules);
  apply("view-floor", show.floor);
  apply("side-panel", show.floor);
  apply("view-pizza", show.pizza);
  apply("view-github", show.github);
  apply("view-chat", show.chat);
  setChatActive(show.chat);

  document.querySelector(".layout").style.gridTemplateColumns =
    show.floor || show.hackathon ? "" : "1fr";

  if (show.floor) {
    requestAnimationFrame(() => {
      const floor = document.getElementById("floor");
      const grid = document.getElementById("floor-tables");
      if (floor && grid) {
        grid.style.setProperty("--floor-cols", String(packColumns(teams.length, floor.clientWidth)));
      }
    });
  }
}

function fmtElapsed(now) {
  const start = new Date(event.start).getTime();
  const ms = Math.max(0, now.getTime() - start);
  const totalMin = Math.floor(ms / 60000);
  const h = Math.floor(totalMin / 60);
  const m = totalMin % 60;
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
}

function tick() {
  const now = new Date();
  document.getElementById("clock").textContent = fmtClock(now);
  const elapsed = document.getElementById("elapsed");
  if (elapsed) elapsed.textContent = fmtElapsed(now);
  renderNow(now);
}

function onPizza(ev) {
  const trays = ev.detail?.trays || [];
  const total = trays.reduce((sum, t) => sum + (t.slices || 0), 0);
  const max = trays.reduce((sum, t) => sum + (t.max || 0), 0);
  document.getElementById("pizza-count").textContent = max ? `${total} / ${max}` : String(total);
  const low = ev.detail?.low || [];
  const source = ev.detail?.source || "sim";
  const label = document.getElementById("pizza-label");
  if (low.length) {
    label.textContent = `low: ${low.map((t) => t.name || t.id).join(", ")}`;
  } else {
    label.textContent = `pizza trays · ${source} cam`;
  }
}

function onGithub() {
  document.querySelectorAll(".table").forEach((btn) => {
    const pulse = getTeamPulse(btn.dataset.teamId);
    const el = btn.querySelector(".table-pulse");
    if (!el) return;
    el.textContent = pulse ? pulseLabel(pulse) : "repo…";
    el.dataset.status = pulse?.status || "";
  });
  if (selectedId) selectTeam(selectedId);
}

function paintHub() {
  const present = document.getElementById("present");
  if (present) present.textContent = `${hub.teams} / ${hub.pool}`;
}

function onJoins() {
  paintHub();
  renderHackathon();
  renderFloor();
  renderPresentations();
  paintJoinPanel(document.getElementById("join-inbox-body"));
  if (selectedId) selectTeam(selectedId);
}

document.getElementById("event-title").textContent = `${event.name} — ${event.month}`;
document.getElementById("venue").textContent = `${event.venue} · ${event.address}`;
document.getElementById("hub-label").textContent = "teams · pool in the room";

initJoins({ teams, people, youAre, hub });
paintHub();

window.addEventListener("nightboard:pizza", onPizza);
window.addEventListener("nightboard:github", onGithub);
window.addEventListener("nightboard:joins", onJoins);

mountOps(document.getElementById("ops-root"), { announcements, people, teams, demoQueue });
mountPizza(document.getElementById("view-pizza"), pizza);
mountGithubBot(document.getElementById("view-github"), { teams, event });
mountChat(document.getElementById("view-chat"), { teams, youAre, timezone: event.timezone });

renderFloor();
renderHackathon();
renderPresentations();
renderRules();
paintJoinPanel(document.getElementById("join-inbox-body"));
selectTeam(selectedId);
tick();
setInterval(tick, 1000);
setView("hackathon");

document.querySelectorAll('.tabs [role="tab"]').forEach((tab) => {
  tab.addEventListener("click", () => setView(tab.dataset.view));
});

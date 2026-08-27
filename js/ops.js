/**
 * Live-ops rail: announcement ticker, help matching, projector toggle, demo clock.
 *
 * Events consumed:
 *   nightboard:pizza  { trays, low, source }
 *   nightboard:github { pulses: { [teamId]: { status, ... } } }
 *
 * Events produced:
 *   nightboard:demo   { currentTeamId, remainingSec, index }
 *
 * Projector: document.body.classList.toggle("projector")
 * Hide wall-display chrome with [data-hide-on-projector].
 */

const TICKER_MS = 6000;
const DEFAULT_SLOT_MIN = 4;

const STOP = new Set([
  "a",
  "an",
  "the",
  "for",
  "to",
  "of",
  "and",
  "or",
  "who",
  "has",
  "have",
  "someone",
  "that",
  "with",
  "in",
  "on",
  "is",
  "need",
  "open",
  "seat",
]);

const EXPAND = [
  {
    when: /design|figma|ux|ui/,
    add: ["design", "designer", "figma", "ux", "ui", "copy", "critique"],
  },
  {
    when: /front|prototype|css|react|html/,
    add: ["frontend", "front-end", "prototype", "prototypes", "ui"],
  },
  {
    when: /data|postgres|sql|backend|api|parking|database/,
    add: ["data", "postgres", "sql", "backend", "api", "apis", "parking"],
  },
  {
    when: /writ|copy|content/,
    add: ["writer", "copy", "writing", "content", "ux"],
  },
  {
    when: /ml|python|eval/,
    add: ["python", "ml", "eval", "prototype", "prototypes"],
  },
];

function ensureOpsStyles() {
  if (document.querySelector("link[data-ops-css]")) return;
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = new URL("../css/ops.css", import.meta.url).href;
  link.dataset.opsCss = "1";
  document.head.appendChild(link);
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

function tokens(text) {
  return String(text || "")
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((t) => t && !STOP.has(t));
}

function expandNeed(need) {
  const raw = String(need || "").toLowerCase();
  const set = new Set(tokens(raw));
  for (const rule of EXPAND) {
    if (rule.when.test(raw) || [...set].some((t) => rule.when.test(t))) {
      for (const extra of rule.add) set.add(extra);
    }
  }
  return [...set];
}

function includesToken(hay, token) {
  if (!hay || !token) return false;
  return hay.includes(token);
}

/**
 * Best people whose skills/role fuzzy-match a team's need.
 * Designer → Figma / design / copy; frontend → prototypes; data → Postgres; writer → copy.
 * @param {string} need
 * @param {Array<{id:string,name:string,role:string,skills:string[]}>} people
 * @returns {typeof people}
 */
export function matchNeed(need, people) {
  if (!need || !Array.isArray(people) || !people.length) return [];
  const query = expandNeed(need);
  if (!query.length) return [];

  const scored = people.map((person) => {
    const skills = (person.skills || []).map((s) => String(s).toLowerCase());
    const role = String(person.role || "").toLowerCase();
    let score = 0;
    for (const token of query) {
      for (const skill of skills) {
        if (includesToken(skill, token) || includesToken(token, skill)) score += 4;
      }
      if (includesToken(role, token)) score += 3;
    }
    return { person, score };
  });

  return scored
    .filter((row) => row.score > 0)
    .sort((a, b) => b.score - a.score || a.person.name.localeCompare(b.person.name))
    .map((row) => row.person);
}

function teamById(teams, id) {
  return (teams || []).find((t) => t.id === id) || null;
}

function trayLeft(tray) {
  const n = tray?.slices ?? tray?.remaining ?? tray?.n ?? tray?.left;
  return n == null ? 0 : n;
}

function collectLowTrays(detail) {
  const trays = Array.isArray(detail?.trays) ? detail.trays : [];
  const low = detail?.low;
  const out = [];
  const push = (tray) => {
    if (!tray) return;
    const key = tray.id || tray.name;
    if (!key || out.some((t) => (t.id || t.name) === key)) return;
    out.push(tray);
  };

  const lowItems = Array.isArray(low) ? low : low && typeof low === "object" ? [low] : null;
  if (lowItems) {
    for (const item of lowItems) {
      if (typeof item === "string") {
        push(trays.find((t) => t.id === item || t.name === item) || { name: item, slices: 0 });
      } else if (item && typeof item === "object") {
        push(item);
      }
    }
    return out;
  }

  for (const tray of trays) {
    if (tray.low === true || tray.status === "low") push(tray);
  }
  return out;
}

function pizzaAlerts(detail) {
  return collectLowTrays(detail).map((tray) => {
    const name = tray.name || tray.id || "tray";
    const n = trayLeft(tray);
    return {
      id: `pizza-${tray.id || name}`,
      text: `Pizza: ${name} is low (${n} left).`,
    };
  });
}

function githubAlerts(detail, teams) {
  const pulses = detail?.pulses && typeof detail.pulses === "object" ? detail.pulses : {};
  const alerts = [];
  for (const [teamId, pulse] of Object.entries(pulses)) {
    if (pulse?.status !== "stalled") continue;
    const team = teamById(teams, teamId);
    const teamName = team?.name || pulse.teamName || teamId;
    alerts.push({
      id: `gh-${teamId}`,
      text: `${teamName} looks stalled on GitHub.`,
    });
  }
  return alerts;
}

function fmtClock(sec) {
  const s = Math.max(0, Math.floor(sec));
  const m = Math.floor(s / 60);
  const r = s % 60;
  return `${m}:${String(r).padStart(2, "0")}`;
}

function slotMinutes(slot) {
  const n = Number(slot?.minutes);
  return n > 0 ? n : DEFAULT_SLOT_MIN;
}

/**
 * @param {HTMLElement} rootEl
 * @param {{ announcements: Array<{id:string,text:string}>, people: object[], teams: object[], demoQueue: Array<{teamId:string,minutes?:number}> }} data
 */
export function mountOps(rootEl, { announcements, people, teams, demoQueue }) {
  if (!rootEl) return;
  ensureOpsStyles();

  if (typeof rootEl._opsCleanup === "function") rootEl._opsCleanup();

  const base = (announcements || []).map((a) => ({ id: a.id, text: a.text }));
  let livePizza = [];
  let liveGithub = [];
  let tickerIndex = 0;
  let tickerTimer = null;

  const queue = Array.isArray(demoQueue) ? demoQueue : [];
  let demoIndex = 0;
  let remainingSec = 0;
  let demoTimer = null;
  let demoRunning = false;

  rootEl.classList.add("ops");
  rootEl.replaceChildren();

  const bar = el("div", "ops-bar");
  const ticker = el("div", "ops-ticker");
  ticker.setAttribute("role", "status");
  ticker.setAttribute("aria-live", "polite");
  const tickerKicker = el("p", "ops-kicker", "Announce");
  const tickerText = el("p", "ops-ticker-text", "");
  ticker.append(tickerKicker, tickerText);

  const projectorBtn = el("button", "ops-btn", "Projector");
  projectorBtn.type = "button";
  projectorBtn.setAttribute("aria-pressed", document.body.classList.contains("projector") ? "true" : "false");
  projectorBtn.addEventListener("click", () => {
    document.body.classList.toggle("projector");
    const on = document.body.classList.contains("projector");
    projectorBtn.setAttribute("aria-pressed", on ? "true" : "false");
  });

  bar.append(ticker, projectorBtn);

  const grid = el("div", "ops-grid");
  const help = el("section", "ops-panel");
  const helpHead = el("div", "ops-panel-head");
  helpHead.append(el("strong", null, "Walk over"), el("span", "ops-muted", "Open seats"));
  const helpBody = el("div", "ops-panel-body");
  help.append(helpHead, helpBody);

  const demo = el("section", "ops-panel");
  const demoHead = el("div", "ops-panel-head");
  demoHead.append(el("strong", null, "Demo clock"), el("span", "ops-muted", `${DEFAULT_SLOT_MIN} min default`));
  const demoBody = el("div", "ops-panel-body");
  demo.append(demoHead, demoBody);

  grid.append(help, demo);
  rootEl.append(bar, grid);

  function rotation() {
    return [...livePizza, ...liveGithub, ...base];
  }

  function paintTicker() {
    const items = rotation();
    if (!items.length) {
      tickerText.textContent = "No announcements.";
      return;
    }
    tickerIndex = tickerIndex % items.length;
    tickerText.textContent = items[tickerIndex].text;
  }

  function startTicker() {
    if (tickerTimer) clearInterval(tickerTimer);
    paintTicker();
    tickerTimer = setInterval(() => {
      const items = rotation();
      if (!items.length) return;
      tickerIndex = (tickerIndex + 1) % items.length;
      paintTicker();
    }, TICKER_MS);
  }

  function onPizza(ev) {
    livePizza = pizzaAlerts(ev.detail || {});
    tickerIndex = 0;
    paintTicker();
  }

  function onGithub(ev) {
    liveGithub = githubAlerts(ev.detail || {}, teams);
    tickerIndex = 0;
    paintTicker();
  }

  window.addEventListener("nightboard:pizza", onPizza);
  window.addEventListener("nightboard:github", onGithub);

  function paintHelp() {
    helpBody.replaceChildren();
    const needy = (teams || []).filter((t) => t.need);
    if (!needy.length) {
      helpBody.append(el("p", "ops-empty", "No open seats."));
      return;
    }
    for (const team of needy) {
      const row = el("article", "ops-need");
      const meta = el("div", "ops-need-meta");
      meta.append(
        el("strong", null, team.name),
        el("span", "ops-table", team.table || "—"),
        el("span", "ops-need-flag", team.need),
      );
      row.append(meta);
      const matches = matchNeed(team.need, people || []).slice(0, 2);
      if (!matches.length) {
        row.append(el("p", "ops-prompt", "No match in the room — ask from the stage."));
      } else {
        for (const person of matches) {
          const line = el("p", "ops-prompt");
          line.textContent = `${person.name} is a match — table ${team.table}.`;
          row.append(line);
        }
      }
      helpBody.append(row);
    }
  }

  function dispatchDemo() {
    const slot = queue[demoIndex];
    window.dispatchEvent(
      new CustomEvent("nightboard:demo", {
        detail: {
          currentTeamId: slot?.teamId ?? null,
          remainingSec,
          index: demoIndex,
        },
      }),
    );
  }

  const nowKicker = el("p", "ops-kicker", "On deck");
  const nowName = el("p", "ops-demo-name", "—");
  const nowClock = el("p", "ops-demo-clock", "");
  const nextKicker = el("p", "ops-kicker", "Up next");
  const nextName = el("p", "ops-demo-name", "—");
  const nextMeta = el("p", "ops-muted", "");
  const startBtn = el("button", "ops-btn ops-btn-ink", "Start demos");
  startBtn.type = "button";
  startBtn.addEventListener("click", startDemos);

  const nowCard = el("div", "ops-demo-card ops-demo-now");
  nowCard.append(nowKicker, nowName, nowClock);
  const nextCard = el("div", "ops-demo-card");
  nextCard.append(nextKicker, nextName, nextMeta);
  const rail = el("div", "ops-demo-rail");
  rail.append(nowCard, nextCard);
  demoBody.append(rail, startBtn);

  function paintDemo() {
    const current = queue[demoIndex] ? teamById(teams, queue[demoIndex].teamId) : null;
    const next = queue[demoIndex + 1] ? teamById(teams, queue[demoIndex + 1].teamId) : null;
    const done = demoIndex >= queue.length;

    nowKicker.textContent = done ? "Done" : demoRunning ? "Now presenting" : "On deck";
    nowName.textContent = done ? "That’s a wrap" : current?.name || "—";
    nowClock.textContent = done
      ? "Queue clear"
      : demoRunning
        ? fmtClock(remainingSec)
        : `${slotMinutes(queue[demoIndex])} min slot`;

    nextName.textContent = done ? "—" : next?.name || "Last in queue";
    nextMeta.textContent = next && !done ? `Table ${next.table}` : current && !done ? `Table ${current.table}` : "";

    startBtn.textContent = demoRunning ? "Running" : "Start demos";
    startBtn.disabled = demoRunning || done || !queue.length;
  }

  function stopDemoTimer() {
    if (demoTimer) {
      clearInterval(demoTimer);
      demoTimer = null;
    }
  }

  function startDemos() {
    if (demoRunning || !queue.length || demoIndex >= queue.length) return;
    demoRunning = true;
    remainingSec = slotMinutes(queue[demoIndex]) * 60;
    dispatchDemo();
    paintDemo();
    stopDemoTimer();
    demoTimer = setInterval(() => {
      remainingSec -= 1;
      if (remainingSec <= 0) {
        remainingSec = 0;
        dispatchDemo();
        demoIndex += 1;
        if (demoIndex >= queue.length) {
          stopDemoTimer();
          demoRunning = false;
          remainingSec = 0;
          dispatchDemo();
          paintDemo();
          return;
        }
        remainingSec = slotMinutes(queue[demoIndex]) * 60;
      }
      dispatchDemo();
      paintDemo();
    }, 1000);
  }

  paintHelp();
  paintDemo();
  startTicker();

  rootEl._opsCleanup = () => {
    if (tickerTimer) clearInterval(tickerTimer);
    stopDemoTimer();
    window.removeEventListener("nightboard:pizza", onPizza);
    window.removeEventListener("nightboard:github", onGithub);
    rootEl._opsCleanup = null;
  };
}

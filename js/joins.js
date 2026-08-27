/**
 * Team join requests and seat offers. Persists to localStorage (`nightboard-joins`).
 * Events: nightboard:joins after any accept / decline / request / offer.
 */

export const STORAGE_KEY = "nightboard-joins";

let ctx = {
  teams: [],
  people: [],
  youAre: "",
  hub: null,
  requests: [],
};

function ensureStyles() {
  if (document.querySelector("link[data-joins-css]")) return;
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = new URL("../css/joins.css", import.meta.url).href;
  link.dataset.joinsCss = "1";
  document.head.appendChild(link);
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

export function memberName(m) {
  return typeof m === "string" ? m : m?.name || "";
}

function memberRole(m) {
  return typeof m === "string" ? "member" : m?.role || "member";
}

function asMember(name, role = "member") {
  return { name, role };
}

function cloneMembers(list) {
  return (list || []).map((m) => asMember(memberName(m), memberRole(m)));
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
  const members = {};
  for (const team of ctx.teams) {
    members[team.id] = cloneMembers(team.members);
  }
  const payload = {
    v: 1,
    requests: ctx.requests,
    members,
    unattached: ctx.people.map((p) => ({ ...p })),
  };
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
  } catch {
    /* ignore quota */
  }
}

function uid(prefix) {
  return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`;
}

function teamById(id) {
  return ctx.teams.find((t) => t.id === id) || null;
}

export function youTeamId() {
  if (ctx.youAre && teamById(ctx.youAre)) return ctx.youAre;
  const mine = ctx.teams.find((t) => t.mine);
  return mine?.id || ctx.teams[0]?.id || "";
}

export function youPersonName() {
  const team = teamById(youTeamId());
  if (!team) return "";
  const samuel = team.members.find((m) => memberName(m) === "Samuel Collins");
  if (samuel) return memberName(samuel);
  return memberName(team.members[0]) || "";
}

function currentTeamIdFor(person) {
  const team = ctx.teams.find((t) => t.members.some((m) => memberName(m) === person));
  return team?.id || null;
}

function pending() {
  return ctx.requests.filter((r) => r.status === "pending");
}

export function inboundFor(teamId) {
  return pending().filter((r) => r.kind === "join" && r.teamId === teamId);
}

export function inboundOffersFor(person) {
  return pending().filter((r) => r.kind === "offer" && r.person === person);
}

function outboundOffersFrom(teamId) {
  return pending().filter((r) => r.kind === "offer" && r.fromTeamId === teamId);
}

function hasPendingJoin(person, teamId) {
  return pending().some((r) => r.kind === "join" && r.person === person && r.teamId === teamId);
}

function hasPendingOffer(fromTeamId, person) {
  return pending().some((r) => r.kind === "offer" && r.fromTeamId === fromTeamId && r.person === person);
}

function seedRequests() {
  const target = youTeamId();
  const out = [];
  const used = new Set();
  const pushSolo = (team) => {
    if (!team || team.id === target || team.members.length !== 1) return;
    const person = memberName(team.members[0]);
    if (!person || used.has(person)) return;
    used.add(person);
    out.push({
      id: `seed-join-${team.id}`,
      kind: "join",
      person,
      teamId: target,
      fromTeamId: team.id,
      status: "pending",
    });
  };

  pushSolo(teamById("jordan"));
  pushSolo(teamById("max-hum"));
  if (out.length < 2) {
    for (const team of ctx.teams) {
      pushSolo(team);
      if (out.length >= 2) break;
    }
  }

  const firstPerson = ctx.people[0];
  if (firstPerson && target) {
    out.push({
      id: `seed-offer-${firstPerson.id || "p0"}`,
      kind: "offer",
      person: firstPerson.name,
      teamId: target,
      fromTeamId: target,
      status: "pending",
    });
  }
  return out;
}

function applyStored(stored) {
  if (stored.members && typeof stored.members === "object") {
    for (const team of ctx.teams) {
      if (Object.prototype.hasOwnProperty.call(stored.members, team.id) && Array.isArray(stored.members[team.id])) {
        team.members.splice(0, team.members.length, ...cloneMembers(stored.members[team.id]));
      }
    }
  }
  if (Array.isArray(stored.unattached)) {
    ctx.people.splice(0, ctx.people.length, ...stored.unattached.map((p) => ({ ...p })));
  }
  ctx.requests = Array.isArray(stored.requests)
    ? stored.requests.filter((r) => r && r.status === "pending")
    : [];
}

export function refreshHubCounts() {
  if (!ctx.hub) return;
  ctx.hub.teams = ctx.teams.filter((t) => t.members.length > 0).length;
  ctx.hub.people = ctx.teams.reduce((n, t) => n + t.members.length, 0) + ctx.people.length;
}

function toast(message) {
  let node = document.getElementById("join-toast");
  if (!node) {
    node = el("div", "join-toast");
    node.id = "join-toast";
    node.setAttribute("role", "status");
    document.body.appendChild(node);
  }
  node.textContent = message;
  node.dataset.show = "true";
  clearTimeout(toast._t);
  toast._t = setTimeout(() => {
    node.dataset.show = "false";
  }, 2800);
}

function commit(message) {
  save();
  refreshHubCounts();
  if (message) toast(message);
  window.dispatchEvent(new CustomEvent("nightboard:joins"));
}

function removePersonFromEverywhere(person) {
  for (const team of ctx.teams) {
    const next = team.members.filter((m) => memberName(m) !== person);
    if (next.length !== team.members.length) team.members.splice(0, team.members.length, ...next);
  }
  const idx = ctx.people.findIndex((p) => p.name === person);
  if (idx >= 0) ctx.people.splice(idx, 1);
}

function movePersonToTeam(person, teamId) {
  const dest = teamById(teamId);
  if (!dest) return false;
  if (dest.members.some((m) => memberName(m) === person)) return true;
  removePersonFromEverywhere(person);
  dest.members.push(asMember(person, "member"));
  return true;
}

function dropRequestsFor(person, exceptId) {
  ctx.requests = ctx.requests.filter((r) => r.id === exceptId || r.person !== person);
}

export function requestJoin(teamId) {
  const person = youPersonName();
  const mine = youTeamId();
  const dest = teamById(teamId);
  if (!person || !dest) return;
  if (dest.id === mine || dest.mine) {
    toast("That’s your team.");
    return;
  }
  if (dest.members.some((m) => memberName(m) === person)) {
    toast("You’re already on this team.");
    return;
  }
  if (hasPendingJoin(person, teamId)) {
    toast("Request already pending.");
    return;
  }
  ctx.requests.push({
    id: uid("join"),
    kind: "join",
    person,
    teamId,
    fromTeamId: currentTeamIdFor(person),
    status: "pending",
  });
  commit(`Requested to join ${dest.name}.`);
}

export function offerSeat(person, personTeamId) {
  const from = youTeamId();
  const fromTeam = teamById(from);
  if (!from || !person || !fromTeam) return;
  if (fromTeam.members.some((m) => memberName(m) === person)) {
    toast("Already on your team.");
    return;
  }
  if (hasPendingOffer(from, person)) {
    toast("Offer already pending.");
    return;
  }
  ctx.requests.push({
    id: uid("offer"),
    kind: "offer",
    person,
    teamId: from,
    fromTeamId: from,
    personTeamId: personTeamId || currentTeamIdFor(person),
    status: "pending",
  });
  commit(`Offered a seat to ${person}.`);
}

export function acceptRequest(id) {
  const req = ctx.requests.find((r) => r.id === id && r.status === "pending");
  if (!req) return;
  const dest = teamById(req.teamId);
  if (!dest) {
    ctx.requests = ctx.requests.filter((r) => r.id !== id);
    commit("That team is gone.");
    return;
  }
  const ok = movePersonToTeam(req.person, req.teamId);
  if (!ok) return;
  dropRequestsFor(req.person, id);
  ctx.requests = ctx.requests.filter((r) => r.id !== id);
  commit(`${req.person} joined ${dest.name}.`);
}

export function declineRequest(id) {
  const req = ctx.requests.find((r) => r.id === id && r.status === "pending");
  if (!req) return;
  ctx.requests = ctx.requests.filter((r) => r.id !== id);
  const label = req.kind === "offer" ? `Declined offer for ${req.person}.` : `Declined ${req.person}.`;
  commit(label);
}

function joinButtonState(team) {
  const person = youPersonName();
  const mine = youTeamId();
  if (!team || !person) return { show: false };
  if (team.id === mine || team.mine) return { show: true, disabled: true, label: "Your team" };
  if (team.members.some((m) => memberName(m) === person)) {
    return { show: true, disabled: true, label: "Already a member" };
  }
  if (hasPendingJoin(person, team.id)) {
    return { show: true, disabled: true, label: "Requested" };
  }
  return { show: true, disabled: false, label: "Request to join" };
}

function offerTarget(team) {
  if (!team || team.id === youTeamId() || team.mine) return null;
  if (team.members.length !== 1) return null;
  return memberName(team.members[0]);
}

function stopCardClick(node) {
  node.addEventListener("click", (ev) => ev.stopPropagation());
}

function acceptBtn(id, label) {
  const btn = el("button", "join-btn join-btn-accept", label || "Accept");
  btn.type = "button";
  btn.addEventListener("click", (ev) => {
    ev.stopPropagation();
    acceptRequest(id);
  });
  return btn;
}

function declineBtn(id, label) {
  const btn = el("button", "join-btn join-btn-decline", label || "Decline");
  btn.type = "button";
  btn.addEventListener("click", (ev) => {
    ev.stopPropagation();
    declineRequest(id);
  });
  return btn;
}

function requestRow(req, extra) {
  const dest = teamById(req.teamId);
  const from = req.fromTeamId ? teamById(req.fromTeamId) : null;
  const row = el("div", "join-row");
  const copy = el("div", "join-row-copy");
  const title = el("strong", null, req.person);
  const meta = el("p", "join-muted");
  if (req.kind === "join") {
    meta.textContent = from
      ? `wants to join ${dest?.name || "this team"} · now on ${from.name}`
      : `solo · wants to join ${dest?.name || "this team"}`;
  } else {
    meta.textContent = `${dest?.name || "A team"} offered a seat`;
  }
  copy.append(title, meta);
  const actions = el("div", "join-row-actions");
  actions.append(acceptBtn(req.id), declineBtn(req.id));
  row.append(copy, actions);
  if (extra) row.append(extra);
  return row;
}

export function paintInbound(root, teamId) {
  if (!root) return;
  const id = teamId || youTeamId();
  const joins = inboundFor(id);
  const offersToYou = inboundOffersFor(youPersonName());
  const items = [...joins, ...offersToYou];

  const wrap = el("div", "join-inbox");
  wrap.append(el("p", "join-kicker", "Join requests"));
  if (!items.length) {
    wrap.append(el("p", "join-muted", "No pending requests."));
  } else {
    for (const req of items) wrap.append(requestRow(req));
  }
  root.append(wrap);
  return wrap;
}

export function paintJoinPanel(root) {
  if (!root) return;
  root.replaceChildren();
  paintInbound(root, youTeamId());

  const sent = outboundOffersFrom(youTeamId());
  const sentWrap = el("div", "join-inbox");
  sentWrap.append(el("p", "join-kicker", "Offers you sent"));
  if (!sent.length) {
    sentWrap.append(el("p", "join-muted", "No outbound offers."));
  } else {
    for (const req of sent) {
      const destName = req.person;
      const row = el("div", "join-row");
      const copy = el("div", "join-row-copy");
      copy.append(el("strong", null, destName), el("p", "join-muted", "Pending — they can accept or decline"));
      row.append(copy);
      sentWrap.append(row);
    }
  }
  root.append(sentWrap);

  const peopleWrap = el("div", "join-inbox");
  peopleWrap.append(el("p", "join-kicker", "Unattached"));
  if (!ctx.people.length) {
    peopleWrap.append(el("p", "join-muted", "Everyone is on a team."));
  } else {
    for (const person of ctx.people) {
      peopleWrap.append(unattachedRow(person));
    }
  }
  root.append(peopleWrap);

  const count = document.getElementById("join-inbox-count");
  if (count) {
    const n = inboundFor(youTeamId()).length + inboundOffersFor(youPersonName()).length;
    count.textContent = n ? `${n} pending` : "none";
  }
}

function unattachedRow(person) {
  const row = el("div", "join-row");
  const copy = el("div", "join-row-copy");
  copy.append(el("strong", null, person.name));
  const looking = el("p", "join-muted", person.lookingFor || person.role || "Unattached");
  copy.append(looking);
  const actions = el("div", "join-row-actions");
  const offer = inboundOffersFor(person.name)[0];
  if (offer) {
    actions.append(acceptBtn(offer.id), declineBtn(offer.id));
  } else if (hasPendingOffer(youTeamId(), person.name)) {
    const pendingLabel = el("span", "join-muted", "Offered");
    actions.append(pendingLabel);
  } else {
    const btn = el("button", "join-btn", "Offer a seat");
    btn.type = "button";
    btn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      offerSeat(person.name, null);
    });
    actions.append(btn);
  }
  row.append(copy, actions);
  return row;
}

export function paintTeamActions(root, team) {
  if (!root || !team) return;
  const box = el("div", "join-actions");
  stopCardClick(box);

  const state = joinButtonState(team);
  if (state.show) {
    const btn = el("button", "join-btn", state.label);
    btn.type = "button";
    btn.disabled = state.disabled;
    if (!state.disabled) {
      btn.addEventListener("click", (ev) => {
        ev.stopPropagation();
        requestJoin(team.id);
      });
    }
    box.append(btn);
  }

  const target = offerTarget(team);
  if (target) {
    const offered = hasPendingOffer(youTeamId(), target);
    const incoming = inboundOffersFor(target).filter((r) => r.fromTeamId === youTeamId());
    if (incoming.length) {
      box.append(el("p", "join-muted", `Pending offer to ${target}`));
      const row = el("div", "join-row-actions");
      row.append(acceptBtn(incoming[0].id), declineBtn(incoming[0].id));
      box.append(row);
    } else {
      const btn = el("button", "join-btn", offered ? "Offer pending" : "Offer a seat");
      btn.type = "button";
      btn.disabled = offered;
      if (!offered) {
        btn.addEventListener("click", (ev) => {
          ev.stopPropagation();
          offerSeat(target, team.id);
        });
      }
      box.append(btn);
    }
  }

  root.append(box);

  if (team.id === youTeamId() || team.mine) {
    paintInbound(root, team.id);
  }
}

/**
 * @param {{ teams: object[], people: object[], youAre: string, hub?: object }} options
 */
export function initJoins({ teams, people, youAre, hub }) {
  ctx.teams = teams;
  ctx.people = people;
  ctx.youAre = youAre;
  ctx.hub = hub || null;
  ensureStyles();

  const stored = loadRaw();
  if (stored) {
    applyStored(stored);
  } else {
    ctx.requests = seedRequests();
    save();
  }
  refreshHubCounts();
}

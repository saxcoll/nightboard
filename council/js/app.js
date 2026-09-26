import { BOOKS } from "./catalog.js";
import { briefFromParams, briefToQuery, convene, DEFAULT_READER, formatHours, hoursFor, yearLabel } from "./convene.js";
import { CHAIR, MEMBERS } from "./members.js";

const form = document.querySelector("#brief");
const shelf = document.querySelector("#shelf");
const shelfMeta = document.querySelector("#shelf-meta");
const modeHint = document.querySelector("#mode-hint");
const minutes = document.querySelector("#minutes");
const filterInput = document.querySelector("#filter");
const rulesList = document.querySelector("#rules-list");

const params = new URLSearchParams(location.search);
const fromUrl = [...params.keys()].some((key) => ["hours", "kind", "theme", "recent", "mode", "shortlist"].includes(key));
const initial = fromUrl ? briefFromParams(params, DEFAULT_READER) : { reader: structuredClone(DEFAULT_READER), mode: "shelf" };

const state = {
  hours: initial.reader.hours,
  kind: initial.reader.kind,
  theme: initial.reader.theme,
  mode: initial.mode,
  recent: new Set(initial.reader.recent),
  shortlist: new Set(initial.mode === "shortlist" ? initial.reader.shortlist : []),
  query: "",
};

document.querySelector("#hours").value = String(state.hours);
document.querySelector("#theme").value = state.theme;
document.querySelector(`input[name="kind"][value="${state.kind}"]`).checked = true;
document.querySelector(`input[name="mode"][value="${state.mode}"]`).checked = true;

for (const member of [...MEMBERS, CHAIR]) {
  const term = document.createElement("dt");
  term.textContent = member.name;
  const detail = document.createElement("dd");
  detail.textContent = member.rule;
  rulesList.append(term, detail);
}

function readerFromState() {
  return {
    hours: state.hours,
    kind: state.kind,
    theme: state.theme.trim(),
    recent: [...state.recent],
    shortlist: state.mode === "shortlist" ? [...state.shortlist] : [],
  };
}

function syncUrl() {
  const query = briefToQuery(readerFromState(), state.mode);
  history.replaceState(null, "", `${location.pathname}?${query}`);
}

function renderShelf() {
  shelf.replaceChildren();
  const query = state.query.trim().toLowerCase();
  const groups = [
    ["Fiction", BOOKS.filter((book) => book.kind === "fiction")],
    ["Nonfiction", BOOKS.filter((book) => book.kind === "nonfiction")],
  ];
  let shown = 0;
  for (const [label, books] of groups) {
    const visible = books.filter((book) => {
      if (!query) return true;
      const hay = [book.title, book.author, book.blurb, ...book.subjects, ...book.keywords].join(" ").toLowerCase();
      return hay.includes(query);
    });
    if (!visible.length) continue;
    const heading = document.createElement("p");
    heading.className = "group-label";
    heading.textContent = label;
    shelf.append(heading);
    for (const book of visible) {
      shown += 1;
      shelf.append(bookRow(book));
    }
  }
  if (!shown) {
    const empty = document.createElement("p");
    empty.className = "hint";
    empty.textContent = "No book on the shelf matches that search.";
    shelf.append(empty);
  }
  const lately = state.recent.size;
  const listed = state.shortlist.size;
  shelfMeta.textContent =
    state.mode === "shortlist"
      ? `${lately} marked lately read. ${listed} on the shortlist.`
      : `${lately} marked lately read. The open shelf ignores the shortlist.`;
  modeHint.textContent =
    state.mode === "shortlist"
      ? "The council may nominate only from the shortlist, skipping anything marked lately read."
      : "The council may nominate any book not marked lately read. If the shelf can answer every word of the theme, the Chair strikes a nomination that misses it.";
}

function bookRow(book) {
  const row = document.createElement("div");
  row.className = "book";
  const toggles = document.createElement("div");
  toggles.className = "toggles";
  toggles.append(
    toggle("Lately", state.recent.has(book.id), () => {
      if (state.recent.has(book.id)) state.recent.delete(book.id);
      else state.recent.add(book.id);
      renderShelf();
    }),
  );
  if (state.mode === "shortlist") {
    toggles.append(
      toggle("List", state.shortlist.has(book.id), () => {
        if (state.shortlist.has(book.id)) state.shortlist.delete(book.id);
        else state.shortlist.add(book.id);
        renderShelf();
      }),
    );
  }
  const body = document.createElement("div");
  const title = document.createElement("h3");
  title.textContent = book.title;
  const meta = document.createElement("p");
  meta.textContent = `${book.author} · ${yearLabel(book.year)} · about ${book.pages} pages · ~${formatHours(hoursFor(book))} h`;
  body.append(title, meta);
  row.append(toggles, body);
  return row;
}

function toggle(label, pressed, onClick) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "toggle";
  button.textContent = label;
  button.setAttribute("aria-pressed", pressed ? "true" : "false");
  button.addEventListener("click", onClick);
  return button;
}

function citeText(text) {
  const titles = BOOKS.map((book) => book.title).sort((a, b) => b.length - a.length);
  const fragment = document.createDocumentFragment();
  let rest = text;
  while (rest) {
    let at = -1;
    let match = "";
    for (const title of titles) {
      const found = rest.indexOf(title);
      if (found !== -1 && (at === -1 || found < at)) {
        at = found;
        match = title;
      }
    }
    if (at === -1) {
      fragment.append(document.createTextNode(rest));
      break;
    }
    if (at > 0) fragment.append(document.createTextNode(rest.slice(0, at)));
    const cite = document.createElement("cite");
    cite.textContent = match;
    fragment.append(cite);
    rest = rest.slice(at + match.length);
  }
  return fragment;
}

function strikeLabel(nomination) {
  if (nomination.strikeReason === "theme") {
    return `First named ${nomination.ideal.title}. Struck: it does not answer the theme.`;
  }
  if (nomination.strikeReason === "both") {
    return `First named ${nomination.ideal.title}. Struck: it misses the theme and the hours.`;
  }
  return `First named ${nomination.ideal.title}. Struck: it does not fit the hours.`;
}

function paragraph(className, text) {
  const node = document.createElement("p");
  node.className = className;
  node.append(citeText(text));
  return node;
}

let lastSession = null;

function renderMinutes(session) {
  lastSession = session;
  minutes.replaceChildren();
  const head = document.createElement("div");
  head.className = "minutes-head";
  const title = document.createElement("h2");
  title.textContent = "Minutes";
  const copy = document.createElement("button");
  copy.type = "button";
  copy.className = "copy";
  copy.textContent = "Copy minutes";
  copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(session.text);
      copy.textContent = "Copied";
    } catch {
      copy.textContent = "Copy failed";
    }
    setTimeout(() => {
      copy.textContent = "Copy minutes";
    }, 1600);
  });
  head.append(title, copy);
  minutes.append(head);

  const kindLabel =
    session.reader.kind === "either"
      ? "Fiction or nonfiction"
      : session.reader.kind === "fiction"
        ? "Fiction only"
        : "Nonfiction only";
  const brief = document.createElement("p");
  brief.className = "brief-line";
  const theme = session.reader.theme ? ` Theme: ${session.reader.theme}.` : "";
  const recentTitles = session.reader.recent
    .map((id) => BOOKS.find((book) => book.id === id)?.title)
    .filter(Boolean);
  brief.textContent = `${kindLabel}. ${formatHours(session.reader.hours)} hours.${theme} ${
    recentTitles.length ? `Just read: ${recentTitles.join("; ")}.` : "Nothing marked as just read."
  }`;
  minutes.append(brief);

  if (!session.winner) {
    minutes.append(paragraph("empty-note", session.resolution));
    return;
  }

  heading("Nominations");
  for (const nomination of session.nominations) {
    const member = MEMBERS.find((entry) => entry.id === nomination.memberId);
    const who = document.createElement("p");
    who.className = "member";
    who.textContent = `${member.name} nominates ${nomination.standing.title}`;
    minutes.append(who);
    if (nomination.struck) {
      const struck = document.createElement("p");
      struck.className = "struck";
      struck.textContent = strikeLabel(nomination);
      minutes.append(struck);
    }
    minutes.append(paragraph("speech", nomination.speech));
  }

  if (session.objections.length) {
    heading("Objections");
    for (const objection of session.objections) {
      const member = MEMBERS.find((entry) => entry.id === objection.memberId);
      const who = document.createElement("p");
      who.className = "member";
      who.textContent = objection.called
        ? `${member.short}, against ${objection.book.title}, called because nobody had spoken against it`
        : `${member.short}, against ${objection.book.title}`;
      minutes.append(who, paragraph("speech", objection.speech));
    }
  }

  if (session.reservations.length) {
    heading("Reservations");
    for (const reservation of session.reservations) {
      const member = MEMBERS.find((entry) => entry.id === reservation.memberId);
      const who = document.createElement("p");
      who.className = "member";
      who.textContent = member.short;
      minutes.append(who, paragraph("speech", reservation.speech));
    }
  }

  if (session.replies.length) {
    heading("Replies");
    for (const reply of session.replies) {
      const member = MEMBERS.find((entry) => entry.id === reply.memberId);
      const objector = MEMBERS.find((entry) => entry.id === reply.objectorId);
      const who = document.createElement("p");
      who.className = "member";
      who.textContent = `${member.short} answers ${objector.short}`;
      minutes.append(who, paragraph("speech", reply.speech));
    }
  }

  heading("The vote");
  const table = document.createElement("table");
  const tableHead = document.createElement("thead");
  const headRow = document.createElement("tr");
  for (const label of ["Member", "Ranking"]) {
    const cell = document.createElement("th");
    cell.textContent = label;
    headRow.append(cell);
  }
  tableHead.append(headRow);
  const body = document.createElement("tbody");
  for (const ballot of session.ballots) {
    const row = document.createElement("tr");
    const memberCell = document.createElement("td");
    memberCell.textContent = MEMBERS.find((entry) => entry.id === ballot.memberId).short;
    const rankCell = document.createElement("td");
    rankCell.textContent = ballot.ranking
      .map((id, index) => {
        const book = session.tally.find((entry) => entry.book.id === id).book;
        return `${index + 1}. ${book.title}`;
      })
      .join("  ");
    row.append(memberCell, rankCell);
    body.append(row);
  }
  for (const row of session.tally) {
    const tr = document.createElement("tr");
    const label = document.createElement("td");
    label.textContent = "Count";
    const value = document.createElement("td");
    value.textContent = `${row.book.title}: ${row.points} points, ${row.firsts} first-place ${row.firsts === 1 ? "vote" : "votes"}`;
    tr.append(label, value);
    body.append(tr);
  }
  table.append(tableHead, body);
  minutes.append(table);

  const resolution = document.createElement("div");
  resolution.className = "resolution";
  const resolutionHeading = document.createElement("h3");
  resolutionHeading.className = "section";
  resolutionHeading.textContent = "Resolution";
  resolution.append(resolutionHeading, paragraph("speech", session.resolution));
  minutes.append(resolution);

  if (session.dissent) {
    const dissent = document.createElement("div");
    dissent.className = "dissent";
    const dissentHeading = document.createElement("h3");
    dissentHeading.textContent = "Dissent";
    dissent.append(dissentHeading, paragraph("speech", session.dissent.speech));
    minutes.append(dissent);
  }

  const pace = document.createElement("p");
  pace.className = "pace";
  pace.textContent =
    "Pace: 55 pages an hour, minus 8 for each step of difficulty. A book may stand when it finishes inside the brief plus a tenth. The Chair does not vote.";
  minutes.append(pace);
}

function heading(text) {
  const node = document.createElement("h3");
  node.className = "section";
  node.textContent = text;
  minutes.append(node);
}

function readForm() {
  state.hours = Number(document.querySelector("#hours").value);
  state.kind = document.querySelector('input[name="kind"]:checked').value;
  state.theme = document.querySelector("#theme").value;
  state.mode = document.querySelector('input[name="mode"]:checked').value;
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  readForm();
  if (!Number.isFinite(state.hours) || state.hours < 1) {
    minutes.replaceChildren();
    const note = document.createElement("p");
    note.className = "empty-note";
    note.textContent = "The brief needs at least one hour.";
    minutes.append(note);
    return;
  }
  syncUrl();
  renderMinutes(convene({ reader: readerFromState() }));
});

form.addEventListener("change", (event) => {
  if (event.target.name === "mode") {
    state.mode = event.target.value;
    renderShelf();
  }
});

filterInput.addEventListener("input", () => {
  state.query = filterInput.value;
  renderShelf();
});

filterInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") event.preventDefault();
});

renderShelf();
renderMinutes(convene({ reader: readerFromState() }));
syncUrl();

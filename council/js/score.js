const STOP = new Set([
  "the",
  "and",
  "for",
  "with",
  "about",
  "from",
  "that",
  "this",
  "book",
  "read",
  "something",
  "into",
  "over",
]);

export function readingRate(book) {
  return 55 - 8 * book.difficulty;
}

export function hoursFor(book) {
  return book.pages / readingRate(book);
}

export function fitsIn(book, reader) {
  const needed = Math.round(hoursFor(book) * 10) / 10;
  const allowed = Math.round(reader.hours * 1.1 * 10) / 10;
  return needed <= allowed;
}

export function formatHours(hours) {
  const rounded = Math.round(hours * 10) / 10;
  return Number.isInteger(rounded) ? String(rounded) : rounded.toFixed(1);
}

export function yearLabel(year) {
  if (year < 0) return `${-year} BCE`;
  return String(year);
}

export function tokenize(theme) {
  return String(theme || "")
    .toLowerCase()
    .split(/[^a-z0-9+]+/)
    .filter((token) => token.length > 2 && !STOP.has(token));
}

function themeHaystack(book) {
  return [book.title, book.author, book.blurb, ...book.subjects, ...book.keywords].join(" ").toLowerCase();
}

export function themeHits(book, theme) {
  const tokens = tokenize(theme);
  if (!tokens.length) return { tokens, hits: 0 };
  const haystack = themeHaystack(book);
  const hits = tokens.filter((token) => haystack.includes(token)).length;
  return { tokens, hits };
}

export function answersTheme(book, theme) {
  const { tokens, hits } = themeHits(book, theme);
  if (!tokens.length) return true;
  return hits === tokens.length;
}

export function themeBoost(book, theme) {
  const { tokens, hits } = themeHits(book, theme);
  if (!tokens.length) return 0;
  return (hits / tokens.length) * 3;
}

export function jaccard(left, right) {
  const a = left instanceof Set ? left : new Set(left);
  const b = right instanceof Set ? right : new Set(right);
  if (!a.size && !b.size) return 0;
  let intersection = 0;
  for (const item of a) {
    if (b.has(item)) intersection += 1;
  }
  const union = a.size + b.size - intersection;
  return union === 0 ? 0 : intersection / union;
}

export function buildContext(books, reader) {
  const byId = new Map(books.map((entry) => [entry.id, entry]));
  const recentBooks = (reader.recent || []).map((id) => byId.get(id)).filter(Boolean);
  const recentSubjects = new Set(recentBooks.flatMap((entry) => entry.subjects));
  const recentKinds = new Set(recentBooks.map((entry) => entry.kind));
  const recentAuthors = new Set(recentBooks.map((entry) => entry.author));
  const dependents = new Map(books.map((entry) => [entry.id, []]));
  for (const entry of books) {
    for (const id of entry.helpfulBefore || []) {
      if (dependents.has(id)) dependents.get(id).push(entry);
    }
  }
  return { byId, recentBooks, recentSubjects, recentKinds, recentAuthors, dependents, books };
}

export function eligibleBooks(books, reader) {
  const recent = new Set(reader.recent || []);
  let pool = books.filter((entry) => !recent.has(entry.id));
  if (reader.kind === "fiction" || reader.kind === "nonfiction") {
    pool = pool.filter((entry) => entry.kind === reader.kind);
  }
  if (reader.shortlist && reader.shortlist.length) {
    const allowed = new Set(reader.shortlist);
    pool = pool.filter((entry) => allowed.has(entry.id));
  }
  return pool;
}

function roundScore(total) {
  return Math.round(total * 1000) / 1000;
}

export function scoreBook(memberId, book, reader, context) {
  const theme = themeBoost(book, reader.theme);
  if (memberId === "canon") {
    const dependents = (context.dependents.get(book.id) || []).length;
    const sequenced = (book.helpfulBefore || []).some((id) => (reader.recent || []).includes(id));
    let total = book.influence * 2.4 + dependents * 0.8 + book.pleasures * 0.15 + theme;
    if (sequenced) total += 3.2;
    if (book.year > 2000) total -= 0.4;
    return roundScore(total);
  }
  if (memberId === "pleasure") {
    const total =
      book.pleasures * 3 +
      book.finishability * 2 -
      book.difficulty * 0.8 -
      (book.hype || 0) * 0.25 +
      theme * 0.6;
    return roundScore(total);
  }
  if (memberId === "opposition") {
    const overlap = jaccard(book.subjects, context.recentSubjects);
    let total = (1 - overlap) * 4 + book.challenge - book.influence * 0.15 + theme * 0.4;
    if (context.recentKinds.size === 1 && !context.recentKinds.has(book.kind)) total += 3;
    if (context.recentAuthors.has(book.author)) total -= 2.5;
    return roundScore(total);
  }
  if (memberId === "time") {
    const hours = hoursFor(book);
    const value =
      book.influence * 1.1 + book.pleasures + book.finishability * 0.5 - (book.hype || 0) * 0.4;
    if (!fitsIn(book, reader)) return roundScore(-20 + value / Math.max(hours, 0.5));
    const utilization = hours / reader.hours;
    return roundScore(value - 2.8 * Math.abs(utilization - 0.65) + theme * 1.2);
  }
  if (memberId === "doubt") {
    let total = book.influence * 0.8 + book.pleasures + book.finishability * 0.3 + theme * 1.3;
    total -= (book.hype || 0) * 1.6;
    if (book.doubtNote) total -= 2.2;
    return roundScore(total);
  }
  throw new Error(`Unknown member ${memberId}`);
}

export function compareBooks(memberId, a, b, reader, context) {
  const delta = scoreBook(memberId, b, reader, context) - scoreBook(memberId, a, reader, context);
  if (delta !== 0) return delta;
  if (memberId === "pleasure") {
    if (a.kind !== b.kind) return a.kind === "fiction" ? -1 : 1;
    return a.pages - b.pages || a.title.localeCompare(b.title);
  }
  if (memberId === "canon") {
    const dependentsA = (context.dependents.get(a.id) || []).length;
    const dependentsB = (context.dependents.get(b.id) || []).length;
    if (dependentsA !== dependentsB) return dependentsB - dependentsA;
    return a.year - b.year || a.title.localeCompare(b.title);
  }
  if (memberId === "time") {
    const gap = (book) => Math.abs(hoursFor(book) / reader.hours - 0.65);
    const deltaGap = gap(a) - gap(b);
    if (deltaGap !== 0) return deltaGap;
  }
  return a.title.localeCompare(b.title);
}

export function rankBooks(memberId, books, reader, context) {
  return [...books].sort((a, b) => compareBooks(memberId, a, b, reader, context));
}

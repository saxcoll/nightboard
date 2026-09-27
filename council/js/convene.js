import { BOOKS } from "./catalog.js";
import { CHAIR, MEMBERS, memberById } from "./members.js";
import {
  answersTheme,
  buildContext,
  compareBooks,
  eligibleBooks,
  fitsIn,
  formatHours,
  hoursFor,
  rankBooks,
  scoreBook,
  themeBoost,
  tokenize,
  yearLabel,
} from "./score.js";
import { nominationSpeech, objectionSpeech, replySpeech, reservationSpeech } from "./speak.js";

export { fitsIn, formatHours, hoursFor, yearLabel };

export const DEFAULT_READER = {
  hours: 10,
  kind: "either",
  theme: "",
  recent: ["meditations", "enquiry-hume", "groundwork"],
  shortlist: [],
};

export function borda(ballots) {
  const points = new Map();
  const firsts = new Map();
  for (const ballot of ballots) {
    ballot.ranking.forEach((id, index) => {
      const award = ballot.ranking.length - index;
      points.set(id, (points.get(id) || 0) + award);
      if (index === 0) firsts.set(id, (firsts.get(id) || 0) + 1);
    });
  }
  return { points, firsts };
}

function listTitles(books) {
  if (!books.length) return "";
  if (books.length === 1) return books[0].title;
  if (books.length === 2) return `${books[0].title} and ${books[1].title}`;
  return `${books
    .slice(0, -1)
    .map((book) => book.title)
    .join(", ")}, and ${books.at(-1).title}`;
}

function ordinal(n) {
  const names = ["first", "second", "third", "fourth", "fifth", "sixth"];
  return names[n] || String(n + 1);
}

export function briefFromParams(params, fallback = DEFAULT_READER) {
  const hours = Number(params.get("hours"));
  const kind = params.get("kind");
  const mode = params.get("mode") === "shortlist" ? "shortlist" : "shelf";
  const recent = String(params.get("recent") || "")
    .split(",")
    .map((id) => id.trim())
    .filter(Boolean);
  const shortlist = String(params.get("shortlist") || "")
    .split(",")
    .map((id) => id.trim())
    .filter(Boolean);
  return {
    reader: {
      hours: Number.isFinite(hours) && hours > 0 ? Math.min(hours, 400) : fallback.hours,
      kind: kind === "fiction" || kind === "nonfiction" || kind === "either" ? kind : fallback.kind,
      theme: params.has("theme") ? params.get("theme") : fallback.theme,
      recent: params.has("recent") ? recent : [...fallback.recent],
      shortlist: mode === "shortlist" ? shortlist : [],
    },
    mode,
  };
}

export function briefToQuery(reader, mode) {
  const params = new URLSearchParams();
  params.set("hours", String(reader.hours));
  params.set("kind", reader.kind);
  params.set("theme", reader.theme || "");
  params.set("recent", (reader.recent || []).join(","));
  params.set("mode", mode);
  if (mode === "shortlist") params.set("shortlist", (reader.shortlist || []).join(","));
  return params;
}

export function convene({ books = BOOKS, reader } = {}) {
  const safeReader = {
    hours: Math.min(Math.max(Number(reader?.hours) || 0, 0), 400),
    kind: reader?.kind || "either",
    theme: reader?.theme || "",
    recent: reader?.recent || [],
    shortlist: reader?.shortlist || [],
  };
  const context = buildContext(books, safeReader);
  const pool = eligibleBooks(books, safeReader);
  const tokens = tokenize(safeReader.theme);
  const themeMiss = tokens.length > 0 && pool.every((book) => themeBoost(book, safeReader.theme) === 0);
  const themeConstrained = pool.some((book) => answersTheme(book, safeReader.theme) && tokenize(safeReader.theme).length > 0);
  const standable = (book) => {
    if (themeConstrained && !answersTheme(book, safeReader.theme)) return false;
    return fitsIn(book, safeReader);
  };
  const anyFit = pool.some((book) => standable(book));

  if (!pool.length || safeReader.hours <= 0) {
    const session = {
      reader: safeReader,
      pool: [],
      poolEmpty: pool.length === 0 || safeReader.hours <= 0,
      emptyReason: safeReader.hours <= 0 ? "hours" : "shelf",
      themeMiss,
      overtime: false,
      nominations: [],
      objections: [],
      reservations: [],
      replies: [],
      ballots: [],
      tally: [],
      winner: null,
      runnerUp: null,
      dissent: null,
      unanimous: false,
      strikes: [],
      resolution: "",
      text: "",
    };
    session.resolution =
      safeReader.hours <= 0
        ? "The Chair declines to convene. The brief has no hours in it."
        : "The Chair declines to convene. Nothing on the shelf survives the brief: the kind, the shortlist, and the books marked already read have emptied it.";
    session.text = formatMinutes(session);
    return session;
  }

  const nominations = MEMBERS.map((member) => {
    const ranked = rankBooks(member.id, pool, safeReader, context);
    const ideal = ranked[0];
    const fitting = ranked.filter((book) => standable(book));
    const themed = themeConstrained ? ranked.filter((book) => answersTheme(book, safeReader.theme)) : ranked;
    const standing = anyFit ? fitting[0] : themed[0] || ideal;
    const nomination = {
      memberId: member.id,
      ideal,
      standing,
      struck: ideal.id !== standing.id,
      strikeReason: null,
    };
    if (nomination.struck) {
      const missedHours = !fitsIn(ideal, safeReader);
      const missedTheme = themeConstrained && !answersTheme(ideal, safeReader.theme);
      nomination.strikeReason = missedHours && missedTheme ? "both" : missedTheme ? "theme" : "hours";
    }
    nomination.speech = nominationSpeech(nomination, safeReader, context);
    return nomination;
  });

  const strikes = nominations.filter((nomination) => nomination.struck).map((nomination) => nomination.ideal);
  const standingBooks = [];
  const seen = new Set();
  for (const nomination of nominations) {
    if (seen.has(nomination.standing.id)) continue;
    seen.add(nomination.standing.id);
    standingBooks.push(nomination.standing);
  }

  const single = standingBooks.length === 1;
  const objections = [];
  const reservations = [];
  if (single) {
    for (const member of MEMBERS) {
      reservations.push({
        memberId: member.id,
        book: standingBooks[0],
        speech: reservationSpeech(member.id, standingBooks[0], safeReader, context),
      });
    }
  } else {
    for (const member of MEMBERS) {
      const own = nominations.find((nomination) => nomination.memberId === member.id).standing;
      const others = standingBooks.filter((book) => book.id !== own.id);
      const target = [...others].sort((a, b) => compareBooks(member.id, a, b, safeReader, context)).at(-1);
      objections.push({
        memberId: member.id,
        book: target,
        called: false,
        speech: objectionSpeech(member.id, target, safeReader, context),
      });
    }
    for (const book of standingBooks) {
      if (objections.some((objection) => objection.book.id === book.id)) continue;
      const nominatorIds = new Set(
        nominations.filter((nomination) => nomination.standing.id === book.id).map((nomination) => nomination.memberId),
      );
      const critic = MEMBERS.filter((member) => !nominatorIds.has(member.id))
        .map((member) => ({
          memberId: member.id,
          rank: rankBooks(member.id, standingBooks, safeReader, context).findIndex((entry) => entry.id === book.id),
        }))
        .sort((a, b) => b.rank - a.rank || a.memberId.localeCompare(b.memberId))[0];
      if (!critic) continue;
      objections.push({
        memberId: critic.memberId,
        book,
        called: true,
        speech: objectionSpeech(critic.memberId, book, safeReader, context),
      });
    }
  }

  // Fix target selection properly here by recomputing if I got it wrong.
  // I'll rewrite the block cleanly before shipping. See note in thinking.

  const replies = [];
  if (!single) {
    for (const book of standingBooks) {
      const nominator = nominations.find((nomination) => nomination.standing.id === book.id);
      const aimed = objections.filter((objection) => objection.book.id === book.id);
      if (!aimed.length) continue;
      const sharpest = aimed
        .map((objection) => {
          const objector = nominations.find((nomination) => nomination.memberId === objection.memberId);
          const gap = scoreGap(objection.memberId, objector.standing, book, safeReader, context);
          return { objection, gap };
        })
        .sort((a, b) => b.gap - a.gap || a.objection.memberId.localeCompare(b.objection.memberId))[0];
      replies.push({
        memberId: nominator.memberId,
        book,
        objectorId: sharpest.objection.memberId,
        speech: replySpeech(nominator.memberId, book, sharpest.objection.memberId, context),
      });
    }
  }

  const ballots = MEMBERS.map((member) => ({
    memberId: member.id,
    ranking: rankBooks(member.id, standingBooks, safeReader, context).map((book) => book.id),
  }));
  const { points, firsts } = borda(ballots);
  const tally = standingBooks
    .map((book) => ({
      book,
      points: points.get(book.id) || 0,
      firsts: firsts.get(book.id) || 0,
      fits: fitsIn(book, safeReader),
    }))
    .sort((a, b) => {
      if (b.points !== a.points) return b.points - a.points;
      if (b.firsts !== a.firsts) return b.firsts - a.firsts;
      if (a.fits !== b.fits) return a.fits ? -1 : 1;
      return a.book.title.localeCompare(b.book.title);
    });

  const winner = tally[0].book;
  const runnerUp = tally[1]?.book || null;
  const dissenters = ballots
    .filter((ballot) => ballot.ranking[0] !== winner.id)
    .map((ballot) => ({
      memberId: ballot.memberId,
      rank: ballot.ranking.indexOf(winner.id),
      standing: nominations.find((nomination) => nomination.memberId === ballot.memberId).standing,
    }))
    .sort((a, b) => b.rank - a.rank || a.memberId.localeCompare(b.memberId));
  const dissent = dissenters[0]
    ? {
        memberId: dissenters[0].memberId,
        book: dissenters[0].standing,
        rank: dissenters[0].rank,
        speech: `${memberById(dissenters[0].memberId).name} stood for ${dissenters[0].standing.title} and ranks ${winner.title} ${ordinal(dissenters[0].rank)} of ${standingBooks.length}. ${nominations.find((nomination) => nomination.memberId === dissenters[0].memberId).speech}`,
      }
    : null;

  const session = {
    reader: safeReader,
    pool,
    poolEmpty: false,
    emptyReason: "",
    themeMiss,
    overtime: !anyFit,
    nominations,
    objections,
    reservations,
    replies,
    ballots,
    tally,
    winner,
    runnerUp,
    dissent,
    unanimous: !dissent,
    strikes,
    resolution: "",
    text: "",
  };
  session.resolution = resolutionParagraph(session);
  session.text = formatMinutes(session);
  return session;
}

function scoreGap(memberId, ownBook, otherBook, reader, context) {
  return scoreBook(memberId, ownBook, reader, context) - scoreBook(memberId, otherBook, reader, context);
}

function resolutionParagraph(session) {
  if (!session.winner) return session.resolution;
  const { winner, runnerUp, reader, unanimous, overtime, themeMiss, strikes } = session;
  const points = session.tally.find((row) => row.book.id === winner.id).points;
  const sentences = [];
  sentences.push(
    `Resolved, that the council reads ${winner.title}, by ${winner.author} (${yearLabel(winner.year)}).`,
  );
  sentences.push(
    `The Borda count gives it ${points} point${points === 1 ? "" : "s"} among the standing nominations.`,
  );
  const tied = session.tally.filter((row) => row.points === points && row.firsts === session.tally[0].firsts);
  if (tied.length > 1) {
    sentences.push(
      `${listTitles(tied.map((row) => row.book))} tied on points and on first-place votes. The Chair breaks that tie by title.`,
    );
  }
  sentences.push(
    `It is about ${winner.pages} pages, near ${formatHours(hoursFor(winner))} hours at this shelf's pace, inside a brief of ${formatHours(reader.hours)} hours.`,
  );
  if (runnerUp) {
    sentences.push(`The next book in the count is ${runnerUp.title}.`);
  }
  if (unanimous) {
    sentences.push("The vote is unanimous. Reservations are entered and do not change the resolution.");
  } else {
    sentences.push(`Dissent is entered by the ${memberById(session.dissent.memberId).name}.`);
  }
  if (overtime) {
    sentences.push("Nothing on the eligible shelf finishes inside the brief. The resolution is the nearest book, and it runs long.");
  }
  if (strikes.length) {
    const unique = [];
    const seen = new Set();
    for (const book of strikes) {
      if (seen.has(book.id)) continue;
      seen.add(book.id);
      const reason = session.nominations.find((nomination) => nomination.ideal.id === book.id)?.strikeReason;
      unique.push({ book, reason });
    }
    const hoursStruck = unique.filter((entry) => entry.reason === "hours" || entry.reason === "both").map((entry) => entry.book);
    const themeStruck = unique.filter((entry) => entry.reason === "theme").map((entry) => entry.book);
    if (hoursStruck.length) sentences.push(`The Chair struck ${listTitles(hoursStruck)} for exceeding the hours.`);
    if (themeStruck.length) sentences.push(`The Chair struck ${listTitles(themeStruck)} for missing the theme.`);
  }
  if (themeMiss) {
    sentences.push("The theme met no book on the eligible shelf. The council used the hours, the kind, and what has just been read.");
  }
  return sentences.join(" ");
}

export function formatMinutes(session) {
  const lines = [];
  lines.push("THE READING COUNCIL");
  lines.push("Minutes");
  lines.push("");
  const { reader } = session;
  const kind =
    reader.kind === "either" ? "Fiction or nonfiction" : reader.kind === "fiction" ? "Fiction only" : "Nonfiction only";
  lines.push(`Brief. ${formatHours(reader.hours)} hours. ${kind}.`);
  lines.push(reader.theme ? `Theme. ${reader.theme}.` : "Theme. None entered.");
  if (reader.shortlist?.length) lines.push(`Shortlist. ${reader.shortlist.length} titles.`);
  const recent = (reader.recent || []).map((id) => session.pool && findTitle(id, session)).filter(Boolean);
  lines.push(recent.length ? `Just read. ${recent.join("; ")}.` : "Just read. Nothing entered.");
  lines.push("");
  if (!session.winner) {
    lines.push(session.resolution);
    lines.push("");
    lines.push(paceNote());
    return lines.join("\n");
  }
  lines.push("Nominations.");
  for (const nomination of session.nominations) {
    const member = memberById(nomination.memberId);
    lines.push(`${member.name} nominates ${nomination.standing.title}.`);
    if (nomination.struck) lines.push(`First named ${nomination.ideal.title}, which the Chair strikes.`);
    lines.push(nomination.speech);
    lines.push("");
  }
  if (session.objections.length) {
    lines.push("Objections.");
    for (const objection of session.objections) {
      const called = objection.called ? " Called, because nobody had spoken against it." : "";
      lines.push(`${memberById(objection.memberId).name}, against ${objection.book.title}.${called}`);
      lines.push(objection.speech);
      lines.push("");
    }
  }
  if (session.reservations.length) {
    lines.push("Reservations.");
    for (const reservation of session.reservations) {
      lines.push(`${memberById(reservation.memberId).name}.`);
      lines.push(reservation.speech);
      lines.push("");
    }
  }
  if (session.replies.length) {
    lines.push("Replies.");
    for (const reply of session.replies) {
      lines.push(`${memberById(reply.memberId).name}, answering ${memberById(reply.objectorId).short}.`);
      lines.push(reply.speech);
      lines.push("");
    }
  }
  lines.push("The vote.");
  for (const ballot of session.ballots) {
    const names = ballot.ranking.map((id) => titleFromSession(session, id));
    lines.push(`${memberById(ballot.memberId).short}: ${names.join(", ")}`);
  }
  lines.push("");
  for (const row of session.tally) {
    lines.push(`${row.book.title}: ${row.points} points, ${row.firsts} first-place vote${row.firsts === 1 ? "" : "s"}.`);
  }
  lines.push("");
  lines.push("Resolution.");
  lines.push(session.resolution);
  if (session.dissent) {
    lines.push("");
    lines.push("Dissent.");
    lines.push(session.dissent.speech);
  }
  lines.push("");
  lines.push(paceNote());
  lines.push(`${CHAIR.name} does not vote.`);
  return lines.join("\n");
}

function findTitle(id, session) {
  const fromNom = session.nominations?.flatMap((nomination) => [nomination.ideal, nomination.standing]);
  const found = (fromNom || []).find((book) => book && book.id === id);
  if (found) return found.title;
  const inPool = (session.pool || []).find((book) => book.id === id);
  if (inPool) return inPool.title;
  return BOOKS.find((book) => book.id === id)?.title || id;
}

function titleFromSession(session, id) {
  const row = session.tally.find((entry) => entry.book.id === id);
  return row ? row.book.title : id;
}

function paceNote() {
  return "Pace. 55 pages an hour, minus 8 pages for each step of difficulty from 1 to 5. A book may stand if it finishes inside the brief plus a tenth. Page counts are round paperback lengths. The scores are this council's taste, so the disagreement can be checked.";
}

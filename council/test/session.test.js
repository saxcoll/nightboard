import assert from "node:assert/strict";
import test from "node:test";
import { BOOKS } from "../js/catalog.js";
import { answersTheme } from "../js/score.js";
import { borda, briefFromParams, convene, DEFAULT_READER, fitsIn, hoursFor } from "../js/convene.js";

test("the shelf is internally consistent", () => {
  const ids = new Set();
  for (const book of BOOKS) {
    assert.equal(ids.has(book.id), false, book.id);
    ids.add(book.id);
    assert.ok(book.pages > 0);
    assert.ok(book.kind === "fiction" || book.kind === "nonfiction");
    for (const key of ["difficulty", "pleasures", "finishability", "influence", "challenge"]) {
      assert.ok(book[key] >= 1 && book[key] <= 5, `${book.id} ${key}`);
    }
    assert.ok(book.hype >= 0 && book.hype <= 3);
    for (const id of book.helpfulBefore) assert.ok(ids.has(id) || BOOKS.some((entry) => entry.id === id), id);
  }
  const fiction = BOOKS.filter((book) => book.kind === "fiction").length;
  const nonfiction = BOOKS.length - fiction;
  assert.ok(fiction >= 8);
  assert.ok(nonfiction >= 8);
});

test("helpful-before links point at real books", () => {
  const ids = new Set(BOOKS.map((book) => book.id));
  for (const book of BOOKS) {
    for (const id of book.helpfulBefore) assert.ok(ids.has(id), `${book.id} -> ${id}`);
  }
});

test("Kant’s Groundwork is a short, slow book", () => {
  const kant = BOOKS.find((book) => book.id === "groundwork");
  assert.ok(Math.abs(hoursFor(kant) - 6) < 0.05);
  assert.equal(fitsIn(kant, { hours: 10 }), true);
  assert.equal(fitsIn(BOOKS.find((book) => book.id === "theory-of-justice"), { hours: 10 }), false);
});

test("Borda count", () => {
  const { points, firsts } = borda([
    { memberId: "a", ranking: ["x", "y"] },
    { memberId: "b", ranking: ["x", "y"] },
    { memberId: "c", ranking: ["y", "x"] },
  ]);
  assert.equal(points.get("x"), 5);
  assert.equal(points.get("y"), 4);
  assert.equal(firsts.get("x"), 2);
  assert.equal(firsts.get("y"), 1);
});

test("after Descartes, Hume, and Kant, the hours strike Rawls and Canon stands for Mill", () => {
  const session = convene({ reader: DEFAULT_READER });
  const canon = session.nominations.find((nomination) => nomination.memberId === "canon");
  assert.equal(canon.ideal.id, "theory-of-justice");
  assert.equal(canon.struck, true);
  assert.equal(canon.standing.id, "on-liberty");
  assert.equal(fitsIn(session.winner, DEFAULT_READER), true);
  for (const nomination of session.nominations) {
    assert.equal(fitsIn(nomination.standing, DEFAULT_READER), true);
    assert.equal(DEFAULT_READER.recent.includes(nomination.standing.id), false);
  }
});

test("Opposition answers a philosophy streak with fiction", () => {
  const session = convene({ reader: DEFAULT_READER });
  const opposition = session.nominations.find((nomination) => nomination.memberId === "opposition");
  assert.equal(opposition.standing.kind, "fiction");
});

test("Doubt does not nominate Sapiens when the shelf is open and no theme is set", () => {
  const session = convene({ reader: { ...DEFAULT_READER, recent: [] } });
  const doubt = session.nominations.find((nomination) => nomination.memberId === "doubt");
  assert.notEqual(doubt.ideal.id, "sapiens");
  assert.notEqual(doubt.standing.id, "sapiens");
});

test("a fiction brief never nominates nonfiction", () => {
  const session = convene({ reader: { hours: 30, kind: "fiction", theme: "", recent: [], shortlist: [] } });
  for (const nomination of session.nominations) {
    assert.equal(nomination.ideal.kind, "fiction");
    assert.equal(nomination.standing.kind, "fiction");
  }
  assert.equal(session.winner.kind, "fiction");
});

test("a shortlist is a closed shelf, and Doubt will not pick the hyped survey", () => {
  const shortlist = ["sapiens", "the-fire-next-time", "the-stranger"];
  const session = convene({
    reader: { hours: 12, kind: "either", theme: "", recent: [], shortlist },
  });
  for (const nomination of session.nominations) {
    assert.ok(shortlist.includes(nomination.standing.id));
  }
  const doubt = session.nominations.find((nomination) => nomination.memberId === "doubt");
  assert.notEqual(doubt.standing.id, "sapiens");
  assert.ok(session.text.includes("Resolved"));
  assert.equal(session.winner.id === "sapiens", false);
});

test("one book on the shortlist still gets reservations", () => {
  const session = convene({
    reader: { hours: 20, kind: "either", theme: "", recent: [], shortlist: ["the-stranger"] },
  });
  assert.equal(session.winner.id, "the-stranger");
  assert.equal(session.unanimous, true);
  assert.equal(session.objections.length, 0);
  assert.equal(session.reservations.length, 5);
});

test("an emptied shelf declines to convene", () => {
  const session = convene({
    reader: { hours: 8, kind: "either", theme: "", recent: ["the-stranger"], shortlist: ["the-stranger"] },
  });
  assert.equal(session.poolEmpty, true);
  assert.equal(session.winner, null);
  assert.match(session.resolution, /declines/i);
});

test("the same brief produces the same minutes", () => {
  const first = convene({ reader: DEFAULT_READER });
  const second = convene({ reader: DEFAULT_READER });
  assert.equal(first.text, second.text);
  assert.equal(first.winner.id, second.winner.id);
});

test("a shared link restores the brief", () => {
  const params = new URLSearchParams("hours=6&kind=fiction&theme=mortality&recent=gilead&mode=shortlist&shortlist=the-stranger,mrs-dalloway");
  const { reader, mode } = briefFromParams(params);
  assert.equal(reader.hours, 6);
  assert.equal(reader.kind, "fiction");
  assert.equal(reader.theme, "mortality");
  assert.deepEqual(reader.recent, ["gilead"]);
  assert.equal(mode, "shortlist");
  assert.deepEqual(reader.shortlist, ["the-stranger", "mrs-dalloway"]);
});

test("a history of science request is answered by a history of science", () => {
  const reader = { hours: 30, kind: "nonfiction", theme: "history of science", recent: [], shortlist: [] };
  const session = convene({ reader });
  const onTheme = new Set(["structure", "sapiens", "atomic-bomb"]);
  assert.ok(onTheme.has(session.winner.id), session.winner.title);
  for (const nomination of session.nominations) {
    assert.ok(onTheme.has(nomination.standing.id), nomination.standing.title);
    assert.equal(answersTheme(nomination.standing, reader.theme), true);
  }
  assert.notEqual(session.winner.id, "the-fire-next-time");
});

test("three hours cannot resolve to Middlemarch", () => {
  const reader = { hours: 3, kind: "either", theme: "", recent: [], shortlist: [] };
  const session = convene({ reader });
  assert.equal(fitsIn(session.winner, reader), true);
  assert.notEqual(session.winner.id, "middlemarch");
  const time = session.nominations.find((nomination) => nomination.memberId === "time");
  assert.equal(fitsIn(time.standing, reader), true);
});

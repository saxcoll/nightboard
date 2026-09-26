import { memberById } from "./members.js";
import { fitsIn, formatHours, hoursFor, yearLabel } from "./score.js";

function titles(books) {
  if (books.length === 1) return books[0].title;
  if (books.length === 2) return `${books[0].title} and ${books[1].title}`;
  return `${books
    .slice(0, -1)
    .map((book) => book.title)
    .join(", ")}, and ${books.at(-1).title}`;
}

function paceClause(book) {
  return `about ${formatHours(hoursFor(book))} hours at this shelf's pace`;
}

function strikeClause(nomination, reader) {
  if (!nomination.struck) return "";
  const ideal = nomination.ideal;
  const named = `I first named ${ideal.title} (${ideal.author}, ${yearLabel(ideal.year)}).`;
  if (nomination.strikeReason === "theme") {
    return `${named} It does not answer the brief, which was ${reader.theme}. The Chair strikes it. `;
  }
  if (nomination.strikeReason === "both") {
    return `${named} It asks ${paceClause(ideal)}, past a brief of ${formatHours(reader.hours)} hours, and it does not answer ${reader.theme}. The Chair strikes it. `;
  }
  return `${named} It asks ${paceClause(ideal)}, and the brief allows ${formatHours(reader.hours)}. The Chair strikes it. `;
}

export function nominationSpeech(nomination, reader, context) {
  const member = memberById(nomination.memberId);
  const book = nomination.standing;
  const ideal = nomination.ideal;
  const strike = strikeClause(nomination, reader);

  if (member.id === "canon") {
    const downstream = context.dependents.get(book.id) || [];
    const sequence = (book.helpfulBefore || [])
      .map((id) => context.byId.get(id))
      .filter((entry) => entry && (reader.recent || []).includes(entry.id));
    let because;
    if (sequence.length) {
      because = `We have just read ${titles(sequence)}. ${book.title} is the sitting that sequence is for.`;
    } else if (downstream.length) {
      because = `${titles(downstream)} sit downstream of ${book.title}. I nominate the book the rest of the shelf already uses.`;
    } else {
      because = `${book.title} carries influence ${book.influence} of 5. I am here for the book other books assume, and this is the one the hours will hold.`;
    }
    return `${strike}${because}`;
  }

  if (member.id === "pleasure") {
    const kind =
      book.kind === "fiction"
        ? "It is fiction, which is the gladness I trust when the scores are close."
        : "It is not a novel. The prose still has to be a place we are willing to stay.";
    return `${strike}${book.title} is about ${book.pages} pages, ${paceClause(book)}. Pleasure ${book.pleasures} of 5, finishability ${book.finishability} of 5. ${kind}`;
  }

  if (member.id === "opposition") {
    if (!context.recentBooks.length) {
      return `${strike}Nothing has been entered as just read, so I take the hardest correction the brief can hold. ${book.title} is challenge ${book.challenge} of 5.`;
    }
    const recentKind = context.recentKinds.size === 1 ? [...context.recentKinds][0] : null;
    const shift =
      recentKind && book.kind !== recentKind
        ? `Those were ${recentKind}. ${book.title} is ${book.kind}, and the change of kind is the point.`
        : `This stays in ${book.kind}, so the correction is the difficulty of the thing itself: challenge ${book.challenge} of 5.`;
    return `${strike}We have just read ${titles(context.recentBooks)}. ${shift}`;
  }

  if (member.id === "time") {
    const share = hoursFor(book) / reader.hours;
    let fit;
    if (share < 0.45) {
      fit = `It leaves hours unused. I allow that because influence is ${book.influence} of 5 and pleasure is ${book.pleasures} of 5, which is a lot of book for the length.`;
    } else if (share > 0.9) fit = "It fills the brief. We finish, with a thin margin.";
    else fit = "It uses the sitting and still leaves a margin before the last page.";
    const over = fitsIn(book, reader) ? "" : " It still runs past the brief. Nothing on the shelf finishes in these hours, so this is the nearest.";
    return `${strike}${book.title} asks ${paceClause(book)}. The brief is ${formatHours(reader.hours)} hours. ${fit}${over}`;
  }

  if (book.doubtNote) {
    return `${strike}I nominate ${book.title} with this entered against it: ${book.doubtNote}`;
  }
  if ((book.hype || 0) >= 2) {
    return `${strike}${book.title} arrives with a reputation. I am nominating the book anyway, and the reputation is not the reason. Influence ${book.influence} of 5.`;
  }
  return `${strike}${book.title} is on the shelf with influence ${book.influence} of 5 and hype ${book.hype || 0} of 3. Nothing in the record asks me to discount it. Fame is not the reason.`;
}

export function objectionSpeech(memberId, book, reader, context) {
  if (memberId === "canon") {
    const downstream = context.dependents.get(book.id) || [];
    if (!downstream.length) {
      if (book.influence >= 5) {
        return `${book.title} has influence ${book.influence} of 5, and still nothing else on this shelf is waiting on it. I want the book the next book needs.`;
      }
      return `${book.title} may be a fine evening. Little else on this shelf depends on it, and influence is ${book.influence} of 5.`;
    }
    return `${book.title} is load-bearing, and I still rank it behind a book the recent reading has already prepared.`;
  }
  if (memberId === "pleasure") {
    return `${book.title} is pleasure ${book.pleasures} of 5 and finishability ${book.finishability} of 5, at difficulty ${book.difficulty}. Importance does not repair a book we set down.`;
  }
  if (memberId === "opposition") {
    const overlap = book.subjects.filter((subject) => context.recentSubjects.has(subject));
    if (context.recentBooks.length && overlap.length) {
      return `${book.title} continues the last sitting. It shares ${overlap.join(", ")} with what we have just read.`;
    }
    if (context.recentKinds.size === 1 && context.recentKinds.has(book.kind)) {
      return `${book.title} keeps us inside ${book.kind}. After ${titles(context.recentBooks)}, that is the rut.`;
    }
    return `${book.title} is the mildest of the nominations. Challenge ${book.challenge} of 5 is not a correction.`;
  }
  if (memberId === "time") {
    const hours = hoursFor(book);
    const share = hours / reader.hours;
    if (!fitsIn(book, reader)) {
      return `${book.title} asks ${paceClause(book)}. The brief is ${formatHours(reader.hours)} hours. I will not vote to begin a book these hours cannot finish.`;
    }
    if (share < 0.45) {
      return `${book.title} is only ${paceClause(book)} inside a ${formatHours(reader.hours)}-hour brief. Most of the sitting would go unused.`;
    }
    if (share > 0.9) {
      return `${book.title} takes ${paceClause(book)} inside a ${formatHours(reader.hours)}-hour brief. The margin is thinner than I want.`;
    }
    return `${book.title} fits, at ${paceClause(book)}. I still rank it behind a book that spends the same hours better.`;
  }
  if (book.doubtNote) return `${book.title}. ${book.doubtNote}`;
  if ((book.hype || 0) >= 2) {
    return `${book.title} is famous for being chosen. The nomination still has to say what the book is for.`;
  }
  return `${book.title} is the respectable nomination. Respectability is not yet a reason.`;
}

export function reservationSpeech(memberId, book, reader, context) {
  if (memberId === "canon") {
    const downstream = context.dependents.get(book.id) || [];
    if (downstream.length) return `My reservation is small. ${titles(downstream)} already depend on it, which is why I can live with a unanimous vote.`;
    return `My reservation: nothing else on this shelf requires ${book.title}. We would be reading it for itself, which is allowed and is not my usual reason.`;
  }
  if (memberId === "pleasure") {
    if (book.pleasures >= 5 && book.finishability >= 4) {
      return "I enter no reservation about the experience of reading it. The other questions are not mine.";
    }
    return `My reservation is the texture: pleasure ${book.pleasures} of 5, finishability ${book.finishability} of 5. We may admire it and still stall.`;
  }
  if (memberId === "opposition") {
    if (!context.recentBooks.length) {
      return `My reservation: with nothing just read, I cannot tell whether ${book.title} corrects a habit or starts one.`;
    }
    return `My reservation: a unanimous shelf can still be a rut. ${book.title} should not become the kind of book we always pick next.`;
  }
  if (memberId === "time") {
    return `My reservation is the clock, stated for the minutes: ${book.title} is ${paceClause(book)} inside ${formatHours(reader.hours)} hours.`;
  }
  if (book.doubtNote) return `My reservation stands even in agreement: ${book.doubtNote}`;
  return `My reservation is only procedural. ${book.title} is not being chosen because everyone else is reading it. If that changes, I will object next time.`;
}

export function replySpeech(_memberId, book, objectorId, context) {
  const objector = memberById(objectorId);
  if (objectorId === "canon") {
    const downstream = context.dependents.get(book.id) || [];
    const answer = downstream.length
      ? `${titles(downstream)} do depend on it.`
      : "Nothing else on this shelf lists it as helpful reading.";
    return `${objector.short} wants a book other books already use. ${answer} I stay with ${book.title}.`;
  }
  if (objectorId === "pleasure") {
    return `${objector.short} says we may set ${book.title} down. Pleasure is ${book.pleasures} of 5 and finishability is ${book.finishability} of 5. That is the risk already in my nomination, and I stay.`;
  }
  if (objectorId === "opposition") {
    return `${objector.short} says ${book.title} fails as a correction to the last sitting. The last sitting is not my ledger. I stay with the book.`;
  }
  if (objectorId === "time") {
    return `${objector.short} is right about the clock. ${book.title} is ${paceClause(book)}. I am staying with it for a reason other than an elegant timetable.`;
  }
  const caveat = book.doubtNote ? ` The record already says: ${book.doubtNote}` : " There is no caveat entered against it.";
  return `${objector.short} asks for a reason other than reputation.${caveat} I stay with ${book.title}.`;
}

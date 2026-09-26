export const MEMBERS = [
  {
    id: "canon",
    name: "Member for the Canon",
    short: "Canon",
    rule: "Read the book that other books already assume. A sequence outranks a novelty. Length is earned by what depends on it.",
  },
  {
    id: "pleasure",
    name: "Member for Pleasure",
    short: "Pleasure",
    rule: "Read the book we will still be glad to be inside late in the evening. A book set down at page forty was the wrong decision, however important.",
  },
  {
    id: "opposition",
    name: "Member for Opposition",
    short: "Opposition",
    rule: "Read against the last book. After arguments, choose a world. After a novel, choose a claim. A council that repeats the reader's habits is not a council.",
  },
  {
    id: "time",
    name: "Member for Time",
    short: "Time",
    rule: "Read only what these hours can finish. Use the sitting. Leave a margin. An unfinished monument is a wish, not a plan.",
  },
  {
    id: "doubt",
    name: "Member for Doubt",
    short: "Doubt",
    rule: "Distrust a book that is famous for being chosen. Reputation is not a reason. A contested book has to be nominated for the argument it actually makes.",
  },
];

export const CHAIR = {
  id: "chair",
  name: "The Chair",
  short: "Chair",
  rule: "The Chair does not nominate. The Chair strikes what the hours cannot finish, counts a Borda vote of the standing nominations, and enters the dissent in the dissenter's own terms.",
};

export function memberById(id) {
  return MEMBERS.find((member) => member.id === id) || null;
}

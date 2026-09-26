# The Reading Council

Five members and a Chair decide what to read next. The shelf holds fiction and nonfiction. Each member nominates by a fixed rule, objects to someone else's nomination, and ranks the books still standing. The Chair strikes anything the hours cannot finish, counts a Borda vote, breaks a remaining tie by title, and enters the dissent.

The members are rule-bound on purpose. A sitting can be checked. The same brief always produces the same minutes.

## Run

From the repository root:

```bash
python3 -m http.server 8000
```

Open [http://localhost:8000/council/](http://localhost:8000/council/).

The opening brief is ten hours, fiction or nonfiction, with Descartes, Hume, and Kant marked as just read. Change the hours, the kind, the theme, or the books, then convene again. The address bar keeps the brief, so a sitting can be shared.

**Open shelf** nominates from anything not marked lately read. **Shortlist** nominates only from the books you mark List.

## The sitting

1. Each member names the book their rule wants, including a book the hours cannot finish.
2. The Chair strikes a book the hours cannot finish, and, when the shelf can meet every word of the theme, a book that misses the theme. The member then stands for their best book that survives both. A book may stand when it finishes inside the brief plus a tenth. Hours are rounded to a tenth, the same way the minutes print them.
3. Pace is 55 pages an hour, minus 8 pages for each step of difficulty from 1 to 5. Kant is short and slow. A long novel at difficulty 2 is faster.
4. Each member objects to the standing nomination they rank worst. If a nomination drew no objection, the Chair calls on the member who ranks it lowest.
5. The nominator replies to the sharpest objection.
6. Each member ranks the standing books. First place among five books is worth 5 points. The Chair does not vote.
7. Ties on points and first-place votes break by title.

## The members

- **Canon** reads the book other books already assume, and the book the last ones prepared.
- **Pleasure** reads the book you will still be glad to be inside late.
- **Opposition** reads against the last book. After arguments, a world. After a novel, a claim.
- **Time** reads what these hours can finish, close to two thirds of the sitting when the book supports it.
- **Doubt** distrusts a book that is famous for being chosen. A note in the catalog is a caveat Doubt will say out loud.
- **The Chair** does not nominate.

## Tests

```bash
node --test council/test/session.test.js
```

Page counts are round paperback lengths. Influence, pleasure, difficulty, and the other scores are this council's taste. They are editorial, and they are the mechanism of the debate.

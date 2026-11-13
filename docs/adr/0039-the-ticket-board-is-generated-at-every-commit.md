---
status: accepted
---

# The ticket board is generated from the tickets at every commit

`docs/tickets.html`, the ticket board, lists every ticket in one row: its ID, its short title (the ticket's `<title>`) linked to the ticket, the text of its Status pill and a link to its plan. Nobody writes it: `coffeecv.ticket_board` generates it from the tickets, and `.githooks/pre-commit` runs it on every commit, putting the board of the staged tickets into the commit and the board of the tickets on disk onto the disk. A ticket's own Status stays the source of truth for its status ([issue-tracker.md](../agents/issue-tracker.md)); the board only copies it. `scripts/check.sh` checks, without writing, that the board on disk is the generated one, which catches a commit made with `--no-verify`.

We chose this because the tracker is one HTML file per ticket, and with 50 of them nothing showed where the work stood without opening each one. We chose a board that copies the status, not one that only lists the tickets, because a board of stale statuses is worse than none: it is read as current. We chose to generate it in the pre-commit hook over three alternatives:

- **Rows written by hand, and a check that fails the commit.** Every field of a row is copied from its ticket, so writing it is pure toil, and closing a ticket would mean editing two files. This was the first form of this ADR, the same day.
- **JavaScript in the page that reads each ticket's status.** Browsers do not let a page opened from disk (`file://`) read other files, so the page would work only when served over HTTP, and it still could not list the tickets.
- **A step that I, Claude, or the owner runs.** The hook runs on every commit, whoever makes it; a step to remember is a step that gets missed.

## Consequences

- The hook changes what is committed: a commit that changes a ticket's Status, title or plan, or adds a ticket, also commits `docs/tickets.html`, and the hook says so. A hand edit to the board is overwritten by the next commit.
- Two branches that each add a ticket both change the board, so their merge conflicts there; resolve it with `python -m coffeecv.ticket_board --write` and `git add docs/tickets.html`.
- A ticket without an eyebrow ID, a `<title>` or a Status pill cannot have a row, and the hook refuses the commit until it has them.

Source: the owner's request, 2026-11-13, made without a ticket.

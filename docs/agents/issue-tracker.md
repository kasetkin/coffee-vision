# Issue tracker: HTML tickets in docs/

Tickets and specs for this repo are HTML files in `docs/`, one file per ticket. The ticket is the source of truth for its decisions and status.

## Conventions

- **Ticket**: `docs/ticket_<slug>.html`. IDs are `ML-<n>` (data, model, training, evaluation) or `OPS-<n>` (deploy, infrastructure), numbered on from the highest existing ID with that prefix: `grep -o 'Ticket [A-Z]*-[0-9]*' docs/ticket_*.html`.
- **New ticket**: copy the newest ticket file and replace its content, keeping the stylesheet, the header table (Status, Triage, Priority, Type, Component, Reporter, Assignee, Created, Blocked by) and the sections (Summary, Background, Before and after, What does not change, Risks, Decisions, Scope and acceptance, Out of scope, Activity).
- **Decisions** are numbered D1, D2, ... in the Decisions section. The owner makes them; record each with its date.
- **Plan**: `docs/plan_<slug>.html`, header "Plan · <ID>", phases P1, P2, ..., owner questions Q1, Q2, .... An answered Q goes into the ticket as the next D-number.
- **Status** row: `Open`, `In progress` or `Closed`, then a short dated line saying what is done (commits, release). The owner closes tickets.
- **Comments and history**: dated entries appended to the Activity list, oldest first.

## When a skill says "publish to the issue tracker"

Create `docs/ticket_<slug>.html` with the next free ID, Status `Open`, the Triage row set to the roles the skill names (else `needs-triage`, see `triage-labels.md`), and a first Activity entry. One file per ticket; a ticket that waits on others names their IDs in Blocked by.

## When a skill says "fetch the relevant ticket"

Find the file by ID (`grep -l 'Ticket ML-3' docs/ticket_*.html`) and read it with its plan, if one exists. Each file opens with a long stylesheet: the content starts at the first `<h1>`.

## Wayfinding operations

Used by `/wayfinder`. The **map** is a ticket with one **child** ticket per question.

- **Map**: a ticket with Type `Wayfinder map` and sections Notes, Decisions so far, Fog.
- **Child ticket**: its own ticket file, with a `Part of` row naming the map's ID and Type `wayfinder:<research|prototype|grilling|task>`.
- **Blocking**: the Blocked by row lists ticket IDs. A ticket is unblocked when every listed ticket is Closed.
- **Frontier**: children of the map that are Open, unblocked and have no Assignee; lowest ID first.
- **Claim**: set Assignee and Status `In progress`, and save, before any work.
- **Resolve**: append the answer under an Answer heading, set Status `Closed`, then add a one-line pointer (gist + file) to the map's Decisions so far.

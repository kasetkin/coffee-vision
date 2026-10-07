# Issue tracker: HTML tickets in docs/

Tickets and their plans are HTML files in `docs/`, one file each. The ticket is also the spec, and the source of truth for its decisions and status.

## Conventions

- **Ticket**: `docs/ticket_<slug>.html`. IDs are `ML-<n>` (data, model, training, evaluation) or `OPS-<n>` (deploy, infrastructure), numbered on from the highest existing ID with that prefix: `grep -o 'Ticket [A-Z]*-[0-9]*' docs/ticket_*.html`.
- **New ticket**: copy `docs/template_ticket.html` and follow the comment at its top: fill every `[[...]]` placeholder (none may be left), drop the optional fields and sections you don't use, and delete the comments. Tickets written before the template have no Triage row; leave them as they are.
- **Decisions** are numbered D1, D2, ... in the Decisions section. The owner makes them; record each with its date.
- **ADR**: a decision that outlives its ticket (it binds later tickets, and is hard to reverse) also gets `docs/adr/NNNN-<slug>.md`, in the format of `/domain-modeling`'s ADR-FORMAT. The ADR names the ticket and D-number it came from, and the D-entry links the ADR. A decision that matters only inside one ticket stays a D-number there.
- **Plan**: `docs/plan_<slug>.html`, copied from `docs/template_plan.html` (fill every `[[...]]`, delete the comments), header "Plan · <ID>", phases P1, P2, ..., owner questions Q1, Q2, .... An answered Q goes into the ticket as the next D-number. Each phase is an `<h2>` ("N. P4 &mdash; title", with a `stop` pill where the owner must act), followed by a `fields` block with its own **Status** (`Open`, `In progress`, `Done` plus a dated line with commits) and **Blocked by** (phases, Qs or ticket IDs, or `None`). Plans written before this rule have no per-phase fields; leave them as they are.
- **Commits** start with the ticket and phase: `ML-3 P4: ...`. This is how `/code-review` finds the spec.
- **Status** row: `Open`, `In progress` or `Closed`, then a short dated line saying what is done (commits, release). The owner closes tickets.
- **Comments and history**: dated entries appended to the Activity list, oldest first.
- **Styles** live in `docs/tickets.css`, which every ticket, plan and template links; no inline `<style>`.

## When a skill says "publish to the issue tracker"

Create `docs/ticket_<slug>.html` with the next free ID, Status `Open`, the Triage row set to the roles the skill names (else `needs-triage`, see `triage-labels.md`), and a first Activity entry. One file per ticket; a ticket that waits on others names their IDs in Blocked by.

## When a skill says "fetch the relevant ticket"

Find the file by ID (`grep -l 'Ticket ML-3' docs/ticket_*.html`) and read it with its plan, if one exists.

## How the engineering skills map onto tickets and plans

- **`/to-spec`** writes the spec into the ticket: a new `ticket_<slug>.html` from the template, or the existing ticket's sections when one is named. The spec template's parts map onto the ticket's sections (Problem and Solution → Summary and Before and after; Implementation and Testing Decisions → Decisions as D-numbers; Out of Scope → Out of scope). Set Triage to `ready-for-agent`.
- **`/to-tickets`** writes the slices as phases in the ticket's `plan_<slug>.html` (one plan per ticket), each with its own Status and Blocked by. It does not create a ticket per slice, and it leaves the ticket's own Status alone.
- **`/implement`** and **`/implement-spec`** take one phase as a ticket: `ML-3 P4` means phase P4 of ML-3's plan. A phase is ready when every entry in its Blocked by is Done, Closed or answered. Set the phase's Status to `In progress` before work and `Done` with its commits after, append a dated Activity entry to the plan, and stop at a `stop` phase for the owner.
- **`/code-review`** reads the ticket and its plan as the spec, found through the `ML-n Pn:` prefix in the commit messages.

## Wayfinding operations

Used by `/wayfinder`. The **map** is a ticket with one **child** ticket per question.

- **Map**: a ticket with Type `Wayfinder map` and sections Notes, Decisions so far, Fog.
- **Child ticket**: its own ticket file, with a `Part of` row naming the map's ID and Type `wayfinder:<research|prototype|grilling|task>`.
- **Blocking**: the Blocked by row lists ticket IDs. A ticket is unblocked when every listed ticket is Closed.
- **Frontier**: children of the map that are Open, unblocked and have no Assignee; lowest ID first.
- **Claim**: set Assignee and Status `In progress`, and save, before any work.
- **Resolve**: append the answer under an Answer heading, set Status `Closed`, then add a one-line pointer (gist + file) to the map's Decisions so far.

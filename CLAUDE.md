Coffee-origin classifier from phone photos. Humans start at [README.md](README.md).

## Publishing

This repo is public: every committed file is published, and history is kept, not rewritten. Personal data lives only in private Claude memory (`.private_claude/`, gitignored). `.githooks/pre-commit` runs `coffeecv.leak_check` on staged files and refuses a commit with a public IP, email address, credential, photo GPS or one of the owner's own patterns: move that content to private memory. SSH aliases, the DVC remote URL, the VM's user and hostname, the live site's domain and Docker-internal addresses are fine to publish.

## Where project knowledge lives

- **Decisions**, `docs/adr/`: read the ones in the area before proposing a design change; a proposal that contradicts one says so.
- **Closed ideas**, `.out-of-scope/`: check before proposing an experiment, a training lever or an infrastructure change.
- **Review rules**, `CODING_STANDARDS.md`: what a change is reviewed against.
- **Runbooks**, `docs/agents/`: `dvc.md` before moving data or trusting a DVC status; `remote-sweeps.md` before running or syncing work on the VM; `experiments.md` before a recorded run, a comparison or a results report; `deploy.md` before a deploy or a claim that something is live.

## Agent skills

### Issue tracker

HTML tickets, `docs/ticket_<slug>.html` (IDs ML-n, OPS-n), with plans in `docs/plan_<slug>.html`. See `docs/agents/issue-tracker.md`.

### Triage labels

Default role names, written in each ticket's Triage row. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` and `docs/adr/` at the root. See `docs/agents/domain.md`.

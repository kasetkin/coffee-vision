Coffee-origin classifier from phone photos. Humans start at [README.md](README.md).

## Publishing

This repo is public: every committed file is published, and history is kept, not rewritten. Personal data lives only in private Claude memory (`.private_claude/`, gitignored). `.githooks/pre-commit` runs `coffeecv.leak_check` on staged files and refuses a commit with a public IP, email address, credential, photo GPS or one of the owner's own patterns: move that content to private memory. SSH aliases, the DVC remote URL, the VM's user and hostname, the live site's domain and Docker-internal addresses are fine to publish.

## Agent skills

### Issue tracker

HTML tickets, `docs/ticket_<slug>.html` (IDs ML-n, OPS-n), with plans in `docs/plan_<slug>.html`. See `docs/agents/issue-tracker.md`.

### Triage labels

Default role names, written in each ticket's Triage row. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` and `docs/adr/` at the root. See `docs/agents/domain.md`.

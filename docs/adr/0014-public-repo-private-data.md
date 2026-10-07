# The repo is public; personal data and the dataset are not

Everything committed is published and history is not rewritten, so personal data is stopped before a commit: public IPs, email addresses, credentials, and GPS or other identifying photo metadata (`coffeecv.leak_check`, run by `.githooks/pre-commit`, `scripts/check.sh` and every deploy). Infrastructure names are fine: SSH aliases, the DVC remote URL, the VM's user and hostname, the live site's domain. The dataset and the DVC remote stay private, so the guardrail runs without them: `scripts/check.sh` has no DVC step, and tests that need real data are marked `@real_data`.

Sources: [OPS-2 D1](../ticket_memory_to_repo.html); owner decision, 2026-10-07 (no DVC in the guardrail).

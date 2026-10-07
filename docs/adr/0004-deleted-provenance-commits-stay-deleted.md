# Deleted provenance commits stay deleted

On 2026-09-07 the owner deleted old branches and an archive tag, which left the recorded `git_commit` of 22 archived experiments unreachable: exp116-129 and exp168-175. Their `experiments/expNNN__*/` records, `index.csv` rows and shipped weights remain; only the code state is gone. This is accepted: such a commit failing to resolve is known, not a defect to report or recover.

Origin: owner decision, 2026-09-07.

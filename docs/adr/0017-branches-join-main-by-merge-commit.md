# A work branch joins `main` by a merge commit, never a fast-forward

Every work branch merged into `main` gets its own merge commit (`git merge --no-ff <branch>`), even when `main` has not moved and a fast-forward is possible. So `git log --first-parent main` lists one entry per branch, the branch's commits stay grouped under it, and the whole branch can be reverted as one (`git revert -m 1 <merge>`) without rewriting history ([ADR 0009](0009-simple-merges-only.md)). Narrows ADR 0009, which allowed either: fast-forwarding `main` is still how the VM sync of OPS-1 §7 moves `main` onto the merge commit it built on a `tmp/*` branch, since that sync joins two copies of `main`, not a branch into it.

Source: owner decision 2026-10-10.

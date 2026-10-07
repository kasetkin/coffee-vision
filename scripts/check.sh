#!/usr/bin/env bash
# The repo's guardrail, in two tiers.
#
#   scripts/check.sh          fast tier (~25 s): .githooks/pre-push runs it before a push to origin
#   scripts/check.sh --full   every test (~4.5 min): scripts/deploy_webapp.sh runs it before a deploy
#
# Runs coffeecv.leak_check (no public IPs, emails, credentials or photo GPS in what git would publish, ~6 s),
# coffeecv.ticket_check (the HTML tickets, plans and ADRs keep docs/agents/issue-tracker.md's rules, <1 s),
# ruff check (the lint rules in pyproject.toml's [tool.ruff], over the working tree, <1 s),
# coffeecv.coverage_report (dataset/ vs classes.txt vs dvc.yaml, ~5 s) and the unittest suite. The
# fast tier sets COFFEECV_FAST_TESTS=1, which skips the tests marked @real_data (tests/_tiers.py: real
# photos or real weights, seconds each). Every step always runs, so one report shows every failure. Full
# output goes to a log; on failure the tail is printed with the log's path. The hook is wired by
# `git config core.hooksPath .githooks`, which the devcontainer's postCreateCommand runs.
set -uo pipefail

case "${1:-}" in
  "")     export COFFEECV_FAST_TESTS=1; TIER=fast ;;
  --full) unset COFFEECV_FAST_TESTS; TIER=full ;;
  *)      echo "usage: scripts/check.sh [--full]" >&2; exit 2 ;;
esac

cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
LOG_DIR=$(mktemp -d /tmp/check.XXXXXX)
failed=()

run() {  # run <name> <command...>
  local name=$1; shift
  local log="$LOG_DIR/${name%% *}.log"
  local start=$SECONDS
  if "$@" > "$log" 2>&1; then
    echo "ok    $name ($((SECONDS - start)) s)"
  else
    echo "FAIL  $name ($((SECONDS - start)) s), full log: $log"
    tail -40 "$log" | sed 's/^/      /'
    failed+=("$name")
  fi
}

run leak_check python -m coffeecv.leak_check
run ticket_check python -m coffeecv.ticket_check
run ruff python -m ruff check --output-format concise
run coverage_report python -m coffeecv.coverage_report
run "unittest ($TIER)" python -m unittest discover -s tests

if [ ${#failed[@]} -gt 0 ]; then
  echo "check ($TIER) FAILED: ${failed[*]}"
  exit 1
fi
echo "check ($TIER) passed"

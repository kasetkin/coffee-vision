#!/usr/bin/env bash
# Link the promoted skills of the mattpocock-skills submodule into this project's .claude/skills/.
#
#   scripts/link_mattpocock_skills.sh
#
# The skill list is the submodule's own .claude-plugin/plugin.json `skills` array (upstream keeps every
# promoted skill in it; in-progress/, misc/ and deprecated/ are not in it). Links are RELATIVE
# (../../mattpocock-skills/...) so they resolve both in the devcontainer and on the host, and can be
# committed. Re-run after `git submodule update --remote`: it adds new skills and removes links to
# skills that left the list. It never deletes a real file or directory -- it stops instead.
set -euo pipefail

REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
SUB=mattpocock-skills
MANIFEST="$REPO/$SUB/.claude-plugin/plugin.json"
DEST="$REPO/.claude/skills"
REL="../../$SUB"   # from $DEST back to the repo root, then into the submodule

if [ ! -f "$MANIFEST" ]; then
  echo "error: $MANIFEST not found -- run: git submodule update --init $SUB" >&2
  exit 1
fi

mapfile -t paths < <(python3 -c '
import json, sys
for p in json.load(open(sys.argv[1]))["skills"]:
    print(p.removeprefix("./"))
' "$MANIFEST")

mkdir -p "$DEST"
declare -A wanted=()
for p in "${paths[@]}"; do
  name=$(basename "$p")
  target="$DEST/$name"
  if [ ! -f "$REPO/$SUB/$p/SKILL.md" ]; then
    echo "error: $SUB/$p/SKILL.md missing (listed in plugin.json)" >&2
    exit 1
  fi
  if [ -n "${wanted[$name]:-}" ]; then
    echo "error: two listed skills are both named $name" >&2
    exit 1
  fi
  if [ -e "$target" ] && [ ! -L "$target" ]; then
    echo "error: $target exists and is not a symlink -- not touching it" >&2
    exit 1
  fi
  wanted[$name]=1
  ln -sfn "$REL/$p" "$target"
done
echo "linked ${#wanted[@]} skills into ${DEST#"$REPO"/}"

# Prune our own links (those pointing into the submodule) whose skill is no longer listed.
for link in "$DEST"/*; do
  [ -L "$link" ] || continue
  name=$(basename "$link")
  case "$(readlink "$link")" in
    "$REL"/*)
      if [ -z "${wanted[$name]:-}" ]; then
        rm "$link"
        echo "removed stale link $name"
      fi
      ;;
  esac
done

#!/usr/bin/env bash
# Makes ~/.ssh a symlink to /workspace/.private_ssh/ (gitignored, in the workspace bind mount), so the SSH
# config (the VM's host alias), its keys and known_hosts survive a container rebuild; ~/.ssh itself is
# inside the container and does not. Runs on container create (postCreateCommand); safe to re-run.
#
# Container setup creates a real ~/.ssh before this runs, holding a known_hosts that lacks the entries
# the container added (likely VS Code copying the host's). Its contents are merged in, then it is removed:
#   - known_hosts: lines .private_ssh/known_hosts lacks are appended;
#   - a file .private_ssh/ lacks is copied in; an identical one is dropped;
#   - any other file that differs is not merged: the directory is then kept as ~/.ssh.pre-link.<epoch>,
#     with a warning, so nothing is lost.
# Also points this clone's git at the devcontainer's GitHub key, when the key exists.
# PRIVATE_SSH_DIR overrides the target (for testing).
set -euo pipefail

src=${PRIVATE_SSH_DIR:-/workspace/.private_ssh}
mkdir -p "$src"
chmod 700 "$src"

if [ -L ~/.ssh ]; then
  if [ "$(readlink ~/.ssh)" != "$src" ]; then
    echo "link-private-ssh: ~/.ssh already links to $(readlink ~/.ssh), not $src; left as it is" >&2
    exit 1
  fi
else
  if [ -d ~/.ssh ]; then
    unmerged=0
    for f in ~/.ssh/* ~/.ssh/.[!.]*; do
      [ -e "$f" ] || continue
      name=$(basename "$f")
      dest="$src/$name"
      if [ ! -e "$dest" ]; then
        cp -a "$f" "$dest"
      elif cmp -s "$f" "$dest"; then
        :
      elif [ "$name" = known_hosts ]; then
        grep -vxF -f "$dest" "$f" >> "$dest" || true
      else
        unmerged=1
        echo "link-private-ssh: ~/.ssh/$name differs from $dest and is not merged" >&2
      fi
    done
    if [ "$unmerged" = 1 ]; then
      aside=~/.ssh.pre-link."$(date +%s)"
      mv ~/.ssh "$aside"
      echo "link-private-ssh: the old ~/.ssh is kept as $aside" >&2
    else
      rm -rf ~/.ssh
    fi
  fi
  ln -s "$src" ~/.ssh
fi

key="$src/coffee-vision-devcontainer"
if [ -f "$key" ]; then
  git -C /workspace config core.sshCommand "ssh -i $key -o IdentitiesOnly=yes"
fi
echo "link-private-ssh: ~/.ssh -> $src"

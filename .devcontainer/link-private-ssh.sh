#!/usr/bin/env bash
# Makes ~/.ssh a symlink to /workspace/.private_ssh/ (gitignored, in the workspace bind mount), so the SSH
# config (the VM's host alias), its keys and known_hosts survive a container rebuild; ~/.ssh itself is
# inside the container and does not. Runs on container create (postCreateCommand); safe to re-run.
#
# A real ~/.ssh found here (a container made before this script) has each file copied in unless
# .private_ssh/ already has one by that name, and is then moved aside to ~/.ssh.pre-link.<epoch>, not
# deleted. Also points this clone's git at the devcontainer's GitHub key, when the key exists.
set -euo pipefail

src=/workspace/.private_ssh
mkdir -p "$src"
chmod 700 "$src"

if [ -L ~/.ssh ]; then
  if [ "$(readlink ~/.ssh)" != "$src" ]; then
    echo "link-private-ssh: ~/.ssh already links to $(readlink ~/.ssh), not $src; left as it is" >&2
    exit 1
  fi
else
  if [ -d ~/.ssh ]; then
    for f in ~/.ssh/* ~/.ssh/.[!.]*; do
      [ -e "$f" ] || continue
      [ -e "$src/$(basename "$f")" ] || cp -a "$f" "$src/"
    done
    mv ~/.ssh ~/.ssh.pre-link."$(date +%s)"
  fi
  ln -s "$src" ~/.ssh
fi

key="$src/coffee-vision-devcontainer"
if [ -f "$key" ]; then
  git -C /workspace config core.sshCommand "ssh -i $key -o IdentitiesOnly=yes"
fi
echo "link-private-ssh: ~/.ssh -> $src"

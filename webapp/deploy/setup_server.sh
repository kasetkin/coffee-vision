#!/usr/bin/env bash
# Prerequisites for the coffee-cv web service on a fresh Debian or Ubuntu box, the inference host (ADR 0031): nginx,
# the sysctl that lets it bind the tailnet address, and the SSH firewall rule (ufw itself is left disabled). Nothing
# else. No TLS certificate and no public port: the front terminates TLS and reaches this host over the tailnet, in
# plain HTTP (ticket OPS-15 D5).
#
# Everything the service itself needs -- its system user, /opt/coffee-cv, uv, the log directory, and
# every rendered file (systemd unit, nginx site, logrotate) -- belongs to scripts/deploy_webapp.sh
# (--bootstrap once, then one command per deploy). A fresh box has no release to serve until that
# first deploy. See webapp/README.md and docs/ops1_release_isolation_plan.html §4.7.
#
# Runs apt, sysctl and ufw: never while a training sweep is running on the box.
#
# Usage: sudo ./webapp/deploy/setup_server.sh
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "ERROR: run as root (sudo) -- needed for apt/nginx/sysctl/ufw." >&2
  exit 1
fi

echo "== 1. nginx =="
apt-get install -y nginx

echo "== 2. binding the tailnet address before it is up =="
# nginx listens only on the host's tailnet address (scripts/deploy_webapp.sh's INFERENCE_ADDR). At boot tailscaled may
# not have brought that address up when nginx starts, and a bind to an address the host does not yet have fails, so
# nginx would not start. ip_nonlocal_bind lets the bind succeed; the socket answers once the address is up.
printf 'net.ipv4.ip_nonlocal_bind = 1\n' > /etc/sysctl.d/60-coffee-cv.conf
sysctl -p /etc/sysctl.d/60-coffee-cv.conf

echo "== 3. firewall rules (ufw is not enabled) =="
# The rules only: enabling ufw is off for now (owner, 2026-10-12), because the host also runs other
# services whose ports these rules don't cover. An inactive ufw records the rules and filters nothing,
# so a later `ufw enable` starts with SSH already allowed.
if command -v ufw >/dev/null; then
  ufw allow 22/tcp
  # No rule for nginx's port: it listens on the tailnet address only, which only tailnet nodes reach.
else
  echo "   ufw is not installed: no rules recorded"
fi

systemctl enable nginx

cat <<'EOF'

Done.

Next, from your workstation (docs/ops1_release_isolation_plan.html, webapp/README.md):

    scripts/deploy_webapp.sh --bootstrap
    DOMAIN=... scripts/deploy_webapp.sh <commit on origin/main> <model name>

The deploy renders /etc/nginx/conf.d/coffee-cv.conf, listening on this host's tailnet address
(`tailscale ip -4`, or INFERENCE_ADDR), plain HTTP: no certificate is needed here.
EOF

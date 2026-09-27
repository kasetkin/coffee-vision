#!/usr/bin/env bash
# Prerequisites for the coffee-cv web service on a fresh Ubuntu 24.04 box: nginx, the TLS cert
# directory, and the firewall. Nothing else.
#
# Everything the service itself needs -- its system user, /opt/coffee-cv, uv, the log directory, and
# every rendered file (systemd unit, nginx site, logrotate) -- belongs to scripts/deploy_webapp.sh
# (--bootstrap once, then one command per deploy). A fresh box has no release to serve until that
# first deploy. See webapp/README.md and docs/ops1_release_isolation_plan.html §4.7.
#
# Runs apt and ufw: never while a training sweep is running on the box.
#
# Usage: sudo ./webapp/deploy/setup_server.sh
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "ERROR: run as root (sudo) -- needed for apt/nginx/ufw." >&2
  exit 1
fi

echo "== 1. nginx =="
apt-get install -y nginx

echo "== 2. cert directory (place your Porkbun files here -- see webapp/README.md) =="
mkdir -p /etc/nginx/ssl/coffee-cv
chmod 700 /etc/nginx/ssl/coffee-cv
# Porkbun's own download bundle uses domain.cert.pem/private.key.pem/
# public.key.pem, world-writable (666) by default -- the nginx config
# (coffee-cv.nginx.conf.template) points directly at those filenames, no
# fullchain.pem build step needed since domain.cert.pem already has the full
# chain concatenated. Fix permissions here too, idempotently, so this doesn't
# have to be redone by hand on every cert renewal -- only touches files that
# are actually present, so it's a no-op before the first cert drop.
if [[ -f /etc/nginx/ssl/coffee-cv/private.key.pem ]]; then
  chmod 600 /etc/nginx/ssl/coffee-cv/private.key.pem
fi
for f in domain.cert.pem public.key.pem; do
  if [[ -f "/etc/nginx/ssl/coffee-cv/${f}" ]]; then
    chmod 644 "/etc/nginx/ssl/coffee-cv/${f}"
  fi
done
# Ubuntu's default site (sites-enabled/default) is left alone -- harmless, and
# touching it is one more thing that can go wrong for no benefit.

echo "== 3. firewall (SSH allowed before enabling, so this can't lock you out) =="
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable

systemctl enable nginx

cat <<'EOF'

Done.

Next, from your workstation (docs/ops1_release_isolation_plan.html, webapp/README.md):

    scripts/deploy_webapp.sh --bootstrap
    DOMAIN=... scripts/deploy_webapp.sh <commit on origin/main> <model name>

The deploy renders /etc/nginx/conf.d/coffee-cv.conf, which references cert files at
/etc/nginx/ssl/coffee-cv/. Put domain.cert.pem and private.key.pem (600) there first,
or its `nginx -t` fails and the deploy stops before the flip.
EOF

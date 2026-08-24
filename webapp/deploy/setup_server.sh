#!/usr/bin/env bash
# Bootstraps the coffee-cv web service on a fresh Ubuntu 24.04 box, or brings an
# existing one back in line with this repo's committed config. Every server-side
# command lives here -- not scattered across ad hoc SSH sessions -- so the exact
# configuration is reviewable and re-runnable rather than living only in shell
# history.
#
# Idempotent: safe to re-run after `git pull` on the VM, or to stand up a second
# identical box. It is NOT the redeploy path for ordinary code/model changes --
# see webapp/README.md for that (rsync + `systemctl restart coffee-cv-web`).
# `systemctl enable --now` is a no-op on an already-running unit, so re-running
# this script will not pick up new application code on its own.
#
# Usage: DOMAIN=yourdomain.example sudo -E ./webapp/deploy/setup_server.sh
set -euo pipefail

if [[ -z "${DOMAIN:-}" ]]; then
  echo "ERROR: DOMAIN environment variable is not set." >&2
  echo "Usage: DOMAIN=yourdomain.example sudo -E $0" >&2
  exit 1
fi

if [[ "${EUID}" -ne 0 ]]; then
  echo "ERROR: run as root (sudo -E) -- needed for apt/systemd/nginx/ufw." >&2
  exit 1
fi

# Resolve paths relative to this script's own location, not the caller's CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

APP_USER=alioth
VENV="/home/${APP_USER}/coffee-vision-venv"

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

echo "== 3. python deps, installed as ${APP_USER} (not root -- the venv is theirs) =="
sudo -u "${APP_USER}" "${VENV}/bin/pip" install --require-hashes -r "${REPO_ROOT}/webapp/requirements.txt"

echo "== 4. nginx site + systemd unit =="
DOMAIN="${DOMAIN}" envsubst '$DOMAIN' \
  < "${REPO_ROOT}/webapp/deploy/coffee-cv.nginx.conf.template" \
  > /etc/nginx/conf.d/coffee-cv.conf
cp "${REPO_ROOT}/webapp/deploy/coffee-cv-web.service" /etc/systemd/system/coffee-cv-web.service
# Ubuntu's default site (sites-enabled/default) is left alone -- harmless, and
# touching it is one more thing that can go wrong for no benefit.

echo "== 5. static frontend =="
mkdir -p /var/www/coffee-cv
cp "${REPO_ROOT}/webapp/static/index.html" /var/www/coffee-cv/index.html
chmod 644 /var/www/coffee-cv/index.html

echo "== 6. firewall (SSH allowed before enabling, so this can't lock you out) =="
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable

echo "== 7. services =="
systemctl daemon-reload
systemctl enable --now coffee-cv-web
systemctl enable nginx

cat <<'EOF'

Done.

coffee-cv-web is running on 127.0.0.1:8000 (check: systemctl status coffee-cv-web).

nginx was deliberately NOT reloaded: /etc/nginx/conf.d/coffee-cv.conf references
cert files that don't exist yet at /etc/nginx/ssl/coffee-cv/. Once your
domain.cert.pem and private.key.pem are in place there (private.key.pem at
600 permissions):

    nginx -t && systemctl reload nginx

If this is a fresh box, also confirm the torch pretrained-weight cache exists
at ~/.cache/torch/hub/checkpoints/ before relying on the service -- without it,
the app will try to reach download.pytorch.org on its next start. See
webapp/README.md.
EOF

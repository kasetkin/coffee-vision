#!/usr/bin/env bash
# Prepare the front (the proxy) for scripts/deploy_webapp.sh, once, as root (ticket OPS-15 plan P5, ADR 0031).
#
#   webapp/deploy/setup_front.sh [--print]
#
# Runs here, from a checkout, and drives the proxy as root through ROOT_HOST, the owner's own SSH alias: the deploy
# never uses it (D9). --print writes the script it would run there, and runs nothing. Environment (defaults in
# brackets): DOMAIN [required: the public site's host name]  ROOT_HOST [webproxy]  FRONT_APP_ROOT [/opt/coffee-cv]
#   DEPLOY_USER [coffee-deploy]  FRONT_SERVICE_USER [coffee-cv-front]  FRONT_PORT [the deploy's: the front site's port]
#   DEPLOY_PUBKEY [the public half of ROOT_HOST's IdentityFile]  NGINX_USER [www-data: nginx's worker user]
#   LOG_GROUP [adm: the group that may read the session log, OPS-3 D18]
#
# On the proxy it creates the deploy user (DEPLOY_PUBKEY may log in as it) and the check service's user; lets the
# deploy user run, through sudo, only what the deploy runs there: stop, start and restart coffee-cv-front, `nginx -t`
# and `systemctl reload nginx` (D9); installs uv and the uv-managed Python 3.12 the deploy pins; creates the release
# directory; and installs the check service's unit (enabled, not started: there is no release yet), conf.d's include
# of the release's site, the stream block's session log for the domain (/etc/nginx/coffee-cv-stream.conf, D20), and
# the logrotate rules, and makes the session logs readable by root and adm only (OPS-3 D18). It touches no other site and not nginx.conf (D8): the include is a glob that matches nothing
# until the first deploy, the session log takes effect only once the owner includes it in `stream {}` (plan P6), and
# `nginx -t` must pass before it ends. Safe to run again.
set -euo pipefail

REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
ROOT_HOST=${ROOT_HOST:-webproxy}
FRONT_APP_ROOT=${FRONT_APP_ROOT:-/opt/coffee-cv}
DEPLOY_USER=${DEPLOY_USER:-coffee-deploy}
FRONT_SERVICE_USER=${FRONT_SERVICE_USER:-coffee-cv-front}
NGINX_USER=${NGINX_USER:-www-data}
LOG_GROUP=${LOG_GROUP:-adm}
PRINT=""
case "${1:-}" in
  --print) PRINT=1 ;;
  "") ;;
  *) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
DOMAIN=${DOMAIN:-}
[[ -n "$DOMAIN" ]] || { echo "DOMAIN is not set (the public site's host name, for the stream block's session log)" >&2; exit 1; }

# The uv and Python the deploy pins (scripts/deploy_webapp.sh): one place for both hosts.
DEPLOY="$REPO/scripts/deploy_webapp.sh"
UV_VERSION=$(sed -n 's/^UV_VERSION=//p' "$DEPLOY")
UV_SHA256=$(sed -n 's/^UV_TARBALL_SHA256=//p' "$DEPLOY")
PY_VERSION=$(sed -n 's/^PY_VERSION=\${PY_VERSION:-\(.*\)}$/\1/p' "$DEPLOY")
FRONT_PORT=${FRONT_PORT:-$(sed -n 's/^FRONT_PORT=\${FRONT_PORT:-\(.*\)}$/\1/p' "$DEPLOY")}
[[ -n "$UV_VERSION" && -n "$UV_SHA256" && -n "$PY_VERSION" && -n "$FRONT_PORT" ]] \
  || { echo "could not read uv, Python and the front's port from $DEPLOY" >&2; exit 1; }

if [[ -z "${DEPLOY_PUBKEY:-}" ]]; then
  key=$(ssh -G "$ROOT_HOST" 2>/dev/null | awk '$1 == "identityfile" { print $2; exit }')
  DEPLOY_PUBKEY="${key/#\~/$HOME}.pub"
fi
[[ -n "$PRINT" || -f "$DEPLOY_PUBKEY" ]] || { echo "no public key at $DEPLOY_PUBKEY: set DEPLOY_PUBKEY" >&2; exit 1; }
# Empty only with --print and no key here: the printed script then leaves the deploy user's keys as they are (a
# re-run, run by hand on the proxy, where the deploy user already logs in).
PUBKEY=$(cat "$DEPLOY_PUBKEY" 2>/dev/null || true)

UNIT=$(python3 "$REPO/webapp/deploy/render_template.py" FRONT_SERVICE_USER="$FRONT_SERVICE_USER" \
         RELEASE_DIR="$FRONT_APP_ROOT/current" < "$REPO/webapp/deploy/coffee-cv-front.service.template")
LOGROTATE=$(python3 "$REPO/webapp/deploy/render_template.py" FRONT_SERVICE_USER="$FRONT_SERVICE_USER" \
              NGINX_USER="$NGINX_USER" LOG_GROUP="$LOG_GROUP" < "$REPO/webapp/deploy/coffee-cv-front.logrotate.template")
STREAM=$(python3 "$REPO/webapp/deploy/render_template.py" DOMAIN="$DOMAIN" < "$REPO/webapp/deploy/coffee-cv-stream.conf.template")
# The same line scripts/deploy_webapp.sh's site_include gives, which its preflight checks.
INCLUDE="include $FRONT_APP_ROOT/current/front.nginx*.conf;"

# The script the proxy runs, as root. Every value it needs is set at its top.
remote=$(cat <<EOF
set -euo pipefail
FRONT_APP_ROOT=$(printf '%q' "$FRONT_APP_ROOT")
DEPLOY_USER=$(printf '%q' "$DEPLOY_USER")
FRONT_SERVICE_USER=$(printf '%q' "$FRONT_SERVICE_USER")
NGINX_USER=$(printf '%q' "$NGINX_USER")
LOG_GROUP=$(printf '%q' "$LOG_GROUP")
UV_VERSION=$(printf '%q' "$UV_VERSION")
UV_SHA256=$(printf '%q' "$UV_SHA256")
PY_VERSION=$(printf '%q' "$PY_VERSION")
PUBKEY=$(printf '%q' "$PUBKEY")
UNIT=$(printf '%q' "$UNIT")
LOGROTATE=$(printf '%q' "$LOGROTATE")
STREAM=$(printf '%q' "$STREAM")
INCLUDE=$(printf '%q' "$INCLUDE")
EOF
)
remote+=$'\n'$(cat <<'EOF'
[[ "$EUID" -eq 0 ]] || { echo "ERROR: run as root"; exit 1; }
command -v nginx >/dev/null || { echo "ERROR: no nginx here"; exit 1; }
nginx -T 2>/dev/null | grep -q 'include /etc/nginx/conf.d/\*.conf;' \
  || { echo "ERROR: nginx.conf does not include /etc/nginx/conf.d/*.conf; this script does not edit nginx.conf (D8)"; exit 1; }

echo "== 1. users"
id "$FRONT_SERVICE_USER" >/dev/null 2>&1 \
  || useradd --system --user-group --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin "$FRONT_SERVICE_USER"
id "$DEPLOY_USER" >/dev/null 2>&1 || useradd --create-home --shell /bin/bash "$DEPLOY_USER"
home=$(getent passwd "$DEPLOY_USER" | cut -d: -f6)
install -d -o "$DEPLOY_USER" -g "$DEPLOY_USER" -m 0700 "$home/.ssh"
touch "$home/.ssh/authorized_keys"
if [[ -n "$PUBKEY" ]]; then
  grep -qxF "$PUBKEY" "$home/.ssh/authorized_keys" || printf '%s\n' "$PUBKEY" >> "$home/.ssh/authorized_keys"
  keys="its key installed"
else
  keys="no key given: its authorized_keys left as they are"
fi
chown "$DEPLOY_USER:$DEPLOY_USER" "$home/.ssh/authorized_keys"; chmod 0600 "$home/.ssh/authorized_keys"
echo "   $FRONT_SERVICE_USER (the check service), $DEPLOY_USER (the deploy; $keys)"

echo "== 2. sudo: only what the deploy runs here (D9)"
systemctl_bin=$(command -v systemctl); nginx_bin=$(command -v nginx)
sudoers=/etc/sudoers.d/coffee-cv-front
cat > "$sudoers.new" <<SUDOERS
# The front's deploy (scripts/deploy_webapp.sh, ticket OPS-15 D9): the check service, and nginx's test and reload.
$DEPLOY_USER ALL=(root) NOPASSWD: $systemctl_bin stop coffee-cv-front, $systemctl_bin start coffee-cv-front, \\
  $systemctl_bin restart coffee-cv-front, $nginx_bin -t, $systemctl_bin reload nginx
SUDOERS
chmod 0440 "$sudoers.new"
visudo -cf "$sudoers.new" >/dev/null || { rm -f "$sudoers.new"; echo "ERROR: the sudoers file does not parse"; exit 1; }
mv "$sudoers.new" "$sudoers"
echo "   $sudoers"

echo "== 3. uv $UV_VERSION and Python $PY_VERSION"
if [[ "$(/usr/local/bin/uv --version 2>/dev/null | awk '{print $2}')" != "$UV_VERSION" ]]; then
  dl=$(mktemp -d); trap 'rm -rf "$dl"' EXIT
  curl -fsSL -o "$dl/uv.tar.gz" "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-x86_64-unknown-linux-gnu.tar.gz"
  echo "$UV_SHA256  $dl/uv.tar.gz" | sha256sum -c --quiet - || { echo "ERROR: uv tarball sha256 mismatch"; exit 1; }
  tar -xzf "$dl/uv.tar.gz" -C "$dl"
  install -m 0755 "$dl/uv-x86_64-unknown-linux-gnu/uv" /usr/local/bin/uv
fi
install -d -o "$DEPLOY_USER" -g "$DEPLOY_USER" -m 0755 "$FRONT_APP_ROOT" "$FRONT_APP_ROOT/releases" \
  "$FRONT_APP_ROOT/uv-cache" "$FRONT_APP_ROOT/python"
# From a directory the deploy user can read, as that user with its own HOME, and with no config file: uv looks for a
# uv.toml in the working directory and its parents, and root's /root is closed to it (2026-11-02, the first run).
(cd "$FRONT_APP_ROOT" && sudo -H -u "$DEPLOY_USER" env UV_NO_CONFIG=1 UV_CACHE_DIR="$FRONT_APP_ROOT/uv-cache" \
  UV_PYTHON_INSTALL_DIR="$FRONT_APP_ROOT/python" UV_PYTHON_PREFERENCE=only-managed \
  /usr/local/bin/uv python install --no-bin "$PY_VERSION") 2>&1 | sed 's/^/   /'
echo "   $(/usr/local/bin/uv --version); Python under $FRONT_APP_ROOT/python; releases under $FRONT_APP_ROOT/releases"

echo "== 4. the check service's unit (enabled; it starts with the first release)"
printf '%s\n' "$UNIT" > /etc/systemd/system/coffee-cv-front.service
systemctl daemon-reload
systemctl enable coffee-cv-front 2>&1 | sed 's/^/   /'

echo "== 5. nginx: conf.d's include of the release's site, then nginx -t"
printf '%s\n' "$INCLUDE" > /etc/nginx/conf.d/coffee-cv-front.conf
nginx -t 2>&1 | sed 's/^/   /'
echo "   no reload: the include matches nothing until the first deploy"

echo "== 6. the stream block's session log (D20): installed here, included by the owner in stream {} (plan P6)"
printf '%s\n' "$STREAM" > /etc/nginx/coffee-cv-stream.conf
nginx -t 2>&1 | sed 's/^/   /'
echo "   /etc/nginx/coffee-cv-stream.conf"

echo "== 7. logrotate"
printf '%s\n' "$LOGROTATE" > /etc/logrotate.d/coffee-cv-front
logrotate -d /etc/logrotate.d/coffee-cv-front >/dev/null 2>&1 && echo "   /etc/logrotate.d/coffee-cv-front (checked)" \
  || { echo "ERROR: logrotate rejects /etc/logrotate.d/coffee-cv-front"; exit 1; }
# OPS-3 D18: the session logs, the only files with the visitor's IP, are root's and adm's to read; logrotate's
# `create` keeps each new one so, and this sets the ones already there.
for f in /var/log/nginx/coffee-cv-front.sessions*; do
  [[ -e "$f" ]] || continue
  chown "$NGINX_USER:$LOG_GROUP" "$f"; chmod 0640 "$f"
done
echo "   the session logs: 0640 $NGINX_USER:$LOG_GROUP"
echo "Done."
EOF
)

if [[ -n "$PRINT" ]]; then printf '%s\n' "$remote"; exit 0; fi
echo "Setting up the front on $ROOT_HOST, as root"
ssh -o BatchMode=yes "$ROOT_HOST" bash -s <<<"$remote"
host=$(ssh -G "$ROOT_HOST" | awk '$1 == "hostname" { print $2 }')
cat <<EOF

Next (plan P5): add the deploy's alias to ~/.ssh/config, beside $ROOT_HOST's:

    Host webproxy-deploy
        HostName $host
        User $DEPLOY_USER
        IdentityFile ${DEPLOY_PUBKEY%.pub}

then check it: ssh webproxy-deploy 'sudo -n nginx -t'.

At the cutover (plan P6), two edits to the stream block in /etc/nginx/nginx.conf, by hand, then nginx -t and a reload:

    upstream server-coffee { server 127.0.0.1:$FRONT_PORT; }   # was yappy's port: now the front's site
    include /etc/nginx/coffee-cv-stream.conf;           # inside stream {}: the session log (D20)

The session log is a stream-level access_log: an access_log of the 443 server's own would override it (none on
2026-11-02). Check after the reload that a visit to https://$DOMAIN/ adds a line to /var/log/nginx/coffee-cv-front.sessions.
EOF

#!/usr/bin/env bash
# Deploy the web service as one immutable release on two hosts (ADR 0031): the inference host and the front.
#
#   scripts/deploy_webapp.sh --bootstrap                     one-time setup of the inference host (user, /opt, uv, logs)
#   scripts/deploy_webapp.sh <git-ref> <model>               full deploy: stage both, stop both, smoke, flip both, verify
#   scripts/deploy_webapp.sh --stage-only <git-ref> <model>  preflight, build, stage both, smoke; no flip
#   scripts/deploy_webapp.sh --compare <release-id>          the release's answers within D7's bars of PyTorch's?
# The smoke and --compare stop the live service while they run (ADR 0026): photo answers are down, the page is up.
#   scripts/deploy_webapp.sh --verify                        check what `current` serves, on both hosts
#   scripts/deploy_webapp.sh --rollback                      switch both hosts back to `previous`
#
# Runs locally and drives both hosts over SSH. <git-ref> must resolve to a commit on origin/main. <model> is a name
# under models/ (e.g. dino3b16_segxl0v3_country_aug8_s123). Environment (defaults in brackets):
#   HOST [webappremote: the inference host]  APP_USER [HOST's SSH user]  APP_ROOT [/opt/coffee-cv]
#   SERVICE_USER [coffee-cv]  DOMAIN [required to deploy or verify: the public site's host name]
#   INFERENCE_ADDR [the host's own `tailscale ip -4`]  INFERENCE_PORT [12346: its nginx's plain-HTTP port]
#   FRONT_HOST [webproxy-deploy: the front]  FRONT_USER [its SSH user]  FRONT_APP_ROOT [/opt/coffee-cv]
#   FRONT_SERVICE_USER [coffee-cv-front]  FRONT_PORT [12347: the front site's loopback TLS port, the stream map's]
#   CHECK_PORT [8100: the check service's]  TLS_CERT, TLS_KEY [required to deploy: the front's certificate files]
#   DEPLOY_SKIP_CHECK [unset; 1 skips check.sh]
#
# One release ID, two trees (webapp/deploy/release_manifest.py): the inference host's (the app, the model, its nginx
# site) and the front's (the page, the check service, its nginx site). Each host's conf.d includes the site from
# `current`, so a flip or a rollback switches the site with the code, and each reload is preceded by `nginx -t` (D8).
# A full deploy stages both, stops both (the front's check service only: the proxy's nginx keeps serving its other
# sites, and the page, D14), flips both, starts both, reloads both nginx, and verifies through the domain; any
# failure after a flip puts both hosts back on `previous` and starts them.
#
# Every step that runs code on the host is a transient systemd unit that can write only /opt/coffee-cv,
# capped at CAP_CPU and CAP_MEM with no swap (default: the whole 4-core, 7.5 GB webappremote; plan §5's
# 1 CPU and 3 GB kept a training sweep on the old VM safe);
# preflight refuses without 10 GB free on /opt; the sweep's PIDs and log size are printed before the
# first VM step and after the last. The host fits one GPU worker, not two (ADR 0026), so before the
# release runs there the live service is stopped and MIN_MEM_GB (5) of available memory is required
# (make_room); it is started again when that is done, or on the new release after the flip, and on any
# failure before the flip. It never touches ~/coffee-vision*, never runs apt, and starts the service at
# a flip only if it is enabled.
set -euo pipefail

usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

# ---------------------------------------------------------------------------------------------- config
HOST=${HOST:-webappremote}
APP_ROOT=${APP_ROOT:-/opt/coffee-cv}
APP_USER=${APP_USER:-$(ssh -G "$HOST" 2>/dev/null | awk '$1 == "user" { print $2 }')}
SERVICE_USER=${SERVICE_USER:-coffee-cv}
DOMAIN=${DOMAIN:-}
LOG_ROOT=${LOG_ROOT:-/var/log/coffee-cv}
# Where the public checks go; the local rehearsal sets it empty to skip them (there is no nginx there).
PUBLIC_URL_GIVEN=${PUBLIC_URL+1}
# The host is the inference host (ADR 0031): its nginx listens only on its tailnet address, plain HTTP, for the front
# to reach. The address is the host's own, read in preflight unless given, so no tailnet address enters git.
INFERENCE_ADDR=${INFERENCE_ADDR:-}
INFERENCE_PORT=${INFERENCE_PORT:-12346}
# The front (ADR 0031): the proxy, set up once by webapp/deploy/setup_front.sh, as root (plan P5). Its deploy user may
# only stop, start and restart the check service, test and reload nginx, and write its own release directory (D8, D9).
FRONT_HOST=${FRONT_HOST:-webproxy-deploy}
FRONT_USER=${FRONT_USER:-$(ssh -G "$FRONT_HOST" 2>/dev/null | awk '$1 == "user" { print $2 }')}
FRONT_APP_ROOT=${FRONT_APP_ROOT:-/opt/coffee-cv}
FRONT_SERVICE_USER=${FRONT_SERVICE_USER:-coffee-cv-front}
FRONT_SYSTEM_ROOT=${FRONT_SYSTEM_ROOT:-}   # as SYSTEM_ROOT below, for the front
FRONT_PORT=${FRONT_PORT:-12347}
CHECK_PORT=${CHECK_PORT:-8100}
TLS_CERT=${TLS_CERT:-}
TLS_KEY=${TLS_KEY:-}
FRONT_MIN_FREE_MB=${FRONT_MIN_FREE_MB:-200}   # a front release is a few hundred KB and its venv; the proxy is full
# The public site is the front, on the proxy in front of the host (ticket OPS-15), so no port.
PUBLIC_URL=${PUBLIC_URL-${DOMAIN:+https://$DOMAIN}}
# Prefix for /etc and /run -- empty on the VM; a scratch directory in the local rehearsal.
SYSTEM_ROOT=${SYSTEM_ROOT:-}
PORT=${PORT:-8000}
SMOKE_PORT=${SMOKE_PORT:-8001}
CAP_CPU=${CAP_CPU:-400%}
SMOKE_CPU=${SMOKE_CPU:-400%}
CAP_MEM=${CAP_MEM:-7G}
MIN_MEM_GB=${MIN_MEM_GB:-5}
MIN_FREE_GB=${MIN_FREE_GB:-10}
LOCAL_PY=${LOCAL_PY:-python3}

UV_VERSION=0.12.19
UV_TARBALL_SHA256=23bf5552d220e0842b65c862097b2ebaeba0064b74eda5e565e77fd25969d8c8
UV_BIN=${UV_BIN:-/usr/local/bin/uv}
# The releases' Python: a uv-managed CPython under $APP_ROOT/python, never the host's own (a Debian
# host has no 3.12). --bootstrap installs it; the newest 3.12 uv $UV_VERSION offers.
PY_VERSION=${PY_VERSION:-3.12.14}
# --compare's PyTorch reference (ADR 0020): this machine's training environment, which has torch.
TORCH_PY=${TORCH_PY:-python}

REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
SCRATCH=${DEPLOY_SCRATCH:-${XDG_CACHE_HOME:-$HOME/.cache}/coffee-cv-deploy}
DEPLOY_DIR="$REPO/webapp/deploy"
FIX="$APP_ROOT/fixtures"          # on the VM: probe script, photos, expected answers -- never in a release
# The fixed photos every release is run through: webapp/deploy/fixtures.txt, one "<role> <sha256> <path>"
# per line (one smoke photo; the compare set). One file for the deploy, the probe, the test and the rehearsal.
FIXTURES="$DEPLOY_DIR/fixtures.txt"
# Ticket ML-19 D34: the fixtures as the inference host reads them, canonicalized here (canonical_fixtures.py): the
# photos, and their list in fixtures.txt's format, sent to the host in place of the originals.
CANON_FIX="$SCRATCH/fixtures"
fixtures() { awk -v r="$1" '$1 == r {print $3}' "$FIXTURES"; }   # fixtures <role>: repo paths

# Keepalives: a silent link is noticed in ~1 min instead of hanging, and an idle one is not dropped.
SSH_OPTS=(-o BatchMode=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=4)

say()  { printf '\n== %s\n' "$*"; }
note() { printf '   %s\n' "$*"; }
die()  { printf '\nDEPLOY FAILED: %s\n' "$*" >&2; exit 1; }

# vm VAR=value ... -- 'script': run a bash script on the VM (or locally, for HOST=local) with those
# variables set. The script travels as an argument, so stdin stays free and cannot be eaten by a child.
PRELUDE=$(cat <<'EOF'
set -euo pipefail
SUDO="sudo -n"
# A capped, sandboxed, transient unit (plan §5): writes only /opt/coffee-cv; the OOM killer and the
# CPU quota act inside it, never on the sweep.
capped() {
  local cpu=$1 wd=$2; shift 2
  $SUDO systemd-run --wait --pipe --collect --quiet --uid="$APP_USER" --gid="$APP_USER" \
    -p ProtectSystem=strict -p ProtectHome=read-only -p PrivateTmp=yes -p ReadWritePaths="$APP_ROOT" \
    -p CPUQuota="$cpu" -p MemoryMax="$CAP_MEM" -p MemorySwapMax=0 --nice=19 \
    --working-directory="$wd" "$@" </dev/null
}
managed_python() {   # managed_python <uv python subcommand> ...: only uv's own CPythons, under $APP_ROOT/python
  UV_PYTHON_INSTALL_DIR="$APP_ROOT/python" UV_PYTHON_PREFERENCE=only-managed "$UV_BIN" python "$@"
}
sweep_state() {
  local pids log="" size=""
  pids=$(pgrep -f '[t]rain_baseline' | sort -n | tr '\n' ' ' || true)
  for p in $pids; do
    l=$(readlink "/proc/$p/fd/1" 2>/dev/null || true)
    if [[ -f "$l" ]]; then log=$l; break; fi
  done
  [[ -n "$log" ]] && size=$(stat -c %s "$log")
  echo "SWEEP_PIDS=${pids% }"
  echo "SWEEP_LOG=$log"
  echo "SWEEP_LOG_SIZE=$size"
  # remote_launch.sh's convention: <name>.log beside <name>_status.log
  if [[ -n "$log" && -f "${log%.log}_status.log" ]]; then
    echo "SWEEP_STATUS=$(tail -1 "${log%.log}_status.log")"
  fi
}
EOF
)
vm() {
  local script=$1; shift
  local full="$PRELUDE"$'\n'"$script"
  local envs=(APP_ROOT="$APP_ROOT" APP_USER="$APP_USER" SERVICE_USER="$SERVICE_USER" LOG_ROOT="$LOG_ROOT"
              SYSTEM_ROOT="$SYSTEM_ROOT" CAP_CPU="$CAP_CPU" CAP_MEM="$CAP_MEM" UV_BIN="$UV_BIN" PY_VERSION="$PY_VERSION" "$@")
  if [[ "$HOST" == local ]]; then
    env "${envs[@]}" PATH="${STUB_PATH:+$STUB_PATH:}$PATH" bash -c "$full" vm </dev/null
  else
    ssh "${SSH_OPTS[@]}" -n "$HOST" "env $(printf '%q ' "${envs[@]}") bash -c $(printf '%q' "$full") vm"
  fi
}
to_vm() {   # to_vm <rsync args...> <local src> <remote dst>: rsync with the VM as destination
  local args=("$@") n=$#
  local dst=${args[$((n-1))]}
  unset 'args[$((n-1))]'
  if [[ "$HOST" == local ]]; then rsync "${args[@]}" "$dst"; return; fi
  # A dropped connection must not cost a 343 MB re-send: retry, resuming a half-sent file from
  # .rsync-partial (which rsync removes on success; the manifest check refuses it otherwise).
  local attempt
  for attempt in 1 2 3; do
    rsync -e "ssh ${SSH_OPTS[*]}" --partial-dir=.rsync-partial "${args[@]}" "$HOST:$dst" && return 0
    note "rsync attempt $attempt failed; retrying in 10 s"
    sleep 10
  done
  return 1
}
render() {  # render <ref> <template path> NAME=value...: a template from the commit being deployed
  local ref=$1 tpl=$2; shift 2
  git -C "$REPO" show "$ref:$tpl" | "$LOCAL_PY" "$DEPLOY_DIR/render_template.py" "$@"
}
kv() { sed -n "s/^$1=//p" <<<"$2" | tail -1; }
# fe VAR=value ... -- 'script': vm on the front, as its deploy user, under its own root and /etc prefix.
fe() {
  local script=$1; shift
  HOST=$FRONT_HOST vm "$script" APP_ROOT="$FRONT_APP_ROOT" APP_USER="$FRONT_USER" SERVICE_USER="$FRONT_SERVICE_USER" \
    SYSTEM_ROOT="$FRONT_SYSTEM_ROOT" "$@"
}
to_fe() { HOST=$FRONT_HOST to_vm "$@"; }
# What each host's conf.d file holds: the site of the release `current` points at. A glob, so the line is valid with
# no release yet, and a flip or a rollback, then a reload, switches the site with the code (OPS-9's goal).
site_include() { printf 'include %s/current/%s.nginx*.conf;' "$1" "$2"; }   # site_include <app root> inference|front

# ---------------------------------------------------------------------------------- preflight / proof
preflight() {
  say "Preflight ($HOST, read-only)"
  local out
  out=$(vm '
    echo "MEM_AVAIL_KB=$(awk "/MemAvailable/ {print \$2}" /proc/meminfo)"
    echo "OPT_FREE_KB=$(df -Pk "$APP_ROOT" | awk "NR==2 {print \$4}")"
    echo "UV=$("$UV_BIN" --version 2>/dev/null | awk "{print \$2}")"
    echo "PY=$(managed_python find "$PY_VERSION" 2>/dev/null || true)"
    echo "TAILNET_ADDR=$(tailscale ip -4 2>/dev/null | head -1 || true)"
    sweep_state')
  local mem free uv py
  mem=$(kv MEM_AVAIL_KB "$out"); free=$(kv OPT_FREE_KB "$out"); uv=$(kv UV "$out"); py=$(kv PY "$out")
  INFERENCE_ADDR=${INFERENCE_ADDR:-$(kv TAILNET_ADDR "$out")}
  PRE_SWEEP=$(grep '^SWEEP_' <<<"$out" || true)
  note "MemAvailable $((mem / 1048576)) GB (need $MIN_MEM_GB, checked by make_room before the release runs here), free on $APP_ROOT $((free / 1048576)) GB (need $MIN_FREE_GB), uv ${uv:-missing}, Python ${py:-missing}"
  note "sweep before: $(tr '\n' ' ' <<<"$PRE_SWEEP")"
  (( free >= MIN_FREE_GB * 1048576 )) || die "only $((free / 1048576)) GB free on $APP_ROOT; need ${MIN_FREE_GB} GB"
  [[ "$uv" == "$UV_VERSION" ]] || die "uv on the host is '${uv:-missing}', not $UV_VERSION -- run --bootstrap"
  [[ -n "$py" ]] || die "no uv-managed Python $PY_VERSION under $APP_ROOT/python -- run --bootstrap"
  note "nginx site listens on ${INFERENCE_ADDR:-?}:$INFERENCE_PORT (the host's tailnet address unless INFERENCE_ADDR is set)"
  [[ "$MODE" == compare || -n "$INFERENCE_ADDR" ]] \
    || die "no tailnet address for the nginx site: set INFERENCE_ADDR, or check \`tailscale ip -4\` on $HOST"
}

front_preflight() {   # the front: read-only, before anything is built or stopped
  say "Preflight the front ($FRONT_HOST, read-only)"
  [[ -n "$DOMAIN" ]] || die "DOMAIN is not set (the public host name, for the front's site)"
  [[ -n "$TLS_CERT" && -n "$TLS_KEY" ]] || die "TLS_CERT and TLS_KEY are not set (the front's certificate and key paths)"
  local out free uv py include unit
  out=$(fe '
    echo "FREE_KB=$(df -Pk "$APP_ROOT" | awk "NR==2 {print \$4}")"
    echo "UV=$("$UV_BIN" --version 2>/dev/null | awk "{print \$2}")"
    echo "PY=$(managed_python find "$PY_VERSION" 2>/dev/null || true)"
    echo "INCLUDE=$(cat "$SYSTEM_ROOT/etc/nginx/conf.d/coffee-cv-front.conf" 2>/dev/null || true)"
    echo "UNIT=$([[ -f "$SYSTEM_ROOT/etc/systemd/system/coffee-cv-front.service" ]] && echo installed || true)"')
  free=$(kv FREE_KB "$out"); uv=$(kv UV "$out"); py=$(kv PY "$out"); include=$(kv INCLUDE "$out"); unit=$(kv UNIT "$out")
  note "free on $FRONT_APP_ROOT $((free / 1024)) MB (need $FRONT_MIN_FREE_MB), uv ${uv:-missing}, Python ${py:-missing}"
  local setup="run webapp/deploy/setup_front.sh on $FRONT_HOST as root (plan P5)"
  (( free >= FRONT_MIN_FREE_MB * 1024 )) || die "only $((free / 1024)) MB free on the front's $FRONT_APP_ROOT"
  [[ "$uv" == "$UV_VERSION" && -n "$py" ]] || die "the front has no uv $UV_VERSION and Python $PY_VERSION: $setup"
  [[ "$include" == "$(site_include "$FRONT_APP_ROOT" front)" ]] || die "the front's nginx does not include the release's site: $setup"
  [[ "$unit" == installed ]] || die "the front has no coffee-cv-front.service: $setup"
}

postflight() {
  local out
  out=$(vm 'sweep_state')
  say "Sweep proof"
  note "before: $(tr '\n' ' ' <<<"${PRE_SWEEP:-}")"
  note "after:  $(tr '\n' ' ' <<<"$out")"
  local pids_before pids_after
  pids_before=$(kv SWEEP_PIDS "${PRE_SWEEP:-}"); pids_after=$(kv SWEEP_PIDS "$out")
  if [[ -n "$pids_before" && "$pids_before" == "$pids_after" ]]; then
    note "same sweep PIDs, log $(kv SWEEP_LOG_SIZE "${PRE_SWEEP:-}") -> $(kv SWEEP_LOG_SIZE "$out") bytes"
  elif [[ -n "$pids_before" ]]; then
    note "sweep PIDs changed -- check the status line above (a sweep also moves to its next fold on its own)"
  fi
}

# --------------------------------------------------------------------------------------------- guard
# The full test suite (scripts/check.sh --full) on the commit being deployed, before anything touches the
# VM. It runs on the working tree, so that tree must be <git-ref>'s commit with no tracked file modified.
# A pass is remembered per commit, so --stage-only and the deploy that follows run it once.
guard() {
  local ref=$1 sha stamp
  if [[ "${DEPLOY_SKIP_CHECK:-}" == 1 ]]; then say "Full check SKIPPED (DEPLOY_SKIP_CHECK=1)"; return; fi
  sha=$(git -C "$REPO" rev-parse --verify "$ref^{commit}") || die "cannot resolve $ref"
  stamp="$SCRATCH/full-check-passed-$sha"
  if [[ -f "$stamp" ]]; then say "Full check: passed earlier on ${sha:0:7}"; return; fi
  say "Full check on ${sha:0:7} (local, ~4.5 min)"
  [[ "$sha" == "$(git -C "$REPO" rev-parse HEAD)" ]] \
    || die "the full check runs on the working tree: check out $ref (${sha:0:7}) first"
  git -C "$REPO" diff --quiet HEAD -- \
    || die "tracked files have uncommitted changes, so the full check would not test ${sha:0:7}"
  "$REPO/scripts/check.sh" --full || die "scripts/check.sh --full failed on ${sha:0:7}"
  touch "$stamp"
}

# --------------------------------------------------------------------------------------------- build
build() {   # sets SHA, ID, STAGE, MANIFEST
  local ref=$1 model=$2
  say "Build $model from $ref (local)"
  git -C "$REPO" fetch -q origin main \
    || note "WARNING: could not fetch origin; checking against the last fetched origin/main ($(git -C "$REPO" rev-parse --short origin/main))"
  SHA=$(git -C "$REPO" rev-parse --verify "$ref^{commit}") || die "cannot resolve $ref"
  git -C "$REPO" merge-base --is-ancestor "$SHA" origin/main \
    || die "$SHA is not on origin/main -- deploy only commits on origin/main (push main first)"
  MANIFEST=$("$LOCAL_PY" "$DEPLOY_DIR/release_manifest.py" --ref "$SHA" --model "$model") \
    || die "no release manifest for $model at $SHA"
  # A complete release of this commit and model is reused, not rebuilt: the full deploy then flips the
  # very release that --stage-only smoked and --compare checked.
  REUSE=$(vm 'shopt -s nullglob
    for d in "$APP_ROOT"/releases/*-"$SUFFIX"; do if [[ -f "$d/.complete" ]]; then basename "$d"; fi; done | tail -1' \
    SUFFIX="${SHA:0:7}-$model")
  ID=${REUSE:-"$(date -u +%Y%m%dT%H%MZ)-${SHA:0:7}-$model"}
  STAGE="$SCRATCH/$ID"
  rm -rf "$STAGE"; mkdir -p "$STAGE"
  note "release $ID${REUSE:+ (already staged on the VM: reused)}"

  # shellcheck disable=SC2046
  git -C "$REPO" archive "$SHA" -- $(jq -r '.git[]' <<<"$MANIFEST") | tar -x -C "$STAGE"

  local path md5 src found sha
  while read -r path md5; do
    found=""
    for src in "$REPO/$path" "$REPO/.dvc/cache/files/md5/${md5:0:2}/${md5:2}"; do
      if [[ -f "$src" && "$(md5sum < "$src" | cut -c1-32)" == "$md5" ]]; then found=$src; break; fi
    done
    [[ -n "$found" ]] || die "$path: no copy in the working tree or the DVC cache has md5 $md5 (its .pt.dvc at $SHA)"
    install -D -m 0644 "$found" "$STAGE/$path"
    note "$path from ${found#"$REPO"/} (md5 $md5)"
  done < <(jq -r '.dvc | to_entries[] | "\(.key) \(.value)"' <<<"$MANIFEST")

  while read -r path sha; do
    [[ -f "$REPO/$path" ]] || die "$path is missing (the warm-up photo: dvc pull it)"
    [[ "$(sha256sum < "$REPO/$path" | cut -c1-64)" == "$sha" ]] \
      || die "$path does not match the sha256 its release manifest pins at $SHA"
    install -D -m 0644 "$REPO/$path" "$STAGE/$path"
    note "$path (sha256 ${sha:0:16}...)"
  done < <(jq -r '.pinned | to_entries[] | "\(.key) \(.value)"' <<<"$MANIFEST")

  "$LOCAL_PY" "$DEPLOY_DIR/write_build_info.py" --ref "$SHA" --model "$model" --out "$STAGE/webapp/BUILD_INFO.json" >/dev/null
  { printf 'COFFEE_CV_CHECKPOINT=models/%s.pt\n' "$model"
    jq -r '.env | to_entries[] | "\(.key)=\(.value)"' <<<"$MANIFEST"; } > "$STAGE/release.env"
  # The host's nginx site, in the release (its conf.d includes it from `current`, site_include).
  render "$SHA" webapp/deploy/coffee-cv.nginx.conf.template INFERENCE_ADDR="$INFERENCE_ADDR" \
    INFERENCE_PORT="$INFERENCE_PORT" PORT="$PORT" > "$STAGE/inference.nginx.conf"
  (cd "$STAGE" && find . -type f -printf '%P\n') \
    | "$LOCAL_PY" "$DEPLOY_DIR/release_manifest.py" --ref "$SHA" --model "$model" --check-staged - \
    || die "the staged tree does not match the manifest"

  check_fixtures
  canonical_fixtures
  note "expected smoke answers: the staged tree's own webapp.app, locally"
  local logs; logs=$(mktemp -d)
  # The models on this machine's CPU (COFFEE_CV_LITERT_ON_CPU): the release's accelerator is the host's (ML-7 P10).
  (cd "$STAGE" && COFFEE_CV_CHECKPOINT="models/$model.pt" COFFEE_CV_LOG_DIR="$logs" PYTHONDONTWRITEBYTECODE=1 \
      COFFEE_CV_LITERT_ON_CPU=1 COFFEE_CV_WARMUP_PHOTO="$REPO/$(fixtures smoke)" \
      "$LOCAL_PY" -B "$DEPLOY_DIR/release_probe.py" smoke "$CANON_FIX/$(basename "$(fixtures smoke)").jxl" \
      2>"$logs/stderr") \
      > "$SCRATCH/$ID.expected.json" || { tail -20 "$logs/stderr"; die "the staged tree failed locally"; }
  rm -rf "$logs"
  jq -r '"   /analyze \(.analyze.parts[1].verdict // .analyze.parts[1].part), top-1 \(.analyze.parts[1].ranked[0].id // "-"), model_sha \(.build.model_sha)"' \
    < <(sed -n 's/^PROBE //p' "$SCRATCH/$ID.expected.json")
  build_front
}

build_front() {   # sets FSTAGE: the front's tree of release $ID, from the same commit
  FSTAGE="$SCRATCH/$ID.front"
  rm -rf "$FSTAGE"; mkdir -p "$FSTAGE"
  # shellcheck disable=SC2046
  git -C "$REPO" archive "$SHA" -- $("$LOCAL_PY" "$DEPLOY_DIR/release_manifest.py" --ref "$SHA" --front | jq -r '.git[]') \
    | tar -x -C "$FSTAGE"
  install -D -m 0644 "$STAGE/webapp/BUILD_INFO.json" "$FSTAGE/webapp/BUILD_INFO.json"
  render "$SHA" webapp/deploy/coffee-cv-front.nginx.conf.template DOMAIN="$DOMAIN" FRONT_PORT="$FRONT_PORT" \
    CHECK_PORT="$CHECK_PORT" APP_ROOT="$FRONT_APP_ROOT" TLS_CERT="$TLS_CERT" TLS_KEY="$TLS_KEY" > "$FSTAGE/front.nginx.conf"
  # The check service's two settings (webapp/front/check_service.py, options): read by its unit at each start.
  printf 'COFFEE_CV_FRONT_LISTEN=127.0.0.1:%s\nCOFFEE_CV_INFERENCE_URL=http://%s:%s\n' \
    "$CHECK_PORT" "$INFERENCE_ADDR" "$INFERENCE_PORT" > "$FSTAGE/front.env"
  (cd "$FSTAGE" && find . -type f -printf '%P\n') \
    | "$LOCAL_PY" "$DEPLOY_DIR/release_manifest.py" --ref "$SHA" --front --check-staged - \
    || die "the front's staged tree does not match its manifest"
  note "the front's tree: $(find "$FSTAGE" -type f | wc -l) files, its site and front.env rendered"
}

check_fixtures() {
  local sha path
  [[ "$(fixtures smoke | wc -l)" == 1 ]] || die "$FIXTURES must name exactly one smoke photo"
  [[ -z "$(awk '!/^#/ && NF {print $3}' "$FIXTURES" | xargs -n1 basename | sort | uniq -d)" ]] \
    || die "$FIXTURES: photo basenames must be unique (the VM stores them flat)"
  while read -r _ sha path; do
    [[ -f "$REPO/$path" ]] || die "fixture photo $path is missing"
    [[ "$(sha256sum < "$REPO/$path" | cut -c1-64)" == "$sha" ]] || die "fixture photo $path does not match its sha256 in $FIXTURES"
  done < <(grep -v '^#' "$FIXTURES" | grep .)
}

canonical_fixtures() {   # ML-19 D34: every fixture canonicalized here, the store's machine (D51), into $CANON_FIX
  "$LOCAL_PY" "$DEPLOY_DIR/canonical_fixtures.py" "$FIXTURES" "$CANON_FIX" >/dev/null \
    || die "the fixtures could not be canonicalized"
}

send_fixtures() {   # the canonical fixtures and their list (ML-19 D34): the host takes canonical photos only
  local files=() name
  while read -r name; do files+=("$CANON_FIX/$name"); done < <(awk '!/^#/ && NF {print $3}' "$CANON_FIX/fixtures.canonical.txt")
  vm 'mkdir -p "$APP_ROOT/fixtures/photos" "$APP_ROOT/fixtures/expected"'
  to_vm -a --chmod=D755,F644 "${files[@]}" "$FIX/photos/"
  to_vm -a --chmod=F644 "$DEPLOY_DIR/release_probe.py" "$CANON_FIX/fixtures.canonical.txt" "$FIX/"
}

# --------------------------------------------------------------------------------------------- stage
stage() {
  say "Stage $ID on $HOST"
  local cur expect
  expect=$(jq -r '(.dvc + .pinned) | to_entries[] | "\(.key) \(.value)"' <<<"$MANIFEST")
  send_fixtures
  to_vm -a --chmod=F644 "$SCRATCH/$ID.expected.json" "$FIX/expected/$ID.json"
  if [[ -n "$REUSE" ]]; then
    vm 'R="$APP_ROOT/releases/$ID"
      while read -r path digest; do
        if (( ${#digest} == 32 )); then got=$(md5sum < "$R/$path" | cut -c1-32); else got=$(sha256sum < "$R/$path" | cut -c1-64); fi
        [[ "$got" == "$digest" ]] || { echo "$path changed since it was staged"; exit 1; }
      done <<<"$EXPECT"' ID="$ID" EXPECT="$expect" || die "the staged release $ID no longer matches its hashes"
    check_staged_listing
    note "reused: hashes and contents re-checked"
    return
  fi
  cur=$(vm '[[ -e "$APP_ROOT/releases/$ID" ]] && { echo "EXISTS"; exit 0; }
            if [[ -e "$APP_ROOT/current" ]]; then readlink -f "$APP_ROOT/current"; fi' ID="$ID")
  [[ "$cur" != EXISTS ]] || die "$APP_ROOT/releases/$ID already exists (incomplete: it is pruned by the next full deploy)"
  local t0=$SECONDS basis=() d
  # --copy-dest: a file found unchanged in an existing release is copied on the host, not sent again
  # (a full copy, never a hard link -- each release stays independent), and a changed one is sent as
  # a delta against it. The current release first, then the newest others: an upload cut off by a
  # dropped connection then costs only what it had not sent (rsync takes at most 20 such dirs).
  if [[ -n "$cur" ]]; then basis+=(--copy-dest="$cur/"); fi
  while read -r d; do
    if [[ -n "$d" && "$d" != "$cur" && ${#basis[@]} -lt 20 ]]; then basis+=(--copy-dest="$d/"); fi
  done < <(vm 'shopt -s nullglob
    for d in "$APP_ROOT"/releases/*/; do echo "$(stat -c %Y "$d") ${d%/}"; done | sort -rn | cut -d" " -f2-')
  to_vm -a --chmod=D755,F644 "${basis[@]}" "$STAGE/" "$APP_ROOT/releases/$ID/"
  note "rsync $((SECONDS - t0)) s"

  t0=$SECONDS
  vm '
    R="$APP_ROOT/releases/$ID"
    while read -r path digest; do
      if (( ${#digest} == 32 )); then got=$(md5sum < "$R/$path" | cut -c1-32); else got=$(sha256sum < "$R/$path" | cut -c1-64); fi
      [[ "$got" == "$digest" ]] || { echo "$path arrived corrupt"; exit 1; }
    done <<<"$EXPECT"
    echo "   venv: uv sync --locked (capped: $CAP_CPU CPU, $CAP_MEM)"
    capped "$CAP_CPU" "$R" \
      -E UV_CACHE_DIR="$APP_ROOT/uv-cache" -E UV_LINK_MODE=copy -E UV_PYTHON_DOWNLOADS=never \
      -E UV_PYTHON_INSTALL_DIR="$APP_ROOT/python" -E UV_PYTHON_PREFERENCE=only-managed \
      -E UV_PROJECT_ENVIRONMENT="$R/.venv" -E UV_NO_PROGRESS=1 \
      "$UV_BIN" sync --project webapp --locked --no-dev --compile-bytecode --python "$PY_VERSION" 2>&1 | tail -8 | sed "s/^/   /"
    # third_party/: the vendored segmenter, only in the release of a segmenter model (ticket ML-2).
    capped "$CAP_CPU" "$R" "$R/.venv/bin/python" -m compileall -q coffeecv webapp $([[ -d "$R/third_party" ]] && echo third_party)
  ' ID="$ID" EXPECT="$expect" || die "building the venv failed"
  note "venv $((SECONDS - t0)) s"

  check_staged_listing
  vm 'date -u +%FT%TZ > "$APP_ROOT/releases/$ID/.complete"' ID="$ID"
  note "complete: $APP_ROOT/releases/$ID"
}

stage_front() {   # release $ID on the front: uploaded unless complete there, checked file by file, its venv built
  say "Stage $ID on the front ($FRONT_HOST)"
  local expect listing unit
  expect=$(cd "$FSTAGE" && find . -type f -printf '%P\n' | sort | while read -r f; do
             printf '%s %s\n' "$f" "$(sha256sum < "$f" | cut -c1-64)"; done)
  # A complete release of this ID is reused if its files are this build's, BUILD_INFO.json aside (it records when it
  # was built, as on the inference host); one that differs is built again, unless the front runs it or keeps it.
  local state
  state=$(fe '
    R="$APP_ROOT/releases/$ID"
    [[ -f "$R/.complete" ]] || { echo NEW; exit 0; }
    while read -r path digest; do
      [[ "$(sha256sum < "$R/$path" 2>/dev/null | cut -c1-64)" == "$digest" ]] || { echo STALE; exit 0; }
    done <<<"$EXPECT"
    echo REUSE' ID="$ID" EXPECT="$(grep -v '^webapp/BUILD_INFO.json ' <<<"$expect")")
  if [[ "$state" == REUSE ]]; then
    expect=$(grep -v '^webapp/BUILD_INFO.json ' <<<"$expect")
    note "already staged on the front: reused"
  else
    if [[ "$state" == STALE ]]; then
      fe '[[ "$(readlink -f "$APP_ROOT/current")" != "$APP_ROOT/releases/$ID" &&
             "$(readlink -f "$APP_ROOT/previous" 2>/dev/null || true)" != "$APP_ROOT/releases/$ID" ]]' ID="$ID" \
        || die "the front's releases/$ID differs from this build, and the front runs or keeps it"
      note "the front's staged releases/$ID differs from this build: staged again"
    fi
    fe 'rm -rf "$APP_ROOT/releases/$ID" && mkdir -p "$APP_ROOT/releases"' ID="$ID"
    to_fe -a --chmod=D755,F644 "$FSTAGE/" "$FRONT_APP_ROOT/releases/$ID/" || die "uploading the front's release failed"
  fi
  fe '
    R="$APP_ROOT/releases/$ID"
    while read -r path digest; do
      [[ "$(sha256sum < "$R/$path" | cut -c1-64)" == "$digest" ]] || { echo "$path arrived corrupt, or changed"; exit 1; }
    done <<<"$EXPECT"
    if [[ ! -f "$R/.complete" ]]; then
      # No dependencies (D7): the venv is the managed Python alone. Not capped: the front deploy user may not run
      # systemd-run (D9), and there is nothing to build.
      UV_CACHE_DIR="$APP_ROOT/uv-cache" UV_PYTHON_DOWNLOADS=never UV_PYTHON_INSTALL_DIR="$APP_ROOT/python" \
        UV_PYTHON_PREFERENCE=only-managed UV_PROJECT_ENVIRONMENT="$R/.venv" UV_NO_PROGRESS=1 \
        "$UV_BIN" sync --project "$R/webapp/front" --locked --no-dev --python "$PY_VERSION" 2>&1 | tail -3 | sed "s/^/   /"
      date -u +%FT%TZ > "$R/.complete"
    fi' ID="$ID" EXPECT="$expect" || die "staging the front's release failed"
  listing=$(fe 'cd "$APP_ROOT/releases/$ID" && find . -path ./.venv -prune -o -type f -printf "%P\n"' ID="$ID")
  "$LOCAL_PY" "$DEPLOY_DIR/release_manifest.py" --ref "$SHA" --front --check-staged - <<<"$listing" \
    || die "the front's release on $FRONT_HOST does not match its manifest"
  # The unit is the owner's to install (D9): it must be this release's, or the release would run under another.
  unit=$(render "$SHA" webapp/deploy/coffee-cv-front.service.template FRONT_SERVICE_USER="$FRONT_SERVICE_USER" \
           RELEASE_DIR="$FRONT_APP_ROOT/current")
  fe '[[ "$(cat "$SYSTEM_ROOT/etc/systemd/system/coffee-cv-front.service")" == "$UNIT" ]]' UNIT="$unit" \
    || die "the front's installed coffee-cv-front.service is not this release's: run webapp/deploy/setup_front.sh on $FRONT_HOST as root (plan P5)"
  note "complete: $FRONT_APP_ROOT/releases/$ID"
}

check_staged_listing() {   # the release on the VM holds exactly the manifest (+ .venv, .complete, bytecode)
  local listing
  listing=$(vm 'cd "$APP_ROOT/releases/$ID" && find . -path ./.venv -prune -o -type f -printf "%P\n"' ID="$ID")
  "$LOCAL_PY" "$DEPLOY_DIR/release_manifest.py" --ref "$SHA" --model "$MODEL" --check-staged - <<<"$listing" \
    || die "the release on the VM does not match the manifest"
}

# ------------------------------------------------------------------------------ the live service (ADR 0026)
# The host fits one GPU worker, not two: a worker costs about 3.3 GB of MemAvailable of webappremote's 7.7 GB
# (OPS-10). So before the release runs on the host (the smoke, --compare's run) the live service is stopped, and
# only once everything the release needs is on the host: every file uploaded and checked against its digest, the
# venv built, the release marked complete, the fixtures and expected answers sent (the owner, 2026-10-20). After
# the stop only the rendered unit and nginx site are written. It starts again when the release's run is done
# (restore_live), on the new release after a flip, or from the EXIT trap on any failure or interrupt. A host where
# stopping it cannot free MIN_MEM_GB is refused before anything stops.
LIVE_STOPPED_AT=""
make_room() {
  local out active avail svc
  out=$(vm '
    echo "ACTIVE=$(systemctl is-active coffee-cv-web 2>/dev/null || true)"
    echo "MEM_AVAIL_KB=$(awk "/MemAvailable/ {print \$2}" /proc/meminfo)"
    cur=$(systemctl show -p MemoryCurrent --value coffee-cv-web 2>/dev/null || true)
    [[ "$cur" =~ ^[0-9]+$ ]] || cur=0
    echo "SERVICE_KB=$((cur / 1024))"')
  active=$(kv ACTIVE "$out"); avail=$(kv MEM_AVAIL_KB "$out"); svc=$(kv SERVICE_KB "$out")
  if [[ "$active" == active ]]; then
    (( avail + svc >= MIN_MEM_GB * 1048576 )) \
      || die "stopping coffee-cv-web would leave about $(((avail + svc) / 1024)) MB available ($((avail / 1024)) MB + its $((svc / 1024)) MB); need ${MIN_MEM_GB} GB -- nothing stopped"
    say "Stop the live service (ADR 0026): photo answers are down until it starts again; the page stays up"
    vm '$SUDO systemctl stop coffee-cv-web'
    LIVE_STOPPED_AT=$SECONDS
    note "stopped at $(date -u +%T) UTC"
    avail=$(vm 'awk "/MemAvailable/ {print \$2}" /proc/meminfo')
  fi
  note "MemAvailable $((avail / 1024)) MB (need $MIN_MEM_GB GB)"
  (( avail >= MIN_MEM_GB * 1048576 )) || die "only $((avail / 1024)) MB available memory; need ${MIN_MEM_GB} GB"
}

restore_live() {   # starts what this run stopped, on either host, if nothing has started it since
  if [[ -n "$LIVE_STOPPED_AT" ]]; then
    local down=$((SECONDS - LIVE_STOPPED_AT))
    LIVE_STOPPED_AT=""
    say "Start the live service again (stopped ${down} s)"
    vm '$SUDO systemctl start coffee-cv-web' \
      || note "WARNING: coffee-cv-web did not start -- start it by hand on $HOST: sudo systemctl start coffee-cv-web"
  fi
  if [[ -n "$FRONT_STOPPED" ]]; then
    FRONT_STOPPED=""
    say "Start the front's check service again"
    fe '$SUDO systemctl start coffee-cv-front' \
      || note "WARNING: coffee-cv-front did not start -- start it by hand on $FRONT_HOST: sudo systemctl start coffee-cv-front"
  fi
}

# The front's side of D14: only its check service stops. The proxy's nginx serves other sites and is never stopped; it
# keeps serving the page, and answers /analyze with D12's "the classifier is offline" while the service is down.
FRONT_STOPPED=""
stop_front() {
  [[ "$(fe 'systemctl is-active coffee-cv-front 2>/dev/null || true')" == active ]] || return 0
  say "Stop the front's check service (D14): /analyze answers \"the classifier is offline\"; the page stays up"
  fe '$SUDO systemctl stop coffee-cv-front'
  FRONT_STOPPED=1
}

# --------------------------------------------------------------------------------------------- smoke
smoke() {   # smoke <release id>: the release, in production's own sandbox, on port $SMOKE_PORT
  local id=$1
  say "Smoke $id in the real sandbox (User=$SERVICE_USER, capped at $SMOKE_CPU / $CAP_MEM)"
  local unit
  unit=$(render "$SHA" webapp/deploy/coffee-cv-web.service.template SERVICE_USER="$SERVICE_USER" \
           RELEASE_DIR="$APP_ROOT/releases/$id" PORT="$SMOKE_PORT" LOG_DIR="$LOG_ROOT/smoke")
  local out rc=0
  out=$(vm '
    U=coffee-cv-smoke.service
    RUN="$SYSTEM_ROOT/run/systemd/system"
    cleanup() {
      $SUDO systemctl stop "$U" 2>/dev/null || true
      $SUDO rm -rf "$RUN/$U" "$RUN/$U.d" "$LOG_ROOT/smoke"
      $SUDO systemctl daemon-reload
    }
    trap cleanup EXIT
    printf "%s\n" "$UNIT" | $SUDO tee "$RUN/$U" >/dev/null
    $SUDO mkdir -p "$RUN/$U.d"
    printf "[Service]\nCPUQuota=%s\nMemoryMax=%s\nMemorySwapMax=0\n" "$SMOKE_CPU" "$CAP_MEM" | $SUDO tee "$RUN/$U.d/limits.conf" >/dev/null
    $SUDO install -d -o "$SERVICE_USER" -g "$SERVICE_USER" -m 0755 "$LOG_ROOT/smoke"
    $SUDO systemctl daemon-reload
    t0=$(date +%s.%N)
    $SUDO systemctl start "$U"
    ready=""
    for _ in $(seq 1 360); do
      [[ "$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$SMOKE_PORT/" || true)" == 404 ]] && { ready=1; break; }
      systemctl is-active --quiet "$U" || break
      sleep 0.5
    done
    if [[ -z "$ready" ]]; then
      echo "SMOKE_FAILED never became ready"
      $SUDO journalctl -u "$U" --no-pager -n 40 -o cat | sed "s/^/   | /"
      exit 1
    fi
    echo "SMOKE_READY_S $(awk -v a="$(date +%s.%N)" -v b="$t0" "BEGIN {printf \"%.1f\", a - b}")"
    P="$APP_ROOT/fixtures/photos/$PHOTO"
    # One upload to /analyze (ticket OPS-7): its status, time and type, then each line of its answer as a SMOKE_PART.
    body=$(mktemp)
    meta=$(curl -s -o "$body" -w "%{http_code} %{time_total} %{content_type}" -F "photo=@$P" "http://127.0.0.1:$SMOKE_PORT/analyze")
    echo "SMOKE_ANALYZE $meta"
    echo "SMOKE_LATENCY_S $(cut -d" " -f2 <<<"$meta")"
    sed "s/^/SMOKE_PART /" "$body"
    rm -f "$body"
    echo "SMOKE_RSS $(systemctl show -p MemoryPeak --value "$U")"
    # The boot record of the worker (ADR 0021): its time, memory and, on a GPU host, the GPU proof of ADR 0020.
    echo "SMOKE_BOOT $($SUDO journalctl -u "$U" --no-pager -o cat | sed -n "s/^INFO:webapp.app:boot //p" | tail -1)"
    echo "SMOKE_LOG $(grep "\"endpoint\": \"/analyze\"" "$LOG_ROOT/smoke/app.jsonl" | grep "\"msg\": \"request\"" | tail -1)"
  ' UNIT="$unit" SMOKE_CPU="$SMOKE_CPU" SMOKE_PORT="$SMOKE_PORT" PHOTO="$(basename "$(fixtures smoke)").jxl") || rc=$?
  printf '%s\n' "$out" > "$SCRATCH/$id.smoke.txt"
  if (( rc != 0 )); then
    grep -v '^SMOKE_LOG \|^SMOKE_PART ' <<<"$out" | sed 's/^/   /'
    die "the smoke test failed"
  fi
  local rss; rss=$(sed -n 's/^SMOKE_RSS //p' <<<"$out"); [[ "$rss" =~ ^[0-9]+$ ]] || rss=0
  note "ready in $(sed -n 's/^SMOKE_READY_S //p' <<<"$out") s; /analyze $(sed -n 's/^SMOKE_LATENCY_S //p' <<<"$out") s; peak memory $(( ${rss:-0} / 1048576 )) MB"
  note "(timings are under the caps; they gate nothing while a sweep may be running)"
  "$LOCAL_PY" "$DEPLOY_DIR/release_probe.py" compare-smoke "$SCRATCH/$id.expected.json" "$SCRATCH/$id.smoke.txt" \
    || die "the smoke answers differ from the local ones"
}

# ------------------------------------------------------------------------------ production files + flip
render_production() {   # the inference host's unit, if it changed; keeps a site from before OPS-15 for a revert
  say "Render production files"
  [[ -n "$DOMAIN" ]] || die "DOMAIN is not set (the public host name, for the checks after the flip)"
  local unit out
  unit=$(render "$SHA" webapp/deploy/coffee-cv-web.service.template SERVICE_USER="$SERVICE_USER" \
           RELEASE_DIR="$APP_ROOT/current" PORT="$PORT" LOG_DIR="$LOG_ROOT")
  out=$(vm '
    UNIT_F="$SYSTEM_ROOT/etc/systemd/system/coffee-cv-web.service"
    SITE_F="$SYSTEM_ROOT/etc/nginx/conf.d/coffee-cv.conf"
    if [[ "$(cat "$UNIT_F" 2>/dev/null)" != "$UNIT" ]]; then
      printf "%s\n" "$UNIT" | $SUDO tee "$UNIT_F" >/dev/null
      $SUDO systemctl daemon-reload
      echo "UNIT_CHANGED=1"
    fi
    # The site an older deploy rendered here, before OPS-15: kept outside conf.d'"'"'s *.conf glob, for a revert or a
    # rollback to a release that carries no site of its own (site_for_current).
    if [[ -f "$SITE_F" && "$(cat "$SITE_F")" != "$INCLUDE" && ! -f "$SITE_F.pre-ops15" ]]; then
      $SUDO cp -p "$SITE_F" "$SITE_F.pre-ops15"
      echo "KEPT=1"
    fi
  ' UNIT="$unit" INCLUDE="$(site_include "$APP_ROOT" inference)")
  UNIT_CHANGED=$(kv UNIT_CHANGED "$out")
  note "unit: $([[ -n "$UNIT_CHANGED" ]] && echo "installed + daemon-reload" || echo unchanged)$([[ -n "$(kv KEPT "$out")" ]] && echo "; the site from before OPS-15 kept as coffee-cv.conf.pre-ops15")"
}

# The inference host's conf.d file for the release `current` points at, before each nginx -t: the include of its site,
# or, for a release from before OPS-15, which carries none, the site that deploy rendered (kept by render_production).
site_for_current() {
  vm '
    SITE_F="$SYSTEM_ROOT/etc/nginx/conf.d/coffee-cv.conf"
    want=$INCLUDE
    if [[ ! -e "$APP_ROOT/current/inference.nginx.conf" && -f "$SITE_F.pre-ops15" ]]; then want=$(cat "$SITE_F.pre-ops15"); fi
    if [[ "$(cat "$SITE_F" 2>/dev/null)" != "$want" ]]; then
      printf "%s\n" "$want" | $SUDO tee "$SITE_F" >/dev/null
      [[ "$want" == "$INCLUDE" ]] && echo "   the inference host: conf.d includes the release'"'"'s site" \
        || echo "   the inference host: conf.d holds the site from before OPS-15 again"
    fi' INCLUDE="$(site_include "$APP_ROOT" inference)"
}

# flip_on vm|fe <release id> <host's name>: previous <- current, current <- releases/<id>, atomically each. A host that
# moved is added to FLIPPED, which revert_both undoes; one already on the release (a redeploy of it) is not.
flip_on() {
  local out
  out=$("$1" '
    cd "$APP_ROOT"
    old=$(readlink current 2>/dev/null || true)
    [[ "$old" != "releases/$ID" ]] || { echo "   $WHERE: already on releases/$ID"; exit 0; }
    if [[ -n "$old" ]]; then
      ln -sfn "$old" previous.new && mv -T previous.new previous
    fi
    ln -sfn "releases/$ID" current.new && mv -T current.new current
    echo "   $WHERE: current -> $(readlink current); previous -> $(readlink previous 2>/dev/null || echo none)"
    echo "MOVED=1"
  ' ID="$2" WHERE="$3")
  grep -v '^MOVED=1$' <<<"$out" || true
  [[ -z "$(kv MOVED "$out")" ]] || FLIPPED+=" $1"
}

# unflip_on vm|fe <release id> <host's name>: undoes flip_on, if it ran: back to previous, or, on a host's first
# release, to no release at all, so that its conf.d include matches nothing and that host's nginx -t passes again.
unflip_on() {
  "$1" '
    cd "$APP_ROOT"
    [[ "$(readlink current 2>/dev/null || true)" == "releases/$ID" ]] || exit 0
    if [[ -L previous && "$(readlink previous)" != "releases/$ID" ]]; then
      prev=$(readlink previous)
      ln -sfn "$prev" current.new && mv -T current.new current
      ln -sfn "releases/$ID" previous.new && mv -T previous.new previous
    else
      rm -f current
    fi
    echo "   $WHERE: current -> $(readlink current 2>/dev/null || echo none)"
  ' ID="$2" WHERE="$3"
}

# rollback_links vm|fe: current and previous swapped, on one host.
rollback_links() {
  "$1" '
    cd "$APP_ROOT"
    [[ -L previous ]] || { echo "no previous release to roll back to"; exit 1; }
    cur=$(readlink current); prev=$(readlink previous)
    [[ -f "$prev/.complete" ]] || { echo "$prev is incomplete"; exit 1; }
    ln -sfn "$prev" current.new && mv -T current.new current
    ln -sfn "$cur" previous.new && mv -T previous.new previous
    echo "   current -> $(readlink current); previous -> $(readlink previous)"'
}

# Both hosts' nginx -t first, then one reload on each (D8): never one host on the new site while the other refuses it.
reload_sites() {
  local on where ok=1
  site_for_current || return 1
  for on in vm fe; do
    where=$([[ $on == vm ]] && echo "the inference host" || echo "the front")
    if "$on" 'msg=$($SUDO nginx -t 2>&1) || { printf "%s\n" "$msg" | sed "s/^/   nginx: /"; exit 1; }'; then
      note "nginx -t ok on $where"
    else
      note "nginx -t failed on $where"; ok=""
    fi
  done
  [[ -n "$ok" ]] || return 1
  vm '$SUDO systemctl reload nginx' && fe '$SUDO systemctl reload nginx' && note "nginx reloaded on both hosts"
}

restart_if_enabled() {   # prints ENABLED=1 when it restarted the service
  vm '
    if [[ "$(systemctl is-enabled coffee-cv-web 2>/dev/null || true)" == enabled ]]; then
      $SUDO systemctl restart coffee-cv-web && echo "ENABLED=1"
    fi'
}

front_restart_if_enabled() {   # prints FRONT_ENABLED=1 when it restarted the check service
  fe '
    if [[ "$(systemctl is-enabled coffee-cv-front 2>/dev/null || true)" == enabled ]]; then
      $SUDO systemctl restart coffee-cv-front && echo "FRONT_ENABLED=1"
    fi'
}

# The hosts flip_on moved (vm, fe), from the flip until the deploy is verified: any failure meanwhile, the EXIT trap's
# included, puts them back on the release they ran before, reloads both sites and starts both services (D14).
FLIPPED=""
REVERTING=""   # set from the flip until verified, even when no host moved (a redeploy of the live release)
revert_both() {
  [[ -n "$REVERTING" ]] || return 0
  REVERTING=""
  say "Revert both hosts to the release they ran before"
  if [[ " $FLIPPED " == *" vm "* ]]; then
    unflip_on vm "$ID" "the inference host" || note "WARNING: the inference host's links could not be reverted -- inspect by hand"
  fi
  if [[ " $FLIPPED " == *" fe "* ]]; then
    unflip_on fe "$ID" "the front" || note "WARNING: the front's links could not be reverted -- inspect by hand"
  fi
  FLIPPED=""
  reload_sites || note "WARNING: nginx -t fails on the previous release's site too -- inspect both hosts by hand"
  restart_if_enabled >/dev/null || note "WARNING: coffee-cv-web did not restart"
  front_restart_if_enabled >/dev/null || note "WARNING: coffee-cv-front did not restart"
}

# -------------------------------------------------------------------------------------------- verify
verify() {   # returns non-zero on a failed check
  say "Verify what current serves"
  [[ -n "$PUBLIC_URL" || -n "$PUBLIC_URL_GIVEN" ]] || die "DOMAIN is not set -- the public checks need it (DOMAIN=... $0 --verify)"
  local out fails=0
  out=$(vm '
    cur=$(readlink -f "$APP_ROOT/current")
    echo "CURRENT=$(basename "$cur")"
    echo "COMMIT=$(sed -n "s/.*\"commit\": \"\([0-9a-f]*\)\".*/\1/p" "$cur/webapp/BUILD_INFO.json")"
    ckpt=$(sed -n "s/^COFFEE_CV_CHECKPOINT=//p" "$cur/release.env")
    echo "MODEL_SHA=$(sha256sum < "$cur/$ckpt" | cut -c1-16)"
    echo "ENABLED=$(systemctl is-enabled coffee-cv-web 2>/dev/null || true)"
    if [[ "$(systemctl is-enabled coffee-cv-web 2>/dev/null || true)" == enabled ]]; then
      for _ in $(seq 1 240); do
        [[ "$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/" || true)" == 404 ]] && break
        sleep 0.5
      done
      echo "GET_ROOT=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/")"
      echo "GET_ANALYZE=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/analyze")"
      inv=$(systemctl show -p InvocationID --value coffee-cv-web)
      echo "STARTUP=$($SUDO journalctl _SYSTEMD_INVOCATION_ID="$inv" -o cat --no-pager | grep -a "serving " | tail -1)"
    fi
    if [[ -f "$APP_ROOT/fixtures/expected/$(basename "$cur").json" ]]; then
      echo "EXPECTED=$(sed -n "s/^PROBE //p" "$APP_ROOT/fixtures/expected/$(basename "$cur").json")"
    fi
  ' PORT="$PORT") || { note "could not read the VM's state"; return 1; }
  local front
  front=$(fe '
    cur=$(readlink -f "$APP_ROOT/current")
    echo "CURRENT=$(basename "$cur")"
    echo "INDEX_SHA=$(sha256sum < "$cur/webapp/static/index.html" | cut -c1-64)"
    echo "ENABLED=$(systemctl is-enabled coffee-cv-front 2>/dev/null || true)"
    if [[ "$(systemctl is-enabled coffee-cv-front 2>/dev/null || true)" == enabled ]]; then
      for _ in $(seq 1 40); do
        [[ "$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$CHECK_PORT/analyze" || true)" == 405 ]] && break
        sleep 0.25
      done
      echo "GET_ANALYZE=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$CHECK_PORT/analyze")"
    fi' CHECK_PORT="$CHECK_PORT") || { note "could not read the front's state"; return 1; }
  local current enabled expected
  current=$(kv CURRENT "$out"); enabled=$(kv ENABLED "$out"); expected=$(kv EXPECTED "$out")
  note "current = $current; coffee-cv-web is ${enabled:-not installed}; coffee-cv-front is $(kv ENABLED "$front" | grep . || echo "not installed")"
  check() { if [[ "$2" == "$3" ]]; then note "ok    $1: $2"; else note "FAIL  $1: got '$2', want '$3'"; fails=$((fails + 1)); fi; }
  # One release on two hosts (ADR 0031): the page and the backend it calls are the same commit's.
  check "the front's current release" "$(kv CURRENT "$front")" "$current"
  if [[ "$(kv ENABLED "$front")" == enabled ]]; then
    check "check service GET /analyze" "$(kv GET_ANALYZE "$front")" 405
  else
    note "the check service is not enabled: its check skipped"
  fi

  if [[ "$enabled" == enabled ]]; then
    check "backend GET /" "$(kv GET_ROOT "$out")" 404
    check "backend GET /analyze" "$(kv GET_ANALYZE "$out")" 405
    local startup; startup=$(kv STARTUP "$out")
    check "startup line: commit" "$(sed -n 's/^[^{]*//p' <<<"$startup" | jq -r '.code.commit' 2>/dev/null)" "$(kv COMMIT "$out")"
    check "startup line: model_sha" "$(sed -n 's/^[^{]*//p' <<<"$startup" | jq -r '.model_sha' 2>/dev/null)" "$(kv MODEL_SHA "$out")"
  else
    note "the service is not enabled: backend checks skipped (the owner enables it: plan Phase B)"
  fi

  # Through the domain: the front's TLS, its page, its check service, and the inference host behind them (ADR 0031).
  if [[ -z "$PUBLIC_URL" ]]; then
    note "SKIPPED public checks (PUBLIC_URL is empty)"
  else
    local body code
    body=$(mktemp)
    code=$(curl -s -o "$body" -w '%{http_code}' "$PUBLIC_URL/")
    check "public GET /" "$code" 200
    check "public page = the release's index.html" "$(sha256sum < "$body" | cut -c1-64)" "$(kv INDEX_SHA "$front")"
    local photo want times hdr
    photo="$REPO/$(fixtures smoke)"
    hdr=$(mktemp)
    # /analyze streams its preview, then its classification (ticket OPS-7 D2): the answer goes to $body, and the
    # seconds at which each of its lines arrived to $times (the upload included; through a proxy that buffers,
    # the two are equal). Reported, not checked (D11). curl's own timers cannot say it: its start of transfer is
    # the upload's.
    post_analyze() {
      times=$(curl -sN -D "$hdr" -F "photo=@$photo" "$PUBLIC_URL/analyze" | tee "$body" \
              | "$LOCAL_PY" -c 'import sys, time; t0 = time.monotonic(); [print(f"{time.monotonic() - t0:.2f}", flush=True) for _ in sys.stdin.buffer]' \
              | paste -sd' ')
      code=$(awk '/^HTTP/ {c = $2} END {print c}' "$hdr")   # the last status line: a 100 Continue comes first
    }
    post_analyze
    if [[ "$code" == 503 ]]; then sleep 3; post_analyze; fi
    rm -f "$hdr"
    if [[ "$enabled" == enabled ]]; then
      check "public POST /analyze" "$code" 200
      check "public /analyze parts" "$(jq -rs 'map(.part) | join(", ")' < "$body" 2>/dev/null)" "preview, classification"
      want=$(jq -r '.analyze.parts[1].ranked[0].id // "-"' <<<"$expected" 2>/dev/null)
      check "public /analyze top-1" "$(jq -rs '.[1].ranked[0].id // "-"' < "$body" 2>/dev/null)" "${want:-?}"
      note "      public /analyze: its lines arrived at ${times:-?} s from the upload's start, the preview's first (reported, not checked: D11)"
      # The page's example photos (ticket OPS-16 D14, D17): each fetched through the domain, held to the release
      # commit's file, posted, and its answer held to its country or a refusal; one after another, about 1.5 min. A
      # wrong one fails verify, which rolls a full deploy back. Not fixtures (D16): nothing here reaches the VM. The
      # commit is the inference host's (its BUILD_INFO.json) while the front serves the files: the same release, as
      # "the front's current release" above checks.
      local examples examples_rc=0 examples_failed line
      examples=$("$LOCAL_PY" "$DEPLOY_DIR/example_check.py" --url "$PUBLIC_URL" --commit "$(kv COMMIT "$out")" \
                 --repo "$REPO") || examples_rc=$?
      while IFS= read -r line; do [[ -n "$line" ]] && note "$line"; done <<<"$examples"
      examples_failed=$(grep -c '^FAIL' <<<"$examples" || true)
      if (( examples_rc != 0 )); then fails=$((fails + (examples_failed > 0 ? examples_failed : 1))); fi
    else
      check "public POST /analyze (no backend yet: the front says offline, D12)" "$code" 503
    fi
    rm -f "$body"
  fi
  return $(( fails > 0 ))
}

on_verify_failure() {
  revert_both
  verify || true
  die "verification failed; both hosts are back on the release they ran before"
}

# ------------------------------------------------------------------------------------ prune / compare
prune() {
  say "Prune: keep current and previous, on both hosts"
  local out kept f id
  out=$(vm '
    cd "$APP_ROOT"
    shopt -s nullglob
    keep="$(readlink -f current) $(readlink -f previous 2>/dev/null || true)"
    for d in releases/*/; do
      d=$(readlink -f "$d")
      [[ -d "$d" && "$d" == "$APP_ROOT"/releases/* ]] || continue
      case " $keep " in *" $d "*) continue ;; esac
      rm -rf "$d" && echo "   removed $(basename "$d")"
    done
    # Expected answers belong to a release; drop any whose release is gone (however it went).
    for f in fixtures/expected/*.json; do
      [[ -d "releases/$(basename "$f" .json)" ]] || { rm -f "$f"; echo "   removed $f"; }
    done
    capped "$CAP_CPU" "$APP_ROOT" -E UV_CACHE_DIR="$APP_ROOT/uv-cache" "$UV_BIN" cache prune --quiet
    echo "   $(du -sh "$APP_ROOT" | cut -f1) in $APP_ROOT"
    echo "KEPT $(basename "$(readlink -f current)") $(basename "$(readlink -f previous 2>/dev/null || true)")"')
  grep -v '^KEPT ' <<<"$out" || true
  kept=$(sed -n 's/^KEPT //p' <<<"$out")
  fe '
    cd "$APP_ROOT"
    shopt -s nullglob
    keep="$(readlink -f current) $(readlink -f previous 2>/dev/null || true)"
    for d in releases/*/; do
      d=$(readlink -f "$d")
      [[ -d "$d" && "$d" == "$APP_ROOT"/releases/* ]] || continue
      case " $keep " in *" $d "*) continue ;; esac
      rm -rf "$d" && echo "   the front: removed $(basename "$d")"
    done
    echo "   the front: $(du -sh "$APP_ROOT" | cut -f1) in $APP_ROOT"' || note "WARNING: pruning the front failed"
  # This machine's staged trees and their records follow the host: keep those of the releases it keeps.
  [[ "$kept" == *-* ]] || { note "WARNING: the host named no current release; $SCRATCH left as it is"; return 0; }
  for f in "$SCRATCH"/2*; do
    [[ -e "$f" ]] || continue
    id=$(basename "$f"); id=${id%%.*}      # <id>, <id>.expected.json, <id>.smoke.txt, ...
    case " $kept " in *" $id "*) continue ;; esac
    chmod -R u+w "$f" 2>/dev/null || true; rm -rf "$f" && note "removed $f"
  done
}

compare() {   # compare <release id>: the release on the host, on its accelerator, against PyTorch here (ADR 0020)
  local id=$1 sha model
  sha=$(cut -d- -f2 <<<"$id"); model=$(cut -d- -f3- <<<"$id")
  # PyTorch's reference runs this checkout's coffeecv, so it must be the release's (ML-7 P11).
  git -C "$REPO" diff --quiet "$sha" -- coffeecv \
    || die "coffeecv/ here is not the release's ($sha): check out $sha for PyTorch's reference"
  preflight
  check_fixtures
  canonical_fixtures
  send_fixtures
  say "Compare $id: the release on $HOST against PyTorch here ($TORCH_PY), $(fixtures compare | wc -l) photos, ML-7 D7's bars"
  mkdir -p "$SCRATCH"
  local t0=$SECONDS photos=()
  while read -r p; do photos+=("$CANON_FIX/$(basename "$p").jxl"); done < <(fixtures compare)   # D34: the same bytes
  (cd "$REPO" && "$TORCH_PY" -B "$DEPLOY_DIR/release_probe.py" classify-torch --checkpoint "models/$model.pt" \
      "${photos[@]}" 2> >(tail -5 >&2)) > "$SCRATCH/$id.compare-torch.json" || die "PyTorch's reference run failed"
  note "PyTorch here: $((SECONDS - t0)) s"
  make_room
  t0=$SECONDS
  # The release as the service runs it: its venv, its release.env (the accelerator), capped.
  vm '
    R="$APP_ROOT/releases/$ID"
    [[ -f "$R/.complete" ]] || { echo "$R is not a complete release"; exit 1; }
    envs=()
    while IFS= read -r line; do [[ -n "$line" ]] && envs+=(-E "$line"); done < "$R/release.env"
    capped "$CAP_CPU" "$R" "${envs[@]}" -E COFFEE_CV_LOG_DIR=/tmp/probe-logs -E PYTHONDONTWRITEBYTECODE=1 \
      "$R/.venv/bin/python" -B "$APP_ROOT/fixtures/release_probe.py" classify --list "$APP_ROOT/fixtures/fixtures.canonical.txt" \
        --role compare --dir "$APP_ROOT/fixtures/photos" 2> >(tail -5 >&2)' ID="$id" \
    > "$SCRATCH/$id.compare-release.json" || die "the release's run failed"
  note "the release on $HOST: $((SECONDS - t0)) s"
  restore_live
  local rc=0
  "$LOCAL_PY" "$DEPLOY_DIR/release_probe.py" compare-classify --d7 "$SCRATCH/$id.compare-torch.json" \
    "$SCRATCH/$id.compare-release.json" || rc=$?
  postflight
  (( rc == 0 )) || die "the release's answers are outside ML-7 D7's bars against PyTorch (ADR 0020)"
  say "Compare OK: within D7's bars against PyTorch"
}

# ----------------------------------------------------------------------------------------- bootstrap
bootstrap() {
  say "Bootstrap $HOST (one-time; safe during a sweep)"
  local before; before=$(vm 'sweep_state'); PRE_SWEEP=$before
  note "sweep before: $(tr '\n' ' ' <<<"$before")"
  local logrotate
  logrotate=$(git -C "$REPO" show HEAD:webapp/deploy/coffee-cv.logrotate.template \
                | "$LOCAL_PY" "$DEPLOY_DIR/render_template.py" SERVICE_USER="$SERVICE_USER")
  vm '
    if ! id "$SERVICE_USER" >/dev/null 2>&1; then
      $SUDO useradd --system --user-group --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin "$SERVICE_USER"
      echo "   created system user $SERVICE_USER"
    else
      echo "   system user $SERVICE_USER exists"
    fi
    $SUDO install -d -o "$APP_USER" -g "$APP_USER" -m 0755 "$APP_ROOT" "$APP_ROOT/releases" "$APP_ROOT/uv-cache" \
      "$APP_ROOT/fixtures" "$APP_ROOT/fixtures/photos" "$APP_ROOT/fixtures/expected"
    echo "   $APP_ROOT owned by $APP_USER"

    if [[ "$("$UV_BIN" --version 2>/dev/null | awk "{print \$2}")" != "$UV_VERSION" ]]; then
      dl=$(mktemp -d "$APP_ROOT/uv-download.XXXXXX")
      trap "rm -rf \"$dl\"" EXIT
      curl -fsSL -o "$dl/uv.tar.gz" "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-x86_64-unknown-linux-gnu.tar.gz"
      echo "$UV_SHA256  $dl/uv.tar.gz" | sha256sum -c --quiet - || { echo "uv tarball sha256 mismatch"; exit 1; }
      tar -xzf "$dl/uv.tar.gz" -C "$dl"
      $SUDO install -m 0755 "$dl/uv-x86_64-unknown-linux-gnu/uv" "$UV_BIN"
    fi
    echo "   $("$UV_BIN" --version) at $UV_BIN"
    $SUDO install -d -o "$APP_USER" -g "$APP_USER" -m 0755 "$APP_ROOT/python"
    UV_CACHE_DIR="$APP_ROOT/uv-cache" managed_python install --no-bin "$PY_VERSION" 2>&1 | sed "s/^/   /"
    echo "   Python $PY_VERSION at $(managed_python find "$PY_VERSION")"

    $SUDO install -d -o "$SERVICE_USER" -g "$SERVICE_USER" -m 0755 "$LOG_ROOT"
    $SUDO find "$LOG_ROOT" -maxdepth 1 -type f -exec chown "$SERVICE_USER:$SERVICE_USER" {} +
    echo "   $LOG_ROOT and its files owned by $SERVICE_USER"
    printf "%s\n" "$LOGROTATE" | $SUDO tee "$SYSTEM_ROOT/etc/logrotate.d/coffee-cv" >/dev/null
    echo "   rendered $SYSTEM_ROOT/etc/logrotate.d/coffee-cv (su $SERVICE_USER $SERVICE_USER)"
  ' UV_VERSION="$UV_VERSION" UV_SHA256="$UV_TARBALL_SHA256" LOGROTATE="$logrotate"
  postflight
}

# -------------------------------------------------------------------------------------------- main
mkdir -p "$SCRATCH"
on_exit() { revert_both; restore_live; }
trap on_exit EXIT
trap 'exit 130' INT TERM
MODE=deploy
case "${1:-}" in
  --bootstrap) bootstrap; exit 0 ;;
  --verify)    verify || die "verification failed"; exit 0 ;;
  --rollback)
    say "Rollback both hosts"
    rollback_links vm || die "rollback refused on the inference host"
    rollback_links fe || { rollback_links vm; die "rollback refused on the front; the inference host is switched back"; }
    if ! reload_sites; then
      rollback_links vm; rollback_links fe; site_for_current || true
      die "nginx -t failed on the previous release's site: both hosts' links are switched back, nothing reloaded"
    fi
    restart_if_enabled >/dev/null
    front_restart_if_enabled >/dev/null
    verify || die "verification after rollback failed -- inspect by hand"
    exit 0 ;;
  --compare)   [[ $# -eq 2 ]] || usage; MODE=compare; compare "$2"; exit 0 ;;
  --stage-only) MODE=stage; shift ;;
  -h|--help|"") usage ;;
esac
[[ $# -eq 2 ]] || usage
REF=$1 MODEL=$2

guard "$REF"
preflight
front_preflight
build "$REF" "$MODEL"
stage
stage_front
make_room
smoke "$ID"
if [[ "$MODE" == stage ]]; then
  restore_live
  postflight
  say "Staged $ID on both hosts (not flipped). Next: scripts/deploy_webapp.sh --compare $ID"
  exit 0
fi
stop_front
render_production
say "Flip both hosts to releases/$ID"
REVERTING=1
flip_on vm "$ID" "the inference host"
flip_on fe "$ID" "the front"
ENABLED=$(restart_if_enabled)
FRONT_ENABLED=$(front_restart_if_enabled)
if [[ -z "$FRONT_ENABLED" && -n "$FRONT_STOPPED" ]]; then
  # Running but not enabled: started again now, on the new release, for verify's checks through the domain.
  fe '$SUDO systemctl start coffee-cv-front'
fi
FRONT_STOPPED=""
# The restart started the new release; a disabled service is started by the EXIT trap, on the new release too.
if [[ -n "$ENABLED" && -n "$LIVE_STOPPED_AT" ]]; then
  note "photo answers were down $((SECONDS - LIVE_STOPPED_AT)) s"; LIVE_STOPPED_AT=""
fi
reload_sites || die "nginx -t rejected a release's site; both hosts go back to the release they ran before"
if ! verify; then on_verify_failure; fi
REVERTING=""; FLIPPED=""
prune
postflight
if [[ -z "$ENABLED" ]]; then
  say "Deployed $ID -- coffee-cv-web is NOT enabled, so nothing serves /analyze yet. To serve it:"
  note "sudo systemctl enable --now coffee-cv-web     (on $HOST)"
  note "DOMAIN=$DOMAIN scripts/deploy_webapp.sh --verify   (here)"
else
  say "Deployed and verified $ID"
fi

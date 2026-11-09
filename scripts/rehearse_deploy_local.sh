#!/usr/bin/env bash
# Rehearse scripts/deploy_webapp.sh end to end on this machine, with no VM and no systemd
# (docs/ops1_release_isolation_plan.html §7): bootstrap, build, stage, smoke, flip, prune, --rollback,
# and the refusals. The deploy script runs unmodified with HOST=local and FRONT_HOST=local against two scratch
# roots, one per host (ticket OPS-15: the inference host and the front), with a failure injected on each.
#
#   scripts/rehearse_deploy_local.sh [scratch-dir]
#
# Both deploys use $REHEARSE_MODEL, by default the served model (the one with a LiteRT export): deploy 1 at
# $REHEARSE_PREV_REF (default origin/main~1), deploy 2 at origin/main, so the rollback switches between two releases
# of it. Until 2026-10-06 deploy 1 was the ResNet18 allrigs_cam_s123; it was retired with every other old model.
# REHEARSE_PREV_REF must be on origin/main and already hold $REHEARSE_MODEL.
# Since ticket OPS-7 the deploy speaks /analyze only (D10), so REHEARSE_PREV_REF must also be from OPS-7 on.
# Since OPS-15 P4 the deploy builds a front release too, from the commit's front templates and its check service's
# unit, so REHEARSE_PREV_REF must also be from OPS-15 P4 on.
#
# Last passed 2026-11-02 (OPS-15 P4, two hosts), with REHEARSE_MODEL=dino3b16_segxl0v6_country_aug8_s42 and
# REHEARSE_PREV_REF=origin/main~1, from a scratch clone whose main was branch ops15-impl plus P4 (deploy 1) and an empty
# commit (deploy 2), since the branch is not on main until the cutover: an nginx -t failure on each host after the flip
# put both hosts back on deploy 1 with both services running, the live service stopped only once both were staged;
# deploy 2 verified both hosts on one release and pruned both; --compare within D7 of PyTorch; a failed redeploy of the
# live release left both hosts on it; a rollback and back moved both hosts; a rollback to a release with no site of its
# own put the site from before OPS-15 back, and the roll-forward the release's include. Run with
# UV_PYTHON_DOWNLOADS and UV_PROJECT_ENVIRONMENT unset: the devcontainer sets them for training.
#
# Stubbed: sudo (runs the command as you), systemd-run (runs the command in its working directory with
# its -E variables -- no sandbox, no caps), systemctl (starts the ExecStart of the rendered unit in the
# background, so the smoke and verify steps hit a real gunicorn from a real release venv), journalctl,
# nginx -t, useradd. NOT exercised here, only on the VM: the sandbox and caps themselves, running as
# coffee-cv, nginx, and the public-site checks.
#
# Runs from a clone whose `origin` is this repository, so the deploy's "commit must be on origin/main"
# check is real: commit to main here first. Needs the DVC-tracked model and its LiteRT export locally (ML-7 P10:
# both deploys must be of commits that have the export), and for --compare the checkpoint and pretrained weights
# checked out in the source (ML-7 P11).
set -euo pipefail

SRC=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
R=${1:-$(mktemp -d)}
R=$(mkdir -p "$R" && cd "$R" && pwd)
echo "rehearsal root: $R"
rm -rf "$R/clone" "$R/opt" "$R/sys" "$R/opt-front" "$R/sys-front" "$R/log" "$R/state" "$R/stubs" "$R/scratch"
mkdir -p "$R"/{opt,opt-front,log,state,stubs,scratch} "$R"/sys/{etc/systemd/system,run/systemd/system,etc/nginx/conf.d,etc/logrotate.d} \
  "$R"/sys-front/{etc/systemd/system,etc/nginx/conf.d}

# ------------------------------------------------------------------------------------------ the clone
git clone -q "$SRC" "$R/clone"
C="$R/clone"
# The checkpoint and its LiteRT export (ticket ML-7) come from the DVC cache by md5; a release needs no pretrained
# weights since P10.
ln -s "$SRC/.dvc/cache" "$C/.dvc/cache"
while read -r photo; do
  mkdir -p "$C/$(dirname "$photo")"; ln -s "$SRC/$photo" "$C/$photo"
done < <(awk '!/^#/ && NF {print $3}' "$SRC/webapp/deploy/fixtures.txt")

# ------------------------------------------------------------------------------------------ the stubs
S="$R/stubs"
cat > "$S/sudo" <<'EOF'
#!/usr/bin/env bash
[[ "${1:-}" == -n ]] && shift
exec "$@"
EOF
cat > "$S/useradd" <<'EOF'
#!/usr/bin/env bash
echo "(stub) useradd $*"
EOF
cat > "$S/nginx" <<'EOF'
#!/usr/bin/env bash
# STUB_NGINX_FAIL names the host whose nginx -t fails, by its SYSTEM_ROOT (the deploy passes each host's).
[[ -n "${STUB_NGINX_FAIL:-}" && "${STUB_NGINX_FAIL}" == "${SYSTEM_ROOT:-}" ]] \
  && { echo "nginx: [emerg] (stub) configured to fail" >&2; exit 1; }
echo "nginx: the configuration file syntax is ok (stub)" >&2
EOF
cat > "$S/systemd-run" <<'EOF'
#!/usr/bin/env bash
envs=() wd=.
while (( $# )); do
  case $1 in
    --wait|--pipe|--collect|--quiet) shift ;;
    --uid=*|--gid=*|--nice=*) shift ;;
    --working-directory=*) wd=${1#*=}; shift ;;
    -p) shift 2 ;;
    -E) envs+=("$2"); shift 2 ;;
    -*) echo "stub systemd-run: unknown option $1" >&2; exit 2 ;;
    *) break ;;
  esac
done
# COFFEE_CV_LITERT_ON_CPU, as for the service below: --compare runs the release here, on the CPU (ML-7 P11).
cd "$wd" && exec env "${envs[@]}" COFFEE_CV_LITERT_ON_CPU=1 "$@"
EOF
cat > "$S/systemctl" <<'EOF'
#!/usr/bin/env bash
# Enough of systemctl for deploy_webapp.sh: units are the rendered files under $SYSTEM_ROOT.
set -euo pipefail
ST=${STUB_STATE:?}
norm() { local u=$1; [[ $u == *.* ]] || u=$u.service; echo "$u"; }
unitfile() {
  local d
  for d in "$SYSTEM_ROOT/run/systemd/system" "$SYSTEM_ROOT/etc/systemd/system"; do
    [[ -f "$d/$1" ]] && { echo "$d/$1"; return; }
  done
  return 1
}
alive() { [[ -f "$ST/$1.pid" ]] && kill -0 "$(cat "$ST/$1.pid")" 2>/dev/null; }
stop() {
  if alive "$1"; then
    local pid; pid=$(cat "$ST/$1.pid")
    kill -TERM -- "-$pid" 2>/dev/null || true
    for _ in $(seq 1 50); do kill -0 "$pid" 2>/dev/null || break; sleep 0.2; done
  fi
  rm -f "$ST/$1.pid"
}
start() {
  local u=$1 f wd envf envs exe logs
  f=$(unitfile "$u") || { echo "stub systemctl: no unit $u" >&2; exit 5; }
  wd=$(sed -n 's/^WorkingDirectory=//p' "$f"); envf=$(sed -n 's/^EnvironmentFile=//p' "$f")
  envs=$(sed -n 's/^Environment=//p' "$f"); exe=$(sed -n 's/^ExecStart=//p' "$f")
  # LogsDirectory= (the check service's, OPS-3 P2): made under $LOG_ROOT's parent and named in $LOGS_DIRECTORY, as
  # systemd makes /var/log/<name> and names it.
  logs=$(sed -n 's/^LogsDirectory=//p' "$f")
  if [[ -n $logs ]]; then mkdir -p "$(dirname "$LOG_ROOT")/$logs"; envs="$envs LOGS_DIRECTORY=$(dirname "$LOG_ROOT")/$logs"; fi
  od -An -N8 -tx8 /dev/urandom | tr -d ' ' > "$ST/$u.inv"
  echo "=== invocation $(cat "$ST/$u.inv")" >> "$ST/$u.log"
  # COFFEE_CV_LITERT_ON_CPU: a GPU release's graphs on this machine's CPU (ML-7 P10); the GPU runs only on the host.
  ( cd "$wd" && set -a && . "$envf" && set +a && exec env $envs COFFEE_CV_LITERT_ON_CPU=1 setsid $exe ) \
    >> "$ST/$u.log" 2>&1 &
  echo $! > "$ST/$u.pid"
}
cmd=$1; shift
case $cmd in
  daemon-reload|reload) exit 0 ;;
  start) start "$(norm "$1")" ;;
  stop) stop "$(norm "$1")" ;;
  restart) u=$(norm "$1"); stop "$u"; start "$u" ;;
  enable)
    now=; [[ $1 == --now ]] && { now=1; shift; }
    u=$(norm "$1"); touch "$ST/$u.enabled"; [[ -n $now ]] && start "$u"; exit 0 ;;
  is-enabled) u=$(norm "$1"); [[ -f "$ST/$u.enabled" ]] && { echo enabled; exit 0; }; echo disabled; exit 1 ;;
  is-active)   # as systemd: prints active or inactive unless --quiet; exit 0 only when active
    q=; [[ $1 == --quiet ]] && { q=1; shift; }
    if alive "$(norm "$1")"; then [[ -n $q ]] || echo active; else [[ -n $q ]] || echo inactive; exit 3; fi ;;
  show)
    prop=; while (( $# > 1 )); do case $1 in -p) prop=$2; shift 2 ;; --value) shift ;; *) shift ;; esac; done
    u=$(norm "$1")
    case $prop in
      InvocationID) cat "$ST/$u.inv" ;;
      MemoryPeak) pid=$(cat "$ST/$u.pid"); ps -o rss= -g "$pid" | awk '{s += $1} END {print s * 1024}' ;;
      *) echo "" ;;
    esac ;;
  *) echo "stub systemctl: $cmd not stubbed" >&2; exit 2 ;;
esac
EOF
cat > "$S/journalctl" <<'EOF'
#!/usr/bin/env bash
ST=${STUB_STATE:?} unit= inv= n=
while (( $# )); do
  case $1 in
    -u) unit=$2; shift 2 ;;
    -n) n=$2; shift 2 ;;
    _SYSTEMD_INVOCATION_ID=*) inv=${1#*=}; shift ;;
    *) shift ;;
  esac
done
if [[ -n $inv ]]; then
  for f in "$ST"/*.inv; do [[ "$(cat "$f")" == "$inv" ]] && unit=$(basename "$f" .inv); done
  awk -v i="=== invocation $inv" '$0 == i {on = 1; next} /^=== invocation/ {on = 0} on' "$ST/$unit.log"
else
  [[ $unit == *.* ]] || unit=$unit.service
  tail -n "${n:-100}" "$ST/$unit.log" 2>/dev/null
fi
EOF
chmod +x "$S"/*

# ------------------------------------------------------------------------------------------ the runs
export HOST=local APP_ROOT="$R/opt/coffee-cv" APP_USER="$(id -un)" SERVICE_USER="$(id -un)"
export SYSTEM_ROOT="$R/sys" LOG_ROOT="$R/log/coffee-cv" DOMAIN=rehearsal.invalid PUBLIC_URL=
export UV_BIN="${UV_BIN:-$(command -v uv)}"   # the releases' Python: uv-managed, installed by --bootstrap
export TORCH_PY="${TORCH_PY:-$(command -v python)}" LOCAL_PY="${LOCAL_PY:-python3}"
export STUB_PATH="$S" STUB_STATE="$R/state" DEPLOY_SCRATCH="$R/scratch"
export MIN_MEM_GB=${MIN_MEM_GB:-2} MIN_FREE_GB=${MIN_FREE_GB:-5} PORT=18000 SMOKE_PORT=18001
# The inference host's nginx site listens here (ticket OPS-15 P2); the rehearsal has no tailnet.
export INFERENCE_ADDR=127.0.0.1 INFERENCE_PORT=18002
# The front (ticket OPS-15 P4): its own root and /etc prefix on this machine. No nginx runs, so the certificate paths
# are never opened.
FA="$R/opt-front/coffee-cv"
export FRONT_HOST=local FRONT_APP_ROOT="$FA" FRONT_SYSTEM_ROOT="$R/sys-front" FRONT_USER="$(id -un)"
export FRONT_SERVICE_USER="$(id -un)" CHECK_PORT=18003 FRONT_PORT=18004 TLS_CERT="$R/tls/cert.pem" TLS_KEY="$R/tls/key.pem"
# The rehearsal tests the deploy machinery on a scratch clone without the data, not the code: skip the
# deploy's full test suite (scripts/check.sh --full).
export DEPLOY_SKIP_CHECK=1
D="$C/scripts/deploy_webapp.sh"
MODEL=${REHEARSE_MODEL:-dino3b16_segxl0v3_country_aug8_s123}
SHA=$(git -C "$C" rev-parse origin/main)
PREV_SHA=$(git -C "$C" rev-parse "${REHEARSE_PREV_REF:-origin/main~1}")
[[ "$PREV_SHA" != "$SHA" ]] || { echo "REHEARSE_PREV_REF resolves to origin/main itself"; exit 1; }
cleanup() {
  PATH="$S:$PATH" systemctl stop coffee-cv-web >/dev/null 2>&1 || true
  PATH="$S:$PATH" SYSTEM_ROOT="$R/sys-front" systemctl stop coffee-cv-front >/dev/null 2>&1 || true
}
both_on() {   # both_on <release id glob> <what>: both hosts' current is that release
  [[ "$(readlink "$APP_ROOT/current")" == releases/$1 && "$(readlink "$FA/current")" == releases/$1 ]] \
    || { echo "$2: the inference host is on $(readlink "$APP_ROOT/current"), the front on $(readlink "$FA/current" || echo none)"; exit 1; }
}
both_active() {   # both services running, under the stub
  PATH="$S:$PATH" systemctl is-active --quiet coffee-cv-web || { echo "$1: coffee-cv-web is not running"; exit 1; }
  PATH="$S:$PATH" SYSTEM_ROOT="$R/sys-front" systemctl is-active --quiet coffee-cv-front \
    || { echo "$1: coffee-cv-front is not running"; exit 1; }
}
trap cleanup EXIT
step() { printf '\n\n############ %s\n' "$*"; }
expect_fail() {   # expect_fail <pattern> <cmd...>: the command must fail and print the pattern
  local out rc=0
  out=$("${@:2}" 2>&1) || rc=$?
  printf '%s\n' "$out" > "$R/expect_fail.log"
  if (( rc != 0 )) && grep -q -- "$1" <<<"$out"; then echo "   refused as expected: $(grep -m1 -- "$1" <<<"$out")"
  else echo "$out" | tail -20; echo "REHEARSAL FAILED: expected a refusal matching '$1' (exit $rc)"; exit 1; fi
}

step "bootstrap"
"$D" --bootstrap

step "the front, as webapp/deploy/setup_front.sh prepares it (plan P5)"
# Its release directory, uv and the managed Python (this machine's, which --bootstrap just installed), the unit, and
# conf.d's include of the release's site.
mkdir -p "$FA/releases" "$FA/uv-cache"
ln -sfn "$APP_ROOT/python" "$FA/python"
git -C "$C" show origin/main:webapp/deploy/coffee-cv-front.service.template \
  | python3 "$C/webapp/deploy/render_template.py" FRONT_SERVICE_USER="$FRONT_SERVICE_USER" RELEASE_DIR="$FA/current" \
  > "$R/sys-front/etc/systemd/system/coffee-cv-front.service"
printf 'include %s/current/front.nginx*.conf;\n' "$FA" > "$R/sys-front/etc/nginx/conf.d/coffee-cv-front.conf"
echo "   $FA, its unit and its conf.d include"

step "refusal: a commit that is not on origin/main"
git -C "$C" -c user.name=rehearsal -c user.email=rehearsal@invalid commit -q --allow-empty -m "local only"
expect_fail "is not on origin/main" "$D" --stage-only HEAD "$MODEL"

step "refusal: a .pt whose md5 does not match its .pt.dvc"
# A stand-in cache without the .pt's object, but with the LiteRT export's listing and manifest, which the release's
# manifest reads (ML-7 P10); the real cache is never touched.
mv "$C/.dvc/cache" "$C/.dvc/cache.off"; printf 'not a model' > "$C/models/$MODEL.pt"
lit=$(git -C "$C" show "$SHA:dvc.lock" | awk -v p="models/$MODEL.litert" '$0 ~ "- path: " p "$" {f = 1} f && /md5:/ {print $2; exit}')
for md5 in "$lit" $(jq -r '.[] | select(.relpath == "manifest.json") | .md5' "$C/.dvc/cache.off/files/md5/${lit:0:2}/${lit:2}"); do
  install -D -m 0644 "$C/.dvc/cache.off/files/md5/${md5:0:2}/${md5:2}" "$C/.dvc/cache/files/md5/${md5:0:2}/${md5:2}"
done
expect_fail "no copy in the working tree or the DVC cache has md5" "$D" --stage-only "$SHA" "$MODEL"
rm -r "$C/models/$MODEL.pt" "$C/.dvc/cache"; mv "$C/.dvc/cache.off" "$C/.dvc/cache"

step "deploy 1: $MODEL at ${PREV_SHA:0:7} (service disabled: flip, no restart)"
# The site a deploy from before OPS-15 rendered into conf.d: kept for a rollback to a release that carries none.
OLD_SITE="# the inference host's site from before OPS-15 (rehearsal)"
printf '%s\n' "$OLD_SITE" > "$SYSTEM_ROOT/etc/nginx/conf.d/coffee-cv.conf"
"$D" "$PREV_SHA" "$MODEL"
[[ "$(readlink "$APP_ROOT/current")" == releases/*-"${PREV_SHA:0:7}-$MODEL" ]] || { echo "current not flipped"; exit 1; }

both_on "*-${PREV_SHA:0:7}-$MODEL" "deploy 1"
[[ "$(cat "$SYSTEM_ROOT/etc/nginx/conf.d/coffee-cv.conf.pre-ops15")" == "$OLD_SITE" ]] \
  || { echo "deploy 1 did not keep the site from before OPS-15"; exit 1; }

step "enable both services (the owner's Phase B), then --verify"
PATH="$S:$PATH" systemctl enable --now coffee-cv-web
PATH="$S:$PATH" SYSTEM_ROOT="$R/sys-front" systemctl enable --now coffee-cv-front
"$D" --verify

# A failure on either host after the flip puts both back (ticket OPS-15 D14): nginx -t failing on each in turn.
for host_root in "$R/sys" "$R/sys-front"; do
  step "refusal: nginx -t fails on $(basename "$host_root") after the flip -> both hosts back on deploy 1"
  expect_fail "nginx -t rejected a release's site" env STUB_NGINX_FAIL="$host_root" "$D" "$SHA" "$MODEL"
  both_on "*-${PREV_SHA:0:7}-$MODEL" "after the failure on $(basename "$host_root")"
  grep -q "Revert both hosts" "$R/expect_fail.log" || { echo "the deploy did not revert both hosts"; exit 1; }
  # ADR 0026: the live service stopped only once both hosts were staged, and started again.
  awk '/^== Stage .* on the front/ {s = 1} /^== Stop the live service/ {if (s) ok = 1} END {exit !ok}' "$R/expect_fail.log" \
    || { echo "the live service was stopped before both hosts were staged"; exit 1; }
  both_active "after the failure on $(basename "$host_root")"
  [[ "$(cat "$SYSTEM_ROOT/etc/nginx/conf.d/coffee-cv.conf")" == "include $APP_ROOT/current/inference.nginx*.conf;" ]] \
    || { echo "the inference host's conf.d does not include the release's site"; exit 1; }
  echo "   both hosts on deploy 1, both services running"
done

step "deploy 2: $MODEL at ${SHA:0:7} (service enabled: stopped for the smoke, restarted on the new release, verify, prune)"
old="$DEPLOY_SCRATCH/20200101T0000Z-0000000-$MODEL"            # a staged tree of a release the host no longer has
mkdir -p "$old" && touch "$old.expected.json" "$old.smoke.txt"
"$D" "$SHA" "$MODEL" | tee "$R/deploy2.log"
grep -q "Stop the live service" "$R/deploy2.log" || { echo "deploy 2 did not stop the live service"; exit 1; }
both_on "*-${SHA:0:7}-$MODEL" "deploy 2"
both_active "after deploy 2"
ls "$APP_ROOT/releases" "$FA/releases"
# The prune keeps this machine's staged trees of the releases the host keeps, and only those.
[[ ! -e "$old" && ! -e "$old.expected.json" && ! -e "$old.smoke.txt" ]] || { echo "the old staged tree was not pruned"; exit 1; }
for rel in "$APP_ROOT"/releases/*; do
  [[ -d "$DEPLOY_SCRATCH/$(basename "$rel")" ]] || { echo "the staged tree of $(basename "$rel") was pruned"; exit 1; }
done
echo "   scratch pruned to the host's releases: $(ls "$DEPLOY_SCRATCH" | grep -c '^2.*Z-[0-9a-f]*-[^.]*$') staged trees"

step "--compare on the current release"
# PyTorch's reference (ML-7 P11) runs the clone's coffeecv on the checkpoint and its pretrained weights, untracked
# files the clone lacks: linked from the source checkout here, after the refusals, which write a fake .pt.
while read -r f; do
  mkdir -p "$C/$(dirname "$f")"; ln -sfn "$SRC/$f" "$C/$f"
done < <(git -C "$SRC" ls-files -o -i --exclude-standard -- models models_pretrained)
"$D" --compare "$(basename "$(readlink "$APP_ROOT/current")")"
PATH="$S:$PATH" systemctl is-active --quiet coffee-cv-web || { echo "--compare did not start the live service again"; exit 1; }

step "refusal: a redeploy of the live release fails after the flip -> both hosts stay on it"
expect_fail "nginx -t rejected a release's site" env STUB_NGINX_FAIL="$R/sys-front" "$D" "$SHA" "$MODEL"
both_on "*-${SHA:0:7}-$MODEL" "after the failed redeploy"
[[ "$(readlink "$APP_ROOT/previous")" == releases/*-"${PREV_SHA:0:7}-$MODEL" ]] \
  || { echo "the failed redeploy moved previous"; exit 1; }
both_active "after the failed redeploy"
echo "   both hosts still on deploy 2, previous still deploy 1"

step "--rollback to ${PREV_SHA:0:7}, and back: both hosts"
"$D" --rollback
both_on "*-${PREV_SHA:0:7}-$MODEL" "the rollback"
"$D" --rollback
both_on "*-${SHA:0:7}-$MODEL" "the roll-forward"
both_active "after the rollbacks"

step "--rollback to a release from before OPS-15 (no site of its own), and back"
# Deploy 1's release made to look like one from before OPS-15, for this step only.
prev_rel=$(readlink -f "$APP_ROOT/previous")
mv "$prev_rel/inference.nginx.conf" "$R/inference.nginx.conf.aside"
"$D" --rollback
[[ "$(cat "$SYSTEM_ROOT/etc/nginx/conf.d/coffee-cv.conf")" == "$OLD_SITE" ]] \
  || { echo "the rollback did not put the site from before OPS-15 back"; exit 1; }
"$D" --rollback
[[ "$(cat "$SYSTEM_ROOT/etc/nginx/conf.d/coffee-cv.conf")" == "include $APP_ROOT/current/inference.nginx*.conf;" ]] \
  || { echo "the roll-forward did not include the release's site again"; exit 1; }
mv "$R/inference.nginx.conf.aside" "$prev_rel/inference.nginx.conf"
both_on "*-${SHA:0:7}-$MODEL" "after the rollback to a release from before OPS-15, and back"
echo "   the old site while on it, the release's include after"

step "REHEARSAL PASSED"

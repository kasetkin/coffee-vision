#!/usr/bin/env bash
# Rehearse scripts/deploy_webapp.sh end to end on this machine, with no VM and no systemd
# (docs/ops1_release_isolation_plan.html §7): bootstrap, build, stage, smoke, flip, prune, --rollback,
# and the refusals. The deploy script runs unmodified with HOST=local against a scratch root.
#
#   scripts/rehearse_deploy_local.sh [scratch-dir]
#
# Both deploys use $REHEARSE_MODEL, by default the one model in models/: deploy 1 at $REHEARSE_PREV_REF
# (default origin/main~1), deploy 2 at origin/main, so the rollback switches between two releases of it. Until
# 2026-10-06 deploy 1 was the ResNet18 allrigs_cam_s123; it was retired with every other old model.
# REHEARSE_PREV_REF must be on origin/main and already hold $REHEARSE_MODEL.
#
# NOT YET RUN in this form (2026-10-06, aedfb93): the one-model rewrite above passed `bash -n` only. The last
# passing rehearsal was at 46ec30e, with allrigs_cam_s123 as deploy 1. Untested: deploy 1 at an older commit,
# and a rollback between two releases of the same model (same model_sha, told apart only by commit). Run it
# before the next release (a new model or segmenter) and drop this note once it passes.
#
# Stubbed: sudo (runs the command as you), systemd-run (runs the command in its working directory with
# its -E variables -- no sandbox, no caps), systemctl (starts the ExecStart of the rendered unit in the
# background, so the smoke and verify steps hit a real gunicorn from a real release venv), journalctl,
# nginx -t, useradd. NOT exercised here, only on the VM: the sandbox and caps themselves, running as
# coffee-cv, nginx, and the public-site checks.
#
# Runs from a clone whose `origin` is this repository, so the deploy's "commit must be on origin/main"
# check is real: commit to main here first. Needs the DVC-tracked models and the DINOv3 backbone locally.
set -euo pipefail

SRC=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
R=${1:-$(mktemp -d)}
R=$(mkdir -p "$R" && cd "$R" && pwd)
echo "rehearsal root: $R"
rm -rf "$R/clone" "$R/opt" "$R/sys" "$R/log" "$R/state" "$R/stubs" "$R/scratch"
mkdir -p "$R"/{opt,log,state,stubs,scratch} "$R"/sys/{etc/systemd/system,run/systemd/system,etc/nginx/conf.d,etc/logrotate.d}

# ------------------------------------------------------------------------------------------ the clone
git clone -q "$SRC" "$R/clone"
C="$R/clone"
ln -s "$SRC/.dvc/cache" "$C/.dvc/cache"
mkdir -p "$C/models_pretrained/dinov3"
ln -s "$SRC/models_pretrained/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth" "$C/models_pretrained/dinov3/"
# The segmenter models' L0 encoder and fine-tuned decoder (ticket ML-2): DVC-tracked, so the clone has only
# their .dvc files, and the deploy reads both from the working tree (the release manifest's `pretrained`).
ln -s "$SRC/models_pretrained/efficientvit_sam/efficientvit_sam_l0.pt" "$C/models_pretrained/efficientvit_sam/"
ln -s "$SRC/models/seg/ft_s123.pt" "$C/models/seg/"
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
[[ -n "${STUB_NGINX_FAIL:-}" ]] && { echo "nginx: [emerg] (stub) configured to fail" >&2; exit 1; }
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
cd "$wd" && exec env "${envs[@]}" "$@"
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
  local u=$1 f wd envf envs exe
  f=$(unitfile "$u") || { echo "stub systemctl: no unit $u" >&2; exit 5; }
  wd=$(sed -n 's/^WorkingDirectory=//p' "$f"); envf=$(sed -n 's/^EnvironmentFile=//p' "$f")
  envs=$(sed -n 's/^Environment=//p' "$f"); exe=$(sed -n 's/^ExecStart=//p' "$f")
  od -An -N8 -tx8 /dev/urandom | tr -d ' ' > "$ST/$u.inv"
  echo "=== invocation $(cat "$ST/$u.inv")" >> "$ST/$u.log"
  ( cd "$wd" && set -a && . "$envf" && set +a && exec env $envs setsid $exe ) >> "$ST/$u.log" 2>&1 &
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
  is-active) [[ $1 == --quiet ]] && shift; alive "$(norm "$1")" ;;
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
export UV_BIN="${UV_BIN:-$(command -v uv)}" PY_BIN="${PY_BIN:-$(command -v python3.12 || command -v python3)}"
export TRAIN_PY="${TRAIN_PY:-$(command -v python3)}" LOCAL_PY="${LOCAL_PY:-python3}"
export STUB_PATH="$S" STUB_STATE="$R/state" DEPLOY_SCRATCH="$R/scratch"
export MIN_MEM_GB=${MIN_MEM_GB:-2} MIN_FREE_GB=${MIN_FREE_GB:-5} PORT=18000 SMOKE_PORT=18001
# The rehearsal tests the deploy machinery on a scratch clone without the data, not the code: skip the
# deploy's full test suite (scripts/check.sh --full).
export DEPLOY_SKIP_CHECK=1
D="$C/scripts/deploy_webapp.sh"
MODEL=${REHEARSE_MODEL:-allrigs_dino3b16_seg_country_s123}
SHA=$(git -C "$C" rev-parse origin/main)
PREV_SHA=$(git -C "$C" rev-parse "${REHEARSE_PREV_REF:-origin/main~1}")
[[ "$PREV_SHA" != "$SHA" ]] || { echo "REHEARSE_PREV_REF resolves to origin/main itself"; exit 1; }
cleanup() { PATH="$S:$PATH" systemctl stop coffee-cv-web >/dev/null 2>&1 || true; }
trap cleanup EXIT
step() { printf '\n\n############ %s\n' "$*"; }
expect_fail() {   # expect_fail <pattern> <cmd...>: the command must fail and print the pattern
  local out rc=0
  out=$("${@:2}" 2>&1) || rc=$?
  if (( rc != 0 )) && grep -q -- "$1" <<<"$out"; then echo "   refused as expected: $(grep -m1 -- "$1" <<<"$out")"
  else echo "$out" | tail -20; echo "REHEARSAL FAILED: expected a refusal matching '$1' (exit $rc)"; exit 1; fi
}

step "bootstrap"
"$D" --bootstrap

step "refusal: a commit that is not on origin/main"
git -C "$C" -c user.name=rehearsal -c user.email=rehearsal@invalid commit -q --allow-empty -m "local only"
expect_fail "is not on origin/main" "$D" --stage-only HEAD "$MODEL"

step "refusal: a .pt whose md5 does not match its .pt.dvc"
mv "$C/.dvc/cache" "$C/.dvc/cache.off"; printf 'not a model' > "$C/models/$MODEL.pt"
expect_fail "no copy in the working tree or the DVC cache has md5" "$D" --stage-only "$SHA" "$MODEL"
rm "$C/models/$MODEL.pt"; mv "$C/.dvc/cache.off" "$C/.dvc/cache"

step "deploy 1: $MODEL at ${PREV_SHA:0:7} (service disabled: flip, no restart)"
"$D" "$PREV_SHA" "$MODEL"
[[ "$(readlink "$APP_ROOT/current")" == releases/*-"${PREV_SHA:0:7}-$MODEL" ]] || { echo "current not flipped"; exit 1; }

step "enable the service (the owner's Phase B), then --verify"
PATH="$S:$PATH" systemctl enable --now coffee-cv-web
"$D" --verify

step "refusal: nginx -t fails -> old site restored, nothing flipped"
before=$(readlink "$APP_ROOT/current")
sed -i 's|^    root .*|    root /changed/for/the/test;|' "$SYSTEM_ROOT/etc/nginx/conf.d/coffee-cv.conf"
cp "$SYSTEM_ROOT/etc/nginx/conf.d/coffee-cv.conf" "$R/site.before"
expect_fail "nginx -t rejected the new site" env STUB_NGINX_FAIL=1 "$D" "$SHA" "$MODEL"
cmp -s "$R/site.before" "$SYSTEM_ROOT/etc/nginx/conf.d/coffee-cv.conf" || { echo "site not restored"; exit 1; }
[[ "$(readlink "$APP_ROOT/current")" == "$before" ]] || { echo "current moved"; exit 1; }
echo "   site restored byte for byte; current unchanged"

step "deploy 2: $MODEL at ${SHA:0:7} (service enabled: restart + verify, prune)"
"$D" "$SHA" "$MODEL"
ls "$APP_ROOT/releases"

step "--compare on the current release"
"$D" --compare "$(basename "$(readlink "$APP_ROOT/current")")"

step "--rollback to ${PREV_SHA:0:7}, and back"
"$D" --rollback
[[ "$(readlink "$APP_ROOT/current")" == releases/*-"${PREV_SHA:0:7}-$MODEL" ]] || { echo "rollback did not switch"; exit 1; }
"$D" --rollback
[[ "$(readlink "$APP_ROOT/current")" == releases/*-"${SHA:0:7}-$MODEL" ]] || { echo "roll-forward did not switch"; exit 1; }

step "REHEARSAL PASSED"

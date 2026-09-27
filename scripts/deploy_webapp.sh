#!/usr/bin/env bash
# Deploy the web service as an immutable release under /opt/coffee-cv (docs/ops1_release_isolation_plan.html).
#
#   scripts/deploy_webapp.sh --bootstrap                     one-time VM setup (user, /opt, uv, logs)
#   scripts/deploy_webapp.sh <git-ref> <model>               full deploy: stage, smoke, flip, verify
#   scripts/deploy_webapp.sh --stage-only <git-ref> <model>  preflight, build, stage, smoke; no flip
#   scripts/deploy_webapp.sh --compare <release-id>          same answers from the training venv and the release venv?
#   scripts/deploy_webapp.sh --verify                        check what `current` serves
#   scripts/deploy_webapp.sh --rollback                      switch back to `previous`
#
# Runs locally and drives the VM over SSH. <git-ref> must resolve to a commit on origin/main. <model> is
# a name under models/ (e.g. allrigs_dino3b16_s123). Environment (defaults in brackets):
#   HOST [powervpsssh]  APP_ROOT [/opt/coffee-cv]  APP_USER [alioth, the deploy user]
#   SERVICE_USER [coffee-cv]  DOMAIN [required to flip or verify: the public site's host name]
#
# Safe while a training sweep runs (plan §5): every step that runs code on the VM is a transient
# systemd unit that can write only /opt/coffee-cv, capped at 1 CPU (smoke: 2) and 3 GB with no swap;
# preflight refuses without 5 GB available memory and 10 GB free on /opt; the sweep's PIDs and log
# size are printed before the first VM step and after the last. It never touches ~/coffee-vision*,
# never runs apt, and starts the service only if it is already enabled.
set -euo pipefail

usage() { sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

# ---------------------------------------------------------------------------------------------- config
HOST=${HOST:-powervpsssh}
APP_ROOT=${APP_ROOT:-/opt/coffee-cv}
APP_USER=${APP_USER:-alioth}
SERVICE_USER=${SERVICE_USER:-coffee-cv}
DOMAIN=${DOMAIN:-}
LOG_ROOT=${LOG_ROOT:-/var/log/coffee-cv}
# Where the public checks go; the local rehearsal sets it empty to skip them (there is no nginx there).
PUBLIC_URL=${PUBLIC_URL-${DOMAIN:+https://$DOMAIN}}
# Prefix for /etc and /run -- empty on the VM; a scratch directory in the local rehearsal.
SYSTEM_ROOT=${SYSTEM_ROOT:-}
PORT=${PORT:-8000}
SMOKE_PORT=${SMOKE_PORT:-8001}
SMOKE_CPU=${SMOKE_CPU:-200%}
CAP_MEM=${CAP_MEM:-3G}
MIN_MEM_GB=${MIN_MEM_GB:-5}
MIN_FREE_GB=${MIN_FREE_GB:-10}
LOCAL_PY=${LOCAL_PY:-python3}

UV_VERSION=0.12.19
UV_TARBALL_SHA256=23bf5552d220e0842b65c862097b2ebaeba0064b74eda5e565e77fd25969d8c8
UV_BIN=${UV_BIN:-/usr/local/bin/uv}
PY_BIN=${PY_BIN:-/usr/bin/python3.12}
TRAIN_PY=${TRAIN_PY:-/home/$APP_USER/coffee-vision-venv/bin/python}

# Fixed photos, pinned by path and sha256. The smoke photo is what every deploy sends to all three
# endpoints; the compare set is one per class across five sessions and three cameras, two of them HEIC.
SMOKE_PHOTO="dataset/2026-08-09__pixel_cam/class_003__Colombia_PinkBourbon/PXL_20260809_132314833.jpg db8ee9220eddad51b073d2ec703e4e31a2d2c7caf890149290a70e5bd143c077"
COMPARE_PHOTOS=(
  "dataset/2026-08-09__pixel_cam/class_001__Ethiopia_Sidamo/PXL_20260809_130838005.jpg 254de232ad1afbb01b56e69639560b03a80dc0df3e731aa2c031b4fae416d9dd"
  "dataset/2026-08-09__sony_cam/class_002__Kenya_AA/PIC_20260809_201618.JPG 4e46f8056de4cd24b830b5e74df272200b3108523cec2cc4f1861b3326ab6eab"
  "dataset/2026-08-25__iphone/class_003__Colombia_PinkBourbon/IMG_6258D.HEIC 072da5af33c26c46ca80fd78678b6662ae1262a996ffcf8f68a8b1ca1a27f03e"
  "dataset/2026-08-25__oneplus/class_004__CostaRica_LaPastora/PXL_20260825_195610599.jpg 474bd0b21d60a1e466fe394572f1dbd190d15ff4cc090525348b803a56f8fb08"
  "dataset/2026-08-07__box_pictures_all_classes/class_005__Guatemala_Tata/PXL_20260807_072408172.jpg 18939747dfd227d78e5ba2cad3afb30b98344f5c8b7c4251167b5b86fd01ac94"
  "dataset/2026-08-09__pixel_cam/class_006__Brazil_Cerrado/PXL_20260809_135958599.jpg 94fb46d397f3f429644200589ac7313dfad2e4fa0afdef26350624ce2296c69f"
  "dataset/2026-08-25__iphone/class_007__Brazil_MonteCristo/IMG_6291D.HEIC ba5293ff264b33261acdf83cfa764c863e513db5b36315c903dbf47e859b564c"
  "dataset/2026-08-09__sony_cam/class_008__Ethiopia_Kochere/PIC_20260809_202937.JPG 5d5d8458ba2bad2318bc359a35652f7e8adfb9b74d1ad1e370ec42d637a79b5e"
  "dataset/2026-08-27__oneplus_flash/class_009__Vietnam_Robusta/PXL_20260827_185019541.jpg 7f1f3fe470a03c83e2b68f6eb48369eb0f12c7ad73df068b7dd4c4c3158663a9"
  "dataset/2026-08-30__sony/class_010__Indonesia_Java/PIC_20260830_200129.JPG 932942572f0e3ad56d81d86cb5dfd963073ee9b2854340648e86ca76f2b5e6d9"
)

REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
SCRATCH=${DEPLOY_SCRATCH:-${XDG_CACHE_HOME:-$HOME/.cache}/coffee-cv-deploy}
DEPLOY_DIR="$REPO/webapp/deploy"
FIX="$APP_ROOT/fixtures"          # on the VM: probe script, photos, expected answers -- never in a release

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
sweep_state() {
  local pids log="" size=""
  pids=$(pgrep -f '[r]un_folds|[t]rain_baseline' | sort -n | tr '\n' ' ' || true)
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
              SYSTEM_ROOT="$SYSTEM_ROOT" CAP_MEM="$CAP_MEM" UV_BIN="$UV_BIN" PY_BIN="$PY_BIN" "$@")
  if [[ "$HOST" == local ]]; then
    env "${envs[@]}" PATH="${STUB_PATH:+$STUB_PATH:}$PATH" bash -c "$full" vm </dev/null
  else
    ssh -o BatchMode=yes -n "$HOST" "env $(printf '%q ' "${envs[@]}") bash -c $(printf '%q' "$full") vm"
  fi
}
to_vm() {   # to_vm <rsync args...> <local src> <remote dst>: rsync with the VM as destination
  local args=("$@") n=$#
  local dst=${args[$((n-1))]}
  unset 'args[$((n-1))]'
  if [[ "$HOST" == local ]]; then rsync "${args[@]}" "$dst"; else rsync "${args[@]}" "$HOST:$dst"; fi
}
render() {  # render <ref> <template path> NAME=value...: a template from the commit being deployed
  local ref=$1 tpl=$2; shift 2
  git -C "$REPO" show "$ref:$tpl" | "$LOCAL_PY" "$DEPLOY_DIR/render_template.py" "$@"
}
kv() { sed -n "s/^$1=//p" <<<"$2" | tail -1; }

# ---------------------------------------------------------------------------------- preflight / proof
preflight() {
  say "Preflight ($HOST, read-only)"
  local out
  out=$(vm '
    echo "MEM_AVAIL_KB=$(awk "/MemAvailable/ {print \$2}" /proc/meminfo)"
    echo "OPT_FREE_KB=$(df -Pk "$APP_ROOT" | awk "NR==2 {print \$4}")"
    echo "UV=$("$UV_BIN" --version 2>/dev/null | awk "{print \$2}")"
    sweep_state')
  local mem free uv
  mem=$(kv MEM_AVAIL_KB "$out"); free=$(kv OPT_FREE_KB "$out"); uv=$(kv UV "$out")
  PRE_SWEEP=$(grep '^SWEEP_' <<<"$out" || true)
  note "MemAvailable $((mem / 1048576)) GB (need $MIN_MEM_GB), free on $APP_ROOT $((free / 1048576)) GB (need $MIN_FREE_GB), uv ${uv:-missing}"
  note "sweep before: $(tr '\n' ' ' <<<"$PRE_SWEEP")"
  (( mem >= MIN_MEM_GB * 1048576 )) || die "only $((mem / 1024)) MB available memory; need ${MIN_MEM_GB} GB"
  (( free >= MIN_FREE_GB * 1048576 )) || die "only $((free / 1048576)) GB free on $APP_ROOT; need ${MIN_FREE_GB} GB"
  [[ "$uv" == "$UV_VERSION" ]] || die "uv on the VM is '${uv:-missing}', not $UV_VERSION -- run --bootstrap"
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
    for d in "$APP_ROOT"/releases/*-"$SUFFIX"; do [[ -f "$d/.complete" ]] && basename "$d"; done | tail -1' \
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
    [[ -f "$REPO/$path" ]] || die "$path is missing -- see models_pretrained/README.md"
    [[ "$(sha256sum < "$REPO/$path" | cut -c1-64)" == "$sha" ]] \
      || die "$path does not match its sha256 in models_pretrained/manifest.json at $SHA"
    install -D -m 0644 "$REPO/$path" "$STAGE/$path"
    note "$path (sha256 ${sha:0:16}...)"
  done < <(jq -r '.pretrained | to_entries[] | "\(.key) \(.value)"' <<<"$MANIFEST")

  "$LOCAL_PY" "$DEPLOY_DIR/write_build_info.py" --ref "$SHA" --model "$model" --out "$STAGE/webapp/BUILD_INFO.json" >/dev/null
  printf 'COFFEE_CV_CHECKPOINT=models/%s.pt\n' "$model" > "$STAGE/release.env"
  (cd "$STAGE" && find . -type f -printf '%P\n') \
    | "$LOCAL_PY" "$DEPLOY_DIR/release_manifest.py" --ref "$SHA" --model "$model" --check-staged - \
    || die "the staged tree does not match the manifest"

  check_fixtures
  note "expected smoke answers: the staged tree's own webapp.app, locally"
  local logs; logs=$(mktemp -d)
  (cd "$STAGE" && COFFEE_CV_CHECKPOINT="models/$model.pt" COFFEE_CV_LOG_DIR="$logs" PYTHONDONTWRITEBYTECODE=1 \
      "$LOCAL_PY" -B "$DEPLOY_DIR/release_probe.py" smoke "$REPO/${SMOKE_PHOTO%% *}" 2>"$logs/stderr") \
      > "$SCRATCH/$ID.expected.json" || { tail -20 "$logs/stderr"; die "the staged tree failed locally"; }
  rm -rf "$logs"
  jq -r '"   /classify \(.classify.body.verdict), top-1 \(.classify.body.ranked[0].id // "-"), model_sha \(.build.model_sha)"' \
    < <(sed -n 's/^PROBE //p' "$SCRATCH/$ID.expected.json")
}

check_fixtures() {
  local entry
  for entry in "$SMOKE_PHOTO" "${COMPARE_PHOTOS[@]}"; do
    [[ -f "$REPO/${entry%% *}" ]] || die "fixture photo ${entry%% *} is missing"
    [[ "$(sha256sum < "$REPO/${entry%% *}" | cut -c1-64)" == "${entry##* }" ]] || die "fixture photo ${entry%% *} changed"
  done
}

send_fixtures() {
  local files=() entry
  for entry in "$SMOKE_PHOTO" "${COMPARE_PHOTOS[@]}"; do files+=("$REPO/${entry%% *}"); done
  vm 'mkdir -p "$APP_ROOT/fixtures/photos" "$APP_ROOT/fixtures/expected"'
  to_vm -a --chmod=D755,F644 "${files[@]}" "$FIX/photos/"
  to_vm -a --chmod=F644 "$DEPLOY_DIR/release_probe.py" "$FIX/"
}

# --------------------------------------------------------------------------------------------- stage
stage() {
  say "Stage $ID on $HOST"
  local cur expect
  expect=$(jq -r '(.dvc + .pretrained) | to_entries[] | "\(.key) \(.value)"' <<<"$MANIFEST")
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
  cur=$(vm '[[ -e "$APP_ROOT/releases/$ID" ]] && { echo "EXISTS"; exit 0; }; readlink -f "$APP_ROOT/current" 2>/dev/null || true' ID="$ID")
  [[ "$cur" != EXISTS ]] || die "$APP_ROOT/releases/$ID already exists (incomplete: it is pruned by the next full deploy)"
  local t0=$SECONDS
  # --copy-dest: files unchanged since the current release are copied on the VM, not sent again
  # (a full copy, never a hard link -- each release stays independent).
  to_vm -a --chmod=D755,F644 ${cur:+--copy-dest="$cur/"} "$STAGE/" "$APP_ROOT/releases/$ID/"
  note "rsync $((SECONDS - t0)) s"

  t0=$SECONDS
  vm '
    R="$APP_ROOT/releases/$ID"
    while read -r path digest; do
      if (( ${#digest} == 32 )); then got=$(md5sum < "$R/$path" | cut -c1-32); else got=$(sha256sum < "$R/$path" | cut -c1-64); fi
      [[ "$got" == "$digest" ]] || { echo "$path arrived corrupt"; exit 1; }
    done <<<"$EXPECT"
    echo "   venv: uv sync --locked (capped: 1 CPU, $CAP_MEM)"
    capped 100% "$R" \
      -E UV_CACHE_DIR="$APP_ROOT/uv-cache" -E UV_LINK_MODE=copy -E UV_PYTHON_DOWNLOADS=never \
      -E UV_PROJECT_ENVIRONMENT="$R/.venv" -E UV_NO_PROGRESS=1 \
      "$UV_BIN" sync --project webapp --locked --no-dev --compile-bytecode --python "$PY_BIN" 2>&1 | tail -8 | sed "s/^/   /"
    capped 100% "$R" "$R/.venv/bin/python" -m compileall -q coffeecv webapp
  ' ID="$ID" EXPECT="$expect" || die "building the venv failed"
  note "venv $((SECONDS - t0)) s"

  check_staged_listing
  vm 'date -u +%FT%TZ > "$APP_ROOT/releases/$ID/.complete"' ID="$ID"
  note "complete: $APP_ROOT/releases/$ID"
}

check_staged_listing() {   # the release on the VM holds exactly the manifest (+ .venv, .complete, bytecode)
  local listing
  listing=$(vm 'cd "$APP_ROOT/releases/$ID" && find . -path ./.venv -prune -o -type f -printf "%P\n"' ID="$ID")
  "$LOCAL_PY" "$DEPLOY_DIR/release_manifest.py" --ref "$SHA" --model "$MODEL" --check-staged - <<<"$listing" \
    || die "the release on the VM does not match the manifest"
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
    body=$(curl -s -w "\n%{http_code} %{time_total}" -F "photo=@$P" "http://127.0.0.1:$SMOKE_PORT/classify")
    echo "SMOKE_CLASSIFY $(tail -1 <<<"$body" | cut -d" " -f1) $(head -n -1 <<<"$body")"
    echo "SMOKE_LATENCY_S $(tail -1 <<<"$body" | cut -d" " -f2)"
    body=$(curl -s -w "\n%{http_code}" -F "photo=@$P" "http://127.0.0.1:$SMOKE_PORT/crop")
    echo "SMOKE_CROP $(tail -1 <<<"$body") $(head -n -1 <<<"$body")"
    echo "SMOKE_PREVIEW $(curl -s -o /dev/null -w "%{http_code} %{content_type} %{size_download}" -F "photo=@$P" "http://127.0.0.1:$SMOKE_PORT/preview")"
    echo "SMOKE_RSS $(systemctl show -p MemoryPeak --value "$U")"
    echo "SMOKE_LOG $(grep "\"endpoint\": \"/classify\"" "$LOG_ROOT/smoke/app.jsonl" | tail -1)"
  ' UNIT="$unit" SMOKE_CPU="$SMOKE_CPU" SMOKE_PORT="$SMOKE_PORT" PHOTO="$(basename "${SMOKE_PHOTO%% *}")") || rc=$?
  printf '%s\n' "$out" > "$SCRATCH/$id.smoke.txt"
  if (( rc != 0 )); then
    grep -v '^SMOKE_LOG ' <<<"$out" | sed 's/^/   /'
    die "the smoke test failed"
  fi
  local rss; rss=$(sed -n 's/^SMOKE_RSS //p' <<<"$out"); [[ "$rss" =~ ^[0-9]+$ ]] || rss=0
  note "ready in $(sed -n 's/^SMOKE_READY_S //p' <<<"$out") s; /classify $(sed -n 's/^SMOKE_LATENCY_S //p' <<<"$out") s; peak memory $(( ${rss:-0} / 1048576 )) MB"
  note "(timings are under the caps; they gate nothing while a sweep may be running)"
  "$LOCAL_PY" "$DEPLOY_DIR/release_probe.py" compare-smoke "$SCRATCH/$id.expected.json" "$SCRATCH/$id.smoke.txt" \
    || die "the smoke answers differ from the local ones"
}

# ------------------------------------------------------------------------------ production files + flip
render_production() {   # installs the unit and nginx site from $SHA if they changed; sets NGINX_CHANGED
  say "Render production files"
  [[ -n "$DOMAIN" ]] || die "DOMAIN is not set (the public host name, for the nginx site)"
  local unit site out
  unit=$(render "$SHA" webapp/deploy/coffee-cv-web.service.template SERVICE_USER="$SERVICE_USER" \
           RELEASE_DIR="$APP_ROOT/current" PORT="$PORT" LOG_DIR="$LOG_ROOT")
  site=$(render "$SHA" webapp/deploy/coffee-cv.nginx.conf.template DOMAIN="$DOMAIN" APP_ROOT="$APP_ROOT")
  out=$(vm '
    UNIT_F="$SYSTEM_ROOT/etc/systemd/system/coffee-cv-web.service"
    SITE_F="$SYSTEM_ROOT/etc/nginx/conf.d/coffee-cv.conf"
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    if [[ "$(cat "$UNIT_F" 2>/dev/null)" != "$UNIT" ]]; then
      [[ -f "$UNIT_F" ]] && $SUDO cp -p "$UNIT_F" "$APP_ROOT/backup-coffee-cv-web.service.$stamp" 2>/dev/null || true
      printf "%s\n" "$UNIT" | $SUDO tee "$UNIT_F" >/dev/null
      $SUDO systemctl daemon-reload
      echo "UNIT_CHANGED=1"
    fi
    if [[ "$(cat "$SITE_F" 2>/dev/null)" != "$SITE" ]]; then
      backup="$SITE_F.bak-$stamp"
      [[ -f "$SITE_F" ]] && $SUDO cp -p "$SITE_F" "$backup"
      printf "%s\n" "$SITE" | $SUDO tee "$SITE_F" >/dev/null
      if ! msg=$($SUDO nginx -t 2>&1); then
        printf "%s\n" "$msg" | sed "s/^/nginx: /"
        if [[ -f "$backup" ]]; then $SUDO mv "$backup" "$SITE_F"; else $SUDO rm -f "$SITE_F"; fi
        exit 1
      fi
      $SUDO rm -f "$backup"
      echo "NGINX_CHANGED=1"
    fi
  ' UNIT="$unit" SITE="$site") || { printf '%s\n' "$out" | sed 's/^/   /'; die "nginx -t rejected the new site; the old one is restored, nothing flipped"; }
  printf '%s\n' "$out" | grep -v '^[A-Z_]*=1$' || true
  UNIT_CHANGED=$(kv UNIT_CHANGED "$out"); NGINX_CHANGED=$(kv NGINX_CHANGED "$out")
  note "unit: ${UNIT_CHANGED:+installed + daemon-reload}${UNIT_CHANGED:-unchanged}; nginx site: ${NGINX_CHANGED:+installed, nginx -t ok}${NGINX_CHANGED:-unchanged}"
}

flip() {   # flip <release id>: previous <- current, current <- releases/<id>, atomically each
  say "Flip current -> releases/$1"
  vm '
    cd "$APP_ROOT"
    old=$(readlink current 2>/dev/null || true)
    if [[ -n "$old" && "$old" != "releases/$ID" ]]; then
      ln -sfn "$old" previous.new && mv -T previous.new previous
    fi
    ln -sfn "releases/$ID" current.new && mv -T current.new current
    echo "   current -> $(readlink current); previous -> $(readlink previous 2>/dev/null || echo none)"
    if [[ -n "$NGINX_CHANGED" ]]; then $SUDO systemctl reload nginx && echo "   nginx reloaded"; fi
  ' ID="$1" NGINX_CHANGED="${NGINX_CHANGED:-}"
}

restart_if_enabled() {   # prints ENABLED=1 when it restarted the service
  vm '
    if [[ "$(systemctl is-enabled coffee-cv-web 2>/dev/null || true)" == enabled ]]; then
      $SUDO systemctl restart coffee-cv-web && echo "ENABLED=1"
    fi'
}

# -------------------------------------------------------------------------------------------- verify
verify() {   # returns non-zero on a failed check
  say "Verify what current serves"
  local out fails=0
  out=$(vm '
    cur=$(readlink -f "$APP_ROOT/current")
    echo "CURRENT=$(basename "$cur")"
    echo "INDEX_SHA=$(sha256sum < "$cur/webapp/static/index.html" | cut -c1-64)"
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
      echo "GET_CLASSIFY=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/classify")"
      inv=$(systemctl show -p InvocationID --value coffee-cv-web)
      echo "STARTUP=$($SUDO journalctl _SYSTEMD_INVOCATION_ID="$inv" -o cat --no-pager | grep -a "serving " | tail -1)"
    fi
    if [[ -f "$APP_ROOT/fixtures/expected/$(basename "$cur").json" ]]; then
      echo "EXPECTED=$(sed -n "s/^PROBE //p" "$APP_ROOT/fixtures/expected/$(basename "$cur").json")"
    fi
  ' PORT="$PORT") || { note "could not read the VM's state"; return 1; }
  local current enabled expected
  current=$(kv CURRENT "$out"); enabled=$(kv ENABLED "$out"); expected=$(kv EXPECTED "$out")
  note "current = $current; coffee-cv-web is ${enabled:-not installed}"
  check() { if [[ "$2" == "$3" ]]; then note "ok    $1: $2"; else note "FAIL  $1: got '$2', want '$3'"; fails=$((fails + 1)); fi; }

  if [[ "$enabled" == enabled ]]; then
    check "backend GET /" "$(kv GET_ROOT "$out")" 404
    check "backend GET /classify" "$(kv GET_CLASSIFY "$out")" 405
    local startup; startup=$(kv STARTUP "$out")
    check "startup line: commit" "$(sed -n 's/^[^{]*//p' <<<"$startup" | jq -r '.code.commit' 2>/dev/null)" "$(kv COMMIT "$out")"
    check "startup line: model_sha" "$(sed -n 's/^[^{]*//p' <<<"$startup" | jq -r '.model_sha' 2>/dev/null)" "$(kv MODEL_SHA "$out")"
  else
    note "the service is not enabled: backend checks skipped (the owner enables it: plan Phase B)"
  fi

  if [[ -z "$PUBLIC_URL" ]]; then
    note "SKIPPED public checks (PUBLIC_URL is empty)"
  else
    local body code
    body=$(mktemp)
    code=$(curl -s -o "$body" -w '%{http_code}' "$PUBLIC_URL/")
    check "public GET /" "$code" 200
    check "public page = the release's index.html" "$(sha256sum < "$body" | cut -c1-64)" "$(kv INDEX_SHA "$out")"
    local photo="$REPO/${SMOKE_PHOTO%% *}" want
    code=$(curl -s -o "$body" -w '%{http_code}' -F "photo=@$photo" "$PUBLIC_URL/classify")
    if [[ "$code" == 503 ]]; then sleep 3; code=$(curl -s -o "$body" -w '%{http_code}' -F "photo=@$photo" "$PUBLIC_URL/classify"); fi
    if [[ "$enabled" == enabled ]]; then
      check "public POST /classify" "$code" 200
      want=$(jq -r '.classify.body.ranked[0].id // "-"' <<<"$expected" 2>/dev/null)
      check "public /classify top-1" "$(jq -r '.ranked[0].id // "-"' < "$body" 2>/dev/null)" "${want:-?}"
    else
      check "public POST /classify (no backend yet)" "$code" 502
    fi
    rm -f "$body"
  fi
  return $(( fails > 0 ))
}

on_verify_failure() {
  local has_prev
  has_prev=$(vm '[[ -L "$APP_ROOT/previous" ]] && echo yes || true')
  if [[ "$has_prev" == yes ]]; then
    note "rolling back to previous"
    rollback_links
    restart_if_enabled >/dev/null
    verify || true
    die "verification failed; rolled back to previous"
  fi
  vm '$SUDO systemctl stop coffee-cv-web || true'
  die "verification failed and there is no previous release: coffee-cv-web stopped (page up, /classify 502)"
}

rollback_links() {
  vm '
    cd "$APP_ROOT"
    [[ -L previous ]] || { echo "no previous release to roll back to"; exit 1; }
    cur=$(readlink current); prev=$(readlink previous)
    [[ -f "$prev/.complete" ]] || { echo "$prev is incomplete"; exit 1; }
    ln -sfn "$prev" current.new && mv -T current.new current
    ln -sfn "$cur" previous.new && mv -T previous.new previous
    echo "   current -> $(readlink current); previous -> $(readlink previous)"'
}

# ------------------------------------------------------------------------------------ prune / compare
prune() {
  say "Prune: keep current and previous"
  vm '
    cd "$APP_ROOT"
    shopt -s nullglob
    keep="$(readlink -f current) $(readlink -f previous 2>/dev/null || true)"
    for d in releases/*/; do
      d=$(readlink -f "$d")
      [[ -d "$d" && "$d" == "$APP_ROOT"/releases/* ]] || continue
      case " $keep " in *" $d "*) continue ;; esac
      rm -rf "$d" && echo "   removed $(basename "$d")"
      rm -f "fixtures/expected/$(basename "$d").json"
    done
    capped 100% "$APP_ROOT" -E UV_CACHE_DIR="$APP_ROOT/uv-cache" "$UV_BIN" cache prune --quiet
    echo "   $(du -sh "$APP_ROOT" | cut -f1) in $APP_ROOT"'
}

compare() {   # compare <release id>: the staged code under the training venv vs its own venv
  local id=$1 names=() entry
  for entry in "${COMPARE_PHOTOS[@]}"; do names+=("$(basename "${entry%% *}")"); done
  preflight
  check_fixtures
  send_fixtures
  say "Compare $id: training venv ($TRAIN_PY -B) vs release venv, ${#names[@]} photos (capped: 1 CPU)"
  local run='
    R="$APP_ROOT/releases/$ID"
    [[ -f "$R/.complete" ]] || { echo "$R is not a complete release"; exit 1; }
    ckpt=$(sed -n "s/^COFFEE_CV_CHECKPOINT=//p" "$R/release.env")
    photos=(); for n in $NAMES; do photos+=("$APP_ROOT/fixtures/photos/$n"); done
    capped 100% "$R" -E COFFEE_CV_CHECKPOINT="$ckpt" -E COFFEE_CV_LOG_DIR=/tmp/probe-logs \
      -E PYTHONDONTWRITEBYTECODE=1 -E OMP_NUM_THREADS=1 \
      "$PY" -B "$APP_ROOT/fixtures/release_probe.py" classify "${photos[@]}" 2> >(tail -5 >&2)'
  mkdir -p "$SCRATCH"
  local t0=$SECONDS
  vm "$run" ID="$id" NAMES="${names[*]}" PY="$TRAIN_PY" > "$SCRATCH/$id.compare-training.json" || die "the training-venv run failed"
  note "training venv: $((SECONDS - t0)) s"; t0=$SECONDS
  vm "$run" ID="$id" NAMES="${names[*]}" PY="$APP_ROOT/releases/$id/.venv/bin/python" > "$SCRATCH/$id.compare-release.json" || die "the release-venv run failed"
  note "release venv: $((SECONDS - t0)) s"
  local rc=0
  "$LOCAL_PY" "$DEPLOY_DIR/release_probe.py" compare-classify "$SCRATCH/$id.compare-training.json" "$SCRATCH/$id.compare-release.json" || rc=$?
  postflight
  (( rc == 0 )) || die "the release venv does not reproduce the training venv bit for bit -- explain every difference before enabling the service"
  say "Compare OK: bit-identical"
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
MODE=deploy
case "${1:-}" in
  --bootstrap) bootstrap; exit 0 ;;
  --verify)    verify || die "verification failed"; exit 0 ;;
  --rollback)
    say "Rollback"
    rollback_links || die "rollback refused"
    restart_if_enabled >/dev/null
    verify || die "verification after rollback failed -- inspect by hand"
    exit 0 ;;
  --compare)   [[ $# -eq 2 ]] || usage; compare "$2"; exit 0 ;;
  --stage-only) MODE=stage; shift ;;
  -h|--help|"") usage ;;
esac
[[ $# -eq 2 ]] || usage
REF=$1 MODEL=$2

preflight
build "$REF" "$MODEL"
stage
smoke "$ID"
if [[ "$MODE" == stage ]]; then
  postflight
  say "Staged $ID (not flipped). Next: scripts/deploy_webapp.sh --compare $ID"
  exit 0
fi
render_production
flip "$ID"
ENABLED=$(restart_if_enabled)
if ! verify; then on_verify_failure; fi
prune
postflight
if [[ -z "$ENABLED" ]]; then
  say "Deployed $ID -- coffee-cv-web is NOT enabled, so nothing serves /classify yet. To serve it:"
  note "sudo systemctl enable --now coffee-cv-web     (on $HOST)"
  note "scripts/deploy_webapp.sh --verify              (here)"
else
  say "Deployed and verified $ID"
fi

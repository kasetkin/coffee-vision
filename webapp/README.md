# Coffee CV web service

Public web page + API for the shipped classifier: one photo in, either a
classification (scores for every class the model knows, 10 today) or a refusal
when the photo does not look like the training data (the OOD guard) out. See the
top-level `README.md` for what the model itself does; this doc is about
running the thing that serves it.

## Architecture

```
Browser --HTTPS--> nginx (TLS termination, static file, rate limit)
                      |-- GET /            -> /opt/coffee-cv/current/webapp/static/index.html
                      |-- POST /preview    -> proxy_pass -> gunicorn (127.0.0.1:8000)
                      |                                     -> Flask app -> coffeecv.dataset.load_rgb_image
                      \-- POST /classify   -> proxy_pass -> gunicorn (127.0.0.1:8000)
                                                              -> Flask app -> coffeecv.infer
```

- `app.py` -- the Flask app. Loads the model once at import time; `classify_one`
  from `coffeecv/infer.py` does the actual work, so this service and the CLI
  can never silently disagree about what counts as a refusal. `/preview`
  decodes whatever format was uploaded and re-encodes it as a JPEG thumbnail --
  it exists so the browser never needs native decode support for HEIC/AVIF/JXL
  to show a picture; the frontend calls it on every file selection, before
  `/classify` is ever hit.
- `static/index.html` -- the entire frontend. One file, inline CSS/JS, no build
  step.
- `deploy/` -- the templates (systemd unit, nginx site, logrotate), the release
  manifest, and the helpers `scripts/deploy_webapp.sh` uses; `setup_server.sh`
  installs only nginx, the cert directory and ufw on a fresh box. See
  "Deploying" below.
- `pyproject.toml` + `uv.lock` -- the service's own environment (uv).

## Where it runs: releases under /opt/coffee-cv

Production does not run from a checkout. Each deploy builds an immutable release from one commit on
`origin/main` and one model (docs/ops1_release_isolation_plan.html):

```
/opt/coffee-cv/                       owner alioth (the deploy user), 0755
  releases/<utc>-<sha7>-<model>/      only the files the service opens (webapp/deploy/release_manifest.py)
    coffeecv/ webapp/ models/ [models_pretrained/]  release.env  .venv/  .complete
  current  -> releases/...            what the service AND nginx use
  previous -> releases/...            the rollback target (only these two are kept)
  uv-cache/ fixtures/                 deploy's download cache; the photos of webapp/deploy/fixtures.txt + expected answers
/var/log/coffee-cv/                   owner coffee-cv
```

- The service runs as the `coffee-cv` system user with `ProtectHome=true`: nothing under `/home` is
  visible to it -- not the training checkout, not the training venv, not `~/.cache/torch`. What a
  sweep does to `~/coffee-vision` or `~/coffee-vision-venv` cannot change what production serves.
- Each release has its own `.venv`, built on the VM by `uv sync --locked` from `webapp/pyproject.toml`
  + `webapp/uv.lock` (versions copied from the production freeze; torch/torchvision from the PyTorch
  CPU index; timm without its download-only deps). Packages are copied, not hard-linked, so every
  release is independent and a rollback restores exactly what that release was tested with.
- **The model is chosen inside the release**, in `release.env` (`COFFEE_CV_CHECKPOINT=models/<name>.pt`),
  so code and model switch -- and roll back -- together. `app.py`'s own default is only for a local
  dev server.
- nginx serves the page from `current/webapp/static`, so the page and the API it calls change together
  at the flip. There is no copy step.
- ResNet18 checkpoints load with `weights=None` (the strict `load_state_dict` overwrites every
  parameter and buffer; `tests/test_model_and_ood_loading.py` proves the logits are bit-identical), so
  no ImageNet download or torch hub cache is needed. DINOv3 checkpoints ship their backbone file in
  `models_pretrained/`, checked against its sha256 at build time and again at every start.

## Which checkpoint is deployed

Whichever model the release was built with: `release.env` in `current`. `cat /opt/coffee-cv/current/release.env`,
or the `model` / `model_sha` fields on any line of `/var/log/coffee-cv/app.jsonl`.

### The OOD probe sidecar

A checkpoint may also carry `<name>.ood_probe.json` (~41 KB). **Its presence is
the switch**: with it, the guard refuses on a fitted linear probe over patch
embeddings; without it, on the Phase 10 centroid distance. There is no flag and
no config key -- a release ships it when the deployed commit has it beside the
model (the manifest includes it only then), and removing it is a commit plus a
deploy. `load_ood_probe` refuses a probe whose `checkpoint_sha` does not
match the weights, so a stale one fails loudly at startup instead of scoring in
the wrong embedding space.

Rebuild it with `python -m coffeecv.fit_ood_probe` (see
`docs/ood_guard_eval.md` for what it is calibrated against and why). It is
small, deterministic, and git-tracked beside the reference -- unlike the
`.ood_embeddings.npz` sidecar, which this service does not need at all.

A shipped checkpoint also carries a frozen `.classes.txt` sidecar, and
`config_for_checkpoint` redirects `classes_file` to it rather than reading
`dataset/classes.txt`. That is deliberate: `dataset/classes.txt` grows when a
new bean is added (class_010 arrived on 2026-08-30) while
a checkpoint's head is fixed at the classes it was trained on. Without the sidecar the head and
the label list would silently desync. Do not "fix" a shipped model's class list
by pointing it back at `dataset/classes.txt`.

## Deploying

Everything is `scripts/deploy_webapp.sh`, run from your workstation; it drives the VM over SSH. Set
`DOMAIN` to the public host name (it is never written into git).

```
scripts/deploy_webapp.sh --bootstrap                        # once per box: coffee-cv user, /opt/coffee-cv, uv, log dir, logrotate
DOMAIN=... scripts/deploy_webapp.sh <sha> <model>           # e.g. <sha> allrigs_dino3b16_s123
scripts/deploy_webapp.sh --stage-only <sha> <model>         # everything up to the flip, then stop
scripts/deploy_webapp.sh --compare <release-id>             # training venv vs release venv, 10 photos, must be bit-identical
DOMAIN=... scripts/deploy_webapp.sh --verify                # what `current` serves, backend + public site
DOMAIN=... scripts/deploy_webapp.sh --rollback              # current <-> previous, restart, verify
```

A full deploy: **preflight** (>= 5 GB available memory, >= 10 GB free on /opt, pinned uv; records the
sweep's PIDs and log size) -> **build locally** (refuses a commit not on `origin/main`; `git archive` of
the manifest; the `.pt` from the working tree or the local DVC cache, refused unless its md5 matches the
commit's `.pt.dvc`; the DINOv3 backbone, refused unless its sha256 matches `models_pretrained/manifest.json`;
`BUILD_INFO.json` + `release.env`; the expected smoke answers from the staged tree itself) -> **stage**
(rsync into `releases/<id>`, venv build, contents checked against the manifest, `.complete`) -> **smoke**
(the production unit template rendered for that release on port 8001, as `coffee-cv`, in the same
sandbox; `/classify`, `/crop`, `/preview` must match the local answers: same top-1, scores within 1e-4,
same `model_sha`) -> **render** the unit and nginx site from the commit (installed only if changed;
`nginx -t` failure restores the old site and stops) -> **flip** (`previous` <- `current`, `current` <-
the new release, each an atomic rename) -> **restart only if enabled** -> **verify** (backend `GET /`
-> 404, `GET /classify` -> 405, the startup journal line's commit and `model_sha`; public page = the
release's `index.html`; public `/classify` top-1 as expected; on failure it rolls back, or stops the
service if there is nothing to roll back to) -> **prune** to `current` + `previous`.

**The deploy never enables the service.** If `coffee-cv-web` is disabled it stops after the flip and
prints the two commands that bring it up: `sudo systemctl enable --now coffee-cv-web` on the VM, then
`scripts/deploy_webapp.sh --verify`.

**Fallback while only one release exists:** `scripts/deploy_webapp.sh <sha> allrigs_cam_s123` (the
ResNet18 model; about 10 minutes). Both models are covered by `tests/test_release_manifest.py`.

### Deploying while a sweep trains

Allowed, up to and including the flip. Every step that runs code on the VM (venv build, compileall,
`--compare`, the smoke) is a transient systemd unit with `ProtectSystem=strict` and `ProtectHome=
read-only`, writable only under `/opt/coffee-cv`, at `CPUQuota=100%` (smoke 200%), `MemoryMax=3G`,
`MemorySwapMax=0` -- an overrun kills the deploy step, never the sweep. The deploy prints the sweep's
PIDs and log size before its first VM step and after its last. Never during a sweep: `setup_server.sh`
(apt, ufw), `remote_launch.sh` with default paths (it deletes `~/sweep.log`), git commands that move the
VM's checkout, reboots.

**Keeping the site on during a sweep** is fine for the frozen DINOv3 model: measured 2026-09-27/28,
`/classify` takes 12.5 s mean and 13.7 s max server-side during a ResNet18 sweep, against 7.0 s idle, well inside
gunicorn's and nginx's 30 s. The site and the sweep share the CPU with no priority between them. Compare
latencies with `latency_ms` in `app.jsonl`: end-to-end times from a slow uplink include several seconds of upload.

### After changing a dependency

Edit `webapp/pyproject.toml`, re-lock with the pinned uv (`uv lock --project webapp`), commit both. A
lock that does not match `pyproject.toml` is refused by `uv sync --locked` at deploy time. A version
change is a change in numerics: run `--compare` on the staged release before enabling it.

## Day-2 commands

```
systemctl status coffee-cv-web nginx
readlink /opt/coffee-cv/current /opt/coffee-cv/previous   # which releases
journalctl -u coffee-cv-web -f         # startup: commit, model_sha, OOD guard
tail -f /var/log/coffee-cv/app.jsonl   # per-request logs: verdicts, timing, errors
                                        # (never image bytes or filenames --
                                        # see docs/logging_plan.html)
tail -f /var/log/nginx/coffee-cv.access  # nginx side of the same requests --
                                        # no client IP, joined to the line
                                        # above by request id
```

## What lives outside git, and where

Three things are deliberately never committed, and never named in this file
either -- see `webapp/deploy/coffee-cv.nginx.conf.template`,
`setup_server.sh` and `scripts/deploy_webapp.sh` for exactly how each is handled:

- **The domain.** Passed as `DOMAIN=...` to `scripts/deploy_webapp.sh`; the
  nginx config template uses a placeholder everywhere the domain would
  appear. Worth noting privately (e.g. your own shell history) since it's not
  written anywhere in this repo.
- **The DNS record.** Point your domain's A record at the VM's public IP
  yourself -- not something a script here can do for you.
- **The TLS cert/key.** `setup_server.sh` creates `/etc/nginx/ssl/coffee-cv/`
  (root-owned, `700`) for you to place your Porkbun files into, using
  Porkbun's own filenames directly -- `ssl_certificate` points at
  `domain.cert.pem` (Porkbun's bundle already concatenates the leaf +
  intermediate chain into this one file, confirmed by counting `BEGIN
  CERTIFICATE` blocks -- no separate fullchain build step needed) and
  `ssl_certificate_key` at `private.key.pem`. `setup_server.sh` can create the
  directory but can't set permissions on files it doesn't create, so this is a
  manual step: `private.key.pem` needs `600` (owner read/write only --
  Porkbun's default download permissions are world-writable `666`, which is
  fine only as long as the containing directory stays `700` root-owned, but
  fix it anyway as defense in depth); `domain.cert.pem` and `public.key.pem`
  (not used by nginx) are fine at `644`. Also worth a one-time check with
  `openssl x509 -noout -text -in domain.cert.pem | grep -A1 'Subject
  Alternative Name'` that the cert's SAN list actually covers whatever
  subdomain you're deploying to (a wildcard `*.yourdomain` or the exact
  hostname) -- Porkbun issues per-domain, not automatically per-subdomain.

  `nginx.service` runs `nginx -t` on start, and a missing cert fails that
  check for the whole config -- so if the VM reboots before the cert files
  are ever placed, nginx won't come up automatically that one time. Once the
  real files are on disk they persist across every future reboot like any
  other file.

## Design choices worth knowing before changing anything

- **One gunicorn worker, sync class, deliberately.** Torch already uses all
  physical cores per inference call on this CPU-only box; a second worker
  would just make both requests slower fighting over the same cores, not add
  real throughput. Serializing behind one worker, backed by nginx's
  `limit_conn` (below), is the better trade for a low-traffic endpoint.
- **Rate limiting is keyed by connection, not IP** (NAT can put many real
  users behind one IP) -- `limit_conn` caps total *concurrent* `/classify`
  requests at exactly 1, matching the single gunicorn worker. A request takes
  about 7 s server-side (12-14 s while a sweep trains), so this alone already enforces well under "1 request/sec"; a
  separate rate-limit zone would just re-enforce a limit this already
  guarantees.
- **The uploaded photo is never stored, logged, or served back to anyone.**
  It's written to a tempfile with a fixed generic suffix (never derived from
  the client's filename), decoded once by Pillow, and deleted in a `finally`.
  This is also why AVIF/JXL/HEIF-in-JPG-clothing-style attacks are inert here
  structurally: nothing in this pipeline ever executes, unzips, or serves the
  upload as anything other than pixel data passed to `PIL.Image.open`.
- **`coffee-cv-web.service` runs sandboxed**, as its own `coffee-cv` system user
  (`NoNewPrivileges`, `PrivateTmp`, `ProtectSystem=strict`, `ProtectHome=true`,
  empty capability set,
  restricted address families, seccomp `@system-service` filter) because this
  process parses attacker-supplied image data over a public endpoint, and no
  code review guarantees a native image codec (libjpeg/libwebp/libheif/
  libavif/libjxl) is bug-free. None of this changes application behavior --
  it only shrinks the blast radius if one of those codecs is ever exploited.
  `MemoryDenyWriteExecute=true` was considered but left out of the committed
  unit pending a real test on the VM (some ML runtimes JIT-compile at
  runtime) -- try it there; add it if torch's CPU inference path still works,
  drop the idea if it doesn't.
- **Format support is pinned in `coffeecv/dataset.py`, not here.** JPG/PNG/
  WEBP need no plugin; HEIF/HEIC, AVIF, and JPEG XL are registered with
  Pillow right next to each other in `load_rgb_image`'s home module, so the
  CLI and this web service can never disagree on what formats they accept.
  Camera RAW (DNG/CR2/CR3/NEF/ARW/RAF/ORF/RW2/PEF/SRW) is different --
  Pillow can't open these at all, so `load_rgb_image` branches explicitly
  and routes them through `rawpy` (LibRaw) instead, normalizing rawpy's own
  exception type to `OSError` so it joins the same "can't read this photo"
  handling every other format's decode failure already gets.

# Coffee CV web service

Public web page + API for the shipped classifier: one photo in, either a
classification (scores for all 9 classes) or an actionable refusal ("move the
camera back", "this doesn't look like the training data") out. See the
top-level `README.md` for what the model itself does; this doc is about
running the thing that serves it.

## Architecture

```
Browser --HTTPS--> nginx (TLS termination, static file, rate limit)
                      |-- GET /            -> static index.html
                      \-- POST /classify   -> proxy_pass -> gunicorn (127.0.0.1:8000)
                                                              -> Flask app -> coffeecv.infer
```

- `app.py` -- the Flask app. Loads the model once at import time; `classify_one`
  from `coffeecv/infer.py` does the actual work, so this service and the CLI
  can never silently disagree about what counts as a refusal.
- `static/index.html` -- the entire frontend. One file, inline CSS/JS, no build
  step.
- `deploy/` -- everything needed to stand the server up, committed so the
  configuration is reviewable and reproducible rather than living in shell
  history. See "Server bootstrap" below.

## Which checkpoint is deployed

`app.py` hardcodes `CHECKPOINT = REPO_ROOT / "models" / "allrigs_mixstyle05_e100p20_s17.pt"`
near the top of the file. To ship a different model: change that one line,
make sure the checkpoint's `.json` card and `.ood_reference.json` are present
beside it (both locally and on the VM -- see "Getting code onto the VM"
below), redeploy (see "Code/model change" below).

## Two separate procedures -- don't conflate them

**Code or model change (the common case):** `rsync` the updated files to the
VM, then:
- Python/model changes: `systemctl restart coffee-cv-web`
- Static file or nginx-config changes: `nginx -t && systemctl reload nginx`

This is *not* a re-run of `setup_server.sh`. `systemctl enable --now` is a
no-op on a unit that's already running, and gunicorn's sync worker doesn't
hot-reload code -- re-running the bootstrap script would silently leave the
old code running.

**Server bootstrap or config drift** (a new box, or `/etc/nginx`, `ufw`, or the
systemd unit itself changed):

```
DOMAIN=yourdomain.example sudo -E ./webapp/deploy/setup_server.sh
```

Idempotent, safe to re-run. It configures the OS/server layer only -- it does
not restart the app service if it's already running, so a config-only re-run
is typically followed by the restart/reload above anyway. Full step list is in
the script itself, with reasoning inline.

## Getting code (and the model) onto the VM

`rsync` the repo per the recipe already used for training deploys (exclude
`.dvc/cache`, `tensorboard/`, etc.). The one wrinkle specific to this service:
**the shipped checkpoint's binary files don't come via `dvc pull`** -- this
repo has no DVC remote configured at all (see top-level `README.md`), so
`models/allrigs_mixstyle05_e100p20_s17.{pt,json,ood_reference.json}` has to be
copied directly (`rsync`/`scp`) alongside the code. After copying, worth a
quick integrity check since this bypasses DVC's own hash verification:

```
sha256sum models/allrigs_mixstyle05_e100p20_s17.*
# compare against the same command run on the VM
```

## Prerequisite: torch's pretrained-weight cache

`coffeecv/model.py`'s `build_model()` unconditionally requests ImageNet
pretrained weights from `download.pytorch.org` when building the backbone,
even though they're immediately overwritten by `load_state_dict` with our own
checkpoint. That means **the app needs `~/.cache/torch/hub/checkpoints/`
populated before its first start**, or it'll try to reach the internet on
every service start for a download whose result is discarded. On the VM this
repo has been training on, that cache already exists from the training setup.
Deploying to a *new* box: copy that directory over first, or the service will
hang/fail to start without internet access.

## Day-2 commands

```
systemctl status coffee-cv-web nginx
systemctl restart coffee-cv-web        # after a code/model change
journalctl -u coffee-cv-web -f         # tail logs (verdicts + exceptions;
                                        # never image bytes or filenames)
nginx -t && systemctl reload nginx     # after a static/nginx-config change
```

## What lives outside git, and where

Three things are deliberately never committed, and never named in this file
either -- see `webapp/deploy/coffee-cv.nginx.conf.template` and
`setup_server.sh` for exactly how each is handled:

- **The domain.** Passed as `DOMAIN=...` when running `setup_server.sh`; the
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
  requests at exactly 1, matching the single gunicorn worker. Inference itself
  takes 6-8s, so this alone already enforces well under "1 request/sec"; a
  separate rate-limit zone would just re-enforce a limit this already
  guarantees.
- **The uploaded photo is never stored, logged, or served back to anyone.**
  It's written to a tempfile with a fixed generic suffix (never derived from
  the client's filename), decoded once by Pillow, and deleted in a `finally`.
  This is also why AVIF/JXL/HEIF-in-JPG-clothing-style attacks are inert here
  structurally: nothing in this pipeline ever executes, unzips, or serves the
  upload as anything other than pixel data passed to `PIL.Image.open`.
- **`coffee-cv-web.service` runs sandboxed** (`NoNewPrivileges`, `PrivateTmp`,
  `ProtectSystem=strict`, `ProtectHome=read-only`, empty capability set,
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
  WEBP need no plugin; HEIF/HEIC and now AVIF/JPEG XL are registered with
  Pillow right next to each other in `load_rgb_image`'s home module, so the
  CLI and this web service can never disagree on what formats they accept.

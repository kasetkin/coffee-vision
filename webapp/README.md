# Coffee CV web service

Public web page + API for the shipped classifier: one photo in, either a
classification (scores for every class the model knows, 10 today) or a refusal
when the photo does not look like the training data (the OOD guard) out. See the
top-level `README.md` for what the model itself does; this doc is about
running the thing that serves it.

## Architecture

```
Browser --HTTPS:443--> the proxy's stream block (reads the SNI; logs the visitor's IP per session)
  --> THE FRONT, the proxy: nginx on 127.0.0.1:12347 (TLS termination, static files, one-at-a-time gate, 40 MB cap)
        |-- GET /            -> /opt/coffee-cv/current/webapp/static/index.html (no-cache)
        |-- GET /robots.txt, /llms.txt, /openapi.json, /.well-known/ard.json
        |                    -> the release's webapp/static/ files; any other path is a 404
        \-- POST /analyze    -> check service, coffee-cv-front (127.0.0.1:8100): the form and the photo's format
              --tailnet, plain HTTP--> THE INFERENCE HOST, yappy: nginx on its tailnet address, port 12346
                  \-- POST /analyze -> gunicorn (127.0.0.1:8000) -> Flask app -> coffeecv.infer
                                       -> a streamed answer: preview line, then classification line
```

Two hosts, one release (ADR 0031, ticket OPS-15, live 2026-11-02). The front is the small always-on proxy that
serves the domain; it has no ML code (`front/`), and since ticket ML-19 (ADR 0033) it canonicalizes each photo with
`coffeecv/canonical.py` and forwards the canonical photo in its place. The inference host has the GPU and a slow link
(2.1 Mbit/s in): the front refuses a photo over 40 MB (ML-19 D23), in a format the app cannot read, or over its
pixel ceiling (60 MP, D38; RAW 16 MP, D24), before it is converted, and only the canonical photo crosses; the host refuses anything that is not a canonical
photo. While the app is down (a deploy,
or the host) the page stays up and `/analyze` answers `503 {"error": "the classifier is offline"}`.

- `app.py` -- the Flask app. Loads the model once at import time; `classify_one`'s
  two steps from `coffeecv/infer.py` do the actual work, so this service and the
  CLI can never silently disagree about what counts as a refusal. One endpoint,
  `POST /analyze` (ticket OPS-7, ADR 0023): the page sends the photo once (form
  field `photo`, and `skip_crop=1` for "Use full photo instead"), the server
  decodes and segments it once, and answers `application/x-ndjson`, one JSON
  object per line. First `{"part": "preview"}`: `thumbnail` (a base64 JPEG, long
  side <= 1024 px, so the browser never needs native HEIC/AVIF/JXL/RAW decoding),
  `cropped`, `box` (the crop as fractions of the decoded photo), `needs_review`,
  `mask` (the photo's bean region: a 1-bit palette PNG of the whole photo, long
  side <= 1024 px, index 0 transparent, base64; null on a fallback and on
  `skip_crop`) and `seg_fallback` (no usable mask, so the whole photo is
  classified). Then `{"part": "classification"}` with the verdict and the ranked
  classes, or `{"part": "error", "message": ...}` if the classification fails
  after the 200 has gone out. An unreadable photo is a plain 400 before any line.
  The app sends `X-Accel-Buffering: no`, so both nginx servers, and the check
  service between them, pass the preview on before the classification is done.
- `static/index.html` -- the entire frontend. One file, inline CSS/JS, no build
  step.
- `static/llms.txt`, `static/openapi.json`, `static/.well-known/ard.json` -- the
  service described for LLM agents: a Markdown guide, the API's OpenAPI 3.1
  description, and the Agentic Resource Discovery manifest pointing at both. See
  "For agents" below.
- `front/` -- the front's check service (`check_service.py`, `upload_check.py`):
  its own `pyproject.toml` + `uv.lock`, with the photo stack canonicalization needs at training's versions
  (`coffeecv.lock_check`); its release carries `coffeecv/canonical.py` and what it imports.
- `deploy/` -- the templates (systemd units, both nginx sites, the stream block's
  session log, logrotate), the release manifest, and the helpers
  `scripts/deploy_webapp.sh` uses; `setup_server.sh` installs only nginx, a
  sysctl and the SSH ufw rule (ufw stays disabled) on a fresh inference host, and
  `setup_front.sh` prepares the front once, as root. See "Deploying" below.
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

The front has the same layout under its own `/opt/coffee-cv/`, with the same release id: the page, the front's
nginx site (`front.nginx.conf`), the check service and its `.venv` (`webapp/deploy/release_manifest.py --front`). Each
host has its own `current` and `previous`; the deploy flips, and `--rollback` reverts, both together.

- The service runs as the `coffee-cv` system user with `ProtectHome=true`: nothing under `/home` is
  visible to it -- not the training checkout, not the training venv, not `~/.cache/torch`. What a
  sweep does to `~/coffee-vision` or `~/coffee-vision-venv` cannot change what production serves.
- Each release has its own `.venv`, built on the host by `uv sync --locked` from `webapp/pyproject.toml`
  + `webapp/uv.lock` on uv's own CPython (`PY_VERSION` in the deploy script, under `/opt/coffee-cv/python`,
  installed by `--bootstrap`; never the host's system Python), versions held to the production freeze. Since ticket ML-7
  it holds no torch, torchvision or timm: the models run on LiteRT (`ai-edge-litert`). Packages are copied, not hard-linked, so every
  release is independent and a rollback restores exactly what that release was tested with.
- **The model is chosen inside the release**, in `release.env` (`COFFEE_CV_CHECKPOINT=models/<name>.pt`),
  so code and model switch -- and roll back -- together. What runs is that model's LiteRT conversion beside it,
  `models/<name>.litert/` (the segmenter's encoder and decoder, the DINOv3 backbone and the head). `app.py`'s own
  default is only for a local dev server.
- The front's nginx serves the page from its `current/webapp/static`, and both hosts flip in one step, so the
  page and the API it calls change together. There is no copy step.
- A release ships no pretrained weights (ticket ML-7, plan P10): the converted directory carries every weight the
  service runs, each file checked at build time by md5 against `dvc.lock`'s record of the `litert_export` stage.
  Only a model with that stage can be deployed; today that is the live one alone.

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

Rebuild it with `python -m coffeecv.fit_ood_probe` and measure it with
`python -m coffeecv.ood_eval` (ADR 0016: a fixed 0.5, fitted on the segmenter
dataset's training and validation negatives, measured on its test split). It is
small, deterministic, and git-tracked beside the reference -- unlike the
`.ood_embeddings.npz` sidecar, which this service does not need at all.

A shipped checkpoint also carries a frozen `.classes.txt` sidecar, and
`config_for_checkpoint` redirects `classes_file` to it rather than reading
`dataset/classes.txt`. That is deliberate: `dataset/classes.txt` grows when a
new bean is added (class_010 arrived on 2026-08-30) while
a checkpoint's head is fixed at the classes it was trained on. Without the sidecar the head and
the label list would silently desync. Do not "fix" a shipped model's class list
by pointing it back at `dataset/classes.txt`.

A frozen list comes in one of two formats, and `coffeecv.class_list.load_classes` reads both. Models
shipped before ticket ML-3 (all retired on 2026-10-06, e.g. `allrigs_cam_s123`) carried the old
`id;label` lines, and each folder id is a class: the page shows `006` and its label. Since ML-3
(`allrigs_dino3b16_seg_country_s123`) the list is in the D15 format, `id;Country[,Region];Misc`, and
the class is the country: the page shows ten keys such as `Brazil` and `CostaRica`, and the request
log's `top1_class` holds a country key. Only a release from before ML-3 would show folder classes again.

## Deploying

Everything is `scripts/deploy_webapp.sh`, run from your workstation; it drives both hosts over SSH. Set
`DOMAIN` to the public host name, and `TLS_CERT` and `TLS_KEY` to the front's certificate files on the proxy. The
inference host is the SSH alias `webappremote` (`HOST`; the deploy user `APP_USER` is its SSH user); its nginx
listens in plain HTTP on its tailnet address, port 12346 (`INFERENCE_PORT`), only. The front is the alias
`webproxy-deploy` (`FRONT_HOST`, the deploy user `coffee-deploy`, whose sudo there covers only the check service,
`nginx -t` and nginx's reload); the site is `https://<DOMAIN>/`. Day-to-day notes: `docs/agents/deploy.md`.

```
webapp/deploy/setup_front.sh                                # once per front, run by the owner as root (ROOT_HOST)
scripts/deploy_webapp.sh --bootstrap                        # once per inference host: coffee-cv user, /opt/coffee-cv, uv, Python, log dir, logrotate
DOMAIN=... TLS_CERT=... TLS_KEY=... scripts/deploy_webapp.sh <sha> <model>   # e.g. <sha> dino3b16_segxl0v3_country_aug8_s123
scripts/deploy_webapp.sh --stage-only <sha> <model>         # everything up to the flip, then stop
scripts/deploy_webapp.sh --compare <release-id>             # the release on the host vs PyTorch here, 12 photos, ML-7 D7's bars
DOMAIN=... scripts/deploy_webapp.sh --verify                # what `current` serves on both hosts + the public site
DOMAIN=... scripts/deploy_webapp.sh --rollback              # current <-> previous on both hosts, restart, verify
```

**The host's GPU is a deploy dependency** (ticket ML-7, ADR 0019). A release whose converted model runs on the
GPU (`yappy`'s Intel N100 iGPU today) needs on the host: Mesa's Vulkan driver for Intel (ANV,
`/usr/share/vulkan/icd.d/intel_icd.x86_64.json`, which `release.env`'s `VK_ICD_FILENAMES` names), the render node
`/dev/dri/renderD128` (the unit's `SupplementaryGroups=render`), and i915's counters in sysfs for the GPU proof.
LiteRT's WebGPU accelerator comes from the release's own venv (the `ai-edge-litert` wheel). The sandbox leaves
no writable home, so no shader cache persists: shaders compile at each boot, within its 20 s (ML-7 P7 measured no
cold-start change with a cache). A worker that cannot prove the GPU refuses to serve (ADR 0020), so a driver upgrade or a
lost render group shows as a failed smoke test or a service that does not start, never as silent CPU serving.

A full deploy: **preflight** on both hosts (on the inference host >= 10 GB free on /opt, pinned uv, its tailnet
address; on the front >= 200 MB free, the pinned uv and Python, the site include and the check service's unit
`setup_front.sh` installed; records the sweep's PIDs and log size) -> **build locally** (refuses a commit not on `origin/main`; `git archive` of
the manifest; the `.pt` from the working tree or the local DVC cache, refused unless its md5 matches the
commit's `.pt.dvc`; its LiteRT export (`models/<name>.litert/`, ticket ML-7) by md5 from `dvc.lock`; the warm-up
photo; `BUILD_INFO.json` + `release.env`; the expected smoke answers from the staged tree itself, on this machine's
CPU) -> **stage**
on both hosts (rsync into `releases/<id>`, venv build, contents checked against each host's manifest, `.complete`)
-> **stop the live service** (ADR 0026: the host fits one GPU worker; then >= 5 GB available memory, and any failure
before the flip starts it again; the front's check service stops too, and the front answers "the classifier is
offline" meanwhile) -> **smoke**
(the production unit template rendered for that release on port 8001, as `coffee-cv`, in the same
sandbox; `/analyze`'s two parts must match the local answer: same verdict and top-1, scores within
ML-7 D7's 0.05, the preview's mask IoU at least 0.999 and its thumbnail a JPEG of the same size, same `model_sha`; on a GPU host the worker's boot must prove the GPU,
ADR 0020) -> **render** the unit from the commit (installed only if changed) -> **flip** both hosts
(`previous` <- `current`, `current` <- the new release, each an atomic rename; both sites ride in the release,
`nginx -t` on both hosts before either reloads, and a failure reverts both) -> **restart only if enabled** ->
**verify** (the front's `current` is the inference host's; the check service's `GET /analyze` -> 405; backend
`GET /` -> 404, `GET /analyze` -> 405, the startup journal line's commit and `model_sha`; public page = the
release's `index.html`; public `/analyze` answers both parts with the top-1 expected, and the time each part
arrived is reported, not checked (OPS-7 D11); on failure it reverts both hosts, or stops the
service if there is nothing to roll back to) -> **prune** to `current` + `previous`, on the host and in this machine's staging directory (`$DEPLOY_SCRATCH`, default `~/.cache/coffee-cv-deploy`).

A reload does not rebind a changed listen address: `docs/agents/deploy.md` has the case, and the rollback to the
release from before OPS-15, which `--rollback` cannot do.

**The deploy never enables the service.** If `coffee-cv-web` is disabled it stops after the flip and
prints the two commands that bring it up: `sudo systemctl enable --now coffee-cv-web` on the host, then
`scripts/deploy_webapp.sh --verify`.

**Fallback:** `--rollback` to `previous`. `models/` still holds the model the live one replaced,
`allrigs_dino3b16_segxl0v3_country_s42` (exp267, live 2026-10-13 to 2026-10-20; `docs/status.md`), with its LiteRT
conversion (dvc.yaml's `litert_export` stage), so it deploys as it stands:
`scripts/deploy_webapp.sh <git-ref> allrigs_dino3b16_segxl0v3_country_s42`. Every older model was retired with its card
kept in git history (`docs/status.md`): the ResNet18 fallback, `allrigs_cam_s123`, on 2026-10-06, the ResNet18 code on
2026-10-20, and the two models whose segmenters left the repo on 2026-10-21. An older release can still be
rebuilt from a commit that holds it, `scripts/deploy_webapp.sh <sha> <model>`, as long as what that commit's build
needs (its `.pt`, and since ML-7 its LiteRT export) is in the local DVC cache (`dvc fetch` at that commit); the
release is built from that commit's tree.

### Deploying while a sweep trains

Every step that runs code on the host (venv build, compileall, `--compare`, the smoke) is a transient
systemd unit with `ProtectSystem=strict` and `ProtectHome=read-only`, writable only under
`/opt/coffee-cv`, at `CPUQuota=$CAP_CPU` (smoke `$SMOKE_CPU`), `MemoryMax=$CAP_MEM`, `MemorySwapMax=0`.
The defaults (400%, 400%, 7G) take the whole of `webappremote`, which trains nothing. On a host that
also runs a sweep, deploy with `CAP_CPU=100% SMOKE_CPU=200% CAP_MEM=3G`, the old VM's caps: then an
overrun kills the deploy step, never the sweep, and deploying is allowed up to and including the flip. The deploy prints the sweep's
PIDs and log size before its first host step and after its last. Never during a sweep: `setup_server.sh`
(apt, ufw), `remote_launch.sh` with default paths (it deletes `~/sweep.log`), git commands that move the
host's checkout, reboots.

**Keeping the site on during a sweep** is fine for the frozen DINOv3 model: measured 2026-09-27/28,
the old `/classify` took 12.5 s mean and 13.7 s max server-side during a ResNet18 sweep, against 7.0 s idle, well inside
gunicorn's and nginx's 30 s. The site and the sweep share the CPU with no priority between them. Compare
latencies with `latency_ms` in `app.jsonl` (and `/analyze`'s `preview_ms`, when its preview went out):
end-to-end times from a slow uplink include several seconds of upload.

### After changing a dependency

Edit `webapp/pyproject.toml`, re-lock with the pinned uv (`uv lock --project webapp`), commit both. A
lock that does not match `pyproject.toml` is refused by `uv sync --locked` at deploy time. A version
change is a change in numerics: run `--compare` on the staged release before enabling it.

## Day-2 commands

On the inference host:

```
systemctl status coffee-cv-web nginx
readlink /opt/coffee-cv/current /opt/coffee-cv/previous   # which releases
journalctl -u coffee-cv-web -f         # startup: commit, model_sha, OOD guard
tail -f /var/log/coffee-cv/app.jsonl   # per-request logs: verdicts, timing, errors
                                        # (never image bytes or filenames --
                                        # see docs/logging_plan.html)
tail -f /var/log/nginx/coffee-cv.access  # nginx side of the same requests --
                                        # no client IP, joined to the line
                                        # above by request id; its last field,
                                        # the bytes nginx got from the app,
                                        # shows an /analyze answer cut short
ss -ltn | grep 12346                    # nginx on the tailnet address only
```

On the front (the proxy):

```
systemctl status coffee-cv-front
journalctl -u coffee-cv-front -f       # the check service: one line per request, no IP
tail -f /var/log/nginx/coffee-cv-front.access    # no IP; nginx's request id, then the app's
                                                 # ("-" for a request refused at the front);
                                                 # last, the calling page's Origin (OPS-19 D3)
tail -f /var/log/nginx/coffee-cv-front.sessions  # the visitor's IP, one line per TLS session
                                                 # (the stream block's; kept 30 days, OPS-15 D20)
```

A streamed `/analyze` answer has its status before it is done, so its `app.jsonl` line also says how it ended
(ticket OPS-7 D19): `outcome` is `complete`, `error` (an error part) or `client_gone` (the client left before the
end), and `parts_sent` counts the lines handed to the server.

## What lives outside git, and where

Three things are deliberately never committed, and never named in this file
either -- see `webapp/deploy/coffee-cv-front.nginx.conf.template`,
`setup_front.sh` and `scripts/deploy_webapp.sh` for exactly how each is handled:

- **The domain.** Passed as `DOMAIN=...` to `scripts/deploy_webapp.sh`; the
  nginx config template uses a placeholder everywhere the domain would
  appear. Worth noting privately (e.g. your own shell history) since it's not
  written anywhere in this repo.
- **The DNS record.** Point your domain's A record at the proxy, the front,
  yourself -- not something a script here can do for you. The proxy's
  `stream {}` block hands the domain's TLS sessions to the front's site on
  `127.0.0.1:12347` (the `server-coffee` upstream, set by hand: plan P6).
- **The TLS cert/key.** On the front only: the inference host has none (it
  speaks plain HTTP on the tailnet). They are the proxy's own files, which its
  other sites use too; the deploy takes their paths as `TLS_CERT` and `TLS_KEY`
  and writes them only into the rendered site. With Porkbun's files,
  `TLS_CERT` is `domain.cert.pem` (Porkbun's bundle already concatenates the
  leaf + intermediate chain into this one file, confirmed by counting `BEGIN
  CERTIFICATE` blocks -- no separate fullchain build step needed) and
  `TLS_KEY` is `private.key.pem`. `private.key.pem` needs `600` (owner
  read/write only -- Porkbun's default download permissions are world-writable
  `666`); `domain.cert.pem` and `public.key.pem` (not used by nginx) are fine
  at `644`. Also worth a one-time check with
  `openssl x509 -noout -text -in domain.cert.pem | grep -A1 'Subject
  Alternative Name'` that the cert's SAN list actually covers whatever
  subdomain you're deploying to (a wildcard `*.yourdomain` or the exact
  hostname) -- Porkbun issues per-domain, not automatically per-subdomain.

  `nginx.service` runs `nginx -t` on start, and a missing cert fails that
  check for the whole config -- the proxy's other sites included, so the
  deploy's `nginx -t` must pass on the front before it reloads.

## For agents

The service is open to LLM agents acting for a person, through the same `POST /analyze` the page uses: no key, no
separate endpoint. Three static files in the release describe it, and the page links all three from its `<head>`:

- `/llms.txt` -- the guide an agent reads first: what the service does, a `curl` call, both answer parts, the
  verdicts, and the one-at-a-time limit (wait and retry on a 503).
- `/openapi.json` -- the same contract, machine-readable (OpenAPI 3.1; `rel="service-desc"`, RFC 8631).
- `/.well-known/ard.json` -- an [Agentic Resource Discovery](https://github.com/ards-project/ard-spec) manifest
  (v0.91, a proposal) listing the two above for agent registries.

nginx serves them with `Access-Control-Allow-Origin: *`, and `robots.txt` allows every path, `/analyze` included,
since agents that honour it would otherwise refuse to call the API. `/analyze` answers every origin too (ADR 0035,
ticket OPS-19), so a cross-origin caller, a script on another site's page, reads the answer and its errors: the front's
nginx adds the CORS headers to every answer, its own refusals included, and answers the preflight before the
one-at-a-time gate. `llms.txt` ("From a web page") gives that site a `fetch` example and asks it to tell its visitors
where their photo goes. **A change to what `/analyze` takes or returns
updates `openapi.json` and `llms.txt` in the same change**; `tests/test_agent_files.py` checks the form fields and
the routes against `app.py`, and that every URL in the three files is one the release serves.

## Design choices worth knowing before changing anything

- **One gunicorn worker, sync class, deliberately.** The worker holds the
  models on the host's one GPU and takes about 3.5 GB of host memory (OPS-10);
  a second worker would share the same GPU and double that memory, not add
  real throughput. Serializing behind one worker, backed by nginx's
  `limit_conn` (below), is the better trade for a low-traffic endpoint.
- **Rate limiting is keyed by connection, not IP** (NAT can put many real
  users behind one IP) -- `limit_conn` caps total *concurrent* `/analyze`
  requests at exactly 1, matching the single gunicorn worker. The front holds
  the gate, so a second upload never starts crossing the slow link, and answers
  `503 {"error": "the classifier is busy with another photo"}`; the inference
  host keeps its own as a backstop. A request takes
  about 8.5 s server-side on the GPU (8.4-8.9 s, its preview at 2.1 s; OPS-7, 2026-10-18), so this alone already enforces well under "1 request/sec"; a
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
  unit pending a real test on the host (some ML runtimes JIT-compile at
  runtime, LiteRT's GPU accelerator likely among them) -- try it there; add it if
  the GPU worker still starts and proves the GPU, drop the idea if it doesn't.
- **Format support is pinned in `coffeecv/dataset.py`, not here.** JPG/PNG/
  WEBP need no plugin; HEIF/HEIC, AVIF, and JPEG XL are registered with
  Pillow right next to each other in `load_rgb_image`'s home module, so the
  CLI and this web service can never disagree on what formats they accept.
  Camera RAW (DNG/CR2/CR3/NEF/ARW/RAF/ORF/RW2/PEF/SRW) is different --
  Pillow can't open these at all, so `load_rgb_image` branches explicitly
  and routes them through `rawpy` (LibRaw) instead, normalizing rawpy's own
  exception type to `OSError` so it joins the same "can't read this photo"
  handling every other format's decode failure already gets.

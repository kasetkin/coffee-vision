"""The release manifest: every repo file the web service opens, for one model, and nothing else.

A release under /opt/coffee-cv/releases/<id> holds exactly these files, the generated ones and the
built ones (docs/ops1_release_isolation_plan.html §4.2). tests/test_release_manifest.py traces the
running app with an audit hook and fails if it opens a repo file outside this list; the deploy checks
the other direction, that a staged release holds nothing beyond it. Stdlib only: the deploy runs it
before any environment exists.

    python webapp/deploy/release_manifest.py --model allrigs_dino3b16_segxl0v3_country_s42 [--ref <sha>]

prints the manifest as JSON, reading the model card and pointers from the ref (default: the working
tree).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parents[2]

APP = [
    "webapp/__init__.py",
    "webapp/app.py",
    "webapp/static/index.html",
    "webapp/pyproject.toml",
    "webapp/uv.lock",
]
# What webapp/app.py imports, found with an import trace and checked by tests/test_release_manifest.py. Since
# ticket ML-7 (plan P10) the models run on LiteRT: no torch-side module, no vendored EfficientViT and no pretrained
# weights; the converted model's directory instead.
LIBRARY = [f"coffeecv/{m}.py" for m in (
    "__init__", "bean_scale", "class_list", "config", "crop_tray", "geometry", "infer", "litert", "model_io",
    "photo_io", "sam_loader", "segment_beans")]
# Beside models/<name>.pt, git-tracked. The first two are required; the rest ship when the ref has them
# (a model without a probe runs the centroid guard -- the probe file's presence is the switch).
MODEL_REQUIRED = (".pt.dvc", ".json")
MODEL_OPTIONAL = (".classes.txt", ".ood_reference.json", ".ood_probe.json")
# The converted model (coffeecv.litert): the dvc.yaml stage litert_export's output, every file of it by md5.
LITERT_STAGE = "litert_export@{model}"
# The export's own records, which serving never opens: they stay in DVC, not in a release.
LITERT_RECORDS = {"convert.json", "check.json"}
FIXTURES = "webapp/deploy/fixtures.txt"
# Written by the deploy into the release, not taken from git.
GENERATED = ["webapp/BUILD_INFO.json", "release.env"]
# Built on the VM: the venv, its marker, and compileall's bytecode beside the release's own modules.
BUILT_TOP = [".venv", ".complete"]
# LiteRT's GPU accelerator directory inside the release's venv: the wheel's own, which holds libLiteRt.so and one GPU
# library, the WebGPU one (ML-7 P2). Read by webapp/app.py from release.env, relative to the release.
LITERT_RUNTIME = ".venv/lib/python3.12/site-packages/ai_edge_litert"
# Vulkan's driver on a GPU host: yappy's Intel iGPU through Mesa's ANV (ML-7 P2), as every P2-P8 GPU run set it.
VK_ICD = "/usr/share/vulkan/icd.d/intel_icd.x86_64.json"


def dvc_md5(pointer_text: str) -> str:
    """The md5 a `<file>.dvc` pointer records -- the same parse webapp/app.py logs as model_dvc_md5."""
    for line in pointer_text.splitlines():
        if line.strip().startswith(("- md5:", "md5:")):
            return line.split("md5:", 1)[1].strip()
    raise ValueError("no md5 in .dvc pointer")


def _dvc_cache(md5: str) -> Path:
    return REPO / ".dvc" / "cache" / "files" / "md5" / md5[:2] / md5[2:]


def litert_files(model: str, lock_text: str, read_object: Callable[[str], str]) -> dict[str, str]:
    """{path: md5} for every file of the model's converted directory, from dvc.lock's record of the litert_export
    stage and the directory's listing (`<md5>.dir`, from the DVC cache by default)."""
    stage = LITERT_STAGE.format(model=model)
    m = re.search(rf"^  {re.escape(stage)}:\n((?:    .*\n?)*)", lock_text, re.MULTILINE)
    if m is None or "outs:" not in m.group(1):
        raise ValueError(f"dvc.lock has no {stage} stage: run `dvc repro -s {stage}` (ML-7 D6)")
    outs = m.group(1).split("outs:", 1)[1]
    path = md5 = None
    for line in outs.splitlines():
        t = line.strip()
        if t.startswith("- path:"):
            path = t.split(":", 1)[1].strip()
        elif t.startswith("md5:") and path == f"models/{model}.litert":
            md5 = t.split(":", 1)[1].strip()
            break
    if md5 is None or not md5.endswith(".dir"):
        raise ValueError(f"dvc.lock's {stage} has no directory output models/{model}.litert")
    return {f"models/{model}.litert/{e['relpath']}": e["md5"] for e in json.loads(read_object(md5))
            if e["relpath"] not in LITERT_RECORDS}


def manifest(model: str, read_text: Callable[[str], str], exists: Callable[[str], bool],
             read_object: Callable[[str], str] = lambda md5: _dvc_cache(md5).read_text()) -> dict:
    """{git: [paths], dvc: {path: md5}, pinned: {path: sha256}, env: {name: value}, generated: [paths]} for `model`:
    a frozen DINOv3 model with the segmenter (crop_method "segment"), served on LiteRT. `env` is what the deploy
    writes into release.env beside COFFEE_CV_CHECKPOINT. `read_object` reads a DVC object by md5 (the converted
    directory's listing and its manifest.json, which say whether a graph runs on the GPU)."""
    stem = f"models/{model}"
    git = APP + LIBRARY
    for suffix in MODEL_REQUIRED:
        if not exists(stem + suffix):
            raise FileNotFoundError(f"{stem}{suffix} is not in the ref")
        git.append(stem + suffix)
    git += [stem + s for s in MODEL_OPTIONAL if exists(stem + s)]
    card = json.loads(read_text(f"{stem}.json"))
    training = card.get("training_config", {})
    if not training.get("model_name", "").startswith("dinov3") or training.get("crop_method") != "segment":
        raise ValueError(f"{model} is {training.get('model_name')!r} with crop_method {training.get('crop_method')!r}, "
                         f"which has no release shape: the service runs a frozen DINOv3 model with the segmenter, on "
                         f"LiteRT (ML-7)")
    litert = litert_files(model, read_text("dvc.lock"), read_object)
    dvc = {f"{stem}.pt": dvc_md5(read_text(f"{stem}.pt.dvc")), **litert}
    converted = json.loads(read_object(litert[f"{stem}.litert/manifest.json"]))
    # The boot's warm-up photo (ADR 0021): the deploy's smoke fixture, pinned by the sha256 fixtures.txt gives it.
    smoke = [ln.split() for ln in read_text(FIXTURES).splitlines() if ln.startswith("smoke ")]
    if len(smoke) != 1:
        raise ValueError(f"{FIXTURES} must name exactly one smoke photo")
    _, sha, photo = smoke[0]
    env = {"COFFEE_CV_WARMUP_PHOTO": photo}
    if any(g["accelerator"] == "gpu" for g in converted["graphs"].values()):
        # LiteRT's WebGPU library from the release venv, and Vulkan pointed at Intel's driver (the host's, ADR 0019).
        env |= {"COFFEE_CV_LITERT_RUNTIME": LITERT_RUNTIME, "VK_ICD_FILENAMES": VK_ICD}
    return {"model": model, "model_name": training["model_name"], "git": git, "dvc": dvc, "pinned": {photo: sha},
            "env": env, "generated": list(GENERATED)}


def all_files(m: dict) -> set[str]:
    """Every file path a staged release may hold outside .venv/ and __pycache__/."""
    return set(m["git"]) | set(m["dvc"]) | set(m["pinned"]) | set(m["generated"])


def unexpected(m: dict, staged: list[str]) -> list[str]:
    """Paths in a staged release (relative, files only) that the manifest does not account for."""
    allowed = all_files(m)
    bad = []
    for p in staged:
        top = p.split("/", 1)[0]
        if top in BUILT_TOP or p in allowed:
            continue
        # compileall's bytecode for a shipped module: <dir>/__pycache__/<mod>.cpython-312.pyc
        parts = p.split("/")
        if len(parts) >= 3 and parts[-2] == "__pycache__" and parts[-1].endswith(".pyc"):
            if "/".join(parts[:-2] + [parts[-1].split(".", 1)[0] + ".py"]) in allowed:
                continue
        bad.append(p)
    return sorted(bad)


def _readers(ref: str | None):
    if ref is None:
        return (lambda p: (REPO / p).read_text()), (lambda p: (REPO / p).is_file())

    def read_text(p: str) -> str:
        return subprocess.run(["git", "show", f"{ref}:{p}"], cwd=REPO, capture_output=True, text=True,
                              check=True).stdout

    def exists(p: str) -> bool:
        return subprocess.run(["git", "cat-file", "-e", f"{ref}:{p}"], cwd=REPO,
                              capture_output=True).returncode == 0
    return read_text, exists


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="model name, e.g. allrigs_dino3b16_seg_country_s123")
    ap.add_argument("--ref", help="git ref to read the card and pointers from (default: working tree)")
    ap.add_argument("--check-staged", metavar="LISTING",
                    help="file listing of a staged release (one relative path per line, '-' = stdin); "
                         "exit 1 listing any path the manifest does not account for, or any it lacks")
    args = ap.parse_args()
    m = manifest(args.model, *_readers(args.ref))
    if args.check_staged is None:
        print(json.dumps(m, indent=1))
        return
    import sys
    text = sys.stdin.read() if args.check_staged == "-" else Path(args.check_staged).read_text()
    staged = [line for line in text.splitlines() if line]
    extra = unexpected(m, staged)
    missing = sorted(all_files(m) - set(staged))
    for p in extra:
        print(f"UNEXPECTED {p}")
    for p in missing:
        print(f"MISSING    {p}")
    raise SystemExit(1 if extra or missing else 0)


if __name__ == "__main__":
    main()

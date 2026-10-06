"""The release manifest: every repo file the web service opens, for one model, and nothing else.

A release under /opt/coffee-cv/releases/<id> holds exactly these files, the generated ones and the
built ones (docs/ops1_release_isolation_plan.html §4.2). tests/test_release_manifest.py traces the
running app with an audit hook and fails if it opens a repo file outside this list; the deploy checks
the other direction, that a staged release holds nothing beyond it. Stdlib only: the deploy runs it
before any environment exists.

    python webapp/deploy/release_manifest.py --model allrigs_dino3b16_s123 [--ref <sha>]

prints the manifest as JSON, reading the model card and pointers from the ref (default: the working
tree).
"""
from __future__ import annotations

import argparse
import json
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
LIBRARY = [f"coffeecv/{m}.py" for m in (
    "__init__", "backbones", "bean_scale", "class_list", "config", "crop_tray", "dataset", "dino_classifier",
    "geometry", "infer", "model", "transforms")]
# Ticket ML-2: a segmenter model (card crop_method "segment") also ships the segmenter's loader and the
# vendored EfficientViT files that building L0 imports, found with an import trace (plan §9) and checked
# by tests/test_release_manifest.py, plus the vendored copy's license and its patch notes.
SEG_LIBRARY = ["coffeecv/sam_loader.py", "coffeecv/segment_beans.py",
               "third_party/efficientvit/LICENSE", "third_party/efficientvit/PATCHES.md"]
SEG_VENDORED = [f"third_party/efficientvit/efficientvit/{m}" for m in (
    "__init__.py",
    "apps/__init__.py",
    "apps/data_provider/__init__.py",
    "apps/data_provider/augment/__init__.py",
    "apps/data_provider/augment/bbox.py",
    "apps/data_provider/augment/color_aug.py",
    "apps/data_provider/base.py",
    "apps/data_provider/random_resolution/__init__.py",
    "apps/data_provider/random_resolution/controller.py",
    "apps/trainer/__init__.py",
    "apps/trainer/base.py",
    "apps/trainer/run_config.py",
    "apps/utils/__init__.py",
    "apps/utils/dist.py",
    "apps/utils/ema.py",
    "apps/utils/export.py",
    "apps/utils/image.py",
    "apps/utils/init.py",
    "apps/utils/lr.py",
    "apps/utils/metric.py",
    "apps/utils/misc.py",
    "apps/utils/opt.py",
    "models/__init__.py",
    "models/efficientvit/__init__.py",
    "models/efficientvit/backbone.py",
    "models/efficientvit/cls.py",
    "models/efficientvit/sam.py",
    "models/efficientvit/seg.py",
    "models/nn/__init__.py",
    "models/nn/act.py",
    "models/nn/drop.py",
    "models/nn/norm.py",
    "models/nn/ops.py",
    "models/utils/__init__.py",
    "models/utils/list.py",
    "models/utils/network.py",
    "models/utils/random.py",
    "sam_model_zoo.py",
)]
# Beside models/<name>.pt, git-tracked. The first two are required; the rest ship when the ref has them
# (a model without a probe runs the centroid guard -- the probe file's presence is the switch).
MODEL_REQUIRED = (".pt.dvc", ".json")
MODEL_OPTIONAL = (".classes.txt", ".ood_reference.json", ".ood_probe.json")
PRETRAINED_MANIFEST = "models_pretrained/manifest.json"
# Written by the deploy into the release, not taken from git.
GENERATED = ["webapp/BUILD_INFO.json", "release.env"]
# Built on the VM: the venv, its marker, and compileall's bytecode beside the release's own modules.
BUILT_TOP = [".venv", ".complete"]


def dvc_md5(pointer_text: str) -> str:
    """The md5 a `<file>.dvc` pointer records -- the same parse webapp/app.py logs as model_dvc_md5."""
    for line in pointer_text.splitlines():
        if line.strip().startswith(("- md5:", "md5:")):
            return line.split("md5:", 1)[1].strip()
    raise ValueError("no md5 in .dvc pointer")


def manifest(model: str, read_text: Callable[[str], str], exists: Callable[[str], bool]) -> dict:
    """{git: [paths], dvc: {path: md5}, pretrained: {path: sha256}, generated: [paths]} for `model`."""
    stem = f"models/{model}"
    git = APP + LIBRARY
    for suffix in MODEL_REQUIRED:
        if not exists(stem + suffix):
            raise FileNotFoundError(f"{stem}{suffix} is not in the ref")
        git.append(stem + suffix)
    git += [stem + s for s in MODEL_OPTIONAL if exists(stem + s)]

    dvc = {f"{stem}.pt": dvc_md5(read_text(f"{stem}.pt.dvc"))}

    # Files pinned by sha256 rather than by a .dvc pointer: pretrained weights, and a segmenter's decoder.
    pretrained = {}
    card = json.loads(read_text(f"{stem}.json"))
    training = card.get("training_config", {})
    model_name = training.get("model_name", "")
    crop_method = training.get("crop_method", "tray_heuristic")
    if model_name.startswith("dinov3") or crop_method == "segment":
        entries = {e["path"]: e for e in json.loads(read_text(PRETRAINED_MANIFEST))}
    if model_name.startswith("dinov3"):
        # A frozen model's .pt holds only the head; the backbone comes from models_pretrained/, checked
        # against the manifest's sha256 (coffeecv.backbones.verify_weights) at every start.
        weights = card["dino"]["weights"]
        pretrained[f"models_pretrained/{weights}"] = entries[weights]["sha256"]
    if crop_method == "segment":
        # Ticket ML-2: the segmenter the model was trained with, as its card pins it (seg_* fields).
        weights, decoder = training["seg_weights"], training["seg_decoder"]
        if entries[weights]["sha256"] != training["seg_weights_sha256"]:
            raise ValueError(f"{model}'s card pins {weights} at sha256 {training['seg_weights_sha256'][:16]}..., "
                             f"but {PRETRAINED_MANIFEST} has {entries[weights]['sha256'][:16]}...")
        pretrained[f"models_pretrained/{weights}"] = training["seg_weights_sha256"]
        if decoder:
            pretrained[decoder] = training["seg_decoder_sha256"]
        git += SEG_LIBRARY + SEG_VENDORED
    elif crop_method != "tray_heuristic":
        raise ValueError(f"{model} uses crop_method {crop_method!r}, which has no release shape")
    if pretrained:
        git.append(PRETRAINED_MANIFEST)
    return {"model": model, "model_name": model_name, "git": git, "dvc": dvc,
            "pretrained": pretrained, "generated": list(GENERATED)}


def all_files(m: dict) -> set[str]:
    """Every file path a staged release may hold outside .venv/ and __pycache__/."""
    return set(m["git"]) | set(m["dvc"]) | set(m["pretrained"]) | set(m["generated"])


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
    ap.add_argument("--model", required=True, help="model name, e.g. allrigs_dino3b16_s123")
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

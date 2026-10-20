"""Ticket ML-8 P2: the one builder of every albumentations pipeline, the segmenter's fine-tune (seg_ft.augment,
ADR 0018) and the classifier's head fit (head_augment, ADR 0025) alike.

A pipeline is a params.yaml list of {transform, p, args}: `transform` names an albumentations class, `args` its
arguments. Lists become tuples; a `transforms` argument (OneOf and the like) nests entries. Mappings stay mappings
(Downscale's interpolation_pair).

Built strictly. albumentations 2.0.8, the last MIT release and pinned (ADR 0018), only warns on an argument it does
not know and runs with its default: ColorJitter(brightness_range=(0.5, 0.6)), the spelling of the newer online
guide, warns "Argument(s) 'brightness_range' are not valid" and jitters brightness by the default (0.8, 1.2). Here
any warning while a pipeline is built is an error naming the transform, so every argument means what it says.

Shared groups (params.yaml `augment`, ADR 0027): an entry {use: <group>} stands for the group's transforms, in
its place. `expand` replaces them before anything is built or recorded, and RunConfig.from_params_yaml calls it on
both fits' pipelines, so a run's config and a model's card hold the transforms that ran, never a reference.

A caller's `check(cls, args)` adds its own rule, applied to every entry, nested ones included: it returns what is
wrong, or None. The segmenter's is that a transform which fills a mask fills it with the label's ignore value.

The library is imported with NO_ALBUMENTATIONS_UPDATE=1: 2.0.8 asks PyPI for a newer version on import otherwise.
"""
from __future__ import annotations

import inspect
import os
import warnings
from typing import Callable

ENTRY_KEYS = {"transform", "p", "args"}


def albumentations():
    """albumentations, imported without its update check."""
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    import albumentations
    return albumentations


def version() -> str:
    return albumentations().__version__


def expand(spec, groups: dict, *, where: str) -> list[dict]:
    """`spec` with each {use: <group>} entry replaced by `groups`' list of that name, in its place. A group is a
    plain list of transforms: it cannot use another."""
    out = []
    for t in spec or ():
        if not (isinstance(t, dict) and "use" in t):
            out.append(t)
            continue
        name = t["use"]
        if set(t) != {"use"}:
            raise ValueError(f"{where}: {t!r}: a shared group is used as {{use: <group>}} alone")
        if name not in groups:
            raise ValueError(f"{where}: no shared group {name!r} in params.yaml augment "
                             f"(it has {', '.join(sorted(groups)) or 'none'})")
        group = groups[name]
        if not isinstance(group, list) or any(isinstance(g, dict) and "use" in g for g in group):
            raise ValueError(f"augment.{name}: a shared group is a list of transforms, and uses no other group")
        out.extend(group)
    return out


def _transform(A, t, where: str, check: Callable[[type, dict], str | None] | None):
    if not isinstance(t, dict) or set(t) - ENTRY_KEYS or "transform" not in t or "p" not in t:
        raise ValueError(f"{where}: {t!r} is not {{transform, p, args}}")
    name = str(t["transform"])
    cls = getattr(A, name, None)
    if not (inspect.isclass(cls) and issubclass(cls, A.BasicTransform | A.BaseCompose)):
        raise ValueError(f"{where}: {name!r} is not an albumentations transform")
    args = dict(t.get("args") or {})
    problem = check(cls, args) if check else None
    if problem:
        raise ValueError(f"{where}: {name} {problem}")
    tup = lambda v: tuple(tup(x) for x in v) if isinstance(v, list) else v  # noqa: E731
    args = {k: [_transform(A, x, where, check) for x in v] if k == "transforms" else tup(v) for k, v in args.items()}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            return cls(**args, p=float(t["p"]))
    except Exception as e:
        raise ValueError(f"{where}: {name}: {e}") from e


def build(spec, *, where: str, seed: int, check: Callable[[type, dict], str | None] | None = None):
    """`spec` as an albumentations Compose seeded by `seed`, so one seed repeats its stream. `where` names the
    spec in errors (its params.yaml key)."""
    A = albumentations()
    return A.Compose([_transform(A, t, where, check) for t in spec or ()], seed=seed, strict=True)

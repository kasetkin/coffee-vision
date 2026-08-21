"""Screen-only test-time augmentation variants beyond the adopted dihedral TTA in
`infer.py`. Promote a variant into `infer.py` proper only if it earns adoption on
the same paired, sign-consistent-across-folds bar as everything else in this
project (see EXPERIMENTS_LOG.md and the coffeecv-project memory).

Both variants here are weaker arguments than dihedral TTA's. Dihedral TTA
averages over the *exact* symmetry group training augments with
(`RandomRightAngleRotation` + both flips), so the averaged predictor is provably
invariant where the base model was only approximately so. Neither variant below
has that guarantee:

- Photometric TTA marginalizes over a *plausible* nuisance (the model was trained
  to tolerate color-jitter perturbation, so averaging over it at test time is a
  reasonable-but-unproven idea), not an exact invariance.
- Multi-scale TTA draws the *same total patch budget*, at the *same patch
  centres*, as a standard draw -- just sized from a few fixed scales instead of
  one continuous log-uniform distribution (see
  `coffeecv/xrig_eval.py`'s `run_photowise` for the position-matched sampling;
  this file no longer implements that arm directly). A plausible-but-unproven
  idea that discretizing might cover the trained range more evenly than a small
  random sample does by chance, not a guaranteed win. (An earlier version of
  this measurement compared unequal patch budgets at independent, unmatched
  positions and was corrected -- see [[project-phase16-screens]].)

Each needs its own screen. A win from one must not be credited to the other.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from coffeecv.transforms import IMAGENET_MEAN, IMAGENET_STD


def photometric_tta_probs(
    model,
    patches: list[Image.Image],
    resize: int,
    jitter_strength: float,
    n_draws: int,
    seed: int,
    batch_size: int = 32,
) -> np.ndarray:
    """Average softmax over the untransformed pass plus `n_draws` *fixed, seeded*
    ColorJitter draws, at the same strength training used
    (`color_jitter_strength`). Operates on PIL patches, not already-normalized
    tensors: unlike dihedral TTA's rotate/flip (position permutations that
    commute with normalization), color jitter changes pixel values and must be
    applied before `Normalize`, exactly where training applies it.

    Deterministic given `seed` -- re-running reproduces the same average, unlike
    ColorJitter's normal per-call randomness during training.
    """
    plain = T.Compose([
        T.Resize((resize, resize), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    gen = torch.Generator().manual_seed(seed)

    def run(transform) -> np.ndarray:
        tensors = torch.stack([transform(p) for p in patches])
        out = []
        with torch.no_grad():
            for i in range(0, len(tensors), batch_size):
                out.append(F.softmax(model(tensors[i:i + batch_size]), dim=1).numpy())
        return np.concatenate(out, axis=0)

    acc = run(plain)
    n = 1
    for _ in range(n_draws):
        # One fixed jitter instance per draw, applied identically to every patch
        # in that draw -- draws vary, not per-patch randomness within a draw,
        # so the average is over `n_draws` distinct "what if the lighting had
        # been like this" hypotheses rather than n_draws independent per-patch
        # noise samples that would mostly cancel out individually anyway.
        jitter = T.ColorJitter(
            brightness=jitter_strength, contrast=jitter_strength,
            saturation=jitter_strength, hue=min(jitter_strength * 0.1, 0.5),
        )
        # Freeze this draw's actual jitter factors from `gen` so the draw is
        # reproducible, then apply that fixed jitter to every patch.
        params = T.ColorJitter.get_params(
            jitter.brightness, jitter.contrast, jitter.saturation, jitter.hue,
        )
        fn_idx, b, c, s, h = params

        def apply_jitter(img: Image.Image, fn_idx=fn_idx, b=b, c=c, s=s, h=h) -> Image.Image:
            for fn_id in fn_idx:
                if fn_id == 0 and b is not None:
                    img = T.functional.adjust_brightness(img, b)
                elif fn_id == 1 and c is not None:
                    img = T.functional.adjust_contrast(img, c)
                elif fn_id == 2 and s is not None:
                    img = T.functional.adjust_saturation(img, s)
                elif fn_id == 3 and h is not None:
                    img = T.functional.adjust_hue(img, h)
            return img

        draw_transform = T.Compose([
            T.Lambda(apply_jitter),
            T.Resize((resize, resize), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        acc = acc + run(draw_transform)
        n += 1
        # Advance `gen` even though it isn't consumed above, so a future draw
        # that does use torch randomness stays reproducible in sequence; kept
        # for parity with how seeded draws are threaded elsewhere in this repo.
        torch.randint(0, 2**31 - 1, (1,), generator=gen)
    return acc / n

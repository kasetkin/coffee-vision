"""LR schedulers for train_baseline.py, behind one small interface.

    step(epoch, val_f1) -> event | None   call once per epoch, after validation
    lrs()               -> [lr per param group]   head first, backbone second if present
    should_stop()       -> bool           only meaningful when `owns_stopping`
    owns_stopping       -> bool           False: the loop's raw-F1 patience rule stays in charge

Two schedulers:

- ``cosine``: the project's original schedule, ``CosineAnnealingLR(T_max=epochs, eta_min)``,
  untouched. Stopping stays with the patience rule in train_baseline's loop.
- ``plateau``: torch's ``ReduceLROnPlateau`` fed a *smoothed* val macro-F1, plus a
  deterministic stop ``floor_epochs`` epochs after the LR floor is reached.

Why the plateau variant looks the way it does (docs/lr_scheduler_plan.md has the evidence):

- The monitor is val macro-F1, not val loss: on the recorded cross-rig curves, checkpointing on
  min val loss lost 0.006 cross-rig F1. It is smoothed (trailing mean) because per-epoch F1
  jitter is sigma~0.005, and torch tracks a raw running max, so one lucky spike would make
  every later epoch look like a plateau. The threshold is absolute for the same reason: torch's
  default (relative 1e-4) sits ~50x below the jitter.
- The floor is ``min(eta_min, base_lr)`` *per group*: the same absolute floor CosineAnnealingLR
  applies to every group. With ``eta_min == backbone_lr`` (the adopted recipe) the backbone
  therefore stays flat at its base LR, exactly as it does under the cosine schedule -- which
  is deliberate, so a plateau-vs-cosine comparison changes the head's schedule and the stop
  rule and nothing else. It is also why a scalar floor is not a bug here.
- Stopping is owned by the scheduler and is a fixed budget at the floor, not another patience
  counter on a noisy metric: after the step that lands the LR on its floor, train exactly
  ``floor_epochs`` more epochs, then stop. ``epochs`` is then only a cap.
"""
from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

if TYPE_CHECKING:
    from coffeecv.config import RunConfig

SCHEDULERS = ("cosine", "plateau")

# Group LRs land on their floor by float arithmetic (max(lr*factor, min_lr) returns min_lr
# exactly), but compare with a hair of slack so a floor that arrives via any other path
# still counts.
_FLOOR_RTOL = 1e-9


class CosineLR:
    """The original schedule, verbatim. The raw-F1 patience stop stays in the training loop."""

    owns_stopping = False

    def __init__(self, optimizer: Optimizer, epochs: int, eta_min: float) -> None:
        self._opt = optimizer
        self._sched = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)
        self._epochs, self._eta_min = epochs, eta_min
        self.smoothed: float | None = None

    def lrs(self) -> list[float]:
        return [g["lr"] for g in self._opt.param_groups]

    def step(self, epoch: int, val_f1: float) -> str | None:
        self._sched.step()
        return None

    def should_stop(self) -> bool:
        return False

    def describe(self) -> str:
        return (f"cosine: CosineAnnealingLR(T_max={self._epochs}, eta_min={self._eta_min:g}); "
                "stopping = raw val macro-F1 patience rule")


class PlateauLR:
    """ReduceLROnPlateau on a trailing mean of val macro-F1, stopping `floor_epochs` after the floor."""

    owns_stopping = True

    def __init__(
        self, optimizer: Optimizer, *, eta_min: float, smooth: int, threshold: float,
        patience: int, cooldown: int, factor: float, floor_epochs: int,
    ) -> None:
        if eta_min <= 0:
            raise ValueError(
                f"scheduler=plateau needs a positive eta_min (got {eta_min}): it is the LR floor, and "
                "reaching the floor is what starts the stopping countdown."
            )
        if smooth < 1 or floor_epochs < 1 or patience < 0 or cooldown < 0:
            raise ValueError(
                f"bad plateau settings: smooth={smooth} floor_epochs={floor_epochs} "
                f"patience={patience} cooldown={cooldown}"
            )
        if not 0.0 < factor < 1.0:
            raise ValueError(f"plateau factor must be in (0, 1), got {factor}")
        base = [g["lr"] for g in optimizer.param_groups]
        if all(b <= eta_min for b in base):
            raise ValueError(
                f"no param group has an LR above the floor (base LRs {base}, eta_min {eta_min}); "
                "there is nothing for the plateau scheduler to reduce."
            )
        self._opt = optimizer
        # One floor per *actual* group, so a single-group optimizer (freeze_mode=full) builds too.
        self.min_lrs = [min(eta_min, b) for b in base]
        self._sched = ReduceLROnPlateau(
            optimizer, mode="max", factor=factor, patience=patience, threshold=threshold,
            threshold_mode="abs", cooldown=cooldown, min_lr=self.min_lrs,
        )
        self._buf: deque[float] = deque(maxlen=smooth)
        self._smooth, self._threshold, self._patience = smooth, threshold, patience
        self._cooldown, self._factor, self._floor_epochs = cooldown, factor, floor_epochs
        self.smoothed: float | None = None
        self.floor_epoch: int | None = None   # epoch whose step first landed every group on its floor
        self._epoch = 0

    def lrs(self) -> list[float]:
        return [g["lr"] for g in self._opt.param_groups]

    def _at_floor(self) -> bool:
        return all(lr <= m * (1.0 + _FLOOR_RTOL) for lr, m in zip(self.lrs(), self.min_lrs))

    def step(self, epoch: int, val_f1: float) -> str | None:
        """Feed this epoch's raw val macro-F1. Returns "drop", "floor" (the drop that reached the
        floor) or None. The event describes the change that takes effect from the *next* epoch."""
        self._epoch = epoch
        self._buf.append(float(val_f1))
        self.smoothed = sum(self._buf) / len(self._buf)
        before = self.lrs()
        self._sched.step(self.smoothed)
        event = "drop" if self.lrs() != before else None
        if self.floor_epoch is None and self._at_floor():
            self.floor_epoch = epoch
            event = "floor"
        return event

    def should_stop(self) -> bool:
        return self.floor_epoch is not None and self._epoch - self.floor_epoch >= self._floor_epochs

    def describe(self) -> str:
        return (
            f"plateau: ReduceLROnPlateau(mode=max) on trailing-{self._smooth} mean val macro-F1, "
            f"abs threshold {self._threshold:g}, patience {self._patience}, cooldown {self._cooldown}, "
            f"factor {self._factor:g}; floors per group {[f'{m:g}' for m in self.min_lrs]}; "
            f"stop {self._floor_epochs} epochs after the floor is reached (epochs is only a cap)"
        )


def build_scheduler(cfg: "RunConfig", optimizer: Optimizer):
    if cfg.scheduler == "cosine":
        return CosineLR(optimizer, cfg.epochs, cfg.eta_min)
    if cfg.scheduler == "plateau":
        return PlateauLR(
            optimizer, eta_min=cfg.eta_min, smooth=cfg.plateau_smooth, threshold=cfg.plateau_threshold,
            patience=cfg.plateau_patience, cooldown=cfg.plateau_cooldown, factor=cfg.plateau_factor,
            floor_epochs=cfg.floor_epochs,
        )
    raise ValueError(f"Unknown scheduler: {cfg.scheduler!r} (want one of {SCHEDULERS})")

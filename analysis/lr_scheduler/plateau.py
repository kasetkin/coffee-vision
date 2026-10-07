import numpy as np
def smooth(x, w):
    """trailing mean over the last w epochs (w=1 -> raw)"""
    x = np.asarray(x, float)
    if w <= 1:
        return x
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = x[max(0, i - w + 1): i + 1].mean()
    return out
def first_trigger(metric, mode, patience, threshold, threshold_mode, cooldown=0):
    """Replicates torch ReduceLROnPlateau's bookkeeping; returns the 1-based epoch at which the
    FIRST LR reduction would fire (num_bad_epochs > patience), or None."""
    best = np.inf if mode == "min" else -np.inf
    bad = 0
    for i, m in enumerate(metric):
        if mode == "min":
            better = m < (best * (1 - threshold) if threshold_mode == "rel" else best - threshold)
        else:
            better = m > (best * (1 + threshold) if threshold_mode == "rel" else best + threshold)
        if better:
            best, bad = m, 0
        else:
            bad += 1
        if bad > patience:
            return i + 1
    return None

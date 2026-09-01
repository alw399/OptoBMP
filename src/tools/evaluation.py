"""
Splits and metrics for a spatially autocorrelated, zero-inflated target.

Two things here are easy to get wrong and both change conclusions:

* SPLITS. A random cell-level split leaves every test cell's own physical
  neighbours in the training set, so a model that only smooths locally still
  scores well. `spatial_block_split` holds out whole TILES instead, which is what
  "generalises to an unseen region" actually requires. Report both -- the gap
  between them is itself the measurement of how much a score depends on spatial
  autocorrelation.

* CEILINGS. R2 on a per-cell zero-inflated target has a hard upper bound well
  below 1, because much of the cell-to-cell variation is single-cell noise no
  spatial model can reach. `oracle_ceiling` measures that bound directly, so a
  score can be read as a fraction of what is achievable rather than of 1.0.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    average_precision_score, r2_score, roc_auc_score,
)

from tools.spatial import BIN_PX, rasterize


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------


def random_split(n: int, val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 0):
    """Held-out CELLS. Neighbours leak across the split -- see the module docstring."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(round(n * val_frac))
    n_test = int(round(n * test_frac))
    return perm[n_val + n_test:], perm[:n_val], perm[n_val:n_val + n_test]


def spatial_block_split(centroids: np.ndarray, block: float = 800.0,
                        val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 0):
    """Held-out REGIONS: tile the field into `block`-px squares and assign whole
    tiles to train/val/test. Returns positional index arrays."""
    by = (centroids[:, 0] // block).astype(int)
    bx = (centroids[:, 1] // block).astype(int)
    bid = by * (bx.max() + 1) + bx
    uniq = np.unique(bid)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * val_frac)))
    n_test = max(1, int(round(len(uniq) * test_frac)))
    is_val = np.isin(bid, uniq[:n_val])
    is_test = np.isin(bid, uniq[n_val:n_val + n_test])
    is_train = ~(is_val | is_test)
    return np.where(is_train)[0], np.where(is_val)[0], np.where(is_test)[0]


# --------------------------------------------------------------------------
# Targets
# --------------------------------------------------------------------------


def hurdle_target(raw: np.ndarray, threshold: float, fit_idx: Optional[np.ndarray] = None):
    """The same target `models.scalers.HurdleScaler` builds: exactly 0 at or below
    `threshold`, `log1p(x - threshold)` rescaled to (0, ~1] above it. Returns
    `(y, y_positive)`. The scale is fit on `fit_idx` (train rows) only."""
    clipped = np.clip(raw - threshold, 0, None)
    t = np.log1p(clipped)
    scale = t[fit_idx].max() if fit_idx is not None else t.max()
    return (t / max(scale, 1e-8)).astype(np.float32), (raw > threshold).astype(np.float32)


def rank_target(raw: np.ndarray) -> np.ndarray:
    """Per-image quantile rank in [0, 1]. Threshold-free, so it stays comparable
    across images whose positive/negative cutoff has not been verified."""
    return pd.Series(raw).rank(pct=True).to_numpy(np.float32)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def evaluate(y_true: np.ndarray, y_pred: np.ndarray,
             y_positive: Optional[np.ndarray] = None) -> dict:
    """Every number worth looking at for a zero-inflated target, in one dict.

    AUROC is reported alongside R2 deliberately: AUROC is invariant to any monotone
    rescaling of the prediction while R2 is not, so a large gap between them points
    at a calibration problem rather than a ranking one."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    out = {
        "r2": r2_score(y_true, y_pred),
        "pearson": pearsonr(y_true, y_pred)[0],
        "spearman": spearmanr(y_true, y_pred).statistic,
    }
    if y_positive is not None:
        yp = np.asarray(y_positive).ravel()
        if 0 < yp.mean() < 1:
            out["auroc"] = roc_auc_score(yp, y_pred)
            out["ap"] = average_precision_score(yp, y_pred)
            out["base_rate"] = float(yp.mean())
        pos = yp > 0.5
        if pos.sum() > 10:
            out["r2_positives"] = r2_score(y_true[pos], y_pred[pos])
    return out


def oracle_ceiling(centroids: np.ndarray, y_continuous: np.ndarray,
                   y_positive: np.ndarray, sigma: float = 30.0,
                   bin_px: float = BIN_PX) -> dict:
    """The best score ANY spatial model could reach on this target.

    Predicts each cell from its NEIGHBOURS' true values, leave-one-out -- i.e. it is
    allowed to peek at the answer everywhere except the cell being scored. Whatever
    fraction of the per-cell variation is independent single-cell noise is
    unreachable, and this measures exactly that, so a real model's score can be read
    as a fraction of what is achievable instead of a fraction of 1.0.

    The leave-one-out step matters: without it a cell contributes to its own
    "neighbourhood" and the ceiling comes out spuriously high.
    """
    shape = (int(centroids[:, 0].max()) + 1, int(centroids[:, 1].max()) + 1)
    all_grid, iy, ix = rasterize(centroids, np.ones(len(centroids), np.float32), bin_px, shape)
    pos_grid, _, _ = rasterize(centroids, y_positive.astype(np.float32), bin_px, shape)
    con_grid, _, _ = rasterize(centroids, y_continuous.astype(np.float32), bin_px, shape)

    s = sigma / bin_px
    den = gaussian_filter(all_grid, sigma=s, mode="nearest")[iy, ix]
    num_p = gaussian_filter(pos_grid, sigma=s, mode="nearest")[iy, ix]
    num_c = gaussian_filter(con_grid, sigma=s, mode="nearest")[iy, ix]

    w0 = 1.0 / (2.0 * np.pi * s * s)          # the kernel's own weight at distance 0
    loo_p = np.clip(num_p - w0 * y_positive, 0, None) / np.maximum(den - w0, 1e-6)
    loo_c = np.clip(num_c - w0 * y_continuous, 0, None) / np.maximum(den - w0, 1e-6)

    ok = np.isfinite(loo_p) & np.isfinite(loo_c) & (den - w0 > 0.05)
    A = np.stack([loo_c[ok], np.ones(int(ok.sum()))], 1)
    coef, *_ = np.linalg.lstsq(A, y_continuous[ok], rcond=None)
    return {
        "sigma": sigma,
        "auroc": roc_auc_score(y_positive[ok], loo_p[ok]),
        "r2": r2_score(y_continuous[ok], A @ coef),
        "n": int(ok.sum()),
    }


def fraction_of_ceiling(metrics: dict, ceiling: dict) -> dict:
    """How much of the achievable signal a model actually captured."""
    return {
        "r2_pct_of_ceiling": 100.0 * metrics["r2"] / ceiling["r2"] if ceiling["r2"] else np.nan,
        "auroc_pct_of_ceiling": 100.0 * (metrics.get("auroc", np.nan) - 0.5) / (ceiling["auroc"] - 0.5),
    }

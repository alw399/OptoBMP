"""
Plots for the "is there anything to predict here?" stage, before any model exists.

These are cheap, model-free and decide what is worth building: how predictive each
spatial scale is on its own, how far a cell can actually see, and how the readout
responds to the pattern's geometry.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


def plot_scale_auroc(df: pd.DataFrame, feature_prefix: str, scales: Sequence[int],
                     targets: Sequence[tuple], ax=None, title: str = ""):
    """Univariate AUROC of each scale's neighbourhood feature against each target.

    Read the SHAPE, not the height: the scale at which the curve departs from 0.5
    is the length scale of the biology, and it is what the graph radius has to be
    able to reach. A curve that is flat at 0.5 up to some scale means every model
    confined below that scale is working with a constant input.
    `targets` is a sequence of `(name, binary_labels)`."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 4.8))
    for name, labels in targets:
        y = np.asarray(labels).ravel().astype(int)
        xs, ys = [], []
        for s in scales:
            col = f"{feature_prefix}{s}"
            if col not in df.columns:
                continue
            v = df[col].to_numpy()
            ok = np.isfinite(v)
            xs.append(s)
            ys.append(roc_auc_score(y[ok], v[ok]))
        ax.plot(xs, ys, "o-", label=name)
    ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("neighbourhood scale $\\sigma$ (px)")
    ax.set_ylabel("univariate AUROC")
    ax.set_title(title or "how predictive is the mask, per spatial scale")
    ax.legend()
    ax.figure.tight_layout()
    return ax


def plot_response_curve(signed_distance, targets: Sequence[tuple], n_bins: int = 40,
                        min_cells: int = 30, ax=None, title: str = ""):
    """Readout vs signed distance to the illuminated boundary (+ inside, - outside).

    This is the dose-response curve the pattern actually delivers, and it shows
    directly whether the effect is an edge effect or a whole-region effect."""
    sd = np.asarray(signed_distance).ravel()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.8))
    edges = np.unique(np.percentile(sd, np.linspace(0, 100, n_bins + 1)))
    which = np.digitize(sd, edges)
    for name, values in targets:
        v = np.asarray(values).ravel().astype(float)
        xs, ys = [], []
        for b in np.unique(which):
            m = which == b
            if m.sum() < min_cells:
                continue
            xs.append(np.median(sd[m]))
            ys.append(v[m].mean())
        ax.plot(xs, ys, "o-", ms=3, label=name)
    ax.axvline(0, color="r", ls="--", lw=1, label="pattern boundary")
    ax.set_xlabel("signed distance to illuminated boundary (px;  + = inside)")
    ax.set_ylabel("mean value")
    ax.set_title(title or "response vs pattern geometry")
    ax.legend()
    ax.figure.tight_layout()
    return ax


def plot_receptive_field_check(df: pd.DataFrame, mask_col: str, radii: Sequence[float],
                               ax=None, title: str = ""):
    """Fraction of cells with NO positive cell inside a given radius.

    The blunt version of the receptive-field question: a cell with no positive
    anywhere in its receptive field has a literally constant input, so nothing
    downstream can separate it from any other such cell. If that fraction is high
    at the radius the graph uses, no amount of architecture will help."""
    from scipy.spatial import cKDTree
    cent = df[["centroid_y", "centroid_x"]].to_numpy(np.float32)
    pos = cent[df[mask_col].to_numpy() > 0.5]
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.4))
    tree = cKDTree(pos)
    d, _ = tree.query(cent, k=1, workers=-1)
    fracs = [float((d > r).mean()) for r in radii]
    ax.plot(radii, fracs, "o-")
    for r, f in zip(radii, fracs):
        ax.annotate(f"{f:.0%}", (r, f), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=8)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("receptive-field radius (px)")
    ax.set_ylabel(f"cells with no {mask_col}+ cell in range")
    ax.set_title(title or "how many cells see a constant input?")
    ax.figure.tight_layout()
    return ax


def plot_field_correlation(rows: Sequence[dict], ax=None, title: str = ""):
    """Smoothed-field Pearson r between a mask field and each readout field, per scale.

    The most generous possible test of "is there any spatial relationship at all":
    per-cell noise is averaged away, so a near-zero value here means there is
    genuinely nothing for a model to find, not that the model was too weak.
    `rows` is a sequence of dicts with keys `sigma`, `pair`, `r`."""
    df = pd.DataFrame(rows)
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 4.8))
    for pair, sub in df.groupby("pair"):
        sub = sub.sort_values("sigma")
        ax.plot(sub["sigma"], sub["r"], "o-", label=pair)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("smoothing $\\sigma$ (px)")
    ax.set_ylabel("field-vs-field Pearson r")
    ax.set_title(title or "spatial relationship between fields")
    ax.legend(fontsize=9)
    ax.figure.tight_layout()
    return ax

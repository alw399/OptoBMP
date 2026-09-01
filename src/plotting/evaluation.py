"""
Reading a trained model's output.

Everything here plots against the ORACLE CEILING where one is available
(`tools.evaluation.oracle_ceiling`) rather than against 1.0. On a zero-inflated
per-cell target most of the variance is single-cell noise, so an R2 of 0.33 can be
three quarters of everything achievable -- plotting against 1.0 makes a good model
look broken.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_training_curves(history: dict, ceiling: Optional[float] = None, title: str = ""):
    """Loss and validation R2. The R2 panel is the one that matters -- it is the
    early-stopping criterion in `models.train.train_calibrated`, and it is what
    tells you whether the run simply needed more epochs."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    axs[0].plot(history["train_loss"], label="train")
    axs[0].plot(history["val_loss"], label="val")
    axs[0].set_xlabel("epoch")
    axs[0].set_ylabel("loss")
    axs[0].legend()
    axs[0].set_title("loss")
    if "val_r2" in history:
        axs[1].plot(history["val_r2"], color="tab:green")
        if ceiling is not None:
            axs[1].axhline(ceiling, color="r", ls="--", label="oracle ceiling")
            axs[1].legend()
        axs[1].set_xlabel("epoch")
        axs[1].set_ylabel("validation $R^2$")
        axs[1].set_title("validation $R^2$ (stopping criterion)")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_pred_vs_actual(y_true, y_pred, y_positive=None, title: str = "",
                        ax=None, max_points: int = 40000, seed: int = 0):
    """Predicted vs measured, split by the hurdle's positive/negative label.

    Splitting matters: on a zero-inflated target the negatives are the overwhelming
    majority and would otherwise hide the only population the magnitude branch is
    actually supervised on."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    idx = np.arange(len(y_true))
    if len(idx) > max_points:
        idx = np.random.default_rng(seed).choice(idx, max_points, replace=False)
    if y_positive is None:
        ax.scatter(y_true[idx], y_pred[idx], s=2, alpha=0.15, linewidths=0)
    else:
        yp = np.asarray(y_positive).ravel()
        for label, sel, c in [("negative", yp[idx] < 0.5, "tab:blue"),
                              ("positive", yp[idx] > 0.5, "tab:red")]:
            ax.scatter(y_true[idx][sel], y_pred[idx][sel], s=2, alpha=0.15,
                       linewidths=0, c=c, label=label)
        ax.legend(markerscale=6)
    lim = [0, max(y_true.max(), y_pred.max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("measured (hurdle-scaled)")
    ax.set_ylabel("predicted")
    ax.set_title(title)
    return ax


def plot_calibration(y_positive, y_prob, n_bins: int = 20, ax=None, title: str = ""):
    """Predicted probability vs observed frequency.

    Worth plotting for any hurdle model trained with `auto_pos_weight`: a curve
    sitting well above the diagonal is the prior-inflation described in
    `models.train.calibrated_predict`, not a modelling failure, and it is invisible
    in AUROC."""
    yp = np.asarray(y_positive).ravel()
    p = np.asarray(y_prob).ravel()
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5.5))
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    xs, ys = [], []
    for b in np.unique(which):
        m = which == b
        if m.sum() < 20:
            continue
        xs.append(p[m].mean())
        ys.append(yp[m].mean())
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfectly calibrated")
    ax.plot(xs, ys, "o-", label="observed")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed positive fraction")
    ax.set_title(title or "calibration")
    ax.legend()
    return ax


def plot_metric_comparison(results: pd.DataFrame, metric: str = "r2",
                           group: str = "config", hue: str = "split",
                           ceiling: Optional[float] = None, ax=None, title: str = ""):
    """Grouped bars for a tidy results table, with the ceiling drawn in."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.5))
    piv = results.pivot_table(index=group, columns=hue, values=metric)
    piv.plot(kind="bar", ax=ax, rot=15, width=0.78)
    if ceiling is not None:
        ax.axhline(ceiling, color="r", ls="--", lw=1.2, label="oracle ceiling")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel(metric)
    ax.set_xlabel("")
    ax.set_title(title or metric)
    ax.legend(fontsize=9)
    ax.figure.tight_layout()
    return ax


def plot_scale_curve(results: pd.DataFrame, x: str = "max_scale",
                     series: Sequence[str] = ("within_blocked", "cross_well"),
                     ax=None, title: str = ""):
    """Score vs the maximum context scale included.

    The characteristic shape to look for: both curves rise together while extra
    context is real information, then separate -- within-image keeps climbing while
    cross-image falls -- at the scale where the features start acting as position
    proxies instead."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 4.8))
    for s in series:
        if s in results.columns:
            ax.plot(results[x], results[s], "o-", label=s.replace("_", " "))
    ax.set_xscale("log")
    ax.set_xlabel("maximum context scale $\\sigma$ (px)")
    ax.set_ylabel("Spearman $\\rho$")
    ax.legend()
    ax.set_title(title)
    ax.figure.tight_layout()
    return ax


def metrics_table(rows: Sequence[dict], ceiling: Optional[dict] = None) -> pd.DataFrame:
    """Tidy metric rows into a display table, adding percent-of-ceiling columns."""
    df = pd.DataFrame(rows)
    if ceiling is not None:
        if "r2" in df:
            df["r2_% of ceiling"] = (100 * df["r2"] / ceiling["r2"]).round(0)
        if "auroc" in df:
            df["auroc_% of ceiling"] = (
                100 * (df["auroc"] - 0.5) / (ceiling["auroc"] - 0.5)).round(0)
    return df.round(4)

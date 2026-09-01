"""
Plots for the "why did it predict that?" stage, after a model exists.

Pairs with `models.interpret`. The recurring theme: an importance number on its own
is not readable, so every plot here puts something next to it -- the two importance
methods side by side, the ICE spread behind a partial-dependence mean, the
self-loop bar next to the neighbour bars.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plotting.fields import plot_cell_map


def plot_attribution_bars(table: pd.DataFrame, ax=None, title: str = "",
                          value: str = "mean_abs"):
    """Per-feature integrated-gradient influence (`interpret.attribution_table`).

    Bar LENGTH is `mean_abs` -- how much the feature moved the prediction. Bar
    COLOUR is the signed mean, so a long red bar means "pushes the prediction up",
    a long blue one "pushes it down", and a long bar in near-white means the
    feature does both for different cells and cancels on average -- which is a real
    and common pattern here (a mask fraction raises the prediction inside the
    pattern and lowers it outside), and is invisible in any single-number ranking.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 0.45 * len(table) + 1.6))
    t = table.iloc[::-1]
    lim = np.abs(t["mean"]).max() or 1.0
    colors = plt.get_cmap("coolwarm")((t["mean"] / (2 * lim) + 0.5).to_numpy())
    ax.barh(t["feature"], t[value], color=colors, edgecolor="0.3", linewidth=0.5)
    for y, (v, share) in enumerate(zip(t[value], t["share"])):
        ax.text(v, y, f"  {share:.0%}", va="center", fontsize=8, color="0.3")
    ax.set_xlabel(f"{value} attribution (prediction units)")
    ax.set_title(title or "integrated gradients: feature influence")
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(-lim, lim))
    ax.figure.colorbar(sm, ax=ax, fraction=0.04, label="signed mean")
    ax.figure.tight_layout()
    return ax


def plot_importance_comparison(ig_table: pd.DataFrame, perm_table: pd.DataFrame,
                               metric: str = "r2_drop", ax=None, title: str = ""):
    """Integrated-gradient share vs permutation score drop, one point per feature.

    The diagonal is agreement. Off-diagonal is the informative part, and each
    corner means something specific:

      high IG, ~0 drop -- REDUNDANT: the model leans on it, but its information
        also lives in a correlated neighbour (the adjacent pyramid scales), so
        destroying it costs nothing.
      ~0 IG, high drop -- rarer, and usually a feature whose effect is nonlinear
        near the baseline, so the straight-line IG path passes through a flat
        region while shuffling still breaks the fit.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    m = ig_table.merge(perm_table, on="feature")
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.scatter(m["share"], m[metric], s=45, color="tab:purple", zorder=3)
    if f"{metric}_std" in m:
        ax.errorbar(m["share"], m[metric], yerr=m[f"{metric}_std"], fmt="none",
                    ecolor="0.6", lw=1, zorder=2)
    for _, r in m.iterrows():
        ax.annotate(r["feature"], (r["share"], r[metric]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("IG share of |attribution|")
    ax.set_ylabel(f"permutation {metric}")
    ax.set_title(title or "attribution vs. what breaks when you destroy it")
    ax.figure.tight_layout()
    return ax


def plot_branch_attribution(prob_table: pd.DataFrame, mag_table: pd.DataFrame,
                            ax=None, title: str = ""):
    """The hurdle model's two branches side by side: which features decide WHETHER a
    cell is positive (classifier logit) vs HOW BRIGHT it is if it is (magnitude).

    Shares, not raw attributions -- the two branches live on different scales (a
    logit is unbounded, a magnitude is in (0, 1)), so only the within-branch
    ranking is comparable across them. A feature that dominates one branch and not
    the other is the most useful thing this plot can show: it means the model has
    learned presence and level from different inputs, which a single 'combined'
    attribution would average away.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 0.5 * len(prob_table) + 1.6))
    order = prob_table.sort_values("share")["feature"].tolist()
    p = prob_table.set_index("feature").loc[order, "share"]
    m = mag_table.set_index("feature").loc[order, "share"]
    y = np.arange(len(order))
    ax.barh(y - 0.2, p, height=0.4, label="presence (logit)", color="tab:blue")
    ax.barh(y + 0.2, m, height=0.4, label="level (magnitude)", color="tab:orange")
    ax.set_yticks(y, order)
    ax.set_xlabel("share of |attribution| within branch")
    ax.set_title(title or "hurdle branches: presence vs level")
    ax.legend()
    ax.figure.tight_layout()
    return ax


def plot_partial_dependence(curves, ax=None, title: str = "", show_ice: bool = True,
                            xlabel: Optional[str] = None, legend: bool = True):
    """Partial-dependence curve(s) from `interpret.partial_dependence`.

    `curves` is one frame or a `{label: frame}` dict (one feature each, shared
    axes). Individual-cell (ICE) curves are drawn faint behind the mean when the
    frame carries them: if they fan out or cross, the mean curve is an average
    over cells that behave differently, and quoting the mean alone would be a
    statement about no actual cell.
    """
    if isinstance(curves, pd.DataFrame):
        curves = {curves.attrs.get("feature", "feature"): curves}
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.6))
    cmap = plt.get_cmap("tab10")
    for k, (label, c) in enumerate(curves.items()):
        color = cmap(k % 10)
        ice_cols = [col for col in c.columns if col.startswith("ice_")]
        if show_ice and ice_cols:
            ax.plot(c["x"], c[ice_cols].to_numpy(), color=color, alpha=0.12, lw=0.7)
        ax.fill_between(c["x"], c["pred_q25"], c["pred_q75"], color=color, alpha=0.15)
        ax.plot(c["x"], c["pred_mean"], "-", color=color, lw=2, label=label)
    ax.set_xlabel(xlabel or "feature value (normalized units)")
    ax.set_ylabel("mean predicted response")
    ax.set_title(title or "partial dependence")
    if legend:
        ax.legend(fontsize=8)
    ax.figure.tight_layout()
    return ax


def plot_attribution_maps(df: pd.DataFrame, ig: dict, features: Sequence[str],
                          ncols: int = 3, figsize_per: tuple = (5.0, 6.0),
                          quantile: float = 0.99, suptitle: str = ""):
    """Where in the well each feature's attribution actually lands.

    Diverging colour, symmetric around 0, clipped at the `quantile`-th percentile of
    |attribution| so a handful of extreme cells don't flatten everything else. Read
    it against the illumination pattern: an attribution map that tracks the pattern
    is the model using the pattern, while one that tracks the image edges or a
    density gradient is the model using an artefact -- which is exactly the failure
    the blocked split exists to catch, seen directly rather than inferred from a
    score gap.
    """
    attr, cols = ig["attr"], list(ig["x_cols"])
    n = len(features)
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, min(ncols, n),
                            figsize=(figsize_per[0] * min(ncols, n), figsize_per[1] * nrows),
                            squeeze=False)
    axs = axs.ravel()
    for ax, feat in zip(axs, features):
        v = attr[:, cols.index(feat)]
        lim = float(np.quantile(np.abs(v), quantile)) or 1.0
        plot_cell_map(df, v, ax=ax, cmap="coolwarm", title=feat, vmin=-lim, vmax=lim)
    for ax in axs[n:]:
        ax.set_visible(False)
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def plot_edge_sensitivity(table: pd.DataFrame, ax=None, title: str = "",
                          radius: Optional[float] = None):
    """Where each message-passing layer's influence comes from, by edge length.

    The `dist = 0` self-loop -- the cell's OWN features -- is drawn as a separate
    bar at the left, because it is a different kind of thing from the neighbour
    curve and putting it on the same axis as a distance would imply a continuum
    that isn't there. Its height against the area under the neighbour curve is the
    answer to "is this a graph model or a per-cell model with extra columns?".
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 4.6))
    cmap = plt.get_cmap("viridis")
    layers = sorted(table["layer"].unique())
    for layer in layers:
        t = table[table["layer"] == layer]
        color = cmap(layer / max(len(layers) - 1, 1) * 0.8)
        nb = t[~t["self_loop"]]
        ax.plot(nb["dist"], nb["share"], "o-", ms=3, color=color, label=f"layer {layer}")
        self_share = float(t.loc[t["self_loop"], "share"].sum())
        ax.bar(-2.5 + layer * 1.6, self_share, width=1.5, color=color, alpha=0.85)
        ax.text(-2.5 + layer * 1.6, self_share, f"{self_share:.0%}", ha="center",
                va="bottom", fontsize=7, color=color)
    if radius:
        ax.axvline(radius, color="k", ls=":", lw=1)
        ax.text(radius, ax.get_ylim()[1], " graph radius", fontsize=7, va="top")
    ax.set_xlabel("edge length (px)      [bars at left: self-loop share]")
    ax.set_ylabel("share of |grad x message| in layer")
    ax.set_title(title or "how much of the prediction arrives through the graph")
    ax.legend(fontsize=8)
    ax.figure.tight_layout()
    return ax


def plot_counterfactual_panel(df: pd.DataFrame, panels: Sequence[tuple],
                              figsize_per: tuple = (4.6, 5.6), suptitle: str = "",
                              cmap: str = "magma"):
    """A row of (mask, prediction) pairs for counterfactual illumination patterns.

    `panels` is a sequence of `(label, mask_values, prediction)`. Predictions share
    one colour scale across every panel -- the comparison IS the magnitude, so
    independent scales would make a pattern that barely responds look identical to
    one that responds strongly.
    """
    preds = [p for _, _, p in panels]
    vmax = float(np.quantile(np.concatenate([np.asarray(p).ravel() for p in preds]), 0.99))
    n = len(panels)
    # constrained_layout, not tight_layout: the shared colorbar below is attached to
    # a whole ROW of axes, and tight_layout has no way to reserve space for that --
    # it either clips the colorbar or shrinks one panel out of alignment.
    fig, axs = plt.subplots(2, n, figsize=(figsize_per[0] * n, figsize_per[1] * 2),
                            squeeze=False, layout="constrained")
    for k, (label, mask, pred) in enumerate(panels):
        plot_cell_map(df, np.asarray(mask).ravel(), ax=axs[0, k], cmap="gray_r",
                      title=label, colorbar=False)
        plot_cell_map(df, np.asarray(pred).ravel(), ax=axs[1, k], cmap=cmap,
                      title="predicted", vmin=0, vmax=vmax, colorbar=False)
    if suptitle:
        fig.suptitle(suptitle)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
    fig.colorbar(sm, ax=axs[1, :].tolist(), fraction=0.025, pad=0.01, label="predicted")
    return fig


def plot_response_range(profiles, ax=None, title: str = "", radii=None):
    """Predicted response vs distance from the pattern centre, one curve per
    counterfactual pattern -- the 1-D read of the model's implied signalling range.

    `profiles` is a `{label: (distance, mean_prediction)}` dict. With the pattern
    EDGE marked (`radii`, one per label), the distance between the edge and where
    the curve returns to baseline is the range the model has learned. That number
    is the interpretability result worth reporting: it is a property of the learned
    rule, comparable against the ~90px the graph can reach and the 240px the
    largest pyramid feature covers.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 4.6))
    cmap = plt.get_cmap("viridis")
    labels = list(profiles)
    for k, label in enumerate(labels):
        d, v = profiles[label]
        color = cmap(k / max(len(labels) - 1, 1) * 0.85)
        ax.plot(d, v, "-", color=color, lw=2, label=label)
        if radii is not None:
            ax.axvline(radii[k], color=color, ls=":", lw=1)
    ax.set_xlabel("distance from pattern centre (px)      [dotted: pattern edge]")
    ax.set_ylabel("mean predicted response")
    ax.set_title(title or "implied signalling range")
    ax.legend(fontsize=8)
    ax.figure.tight_layout()
    return ax

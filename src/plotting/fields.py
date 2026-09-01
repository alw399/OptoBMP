"""
Looking at a cell table in space.

The per-cell scatter is the honest view but at ~100k cells it is too dense to read;
the smoothed field is readable but hides per-cell noise. Both are here because the
two together are what actually answers "is the readout following the pattern".
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from tools.spatial import BIN_PX, rasterize, smoothed_field


def plot_cell_map(df: pd.DataFrame, values, ax=None, cmap: str = "magma",
                  title: str = "", s: float = 1.2, vmin=None, vmax=None,
                  colorbar: bool = True):
    """One panel: every cell as a dot, coloured by `values`."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 9))
    cent = df[["centroid_y", "centroid_x"]].to_numpy(np.float32)
    v = df[values].to_numpy() if isinstance(values, str) else np.asarray(values)
    sc = ax.scatter(cent[:, 1], cent[:, 0], c=v, s=s, cmap=cmap, linewidths=0,
                    vmin=vmin, vmax=vmax)
    ax.set_title(title or (values if isinstance(values, str) else ""))
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if colorbar:
        ax.figure.colorbar(sc, ax=ax, fraction=0.046)
    return ax


def plot_cell_maps(df: pd.DataFrame, panels: Sequence[tuple], ncols: Optional[int] = None,
                   figsize_per: tuple = (5.5, 7), suptitle: str = ""):
    """A row of `plot_cell_map` panels. `panels` is a sequence of
    `(title, values, cmap)`, where `values` is a column name or an array."""
    n = len(panels)
    ncols = ncols or n
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols,
                            figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows),
                            squeeze=False)
    axs = axs.ravel()
    for ax, (title, values, cmap) in zip(axs, panels):
        plot_cell_map(df, values, ax=ax, cmap=cmap, title=title)
    for ax in axs[n:]:
        ax.set_visible(False)
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def plot_field_grid(df: pd.DataFrame, rows: Sequence[tuple], sigmas: Sequence[float],
                    cmap: str = "magma", bin_px: float = BIN_PX):
    """Rows x scales grid of smoothed fields -- the single most useful diagnostic
    picture for this kind of data.

    `rows` is a sequence of `(name, values_or_None)`; a `None` value plots raw cell
    DENSITY for that row, which is worth including in every comparison because a
    marker field and the density field are easy to confuse by eye.
    """
    cent = df[["centroid_y", "centroid_x"]].to_numpy(np.float32)
    shape = (int(cent[:, 0].max()) + 1, int(cent[:, 1].max()) + 1)
    all_grid, _, _ = rasterize(cent, np.ones(len(df), np.float32), bin_px, shape)

    fig, axs = plt.subplots(len(rows), len(sigmas),
                            figsize=(4.2 * len(sigmas), 5.4 * len(rows)), squeeze=False)
    for r, (name, values) in enumerate(rows):
        for c, sig in enumerate(sigmas):
            ax = axs[r, c]
            if values is None:
                field = gaussian_filter(all_grid, sigma=sig / bin_px, mode="nearest")
            else:
                v = df[values].to_numpy(np.float32) if isinstance(values, str) else np.asarray(values, np.float32)
                field = smoothed_field(cent, v, sig, bin_px=bin_px, shape_px=shape)
            im = ax.imshow(field, cmap=cmap if values is not None else "viridis",
                           interpolation="nearest")
            ax.set_title(f"{name}   $\\sigma$={sig:g}px", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def plot_prediction_panel(df: pd.DataFrame, mask_values, y_true, y_pred,
                          target_name: str = "", mask_name: str = "input mask",
                          smooth_sigma: Optional[float] = None):
    """Input / measured / predicted, side by side and on a shared colour scale.

    A shared scale matters: a prediction that is right in shape but systematically
    off in magnitude looks correct on independent scales, and that is exactly the
    failure mode an uncalibrated hurdle head produces.
    """
    cent = df[["centroid_y", "centroid_x"]].to_numpy(np.float32)
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mask_values = df[mask_values].to_numpy() if isinstance(mask_values, str) else np.asarray(mask_values)

    vmax = np.percentile(y_true[y_true > 0], 95) if (y_true > 0).any() else 1.0
    fig, axs = plt.subplots(1, 3, figsize=(19, 8))
    if smooth_sigma:
        shape = (int(cent[:, 0].max()) + 1, int(cent[:, 1].max()) + 1)
        panels = [(mask_name, smoothed_field(cent, mask_values.astype(np.float32), smooth_sigma, shape_px=shape), "gray_r", None),
                  (f"measured {target_name}", smoothed_field(cent, y_true, smooth_sigma, shape_px=shape), "magma", vmax),
                  (f"predicted {target_name}", smoothed_field(cent, y_pred, smooth_sigma, shape_px=shape), "magma", vmax)]
        for ax, (t, field, cm, vm) in zip(axs, panels):
            im = ax.imshow(field, cmap=cm, vmax=vm, interpolation="nearest")
            ax.set_title(t)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046)
    else:
        panels = [(mask_name, mask_values, "gray_r", 1.0),
                  (f"measured {target_name}", y_true, "magma", vmax),
                  (f"predicted {target_name}", y_pred, "magma", vmax)]
        for ax, (t, v, cm, vm) in zip(axs, panels):
            sc = ax.scatter(cent[:, 1], cent[:, 0], c=v, s=1.2, cmap=cm, linewidths=0,
                            vmin=0, vmax=vm)
            ax.set_title(t)
            ax.invert_yaxis(); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(sc, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def plot_radial_profile(df: pd.DataFrame, centre: tuple[float, float],
                        series: Sequence[tuple], step: float = 50.0, min_cells: int = 50):
    """Marker fraction vs distance from a pattern centre -- the cleanest 1-D read of
    a radially symmetric response. `series` is a sequence of `(label, values)`."""
    cent = df[["centroid_y", "centroid_x"]].to_numpy(np.float32)
    rad = np.hypot(cent[:, 0] - centre[0], cent[:, 1] - centre[1])
    edges = np.arange(0, rad.max(), step)
    which = np.digitize(rad, edges)

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, values in series:
        v = df[values].to_numpy(np.float32) if isinstance(values, str) else np.asarray(values, np.float32)
        xs, ys = [], []
        for k in np.unique(which):
            m = which == k
            if m.sum() < min_cells:
                continue
            xs.append(rad[m].mean())
            ys.append(v[m].mean())
        ax.plot(xs, ys, "-", label=label)
    ax.set_xlabel("distance from pattern centre (px)")
    ax.set_ylabel("mean value")
    ax.legend()
    fig.tight_layout()
    return fig


def estimate_pattern_centre(df: pd.DataFrame, mask_col: str, sigma: float = 200.0,
                            bin_px: float = BIN_PX) -> tuple[float, float]:
    """Centre of the illuminated pattern: the peak of a heavily smoothed mask density."""
    cent = df[["centroid_y", "centroid_x"]].to_numpy(np.float32)
    shape = (int(cent[:, 0].max()) + 1, int(cent[:, 1].max()) + 1)
    grid, _, _ = rasterize(cent, df[mask_col].to_numpy(np.float32), bin_px, shape)
    sm = gaussian_filter(grid, sigma=sigma / bin_px, mode="constant")
    cy, cx = np.unravel_index(np.argmax(sm), sm.shape)
    return float(cy * bin_px), float(cx * bin_px)

"""
Multi-scale neighbourhood features over a cell table.

A binary per-cell mask (e.g. thresholded BMP4) carries no per-cell intensity to
learn from -- the ONLY thing a model can use is the mask's spatial arrangement. So
the whole question is at which SCALE that arrangement matters, which is what these
features parameterise.

Everything is computed by rasterising the cells onto a coarse grid and convolving,
not by KD-tree neighbour lists: at sigma ~ 1000px each cell has ~7000 neighbours,
so explicit neighbour lists run to ~4e8 entries and exhaust memory, while grid
convolution is O(pixels) per scale regardless of sigma. At the default `bin_px=8`
(well under one cell diameter) the discretisation error is negligible next to the
scales being probed.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, uniform_filter
from scipy.spatial import cKDTree

PYRAMID_SCALES = (15, 30, 60, 120, 240, 480, 960, 1920)
BIN_PX = 8.0


# --------------------------------------------------------------------------
# Rasterisation
# --------------------------------------------------------------------------


def rasterize(centroids: np.ndarray, values: np.ndarray, bin_px: float = BIN_PX,
              shape_px: Optional[tuple[float, float]] = None):
    """Sum `values` into a coarse 2D grid of `bin_px`-sized bins.

    Returns `(grid, iy, ix)` -- the grid plus each cell's own bin indices, so a
    smoothed grid can be sampled straight back at the cells with `grid[iy, ix]`."""
    if shape_px is None:
        shape_px = (centroids[:, 0].max() + 1, centroids[:, 1].max() + 1)
    h, w = shape_px
    ny, nx = int(np.ceil(h / bin_px)), int(np.ceil(w / bin_px))
    iy = np.clip((centroids[:, 0] / bin_px).astype(int), 0, ny - 1)
    ix = np.clip((centroids[:, 1] / bin_px).astype(int), 0, nx - 1)
    flat = np.bincount(iy * nx + ix, weights=values, minlength=ny * nx)
    return flat.reshape(ny, nx).astype(np.float32), iy, ix


def smoothed_field(centroids: np.ndarray, values: np.ndarray, sigma: float,
                   bin_px: float = BIN_PX, shape_px=None, min_density: float = 0.02):
    """The local MEAN of `values` -- `smooth(sum of values) / smooth(cell count)` --
    as a grid, with cell-free regions set to NaN so they do not read as zeros.

    This is the right way to look at a marker across an image with uneven cell
    coverage: a raw smoothed count conflates "few positive cells" with "few cells"."""
    all_grid, _, _ = rasterize(centroids, np.ones(len(centroids), np.float32), bin_px, shape_px)
    val_grid, _, _ = rasterize(centroids, values.astype(np.float32), bin_px, shape_px)
    s = sigma / bin_px
    num = gaussian_filter(val_grid, sigma=s, mode="nearest")
    den = gaussian_filter(all_grid, sigma=s, mode="nearest")
    out = num / np.maximum(den, 1e-6)
    out[den < min_density] = np.nan
    return out


# --------------------------------------------------------------------------
# Multi-scale summaries of a binary mask
# --------------------------------------------------------------------------


def multiscale_fraction(centroids: np.ndarray, binary: np.ndarray,
                        scales: Sequence[int] = PYRAMID_SCALES,
                        bin_px: float = BIN_PX, shape_px=None,
                        box: bool = False) -> pd.DataFrame:
    """Per cell, per scale: the positive FRACTION of the mask in its neighbourhood.

    Gaussian kernel by default (smooth, no discontinuous window edge); `box=True`
    additionally returns hard-disc counts (`n_{r}`, `pos_{r}`, `frac_{r}`), which
    are easier to reason about but noisier."""
    if shape_px is None:
        shape_px = (centroids[:, 0].max() + 1, centroids[:, 1].max() + 1)
    all_grid, iy, ix = rasterize(centroids, np.ones(len(centroids), np.float32), bin_px, shape_px)
    pos_grid, _, _ = rasterize(centroids, binary.astype(np.float32), bin_px, shape_px)

    out: dict[str, np.ndarray] = {}
    for s in scales:
        sig = s / bin_px
        num = gaussian_filter(pos_grid, sigma=sig, mode="nearest")
        den = gaussian_filter(all_grid, sigma=sig, mode="nearest")
        out[f"gauss_{s}"] = (num / np.maximum(den, 1e-8))[iy, ix]
    if box:
        for s in scales:
            size = max(1, int(round(2 * s / bin_px)))
            a = uniform_filter(all_grid, size=size, mode="nearest")
            p = uniform_filter(pos_grid, size=size, mode="nearest")
            out[f"n_{s}"] = (a * size * size)[iy, ix]
            out[f"pos_{s}"] = (p * size * size)[iy, ix]
            out[f"frac_{s}"] = (p / np.maximum(a, 1e-8))[iy, ix]
    return pd.DataFrame({k: v.astype(np.float32) for k, v in out.items()})


def multiscale_density(centroids: np.ndarray, scales: Sequence[int] = PYRAMID_SCALES,
                       bin_px: float = BIN_PX, shape_px=None) -> pd.DataFrame:
    """Per cell, per scale: smoothed local CELL density. No marker channel involved."""
    grid, iy, ix = rasterize(centroids, np.ones(len(centroids), np.float32), bin_px, shape_px)
    out = {}
    for s in scales:
        out[f"dens_{s}"] = gaussian_filter(grid, sigma=s / bin_px, mode="nearest")[iy, ix]
    return pd.DataFrame({k: v.astype(np.float32) for k, v in out.items()})


def multiscale_mean(centroids: np.ndarray, values: np.ndarray,
                    scales: Sequence[int] = PYRAMID_SCALES,
                    bin_px: float = BIN_PX, shape_px=None) -> pd.DataFrame:
    """Per cell, per scale: the local MEAN of a CONTINUOUS `values` column --
    `smooth(sum of values) / smooth(cell count)`, i.e. `smoothed_field` evaluated at
    every cell's own position, at each scale. The multiscale analog of
    `multiscale_fraction` for a channel that isn't a 0/1 indicator -- e.g. raw
    marker/mask INTENSITY rather than a thresholded positive/negative call, where a
    positive fraction has no meaning but a local average intensity still does."""
    if shape_px is None:
        shape_px = (centroids[:, 0].max() + 1, centroids[:, 1].max() + 1)
    all_grid, iy, ix = rasterize(centroids, np.ones(len(centroids), np.float32), bin_px, shape_px)
    val_grid, _, _ = rasterize(centroids, values.astype(np.float32), bin_px, shape_px)
    out = {}
    for s in scales:
        sig = s / bin_px
        num = gaussian_filter(val_grid, sigma=sig, mode="nearest")
        den = gaussian_filter(all_grid, sigma=sig, mode="nearest")
        out[f"mean_{s}"] = (num / np.maximum(den, 1e-8))[iy, ix]
    return pd.DataFrame({k: v.astype(np.float32) for k, v in out.items()})


# --------------------------------------------------------------------------
# Pattern geometry
# --------------------------------------------------------------------------


def distance_to(centroids: np.ndarray, selected: np.ndarray) -> np.ndarray:
    """Euclidean distance from every cell to the nearest cell in `selected`."""
    pts = centroids[selected > 0.5]
    if len(pts) == 0:
        return np.full(len(centroids), np.inf, dtype=np.float32)
    d, _ = cKDTree(pts).query(centroids, k=1, workers=-1)
    return d.astype(np.float32)


def signed_boundary_distance(centroids: np.ndarray, binary: np.ndarray) -> np.ndarray:
    """Signed distance to the illuminated-region boundary: POSITIVE inside a
    positive region, NEGATIVE outside, magnitude = distance to the nearest cell of
    the opposite class.

    This is the only feature that distinguishes *inside a ring, with no positive
    cell nearby* from *far outside the pattern, with no positive cell nearby*.
    Every neighbourhood-density feature at every scale reports those two the same
    way, and they are exactly the two the readouts differ most between."""
    d_pos = distance_to(centroids, binary)
    d_neg = distance_to(centroids, 1.0 - binary)
    return np.where(binary > 0.5, d_neg, -d_pos).astype(np.float32)


# --------------------------------------------------------------------------
# DataFrame-level convenience
# --------------------------------------------------------------------------


def add_mask_pyramid(df: pd.DataFrame, mask_col: str = "BMP4_bin",
                     scales: Sequence[int] = PYRAMID_SCALES,
                     geometry: bool = True) -> list[str]:
    """Attach the mask pyramid (and optionally the pattern geometry) to `df` IN
    PLACE, prefixed with `mask_col`. Returns the new column names."""
    cent = df[["centroid_y", "centroid_x"]].to_numpy(np.float32)
    binary = df[mask_col].to_numpy(np.float32)
    shape = (int(cent[:, 0].max()) + 1, int(cent[:, 1].max()) + 1)
    ms = multiscale_fraction(cent, binary, scales=scales, shape_px=shape)
    names = []
    for c in ms.columns:
        name = f"{mask_col}_{c}"
        df[name] = ms[c].to_numpy(np.float32)
        names.append(name)
    if geometry:
        df[f"{mask_col}_signed_dist"] = signed_boundary_distance(cent, binary)
        df[f"{mask_col}_log_dist_pos"] = np.log1p(distance_to(cent, binary))
        df[f"{mask_col}_log_dist_neg"] = np.log1p(distance_to(cent, 1.0 - binary))
        names += [f"{mask_col}_signed_dist", f"{mask_col}_log_dist_pos",
                  f"{mask_col}_log_dist_neg"]
    return names


def add_density_pyramid(df: pd.DataFrame, scales: Sequence[int] = PYRAMID_SCALES) -> list[str]:
    """Attach smoothed cell density at each scale to `df` IN PLACE."""
    cent = df[["centroid_y", "centroid_x"]].to_numpy(np.float32)
    shape = (int(cent[:, 0].max()) + 1, int(cent[:, 1].max()) + 1)
    ms = multiscale_density(cent, scales=scales, shape_px=shape)
    for c in ms.columns:
        df[c] = ms[c].to_numpy(np.float32)
    return list(ms.columns)


def add_intensity_pyramid(df: pd.DataFrame, value_col: str,
                          scales: Sequence[int] = PYRAMID_SCALES) -> list[str]:
    """Attach a multiscale local-MEAN pyramid of a CONTINUOUS channel (e.g. raw
    marker/mask intensity, not a thresholded indicator) to `df` IN PLACE, prefixed
    with `value_col`. Same role as `add_mask_pyramid`'s fraction pyramid, but for a
    real intensity rather than a 0/1 mask -- no `geometry` option, since signed
    boundary distance (`signed_boundary_distance`) needs a binary class to be
    "inside"/"outside" of, which a continuous channel doesn't have on its own."""
    cent = df[["centroid_y", "centroid_x"]].to_numpy(np.float32)
    values = df[value_col].to_numpy(np.float32)
    shape = (int(cent[:, 0].max()) + 1, int(cent[:, 1].max()) + 1)
    ms = multiscale_mean(cent, values, scales=scales, shape_px=shape)
    names = []
    for c in ms.columns:
        name = f"{value_col}_{c}"
        df[name] = ms[c].to_numpy(np.float32)
        names.append(name)
    return names


def shuffle_mask(df: pd.DataFrame, mask_col: str = "BMP4_bin", seed: int = 0,
                 out_col: str = "mask_shuffled") -> str:
    """Permute a binary mask across cells: destroys its spatial pattern while
    preserving the number of positives AND the cell-density field exactly.

    The negative control for "is the model using the mask's ARRANGEMENT, or just
    the cell layout that happens to come with it?"."""
    rng = np.random.default_rng(seed)
    v = df[mask_col].to_numpy(np.float32).copy()
    rng.shuffle(v)
    df[out_col] = v
    return out_col

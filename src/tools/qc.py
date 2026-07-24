"""
Scalable sanity checks for a Cellpose segmentation result.

With 100k+ cells, eyeballing the whole mosaic isn't feasible and rendering
masks one-by-one isn't either. Instead: (1) look at the *distribution* of
morphology stats to catch fragments/merges, and (2) bin cell centroids into a
spatial density map to catch regions that were under- or over-segmented --
then only crop and visually inspect the flagged regions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

_COLOR_RGB = {
    "blue": (0, 0, 1),
    "green": (0, 1, 0),
    "red": (1, 0, 0),
    "magenta": (1, 0, 1),
    "cyan": (0, 1, 1),
    "yellow": (1, 1, 0),
    "gray": (1, 1, 1),
}


def plot_feature_histograms(
    df: pd.DataFrame,
    features: list[str],
    bins: int = 100,
    view_thresholds: dict[str, float] | None = None,
    log_scale=False
):
    """
    Distribution sanity check for any column(s) in a cell feature table -- morphology
    (e.g. spikes at very small `area` = noise/fragments; a heavy tail at large `area`
    = merged/under-segmented blobs; a pileup of low `circularity`/`solidity` = two
    touching cells merged into one mask) or marker intensity (e.g. `HAND1_mean`).

    `view_thresholds` optionally maps a subset of `features` to an x position to mark
    with a dashed red line (e.g. a candidate cutoff you're considering) -- may be
    empty or only cover some of `features`.
    """
    view_thresholds = view_thresholds or {}
    fig, axes = plt.subplots(1, len(features), figsize=(5 * len(features), 4))
    if len(features) == 1:
        axes = [axes]
    for ax, col in zip(axes, features):
        ax.hist(df[col], bins=bins)
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("count")
        if col in view_thresholds:
            ax.axvline(view_thresholds[col], color="red", linestyle="--")
        if log_scale:
            ax.set_yscale("log")
    fig.tight_layout()
    return fig

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.signal import argrelextrema

def get_bimodal_threshold(data, col_name, log_transform=True):
    # Filter out NaNs and non-positive values
    vals = data[col_name].dropna()
    if log_transform:
        vals = vals[vals > 0]
        x_data = np.log10(vals)
    else:
        x_data = vals.to_numpy()
    
    # Estimate probability density
    kde = gaussian_kde(x_data)
    x_grid = np.linspace(x_data.min(), x_data.max(), 1000)
    y_grid = kde(x_grid)
    
    minima_idx = argrelextrema(y_grid, np.less)[0]
    maxima_idx = argrelextrema(y_grid, np.greater)[0]
    
    threshold = None
    if len(minima_idx) > 0:
        if len(maxima_idx) >= 2:
            # Sort peaks by density height to find the negative and positive peak modes
            top_peaks_idx = sorted(maxima_idx, key=lambda idx: y_grid[idx], reverse=True)[:2]
            left_peak = min(top_peaks_idx)
            right_peak = max(top_peaks_idx)
            
            # Find valleys that lie between the two peaks
            valleys_between = [idx for idx in minima_idx if left_peak < idx < right_peak]
            if valleys_between:
                best_valley_idx = min(valleys_between, key=lambda idx: y_grid[idx])
                threshold = x_grid[best_valley_idx]
        
        # Fallback: select the overall lowest valley
        if threshold is None:
            best_valley_idx = min(minima_idx, key=lambda idx: y_grid[idx])
            threshold = x_grid[best_valley_idx]
            
    if threshold is not None:
        return 10**threshold if log_transform else threshold
    return None


def cell_density_map(df: pd.DataFrame, image_shape: tuple[int, int], bin_size: int = 200):
    """2D histogram of cell centroids: (n_bins_y, n_bins_x) array of cell counts per bin."""
    h, w = image_shape
    y_edges = np.arange(0, h + bin_size, bin_size)
    x_edges = np.arange(0, w + bin_size, bin_size)
    density, _, _ = np.histogram2d(df["centroid_y"], df["centroid_x"], bins=[y_edges, x_edges])
    return density, y_edges, x_edges


def plot_density_map(df: pd.DataFrame, image_shape: tuple[int, int], bin_size: int = 200, ax=None):
    density, y_edges, x_edges = cell_density_map(df, image_shape, bin_size)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        density,
        extent=[x_edges[0], x_edges[-1], y_edges[-1], y_edges[0]],
        cmap="viridis",
    )
    plt.colorbar(im, ax=ax, label="cells / bin")
    ax.set_title(f"Cell density ({bin_size}px bins)")
    return ax


def flag_outlier_bins(
    df: pd.DataFrame,
    image_shape: tuple[int, int],
    bin_size: int = 200,
    low_pct: float = 5,
    high_pct: float = 95,
) -> pd.DataFrame:
    """
    Returns one row per spatial bin whose cell density (cells / actual pixel
    area covered) is a low/high outlier relative to the rest of the image,
    sorted worst-first -- candidates for a closer visual check. `low_density`
    = likely missed cells (e.g. dim region the network under-detected);
    `high_density` = likely over-segmentation/noise fragments.

    Density is normalized by each bin's *actual* pixel area rather than raw
    count, since bins along the bottom/right edge are clipped by the image
    boundary (image dimensions are rarely an exact multiple of `bin_size`)
    and would otherwise look artificially low-density.
    """
    h, w = image_shape
    density, y_edges, x_edges = cell_density_map(df, image_shape, bin_size)

    bin_h = np.minimum(y_edges[1:], h) - y_edges[:-1]
    bin_w = np.minimum(x_edges[1:], w) - x_edges[:-1]
    bin_area = np.outer(bin_h, bin_w)  # (n_bins_y, n_bins_x), in px^2
    density_per_area = density / bin_area

    nonzero = density_per_area[density > 0]
    low_thresh = np.percentile(nonzero, low_pct)
    high_thresh = np.percentile(nonzero, high_pct)

    ii, jj = np.nonzero((density_per_area <= low_thresh) | (density_per_area >= high_thresh))
    rows = []
    for i, j in zip(ii, jj):
        rows.append(
            {
                "bin_y": i,
                "bin_x": j,
                "count": density[i, j],
                "density_per_area": density_per_area[i, j],
                "y0": int(y_edges[i]),
                "y1": int(min(y_edges[i + 1], h)),
                "x0": int(x_edges[j]),
                "x1": int(min(x_edges[j + 1], w)),
                "flag": "low_density" if density_per_area[i, j] <= low_thresh else "high_density",
            }
        )
    return pd.DataFrame(rows).sort_values("density_per_area").reset_index(drop=True)


def crop_region(image: np.ndarray, masks: np.ndarray, y0: int, y1: int, x0: int, x1: int):
    """Crop both the intensity image (C, H, W) and the label mask (H, W) to the same window."""
    return image[:, y0:y1, x0:x1], masks[y0:y1, x0:x1]


def channels_to_rgb(image: np.ndarray, colors: list[str], percentile: float = 99.5) -> np.ndarray:
    """Additive false-color blend of a (C, H, W) intensity crop into an (H, W, 3) RGB array in [0, 1].

    Each channel is scaled by its own percentile (not a shared/global one) so a
    small crop with a dim channel doesn't just disappear.
    """
    rgb = np.zeros((*image.shape[1:], 3), dtype=np.float32)
    for c, color in enumerate(colors):
        chan = image[c].astype(np.float32)
        hi = np.percentile(chan, percentile)
        chan = np.clip(chan / max(hi, 1e-6), 0, 1)
        rgb += chan[..., None] * np.array(_COLOR_RGB[color], dtype=np.float32)
    return np.clip(rgb, 0, 1)


def plot_flagged_crops(
    image: np.ndarray,
    masks: np.ndarray,
    flagged: pd.DataFrame,
    colors: list[str],
    n: int = 6,
    ncols: int = 3,
):
    """
    Grid of the `n` worst rows of `flagged` (see `flag_outlier_bins`), each shown as a
    false-color blend with mask boundaries drawn on top (`skimage.mark_boundaries`).

    Pure matplotlib, no napari -- safe to call repeatedly / for many regions at once,
    unlike spinning up a new `napari.Viewer()` per crop (each one is a full Qt/OpenGL
    window; a handful of those alongside the whole-mosaic viewer is a common way to
    exhaust GPU memory and crash the kernel).
    """
    n = min(n, len(flagged))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)

    for idx in range(nrows * ncols):
        ax = axes[idx // ncols, idx % ncols]
        ax.axis("off")
        if idx >= n:
            continue
        row = flagged.iloc[idx]
        crop_img, crop_masks = crop_region(image, masks, row.y0, row.y1, row.x0, row.x1)
        rgb = channels_to_rgb(crop_img, colors)
        overlay = mark_boundaries(rgb, crop_masks, color=(1, 1, 1), mode="thick")
        ax.imshow(overlay)
        n_cells = len(np.unique(crop_masks)) - (1 if 0 in crop_masks else 0)
        ax.set_title(f"{row.flag}, y[{row.y0}:{row.y1}] x[{row.x0}:{row.x1}], {n_cells} cells", fontsize=9)

    fig.tight_layout()
    return fig

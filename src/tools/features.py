"""
Per-cell morphology + fluorescence feature extraction from a Cellpose label
mask and its corresponding multi-channel intensity image.

Usage
-----
    df = extract_cell_features(img, masks, channel_names=['HAND1', 'BMP4', 'T', 'Hoechst'])

`df` has one row per cell (indexed by mask label / cell_id), with:
  - centroid_y, centroid_x   -- pixel coordinates, for spatial graph construction
  - morphology columns       -- area, perimeter, circularity, eccentricity, ...
  - '{channel}_mean' / '{channel}_sum'  -- per-channel marker intensity
"""

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve
from skimage.measure import regionprops_table

MORPHOLOGY_PROPERTIES = [
    "label",  # integer label of region
    "centroid",  # centroid coordinate tuple (y, x)
    "area",  # total number of pixels in region
    "perimeter",  # boundary length tracing pixel edges
    "perimeter_crofton",  # grid-bias corrected perimeter approximation
    "eccentricity",  # elongation: 0 (circle) to 1 (line segment)
    "solidity",  # boundary smoothness/convexity: area / convex_area
    "extent",  # bounding box filling ratio: area / bounding_box_area
    "major_axis_length",  # length of major axis of equivalent ellipse
    "minor_axis_length",  # length of minor axis of equivalent ellipse
    "orientation",  # angle of major axis relative to x-axis (-pi/2 to pi/2)
    "equivalent_diameter_area",  # diameter of circle with same area
]


def extract_cell_features(
    image: np.ndarray,
    masks: np.ndarray,
    channel_names: list[str],
) -> pd.DataFrame:
    """
    Parameters
    ----------
    image : (C, H, W) intensity image, same footprint as `masks`
    masks : (H, W) integer label image from Cellpose (0 = background)
    channel_names : names for each of the C channels, e.g. ['HAND1', 'BMP4', 'T', 'Hoechst']

    Returns
    -------
    pd.DataFrame indexed by cell_id (the mask label), one row per cell.
    """
    if image.shape[0] != len(channel_names):
        raise ValueError(
            f"image has {image.shape[0]} channels but got {len(channel_names)} channel_names"
        )

    intensity = np.moveaxis(image, 0, -1)  # (H, W, C) for multichannel regionprops

    props = regionprops_table(
        masks,
        intensity_image=intensity,
        properties=MORPHOLOGY_PROPERTIES + ["intensity_mean"],
    )
    df = pd.DataFrame(props)

    df = df.rename(
        columns={
            "label": "cell_id",
            "centroid-0": "centroid_y",
            "centroid-1": "centroid_x",
            "equivalent_diameter_area": "equivalent_diameter",
        }
    )

    # Circularity: 1.0 for a perfect circle, lower for elongated/irregular shapes.
    # Use the Crofton perimeter (less biased on pixelated boundaries) when available.
    perimeter = df["perimeter_crofton"].replace(0, np.nan)
    df["circularity"] = (4 * np.pi * df["area"] / perimeter**2).fillna(0.0).clip(upper=1.0)

    # Aspect ratio: ~1 for round/compact shapes, large for thin/elongated ones
    # (e.g. crescent-shaped "gap between cells" segmentation artifacts).
    minor_axis = df["minor_axis_length"].replace(0, np.nan)
    df["aspect_ratio"] = (df["major_axis_length"] / minor_axis).fillna(1.0)

    # Per-channel intensity: regionprops_table expands multichannel `intensity_mean`
    # into `intensity_mean-0`, `intensity_mean-1`, ... in channel order.
    for c, name in enumerate(channel_names):
        mean_col = f"intensity_mean-{c}"
        df[f"{name}_mean"] = df[mean_col]
        df = df.drop(columns=[mean_col])

    df = df.set_index("cell_id")
    return df


def filter_cells(
    df: pd.DataFrame,
    min_thresholds: dict[str, float] | None = None,
    max_thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Combinatorial filter over any columns of a cell feature table -- a cell is kept
    only if it satisfies every given bound. `min_thresholds`/`max_thresholds` map a
    column name to an inclusive lower/upper bound; either may be omitted or partial.

    e.g. to drop small fragments and thin/concave "gap between cells" artifacts
    (real nuclei are close to convex and roughly round; a crescent-shaped sliver
    along a cell-cell boundary has low solidity/circularity and a high aspect ratio):

        filter_cells(
            df,
            min_thresholds={"area": 100, "solidity": 0.85, "circularity": 0.5},
            max_thresholds={"aspect_ratio": 3.0},
        )
    """
    min_thresholds = min_thresholds or {}
    max_thresholds = max_thresholds or {}
    keep = pd.Series(True, index=df.index)
    for col, lo in min_thresholds.items():
        keep &= df[col] >= lo
    for col, hi in max_thresholds.items():
        keep &= df[col] <= hi
    return df[keep]


def radius_total_signal(
    image: np.ndarray,
    df: pd.DataFrame,
    radius: float,
    channel_names: list[str],
    cell_mask: np.ndarray | dict[str, np.ndarray] | None = None,
    use_mask: bool = False,
) -> pd.DataFrame:
    """
    Per-cell TOTAL fluorescence within a `radius`-pixel disk centered on each
    cell's centroid ('centroid_y'/'centroid_x' columns of `df`) -- a patch-level
    signal aggregate, in contrast to a k-NN graph (`tools.morphology.knn_edge_index`)
    which only looks at a fixed NUMBER of nearest neighbors regardless of how much
    total signal actually surrounds a cell.

    If `use_mask` is True, only pixels inside a real, kept cell contribute (sums
    `image * cell_mask` instead of raw `image`), so background fluorescence between
    cells never counts toward "local signal". `cell_mask` is either a single (H, W)
    boolean applied to every channel, or a {channel_name: (H, W) boolean} dict for a
    different mask per channel -- e.g. restricting each channel to only its own
    IF-positive cells, via `cells_mask_from_labels(label_mask, df.index[df[f'{{c}}_mean'] != 0])`
    per channel (see `tools.morphology.cells_mask_from_labels`). Required (either form)
    when `use_mask=True`.

    Implemented as one FFT convolution per channel with a disk-shaped (1/0, not
    normalized -- this is a TOTAL, not an average) kernel, giving the local sum at
    every pixel in O(H*W log(H*W)) regardless of `radius` or cell count, then
    sampled at each cell's centroid pixel -- much faster than looping over cells
    and summing a radius**2 crop directly.

    Returns
    -------
    pd.DataFrame indexed like `df`, with one '{channel}_total' column per channel.
    """
    if use_mask and cell_mask is None:
        raise ValueError("cell_mask is required when use_mask=True")

    offsets = np.arange(-int(radius), int(radius) + 1)
    yy, xx = np.meshgrid(offsets, offsets)
    disk = (yy**2 + xx**2 <= radius**2).astype(np.float32)

    cy = df["centroid_y"].to_numpy().round().astype(int).clip(0, image.shape[1] - 1)
    cx = df["centroid_x"].to_numpy().round().astype(int).clip(0, image.shape[2] - 1)

    result = pd.DataFrame(index=df.index)
    for c, name in enumerate(channel_names):
        channel = image[c].astype(np.float32)
        if use_mask:
            mask = cell_mask[name] if isinstance(cell_mask, dict) else cell_mask
            channel = channel * mask
        local_total = fftconvolve(channel, disk, mode="same")
        result[f"{name}_total"] = local_total[cy, cx]
    return result

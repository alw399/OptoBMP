"""
Synthetic Cellpose-like label masks + cell_df feature tables, for testing/
benchmarking the neighbor-prediction pipeline (models.predictors/models.gat) against data
with KNOWN, controllable spatial structure, instead of only real microscopy data.

Real reference (this project's actual data, `20260717T120225_pA_Activin_D4_48hr_
W0001`): ~8800 cells in a 1998x1998px image; `area` mean=381px^2 (median 368, std
166) -> implied radius (`area = pi * r^2`) mean~10.7px (median 10.8, std 2.5px);
16% background (84% of pixels belong to some nucleus). `RADIUS_MEAN_DEFAULT`/
`SPACING_DEFAULT` below are calibrated to this -- but NOT via the real data's raw
nearest-neighbor centroid distance (~17px). That number is arrangement-dependent:
a regular hex lattice is the MOST uniform packing possible at a given density, so
it necessarily has a LARGER nearest-neighbor distance than a naturally irregular
real point pattern at the same density (real nuclei cluster closer in some spots
and leave bigger gaps elsewhere; a perfect grid can't reproduce that unevenness).
What IS arrangement-independent is the mean Voronoi tile area, `image_area /
n_cells` -- that's a fixed identity regardless of how regular or irregular the
packing is. Solving `tile_area = (sqrt(3)/2) * spacing^2` for this project's real
tile_area (453.9px^2) gives `SPACING_DEFAULT ~= 22.9px`, noticeably bigger than
the raw ~17px nearest-neighbor number -- using 17px directly would already run
out of room for cells sized like the real data's before any radius is applied at
all (i.e. cells would overlap enough to be Voronoi-capped well below their
intended size, independent of `radius_mean`; verified this by sweeping radius at
spacing=17 and finding area saturates around ~250px^2 no matter how large
`radius_mean` gets). Even at the corrected density-implied spacing, a "perfect"
hex grid still won't match every real statistic AT ONCE -- `radius_mean=11.0`
reproduces real mean area almost exactly (380 vs 381) but with background ~30%
instead of 16%; `radius_mean=12.0` gets closer on background (~20%) but
overshoots area (~437). `RADIUS_MEAN_DEFAULT=10.7` below prioritizes matching
real cell SIZE (the more directly meaningful/tunable dial) over background
fraction -- both are real, geometrically-forced trade-offs of an idealized
regular-grid model, not simulator bugs, and jittering (`radius_std`/
`position_jitter_std` > 0) moves further away from all of these idealized-grid
numbers anyway, same as real data deviates from a perfect lattice.

Label masks use the SAME convention as a real Cellpose `.npy` mask (background=0,
integer labels 1..N). The returned cell_df is indexed by `cell_id` (= the label
value) with the SAME geometry columns as a real `..._features.parquet` (area,
perimeter, perimeter_crofton, eccentricity, solidity, extent, major/
minor_axis_length, orientation, equivalent_diameter, circularity, aspect_ratio),
measured directly from the rasterized mask via the same `regionprops_table`
properties list and the same circularity/aspect_ratio formulas
`tools.features.extract_cell_features` uses on real masks -- kept in sync by hand
rather than a shared function, since this module never has an intensity image to
also derive `_mean` columns from. There's no simulated fluorescence here at all;
add marker columns yourself downstream if a notebook needs them.

Only SIZE and XY POSITION are ever jittered (`radius_std`/`position_jitter_std`)
-- shape stays circular in both modes; this is not about introducing elongated/
irregular cells, just size and placement variability. Cells never overlap: a
pixel is assigned to whichever cell's CENTER is nearest among cells whose own
disk reaches that pixel (a Voronoi-style tie-break via `cKDTree`, not "whichever
circle got drawn last"), the same way a real segmentation always partitions
space non-overlappingly.
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from skimage.measure import regionprops_table

from tools.features import MORPHOLOGY_PROPERTIES

# Calibrated against this project's real data -- see module docstring for why
# SPACING_DEFAULT is derived from cell DENSITY, not the real data's raw
# nearest-neighbor centroid distance.
RADIUS_MEAN_DEFAULT = 10.7  # px
SPACING_DEFAULT = 22.9  # px, center-to-center


def hex_grid_centers(
    image_shape: tuple[int, int], spacing: float, margin: Optional[float] = None
) -> np.ndarray:
    """
    Cell centers on a hexagonal (triangular) grid filling `image_shape`, `spacing`
    px apart center-to-center -- denser/more tissue-like than a square grid, and
    a reasonable approximation of how a confluent cell monolayer actually packs.

    `margin` (default `spacing`) keeps every center at least this far from the
    image edge, so border cells aren't obviously half-clipped relative to
    interior ones.

    Returns (N, 2) float64 array of (y, x) centers.
    """
    h, w = image_shape
    margin = spacing if margin is None else margin
    row_spacing = spacing * np.sqrt(3) / 2

    centers = []
    row = 0
    y = margin
    while y <= h - margin:
        x_offset = (spacing / 2) if row % 2 else 0.0
        x = margin + x_offset
        while x <= w - margin:
            centers.append((y, x))
            x += spacing
        y += row_spacing
        row += 1
    return np.asarray(centers, dtype=np.float64)


def _rasterize_disks(image_shape: tuple[int, int], centers: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """
    Draws N circular disks (`centers[i]`, radius `radii[i]`) into one (H, W)
    int32 label mask (background=0, labels 1..N), never letting two disks claim
    the same pixel: for each disk, a candidate pixel (within `radii[i]` of
    `centers[i]`) is only actually labeled if `centers[i]` is the GLOBALLY
    nearest center to that pixel (checked via a `cKDTree` built once over every
    center) -- a deterministic, order-independent rule, unlike naively
    overwriting overlapping circles in a loop.
    """
    h, w = image_shape
    label_mask = np.zeros((h, w), dtype=np.int32)
    tree = cKDTree(centers)

    for i in range(len(centers)):
        cy, cx = centers[i]
        r = radii[i]
        y0, y1 = max(0, int(np.floor(cy - r))), min(h, int(np.ceil(cy + r)) + 1)
        x0, x1 = max(0, int(np.floor(cx - r))), min(w, int(np.ceil(cx + r)) + 1)
        if y0 >= y1 or x0 >= x1:
            continue  # disk entirely off-canvas

        yy, xx = np.mgrid[y0:y1, x0:x1]
        own_disk = np.hypot(yy - cy, xx - cx) <= r
        if not own_disk.any():
            continue

        pts = np.column_stack([yy[own_disk], xx[own_disk]])
        _, nearest_idx = tree.query(pts)
        claimed = nearest_idx == i
        label_mask[yy[own_disk][claimed], xx[own_disk][claimed]] = i + 1  # 1-indexed, matches Cellpose

    return label_mask


def _morphology_from_mask(label_mask: np.ndarray) -> pd.DataFrame:
    """Geometry-only cell_df from a label mask -- same properties list and
    circularity/aspect_ratio formulas as `tools.features.extract_cell_features`
    uses on real Cellpose masks."""
    props = regionprops_table(label_mask, properties=MORPHOLOGY_PROPERTIES)
    df = pd.DataFrame(props).rename(
        columns={
            "label": "cell_id",
            "centroid-0": "centroid_y",
            "centroid-1": "centroid_x",
            "equivalent_diameter_area": "equivalent_diameter",
        }
    )
    perimeter = df["perimeter_crofton"].replace(0, np.nan)
    df["circularity"] = (4 * np.pi * df["area"] / perimeter**2).fillna(0.0).clip(upper=1.0)
    minor_axis = df["minor_axis_length"].replace(0, np.nan)
    df["aspect_ratio"] = (df["major_axis_length"] / minor_axis).fillna(1.0)
    return df.set_index("cell_id")


def simulate_cells(
    image_shape: tuple[int, int] = (1998, 1998),
    radius_mean: float = RADIUS_MEAN_DEFAULT,
    radius_std: float = 0.0,
    spacing: Optional[float] = None,
    position_jitter_std: float = 0.0,
    margin: Optional[float] = None,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Builds one synthetic (label_mask, cell_df) pair, in the style of a real
    Cellpose segmentation + `tools.features.extract_cell_features` run.

    `radius_std=0` and `position_jitter_std=0` (the defaults) give a "perfect"
    layout: an evenly-spaced hex grid of identical, perfectly round cells. Set
    either > 0 to jitter cell SIZE and/or XY POSITION respectively, each drawn
    from its own independent Gaussian -- `radius_mean`/`radius_std` for size
    (clipped at a small positive minimum so an unlucky draw can't produce a
    zero/negative-radius cell), `position_jitter_std` for a per-cell (dy, dx)
    offset applied to each hex-grid center.

    `spacing` (center-to-center grid spacing) defaults to scale with
    `radius_mean` at the same ratio as this project's real cell density
    (`SPACING_DEFAULT / RADIUS_MEAN_DEFAULT` ~= 2.1x radius -- see module
    docstring for why this is derived from density rather than the real data's
    raw nearest-neighbor distance). Shrink it for more tightly-packed/touching
    cells, grow it for a sparser layout. Small
    `spacing` combined with large `position_jitter_std`/`radius_std` can cause
    heavy overlap; overlapping cells don't disappear (the nearest-center rule
    always gives each one SOME area) but can end up much smaller than
    `radius_mean` implies.

    Returns
    -------
    label_mask : (H, W) int32 array, 0 = background, 1..N = cell labels.
    cell_df : DataFrame indexed by `cell_id` (matching `label_mask`'s label
        values), with the same geometry columns as a real
        `..._features.parquet` -- no marker/`_mean` columns, since there's no
        simulated fluorescence here.
    """
    rng = np.random.default_rng(seed)
    if spacing is None:
        spacing = radius_mean * (SPACING_DEFAULT / RADIUS_MEAN_DEFAULT)

    centers = hex_grid_centers(image_shape, spacing, margin)
    n = len(centers)

    if position_jitter_std > 0:
        centers = centers + rng.normal(0.0, position_jitter_std, size=centers.shape)
    if radius_std > 0:
        radii = np.clip(rng.normal(radius_mean, radius_std, size=n), 1.0, None)
    else:
        radii = np.full(n, radius_mean)

    label_mask = _rasterize_disks(image_shape, centers, radii)
    cell_df = _morphology_from_mask(label_mask)
    return label_mask, cell_df


def jitter_real_cells(
    label_mask: np.ndarray,
    cell_df: pd.DataFrame,
    position_jitter_std: float,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Takes a REAL segmentation's cell centers (`cell_df`'s `centroid_y`/
    `centroid_x`) and per-cell radius implied by its real `area` (`r =
    sqrt(area / pi)`, i.e. treating each irregular real nucleus as a circle of
    equal area), adds iid Gaussian XY jitter (std `position_jitter_std`, mean
    0) to every center, and re-rasterizes into a brand new label mask --
    same `image_shape` as `label_mask`, same nearest-center Voronoi tie-break
    as `simulate_cells`/`_rasterize_disks`.

    This changes each cell's POSITION (and, as a side effect of the Voronoi
    re-tiling, its exact pixel-level size/shape) while every OTHER column of
    `cell_df` -- markers, `_mean`/`+` columns, etc. -- travels with that same
    cell unchanged. That's the point: simulating "what if these exact cells,
    with these exact marker readouts, had been arranged slightly
    differently", rather than simulating new cells from scratch the way
    `simulate_cells` does.

    The output mask is relabeled so pixel value == the surviving cell's
    original `cell_df` index (same convention as the input), not the
    positional 1..N label `_rasterize_disks` assigns internally. Any cell
    whose jittered disk ends up with zero surviving pixels (fully occluded by
    a neighbor that jittered on top of it) is dropped from the returned
    `cell_df` -- a warning is printed with the count, since a high drop rate
    means `position_jitter_std` is too large relative to cell spacing/size
    for the result to be trustworthy.

    Returns
    -------
    jittered_label_mask : (H, W) int32 array, 0 = background, other values =
        the SAME `cell_id`s as `cell_df.index` (a subset, if any were dropped).
    jittered_cell_df : DataFrame indexed by `cell_id` (matching
        `jittered_label_mask`), with recomputed geometry columns (from the new
        rasterized shapes) plus every other original `cell_df` column carried
        over unchanged from the matching input cell.
    """
    if position_jitter_std < 0:
        raise ValueError(f"position_jitter_std must be >= 0, got {position_jitter_std}")

    rng = np.random.default_rng(seed)
    cell_ids = cell_df.index.to_numpy()
    centers = cell_df[["centroid_y", "centroid_x"]].to_numpy(dtype=np.float64)
    radii = np.sqrt(cell_df["area"].to_numpy(dtype=np.float64) / np.pi)

    if position_jitter_std > 0:
        centers = centers + rng.normal(0.0, position_jitter_std, size=centers.shape)

    positional_mask = _rasterize_disks(label_mask.shape, centers, radii)  # positional labels 1..N

    lut = np.zeros(len(cell_ids) + 1, dtype=positional_mask.dtype)
    lut[1:] = cell_ids
    jittered_label_mask = lut[positional_mask]  # relabel: pixel value == original cell_id

    geom_df = _morphology_from_mask(jittered_label_mask)  # index already real cell_id, post-relabel
    n_dropped = len(cell_ids) - len(geom_df)
    if n_dropped:
        print(
            f"jitter_real_cells: {n_dropped} / {len(cell_ids)} cells lost all pixels at "
            f"position_jitter_std={position_jitter_std} and were dropped"
        )

    carry_cols = [c for c in cell_df.columns if c not in geom_df.columns]
    jittered_cell_df = geom_df.join(cell_df[carry_cols])
    return jittered_label_mask, jittered_cell_df


def build_scaled_feature_frame(
    label_mask: np.ndarray,
    cell_df: pd.DataFrame,
    if_cols: list[str],
    global_x_cols: list[str],
    neighbor_x_cols: list[str],
    y_cols: list[str],
    pred_radius: float,
    density_radius: float,
    scaling: str,
    normalize_data: bool,
    seed: int,
) -> pd.DataFrame:
    """
    Runs the same `models.data.build_prediction_data` pipeline `sender_predict.ipynb` uses
    for actual training, but only to recover its SCALED feature columns (plus
    `centroid_y`/`centroid_x`) as a plain DataFrame -- for a "does this look different
    before vs. after jitter" spatial-scatter comparison, not for training a model.

    Meant to be called TWICE with the SAME hyperparameters -- once on a real
    `(label_mask, cell_df)` pair, once on its `jitter_real_cells` output -- so both
    results share identical scaling/train-split logic and are directly comparable
    (e.g. with a shared `vmin`/`vmax` per column across both calls' outputs).

    Imports the model modules/torch lazily, so importing `tools.simulated` itself stays free
    of a torch_geometric dependency for callers who only need `simulate_cells`/
    `jitter_real_cells`.
    """
    from models.graph import border_mask
    from models.data import build_prediction_data
    from tools.morphology import local_density, split_indices

    cell_df = cell_df.copy()
    centroids = cell_df[["centroid_y", "centroid_x"]].to_numpy(dtype=np.float32)
    cell_df["local_density"] = local_density(centroids, radius=density_radius)

    is_border = border_mask(centroids, label_mask.shape, pred_radius)
    valid_target_idx = np.where(~is_border)[0]
    train_pos, _, _ = split_indices(len(valid_target_idx), val_frac=0.15, test_frac=0.15, seed=seed)
    train_idx = valid_target_idx[train_pos]

    pred_data = build_prediction_data(
        cell_df, radius=pred_radius,
        global_x_cols=global_x_cols, neighbor_x_cols=neighbor_x_cols, y_cols=y_cols,
        train_idx=train_idx, scaling=scaling,
        no_background_cols=["local_density"], no_transform_cols=["local_density"],
        normalize_data=normalize_data,
    )

    frame = pd.DataFrame(pred_data["y"], columns=pred_data["y_cols"])
    for i, col in enumerate(pred_data["global_x_cols"]):
        frame[col] = pred_data["x_global"][:, i]
    for i, col in enumerate(pred_data["neighbor_x_cols"]):
        frame[col] = pred_data["x_neighbor"][:, i]  # overwrites same-named global_x_cols column
    frame["centroid_x"] = cell_df["centroid_x"].values
    frame["centroid_y"] = cell_df["centroid_y"].values
    return frame

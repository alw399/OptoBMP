"""
Assembling a training-ready dict: the graph, the scaled inputs and the scaled target.

`build_prediction_data` handles one image; `build_multi_image_prediction_data`
concatenates several (one independent graph each -- cells in different wells are
never physically adjacent, so never connected).
"""

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

from models.graph import build_radius_graph
from models.scalers import _fit_hurdle_y, _scale_columns

def build_prediction_data(
    df: pd.DataFrame,
    radius: float,
    x_cols: Sequence[str],
    y_cols: Sequence[str],
    min_neighbors: int = 1,
    length_scale: Optional[float] = None,
    normalize_data: bool = True,
    train_idx: Optional[np.ndarray] = None,
    scaling: str = "log1p_standard",
    arcsinh_cofactor: float = 5.0,
    subtract_background: bool = True,
    no_background_cols: Sequence[str] = (),
    no_transform_cols: Sequence[str] = (),
    no_scale_cols: Sequence[str] = (),
    custom_background: Optional[dict[str, float]] = None,
    two_part: bool = False,
) -> dict:
    """
    Builds the (self-inclusive) radius graph (`build_radius_graph`) plus scaled x/y
    tensors for `IFPredictor` (or `models.gat.IFPredictorGAT`, same tensors).
    `scaling`/`arcsinh_cofactor`/`subtract_background` select how each column is
    scaled -- see `ColumnScaler`.

    `normalize_data=False`: see `build_radius_graph`/`normalize_distance_weights`
    -- an alternative to an explicit `local_density` feature, letting the GNN see
    neighbor count/density implicitly via a weighted-SUM aggregation instead of a
    weighted-average one. If used with `IFPredictor`, pass this SAME value to its
    `normalize_data` constructor arg too -- see its docstring.

    `no_background_cols`: column names to exempt from `subtract_background`
    even when it's True for the rest of the group -- e.g. `local_density`,
    which unlike an intensity channel has no real "background" floor to
    subtract (see `ColumnScaler`'s per-column `subtract_background` support).

    `no_transform_cols`: column names to exempt from the `scaling` mode's
    log1p/arcsinh nonlinear step (still standardized, just not transformed
    first) -- e.g. `local_density` again: log1p exists to pull in a long
    right tail from multiplicative noise, which a density count doesn't have
    (measured skew ~0.3, sometimes negative), so it only makes the
    distribution more skewed, not less.

    `no_scale_cols`: column names to skip the FINAL StandardScaler/RobustScaler
    step for entirely (see `ColumnScaler`'s per-column `apply_scaling` support) --
    independent of `no_background_cols`/`no_transform_cols`, combine as needed.
    Meant for an already-binary {0, 1} column (e.g. a thresholded `{channel}+`
    cast to float): standardizing it can hurt rather than help, since a skewed
    positive rate gives a small `std`, and dividing by a small `std` inflates
    the rarer class into an outlier-sized value purely from class imbalance,
    not real signal -- the opposite of what standardization is supposed to do.

    `custom_background` ({column_name: value}, default None): overrides
    `estimate_background`'s automatic bimodal-threshold estimate for whichever
    columns are given, using the supplied value directly instead -- for a
    channel where that automatic estimate doesn't hold up for a particular
    system/dataset (no clean bimodal split, or a train slice degenerate enough
    to make its `gaussian_kde` call raise). Only affects the columns named as
    keys; every other column still auto-estimates as usual. Takes priority over
    `subtract_background`/`no_background_cols` for that column -- supplying an
    override IS the decision to subtract that value, so a column can't be both
    overridden and exempted from background subtraction. `tools.qc.
    get_bimodal_threshold` (run yourself, with whatever settings actually work
    for that channel) is one reasonable way to compute the value, but any float
    you trust for that channel's background level is fine. For a `two_part`
    `y_col`, this SAME dict doubles as the hurdle threshold override -- see
    `_fit_hurdle_y`'s docstring for why it's the same mechanism rather than a
    second one.

    leakage, not a deliberate choice the way it briefly was under the old
    `global_x_cols`/`neighbor_x_cols` split.

    `two_part` (default False): use `HurdleScaler` instead of `ColumnScaler` for
    `y_cols`, and also compute+return `y_positive` -- for `IFPredictor`/
    `IFPredictorGAT`'s `two_part` hurdle-model option (see `TwoPartHead`'s and
    `HurdleScaler`'s docstrings). Meant for a `y_col` that's heavily zero-inflated
    (most cells at background, a small real positive population) rather than
    smoothly continuous -- e.g. this project's Sox17_mean. When True,
    `no_background_cols`/`no_transform_cols` still apply to `y_cols`, now as
    `HurdleScaler`'s per-column `apply_background`/`apply_transform` (see its
    docstring for what `no_background_cols` means there specifically -- floors
    at raw 0 instead of the classification threshold, NOT "no floor at all").
    `subtract_background`/`no_scale_cols` have no equivalent in `HurdleScaler`
    (the floor is the threshold itself, always on; the [0, ~1] output range
    isn't optional the way `ColumnScaler`'s final standardization step is) --
    both are simply UNUSED for `y_cols` under `two_part=True`. `custom_
    background` still applies, as the per-column threshold override (see above).
    `x_cols` are entirely unaffected by `two_part` either way.

    Scalers are fit on `train_idx` (positions, not labels) if given, else the whole
    of `df` -- pass the training split's indices to avoid leakage from val/test cells.
    """
    x_cols = list(x_cols)
    y_cols = list(y_cols)
    overlap = set(x_cols) & set(y_cols)
    if overlap:
        raise ValueError(f"x_cols/y_cols overlap on {sorted(overlap)} -- would leak a column into predicting itself")

    centroids = df[["centroid_y", "centroid_x"]].to_numpy(dtype=np.float32)
    edge_index, edge_weight = build_radius_graph(centroids, radius, min_neighbors, length_scale, normalize_data)

    fit_slice = df if train_idx is None else df.iloc[train_idx]
    x, x_scaler = _scale_columns(
        df, fit_slice, x_cols, scaling, arcsinh_cofactor, subtract_background,
        no_background_cols, no_transform_cols, no_scale_cols, custom_background,
    )

    result = dict(edge_index=edge_index, edge_weight=edge_weight, x=torch.from_numpy(x), x_scaler=x_scaler, x_cols=x_cols, y_cols=y_cols)

    if two_part:
        y, y_positive, y_scaler = _fit_hurdle_y(
            df, fit_slice, y_cols, scaling, arcsinh_cofactor, custom_background, no_background_cols, no_transform_cols,
        )
        result["y_positive"] = y_positive
        result["y"] = torch.from_numpy(y)
        result["y_scaler"] = y_scaler
        return result

    y, y_scaler = _scale_columns(
        df, fit_slice, y_cols, scaling, arcsinh_cofactor, subtract_background,
        no_background_cols, no_transform_cols, no_scale_cols, custom_background,
    )

    result["y"] = torch.from_numpy(y)
    result["y_scaler"] = y_scaler
    return result


def build_multi_image_prediction_data(
    cell_dfs: Sequence[pd.DataFrame],
    image_shapes: Sequence[tuple[int, int]],
    radius: float,
    x_cols: Sequence[str],
    y_cols: Sequence[str],
    min_neighbors: int = 1,
    length_scale: Optional[float] = None,
    normalize_data: bool = True,
    train_idx: Optional[np.ndarray] = None,
    scaling: str = "log1p_standard",
    arcsinh_cofactor: float = 5.0,
    subtract_background: bool = True,
    no_background_cols: Sequence[str] = (),
    no_transform_cols: Sequence[str] = (),
    no_scale_cols: Sequence[str] = (),
    custom_background: Optional[dict[str, float]] = None,
    two_part: bool = False,
) -> dict:
    """
    Like `build_prediction_data`, but trains across MULTIPLE images at once
    (`notebooks/combined.ipynb`): builds an INDEPENDENT radius graph per image
    -- cells in different images/wells are never physically adjacent, so
    never connected to each other -- then concatenates all of them into one
    combined graph/feature set. `train_idx` indexes POSITIONS in
    `pd.concat(cell_dfs)` (same convention as `build_prediction_data`'s
    single-df `train_idx`, just over the concatenation). Scalers are fit ONCE
    across every image's train rows together, so one checkpoint's scaler
    applies uniformly regardless of which image a cell came from -- this is
    also why `local_density`/border exclusion must be computed per-image
    BEFORE calling this (each image has its own neighborhoods/edges), not
    recomputed here.

    `no_background_cols`/`no_transform_cols`/`no_scale_cols`/`custom_background`/
    `two_part`: same per-column exceptions/options as `build_prediction_data` --
    see its docstring. Same `x_cols`/`y_cols` overlap check as
    `build_prediction_data` too.

    `image_shapes[i]` is unused by this function directly -- it exists so
    callers who already loop over `(cell_df, image_shape)` pairs for
    `border_mask` can pass the same list through without restructuring, but
    graph-building only needs centroids. Kept as a parameter for symmetry/
    documentation of the expected per-image inputs.
    """
    assert len(cell_dfs) == len(image_shapes), "one image_shape per cell_df"
    x_cols = list(x_cols)
    y_cols = list(y_cols)
    overlap = set(x_cols) & set(y_cols)
    if overlap:
        raise ValueError(f"x_cols/y_cols overlap on {sorted(overlap)} -- would leak a column into predicting itself")

    edge_index_parts, edge_weight_parts = [], []
    offset = 0
    for cell_df in cell_dfs:
        centroids = cell_df[["centroid_y", "centroid_x"]].to_numpy(dtype=np.float32)
        ei, ew = build_radius_graph(centroids, radius, min_neighbors, length_scale, normalize_data)
        edge_index_parts.append(ei + offset)
        edge_weight_parts.append(ew)
        offset += len(cell_df)
    edge_index = torch.cat(edge_index_parts, dim=1)
    edge_weight = torch.cat(edge_weight_parts)

    df = pd.concat(cell_dfs, ignore_index=True)
    fit_slice = df if train_idx is None else df.iloc[train_idx]
    x, x_scaler = _scale_columns(
        df, fit_slice, x_cols, scaling, arcsinh_cofactor, subtract_background,
        no_background_cols, no_transform_cols, no_scale_cols, custom_background,
    )

    result = dict(edge_index=edge_index, edge_weight=edge_weight, x=torch.from_numpy(x), x_scaler=x_scaler, x_cols=x_cols, y_cols=y_cols)

    if two_part:
        y, y_positive, y_scaler = _fit_hurdle_y(
            df, fit_slice, y_cols, scaling, arcsinh_cofactor, custom_background, no_background_cols, no_transform_cols,
        )
        result["y_positive"] = y_positive
        result["y"] = torch.from_numpy(y)
        result["y_scaler"] = y_scaler
        return result

    y, y_scaler = _scale_columns(
        df, fit_slice, y_cols, scaling, arcsinh_cofactor, subtract_background,
        no_background_cols, no_transform_cols, no_scale_cols, custom_background,
    )
    result["y"] = torch.from_numpy(y)
    result["y_scaler"] = y_scaler
    return result

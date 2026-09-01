"""
Turning raw immunofluorescence intensity into something a network can be trained on.

`ColumnScaler` is the general case (background-subtract, variance-stabilise,
standardise). `HurdleScaler` is the zero-inflated case that `TwoPartHead` needs:
at-or-below-threshold maps to exactly 0, above it into (0, ~1].
"""

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler, StandardScaler

from tools.qc import get_bimodal_threshold

SCALING_MODES = ("log1p_standard", "arcsinh", "robust")


def estimate_background(values: np.ndarray) -> float:
    """
    Per-channel background level: raw IF intensity is never zero for a true
    negative cell -- camera offset, tissue autofluorescence, and non-specific
    antibody binding all add a baseline every cell shares, positive or not. Taking
    log1p/arcsinh of the RAW value (background included) compresses real signal
    unevenly depending on where a cell happens to sit relative to that baseline,
    rather than starting every cell from a true zero.

    Reuses `tools.qc.get_bimodal_threshold` (the same negative/positive valley
    already used to build `cell_df`'s '{channel}+' columns) to split `values` into
    a negative (background) and positive (true signal) population, then returns
    the MEDIAN of the negative population as the background estimate. Falls back
    to the median of all `values` if no bimodal split is found (e.g. a channel
    with no clear positive population in this particular slice of cells).
    """
    series = pd.Series(np.asarray(values).ravel())
    threshold = get_bimodal_threshold(pd.DataFrame({"v": series}), "v")
    if threshold is None:
        return float(series.median())
    negative = series[series < threshold]
    # return float(negative.median()) if len(negative) > 0 else float(series.median())
    threshold = float(np.percentile(negative, 80)) if len(negative) > 0 else float(series.median())
    return threshold


class ColumnScaler:
    """
    Fluorescence intensity is heavily right-skewed by a handful of very bright
    true-positive cells (e.g. this data's `Sox17_mean`: mean=272, std=480,
    max=23266) -- a plain `StandardScaler` lets those outliers set the scale for
    every other cell. Wraps a variance-stabilizing transform with a final
    sklearn scaler, but still exposes plain `fit`/`transform`/`inverse_transform`
    (so callers like `predict_df` don't need to know which mode built it):

      'log1p_standard' (default): `log1p(x)`, then `StandardScaler`. Standard for
        imaging-based multiplexed IF (roughly log-normal intensity -- antibody
        binding/amplification is a multiplicative process).
      'arcsinh': `arcsinh(x / arcsinh_cofactor)`, then `StandardScaler` -- the
        standard transform in flow/mass cytometry. Behaves like log1p for large
        values but stays linear near zero, so background-level cells don't get
        compressed the way log can. `arcsinh_cofactor` should be tuned to the
        platform's noise floor (CyTOF-style analyses commonly use ~5).
      'robust': `RobustScaler` alone (median/IQR, no log-style transform) --
        centers/scales using statistics the outlier tail can't drag around,
        without changing the units of the data.

    Both transform-based modes still end in an explicit `StandardScaler` fit on
    the transformed TRAIN values -- not just the transform alone, which is
    sometimes used as-is in cytometry pipelines -- because `RadiusGNN`'s
    Kaiming init (`tools.morphology.init_weights`) assumes standardized input.

    `subtract_background` (default True): before any of the above, subtract each
    column's `estimate_background` (fit on TRAIN only) and clip at 0. Raw IF
    intensity is never truly zero -- camera offset, autofluorescence, non-specific
    binding -- so every cell shares a nonzero baseline regardless of real signal;
    subtracting it first means log1p/arcsinh sees actual signal-above-background
    starting from zero, instead of compressing signal unevenly depending on where
    a cell happens to sit relative to that shared baseline. Clipping at 0 (rather
    than letting background-negative noise go negative) keeps "no signal" cells
    at exactly 0 pre-transform instead of scattering them to small negative
    values with no physical meaning.

    Also accepts a per-column bool sequence (one entry per column this scaler is
    fit on) instead of a single bool for BOTH `subtract_background` and
    `apply_transform` -- needed for a column like `local_density` that's fit
    ALONGSIDE an intensity column (e.g. `Activin_mean`) in the same `x_cols`
    group but shouldn't get the same treatment:

      - a density COUNT has no "background" floor to subtract -- unlike
        intensity, values below some threshold aren't noise, they're just low
        density -- so clipping them to a shared floor destroys real variation
        instead of removing noise. `background_` is simply 0 for any column
        with subtraction off, so `_forward`/`_inverse` don't need to branch on
        this at all.
      - `apply_transform=False` skips the log1p/arcsinh nonlinear step for a
        column, leaving `StandardScaler` to standardize it directly. log1p/
        arcsinh exist to pull in a long right tail from multiplicative noise
        (real for intensity: e.g. this data's `Activin_mean` has skew ~10) --
        `local_density` has no such tail (measured skew ~0.3, sometimes even
        NEGATIVE depending on radius), so log1p only compresses the wrong end
        and makes the distribution more skewed, not less.
      - `apply_scaling=False` skips the FINAL `StandardScaler`/`RobustScaler` step
        too, leaving a column completely untouched by this point (still subject
        to whatever `subtract_background`/`apply_transform` did first -- set
        those False too for a column that should be truly raw end to end).
        Meant for an already-binary {0, 1} indicator column (e.g. a thresholded
        `{channel}+` cast to float): standardizing a column like that can
        actively hurt rather than help -- with a skewed positive rate `p`, `std =
        sqrt(p(1-p))` is small, so dividing by it inflates the rarer class into
        an outlier-sized value purely from how imbalanced the split happened to
        be, not from any real signal, which is the opposite of what
        standardization is for.

    `background_overrides` (default `None`, i.e. every column auto-estimates):
    an optional per-column sequence, one entry per column, where a non-`None`
    entry is used AS `background_` directly for that column INSTEAD OF ever
    calling `estimate_background` -- for a channel where the automatic bimodal-
    threshold estimate (`tools.qc.get_bimodal_threshold`'s `gaussian_kde` call)
    doesn't hold up for a particular system/dataset (e.g. no clear bimodal
    split, or a degenerate train slice that makes `gaussian_kde` raise). Takes
    priority over `subtract_background`/`no_background_cols` entirely for that
    column -- providing an override IS the decision to subtract that value, so
    a column can't be BOTH overridden and exempted from background subtraction.
    """

    def __init__(
        self,
        scaling: str = "log1p_standard",
        arcsinh_cofactor: float = 5.0,
        subtract_background: Union[bool, Sequence[bool]] = True,
        apply_transform: Union[bool, Sequence[bool]] = True,
        apply_scaling: Union[bool, Sequence[bool]] = True,
        background_overrides: Optional[Sequence[Optional[float]]] = None,
    ):
        if scaling not in SCALING_MODES:
            raise ValueError(f"scaling must be one of {SCALING_MODES}, got {scaling!r}")
        self.scaling = scaling
        self.arcsinh_cofactor = arcsinh_cofactor
        self.subtract_background = subtract_background
        self.apply_transform = apply_transform
        self.apply_scaling = apply_scaling
        self.background_overrides = background_overrides
        self.background_: Optional[np.ndarray] = None
        self.base_scaler = RobustScaler() if scaling == "robust" else StandardScaler()

    @staticmethod
    def _column_mask(value: Union[bool, Sequence[bool]], n_cols: int, name: str) -> np.ndarray:
        if isinstance(value, bool):
            return np.full(n_cols, value)
        mask = np.asarray(value, dtype=bool)
        assert mask.shape == (n_cols,), f"{name} must have one entry per column ({n_cols}), got {mask.shape}"
        return mask

    def _forward(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x - self.background_, a_min=0.0, a_max=None)
        transform_mask = self._column_mask(self.apply_transform, x.shape[1], "apply_transform")
        out = x.copy()
        if self.scaling == "log1p_standard":
            out[:, transform_mask] = np.log1p(x[:, transform_mask])
        elif self.scaling == "arcsinh":
            out[:, transform_mask] = np.arcsinh(x[:, transform_mask] / self.arcsinh_cofactor)
        # 'robust': no variance-stabilizing transform regardless of the mask
        return out

    def _inverse(self, x: np.ndarray) -> np.ndarray:
        transform_mask = self._column_mask(self.apply_transform, x.shape[1], "apply_transform")
        out = x.copy()
        if self.scaling == "log1p_standard":
            out[:, transform_mask] = np.expm1(x[:, transform_mask])
        elif self.scaling == "arcsinh":
            out[:, transform_mask] = np.sinh(x[:, transform_mask]) * self.arcsinh_cofactor
        return out + self.background_

    def fit(self, x: np.ndarray) -> "ColumnScaler":
        bg_mask = self._column_mask(self.subtract_background, x.shape[1], "subtract_background")
        overrides = self.background_overrides
        def _background(j: int) -> float:
            if overrides is not None and overrides[j] is not None:
                return float(overrides[j])  # bypasses estimate_background entirely -- see class docstring
            return estimate_background(x[:, j]) if bg_mask[j] else 0.0
        self.background_ = np.array([_background(j) for j in range(x.shape[1])], dtype=np.float32)
        self.base_scaler.fit(self._forward(x))

        scale_mask = self._column_mask(self.apply_scaling, x.shape[1], "apply_scaling")
        if not scale_mask.all():
            # Neither StandardScaler nor RobustScaler has a per-column skip built in --
            # force an identity transform for these columns by overwriting their fitted
            # center/scale directly, AFTER the normal fit above (so it doesn't affect
            # any other column's statistics). transform()/inverse_transform() then act
            # as a no-op for these columns with no extra branching needed there.
            center_attr = "mean_" if hasattr(self.base_scaler, "mean_") else "center_"
            getattr(self.base_scaler, center_attr)[~scale_mask] = 0.0
            self.base_scaler.scale_[~scale_mask] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return self.base_scaler.transform(self._forward(x))

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return self._inverse(self.base_scaler.inverse_transform(x))


def _scale_columns(
    df: pd.DataFrame,
    fit_slice: pd.DataFrame,
    cols: list[str],
    scaling: str,
    arcsinh_cofactor: float,
    subtract_background: bool,
    no_background_cols: Sequence[str],
    no_transform_cols: Sequence[str],
    no_scale_cols: Sequence[str] = (),
    custom_background: Optional[dict[str, float]] = None,
) -> tuple[np.ndarray, Optional[ColumnScaler]]:
    """Shared by `build_prediction_data`/`build_multi_image_prediction_data`: fits a
    `ColumnScaler` on `fit_slice[cols]` (the training rows) and transforms `df[cols]`
    (every row). `no_background_cols`/`no_transform_cols`/`no_scale_cols` carve out
    per-column exceptions to the group's otherwise-uniform `subtract_background`/
    `scaling`/final-StandardScaler-or-RobustScaler step, independently of each other.
    `custom_background` (name -> value) becomes `ColumnScaler`'s positional
    `background_overrides`, looked up by name for whichever of `cols` it applies to."""
    if not cols:
        return np.zeros((len(df), 0), dtype=np.float32), None
    bg_mask = [subtract_background and c not in no_background_cols for c in cols]
    transform_mask = [c not in no_transform_cols for c in cols]
    scale_mask = [c not in no_scale_cols for c in cols]
    bg_overrides = [(custom_background or {}).get(c) for c in cols]
    scaler = ColumnScaler(
        scaling, arcsinh_cofactor, bg_mask, transform_mask, scale_mask, bg_overrides,
    ).fit(fit_slice[cols].to_numpy(dtype=np.float32))
    return scaler.transform(df[cols].to_numpy(dtype=np.float32)).astype(np.float32), scaler


class HurdleScaler:
    """
    Scaler for a `two_part` `y_col`, paired with `TwoPartHead`. Unlike
    `ColumnScaler` (background-subtract, log1p/arcsinh, then center/scale to
    zero-mean-unit-variance -- which maps "at background" to some arbitrary
    non-zero point, and lets a bright positive land many standard deviations
    above 1, which `TwoPartHead`'s sigmoid-bounded `magnitude` head can't ever
    reach), this scaler makes "at or below `threshold_`" map to EXACTLY 0, and
    scales everything above it into (0, ~1] -- so a hurdle model's negative
    branch can just predict a combined value of 0 directly, and the positive
    branch's target actually lives in the range `TwoPartHead` assumes.

    `threshold_` is the SAME per-column cutoff `two_part`'s classification labels
    (`y_positive`) are built from (see `_fit_hurdle_y`) -- one shared boundary
    decides "is this cell positive" for classification purposes. `floor_` is
    what SCALING actually subtracts before clipping -- equal to `threshold_` by
    default, or 0 for a column with `apply_background=False` (see below).

    `fit(x, thresholds)`:
      1. `floor_ = threshold_` per column, unless `apply_background=False` for
         that column, in which case `floor_ = 0` for it instead.
      2. `clipped = clip(x - floor_, min=0)` -- exactly 0 at/below `floor_`,
         positive above it.
      3. `transformed = log1p(clipped)` or `arcsinh(clipped / arcsinh_cofactor)`
         (still maps 0 -> 0, so the floor survives the variance-stabilizing step)
         or `clipped` itself if `apply_transform=False` for that column, or
         globally if `scaling="robust"` (same convention as `ColumnScaler` --
         no log-style transform, just the raw above-floor magnitude).
      4. `scale_ = max(transformed)` over the FIT (train) rows -- the divisor
         that sends the train set's brightest positive cell to exactly 1.

    `transform(x) = transformed(clip(x - floor_, min=0)) / scale_`. Cells above
    the train set's brightest positive (e.g. in later eval data) are NOT clipped --
    they simply land above 1, same "fit on train, no forced range on eval"
    philosophy as `StandardScaler` not clipping at +-3 either.

    `inverse_transform(s) = inverse_transformed(s * scale_) + floor_` -- for a
    column with background subtraction on (the default), `s == 0` decodes back
    to exactly `threshold_`: there's no way to recover which below-threshold raw
    value a scaled 0 actually came from (that's the whole point of the hurdle
    split), so the boundary is the least-arbitrary choice of what "we called
    this background" decodes back to.

    `apply_background` (default True, per-column like `ColumnScaler`'s
    `subtract_background`): when False for a column, that column's floor
    (`floor_`) is 0 instead of `threshold_` -- `clip(x, min=0)` rather than
    `clip(x - threshold_, min=0)`. `y_positive`/the classification boundary is
    UNCHANGED either way (still `threshold_`, from `_fit_hurdle_y`) -- only
    where SCALING floors to exactly 0 shifts.

    For a column whose raw values sit well above 0 even at "background" (true
    of essentially any raw fluorescence intensity -- camera offset and
    autofluorescence mean it's never actually 0), this is NOT a narrow edge
    case: verified empirically on a synthetic zero-inflated intensity column,
    turning this off made ~95% of `y_positive`-negative cells get a nonzero
    scaled `y` (since their floor dropped from `threshold_`, which sits near
    where their raw values actually are, down to a floor of 0, which they're
    all well above regardless of true signal) -- breaking the "negative maps to
    exactly 0" invariant `TwoPartHead.combined = prob * magnitude` depends on
    for MOST of the negative population, not a handful of borderline cells.
    Reserve this for a column that genuinely has meaningful values at/near raw
    0 as its "no signal" state (e.g. `ColumnScaler`'s own motivating case,
    `local_density` -- a count, not an intensity) -- not for an IF channel.

    `apply_transform` (default True, per-column): when False for a column, the
    log1p/arcsinh nonlinear step is skipped -- the clipped value is scaled
    directly. Globally off regardless of this mask when `scaling="robust"` (same
    convention as `ColumnScaler`).
    """

    def __init__(
        self,
        scaling: str = "log1p_standard",
        arcsinh_cofactor: float = 5.0,
        apply_background: Union[bool, Sequence[bool]] = True,
        apply_transform: Union[bool, Sequence[bool]] = True,
    ):
        if scaling not in SCALING_MODES:
            raise ValueError(f"scaling must be one of {SCALING_MODES}, got {scaling!r}")
        self.scaling = scaling
        self.arcsinh_cofactor = arcsinh_cofactor
        self.apply_background = apply_background
        self.apply_transform = apply_transform
        self.threshold_: Optional[np.ndarray] = None
        self.floor_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def _clip(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x - self.floor_, a_min=0.0, a_max=None)

    def _transform_nonlinear(self, x: np.ndarray) -> np.ndarray:
        if self.scaling == "robust":
            return x
        mask = ColumnScaler._column_mask(self.apply_transform, x.shape[1], "apply_transform")
        out = x.copy()
        if self.scaling == "arcsinh":
            out[:, mask] = np.arcsinh(x[:, mask] / self.arcsinh_cofactor)
        else:
            out[:, mask] = np.log1p(x[:, mask])
        return out

    def _inverse_nonlinear(self, x: np.ndarray) -> np.ndarray:
        if self.scaling == "robust":
            return x
        mask = ColumnScaler._column_mask(self.apply_transform, x.shape[1], "apply_transform")
        out = x.copy()
        if self.scaling == "arcsinh":
            out[:, mask] = np.sinh(x[:, mask]) * self.arcsinh_cofactor
        else:
            out[:, mask] = np.expm1(x[:, mask])
        return out

    def fit(self, x: np.ndarray, thresholds: np.ndarray) -> "HurdleScaler":
        self.threshold_ = np.asarray(thresholds, dtype=np.float32)
        bg_mask = ColumnScaler._column_mask(self.apply_background, x.shape[1], "apply_background")
        self.floor_ = np.where(bg_mask, self.threshold_, 0.0).astype(np.float32)
        transformed = self._transform_nonlinear(self._clip(x))
        max_val = transformed.max(axis=0)
        bad = np.where(max_val <= 0)[0]
        if len(bad) > 0:
            raise ValueError(
                f"HurdleScaler: no TRAIN values above threshold for column index {bad.tolist()} -- "
                "can't calibrate a scale from zero positive examples"
            )
        self.scale_ = max_val.astype(np.float32)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (self._transform_nonlinear(self._clip(x)) / self.scale_).astype(np.float32)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return (self._inverse_nonlinear(np.asarray(x) * self.scale_) + self.floor_).astype(np.float32)


def _fit_hurdle_y(
    df: pd.DataFrame,
    fit_slice: pd.DataFrame,
    y_cols: list[str],
    scaling: str,
    arcsinh_cofactor: float,
    custom_background: Optional[dict[str, float]],
    no_background_cols: Sequence[str] = (),
    no_transform_cols: Sequence[str] = (),
) -> tuple[np.ndarray, torch.Tensor, HurdleScaler]:
    """
    For `build_prediction_data`'s `two_part` option: one THRESHOLD per `y_col` --
    `custom_background[col]` if given (same override dict `ColumnScaler`-scaled
    columns already use for their background floor -- see `build_prediction_data`'s
    docstring), else `tools.qc.get_bimodal_threshold` fit on `fit_slice` (train rows
    only, same leakage discipline as every other scaler/threshold in this module).
    That threshold is ALWAYS the classification boundary (`y_positive`) -- and,
    unless the column is named in `no_background_cols`, also `HurdleScaler`'s
    scaling floor (see its `apply_background` docstring for the narrow
    consequence of exempting a column from that).

    Reusing `custom_background` here (rather than a separate override dict) is
    deliberate: `get_bimodal_threshold`'s automatic estimate can land too HIGH for
    a given channel/dataset (a spurious second mode, or a valley that doesn't sit
    where the real biological cutoff does) with no single flag that detects that
    automatically -- `custom_background` is already the established escape hatch
    for exactly this kind of "the automatic estimate doesn't hold up here" case
    (see `ColumnScaler`'s docstring), so a `two_part` column reuses it instead of
    introducing a second, parallel override mechanism.

    `no_transform_cols`: per-column skip of `HurdleScaler`'s log1p/arcsinh step
    (still floored and scaled, just not log/arcsinh'd first) -- same convention
    and same motivating case (e.g. a column with no long right tail to compress)
    as `ColumnScaler`'s `no_transform_cols`.

    Raises if a column has no override AND no clear bimodal split -- `two_part=True`
    is an explicit claim that a column IS meaningfully bimodal; silently treating an
    ambiguous column as "all positive" or "all negative" would hide that assumption
    failing rather than surface it.

    Returns
    -------
    y : (N, len(y_cols)) float32 array, `HurdleScaler`-scaled -- exactly 0 at or
        below threshold, in (0, ~1] above it.
    y_positive : (N, len(y_cols)) float32 tensor, 1.0 where `df[col] > threshold`.
    y_scaler : the fitted `HurdleScaler` (same `transform`/`inverse_transform`
        interface as `ColumnScaler`, so `predict_df`/`save_model`/
        `apply_prediction_data` don't need to know which kind built it).
    """
    overrides = custom_background or {}
    thresholds = []
    for col in y_cols:
        if overrides.get(col) is not None:
            thresholds.append(float(overrides[col]))
            continue
        threshold = get_bimodal_threshold(pd.DataFrame({"v": fit_slice[col]}), "v")
        if threshold is None:
            raise ValueError(
                f"two_part=True but no bimodal split found for {col!r} on the train slice -- "
                f"pass an explicit threshold via custom_background={{{col!r}: ...}} instead"
            )
        thresholds.append(float(threshold))
    threshold_arr = np.array(thresholds, dtype=np.float32)

    y_positive = torch.from_numpy((df[y_cols].to_numpy(dtype=np.float32) > threshold_arr).astype(np.float32))

    bg_mask = [c not in no_background_cols for c in y_cols]
    transform_mask = [c not in no_transform_cols for c in y_cols]
    y_scaler = HurdleScaler(
        scaling=scaling, arcsinh_cofactor=arcsinh_cofactor, apply_background=bg_mask, apply_transform=transform_mask,
    )
    y_scaler.fit(fit_slice[y_cols].to_numpy(dtype=np.float32), threshold_arr)
    y = y_scaler.transform(df[y_cols].to_numpy(dtype=np.float32))

    return y, y_positive, y_scaler

"""
The recommended configuration, in one place.

Every constant here is a conclusion from the investigation in
`notebooks/01_predictive_power.ipynb`, not a preference.

* `MAX_SIGMA = 240`. `radius=30` x 3 layers gives each cell a ~90px receptive
  field, but the BMP4 response region is a disc of radius ~600px, and 86% of cells
  have no BMP4+ cell within 30px at all -- their input is a constant. Context
  features fix that. But pyramid levels ABOVE ~240px stop being context and start
  acting as position proxies: at sigma = 960px a feature varies over the same
  length scale as the image, so within one image it can encode "where am I", which
  is worthless in a well whose pattern sits elsewhere. On the one comparable well
  pair (W8_pattern1 <-> W9_pattern1 -- same illumination power, same imaging
  session) capping at 120-240px maximises BOTH the within-image blocked score and
  the cross-well score, while extending to 1920px keeps nudging the within-image
  number up and drops cross-well T rho from 0.44 to 0.31.

* Geometry features are kept, EXCEPT `signed_dist`. `log_dist_pos`/`log_dist_neg`
  ("how close is the nearest BMP4+/BMP4- cell") add a small consistent gain and are
  kept. `signed_dist` (same idea, but signed +inside/-outside, magnitude = distance
  to the boundary -- see `tools.spatial.signed_boundary_distance`) is deliberately
  EXCLUDED from `mask_features`'s returned x_cols: it's an unbounded raw distance
  whose scale tracks the SIZE of the illuminated pattern in a given well, so a
  scaler fit on one well's pattern doesn't obviously transfer to a well with a
  differently-sized one -- and it has only ever been validated on the one matched
  well pair, which shares the same ring geometry. `tools.spatial.add_mask_pyramid`
  still computes and attaches it to the dataframe (notebook 01's response-vs-
  geometry section uses it directly as a standalone diagnostic, independent of any
  trained model), it's just not fed to the model here.

* `train_calibrated`, not `train_predictor` -- see `models.train.calibrated_predict`
  for the prior-correction issue, which is worth reading before quoting any R2 off
  a hurdle model.

* INTENSITY NORMALIZATION HAPPENS ONCE, UP FRONT, ON THE DATAFRAME (`normalize`),
  before any feature is computed -- not per-column inside the model pipeline. Every
  x column this module hands back is therefore already in normalized units:
  a mask FRACTION in [0, 1] (`mask_features`), a log-distance (its geometry
  columns), a 0/1 flag, or `log1p(intensity - background)` (`normalize` +
  `intensity_features`). `build_data`/`fit` consequently treat all x columns
  identically -- standardize, nothing else -- and need no `intensity_cols`
  argument to tell raw-intensity columns apart from already-bounded ones.
  Two consequences worth knowing:
    - the intensity pyramid is now the local mean of log1p'd values (~a log
      geometric mean) rather than log1p of the local mean of raw values. Same
      information, slightly different numbers than the pre-`normalize` code.
    - `y_col` is also normalized up front: `two_part` uses a threshold of 0 on the 
      normalized scale.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

from models.data import build_prediction_data
from models.predictors import IFPredictor
from models.scalers import estimate_background
from models.train import train_calibrated, calibrated_predict
from tools.spatial import PYRAMID_SCALES, add_mask_pyramid, add_density_pyramid, add_intensity_pyramid

MAX_SIGMA = 240
RADIUS = 30
NUM_LAYERS = 3
HIDDEN = 64
LR = 1e-2
EPOCHS = 2500
PATIENCE = 400
LAMBDA_COMBINED = 10.0

RAW_SUFFIX = "_raw"


def normalize(df: pd.DataFrame, cols: Sequence[str],
              backgrounds: Optional[dict[str, float]] = None,
              train_idx=None) -> dict[str, float]:
    """Background-subtract + log1p each raw-intensity column IN PLACE, keeping the
    original values in `{col}{RAW_SUFFIX}`. Returns `{col: background used}`.

    This is THE normalization step (see the module docstring): run it on the
    dataframe before computing any feature, and everything downstream --
    `intensity_features`' pyramid, `build_data`, `fit` -- is uniformly in
    normalized units with no per-column special-casing.

    `df[col] = log1p(clip(raw - background, 0))`, the same transform
    `ColumnScaler` applies internally to an intensity column, just done once here
    instead of separately per derived feature. The background is
    `backgrounds[col]` when given (the eye-verified constant -- preferred, and
    leak-free by construction), else `models.scalers.estimate_background` on
    `train_idx`'s rows if given, else on every row. Passing a `backgrounds` dict
    that doesn't mention every column is fine and normal: e.g. project
    `IF_BACKGROUNDS` covers the readout channels, so `BMP4_mean` falls through to
    the estimate, and the return value reports what it actually used.

    re-executing the cell in a notebook is safe.
    """
    backgrounds = backgrounds or {}
    used: dict[str, float] = {}
    for col in cols:
        raw_col = f"{col}{RAW_SUFFIX}"
        raw = df[raw_col if raw_col in df.columns else col].to_numpy(np.float32)
        background = backgrounds.get(col)
        if background is None:
            fit_values = raw if train_idx is None else raw[np.asarray(train_idx)]
            background = estimate_background(fit_values)
        background = float(background)
        df[raw_col] = raw
        df[col] = np.log1p(np.clip(raw - background, 0.0, None)).astype(np.float32)
        used[col] = background
    return used


def mask_features(df: pd.DataFrame, mask_col: str = "BMP4_bin",
                  max_sigma: int = MAX_SIGMA, geometry: bool = True) -> list[str]:
    """Attach the BMP4-mask pyramid (+ pattern geometry) and return the x_cols.

    `{mask_col}_signed_dist` is computed by `add_mask_pyramid` (still attached to
    `df`, still useful as a standalone diagnostic) but deliberately EXCLUDED from
    the returned x_cols -- see the module docstring's "Geometry features" bullet
    for why. `log_dist_pos`/`log_dist_neg` are kept."""
    pyr = add_mask_pyramid(df, mask_col=mask_col, geometry=geometry)
    keep = [c for c in pyr if not c.startswith(f"{mask_col}_gauss_")]
    keep = [c for c in keep if not c.endswith("_signed_dist")]
    keep += [f"{mask_col}_gauss_{s}" for s in PYRAMID_SCALES if s <= max_sigma]
    return [mask_col] + [c for c in keep if c in df.columns]


def density_features(df: pd.DataFrame, max_sigma: int = MAX_SIGMA) -> list[str]:
    """Attach smoothed CELL density at the same scales and return the x_cols.
    No marker channel is involved -- this is the control that says how much of any
    score is available with no BMP4 information at all."""
    dens = add_density_pyramid(df)
    return [c for c in dens if int(c.rsplit("_", 1)[1]) <= max_sigma]


def intensity_features(df: pd.DataFrame, value_col: str = "BMP4_mean",
                       max_sigma: int = MAX_SIGMA) -> list[str]:
    """Attach a multiscale local-mean pyramid of a CONTINUOUS channel (e.g. BMP4
    intensity, not the thresholded `{channel}+`/`{channel}_bin` mask) and return
    the x_cols.

    `value_col` must ALREADY be normalized by `normalize` -- the pyramid is the
    local mean of the normalized channel, so its columns come out in the same
    units as the column itself and need no further treatment downstream. Raises if
    the column hasn't been through `normalize`, since raw intensities would
    otherwise reach `build_data`, which no longer background-subtracts anything."""
    raw_col = f"{value_col}{RAW_SUFFIX}"
    if raw_col not in df.columns:
        raise ValueError(
            f"{value_col!r} has not been normalized -- call "
            f"recipe.normalize(df, [{value_col!r}], backgrounds=...) first "
            "(see the module docstring's normalization bullet)"
        )
    pyr = add_intensity_pyramid(df, value_col=value_col)
    keep = [c for c in pyr if int(c.rsplit("_", 1)[1]) <= max_sigma]
    return [value_col] + keep


def build_data(df: pd.DataFrame, x_cols: Sequence[str], y_col: str, train_idx: float = None,
               radius: float = RADIUS, two_part: bool = True) -> dict:
    """`build_prediction_data` with the one exception these features need: every x
    column and y column is already in normalized units -- a fraction, a log-distance, a 0/1 flag
    or a `normalize`d intensity -- so none of them get background subtraction or a
    (second) log1p, only standardization. Genuinely binary columns skip even that.
    """
    x_cols = list(x_cols)
    return build_prediction_data(
        df, radius=radius, x_cols=x_cols, train_idx=train_idx,
        scaling="log1p_standard", y_cols=[y_col],
        no_background_cols=x_cols, no_transform_cols=x_cols + [y_col],
        no_scale_cols=[c for c in x_cols if df[c].dropna().isin([0.0, 1.0]).all()],
        custom_background={y_col: 0.0} if two_part else None, normalize_data=False, two_part=two_part,
    )


def fit(df: pd.DataFrame, x_cols: Sequence[str], y_col: str, train_mask, val_mask,
        train_idx, seed: int = 0, device: str = "cpu", verbose: bool = True, **train_kw):
    """Train the recommended model. Returns `(model, data, log_w, history)`.
    `x_cols` must come from `mask_features`/`intensity_features`/`density_features`
    on a dataframe already passed through `normalize`. `y_col` must also be normalized."""
    data = build_data(df, x_cols, y_col, train_idx)
    torch.manual_seed(seed)
    model = IFPredictor(in_channels=len(x_cols), num_outputs=1, num_layers=NUM_LAYERS,
                        hidden_channels=HIDDEN, normalize_data=False, two_part=True)
    kw = dict(epochs=EPOCHS, patience=PATIENCE, lr=LR, device=device, desc=str(y_col),
              verbose=verbose, lambda_combined=LAMBDA_COMBINED)
    kw.update(train_kw)
    history, log_w = train_calibrated(model, data, train_mask, val_mask, **kw)
    return model, data, log_w, history


def predict(model, data: dict, log_w: float) -> np.ndarray:
    return calibrated_predict(model, data, log_w)


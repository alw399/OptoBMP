"""
Persisting a trained model together with everything needed to re-apply it.

Bare weights alone cannot be re-run: reproducing a prediction needs the exact
columns, the FITTED scalers (reused, never refit) and the graph parameters too.
"""

from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from models.graph import build_radius_graph
from models.scalers import ColumnScaler

def save_model(
    path: str,
    model: nn.Module,
    model_kwargs: dict,
    pred_data: dict,
    radius: float,
    min_neighbors: int = 1,
    length_scale: Optional[float] = None,
    density_radius: Optional[float] = None,
    normalize_data: bool = True,
) -> None:
    """
    Saves `model.state_dict()` (just the weight tensors -- much smaller and more
    portable than pickling the whole `nn.Module`, the standard way to persist a
    torch model) plus everything `load_model`/`apply_prediction_data` need to
    reconstruct the model and correctly preprocess a NEW cell_df the same way:
    `model_kwargs` (the exact constructor kwargs used, so the architecture can be
    rebuilt before loading weights into it), the fitted `x_scaler`/`y_scaler` (see
    `ColumnScaler` -- must be REUSED, not refit, for predictions on new data to be
    meaningful/invertible), the `x_cols`/`y_cols` used, and the radius-graph params.

    `radius` is the message-passing/graph radius. `density_radius` is a SEPARATE
    radius, only needed if `local_density` is one of the x cols -- it's whatever
    radius was passed to `tools.morphology.local_density` when that column was
    computed, which need not match the graph radius. Saved so a later
    `apply_prediction_data` call (e.g. in `cross_predict.ipynb`) recomputes
    `local_density` the same way training did, instead of assuming it equals
    the graph radius.

    `normalize_data` MUST match whatever `build_prediction_data`/
    `build_multi_image_prediction_data` was called with -- it changes what the
    saved `edge_weight` actually MEANS (a weighted-average vs. weighted-sum
    adjacency, see `normalize_distance_weights`), so `apply_prediction_data`
    needs the exact same setting to rebuild an eval graph the trained weights
    still interpret correctly. If `model` is an `IFPredictor`, its own
    `normalize_data` constructor arg should have been set to this same value
    too (`model_kwargs` already carries whatever that was, independently of
    what's saved here).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(
            state_dict=model.state_dict(),
            model_kwargs=model_kwargs,
            x_cols=pred_data["x_cols"],
            y_cols=pred_data["y_cols"],
            x_scaler=pred_data["x_scaler"],
            y_scaler=pred_data["y_scaler"],
            radius=radius,
            min_neighbors=min_neighbors,
            length_scale=length_scale,
            density_radius=density_radius,
            normalize_data=normalize_data,
        ),
        path,
    )


def load_model(path: str, model_cls: type) -> tuple[nn.Module, dict]:
    """
    Reloads a checkpoint written by `save_model`. `model_cls` is whichever class
    it was saved from (`IFPredictor` or `models.gat.IFPredictorGAT`) -- returns
    `(model, checkpoint)` in eval mode; `checkpoint` also carries the
    scalers/columns/radius `apply_prediction_data` needs.
    """
    checkpoint = torch.load(path, weights_only=False)
    model = model_cls(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def apply_prediction_data(df: pd.DataFrame, checkpoint: dict) -> dict:
    """
    Builds a `predict_df`-ready data dict (same keys as `build_prediction_data`)
    for a NEW `df`, using a checkpoint's saved scalers/columns/radius instead of
    fitting new ones -- for evaluating a model trained on one image against a
    DIFFERENT image's cells (`cross_predict.ipynb`). No train/val split here:
    this is for evaluation only, not further training.
    """
    centroids = df[["centroid_y", "centroid_x"]].to_numpy(dtype=np.float32)
    # "normalize_weights" fallback: checkpoints saved by `save_model` before it was
    # renamed to `normalize_data` still have the old key -- not a model_kwargs concern,
    # since this is the DATA-side flag (see `save_model`'s docstring).
    normalize_data = checkpoint.get("normalize_data", checkpoint.get("normalize_weights", True))
    edge_index, edge_weight = build_radius_graph(
        centroids, checkpoint["radius"], checkpoint.get("min_neighbors", 1), checkpoint.get("length_scale"),
        normalize_data,
    )

    def scale(cols: list[str], scaler: Optional[ColumnScaler]) -> np.ndarray:
        if not cols:
            return np.zeros((len(df), 0), dtype=np.float32)
        return scaler.transform(df[cols].to_numpy(dtype=np.float32)).astype(np.float32)

    x_cols, y_cols = checkpoint["x_cols"], checkpoint["y_cols"]
    x_scaler, y_scaler = checkpoint["x_scaler"], checkpoint["y_scaler"]

    return dict(
        edge_index=edge_index,
        edge_weight=edge_weight,
        x=torch.from_numpy(scale(x_cols, x_scaler)),
        y=torch.from_numpy(scale(y_cols, y_scaler)),
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        x_cols=x_cols,
        y_cols=y_cols,
    )

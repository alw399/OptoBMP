"""
Radius-based (NOT k-NN) neighbor-only GNN for predicting a cell's own
immunofluorescence from its spatial NEIGHBORS' immunofluorescence.

Two disjoint blocks of input columns:
  - `global_x_cols`: seen directly by every cell for ITSELF (no message passing --
    "global" because every node gets this, not just its neighbors). e.g. letting the
    model see a cell's own Activin while predicting its other markers.
  - `neighbor_x_cols`: only seen through neighbors -- a cell's OWN value for these
    never reaches its own prediction directly, only via aggregating its neighbors'
    values (self-excluded message passing, same idea as
    `tools.morphology.NeighborFeaturePredictor`, but over a radius graph instead of
    k-NN, and with fixed distance-normalized weights instead of learned attention).

`y_cols` defaults to every IF channel not already in `global_x_cols` (predicting a
channel the model already sees directly for itself would be trivial leakage).
"""

from typing import Optional, Sequence, Union

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import RobustScaler, StandardScaler
# pyrefly: ignore [missing-import]
from torch_geometric.nn import MessagePassing
from tqdm.auto import trange

from tools.morphology import radius_edge_index, init_weights, init_output_layer
from tools.qc import get_bimodal_threshold

# --------------------------------------------------------------------------
# Radius graph with distance baked into the (fixed, non-learned) edge weights
# --------------------------------------------------------------------------


def normalize_distance_weights(
    edge_index: np.ndarray, edge_dist: np.ndarray, n_nodes: int, length_scale: float, normalize: bool = True
) -> torch.Tensor:
    """
    Exponential distance decay (`exp(-dist / length_scale)`, closer neighbors get a
    larger raw weight, always positive so a `min_neighbors` fallback edge -- see
    `radius_edge_index` -- never gets a zero/negative weight just for sitting past
    `radius`), then (if `normalize`) row-normalized so each node's INCOMING edge
    weights sum to exactly 1. That normalization is what turns "a decreasing function
    of distance" into an actual weighted-AVERAGING adjacency matrix, the spatial-graph
    analog of GCN's degree normalization.

    `normalize=False` skips that step, leaving the raw per-edge decay weight as-is.
    `WeightedNeighborConv` aggregates with `aggr="add"`, so a normalized weight (sums
    to 1) makes that a weighted AVERAGE -- insensitive to neighbor COUNT, since a
    5-neighbor and a 50-neighbor node can produce the same aggregate if their feature
    distributions look similar (this is exactly why `tools.morphology.local_density`
    exists as a separate explicit feature: mean/attention aggregators normalize count
    information away). Without normalization, aggregation becomes a weighted SUM
    (GraphSAGE/GIN-style) -- a node with more/closer neighbors accumulates a strictly
    larger total signal, so degree/density becomes implicitly recoverable from the
    embedding's magnitude alone, without needing `local_density` as an input at all.
    Only meaningfully changes GNN behavior: GAT's attention is softmax-normalized by
    `GATConv` regardless of this flag (that normalization is intrinsic to how
    attention is computed, not something this edge weight controls for GAT) -- this
    only changes the edge FEATURE GAT conditions its (still sum-to-1) attention on.
    """
    dist = torch.as_tensor(edge_dist, dtype=torch.float32)
    dst = torch.as_tensor(edge_index[1], dtype=torch.long)
    w = torch.exp(-dist / length_scale)
    if not normalize:
        return w
    denom = torch.zeros(n_nodes, dtype=torch.float32).scatter_add_(0, dst, w)
    return w / denom[dst].clamp(min=1e-8)


def build_radius_graph(
    centroids: np.ndarray,
    radius: float,
    min_neighbors: int = 1,
    length_scale: Optional[float] = None,
    normalize_weights: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Connects every pair of cells within `radius` pixels (see
    `tools.morphology.radius_edge_index` -- NOT a fixed-k graph, so node degree is a
    direct density signal), then converts distance into an edge weight
    (`normalize_distance_weights`, `length_scale` defaults to `radius`).

    `normalize_weights=False`: see `normalize_distance_weights` -- turns the GNN's
    aggregation from a weighted average into a weighted sum, an alternative to an
    explicit `local_density` feature for letting the model see neighbor count/density.

    Returns
    -------
    edge_index : (2, E) int64 tensor, COO format, symmetric.
    edge_weight : (E,) float32 tensor -- sums to 1 per destination node if
        `normalize_weights` (default), else the raw per-edge distance-decay weight.
    """
    edge_index_np, edge_dist_np = radius_edge_index(centroids, radius=radius, min_neighbors=min_neighbors)
    edge_weight = normalize_distance_weights(
        edge_index_np, edge_dist_np, len(centroids), length_scale or radius, normalize=normalize_weights
    )
    return torch.from_numpy(edge_index_np), edge_weight


def border_mask(centroids: np.ndarray, image_shape: tuple[int, int], radius: float) -> np.ndarray:
    """
    True for cells whose `radius`-px neighborhood extends past the image edge -- their
    true neighbor count/signal is undercounted simply because part of the neighborhood
    was never imaged, not because there's actually nothing there. Meant to exclude
    these cells from ever being a train/val/test TARGET (see `build_prediction_data`'s
    `train_idx`, and mask out the same cells for val/test the same way) -- they should
    still take part in message passing as a neighbor for cells further from the edge,
    just never be trained on or evaluated themselves.
    """
    h, w = image_shape
    y, x = centroids[:, 0], centroids[:, 1]
    return (y < radius) | (y > h - radius) | (x < radius) | (x > w - radius)


# --------------------------------------------------------------------------
# Model: fixed distance-weighted neighbor-only conv (no self term, no learned
# attention -- `models.gat.NeighborRadiusGAT` is the learned-attention counterpart)
# --------------------------------------------------------------------------


class WeightedNeighborConv(MessagePassing):
    """One neighbor-only message-passing layer: destination node i's output is the
    weighted average (using the fixed, precomputed `edge_weight`) of its neighbors'
    (linearly transformed) features -- i never receives a message from itself."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="add", node_dim=0)
        self.lin = nn.Linear(in_channels, out_channels)
        init_weights(self.lin)  # standardized input, ReLU follows (except possibly the last layer)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        x = self.lin(x)
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        return x_j * edge_weight.unsqueeze(-1)  # already row-normalized to sum to 1 per destination


class NeighborRadiusGNN(nn.Module):
    """Stack of `WeightedNeighborConv` layers -- an embedding built purely from
    neighbors' `neighbor_x_cols`, over the radius graph. Same `num_layers=1` default
    and 2-hop leakage caveat as `tools.morphology.NeighborOnlyGNN`: at 1 layer, node
    i's embedding is strictly a function of its direct neighbors' raw features."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        self.convs = nn.ModuleList(WeightedNeighborConv(dims[i], dims[i + 1]) for i in range(num_layers))
        self.norms = nn.ModuleList(nn.BatchNorm1d(dims[i + 1]) for i in range(num_layers - 1))
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            if i < len(self.convs) - 1:
                x = F.relu(self.norms[i](x))
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class NeighborIFPredictor(nn.Module):
    """Encoder (`NeighborRadiusGNN` over `neighbor_x_cols`) + a linear head that also
    sees `x_global` (the cell's OWN `global_x_cols`, concatenated in directly -- no
    message passing) -- predicts `y_cols`. `global_in_channels=0` (i.e.
    `global_x_cols=[]`) is fine: concatenating a zero-width tensor is a no-op."""

    def __init__(
        self,
        neighbor_in_channels: int,
        global_in_channels: int,
        num_outputs: int,
        hidden_channels: int = 64,
        embedding_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = NeighborRadiusGNN(neighbor_in_channels, embedding_dim, hidden_channels, num_layers, dropout)
        self.head = nn.Linear(embedding_dim + global_in_channels, num_outputs)
        init_output_layer(self.head)  # no activation follows this layer

    def forward(
        self, x_neighbor: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor, x_global: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x_neighbor, edge_index, edge_weight)
        z_full = torch.cat([z, x_global], dim=1)
        return self.head(z_full), z


# --------------------------------------------------------------------------
# Data prep + training
# --------------------------------------------------------------------------

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
    return float(negative.median()) if len(negative) > 0 else float(series.median())


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
    sometimes used as-is in cytometry pipelines -- because `NeighborRadiusGNN`'s
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
    ALONGSIDE an intensity column (e.g. `Activin_mean`) in the same
    `global_x_cols`/`neighbor_x_cols` group but shouldn't get the same treatment:

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
    """

    def __init__(
        self,
        scaling: str = "log1p_standard",
        arcsinh_cofactor: float = 5.0,
        subtract_background: Union[bool, Sequence[bool]] = True,
        apply_transform: Union[bool, Sequence[bool]] = True,
    ):
        if scaling not in SCALING_MODES:
            raise ValueError(f"scaling must be one of {SCALING_MODES}, got {scaling!r}")
        self.scaling = scaling
        self.arcsinh_cofactor = arcsinh_cofactor
        self.subtract_background = subtract_background
        self.apply_transform = apply_transform
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
        self.background_ = np.array(
            [estimate_background(x[:, j]) if bg_mask[j] else 0.0 for j in range(x.shape[1])], dtype=np.float32
        )
        self.base_scaler.fit(self._forward(x))
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
) -> tuple[np.ndarray, Optional[ColumnScaler]]:
    """Shared by `build_prediction_data`/`build_multi_image_prediction_data`: fits a
    `ColumnScaler` on `fit_slice[cols]` (the training rows) and transforms `df[cols]`
    (every row). `no_background_cols`/`no_transform_cols` carve out per-column
    exceptions to the group's otherwise-uniform `subtract_background`/`scaling`."""
    if not cols:
        return np.zeros((len(df), 0), dtype=np.float32), None
    bg_mask = [subtract_background and c not in no_background_cols for c in cols]
    transform_mask = [c not in no_transform_cols for c in cols]
    scaler = ColumnScaler(scaling, arcsinh_cofactor, bg_mask, transform_mask).fit(
        fit_slice[cols].to_numpy(dtype=np.float32)
    )
    return scaler.transform(df[cols].to_numpy(dtype=np.float32)).astype(np.float32), scaler


def build_prediction_data(
    df: pd.DataFrame,
    if_channels: Sequence[str],
    radius: float,
    global_x_cols: Sequence[str],
    neighbor_x_cols: Optional[Sequence[str]] = None,
    y_cols: Optional[Sequence[str]] = None,
    min_neighbors: int = 1,
    length_scale: Optional[float] = None,
    normalize_weights: bool = True,
    train_idx: Optional[np.ndarray] = None,
    scaling: str = "log1p_standard",
    arcsinh_cofactor: float = 5.0,
    subtract_background: bool = True,
    no_background_cols: Sequence[str] = (),
    no_transform_cols: Sequence[str] = (),
) -> dict:
    """
    Builds the radius graph (`build_radius_graph`) plus scaled x_global /
    x_neighbor / y tensors for `NeighborIFPredictor` (or
    `models.gat.NeighborIFPredictorGAT`, same tensors). `scaling`/
    `arcsinh_cofactor`/`subtract_background` select how each column is scaled --
    see `ColumnScaler`.

    `normalize_weights=False`: see `build_radius_graph`/`normalize_distance_weights`
    -- an alternative to an explicit `local_density` feature, letting the GNN see
    neighbor count/density implicitly via a weighted-SUM aggregation instead of a
    weighted-average one.

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

    `neighbor_x_cols` defaults to every column in `if_channels`; `y_cols` defaults to
    every column in `if_channels` EXCLUDING `global_x_cols` (a channel the model
    already sees directly for itself is not worth predicting).

    Scalers are fit on `train_idx` (positions, not labels) if given, else the whole
    of `df` -- pass the training split's indices to avoid leakage from val/test cells.
    """
    global_x_cols = list(global_x_cols)
    neighbor_x_cols = list(if_channels) if neighbor_x_cols is None else list(neighbor_x_cols)
    y_cols = [c for c in if_channels if c not in global_x_cols] if y_cols is None else list(y_cols)

    centroids = df[["centroid_y", "centroid_x"]].to_numpy(dtype=np.float32)
    edge_index, edge_weight = build_radius_graph(centroids, radius, min_neighbors, length_scale, normalize_weights)

    fit_slice = df if train_idx is None else df.iloc[train_idx]
    x_global, global_scaler = _scale_columns(
        df, fit_slice, global_x_cols, scaling, arcsinh_cofactor, subtract_background, no_background_cols, no_transform_cols
    )
    x_neighbor, neighbor_scaler = _scale_columns(
        df, fit_slice, neighbor_x_cols, scaling, arcsinh_cofactor, subtract_background, no_background_cols, no_transform_cols
    )
    y, y_scaler = _scale_columns(
        df, fit_slice, y_cols, scaling, arcsinh_cofactor, subtract_background, no_background_cols, no_transform_cols
    )

    return dict(
        edge_index=edge_index,
        edge_weight=edge_weight,
        x_global=torch.from_numpy(x_global),
        x_neighbor=torch.from_numpy(x_neighbor),
        y=torch.from_numpy(y),
        global_scaler=global_scaler,
        neighbor_scaler=neighbor_scaler,
        y_scaler=y_scaler,
        global_x_cols=global_x_cols,
        neighbor_x_cols=neighbor_x_cols,
        y_cols=y_cols,
    )


def build_multi_image_prediction_data(
    cell_dfs: Sequence[pd.DataFrame],
    image_shapes: Sequence[tuple[int, int]],
    if_channels: Sequence[str],
    radius: float,
    global_x_cols: Sequence[str],
    neighbor_x_cols: Optional[Sequence[str]] = None,
    y_cols: Optional[Sequence[str]] = None,
    min_neighbors: int = 1,
    length_scale: Optional[float] = None,
    normalize_weights: bool = True,
    train_idx: Optional[np.ndarray] = None,
    scaling: str = "log1p_standard",
    arcsinh_cofactor: float = 5.0,
    subtract_background: bool = True,
    no_background_cols: Sequence[str] = (),
    no_transform_cols: Sequence[str] = (),
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

    `image_shapes[i]` is unused by this function directly -- it exists so
    callers who already loop over `(cell_df, image_shape)` pairs for
    `border_mask` can pass the same list through without restructuring, but
    graph-building only needs centroids. Kept as a parameter for symmetry/
    documentation of the expected per-image inputs.
    """
    assert len(cell_dfs) == len(image_shapes), "one image_shape per cell_df"
    global_x_cols = list(global_x_cols)
    neighbor_x_cols = list(if_channels) if neighbor_x_cols is None else list(neighbor_x_cols)
    y_cols = [c for c in if_channels if c not in global_x_cols] if y_cols is None else list(y_cols)

    edge_index_parts, edge_weight_parts = [], []
    offset = 0
    for cell_df in cell_dfs:
        centroids = cell_df[["centroid_y", "centroid_x"]].to_numpy(dtype=np.float32)
        ei, ew = build_radius_graph(centroids, radius, min_neighbors, length_scale, normalize_weights)
        edge_index_parts.append(ei + offset)
        edge_weight_parts.append(ew)
        offset += len(cell_df)
    edge_index = torch.cat(edge_index_parts, dim=1)
    edge_weight = torch.cat(edge_weight_parts)

    df = pd.concat(cell_dfs, ignore_index=True)
    fit_slice = df if train_idx is None else df.iloc[train_idx]
    x_global, global_scaler = _scale_columns(
        df, fit_slice, global_x_cols, scaling, arcsinh_cofactor, subtract_background, no_background_cols, no_transform_cols
    )
    x_neighbor, neighbor_scaler = _scale_columns(
        df, fit_slice, neighbor_x_cols, scaling, arcsinh_cofactor, subtract_background, no_background_cols, no_transform_cols
    )
    y, y_scaler = _scale_columns(
        df, fit_slice, y_cols, scaling, arcsinh_cofactor, subtract_background, no_background_cols, no_transform_cols
    )

    return dict(
        edge_index=edge_index,
        edge_weight=edge_weight,
        x_global=torch.from_numpy(x_global),
        x_neighbor=torch.from_numpy(x_neighbor),
        y=torch.from_numpy(y),
        global_scaler=global_scaler,
        neighbor_scaler=neighbor_scaler,
        y_scaler=y_scaler,
        global_x_cols=global_x_cols,
        neighbor_x_cols=neighbor_x_cols,
        y_cols=y_cols,
    )


def train_neighbor_predictor(
    model: nn.Module,
    data: dict,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 20,
    device: str = "cpu",
    desc: str = "training",
) -> dict[str, list[float]]:
    """
    Full-batch transductive training (mirrors `tools.morphology.train_gnn`): every
    cell always participates in message passing, `train_mask`/`val_mask` only
    control which cells' own loss/gradients/early-stopping count. Works for both
    `NeighborIFPredictor` (this module) and `models.gat.NeighborIFPredictorGAT` --
    same `forward(x_neighbor, edge_index, edge_weight, x_global)` signature.
    """
    model = model.to(device)
    x_neighbor = data["x_neighbor"].to(device)
    x_global = data["x_global"].to(device)
    edge_index = data["edge_index"].to(device)
    edge_weight = data["edge_weight"].to(device)
    y = data["y"].to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val, best_state, bad_epochs = float("inf"), None, 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    pbar = trange(epochs, desc=desc, leave=True)
    for _ in pbar:
        model.train()
        opt.zero_grad()
        pred, _ = model(x_neighbor, edge_index, edge_weight, x_global)
        loss = F.mse_loss(pred[train_mask], y[train_mask])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            pred, _ = model(x_neighbor, edge_index, edge_weight, x_global)
            val_loss = F.mse_loss(pred[val_mask], y[val_mask]).item()

        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_loss)
        pbar.set_postfix(train=f"{loss.item():.4f}", val=f"{val_loss:.4f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


_AUTO = object()  # sentinel: "use data['y_scaler']" -- None is a legitimate override, not a valid default


@torch.no_grad()
def predict_df(
    model: nn.Module, data: dict, mask: torch.Tensor, index: pd.Index, y_scaler=_AUTO
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs `model` over the cells in `mask` and returns (pred_df, true_df), both in
    the SAME units -- original units by default (`data['y_scaler']`), or pass
    `y_scaler=None` explicitly to see both in the model's raw SCALED output space
    instead (no inverse-transform) -- same convention as
    `models.cnn.predict_patch_df`.
    """
    if y_scaler is _AUTO:
        y_scaler = data["y_scaler"]
    model.eval()
    pred, _ = model(data["x_neighbor"], data["edge_index"], data["edge_weight"], data["x_global"])
    mask_np = mask.cpu().numpy()
    pred_scaled = pred[mask].cpu().numpy()
    true_scaled = data["y"][mask].cpu().numpy()
    if y_scaler is not None:
        pred_scaled = y_scaler.inverse_transform(pred_scaled)
        true_scaled = y_scaler.inverse_transform(true_scaled)
    idx = index[mask_np]
    pred_df = pd.DataFrame(pred_scaled, columns=data["y_cols"], index=idx)
    true_df = pd.DataFrame(true_scaled, columns=data["y_cols"], index=idx)
    return pred_df, true_df


# --------------------------------------------------------------------------
# Checkpointing -- save enough to re-apply a trained model to a DIFFERENT
# cell_df later (`apply_prediction_data`, see cross_predict.ipynb), not just
# the bare weights, which alone can't be re-run without knowing the exact
# columns/scalers/graph radius the model was trained with.
# --------------------------------------------------------------------------


def save_model(
    path: str,
    model: nn.Module,
    model_kwargs: dict,
    pred_data: dict,
    radius: float,
    min_neighbors: int = 1,
    length_scale: Optional[float] = None,
    density_radius: Optional[float] = None,
    normalize_weights: bool = True,
) -> None:
    """
    Saves `model.state_dict()` (just the weight tensors -- much smaller and more
    portable than pickling the whole `nn.Module`, the standard way to persist a
    torch model) plus everything `load_model`/`apply_prediction_data` need to
    reconstruct the model and correctly preprocess a NEW cell_df the same way:
    `model_kwargs` (the exact constructor kwargs used, so the architecture can be
    rebuilt before loading weights into it), the fitted `global_scaler`/
    `neighbor_scaler`/`y_scaler` (see `ColumnScaler` -- must be REUSED, not
    refit, for predictions on new data to be meaningful/invertible), the
    `neighbor_x_cols`/`global_x_cols`/`y_cols` used, and the radius-graph params.

    `radius` is the message-passing/graph radius. `density_radius` is a SEPARATE
    radius, only needed if `local_density` is one of the x cols -- it's whatever
    radius was passed to `tools.morphology.local_density` when that column was
    computed, which need not match the graph radius. Saved so a later
    `apply_prediction_data` call (e.g. in `cross_predict.ipynb`) recomputes
    `local_density` the same way training did, instead of assuming it equals
    the graph radius.

    `normalize_weights` MUST match whatever `build_prediction_data`/
    `build_multi_image_prediction_data` was called with -- it changes what the
    saved `edge_weight` actually MEANS (a weighted-average vs. weighted-sum
    adjacency, see `normalize_distance_weights`), so `apply_prediction_data`
    needs the exact same setting to rebuild an eval graph the trained weights
    still interpret correctly.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(
            state_dict=model.state_dict(),
            model_kwargs=model_kwargs,
            neighbor_x_cols=pred_data["neighbor_x_cols"],
            global_x_cols=pred_data["global_x_cols"],
            y_cols=pred_data["y_cols"],
            global_scaler=pred_data["global_scaler"],
            neighbor_scaler=pred_data["neighbor_scaler"],
            y_scaler=pred_data["y_scaler"],
            radius=radius,
            min_neighbors=min_neighbors,
            length_scale=length_scale,
            density_radius=density_radius,
            normalize_weights=normalize_weights,
        ),
        path,
    )


def load_model(path: str, model_cls: type) -> tuple[nn.Module, dict]:
    """
    Reloads a checkpoint written by `save_model`. `model_cls` is whichever class
    it was saved from (`NeighborIFPredictor` or `models.gat.NeighborIFPredictorGAT`)
    -- returns `(model, checkpoint)` in eval mode; `checkpoint` also carries the
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
    edge_index, edge_weight = build_radius_graph(
        centroids, checkpoint["radius"], checkpoint.get("min_neighbors", 1), checkpoint.get("length_scale"),
        checkpoint.get("normalize_weights", True),
    )

    def scale(cols: list[str], scaler: Optional[ColumnScaler]) -> np.ndarray:
        if not cols:
            return np.zeros((len(df), 0), dtype=np.float32)
        return scaler.transform(df[cols].to_numpy(dtype=np.float32)).astype(np.float32)

    global_x_cols, neighbor_x_cols, y_cols = checkpoint["global_x_cols"], checkpoint["neighbor_x_cols"], checkpoint["y_cols"]
    global_scaler, neighbor_scaler, y_scaler = checkpoint["global_scaler"], checkpoint["neighbor_scaler"], checkpoint["y_scaler"]

    return dict(
        edge_index=edge_index,
        edge_weight=edge_weight,
        x_global=torch.from_numpy(scale(global_x_cols, global_scaler)),
        x_neighbor=torch.from_numpy(scale(neighbor_x_cols, neighbor_scaler)),
        y=torch.from_numpy(scale(y_cols, y_scaler)),
        global_scaler=global_scaler,
        neighbor_scaler=neighbor_scaler,
        y_scaler=y_scaler,
        global_x_cols=global_x_cols,
        neighbor_x_cols=neighbor_x_cols,
        y_cols=y_cols,
    )

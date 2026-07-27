"""
Consolidated morphology-pipeline utilities.

This project's active work now lives in stitch.ipynb/segment.ipynb (see
tools/restitch_fovs.py, tools/qc.py, tools/features.py); everything below is
only used by the older per-cell morphology-modeling notebooks under
notebooks/morphology/ (cropped.ipynb, extrinsic.ipynb, intrinsic.ipynb) and is
kept here, consolidated into one module, for that purpose. Originally split
across tools/graph.py, tools/dataset.py, tools/modeling.py, tools/cnn.py,
tools/gnn.py, and tools/patches.py -- combined section-by-section below in
that dependency order (each section's own docstring, kept as a comment
block, explains its purpose).
"""

from __future__ import annotations

import random
import warnings
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm.auto import trange
# pyrefly: ignore [missing-import]
from torch_geometric.data import Data, Dataset
# pyrefly: ignore [missing-import]
from torch_geometric.nn import GATConv, GCNConv, MessagePassing, SAGEConv
# pyrefly: ignore [missing-import]
from torch_geometric.utils import softmax



# ==============================================================================
# graph.py -- spatial neighbor graphs
# -----------------------------------
# Build a spatial neighbor graph over cells from their centroids.
#
# Cells that are physically close in the tissue become graph neighbors -- this
# is the standard construction used to feed segmented, multiplexed imaging data
# into a graph neural network (each node = one cell, edges = local neighborhood).

def knn_edge_index(
    centroids: np.ndarray,
    k: int = 6,
    mutual: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    centroids : (N, 2) array of (y, x) pixel coordinates, one row per cell.
    k : number of nearest neighbors per cell.
    mutual : if True, only keep an edge when both endpoints list each other as
        a neighbor (sparser, more conservative graph). If False (default),
        keep an edge whenever either endpoint lists the other (denser).

    Returns
    -------
    edge_index : (2, E) int64 array, COO format, symmetric (both directions present).
    edge_dist  : (E,) float32 array of Euclidean distances, aligned with edge_index.
    """
    n = centroids.shape[0]
    k_eff = min(k + 1, n)  # +1 because a point is its own nearest neighbor
    nn = NearestNeighbors(n_neighbors=k_eff).fit(centroids)
    dist, idx = nn.kneighbors(centroids)

    src = np.repeat(np.arange(n), k_eff - 1)
    dst = idx[:, 1:].reshape(-1)  # drop self-match in column 0
    d = dist[:, 1:].reshape(-1)

    # dedupe (i, j) / (j, i) pairs into a plain adjacency set first
    forward = {(i, j) for i, j in zip(src.tolist(), dst.tolist())}

    if mutual:
        pairs = {(i, j) for (i, j) in forward if (j, i) in forward}
    else:
        pairs = forward | {(j, i) for (i, j) in forward}

    if not pairs:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    edge_index = np.array(sorted(pairs), dtype=np.int64).T
    edge_dist = np.linalg.norm(
        centroids[edge_index[0]] - centroids[edge_index[1]], axis=1
    ).astype(np.float32)
    return edge_index, edge_dist


def radius_edge_index(
    centroids: np.ndarray,
    radius: float,
    min_neighbors: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Connect every pair of cells within `radius` pixels of each other -- unlike
    `knn_edge_index`, the NUMBER of neighbors per cell varies with local density
    (more edges in a dense region, fewer in a sparse one), so node degree becomes
    a direct, built-in density signal rather than something a fixed `k` fixes away.
    Already symmetric by construction (distance(i, j) == distance(j, i)), so no
    `mutual`/union step is needed the way `knn_edge_index` requires.

    Parameters
    ----------
    centroids : (N, 2) array of (y, x) pixel coordinates, one row per cell.
    radius : neighbors within this many pixels are connected.
    min_neighbors : cells with fewer than this many neighbors within `radius`
        (e.g. sitting in an unusually sparse patch) fall back to their
        `min_neighbors` nearest neighbors regardless of distance, so no node is
        ever left isolated.

    Returns
    -------
    edge_index : (2, E) int64 array, COO format, symmetric.
    edge_dist  : (E,) float32 array of Euclidean distances, aligned with edge_index.
    """
    n = centroids.shape[0]
    nn = NearestNeighbors(radius=radius).fit(centroids)
    neighbor_dists, neighbor_idxs = nn.radius_neighbors(centroids)

    k_eff = min(min_neighbors + 1, n)  # +1 for self, dropped below
    knn = NearestNeighbors(n_neighbors=k_eff).fit(centroids)
    _, fallback_idx = knn.kneighbors(centroids)

    pairs = set()
    for i in range(n):
        idx = neighbor_idxs[i]
        idx = idx[idx != i]  # radius_neighbors includes the point itself at distance 0
        if len(idx) < min_neighbors:
            idx = fallback_idx[i, 1:]  # drop self (always column 0, distance 0)
        for j in idx:
            j = int(j)
            pairs.add((i, j))
            pairs.add((j, i))

    if not pairs:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    edge_index = np.array(sorted(pairs), dtype=np.int64).T
    edge_dist = np.linalg.norm(
        centroids[edge_index[0]] - centroids[edge_index[1]], axis=1
    ).astype(np.float32)
    return edge_index, edge_dist


def local_density(centroids: np.ndarray, radius: float) -> np.ndarray:
    """
    Number of OTHER cells within `radius` pixels of each cell -- a direct, explicit
    density feature. Worth adding alongside `radius_edge_index` as an input feature:
    message-passing aggregators (mean/attention, as used by SAGEConv/GATConv here)
    largely normalize away raw neighbor count, so a dense and a sparse neighborhood
    with similar per-cell values can still produce similar messages -- this gives
    the model an explicit density readout it wouldn't otherwise reliably get from
    the graph structure alone.
    """
    nn = NearestNeighbors(radius=radius).fit(centroids)
    neighbor_idxs = nn.radius_neighbors(centroids, return_distance=False)
    return np.array([len(idx) - 1 for idx in neighbor_idxs], dtype=np.float32)  # -1 excludes self


# ==============================================================================
# dataset.py -- per-cell feature tables -> torch_geometric graphs
# ---------------------------------------------------------------
# Turns per-cell feature tables (see `features.extract_cell_features`) into
# torch_geometric graphs, one graph per image (FOV / well / condition).
#
# Each node = one cell. Edges connect spatially nearby cells (k-NN on
# centroids, see `graph.knn_edge_index`). Every Data object also carries
# `cell_id` and `pos` (centroid) so node-level predictions/embeddings can
# always be traced back to a specific cell in the original mask.

MORPHOLOGY_COLS = [
    "area",  
    # "perimeter",  
    "perimeter_crofton",    
    "circularity",  
    "eccentricity",  
    # "solidity", 
    # "extent",  
    # "major_axis_length",  
    # "minor_axis_length",  
    # "orientation", 
    # "equivalent_diameter",  
]


def marker_cols(df: pd.DataFrame, stat: str = "mean") -> list[str]:
    """Fluorescence columns produced by extract_cell_features, e.g. '{channel}_mean'."""
    suffix = f"_{stat}"
    return [c for c in df.columns if c.endswith(suffix) and c not in MORPHOLOGY_COLS]


def dataframe_to_graph(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    k: int = 6,
    image_id: int = 0,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> tuple[Data, np.ndarray, np.ndarray]:
    """
    Build a single graph from one image's cell feature table.

    Standardizes `feature_cols` using `mean`/`std` if given, otherwise fits
    them from this table (pass the training set's mean/std when building
    val/test graphs, to avoid leakage).
    """
    centroids = np.ascontiguousarray(df[["centroid_y", "centroid_x"]].to_numpy(dtype=np.float32))
    x = df[feature_cols].to_numpy(dtype=np.float32)

    mean = x.mean(axis=0) if mean is None else mean
    std = x.std(axis=0) if std is None else std
    std = np.where(std < 1e-8, 1.0, std)
    x = (x - mean) / std

    edge_index, edge_dist = knn_edge_index(centroids, k=k)

    data = Data(
        x=torch.from_numpy(x),
        edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(edge_dist).unsqueeze(-1),
        pos=torch.from_numpy(centroids),
    )
    data.cell_id = torch.from_numpy(np.asarray(df.index, dtype=np.int64))
    data.image_id = torch.full((len(df),), image_id, dtype=torch.long)
    return data, mean, std


class CellGraphDataset(Dataset):
    """
    A torch_geometric Dataset where each sample is one whole-image cell graph.

    Works directly with torch_geometric.loader.DataLoader, which batches
    multiple image-graphs into a single disjoint-union graph per training step.

    Example
    -------
        df = extract_cell_features(img, masks, channel_names)
        feature_cols = MORPHOLOGY_COLS + marker_cols(df, 'mean')
        ds = CellGraphDataset.from_dataframes([df], feature_cols=feature_cols, k=6)

        from torch_geometric.loader import DataLoader
        loader = DataLoader(ds, batch_size=4, shuffle=True)
        batch = next(iter(loader))
    """

    def __init__(
        self,
        graphs: list[Data],
        feature_cols: list[str],
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
    ):
        super().__init__()
        self._graphs = graphs
        self.feature_cols = list(feature_cols)
        self.feature_mean = feature_mean
        self.feature_std = feature_std

    @classmethod
    def from_dataframes(
        cls,
        dataframes: list[pd.DataFrame],
        feature_cols: Optional[Sequence[str]] = None,
        k: int = 6,
        image_ids: Optional[Sequence[int]] = None,
    ) -> "CellGraphDataset":
        """
        One graph per DataFrame (one DataFrame per image). Standardization
        (mean/std) is fit once, pooling cells across all given images, then
        applied identically to every graph.
        """
        if feature_cols is None:
            feature_cols = MORPHOLOGY_COLS + marker_cols(dataframes[0], "mean")
        if image_ids is None:
            image_ids = range(len(dataframes))

        pooled = pd.concat([df[feature_cols] for df in dataframes], axis=0)
        mean = pooled.to_numpy(dtype=np.float32).mean(axis=0)
        std = pooled.to_numpy(dtype=np.float32).std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)

        graphs = []
        for df, image_id in zip(dataframes, image_ids):
            data, _, _ = dataframe_to_graph(
                df, feature_cols, k=k, image_id=image_id, mean=mean, std=std
            )
            graphs.append(data)

        return cls(graphs, feature_cols=feature_cols, feature_mean=mean, feature_std=std)

    def len(self) -> int:
        return len(self._graphs)

    def get(self, idx: int) -> Data:
        return self._graphs[idx]


# ==============================================================================
# modeling.py -- shared training / evaluation / visualization utilities
# ---------------------------------------------------------------------
# Shared training / evaluation / visualization utilities for the two
# feature-prediction notebooks:
#
# - intrinsic.ipynb  -- per-cell MLP: predict a cell's own features from its
#   OWN other features (morphology <-> immunofluorescence). Cell-autonomous.
# - extrinsic.ipynb  -- spatial GNN: predict a cell's own features from its
#   spatial NEIGHBORS' features only (see `tools.gnn.NeighborFeaturePredictor`).
#   Non-cell-autonomous / community effects.
#
# Both notebooks follow the same pattern: a small torch.nn.Module whose
# __init__ pins down which dataframe columns are "X" and which are "y", fit
# on a train split with per-column standardization (fit on train only, to
# avoid leakage), then evaluated/visualized/interpreted the same way. Keeping
# that shared plumbing here means the two notebooks stay focused on the
# actual scientific question rather than boilerplate.

def make_scaler(kind: str = "standard"):
    """
    `kind="standard"` (default): zero mean / unit variance per column. Good
    general-purpose choice, robust when different columns have very
    different absolute scales (e.g. `area` in the hundreds of pixels vs.
    `eccentricity` in [0, 1]) -- it puts every column on a comparable
    footing without assuming a fixed range.

    `kind="minmax"`: rescales each column into [0, 1] using the TRAIN
    split's min/max. Reasonable if you want everything on a literal 0-1
    scale (e.g. for direct comparison across features in a plot), but it's
    sensitive to outliers -- a single unusually large `area` value compresses
    every other cell's `area` toward 0.
    """
    if kind == "standard":
        return StandardScaler()
    if kind == "minmax":
        return MinMaxScaler()
    raise ValueError(f"kind must be 'standard' or 'minmax', got {kind!r}")


# --------------------------------------------------------------------------
# Reproducibility + splits
# --------------------------------------------------------------------------


def set_seed(seed: int = 0) -> None:
    """Seed python/numpy/torch (+ CUDA if present) for reproducible splits/init."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_indices(
    n: int, val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Random train/val/test index split. Used two ways in these notebooks:
      - intrinsic: each split is simply a subset of independent cell *rows*.
      - extrinsic: each split is a subset of *nodes* in one big spatial graph
        (transductive setting -- all nodes stay in the graph for message
        passing, only the loss/eval is masked to the relevant split).
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    test_idx = idx[:n_test]
    val_idx = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val :]
    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def indices_to_mask(idx: np.ndarray, n: int) -> torch.Tensor:
    mask = torch.zeros(n, dtype=torch.bool)
    mask[torch.as_tensor(idx, dtype=torch.long)] = True
    return mask


# --------------------------------------------------------------------------
# Shared EXPERIMENTS-dict helper (intrinsic.ipynb and extrinsic.ipynb each
# build an EXPERIMENTS dict of {name: config}, just with different config
# keys -- x_cols/y_cols for the same-cell mapper, neighbor_cols/target_cols
# for the neighbor-only GNN)
# --------------------------------------------------------------------------


def leave_one_out_configs(
    cols: Sequence[str],
    prefix: str,
    x_key: str = "x_cols",
    y_key: str = "y_cols",
) -> dict[str, dict]:
    """
    One config per column in `cols`: predict that column from all the OTHER
    columns in the same list -- e.g. predict T_mean from [HAND1_mean,
    BMP4_mean]. Tests redundancy/correlation within a single modality.

    `x_key`/`y_key` select which EXPERIMENTS convention to emit: the default
    `x_cols`/`y_cols` (intrinsic.ipynb, same-cell features) or
    `neighbor_cols`/`target_cols` (extrinsic.ipynb -- predicting a cell's own
    value from its neighbors' OTHER markers, excluding the neighbors' value of
    that same marker).
    """
    return {f"{prefix}_{col}": {x_key: [c for c in cols if c != col], y_key: [col]} for col in cols}


# --------------------------------------------------------------------------
# Weight initialization -- shared by every custom nn.Linear/Conv2d/ConvTranspose2d
# in this project (MLP here, DistanceAttentionConv/CellGNNWithHead/
# NeighborFeaturePredictor in tools/gnn.py, UNet in tools/cnn.py). All of our
# data is standardized (zero mean, unit variance) before it ever reaches a
# model, and every hidden layer is followed by ReLU, so Kaiming/He init (tuned
# for exactly that combination -- it keeps activation variance ~stable across
# layers at initialization, rather than shrinking/exploding) is the appropriate
# default. PyTorch's own default init (kaiming_uniform_ with a=sqrt(5)) predates
# He et al. and is tuned for neither case, so it's worth overriding explicitly
# rather than relying on it. The one exception is a final regression layer with
# NO activation after it (a prediction head, or a UNet's output conv) -- Kaiming
# assumes a ReLU follows and would otherwise leave the initial output variance
# too large; Xavier/Glorot is the correct choice there instead.
# --------------------------------------------------------------------------


def init_weights(module: nn.Module, nonlinearity: str = "relu") -> None:
    """Kaiming/He init for a Linear/Conv2d/ConvTranspose2d that feeds into a ReLU-
    family activation. Use via `model.apply(init_weights)`; biases start at zero."""
    if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity=nonlinearity)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def init_output_layer(layer: nn.Module) -> None:
    """Xavier/Glorot init for a final layer with NO activation following it (a
    regression head/output conv) -- call AFTER `model.apply(init_weights)`, to
    override that layer specifically."""
    nn.init.xavier_normal_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


# --------------------------------------------------------------------------
# Generic MLP block (shared by the intrinsic mapper and the GNN's readout)
# --------------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.1,
    ):
        super().__init__()
        dims = [in_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.BatchNorm1d(b), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(dims[-1], out_dim))
        self.net = nn.Sequential(*layers)

        self.apply(init_weights)  # Kaiming for every Linear (each is followed by ReLU except the last)
        init_output_layer(self.net[-1])  # override: final layer has no activation after it

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------------
# Intrinsic model: X and y are different columns of the SAME cell
# --------------------------------------------------------------------------


class TabularFeatureMapper(nn.Module):
    """
    Predicts `y_cols` from `x_cols` for the same cell -- e.g. morphology ->
    immunofluorescence, or the reverse. `x_cols`/`y_cols` are fixed at
    construction time, exactly like specifying a formula. Rescales both X
    and y column-by-column (fit on the training split only, see
    `make_scaler`) since morphology (pixels, ratios) and marker intensity
    (arbitrary fluorescence units, heavy-tailed) live on completely
    different scales -- pass `scaler_kind="minmax"` to rescale to [0, 1]
    instead of the default zero-mean/unit-variance standardization.

    Since `x_cols`/`y_cols` refer to the SAME cell's own features, any column
    appearing in both would let the model "predict" that column from itself
    -- a trivial same-cell leakage that a config edit could introduce by
    accident. Any such column is dropped from `x_cols` (with a warning)
    before the network is built.
    """

    def __init__(
        self,
        x_cols: Sequence[str],
        y_cols: Sequence[str],
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.1,
        scaler_kind: str = "standard",
    ):
        super().__init__()
        y_cols = list(y_cols)
        y_set = set(y_cols)
        overlap = [c for c in x_cols if c in y_set]
        x_cols = [c for c in x_cols if c not in y_set]
        if overlap:
            warnings.warn(
                f"x_cols/y_cols overlap on {overlap} -- dropped from x_cols to avoid "
                "predicting a column from itself. Pass disjoint x_cols/y_cols to silence this."
            )
        if not x_cols:
            raise ValueError("x_cols is empty after removing columns shared with y_cols")

        self.x_cols = x_cols
        self.y_cols = y_cols
        self.scaler_kind = scaler_kind
        self.x_scaler = make_scaler(scaler_kind)
        self.y_scaler = make_scaler(scaler_kind)
        self._fitted = False
        self.net = MLP(len(self.x_cols), len(self.y_cols), hidden_dims, dropout)

    def fit_scalers(self, df_train: pd.DataFrame) -> "TabularFeatureMapper":
        self.x_scaler.fit(df_train[self.x_cols].to_numpy(dtype=np.float32))
        self.y_scaler.fit(df_train[self.y_cols].to_numpy(dtype=np.float32))
        self._fitted = True
        return self

    def to_arrays(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        assert self._fitted, "call fit_scalers(df_train) first"
        X = self.x_scaler.transform(df[self.x_cols].to_numpy(dtype=np.float32)).astype(np.float32)
        y = self.y_scaler.transform(df[self.y_cols].to_numpy(dtype=np.float32)).astype(np.float32)
        return X, y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def predict_df(self, df: pd.DataFrame, device: str = "cpu", scaled: bool = False) -> pd.DataFrame:
        """
        Predict `y_cols` for `df`. By default returns ORIGINAL (unscaled) units,
        i.e. the same units as `df` itself. Pass `scaled=True` to instead get
        the standardized/min-max units the network actually sees -- useful for
        plots that put several targets with very different native scales
        (e.g. `area` in pixels vs. `eccentricity` in [0, 1]) on comparable axes.
        """
        self.eval()
        X = self.x_scaler.transform(df[self.x_cols].to_numpy(dtype=np.float32)).astype(np.float32)
        pred_scaled = self(torch.from_numpy(X).to(device)).cpu().numpy()
        if scaled:
            return pd.DataFrame(pred_scaled, columns=self.y_cols, index=df.index)
        pred = self.y_scaler.inverse_transform(pred_scaled)
        return pd.DataFrame(pred, columns=self.y_cols, index=df.index)

    def scale_x(self, df: pd.DataFrame) -> pd.DataFrame:
        """`x_cols` of `df` in the same standardized/min-max units the network was trained on."""
        X = self.x_scaler.transform(df[self.x_cols].to_numpy(dtype=np.float32))
        return pd.DataFrame(X, columns=self.x_cols, index=df.index)

    def scale_y(self, df: pd.DataFrame) -> pd.DataFrame:
        """`y_cols` of `df` in the same standardized/min-max units the network was trained on."""
        y = self.y_scaler.transform(df[self.y_cols].to_numpy(dtype=np.float32))
        return pd.DataFrame(y, columns=self.y_cols, index=df.index)


def train_feature_mapper(
    model: TabularFeatureMapper,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    epochs: int = 150,
    lr: float = 1e-3,
    batch_size: int = 512,
    weight_decay: float = 1e-5,
    patience: int = 15,
    device: str = "cpu",
    desc: Optional[str] = None,
) -> dict[str, list[float]]:
    """Minibatch Adam training with early stopping on val loss. Returns loss history."""
    model.fit_scalers(df_train)
    X_train, y_train = model.to_arrays(df_train)
    X_val, y_val = model.to_arrays(df_val)

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    n = X_train_t.shape[0]
    best_val, best_state, bad_epochs = float("inf"), None, 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    desc = desc or f"{'+'.join(model.x_cols[:2])}.. -> {'+'.join(model.y_cols[:2])}.."

    pbar = trange(epochs, desc=desc, leave=True)
    for _ in pbar:
        model.train()
        perm = torch.randperm(n)
        running = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb, yb = X_train_t[idx].to(device), y_train_t[idx].to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            opt.step()
            running += loss.item() * len(idx)
        train_loss = running / n

        model.eval()
        with torch.no_grad():
            val_loss = F.mse_loss(model(X_val_t), y_val_t).item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        pbar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}")

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


# --------------------------------------------------------------------------
# Transductive GNN training (single big graph, mask out train/val/test loss)
# --------------------------------------------------------------------------


def train_gnn(
    model: nn.Module,
    data,
    y: torch.Tensor,
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
    Full-batch training on one spatial graph. All nodes are always visible
    to message passing (transductive); the loss (and therefore gradients)
    only ever sees `train_mask` nodes, `val_mask` is used purely for early
    stopping / model selection.
    """
    model = model.to(device)
    data = data.to(device)
    y = y.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val, best_state, bad_epochs = float("inf"), None, 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    pbar = trange(epochs, desc=desc, leave=True)
    for _ in pbar:
        model.train()
        opt.zero_grad()
        pred, _ = model(data.x, data.edge_index, data.edge_attr)
        loss = F.mse_loss(pred[train_mask], y[train_mask])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            pred, _ = model(data.x, data.edge_index, data.edge_attr)
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


# --------------------------------------------------------------------------
# Inductive GNN training (MANY independent small graphs, e.g. one per image
# patch -- see cropped.ipynb -- rather than one big transductive graph)
# --------------------------------------------------------------------------


def train_patch_gnn(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 12,
    device: str = "cpu",
    desc: str = "training",
) -> dict[str, list[float]]:
    """
    Minibatch training over many independent graphs (e.g. one per image patch),
    each minibatch a disjoint union of several graphs via torch_geometric's
    `Batch` (handled automatically by `torch_geometric.loader.DataLoader`).
    Contrast with `train_gnn` above: there, one transductive graph is trained
    with train/val NODE masks; here, whole GRAPHS are held out for val/test via
    separate DataLoaders (see `split_patches` in tools/patches.py).

    Expects `model(x, edge_index) -> (pred, embedding)`, matching
    `CellGNN`/`CellGNNWithHead` (self+neighbor mixing) -- NOT the edge_attr-aware
    `NeighborFeaturePredictor` (extrinsic.ipynb's neighbor-ONLY model), which
    also needs `edge_attr` passed through.
    """
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val, best_state, bad_epochs = float("inf"), None, 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    pbar = trange(epochs, desc=desc, leave=True)
    for _ in pbar:
        model.train()
        running, n = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            pred, _ = model(batch.x, batch.edge_index)
            loss = F.mse_loss(pred, batch.y)
            loss.backward()
            opt.step()
            running += loss.item() * batch.num_nodes
            n += batch.num_nodes
        train_loss = running / n

        model.eval()
        val_running, val_n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred, _ = model(batch.x, batch.edge_index)
                val_running += F.mse_loss(pred, batch.y, reduction="sum").item()
                val_n += batch.y.numel()
        val_loss = val_running / val_n

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        pbar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}")

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


@torch.no_grad()
def predict_patch_gnn(model: nn.Module, loader, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """
    Runs `model` over every graph in `loader` and concatenates (pred, true) across
    all of them, in the STANDARDIZED units the model trained in -- inverse-
    transform with the y-scaler for original units (see cropped.ipynb).
    """
    model.eval()
    preds, trues = [], []
    for batch in loader:
        batch = batch.to(device)
        pred, _ = model(batch.x, batch.edge_index)
        preds.append(pred.cpu().numpy())
        trues.append(batch.y.cpu().numpy())
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)


# --------------------------------------------------------------------------
# Visualization
# --------------------------------------------------------------------------


def plot_loss_curves(histories: dict[str, dict[str, list[float]]], title: str = "") -> plt.Figure:
    """`histories`: {run_name: {"train_loss": [...], "val_loss": [...]}}"""
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for (name, hist), color in zip(histories.items(), colors):
        epochs = np.arange(1, len(hist["train_loss"]) + 1)
        ax.plot(epochs, hist["train_loss"], color=color, label=f"{name} (train)")
        ax.plot(epochs, hist["val_loss"], color=color, linestyle="--", label=f"{name} (val)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_pred_vs_actual(
    y_true_df: pd.DataFrame, y_pred_df: pd.DataFrame, cols: Optional[Sequence[str]] = None, title: str = "", max_cols: int = 5, s=5,
) -> plt.Figure:
    cols = list(cols) if cols is not None else list(y_true_df.columns)
    n_features = len(cols)
    n_cols = max(1, min(n_features, max_cols))
    n_rows = max(1, (n_features + n_cols - 1) // n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    for i, col in enumerate(cols):
        ax = axes_flat[i]
        yt = y_true_df[col].to_numpy()
        yp = y_pred_df[col].to_numpy()
        ax.scatter(yt, yp, s=s, alpha=0.9, linewidths=0)
        lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
        r2 = r2_score(yt, yp)
        mae = mean_absolute_error(yt, yp)
        ax.set_xlabel(f"actual {col}")
        ax.set_ylabel(f"predicted {col}")
        ax.set_title(f"{col}\n$R^2$={r2:.3f}  MAE={mae:.3g}")

    # Remove any unused axes from the grid
    for j in range(n_features, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_residuals(
    y_true_df: pd.DataFrame, y_pred_df: pd.DataFrame, cols: Optional[Sequence[str]] = None, title: str = "", max_cols: int = 5
) -> plt.Figure:
    cols = list(cols) if cols is not None else list(y_true_df.columns)
    n_features = len(cols)
    n_cols = max(1, min(n_features, max_cols))
    n_rows = max(1, (n_features + n_cols - 1) // n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    for i, col in enumerate(cols):
        ax = axes_flat[i]
        yt = y_true_df[col].to_numpy()
        resid = y_pred_df[col].to_numpy() - yt
        ax.scatter(yt, resid, s=3, alpha=0.15, linewidths=0)
        ax.axhline(0, color="r", linestyle="--", linewidth=1)
        ax.set_xlabel(f"actual {col}")
        ax.set_ylabel("residual (pred - actual)")
        ax.set_title(col)

    # Remove any unused axes from the grid
    for j in range(n_features, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def r2_table(y_true_df: pd.DataFrame, y_pred_df: pd.DataFrame, cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    cols = list(cols) if cols is not None else list(y_true_df.columns)
    rows = []
    for col in cols:
        yt, yp = y_true_df[col].to_numpy(), y_pred_df[col].to_numpy()
        rows.append({"feature": col, "R2": r2_score(yt, yp), "MAE": mean_absolute_error(yt, yp)})
    return pd.DataFrame(rows).set_index("feature")


def plot_feature_importance(
    importance_df: pd.DataFrame, target_col: Optional[str] = None, top_n: Optional[int] = None, title: str = ""
) -> plt.Figure:
    """`importance_df` has columns: feature, target, importance_mean, importance_std."""
    df = importance_df if target_col is None else importance_df[importance_df["target"] == target_col]
    agg = df.groupby("feature")["importance_mean"].mean().sort_values()
    err = df.groupby("feature")["importance_std"].mean().reindex(agg.index)
    if top_n:
        agg, err = agg.tail(top_n), err.tail(top_n)
    fig, ax = plt.subplots(figsize=(6, 0.4 * len(agg) + 1.5))
    ax.barh(agg.index, agg.values, xerr=err.values, color="steelblue")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("mean drop in $R^2$ when permuted (higher = more informative)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_feature_importance_by_target(
    importance_df: pd.DataFrame,
    top_n_features: Optional[int] = None,
    title: str = "",
) -> plt.Figure:
    """
    Every (feature, target) importance at once: one group of bars per TARGET
    on the x-axis, one color per input FEATURE within each group. Complements
    `plot_feature_importance`'s single-target/averaged views by making it easy
    to spot patterns like "this one neighbor feature matters for every target"
    vs. "this feature only matters for eccentricity".

    `top_n_features`: if the feature list is long, keep only the N features
    with the largest importance averaged across all targets (still computed
    from every feature -- this only controls how many bars/colors get drawn).
    """
    pivot = importance_df.pivot(index="target", columns="feature", values="importance_mean")
    err_pivot = importance_df.pivot(index="target", columns="feature", values="importance_std")

    if top_n_features:
        keep = pivot.abs().mean(axis=0).sort_values(ascending=False).index[:top_n_features]
        pivot, err_pivot = pivot[keep], err_pivot[keep]

    targets = list(pivot.index)
    features = list(pivot.columns)
    n_targets, n_features = len(targets), len(features)

    x = np.arange(n_targets)
    width = 0.8 / max(n_features, 1)
    cmap = plt.get_cmap("tab20" if n_features > 10 else "tab10")

    fig, ax = plt.subplots(figsize=(max(7, n_targets * 1.3), 5))
    for i, feat in enumerate(features):
        offset = (i - (n_features - 1) / 2) * width
        ax.bar(
            x + offset,
            pivot[feat].to_numpy(),
            width=width,
            yerr=err_pivot[feat].to_numpy(),
            label=feat,
            color=cmap(i % cmap.N),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=30, ha="right")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("mean drop in $R^2$ when permuted")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=1, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    cols: Sequence[str],
    title: str = "",
    cmap: str = "RdBu_r",
    annot: bool = True,
    corr_type: str = 'pearson',
    ax: Optional[plt.Axes] = None,
) -> tuple[plt.Figure, pd.DataFrame]:
    """
    Correlation matrix across `cols` (Pearson or Spearman), rendered as a heatmap
    (red = positively correlated, blue = anti-correlated). A fast, training-free
    complement to the permutation-importance/leave-one-out experiments --
    e.g. `area` and `equivalent_diameter` should show up strongly correlated
    since they're nearly the same quantity, while a real biological
    relationship (e.g. two markers of different lineages) might show up
    anti-correlated. Returns the figure AND the correlation matrix itself
    (feed it to `top_correlated_pairs` for a ranked list).
    """
    assert corr_type in ['pearson', 'spearman']
    corr = df[cols].corr(method=corr_type)
    n = len(cols)
    owns_fig = ax is None
    if owns_fig:
        fig, ax = plt.subplots(figsize=(0.55 * n + 2, 0.55 * n + 2))
    else:
        fig = ax.get_figure()
    im = ax.imshow(corr.to_numpy(), cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(cols)
    if annot:
        for i in range(n):
            for j in range(n):
                val = corr.iloc[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(val) > 0.6 else "black")
    cbar_label = "Pearson $r$" if corr_type == 'pearson' else "Spearman $\\rho$"
    fig.colorbar(im, ax=ax, label=cbar_label, shrink=0.5)
    ax.set_title(title)
    if owns_fig:
        fig.tight_layout()
    return fig, corr


def top_correlated_pairs(corr: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """
    Every feature pair (each unordered pair once, no self-pairs) from a
    correlation matrix (as returned by `plot_correlation_heatmap`), sorted by
    |r| descending -- the strongest correlations and anti-correlations at
    the top, regardless of sign.
    """
    cols = list(corr.columns)
    rows = [
        {"feature_1": cols[i], "feature_2": cols[j], "r": corr.iloc[i, j]}
        for i in range(len(cols))
        for j in range(i + 1, len(cols))
    ]
    return (
        pd.DataFrame(rows)
        .sort_values("r", key=lambda s: s.abs(), ascending=False)
        .reset_index(drop=True)
        .head(n)
    )


# --------------------------------------------------------------------------
# Spatial autocorrelation (Moran's I) -- the spatial analog of
# plot_correlation_heatmap/top_correlated_pairs for extrinsic.ipynb: instead
# of "do these two FEATURES move together across cells", "does this one
# feature's value move together across NEIGHBORING cells".
# --------------------------------------------------------------------------


def _edge_index_to_numpy(edge_index) -> np.ndarray:
    if hasattr(edge_index, "numpy"):
        edge_index = edge_index.detach().cpu().numpy()
    return np.asarray(edge_index)


def morans_i(values: np.ndarray, edge_index, weights: Optional[np.ndarray] = None) -> float:
    """
    Global Moran's I for one feature, using the (symmetric, no self-loops)
    graph in `edge_index` (shape (2, E), e.g. from `knn_edge_index` or the
    `edge_index` already built in extrinsic.ipynb) as the spatial weight
    matrix. `weights` (optional, shape (E,), e.g. `edge_attr` for an
    inverse-distance-style weighting) defaults to 1 for every edge (every
    neighbor counts equally regardless of exact distance).

    I > 0: neighboring cells tend to have similar values (spatially
    clustered/patchy). I < 0: neighboring cells tend to differ (spatially
    dispersed, checkerboard-like). I ~ 0: no spatial structure -- a cell's
    value tells you nothing about its neighbors'.
    """
    x = np.asarray(values, dtype=np.float64)
    n = x.shape[0]
    z = x - x.mean()

    src, dst = _edge_index_to_numpy(edge_index)
    w = np.ones(src.shape[0], dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)

    denom = (z**2).sum()
    if denom == 0:
        return 0.0
    return (n / w.sum()) * (w * z[src] * z[dst]).sum() / denom


def morans_i_table(
    df: pd.DataFrame,
    cols: Sequence[str],
    edge_index,
    weights: Optional[np.ndarray] = None,
    n_permutations: int = 99,
    seed: int = 0,
) -> pd.DataFrame:
    """
    `morans_i` for every column in `cols`, with a permutation-based p-value
    (shuffle which cell has which value, recompute I, repeat
    `n_permutations` times -- there's no simple closed-form null under an
    arbitrary graph, so this is the standard way to assess significance).
    Sorted by I descending (most spatially clustered first).
    """
    rng = np.random.default_rng(seed)
    src, dst = _edge_index_to_numpy(edge_index)
    w = np.ones(src.shape[0], dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)
    n = len(df)
    W = w.sum()

    rows = []
    for col in cols:
        x = df[col].to_numpy(dtype=np.float64)
        z = x - x.mean()
        denom = (z**2).sum()
        observed = (n / W) * (w * z[src] * z[dst]).sum() / denom if denom else 0.0

        null = np.empty(n_permutations)
        for p in range(n_permutations):
            z_perm = rng.permutation(z)
            null[p] = (n / W) * (w * z_perm[src] * z_perm[dst]).sum() / denom if denom else 0.0

        p_value = (np.sum(np.abs(null) >= abs(observed)) + 1) / (n_permutations + 1)
        rows.append(
            {"feature": col, "morans_i": observed, "null_mean": null.mean(), "null_std": null.std(), "p_value": p_value}
        )
    return pd.DataFrame(rows).set_index("feature").sort_values("morans_i", ascending=False)


def plot_morans_i(morans_df: pd.DataFrame, title: str = "", alpha: float = 0.05, ax: Optional[plt.Axes] = None) -> plt.Figure:
    """Bar chart of Moran's I per feature (from `morans_i_table`), colored by sign
    (clustered vs. dispersed), with `*` marking permutation-test p < `alpha`."""
    df = morans_df.sort_values("morans_i")
    colors = ["crimson" if v < 0 else "steelblue" for v in df["morans_i"]]
    owns_fig = ax is None
    if owns_fig:
        fig, ax = plt.subplots(figsize=(6, 0.4 * len(df) + 1.5))
    else:
        fig = ax.get_figure()
    ax.barh(df.index, df["morans_i"], color=colors)
    for i, (_, row) in enumerate(df.iterrows()):
        if row["p_value"] < alpha:
            offset = 0.01 if row["morans_i"] >= 0 else -0.01
            ax.text(row["morans_i"] + offset, i, "*", va="center", ha="left" if offset > 0 else "right", fontsize=12)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Moran's I (spatial autocorrelation among k-NN neighbors)")
    ax.set_title(title)
    if owns_fig:
        fig.tight_layout()
    return fig


def spatial_cross_correlation_matrix(
    df: pd.DataFrame,
    cols: Sequence[str],
    edge_index,
    weights: Optional[np.ndarray] = None,
    corr_type: str = "pearson",
) -> pd.DataFrame:
    """
    Bivariate spatial cross-correlation between every PAIR of `cols` -- e.g.
    does a high `BMP4_mean` at a cell tend to sit next to a high (or low)
    `eccentricity` at its neighbors, not just at the same cell? For features
    a, b:

        L_ab = (n / W) * sum_edges w_ij * z_a[i] * z_b[j]  /  sqrt(sum z_a^2 * sum z_b^2)

    Symmetric (L_ab == L_ba, since every edge (i, j) in `edge_index` has a
    matching (j, i)) and bounded roughly in [-1, 1], directly comparable to
    `plot_correlation_heatmap`'s Pearson matrix. The diagonal (a == b) is
    exactly the ordinary single-feature Moran's I (`morans_i`) for that
    feature -- so this one matrix subsumes both "is this feature spatially
    clustered" and "do these two features cluster together in space".
    """
    assert corr_type in ['pearson', 'spearman']
    src, dst = _edge_index_to_numpy(edge_index)
    w = np.ones(src.shape[0], dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)
    W = w.sum()
    n = len(df)

    if corr_type == "pearson":
        values = {c: df[c].to_numpy(dtype=np.float64) for c in cols}
    elif corr_type == "spearman":
        # ranks, not sort positions -- argsort() gives the permutation that would
        # sort the array, not each element's rank among the others
        values = {c: df[c].rank().to_numpy(dtype=np.float64) for c in cols}
    z = {c: values[c] - values[c].mean() for c in cols}
    ss = {c: (z[c] ** 2).sum() for c in cols}

    mat = np.zeros((len(cols), len(cols)), dtype=np.float64)
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if j < i:
                mat[i, j] = mat[j, i]  # symmetric -- already computed
                continue
            denom = np.sqrt(ss[a] * ss[b])
            mat[i, j] = (n / W) * (w * z[a][src] * z[b][dst]).sum() / denom if denom else 0.0
    return pd.DataFrame(mat, index=list(cols), columns=list(cols))


def plot_spatial_cross_correlation(
    df: pd.DataFrame,
    cols: Sequence[str],
    edge_index,
    weights: Optional[np.ndarray] = None,
    title: str = "",
    cmap: str = "RdBu_r",
    annot: bool = True,
    corr_type: str = "pearson",
    ax: Optional[plt.Axes] = None,
) -> tuple[plt.Figure, pd.DataFrame]:
    """
    Heatmap version of `spatial_cross_correlation_matrix` -- same layout as
    `plot_correlation_heatmap`, but each cell asks whether feature A at a
    cell sits next to feature B at its NEIGHBORS, rather than whether A and B
    move together within the same cell. The diagonal is the ordinary
    (univariate) Moran's I for that one feature. Returns the figure AND the
    matrix (feed it to `top_correlated_pairs` for a ranked list of the
    strongest spatial cross-correlations/anti-correlations, same as you'd do
    with the Pearson correlation matrix).
    """
    mat = spatial_cross_correlation_matrix(df, cols, edge_index, weights=weights, corr_type=corr_type)
    n = len(cols)
    owns_fig = ax is None
    if owns_fig:
        fig, ax = plt.subplots(figsize=(0.55 * n + 2, 0.55 * n + 2))
    else:
        fig = ax.get_figure()
    im = ax.imshow(mat.to_numpy(), cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(cols)
    if annot:
        for i in range(n):
            for j in range(n):
                val = mat.iloc[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(val) > 0.6 else "black")
    fig.colorbar(im, ax=ax, label="Bivariate Moran's I", shrink=0.5)
    ax.set_title(title)
    if owns_fig:
        fig.tight_layout()
    return fig, mat


def plot_experiment_comparison(
    results: dict[str, dict],
    target_cols_by_experiment: dict[str, Sequence[str]],
    title: str = "",
) -> pd.DataFrame:
    """
    One heatmap of test R^2 across every experiment in `results` (as produced by
    `run_neighbor_experiment`/`run_tabular_experiment`'s `EXPERIMENTS` loop):
    rows = every target column that appears in ANY experiment (first-seen
    order), columns = experiment (in `results` order), cell = that experiment's
    test R^2 for that target -- blank/gray wherever an experiment doesn't
    predict that target (e.g. a leave-one-out run only covers one target).
    Answers "which experiment predicts which target well" in a single glance,
    rather than one bar chart per target-group.

    `target_cols_by_experiment`: {experiment_name: target_cols}, e.g.
    `{name: cfg['target_cols'] for name, cfg in EXPERIMENTS.items()}`.

    Calls `plt.show()` itself and returns the R^2 matrix shown in the heatmap
    (pass it to `display` for the exact numbers alongside the figure).
    """
    rows = []
    for name, res in results.items():
        target_cols = list(target_cols_by_experiment[name])
        r2 = res["r2"]["R2"]
        for target in target_cols:
            rows.append({"experiment": name, "target": target, "R2": r2.loc[target]})
    long_df = pd.DataFrame(rows)

    target_order = list(dict.fromkeys(t for cols in target_cols_by_experiment.values() for t in cols))
    exp_order = list(target_cols_by_experiment.keys())
    pivot = long_df.pivot(index="target", columns="experiment", values="R2").reindex(
        index=target_order, columns=exp_order
    )

    n_targets, n_exps = pivot.shape
    fig, ax = plt.subplots(figsize=(0.9 * n_exps + 3, 0.5 * n_targets + 2))
    masked = np.ma.masked_invalid(pivot.to_numpy())
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="0.85")
    im = ax.imshow(np.clip(masked, -1, 1), cmap=cmap, vmin=-1, vmax=1)

    ax.set_xticks(range(n_exps))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(n_targets))
    ax.set_yticklabels(pivot.index)
    for i in range(n_targets):
        for j in range(n_exps):
            val = pivot.iat[i, j]
            if pd.notna(val):
                ax.text(
                    j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(val) > 0.6 else "black",
                )
    fig.colorbar(im, ax=ax, label="test $R^2$ (clipped to [-1, 1])", shrink=0.6)
    ax.set_title(title)
    fig.tight_layout()
    plt.show()

    return pivot


# --------------------------------------------------------------------------
# Feature importance via permutation
# --------------------------------------------------------------------------


def permutation_importance_mlp(
    model: TabularFeatureMapper,
    df: pd.DataFrame,
    n_repeats: int = 5,
    seed: int = 0,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    For each input column, shuffle it (breaking its link to the target)
    and measure the drop in test R^2 per target column, averaged over
    `n_repeats` shuffles. Operates in standardized space -- R^2 is
    invariant to the per-column affine rescaling, so this is identical to
    computing it in original units.
    """
    model.eval()
    rng = np.random.default_rng(seed)
    X, y = model.to_arrays(df)
    with torch.no_grad():
        base_pred = model(torch.from_numpy(X).to(device)).cpu().numpy()
    base_r2 = np.atleast_1d(r2_score(y, base_pred, multioutput="raw_values"))

    rows = []
    for j, fname in enumerate(model.x_cols):
        drops = np.zeros((n_repeats, len(model.y_cols)))
        for r in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
            with torch.no_grad():
                pred = model(torch.from_numpy(Xp).to(device)).cpu().numpy()
            score = np.atleast_1d(r2_score(y, pred, multioutput="raw_values"))
            drops[r] = base_r2 - score
        for t, tname in enumerate(model.y_cols):
            rows.append(
                {
                    "feature": fname,
                    "target": tname,
                    "importance_mean": drops[:, t].mean(),
                    "importance_std": drops[:, t].std(),
                }
            )
    return pd.DataFrame(rows)


def permutation_importance_gnn(
    model: nn.Module,
    data,
    y: torch.Tensor,
    mask: torch.Tensor,
    feature_names: Sequence[str],
    target_names: Sequence[str],
    n_repeats: int = 5,
    seed: int = 0,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Same idea as `permutation_importance_mlp`, but the thing being shuffled
    is a NODE feature column -- i.e. we scramble which cell has which value
    for that feature, then re-run message passing and see how much worse
    the *center* cell's prediction gets. This tells you which of the
    neighbors' features actually carry information about the center cell,
    as opposed to just being structurally present in the graph.
    """
    model.eval()
    rng = np.random.default_rng(seed)
    data = data.to(device)
    y = y.to(device)
    mask = mask.to(device)

    with torch.no_grad():
        base_pred, _ = model(data.x, data.edge_index, data.edge_attr)
    y_true = y[mask].cpu().numpy()
    base_r2 = np.atleast_1d(r2_score(y_true, base_pred[mask].cpu().numpy(), multioutput="raw_values"))

    x_np = data.x.cpu().numpy()
    rows = []
    for j, fname in enumerate(feature_names):
        drops = np.zeros((n_repeats, len(target_names)))
        for r in range(n_repeats):
            perm = rng.permutation(x_np.shape[0])
            x_perm = data.x.clone()
            x_perm[:, j] = torch.from_numpy(x_np[perm, j]).to(device)
            with torch.no_grad():
                pred_perm, _ = model(x_perm, data.edge_index, data.edge_attr)
            score = np.atleast_1d(r2_score(y_true, pred_perm[mask].cpu().numpy(), multioutput="raw_values"))
            drops[r] = base_r2 - score
        for t, tname in enumerate(target_names):
            rows.append(
                {
                    "feature": fname,
                    "target": tname,
                    "importance_mean": drops[:, t].mean(),
                    "importance_std": drops[:, t].std(),
                }
            )
    return pd.DataFrame(rows)


# ==============================================================================
# cnn.py -- pixel-level channel-to-channel U-Net (cropped.ipynb)
# --------------------------------------------------------------
# Pixel-level image-to-image channel translation on cropped patches (cropped.ipynb,
# Task 1): given one or more fluorescence channel IMAGES, predict another channel's
# image directly in pixel space -- the image analog of intrinsic.ipynb's per-cell
# tabular morphology<->IF mapping.
#
# A small U-Net (encoder-decoder with skip connections) is the standard architecture
# for this kind of dense image regression -- skip connections let it reproduce
# fine spatial detail (individual nuclei) while the bottleneck captures broader
# context (local tissue neighborhood/density).

# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    """
    Small U-Net predicting `out_channels` pixel maps from `in_channels` input
    channels. `base_channels` sets the width of the first stage (doubles every
    downsampling stage); `depth` controls how many downsample/upsample stages
    (2-3 is plenty for ~128-256px crops -- each stage halves spatial resolution,
    so `depth` stages need input dims divisible by 2**depth).
    """

    def __init__(self, in_channels: int, out_channels: int = 1, base_channels: int = 16, depth: int = 3):
        super().__init__()
        chs = [in_channels] + [base_channels * (2**i) for i in range(depth + 1)]

        self.downs = nn.ModuleList(ConvBlock(chs[i], chs[i + 1]) for i in range(depth))
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(chs[depth], chs[depth + 1])
        self.ups = nn.ModuleList(
            nn.ConvTranspose2d(chs[depth + 1 - i], chs[depth - i], 2, stride=2) for i in range(depth)
        )
        self.up_convs = nn.ModuleList(ConvBlock(chs[depth - i] * 2, chs[depth - i]) for i in range(depth))
        self.out_conv = nn.Conv2d(chs[1], out_channels, 1)

        self.apply(init_weights)  # Kaiming for every Conv2d/ConvTranspose2d (each feeds a ReLU)
        init_output_layer(self.out_conv)  # override: no activation follows the output conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, up_conv, skip in zip(self.ups, self.up_convs, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:  # odd input dims not divisible by 2**depth
                x = F.interpolate(x, size=skip.shape[-2:])
            x = torch.cat([x, skip], dim=1)
            x = up_conv(x)
        return self.out_conv(x)


# --------------------------------------------------------------------------
# Dataset + channel standardization (fit on TRAIN patches only, same
# discipline as make_scaler in tools/modeling.py)
# --------------------------------------------------------------------------


def fit_channel_stats(
    image: np.ndarray,
    cell_mask: np.ndarray,
    train_patches: list[dict],
    channel_indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-channel (mean, std) over IN-CELL pixels (`cell_mask` True) of the TRAIN
    patches only, for each channel in `channel_indices`. Returns (mean, std) arrays
    of shape (C, 1, 1), broadcastable against a (C, H, W) crop. Restricting to
    in-cell pixels (rather than every pixel) keeps background fluorescence --
    which we never train or evaluate on, see `PatchImageDataset` -- from skewing
    the standardization. Accumulates sum/sum-of-squares rather than concatenating
    pixels, since a few hundred patches' worth of pixels adds up fast.
    """
    means, stds = [], []
    for c in channel_indices:
        total, total_sq, count = 0.0, 0.0, 0
        for p in train_patches:
            vals = image[c, p["y0"] : p["y1"], p["x0"] : p["x1"]].astype(np.float64)
            m = cell_mask[p["y0"] : p["y1"], p["x0"] : p["x1"]]
            vals = vals[m]
            total += vals.sum()
            total_sq += np.square(vals).sum()
            count += vals.size
        mean = total / count if count > 0 else 0.0
        var = max(total_sq / count - mean**2, 0.0) if count > 0 else 1.0
        means.append(mean)
        stds.append(np.sqrt(var))
    mean = np.array(means, dtype=np.float32).reshape(-1, 1, 1)
    std = np.array(stds, dtype=np.float32).reshape(-1, 1, 1)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


class PatchImageDataset(torch.utils.data.Dataset):
    """
    One sample = one cropped image patch. `__getitem__` returns (x, y, cell_mask):
    `x` stacks `input_channels`, `y` is the single `target_channel` (both
    standardized with `x_mean`/`x_std`/`y_mean`/`y_std` -- from `fit_channel_stats`
    on TRAIN patches, in-cell pixels only), and `cell_mask` is a (1, H, W) float
    tensor, 1 where the pixel falls inside a real (kept, non-filtered) segmented
    cell and 0 for background. `x` itself is NOT masked -- the network still sees
    the whole patch as spatial context -- but `cell_mask` is what restricts the
    training loss and evaluation (`train_cnn`/`pixel_r2`) to actual cell pixels,
    so background fluorescence never counts toward "predicting immunofluorescence."
    """

    def __init__(
        self,
        image: np.ndarray,
        cell_mask: np.ndarray,
        patches: list[dict],
        input_channels: list[str],
        target_channel: str,
        channel_names: list[str],
        x_mean: np.ndarray,
        x_std: np.ndarray,
        y_mean: np.ndarray,
        y_std: np.ndarray,
    ):
        self.image = image
        self.cell_mask = cell_mask
        self.patches = patches
        self.input_idx = [channel_names.index(c) for c in input_channels]
        self.target_idx = channel_names.index(target_channel)
        self.x_mean, self.x_std = x_mean, x_std
        self.y_mean, self.y_std = y_mean, y_std

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p = self.patches[idx]
        crop = self.image[:, p["y0"] : p["y1"], p["x0"] : p["x1"]].astype(np.float32)
        mask_crop = self.cell_mask[p["y0"] : p["y1"], p["x0"] : p["x1"]].astype(np.float32)[None]
        x = (crop[self.input_idx] - self.x_mean) / self.x_std
        y = (crop[self.target_idx : self.target_idx + 1] - self.y_mean) / self.y_std
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask_crop)


# --------------------------------------------------------------------------
# Training + evaluation
# --------------------------------------------------------------------------


def train_cnn(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int = 60,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 8,
    device: str = "cpu",
    desc: str = "training",
) -> dict[str, list[float]]:
    """
    Minibatch Adam training with early stopping on val loss -- pixel-wise MSE,
    restricted to in-cell pixels via each batch's `cell_mask` (see
    `PatchImageDataset`) so background fluorescence never contributes to the loss.
    """
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val, best_state, bad_epochs = float("inf"), None, 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    pbar = trange(epochs, desc=desc, leave=True)
    for _ in pbar:
        model.train()
        running, n = 0.0, 0
        for xb, yb, mb in train_loader:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            opt.zero_grad()
            pred = model(xb)
            n_valid = mb.sum().clamp(min=1)
            loss = ((pred - yb) ** 2 * mb).sum() / n_valid
            loss.backward()
            opt.step()
            running += loss.item() * n_valid.item()
            n += n_valid.item()
        train_loss = running / max(n, 1)

        model.eval()
        val_running, val_n = 0.0, 0
        with torch.no_grad():
            for xb, yb, mb in val_loader:
                xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                pred = model(xb)
                val_running += ((pred - yb) ** 2 * mb).sum().item()
                val_n += mb.sum().item()
        val_loss = val_running / max(val_n, 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        pbar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}")

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


def pixel_r2(model: nn.Module, loader: torch.utils.data.DataLoader, device: str = "cpu") -> float:
    """
    Pixel-wise R^2 (standardized units), pooled across every IN-CELL pixel
    (`cell_mask` True) of every patch in `loader` -- background pixels are
    excluded, same restriction as the training loss in `train_cnn`.
    """
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb, mb in loader:
            pred = model(xb.to(device)).cpu().numpy()
            mask_np = mb.numpy().astype(bool)
            preds.append(pred[mask_np])
            trues.append(yb.numpy()[mask_np])
    return r2_score(np.concatenate(trues), np.concatenate(preds))


@torch.no_grad()
def predict_patch(model: nn.Module, dataset: PatchImageDataset, idx: int, device: str = "cpu") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict a single patch (standardized units). Returns (x, y_true, y_pred,
    cell_mask), each a numpy array -- `x` is (C_in, H, W), the rest are (1, H, W).
    `cell_mask` marks which pixels actually counted toward training/evaluation.
    """
    model.eval()
    x, y, mask = dataset[idx]
    pred = model(x.unsqueeze(0).to(device)).cpu().numpy()[0]
    return x.numpy(), y.numpy(), pred, mask.numpy()


# ==============================================================================
# gnn.py -- cell graph neural nets
# --------------------------------
# GNN scaffold for learning over cell graphs (nodes = cells, edges = spatial
# neighbors; see dataset.CellGraphDataset).
#
# `CellGNN` is the encoder: stack of graph conv layers producing one embedding
# per cell from its morphology + marker features. `CellGNNWithHead` adds a
# linear readout on top for whatever supervised task you plug in later (e.g.
# predicting a discrete cell state/fate from morphology+markers, or regressing
# a marker from the others) -- swap `num_outputs` and the loss to match.

CONV_LAYERS = {"sage": SAGEConv, "gcn": GCNConv, "gat": GATConv}


class CellGNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 32,
        num_layers: int = 3,
        conv_type: str = "sage",
        dropout: float = 0.1,
    ):
        super().__init__()
        if conv_type not in CONV_LAYERS:
            raise ValueError(f"conv_type must be one of {list(CONV_LAYERS)}, got {conv_type!r}")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        conv_cls = CONV_LAYERS[conv_type]
        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]

        self.convs = nn.ModuleList(conv_cls(dims[i], dims[i + 1]) for i in range(num_layers))
        self.norms = nn.ModuleList(nn.BatchNorm1d(dims[i + 1]) for i in range(num_layers - 1))
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.norms[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x  # (num_nodes, out_channels) per-cell embedding


class CellGNNWithHead(nn.Module):
    """Encoder + linear readout for a node-level supervised task."""

    def __init__(
        self,
        in_channels: int,
        num_outputs: int,
        hidden_channels: int = 64,
        embedding_dim: int = 32,
        num_layers: int = 3,
        conv_type: str = "sage",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = CellGNN(
            in_channels, hidden_channels, embedding_dim, num_layers, conv_type, dropout
        )
        self.head = nn.Linear(embedding_dim, num_outputs)
        init_output_layer(self.head)  # no activation follows this layer

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        z = self.encoder(x, edge_index)
        return self.head(z), z


# --------------------------------------------------------------------------
# Neighbor-only, distance-aware conv (used by extrinsic.ipynb)
#
# `CellGNN` above mixes each cell's own features into its next-layer
# embedding (SAGEConv/GCNConv/GATConv all have a "self"/root term). That's
# the right choice for a general per-cell embedding, but it's the wrong
# choice if the scientific question is "how much of this cell's state is
# explained by its NEIGHBORS alone" (non-cell-autonomous / community
# effects) -- any self term would let the model partially answer that
# question by cheating and looking at the cell's own value.
#
# `DistanceAttentionConv` never includes x_i (the center/self node) in the
# message for node i, only x_j for its neighbors j. Neighbors are combined
# with attention weights that (optionally) decay with spatial distance:
# closer neighbors get a bigger say, exactly like a distance-limited
# paracrine or mechanical-coupling signal would. Setting `use_distance=False`
# makes the attention logits identical for every edge, which collapses the
# softmax to a plain unweighted mean over neighbors -- so toggling this flag
# is a clean, direct A/B test of "does distance matter" (see extrinsic.ipynb).
# --------------------------------------------------------------------------


class DistanceAttentionConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, use_distance: bool = True):
        super().__init__(aggr="add", node_dim=0)
        self.lin = nn.Linear(in_channels, out_channels)
        init_weights(self.lin)  # standardized input, ReLU follows (except possibly the last layer)
        self.use_distance = use_distance
        if use_distance:
            # exp(length_scale) starts around a typical neighbor distance
            # (tens of pixels here) so the initial attention isn't already
            # saturated to one-hot or perfectly uniform; the model can
            # still learn to sharpen or flatten it during training.
            self.length_scale = nn.Parameter(torch.tensor(25.0))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        x = self.lin(x)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        if self.use_distance:
            dist = edge_attr.squeeze(-1)
            logits = -dist / self.length_scale.abs().clamp(min=1e-3)
        else:
            logits = torch.zeros(edge_attr.size(0), device=edge_attr.device)
        alpha = softmax(logits, index)  # normalized per destination node -> weighted mean
        return x_j * alpha.unsqueeze(-1)


class NeighborOnlyGNN(nn.Module):
    """Stack of `DistanceAttentionConv` layers -- an embedding built purely from neighbors.

    Note on `num_layers`: with `num_layers=1`, node i's embedding is
    strictly a function of its direct neighbors' raw features -- the
    cleanest version of "what do my neighbors say about me". With
    `num_layers>=2` on an undirected graph, a 2-hop path (i -> j -> i) means
    j's layer-1 embedding already depends on i, so a little bit of i's own
    signal can leak back in indirectly. Default is 1 layer for that reason;
    increase only if you're aware of (and want) that tradeoff.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 1,
        use_distance: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        self.convs = nn.ModuleList(
            DistanceAttentionConv(dims[i], dims[i + 1], use_distance=use_distance) for i in range(num_layers)
        )
        self.norms = nn.ModuleList(nn.BatchNorm1d(dims[i + 1]) for i in range(num_layers - 1))
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            if i < len(self.convs) - 1:
                x = self.norms[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class NeighborFeaturePredictor(nn.Module):
    """Neighbor-only encoder + linear head: predict a cell's own features from its neighbors' features."""

    def __init__(
        self,
        in_channels: int,
        num_outputs: int,
        hidden_channels: int = 64,
        embedding_dim: int = 32,
        num_layers: int = 1,
        use_distance: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = NeighborOnlyGNN(
            in_channels, embedding_dim, hidden_channels, num_layers, use_distance, dropout
        )
        self.head = nn.Linear(embedding_dim, num_outputs)
        init_output_layer(self.head)  # no activation follows this layer

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        z = self.encoder(x, edge_index, edge_attr)
        return self.head(z), z


# ==============================================================================
# patches.py -- tiling a stitched image into patches
# --------------------------------------------------
# Tile a large stitched image (and its per-cell feature table) into many smaller,
# independent patches -- the training samples for cropped.ipynb.
#
# Why patches at all: a single stitched mosaic is one biological image, i.e. one
# sample. Neither a CNN (pixel-to-pixel) nor a per-patch graph model can be trained
# or *validated* on a sample size of one. Cropping into a grid of non-overlapping
# patches turns it into hundreds of independent samples, and -- because patches
# never overlap and each patch's k-NN graph only ever connects cells within that
# same patch -- a plain per-patch train/val/test split is already a genuine
# spatially-blocked split: held-out patches share no pixels, cells, or graph edges
# with training patches (contrast with intrinsic.ipynb/extrinsic.ipynb, where the
# split is over individual cells/nodes within one contiguous tissue).
#
# The one approximation this introduces: a cell near a patch's edge loses whatever
# true nearest neighbors would have fallen just across the boundary, in the
# adjacent patch. That's a boundary effect on graph connectivity, not a train/test
# leakage issue.

def make_patch_grid(image_shape: tuple[int, int], patch_size: int, stride: Optional[int] = None) -> list[dict]:
    """
    Returns one dict per patch: {row, col, y0, y1, x0, x1} (row/col = grid position,
    y/x = pixel bounds). `stride` defaults to `patch_size` (non-overlapping tiling);
    a smaller stride gives overlapping patches (more samples, but adjacent patches
    then DO share pixels/cells -- only use that for the CNN task, where a few
    overlapping training crops are harmless, not for the patch-graph task, where
    it would violate the "held-out patches share nothing" property above).

    Partial edge patches (where the image dimension isn't a multiple of
    `patch_size`) are dropped rather than padded, so every patch is the same size.
    """
    stride = stride or patch_size
    h, w = image_shape
    patches = []
    for row, y0 in enumerate(range(0, h - patch_size + 1, stride)):
        for col, x0 in enumerate(range(0, w - patch_size + 1, stride)):
            patches.append({"row": row, "col": col, "y0": y0, "y1": y0 + patch_size, "x0": x0, "x1": x0 + patch_size})
    return patches


def split_patches(patches: list[dict], val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 0) -> np.ndarray:
    """
    Assigns each patch (by position in `patches`) to 'train'/'val'/'test', shuffled
    at the PATCH level. Since patches are non-overlapping and self-contained (see
    module docstring), this is already a spatially-blocked split -- no need to
    additionally group patches into larger contiguous blocks.
    """
    n = len(patches)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    split = np.full(n, "train", dtype=object)
    split[idx[:n_test]] = "test"
    split[idx[n_test : n_test + n_val]] = "val"
    return split


def crop_image(image: np.ndarray, patch: dict) -> np.ndarray:
    """image: (C, H, W) -> (C, patch_size, patch_size)."""
    return image[:, patch["y0"] : patch["y1"], patch["x0"] : patch["x1"]]


def crop_mask(mask: np.ndarray, patch: dict) -> np.ndarray:
    """mask: (H, W) -> (patch_size, patch_size). For a single-channel array (e.g. a
    label mask or the boolean in-cell mask), unlike `crop_image` which expects (C, H, W)."""
    return mask[patch["y0"] : patch["y1"], patch["x0"] : patch["x1"]]


def cells_mask_from_labels(label_mask: np.ndarray, valid_cell_ids) -> np.ndarray:
    """
    Boolean (H, W) array, True wherever a pixel belongs to one of `valid_cell_ids`
    (e.g. `cell_df.index`) in the Cellpose `label_mask` -- False for background AND
    for any segmented blob that isn't in `valid_cell_ids` (e.g. filtered out by
    `features.filter_cells` as a fragment/sliver artifact). Used to restrict
    pixel-level training/evaluation (cropped.ipynb Task 1) to real, kept cells only.
    """
    valid_ids = np.asarray(list(valid_cell_ids))
    lookup = np.zeros(int(label_mask.max()) + 1, dtype=bool)
    lookup[valid_ids] = True
    return lookup[label_mask]


def crop_cell_df(cell_df: pd.DataFrame, patch: dict) -> pd.DataFrame:
    """
    Cells whose centroid falls inside this patch, with `centroid_y`/`centroid_x`
    shifted to be relative to the patch's own top-left corner (so downstream code
    -- e.g. `knn_edge_index` -- doesn't need to know the patch's position in the
    original mosaic).
    """
    in_patch = (
        (cell_df["centroid_y"] >= patch["y0"]) & (cell_df["centroid_y"] < patch["y1"]) &
        (cell_df["centroid_x"] >= patch["x0"]) & (cell_df["centroid_x"] < patch["x1"])
    )
    sub = cell_df.loc[in_patch].copy()
    sub["centroid_y"] -= patch["y0"]
    sub["centroid_x"] -= patch["x0"]
    return sub


# ==============================================================================
# features.py -- zero_out_negative_if (rest of features.py stays in tools/features.py,
# still used by segment.ipynb/stitch.ipynb)

def zero_out_negative_if(
    df: pd.DataFrame,
    if_channels: list[str],
    zero_out: bool = True,
) -> pd.DataFrame:
    """
    If `zero_out` is True, zero out each channel's mean-intensity column wherever its
    binarized positivity column is False -- e.g. sets `HAND1_mean` to 0 for every cell
    where `HAND1+` is False. Treats "IF-negative" (below the bimodal threshold used to
    build the `+` column) as true zero signal rather than whatever background/baseline
    intensity the mean happened to measure.

    `if_channels` are the mean-intensity column names (e.g. ['HAND1_mean', 'BMP4_mean']);
    the matching positivity column is assumed to be named '{channel}+' (dropping the
    '_mean' suffix), the same convention used when the '+' columns were built. A channel
    missing its '+' column is left untouched.

    No-op (returns `df` unchanged, no copy) when `zero_out` is False -- flip this one
    parameter to compare "IF-negative counts as zero" vs. "use the measured mean as-is"
    everywhere downstream, without touching the rest of the pipeline.
    """
    if not zero_out:
        return df
    df = df.copy()
    for mean_col in if_channels:
        pos_col = mean_col.replace("_mean", "+")
        if mean_col not in df.columns or pos_col not in df.columns:
            continue
        df.loc[~df[pos_col], mean_col] = 0.0
    return df

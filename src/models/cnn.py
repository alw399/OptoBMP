"""
Patch-grid (segmentation-free) CNN for predicting a patch's own immunofluorescence
from its spatially NEIGHBORING patches' immunofluorescence -- the patch-grid analog
of `models.gnn`'s cell-graph neighbor predictor, built for exactly the case where
cell segmentation itself is suspect: nothing here depends on a Cellpose mask, only
on tiling the raw multi-channel image into a fixed grid.

Same x_cols / neighbor_cols / y_cols split as `models.gnn.build_prediction_data`:
  - `x_cols`: channels the model sees directly from the CENTER patch's own pixels
    (its own CNN encoder branch). May be empty (`x_cols=[]`, the default) for a
    pure "do neighbors ALONE predict this patch" test -- no self-information at
    all, only `neighbor_cols` from the 8 surrounding patches.
  - `neighbor_cols`: channels seen only through the up-to-8 grid-adjacent NEIGHBOR
    patches (a shared CNN encoder per neighbor, mean-pooled across neighbors) --
    the center patch's own value for these channels is never used.
  - `y_cols` defaults to every IF channel not already in `x_cols` (every channel,
    if `x_cols=[]`); the prediction target is the center patch's own MEAN raw
    intensity per y_cols channel (the patch-level analog of a cell's mean IF).

A patch on the outer ring of the grid doesn't have a full 3x3 (Moore) neighborhood
-- part of it falls outside the tiled image, genuinely missing data, not empty --
so `border_patch_mask` flags these to be excluded as prediction TARGETS while still
serving as a neighbor for interior patches (mirrors `models.gnn.border_mask`).
"""

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import trange

from tools.morphology import make_patch_grid, init_weights, init_output_layer
from models.gnn import ColumnScaler


def _channel_index(col: str, channel_names: list[str]) -> int:
    """`x_cols`/`neighbor_cols`/`y_cols` use the '{channel}_mean' convention (same
    as `if_channels`/`extract_cell_features`'s columns); `channel_names` are the
    bare channel names indexing `image`'s first axis."""
    name = col[:-5] if col.endswith("_mean") else col
    return channel_names.index(name)


# --------------------------------------------------------------------------
# Patch grid + neighbor lookup + border exclusion
# --------------------------------------------------------------------------


def build_patch_neighbors(patches: list[dict]) -> tuple[dict[tuple[int, int], int], list[list[int]]]:
    """
    `patches` (see `tools.morphology.make_patch_grid`) is a flat list, each with a
    `row`/`col` grid position. Returns:
      grid_lookup : {(row, col): index into `patches`}
      neighbor_idx: for each patch, the indices (into `patches`) of its up-to-8
        grid-adjacent (Moore neighborhood) neighbors that actually exist.
    """
    grid_lookup = {(p["row"], p["col"]): i for i, p in enumerate(patches)}
    neighbor_idx = []
    for p in patches:
        r, c = p["row"], p["col"]
        idx = [
            grid_lookup[(r + dr, c + dc)]
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if not (dr == 0 and dc == 0) and (r + dr, c + dc) in grid_lookup
        ]
        neighbor_idx.append(idx)
    return grid_lookup, neighbor_idx


def border_patch_mask(neighbor_idx: list[list[int]]) -> np.ndarray:
    """
    True for any patch with fewer than 8 grid neighbors -- part of its Moore
    neighborhood falls outside the tiled grid (the outer ring), so its neighbor
    context is genuinely incomplete, not just sparse. Exclude these as prediction
    TARGETS (see `models.gnn.border_mask` for the cell-graph equivalent); they're
    still valid neighbors for whichever interior patch is adjacent to them.
    """
    return np.array([len(idx) < 8 for idx in neighbor_idx])


# --------------------------------------------------------------------------
# Pixel normalization -- every pixel counts, deliberately no cell mask (unlike
# `tools.morphology.fit_channel_stats`, which restricts to in-cell pixels)
# --------------------------------------------------------------------------


def fit_pixel_stats(image: np.ndarray, patches: Sequence[dict], channel_indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-channel (mean, std) of `log1p(pixel value)` over EVERY pixel of the given
    (TRAIN) `patches`, shape (C, 1, 1), broadcastable against a (C, H, W) crop.
    log1p first for the same reason `models.gnn.ColumnScaler` does -- raw
    fluorescence is heavily right-skewed by a few very bright pixels, which would
    otherwise dominate the mean/std. Accumulates sum/sum-of-squares rather than
    concatenating pixels (same discipline as `tools.morphology.fit_channel_stats`),
    since a few hundred patches' worth of pixels adds up fast.
    """
    means, stds = [], []
    for c in channel_indices:
        total, total_sq, count = 0.0, 0.0, 0
        for p in patches:
            vals = np.log1p(image[c, p["y0"] : p["y1"], p["x0"] : p["x1"]].astype(np.float64))
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


def normalize_patch(crop: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((np.log1p(crop.astype(np.float32)) - mean) / std).astype(np.float32)


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------


class PatchNeighborDataset(Dataset):
    """
    One sample = one (non-border) center patch. `__getitem__` returns:
      x_center   : (len(x_cols), patch_size, patch_size) -- center patch's OWN
                   pixels for x_cols channels, log1p+z-scored (`fit_pixel_stats`).
      x_neighbors: (8, len(neighbor_cols), patch_size, patch_size) -- always
                   exactly 8, since only non-border patches (`border_patch_mask`)
                   are ever used as centers.
      y          : (len(y_cols),) -- the center patch's own mean raw intensity per
                   y_cols channel, scaled by a `models.gnn.ColumnScaler` fit on
                   TRAIN centers only (see `build_patch_data`).
    """

    def __init__(
        self,
        image: np.ndarray,
        patches: list[dict],
        neighbor_idx: list[list[int]],
        center_indices: Sequence[int],
        x_channel_idx: Sequence[int],
        neighbor_channel_idx: Sequence[int],
        x_mean: np.ndarray,
        x_std: np.ndarray,
        neighbor_mean: np.ndarray,
        neighbor_std: np.ndarray,
        y: np.ndarray,
    ):
        self.image = image
        self.patches = patches
        self.neighbor_idx = neighbor_idx
        self.center_indices = list(center_indices)
        self.x_channel_idx = list(x_channel_idx)
        self.neighbor_channel_idx = list(neighbor_channel_idx)
        self.x_mean, self.x_std = x_mean, x_std
        self.neighbor_mean, self.neighbor_std = neighbor_mean, neighbor_std
        self.y = y

    def __len__(self) -> int:
        return len(self.center_indices)

    def _crop(self, patch_idx: int, channel_idx: list[int]) -> np.ndarray:
        p = self.patches[patch_idx]
        return self.image[channel_idx, p["y0"] : p["y1"], p["x0"] : p["x1"]]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        center_i = self.center_indices[i]
        x_center = normalize_patch(self._crop(center_i, self.x_channel_idx), self.x_mean, self.x_std)
        x_neighbors = np.stack(
            [
                normalize_patch(self._crop(n_i, self.neighbor_channel_idx), self.neighbor_mean, self.neighbor_std)
                for n_i in self.neighbor_idx[center_i]
            ]
        )
        return torch.from_numpy(x_center), torch.from_numpy(x_neighbors), torch.from_numpy(self.y[i])


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


class PatchEncoder(nn.Module):
    """Small conv encoder: (C_in, patch_size, patch_size) -> (embedding_dim,) via a
    few stride-2 conv blocks ending in global average pooling -- output doesn't
    depend on patch_size, so the same encoder works for any patch_size."""

    def __init__(self, in_channels: int, embedding_dim: int = 32, base_channels: int = 16, depth: int = 3):
        super().__init__()
        chs = [in_channels] + [base_channels * (2**i) for i in range(depth)]
        layers = []
        for a, b in zip(chs[:-1], chs[1:]):
            layers += [nn.Conv2d(a, b, 3, stride=2, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True)]
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(chs[-1], embedding_dim)
        self.conv.apply(init_weights)  # Kaiming for every Conv2d (each feeds a ReLU)
        init_weights(self.proj)  # ReLU follows (below), not a final output layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x).flatten(1)
        return F.relu(self.proj(x))


class NeighborPatchPredictor(nn.Module):
    """A SHARED `PatchEncoder` for each of the 8 neighbor patches' `neighbor_cols`,
    mean-pooled across neighbors (permutation-invariant, same role as
    `models.gnn.WeightedNeighborConv`'s weighted average -- grid neighbors are all
    roughly equidistant, so a plain mean is the natural choice here, no distance
    weighting needed) -- this embedding ALONE predicts `y_cols` when
    `x_in_channels=0` (i.e. `x_cols=[]`, the pure "do neighbors alone predict this
    patch" test). If `x_in_channels>0`, a second `PatchEncoder` for the center
    patch's OWN `x_cols` pixels is concatenated in before the head, the same
    optional "global" self-information as `models.gnn.NeighborIFPredictor`'s
    `x_global`."""

    def __init__(
        self,
        x_in_channels: int,
        neighbor_in_channels: int,
        num_outputs: int,
        embedding_dim: int = 32,
        base_channels: int = 16,
        depth: int = 3,
    ):
        super().__init__()
        self.center_encoder = (
            PatchEncoder(x_in_channels, embedding_dim, base_channels, depth) if x_in_channels > 0 else None
        )
        self.neighbor_encoder = PatchEncoder(neighbor_in_channels, embedding_dim, base_channels, depth)
        head_in = embedding_dim * (2 if x_in_channels > 0 else 1)
        self.head = nn.Linear(head_in, num_outputs)
        init_output_layer(self.head)  # no activation follows this layer

    def forward(self, x_center: torch.Tensor, x_neighbors: torch.Tensor) -> torch.Tensor:
        b, k, c, h, w = x_neighbors.shape
        z_neighbors = self.neighbor_encoder(x_neighbors.reshape(b * k, c, h, w))
        z_neighbors = z_neighbors.reshape(b, k, -1).mean(dim=1)
        if self.center_encoder is None:
            return self.head(z_neighbors)
        z_center = self.center_encoder(x_center)
        return self.head(torch.cat([z_center, z_neighbors], dim=1))


# --------------------------------------------------------------------------
# Data prep + training
# --------------------------------------------------------------------------


def build_patch_data(
    image: np.ndarray,
    channel_names: Sequence[str],
    if_channels: Sequence[str],
    patch_size: int,
    x_cols: Sequence[str],
    neighbor_cols: Optional[Sequence[str]] = None,
    y_cols: Optional[Sequence[str]] = None,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
    scaling: str = "log1p_standard",
    arcsinh_cofactor: float = 5.0,
    subtract_background: bool = True,
) -> dict:
    """
    Tiles `image` into a `patch_size`-px grid (`tools.morphology.make_patch_grid`),
    finds each patch's up-to-8 grid neighbors, excludes border patches (incomplete
    neighborhood -- `border_patch_mask`) from ever being a train/val/test TARGET
    (they still act as a neighbor for interior patches), and returns train/val/test
    `PatchNeighborDataset`s plus the fitted pixel-normalization stats and y
    `ColumnScaler` (see `models.gnn.ColumnScaler` for `scaling`/`arcsinh_cofactor`/
    `subtract_background`).

    `neighbor_cols` defaults to every column in `if_channels`; `y_cols` defaults to
    every column in `if_channels` EXCLUDING `x_cols` (same convention as
    `models.gnn.build_prediction_data`).
    """
    channel_names = list(channel_names)
    x_cols = list(x_cols)
    neighbor_cols = list(if_channels) if neighbor_cols is None else list(neighbor_cols)
    y_cols = [c for c in if_channels if c not in x_cols] if y_cols is None else list(y_cols)

    patches = make_patch_grid(image.shape[1:], patch_size)
    _, neighbor_idx = build_patch_neighbors(patches)
    is_border = border_patch_mask(neighbor_idx)
    valid_idx = np.where(~is_border)[0]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(valid_idx))
    n_test = int(round(len(valid_idx) * test_frac))
    n_val = int(round(len(valid_idx) * val_frac))
    test_centers = valid_idx[perm[:n_test]]
    val_centers = valid_idx[perm[n_test : n_test + n_val]]
    train_centers = valid_idx[perm[n_test + n_val :]]

    x_channel_idx = [_channel_index(c, channel_names) for c in x_cols]
    neighbor_channel_idx = [_channel_index(c, channel_names) for c in neighbor_cols]
    x_mean, x_std = fit_pixel_stats(image, [patches[i] for i in train_centers], x_channel_idx)
    neighbor_mean, neighbor_std = fit_pixel_stats(image, [patches[i] for i in train_centers], neighbor_channel_idx)

    y_channel_idx = [_channel_index(c, channel_names) for c in y_cols]

    def patch_means(indices: np.ndarray) -> np.ndarray:
        return np.array(
            [
                [image[c, patches[i]["y0"] : patches[i]["y1"], patches[i]["x0"] : patches[i]["x1"]].mean() for c in y_channel_idx]
                for i in indices
            ],
            dtype=np.float32,
        )

    y_train_raw = patch_means(train_centers)
    y_scaler = ColumnScaler(scaling, arcsinh_cofactor, subtract_background).fit(y_train_raw)

    def make_ds(centers: np.ndarray, y_raw: np.ndarray) -> PatchNeighborDataset:
        return PatchNeighborDataset(
            image, patches, neighbor_idx, centers, x_channel_idx, neighbor_channel_idx,
            x_mean, x_std, neighbor_mean, neighbor_std, y_scaler.transform(y_raw),
        )

    return dict(
        train_ds=make_ds(train_centers, y_train_raw),
        val_ds=make_ds(val_centers, patch_means(val_centers)),
        test_ds=make_ds(test_centers, patch_means(test_centers)),
        y_scaler=y_scaler,
        x_cols=x_cols,
        neighbor_cols=neighbor_cols,
        y_cols=y_cols,
        patches=patches,
        is_border=is_border,
    )


def train_patch_predictor(
    model: nn.Module,
    train_ds: PatchNeighborDataset,
    val_ds: PatchNeighborDataset,
    epochs: int = 60,
    lr: float = 1e-3,
    batch_size: int = 32,
    weight_decay: float = 1e-5,
    patience: int = 8,
    device: str = "cpu",
    desc: str = "training",
) -> dict[str, list[float]]:
    """Minibatch Adam training with early stopping on val loss -- every non-border
    patch is an independent sample (unlike `models.gnn.train_neighbor_predictor`'s
    full-graph transductive setting), so this is an ordinary DataLoader loop,
    mirroring `tools.morphology.train_cnn`."""
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val, best_state, bad_epochs = float("inf"), None, 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    pbar = trange(epochs, desc=desc, leave=True)
    for _ in pbar:
        model.train()
        running, n = 0.0, 0
        for x_center, x_neighbors, y in train_loader:
            x_center, x_neighbors, y = x_center.to(device), x_neighbors.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x_center, x_neighbors)
            loss = F.mse_loss(pred, y)
            loss.backward()
            opt.step()
            running += loss.item() * len(y)
            n += len(y)
        train_loss = running / max(n, 1)

        model.eval()
        val_running, val_n = 0.0, 0
        with torch.no_grad():
            for x_center, x_neighbors, y in val_loader:
                x_center, x_neighbors, y = x_center.to(device), x_neighbors.to(device), y.to(device)
                pred = model(x_center, x_neighbors)
                val_running += F.mse_loss(pred, y, reduction="sum").item()
                val_n += y.numel()
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


@torch.no_grad()
def predict_patch_df(
    model: nn.Module, ds: PatchNeighborDataset, y_cols: Sequence[str], y_scaler: ColumnScaler = None, device: str = "cpu", batch_size: int = 64
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Runs `model` over every sample in `ds` and returns (pred_df, true_df), both
    in ORIGINAL units (inverse-transformed by `y_scaler`) -- feed directly to
    `tools.morphology.r2_table`/`plot_pred_vs_actual`.
    if y_scaler is not provided, then dont inverse transform 
    
    """
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size)
    preds, trues = [], []
    for x_center, x_neighbors, y in loader:
        pred = model(x_center.to(device), x_neighbors.to(device))
        preds.append(pred.cpu().numpy())
        trues.append(y.numpy())
    pred_scaled = np.concatenate(preds, axis=0)
    true_scaled = np.concatenate(trues, axis=0)
    y_cols = list(y_cols)
    if y_scaler:
        pred_scaled = y_scaler.inverse_transform(pred_scaled)
        true_scaled = y_scaler.inverse_transform(true_scaled)
    pred_scaled = pd.DataFrame(pred_scaled, columns=y_cols)
    true_scaled = pd.DataFrame(true_scaled, columns=y_cols)
    return pred_scaled, true_scaled


def neighbor_average_df(
    image: np.ndarray,
    channel_names: Sequence[str],
    patches: list[dict],
    neighbor_idx: list[list[int]],
    center_indices: Sequence[int],
    y_cols: Sequence[str],
    y_scaler: Optional[ColumnScaler] = None,
) -> pd.DataFrame:
    """
    Naive, model-free baseline: for each center patch, the plain (unweighted) mean of
    its 8 neighbor patches' own mean raw pixel intensity, per `y_cols` channel -- "what
    do my neighbors look like on average", no CNN involved. Always in ORIGINAL units (unless
    y_scaler is provided), since it's computed directly from `image` rather than through
    any `ColumnScaler`.

    Compare against `predict_patch_df`'s `pred_df` (with `y_scaler` given, i.e. also in
    original units) to check whether `NeighborPatchPredictor` is actually learning
    something beyond what a plain neighbor average already captures.
    """
    channel_idx = [_channel_index(c, channel_names) for c in y_cols]
    rows = []
    for center_i in center_indices:
        row = []
        for c in channel_idx:
            neighbor_means = [
                image[c, patches[n]["y0"] : patches[n]["y1"], patches[n]["x0"] : patches[n]["x1"]].mean()
                for n in neighbor_idx[center_i]
            ]
            row.append(np.mean(neighbor_means))
        rows.append(row)
    df = pd.DataFrame(rows, columns=list(y_cols), dtype=np.float32)
    if y_scaler is not None:
        df = pd.DataFrame(y_scaler.transform(df.to_numpy()), columns=df.columns, index=df.index, dtype=np.float32)
    return df

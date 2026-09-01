"""
The spatial graph: which cells talk to which, and how much.

Every pair of cells within `radius` pixels is connected (NOT k-NN, so node degree
is itself a density signal), plus one self-loop per node at distance 0 so a cell's
own value reaches its own embedding through the same pathway a neighbour's does
rather than needing a separate concatenated-in feature at the head.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
# pyrefly: ignore [missing-import]
from torch_geometric.nn import MessagePassing

from tools.morphology import radius_edge_index, init_weights


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
    analog of GCN's degree normalization. A node's own SELF-LOOP edge (see
    `build_radius_graph`) is just another row in `edge_index`/`edge_dist` at
    distance 0 to this function -- `exp(-0 / length_scale) == 1`, the maximum weight
    the decay can produce, so self competes for aggregation weight on the same
    footing as a neighbor would, rather than needing separate handling here.

    `normalize=False` skips that step, leaving the raw per-edge decay weight as-is.
    `WeightedRadiusConv` aggregates with `aggr="add"`, so a normalized weight (sums
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
    normalize_data: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Connects every pair of cells within `radius` pixels (see
    `tools.morphology.radius_edge_index` -- NOT a fixed-k graph, so node degree is a
    direct density signal), adds a SELF-LOOP edge for every node at distance 0 (so a
    cell's own value participates in its own aggregation instead of needing a
    separate global/self pathway concatenated in later -- see the module docstring),
    then converts distance into an edge weight (`normalize_distance_weights`,
    `length_scale` defaults to `radius`).

    `normalize_data=False`: see `normalize_distance_weights` -- turns the GNN's
    aggregation from a weighted average into a weighted sum, an alternative to an
    explicit `local_density` feature for letting the model see neighbor count/density.
    Pass this SAME value as `IFPredictor`'s `normalize_data` constructor arg too (see
    its docstring) -- one decision, shared by the data pipeline and the model
    architecture.

    Returns
    -------
    edge_index : (2, E + N) int64 tensor, COO format, symmetric neighbor edges plus
        one self-loop per node.
    edge_weight : (E + N,) float32 tensor -- sums to 1 per destination node if
        `normalize_data` (default), else the raw per-edge distance-decay weight.
    """
    edge_index_np, edge_dist_np = radius_edge_index(centroids, radius=radius, min_neighbors=min_neighbors)
    n = len(centroids)
    self_idx = np.arange(n, dtype=np.int64)
    edge_index_np = np.concatenate([edge_index_np, np.stack([self_idx, self_idx])], axis=1)
    edge_dist_np = np.concatenate([edge_dist_np, np.zeros(n, dtype=np.float32)])
    edge_weight = normalize_distance_weights(
        edge_index_np, edge_dist_np, n, length_scale or radius, normalize=normalize_data
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


class WeightedRadiusConv(MessagePassing):
    """One self-inclusive message-passing layer: destination node i's output is the
    weighted average (using the fixed, precomputed `edge_weight`) of ITSELF and its
    neighbors' (linearly transformed) features -- self participates via the
    self-loop edge `build_radius_graph` adds at distance 0, not a separate pathway."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="add", node_dim=0)
        self.lin = nn.Linear(in_channels, out_channels)
        init_weights(self.lin)  # standardized input, ReLU follows (except possibly the last layer)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        x = self.lin(x)
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        return x_j * edge_weight.unsqueeze(-1)  # already row-normalized to sum to 1 per destination

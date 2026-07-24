"""
GAT (learned-attention) counterpart to `models.gnn.NeighborIFPredictor`.

Same radius graph, same `global_x_cols`/`neighbor_x_cols`/`y_cols` split, same
`build_prediction_data`/`train_neighbor_predictor` -- the only thing that changes is
the neighbor encoder: `gnn.NeighborRadiusGNN` aggregates neighbors with a FIXED,
precomputed distance weight (no learning involved in how much a neighbor counts);
`NeighborRadiusGAT` below instead learns the attention, but the same precomputed
distance weight (`gnn.build_radius_graph`'s `edge_weight`) is still fed in as an edge
feature (`edge_dim=1`) that `GATConv`'s attention linearly incorporates -- so
distance still informs the (now-learned) adjacency, it's just no longer the ONLY
thing that does.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from tools.morphology import init_output_layer

# re-exported so a notebook only needs `from models.gat import ...` for the GAT run
from models.gnn import build_radius_graph, build_prediction_data, train_neighbor_predictor, predict_df  # noqa: F401


class NeighborRadiusGAT(nn.Module):
    """Neighbor-only (`add_self_loops=False`, so a node never attends to itself)
    stack of `GATConv` layers -- same shape/role as `gnn.NeighborRadiusGNN`, but
    with learned, distance-informed (`edge_dim=1`) attention instead of a fixed
    distance weight. Every non-final layer concatenates its `heads` outputs; the
    final layer averages them (`concat=False`) so the output is exactly
    `out_channels` wide regardless of `heads`."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 1,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        in_dim = in_channels
        for i in range(num_layers):
            is_last = i == num_layers - 1
            layer_out = out_channels if is_last else hidden_channels
            layer_heads = 1 if is_last else heads
            self.convs.append(
                GATConv(
                    in_dim, layer_out, heads=layer_heads, concat=not is_last,
                    edge_dim=1, add_self_loops=False, dropout=dropout,
                )
            )
            in_dim = layer_out * layer_heads
            if not is_last:
                self.norms.append(nn.BatchNorm1d(in_dim))
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            if i < len(self.convs) - 1:
                x = F.relu(self.norms[i](x))
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    @torch.no_grad()
    def attention_by_layer(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> list[torch.Tensor]:
        """
        Same computation as `forward` (same intermediate `x` fed to each layer in
        turn, including the ReLU/BatchNorm/dropout between non-final layers), but
        also captures each layer's attention via `GATConv`'s
        `return_attention_weights=True` -- needed with `num_layers > 1`, where a
        single `conv(..., return_attention_weights=True)` call on ONE layer isn't
        enough: every non-final layer has `heads > 1` (`concat=True`), so its
        attention comes back as `(E, heads)`, not the `(E,)` a single scalar-per-edge
        plot needs (only the FINAL layer is naturally `(E,)`, since `num_layers - 1`
        forces `heads=1` there). Head-averages every layer's attention down to `(E,)`
        so callers get one comparable tensor per layer regardless of layer position
        -- a no-op for the final layer, which already has exactly 1 head.

        Always runs in eval mode with dropout disabled (`training=False` explicitly,
        not just relying on the caller having called `.eval()`), so the SAME
        deterministic computation is used to attribute attention as would be used at
        inference -- dropout during a `forward` call under `.train()` would make
        each layer's captured attention reflect a different random mask than a
        prediction call would, which isn't what you want when explaining a
        specific prediction.
        """
        was_training = self.training
        self.eval()
        try:
            layer_attentions = []
            for i, conv in enumerate(self.convs):
                x, (att_edge_index, alpha) = conv(x, edge_index, edge_attr, return_attention_weights=True)
                assert torch.equal(att_edge_index, edge_index), "edge order changed -- add_self_loops must be False"
                layer_attentions.append(alpha.mean(dim=-1))  # (E, heads) -> (E,); no-op when heads == 1
                if i < len(self.convs) - 1:
                    x = F.relu(self.norms[i](x))
                    x = F.dropout(x, p=self.dropout, training=False)
            return layer_attentions
        finally:
            self.train(was_training)


class NeighborIFPredictorGAT(nn.Module):
    """`NeighborRadiusGAT` encoder + a linear head that also sees `x_global` (the
    cell's OWN `global_x_cols`, concatenated in directly) -- predicts `y_cols`.
    Same `forward` signature as `gnn.NeighborIFPredictor`, so
    `gnn.train_neighbor_predictor`/`gnn.predict_df` work on this model unchanged."""

    def __init__(
        self,
        neighbor_in_channels: int,
        global_in_channels: int,
        num_outputs: int,
        hidden_channels: int = 64,
        embedding_dim: int = 32,
        num_layers: int = 1,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = NeighborRadiusGAT(
            neighbor_in_channels, embedding_dim, hidden_channels, num_layers, heads, dropout
        )
        self.head = nn.Linear(embedding_dim + global_in_channels, num_outputs)
        init_output_layer(self.head)  # no activation follows this layer

    def forward(
        self, x_neighbor: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor, x_global: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        edge_attr = edge_weight.unsqueeze(-1)  # GATConv wants (E, edge_dim), edge_weight is (E,)
        z = self.encoder(x_neighbor, edge_index, edge_attr)
        z_full = torch.cat([z, x_global], dim=1)
        return self.head(z_full), z

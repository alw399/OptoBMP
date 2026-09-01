"""
GAT (learned-attention) counterpart to `models.predictors.IFPredictor`.

Same self-inclusive radius graph, same `x_cols`/`y_cols` split, same
`build_prediction_data`/`train_predictor` -- the only thing that changes is the
encoder: `predictors.RadiusGNN` aggregates with a FIXED, precomputed distance weight (no
learning involved in how much a neighbor -- or self -- counts); `RadiusGAT` below
instead learns the attention, but the same precomputed distance weight
(`graph.build_radius_graph`'s `edge_weight`, including each node's self-loop weight)
is still fed in as an edge feature (`edge_dim=1`) that `GATConv`'s attention
linearly incorporates -- so distance (0 for the self-loop) still informs the
(now-learned) adjacency, it's just no longer the ONLY thing that does.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
# pyrefly: ignore [missing-import]
from torch_geometric.nn import GATConv

from tools.morphology import init_output_layer

# re-exported so a notebook only needs `from models.gat import ...` for the GAT run
from models.graph import build_radius_graph  # noqa: F401
from models.data import build_prediction_data  # noqa: F401
from models.train import train_predictor, train_calibrated, calibrated_predict, predict_df  # noqa: F401
from models.predictors import TwoPartHead


def _init_distance_attention(conv: GATConv) -> None:
    """
    Re-initializes one `GATConv` layer's attention so it starts out EXACTLY
    reproducing the row-normalized distance-decay weighting
    `models.graph.WeightedRadiusConv` uses -- i.e. `attention(i, j) == w_ij /
    sum_k(w_ik)`, where `w` is the same distance-decay edge weight
    (`models.graph.normalize_distance_weights`) this conv receives as
    `log(edge_weight)` -- instead of `GATConv`'s usual Glorot-random, nothing-
    to-do-with-distance starting point. Applies equally to the self-loop edge
    (distance 0, the largest raw weight): at init, GAT starts out attending to
    self exactly as much as the fixed-weight GNN would.

    Zeros `att_src`/`att_dst` (the NODE-feature-driven attention terms) so at
    init the attention logit depends ONLY on the edge feature; sets
    `lin_edge`'s weight to 1 and `att_edge` to `1 / (negative_slope *
    out_channels)`. That specific constant is chosen so
    `LeakyReLU(alpha_edge) == log(w_ij)` exactly -- the `negative_slope`
    scaling in `LeakyReLU`'s negative branch exactly cancels the `1/
    negative_slope` we baked into `att_edge` -- which makes `softmax_j(log(
    w_ij)) == w_ij / sum_j(w_j)`, the standard softmax-of-log identity.
    Requires `RadiusGAT.forward` to feed `log(edge_weight)`, not the raw
    `edge_weight`, as `edge_attr` -- see `distance_init` below.

    A STARTING point only, not a constraint: `att_src`/`att_dst` still receive
    gradients from the node features' contribution to the attention logit
    (their initial value is 0, not a frozen 0), so training remains free to
    learn something else entirely.
    """
    nn.init.zeros_(conv.att_src)
    nn.init.zeros_(conv.att_dst)
    nn.init.constant_(conv.lin_edge.weight, 1.0)
    nn.init.constant_(conv.att_edge, 1.0 / (conv.negative_slope * conv.out_channels))


class RadiusGAT(nn.Module):
    """Self-inclusive stack of `GATConv` layers -- same shape/role as `gnn.RadiusGNN`,
    but with learned, distance-informed (`edge_dim=1`) attention instead of a fixed
    distance weight. Every non-final layer concatenates its `heads` outputs; the
    final layer averages them (`concat=False`) so the output is exactly
    `out_channels` wide regardless of `heads`.

    `add_self_loops=False`: NOT an exclusion of self -- the graph handed in already
    carries an explicit self-loop edge per node (`gnn.build_radius_graph`, weighted
    the same distance-decay way as any other edge), so `GATConv`'s own automatic
    self-loop insertion is disabled here purely to avoid adding a SECOND, differently-
    weighted self-loop on top of the one already present.

    `distance_init=True` warm-starts EVERY layer's attention via
    `_init_distance_attention` -- see its docstring -- so GAT starts out
    computing the exact same row-normalized distance-decay weighting the fixed
    GNN uses (self-loop included), rather than near-uniform random attention,
    while remaining free to learn away from it during training. Off by default:
    it doesn't change any parameter's SHAPE (old checkpoints still load either
    way), only its INITIAL VALUE, so this is purely a training-dynamics choice,
    not a backwards-compatibility one -- opt in explicitly to use it.

    `use_checkpoint=True` wraps each layer's `GATConv` call in
    `torch.utils.checkpoint.checkpoint` -- trades extra compute (each layer's
    forward runs twice: once discarded, once replayed during backward) for
    lower activation memory, with the EXACT same result (no approximation),
    since it changes what's stored for backward, not what's computed. Only the
    conv call itself is checkpointed, not the following BatchNorm/dropout --
    BatchNorm's running-stats update is a buffer write with a side effect, not
    a tracked autograd op, so it would silently double-update if it were
    inside the checkpointed region (checkpoint replays its function during
    backward). Off by default; a plain memory/speed knob with no accuracy
    impact either way."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 1,
        heads: int = 4,
        dropout: float = 0.1,
        distance_init: bool = False,
        use_checkpoint: bool = False,
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
            conv = GATConv(
                in_dim, layer_out, heads=layer_heads, concat=not is_last,
                edge_dim=1, add_self_loops=False, dropout=dropout,
            )
            if distance_init:
                _init_distance_attention(conv)
            self.convs.append(conv)
            in_dim = layer_out * layer_heads
            if not is_last:
                self.norms.append(nn.BatchNorm1d(in_dim))
        self.dropout = dropout
        self.distance_init = distance_init
        self.use_checkpoint = use_checkpoint

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        if self.distance_init:
            edge_attr = torch.log(edge_attr.clamp_min(1e-8))  # see _init_distance_attention
        for i, conv in enumerate(self.convs):
            if self.use_checkpoint and torch.is_grad_enabled():
                x = checkpoint(conv, x, edge_index, edge_attr, use_reentrant=False)
            else:
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
        -- a no-op for the final layer, which already has exactly 1 head. `E` here
        includes each node's self-loop edge (see `graph.build_radius_graph`), so a
        node's attention to ITSELF is directly readable off the same tensor as its
        attention to any neighbor.

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
            if self.distance_init:
                edge_attr = torch.log(edge_attr.clamp_min(1e-8))  # match forward's edge_attr transform
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


class IFPredictorGAT(nn.Module):
    """`RadiusGAT` encoder + a head -- predicts `y_cols` from the graph embedding
    alone. Same `forward(x, edge_index, edge_weight)` signature as `predictors.IFPredictor`,
    so `train.train_predictor`/`train.predict_df` work on this model unchanged.

    `two_part=True` swaps the plain linear head for a `gnn.TwoPartHead` -- see its
    docstring, and `data.build_prediction_data`'s `two_part` option, which scales
    `y` the way `TwoPartHead` expects (`scalers.HurdleScaler`, not `gnn.ColumnScaler`).
    `False` (default): a plain `nn.Linear` head, completely unaffected by any of
    this -- old checkpoints load unchanged.

    `distance_init`/`use_checkpoint` are passed straight through to `RadiusGAT` --
    see its docstring."""

    def __init__(
        self,
        in_channels: int,
        num_outputs: int,
        hidden_channels: int = 64,
        embedding_dim: int = 32,
        num_layers: int = 1,
        heads: int = 4,
        dropout: float = 0.1,
        distance_init: bool = False,
        use_checkpoint: bool = False,
        two_part: bool = False,
    ):
        super().__init__()
        self.encoder = RadiusGAT(
            in_channels, embedding_dim, hidden_channels, num_layers, heads, dropout, distance_init,
            use_checkpoint=use_checkpoint,
        )
        self.two_part = two_part
        if two_part:
            self.head = TwoPartHead(embedding_dim, num_outputs)
        else:
            self.head = nn.Linear(embedding_dim, num_outputs)
            init_output_layer(self.head)  # no activation follows this layer

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        edge_attr = edge_weight.unsqueeze(-1)  # GATConv wants (E, edge_dim), edge_weight is (E,)
        z = self.encoder(x, edge_index, edge_attr)
        if self.two_part:
            combined, _, _, _ = self.head(z)
            return combined, z
        return self.head(z), z

    def forward_two_part(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Only valid when `two_part=True` -- see `gnn.IFPredictor.forward_two_part`,
        same role, same returned tuple."""
        assert self.two_part, "forward_two_part requires two_part=True"
        edge_attr = edge_weight.unsqueeze(-1)
        z = self.encoder(x, edge_index, edge_attr)
        combined, logit, magnitude, _ = self.head(z)
        return combined, logit, magnitude, z

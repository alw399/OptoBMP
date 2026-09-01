"""
Encoders and prediction heads.

`RadiusGNN` stacks fixed distance-weighted convolutions over the radius graph;
`IFPredictor` puts a head on it. `TwoPartHead` is the hurdle head for a
zero-inflated target. `MLPPredictor` is the graph-free control -- same head, same
signature, no message passing -- which is how you tell whether the graph is
actually contributing or the node features are doing all the work.
(Measured on this data: GNN R2 0.184 vs MLP 0.148 on identical features.)
"""

from matplotlib import path
import textwrap
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models.graph import WeightedRadiusConv
from tools.morphology import init_weights, init_output_layer


class RadiusGNN(nn.Module):
    """Stack of `WeightedRadiusConv` layers -- an embedding built from a cell and its
    neighbors' `x_cols`, over the (self-inclusive) radius graph.

    `normalize_data` MUST be the SAME value passed to `build_radius_graph`/
    `build_prediction_data` when this data's edge weights were built. When
    False, `WeightedRadiusConv`'s fixed-weight aggregation becomes a weighted
    SUM instead of a weighted AVERAGE (see `normalize_distance_weights`), whose
    magnitude scales with local density (the same signal an explicit
    `local_density` column would otherwise supply). When
    `normalize_data=False`, an extra `BatchNorm1d` is added after the last conv
    (ReLU/dropout still skipped) to fix that -- BatchNorm rescales per-CHANNEL
    scale, not cross-node relative magnitude, so a denser node's embedding is
    still larger than a sparser node's after it, just consistently scaled, so
    the density signal survives. When True (the default -- weighted-average
    output is already ~unit-scale), no extra norm is added: `self.final_norm`
    stays `None`, so old checkpoints trained before this parameter existed
    still load unchanged.

    `use_checkpoint=True` wraps each layer's `WeightedRadiusConv` call in
    `torch.utils.checkpoint.checkpoint` -- trades extra compute (each layer's
    forward runs twice: once discarded, once replayed during backward) for
    lower activation memory, with the EXACT same result (no approximation),
    since it changes what's stored for backward, not what's computed. Only
    the conv call itself is checkpointed, not the following BatchNorm/dropout
    -- BatchNorm's running-stats update is a buffer write with a side effect,
    not a tracked autograd op, so it would silently double-update if it were
    inside the checkpointed region. Off by default; a plain memory/speed knob
    with no accuracy impact either way."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        normalize_data: bool = False,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        self.convs = nn.ModuleList(WeightedRadiusConv(dims[i], dims[i + 1]) for i in range(num_layers))
        self.norms = nn.ModuleList(nn.BatchNorm1d(dims[i + 1]) for i in range(num_layers - 1))
        self.final_norm = None if normalize_data else nn.BatchNorm1d(out_channels)
        self.dropout = dropout
        self.use_checkpoint = use_checkpoint

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            if self.use_checkpoint and torch.is_grad_enabled():
                x = checkpoint(conv, x, edge_index, edge_weight, use_reentrant=False)
            else:
                x = conv(x, edge_index, edge_weight)
            if i < len(self.convs) - 1:
                x = F.relu(self.norms[i](x))
                x = F.dropout(x, p=self.dropout, training=self.training)
        if self.final_norm is not None:
            x = self.final_norm(x)
        return x


class TwoPartHead(nn.Module):
    """
    Hurdle-model head for a zero-inflated `y_col` (e.g. this project's Sox17_mean:
    ~99% of cells sit at background, ~1% are real positives). A single continuous
    MSE regression over a target shaped like that lets the loss be dominated by
    correctly predicting background for the overwhelming majority, with little
    pressure left to get the sparse positive population right -- this splits the
    problem into two linear heads over the SAME embedding instead:

      - `classifier`: a logit for "is this cell positive for this y_col" (the BCE
        target is `data['y_positive']`).
      - `magnitude`: a value in (0, 1), via a final `sigmoid` -- only ever
        meaningfully SUPERVISED on positive rows (see `train_predictor`'s two-part
        loss), so its value for a confidently-negative cell is unconstrained/
        meaningless on its own.

    Combined as `prob * magnitude`, both in (0, 1) -- this ASSUMES `y` itself is
    scaled so a real positive cell's target also lands in (0, 1] and a background
    cell's lands at/near 0, which is exactly what `build_prediction_data`'s
    `two_part` option arranges via `HurdleScaler` rather than the general-purpose
    `ColumnScaler` (zero-mean/unit-variance, which lets a bright positive land at
    5-7+ standard deviations above the mean -- past where this sigmoid-bounded
    head could ever reach). If magnitude-branch loss looks stuck at a large,
    non-decreasing value, check that `y` actually came from `HurdleScaler` (i.e.
    `build_prediction_data` was called with `two_part=True`) before assuming the
    model is just undertrained.
    """

    def __init__(self, embedding_dim: int, num_outputs: int):
        super().__init__()
        self.classifier = nn.Linear(embedding_dim, num_outputs)
        self.magnitude = nn.Linear(embedding_dim, num_outputs)
        init_output_layer(self.classifier)  # logit -- sigmoid follows, Xavier is still the right init
        init_output_layer(self.magnitude)   # sigmoid follows here too -- see class docstring

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logit = self.classifier(z)
        magnitude = torch.sigmoid(self.magnitude(z))
        prob = torch.sigmoid(logit)
        combined = prob * magnitude
        return combined, logit, magnitude, prob


class IFPredictor(nn.Module):
    """`RadiusGNN` encoder + a head -- predicts `y_cols` from the graph embedding
    alone.

    `two_part=True` swaps the plain linear head for a `TwoPartHead` -- see its
    docstring, and `build_prediction_data`'s `two_part` option, which scales `y`
    the way `TwoPartHead` expects (`HurdleScaler`, not `ColumnScaler`). `False`
    (default): a plain `nn.Linear` head

    `normalize_data` is passed straight through to `RadiusGNN` -- see its
    docstring. MUST be the SAME value `build_prediction_data` was called with for
    this data, same as `save_model`'s `normalize_data` arg. `use_checkpoint` is
    also passed straight through -- see `RadiusGNN`'s docstring."""

    def __init__(
        self,
        in_channels: int,
        num_outputs: int,
        hidden_channels: int = 64,
        embedding_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.1,
        normalize_data: bool = False,
        use_checkpoint: bool = False,
        two_part: bool = False,
    ):
        super().__init__()
        self.normalize_data = normalize_data
        self.encoder = RadiusGNN(
            in_channels, embedding_dim, hidden_channels, num_layers, dropout, normalize_data,
            use_checkpoint=use_checkpoint,
        )
        self.two_part = two_part
        self.num_outputs = num_outputs
        if two_part:
            self.head = TwoPartHead(embedding_dim, num_outputs)
        else:
            self.head = nn.Linear(embedding_dim, num_outputs)
            init_output_layer(self.head)  # no activation follows this layer

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x, edge_index, edge_weight)
        if self.two_part:
            combined, _, _, _ = self.head(z)
            return combined, z
        return self.head(z), z

    def forward_two_part(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Only valid when `two_part=True`. Same encoder pass as `forward`, but also
        returns the classifier logit and raw magnitude `train_predictor`'s
        two-part loss needs (BCE on the logit against `data['y_positive']`, MSE on
        the magnitude restricted to positive rows -- see `train_predictor`'s
        docstring) -- `forward` alone only exposes the already-combined prediction.
        """
        assert self.two_part, "forward_two_part requires two_part=True"
        z = self.encoder(x, edge_index, edge_weight)
        combined, logit, magnitude, _ = self.head(z)
        return combined, logit, magnitude, z

    def message_sensitivity_by_layer(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor
    ) -> list[torch.Tensor]:
        """
        Edge-level explanation for a model with no learned attention to read off
        directly -- `WeightedRadiusConv`'s `edge_weight` is a FIXED function of
        distance alone (see `normalize_distance_weights`), so unlike
        `gat.RadiusGAT.attention_by_layer` there's no learned per-edge scalar
        already sitting in the forward pass to inspect. This substitutes a
        gradient-based one: for each layer, captures that layer's per-edge message
        `m_e = edge_weight_e * lin(x_j)` (exactly what `WeightedRadiusConv.message`
        computes internally, including each node's own self-loop message), replays
        the REST of the forward pass (remaining conv layers, then this model's
        `head`) -- unlike `attention_by_layer`, which stops at the encoder since
        attention doesn't involve the head, this goes all the way to the actual
        prediction, since there's no output-independent quantity to fall back on
        here -- then backpropagates the prediction summed over every cell ONCE
        back to each layer's captured messages.

        Summing before backpropagating (rather than one backward per cell) is exact,
        not an approximation: cell i's final output depends only on ITS OWN incoming
        edges at every layer (`WeightedRadiusConv` never lets a message meant for
        one destination leak into another), so `d(sum_i out_i)/d(m_e)` already equals
        `d(out_dst(e))/d(m_e)` -- no cross-cell contamination to worry about.

        Returns one (E,) tensor per layer: `(grad_m * m).sum(-1)`, i.e. grad-times-
        message ("how much did this edge's ACTUAL contribution move the prediction",
        the standard attribution choice) rather than the raw gradient alone (which
        would only say "how sensitive is the prediction to an infinitesimal nudge
        here", ignoring how large that edge's real contribution actually is).

        Same layering caveat as `attention_by_layer`: only the FIRST layer's messages
        are a direct function of the raw `x_cols` values -- every later layer's
        messages are built from the previous layer's embeddings, so correlating a
        later layer's sensitivity against raw values reflects an indirect/aggregate
        relationship, not a direct one.

        Always runs in eval mode with dropout disabled and BatchNorm using its
        running stats, for the same reason as `attention_by_layer`: this attributes
        the SAME deterministic computation a real prediction call would use, not one
        with a different random dropout mask.
        """
        was_training = self.training
        self.eval()
        try:
            with torch.enable_grad():
                encoder = self.encoder
                src, dst = edge_index[0], edge_index[1]
                messages = []
                for i, conv in enumerate(encoder.convs):
                    x_lin = conv.lin(x)
                    m = x_lin[src] * edge_weight.unsqueeze(-1)
                    messages.append(m)
                    z = torch.zeros(x.shape[0], m.shape[-1], dtype=m.dtype, device=m.device)
                    z = z.scatter_add(0, dst.unsqueeze(-1).expand_as(m), m)
                    if i < len(encoder.convs) - 1:
                        z = F.relu(encoder.norms[i](z))
                        z = F.dropout(z, p=encoder.dropout, training=False)
                    x = z
                if encoder.final_norm is not None:
                    x = encoder.final_norm(x)
                out = self.head(x)[0] if self.two_part else self.head(x)  # TwoPartHead returns a tuple; take combined
                grads = torch.autograd.grad(out.sum(), messages)
                return [(g * m).sum(-1).detach() for g, m in zip(grads, messages)]
        finally:
            self.train(was_training)
    
    def save_checkpoint(self, path: str):
        checkpoint = {
            "model_config": {
                "in_channels": self.encoder.in_channels,
                "embedding_dim": self.encoder.out_channels,
                "hidden_channels": self.encoder.hidden_channels,
                "num_layers": self.encoder.num_layers,
                "num_outputs": self.num_outputs,
                "num_layers": len(self.encoder.convs),
                "dropout": self.encoder.dropout,
                "normalize_data": self.normalize_data,
                "use_checkpoint": self.encoder.use_checkpoint,
                "two_part": self.two_part,
            },
            "model_state_dict": self.state_dict(),
        }

        torch.save(checkpoint, path)
    
    @classmethod
    def load_checkpoint(cls, path: str, device="cpu"):
        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )

        model = cls(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        return model



class MLPPredictor(nn.Module):
    """Per-cell MLP with the same `TwoPartHead` -- ignores `edge_index`/`edge_weight`
    entirely, so it is the exact ablation for "does message passing add anything
    once each node already carries a multi-scale summary of its neighbourhood?".
    Same `forward`/`forward_two_part` signature as `IFPredictor`, so it drops into
    the same training and scoring code unchanged."""

    two_part = True

    def __init__(self, in_channels: int, num_outputs: int = 1, hidden_channels: int = 128,
                 num_layers: int = 3, embedding_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [embedding_dim]
        layers = []
        for i in range(len(dims) - 1):
            lin = nn.Linear(dims[i], dims[i + 1])
            init_weights(lin)
            layers.append(lin)
            if i < len(dims) - 2:
                layers += [nn.BatchNorm1d(dims[i + 1]), nn.ReLU(), nn.Dropout(dropout)]
        self.encoder_mlp = nn.Sequential(*layers)
        self.head = TwoPartHead(embedding_dim, num_outputs)

    def forward(self, x, edge_index=None, edge_weight=None):
        z = self.encoder_mlp(x)
        if not self.two_part:  # plain regression head swapped in by an ablation
            return self.head(z), z
        combined, _, _, _ = self.head(z)
        return combined, z

    def forward_two_part(self, x, edge_index=None, edge_weight=None):
        z = self.encoder_mlp(x)
        combined, logit, magnitude, _ = self.head(z)
        return combined, logit, magnitude, z

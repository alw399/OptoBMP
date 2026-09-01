"""
Training loops.

`train_predictor` is the original full-batch/minibatch loop. `train_calibrated` is
the one to use for a `two_part` model -- see its docstring for the prior-correction
issue it exists to fix, which is worth understanding before reading any R2 off a
hurdle model trained with `auto_pos_weight`.
"""

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import trange
# pyrefly: ignore [missing-import]
from torch_geometric.data import Data
# pyrefly: ignore [missing-import]
from torch_geometric.loader import NeighborLoader


def _two_part_loss(
    logit: torch.Tensor,
    magnitude: torch.Tensor,
    y: torch.Tensor,
    y_positive: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    `train_predictor`'s loss when `model.two_part` -- BCE-with-logits over EVERY
    row passed in (needs both classes present to learn a real decision boundary)
    plus MSE over only the rows `y_positive` marks positive (`magnitude` is only
    ever supervised for cells the ground truth calls positive -- a negative cell's
    target is already known deterministically to be exactly 0 under `HurdleScaler`,
    so training magnitude against it would just be redundant, noisy signal). All
    tensors here are already restricted to whichever rows count for this particular
    loss computation (a `train_mask`/`val_mask` slice, or a minibatch's target
    slice) -- this function itself does no masking beyond the positive/negative
    split. `pos_weight` (per-`y_col`, `nn.BCEWithLogitsLoss` convention) upweights
    the minority positive class -- see `train_predictor`'s `auto_pos_weight`.
    """
    bce = F.binary_cross_entropy_with_logits(logit, y_positive, pos_weight=pos_weight)
    pos_sel = y_positive > 0.5
    if pos_sel.any():
        mag_loss = F.mse_loss(magnitude[pos_sel], y[pos_sel])
    else:
        mag_loss = torch.zeros((), device=y.device, dtype=y.dtype)
    return bce + mag_loss


def train_predictor(
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
    batch_size: Optional[int] = None,
    num_neighbors: Optional[Sequence[int]] = None,
    use_amp: bool = False,
    auto_pos_weight: bool = True,
    classifier_pos_weight: Optional[torch.Tensor] = None,
) -> dict[str, list[float]]:
    """
    Full-batch transductive training (mirrors `tools.morphology.train_gnn`): every
    cell always participates in message passing, `train_mask`/`val_mask` only
    control which cells' own loss/gradients/early-stopping count. Works for both
    `IFPredictor` (this module) and `models.gat.IFPredictorGAT` -- same
    `forward(x, edge_index, edge_weight)` signature.

    `use_amp=True` runs the forward pass (and loss) under `torch.autocast` in
    fp16, with a `GradScaler` handling the backward/step -- roughly halves
    activation/attention-tensor memory, at the cost of some numerical
    precision (occasionally an unstable loss on aggressive learning rates;
    the scaler mitigates but doesn't eliminate this). Only takes effect on
    CUDA -- a no-op elsewhere, since fp16 autocast isn't well-supported on
    CPU/MPS for this kind of model. Combines with `use_checkpoint` on the
    model (`RadiusGNN`/`RadiusGAT`) for further savings; the two are
    independent knobs affecting different things (numeric precision vs. what's
    stored for backward).

    `batch_size` (default `None` -- the full-batch behavior above, unchanged): set
    it to switch to neighbor-sampled MINIBATCH training via `torch_geometric.loader.
    NeighborLoader` instead, for graphs too large to fit one full-graph forward +
    backward pass in GPU/MPS memory (`GATConv`'s per-edge message tensor scales with
    TOTAL edge count, which for a large radius graph can be tens of GB in a single
    allocation). Each optimizer step then only processes `batch_size` TARGET cells
    plus whichever neighbor cells their receptive field pulls in -- not every cell
    in the graph at once. Requires the `pyg-lib` (or `torch-sparse`) package for
    `NeighborLoader`'s sampling backend.

    `num_neighbors` (only used when `batch_size` is set) is the per-hop neighbor cap
    `NeighborLoader` samples -- one entry per message-passing layer, inferred from
    `len(model.encoder.convs)` if not given. Defaults to `-1` per hop (keep EVERY
    neighbor, no subsampling), so `batch_size` changes ONLY how many target cells are
    scored per step, not what each one's neighborhood aggregate actually is: verified
    numerically that a `-1`-per-hop batched forward pass reproduces the full-batch
    forward pass EXACTLY (to float32 precision) for whichever cells land in a given
    batch, in eval mode. In train mode, `BatchNorm1d` layers still see only that
    batch's statistics rather than the whole graph's -- the same, expected way
    BatchNorm behaves under minibatching in any other architecture, not a bug or an
    approximation introduced by the sampling itself. Pass a smaller `num_neighbors`
    yourself for a further memory/speed tradeoff, at the cost of an actually-
    approximated (randomly subsampled) neighborhood per step.

    `auto_pos_weight`/`classifier_pos_weight`: only relevant when `model.two_part`
    (see `IFPredictor`'s docstring) -- `classifier_pos_weight` (per-`y_col`, same
    convention as `nn.BCEWithLogitsLoss`'s `pos_weight`) explicitly sets how much
    more a missed POSITIVE costs than a missed negative in the classifier's BCE
    loss. If left `None` and `auto_pos_weight` (default True), it's computed from
    the TRAIN split as `n_negative / n_positive` per column -- the standard
    class-imbalance heuristic, since a target this skewed (this project's
    Sox17_mean is ~1% positive) makes plain unweighted BCE trivially minimizable
    by predicting "negative" for every cell, learning nothing about the real
    positive population. Set `auto_pos_weight=False` to disable reweighting
    entirely (plain BCE) instead.
    """
    model = model.to(device)
    two_part = getattr(model, "two_part", False)
    amp_enabled = use_amp and str(device).startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    pos_weight = None
    if two_part:
        if classifier_pos_weight is not None:
            pos_weight = torch.as_tensor(classifier_pos_weight, dtype=torch.float32)
        elif auto_pos_weight:
            # Computed from the (still-CPU, un-moved) train labels, before either
            # branch below moves anything to `device` -- one computation shared by
            # both the full-batch and minibatch training loops.
            train_labels = data["y_positive"][train_mask]
            n_pos = train_labels.sum(dim=0).clamp(min=1.0)  # avoid div-by-zero if a column has 0 train positives
            n_neg = train_labels.shape[0] - train_labels.sum(dim=0)
            pos_weight = (n_neg / n_pos).clamp(min=1.0)
        if pos_weight is not None:
            pos_weight = pos_weight.to(device)

    if batch_size is None:
        x = data["x"].to(device)
        edge_index = data["edge_index"].to(device)
        edge_weight = data["edge_weight"].to(device)
        y = data["y"].to(device)
        train_mask = train_mask.to(device)
        val_mask = val_mask.to(device)
        if two_part:
            y_positive = data["y_positive"].to(device)
    else:
        # NeighborLoader's sampling backend runs on CPU regardless of `device` --
        # build the graph from the ORIGINAL (un-moved) tensors, and move only each
        # already-small sampled batch to `device` inside the loop below.
        if num_neighbors is None:
            num_neighbors = [-1] * len(model.encoder.convs)
        # `.contiguous()`: `edge_index` comes from a `.T`-transposed numpy array
        # (`tools.morphology.radius_edge_index`), so the tensor `torch.from_numpy`
        # produces is a non-contiguous view -- `pyg_lib`'s sampler backend requires
        # contiguous input.
        graph_kwargs = dict(
            x=data["x"], y=data["y"],
            edge_index=data["edge_index"].contiguous(), edge_weight=data["edge_weight"],
            num_nodes=data["x"].size(0),
        )
        if two_part:
            graph_kwargs["y_positive"] = data["y_positive"]
        graph = Data(**graph_kwargs)
        train_loader = NeighborLoader(
            graph, num_neighbors=list(num_neighbors), batch_size=batch_size,
            input_nodes=train_mask, shuffle=True,
        )
        val_loader = NeighborLoader(
            graph, num_neighbors=list(num_neighbors), batch_size=batch_size,
            input_nodes=val_mask, shuffle=False,
        )

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val, best_state, bad_epochs = float("inf"), None, 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    pbar = trange(epochs, desc=desc, leave=True)
    for _ in pbar:
        if batch_size is None:
            model.train()
            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                if two_part:
                    _, logit, magnitude, _ = model.forward_two_part(x, edge_index, edge_weight)
                    train_loss = _two_part_loss(
                        logit[train_mask], magnitude[train_mask], y[train_mask], y_positive[train_mask], pos_weight,
                    )
                else:
                    pred, _ = model(x, edge_index, edge_weight)
                    train_loss = F.mse_loss(pred[train_mask], y[train_mask])
            scaler.scale(train_loss).backward()
            scaler.step(opt)
            scaler.update()
            train_loss = train_loss.item()

            model.eval()
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                if two_part:
                    _, logit, magnitude, _ = model.forward_two_part(x, edge_index, edge_weight)
                    val_loss = _two_part_loss(
                        logit[val_mask], magnitude[val_mask], y[val_mask], y_positive[val_mask], pos_weight,
                    ).item()
                else:
                    pred, _ = model(x, edge_index, edge_weight)
                    val_loss = F.mse_loss(pred[val_mask], y[val_mask]).item()
        else:
            model.train()
            total_loss = total_n = 0
            for batch in train_loader:
                batch = batch.to(device)
                opt.zero_grad()
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                    bs = batch.batch_size
                    # Only the first `bs` nodes are TARGETS -- the rest are sampled
                    # neighbors pulled in purely to support message passing, the same
                    # role border-excluded/non-train cells play in the full-batch path.
                    if two_part:
                        _, logit, magnitude, _ = model.forward_two_part(batch.x, batch.edge_index, batch.edge_weight)
                        loss = _two_part_loss(
                            logit[:bs], magnitude[:bs], batch.y[:bs], batch.y_positive[:bs], pos_weight,
                        )
                    else:
                        pred, _ = model(batch.x, batch.edge_index, batch.edge_weight)
                        loss = F.mse_loss(pred[:bs], batch.y[:bs])
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                total_loss += loss.item() * bs
                total_n += bs
            train_loss = total_loss / total_n

            model.eval()
            total_val_loss = total_val_n = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                        bs = batch.batch_size
                        if two_part:
                            _, logit, magnitude, _ = model.forward_two_part(
                                batch.x, batch.edge_index, batch.edge_weight
                            )
                            vloss = _two_part_loss(
                                logit[:bs], magnitude[:bs], batch.y[:bs], batch.y_positive[:bs], pos_weight,
                            )
                        else:
                            pred, _ = model(batch.x, batch.edge_index, batch.edge_weight)
                            vloss = F.mse_loss(pred[:bs], batch.y[:bs])
                    total_val_loss += vloss.item() * bs
                    total_val_n += bs
            val_loss = total_val_loss / total_val_n

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
    `models.checkpoint.apply_prediction_data`.

    Always runs on CPU, regardless of what device `model` was trained on:
    `train_predictor` moves `model` to its `device` arg, but never moves `data`
    (its full-batch forward path only moves its OWN local copies; its minibatch
    path moves each sampled batch, not the underlying `data`), so `data` here is
    always CPU tensors -- moving `model` back to CPU is what keeps this callable
    at all after GPU/MPS training, not just what avoids reintroducing a
    full-graph GPU memory allocation for eval alone.
    """
    if y_scaler is _AUTO:
        y_scaler = data["y_scaler"]
    model = model.to("cpu")
    model.eval()
    pred, _ = model(data["x"], data["edge_index"], data["edge_weight"])
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
# Prior-corrected training for a `two_part` model
# --------------------------------------------------------------------------


@torch.no_grad()
def calibrated_predict(model: nn.Module, data: dict, log_w: float, device: str = "cpu") -> np.ndarray:
    """
    `TwoPartHead`'s combined output with the classifier's re-balancing prior divided
    back out.

    `train_predictor(auto_pos_weight=True)` trains the classifier's BCE with
    `pos_weight = n_neg / n_pos`. That is the right thing for LEARNING an imbalanced
    boundary, but it also means the head converges to

        sigmoid(z) = w*p / (w*p + 1 - p),   not p

    so the probability is calibrated to a RE-BALANCED prior, not the real one, and
    `combined = sigmoid(z) * magnitude` is inflated by roughly `w`. Measured on
    `W8_pattern1` Sox17: true positive rate 0.165, mean predicted probability 0.479
    (`pos_weight` = 5.2). R2 is scale-sensitive, so this alone is enough to drive a
    reported R2 negative on a model whose ranking is fine: on an already-trained model,
    subtracting `log(pos_weight)` from the logit moved R2 from -0.55 to +0.08 with no
    retraining.

    The correction `p = sigmoid(z - log(w))` is exact and monotone IN THE LOGIT, so the
    classifier's own ranking is untouched. The returned quantity is `prob * magnitude`
    though, and rescaling only `prob` can reorder that product slightly -- so AUROC
    shifts a little (<0.01 in practice) rather than not at all. A large R2 change
    alongside a negligible AUROC change is the signature to look for.
    """
    model = model.to(device).eval()
    x = data["x"].to(device)
    ei = data["edge_index"].to(device)
    ew = data["edge_weight"].to(device)
    _, logit, magnitude, _ = model.forward_two_part(x, ei, ew)
    prob = torch.sigmoid(logit - log_w)
    return (prob * magnitude).cpu().numpy().ravel()


def train_calibrated(
    model: nn.Module,
    data: dict,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    epochs: int = 2500,
    lr: float = 1e-2,
    weight_decay: float = 1e-5,
    patience: int = 400,
    device: str = "cpu",
    desc: str = "train",
    lambda_combined: float = 10.0,
    lambda_posmag: float = 10.0,
    auto_pos_weight: bool = True,
    verbose: bool = True,
) -> tuple[dict[str, list[float]], float]:
    """
    Full-batch transductive training for a `two_part` model, differing from
    `train_predictor` in three measured ways:

    1. The loss includes an MSE term on the PRIOR-CORRECTED combined output (see
       `calibrated_predict`), weighted by `lambda_combined`, so the model is trained
       for the quantity it gets scored on. Measured: R2 0.184 at `lambda_combined=10`
       vs 0.168 at 1, and vs 0.101 for a plain single-output MSE head -- the hurdle
       structure is worth keeping, it just has to be optimised for the right target.
    2. Early stopping is on validation R2 of that corrected output, not on the raw
       BCE+MSE sum. Once the classifier is reweighted the two are no longer
       monotonically related, so the raw loss stops being a usable stopping signal.
    3. `lr` defaults to 1e-2 with AdamW and a plateau schedule. At the 1e-4 the
       original notebook used, validation R2 is still climbing after 1500 epochs.

    Returns `(history, log_w)`. `log_w` is required by `calibrated_predict` at
    inference time -- save it alongside the checkpoint.
    """
    model = model.to(device)
    x = data["x"].to(device)
    ei = data["edge_index"].to(device)
    ew = data["edge_weight"].to(device)
    y = data["y"].to(device)
    ypos = data["y_positive"].to(device)
    trm = train_mask.to(device)
    vam = val_mask.to(device)

    if auto_pos_weight:
        lab = ypos[trm]
        n_pos = lab.sum(dim=0).clamp(min=1.0)
        n_neg = lab.shape[0] - lab.sum(dim=0)
        pos_weight = (n_neg / n_pos).clamp(min=1.0)
    else:
        pos_weight = torch.ones(ypos.shape[1], device=device)
    log_w = float(torch.log(pos_weight).mean().item())

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=max(20, patience // 4))

    y_val = y[vam]
    val_var = y_val.var(unbiased=False).clamp(min=1e-12)

    def losses(mask):
        _, logit, magnitude, _ = model.forward_two_part(x, ei, ew)
        lg, mg, yy, yp = logit[mask], magnitude[mask], y[mask], ypos[mask]
        bce = F.binary_cross_entropy_with_logits(lg, yp, pos_weight=pos_weight)
        sel = yp > 0.5
        mag = F.mse_loss(mg[sel], yy[sel]) if sel.any() else torch.zeros((), device=device)
        comb = torch.sigmoid(lg - log_w) * mg          # the CALIBRATED output
        return bce + (lambda_posmag * mag) + (lambda_combined * F.mse_loss(comb, yy)), comb

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_r2": []}
    best_r2, best_state, bad = -np.inf, None, 0
    pbar = trange(epochs, desc=desc, leave=False, disable=not verbose)
    for _ in pbar:
        model.train()
        opt.zero_grad()
        loss, _ = losses(trm)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            vloss, vcomb = losses(vam)
            val_r2 = float(1.0 - F.mse_loss(vcomb, y_val) / val_var)
        history["train_loss"].append(float(loss.detach()))
        history["val_loss"].append(float(vloss.detach()))
        history["val_r2"].append(val_r2)
        sched.step(val_r2)
        if verbose:
            pbar.set_postfix(train=f"{float(loss.detach()):.4f}", val_r2=f"{val_r2:.4f}")

        if val_r2 > best_r2 + 1e-6:
            best_r2, bad = val_r2, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history, log_w

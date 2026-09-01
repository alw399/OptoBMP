"""
Reading a trained `IFPredictor` back out: which features it uses, what shape its
response has, and how far across the graph its information actually travels.

Five complementary views, deliberately not one -- each is wrong in a different,
known way, so the useful signal is where they AGREE:

  `integrated_gradients` -- exact credit assignment for the prediction the model
      actually made: attributions sum to `f(x) - f(baseline)` (completeness, which
      `ig_completeness` checks rather than assumes). Its known failure mode is
      CORRELATED inputs: the pyramid scales here run ~0.9+ correlated with each
      other, and IG splits credit between correlated features along the chosen
      path, so a single scale's share is not "the information lives at this scale".

  `permutation_importance` -- the model-agnostic cross-check: how much score is
      actually lost when a feature is destroyed. Correlated features bias this the
      OPPOSITE way (a feature whose information is duplicated by its neighbours in
      the pyramid looks free to destroy, however much the model leans on it), so a
      feature that scores high on both is genuinely load-bearing, and a feature
      high on IG but ~0 on permutation is redundant rather than unimportant.

  `partial_dependence` -- the SHAPE of the learned response (monotone? saturating?
      thresholded?), which no importance ranking gives you.

  `edge_sensitivity` -- how much of the prediction arrives through the graph at
      all, resolved by edge distance and layer. This is the only view that
      separates "the cell's own features" from "its neighbours'".

  `counterfactual_prediction` -- what the model predicts for an illumination
      pattern that was never imaged (e.g. a synthetic disc). The other four
      explain the fit; this one tests whether the learned rule is a spatial
      dose-response or a memorised map of this particular well.

Everything here explains the CALIBRATED output (`models.train.calibrated_predict`:
`prob = sigmoid(logit - log_w)`, prediction = `prob * magnitude`), because that is
the quantity every reported R2/AUROC is computed on. The correction is a constant
shift in the logit, but `sigmoid` is nonlinear, so attributions of the calibrated
and raw outputs are genuinely different -- do not mix them.

Two model-specific gotchas that apply to every function here:

* `IFPredictor.save_checkpoint` does NOT store `log_w`, the `x_scaler`, or the
  train split. Reconstruct all three the same way training did -- rebuild `data`
  with `recipe.build_data(df, x_cols, y_col, train_idx)` using the SAME
  `train_idx`, then `log_w = train_log_w(data, train_mask)`. `check_reconstruction`
  exists to verify you did: if the reconstructed R2/AUROC don't match the training
  notebook's, the scaler or the split is different and every attribution below is
  being computed for a model/input pair that never existed.

* `BatchNorm` layers are put in eval mode (running statistics) throughout, so `f`
  is a deterministic function of `x` alone. In train mode a perturbed input would
  also move the normalisation statistics, and attributions would partly describe
  that instead of the model.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from tools.evaluation import evaluate

TARGETS = ("combined", "prob", "logit", "magnitude")


# --------------------------------------------------------------------------
# The function being explained
# --------------------------------------------------------------------------


def train_log_w(data: dict, train_mask: Optional[torch.Tensor] = None) -> float:
    """Recompute `train_calibrated`'s `log_w` from the data + train split.

    `train_calibrated` sets `pos_weight = n_negative / n_positive` on the TRAIN
    rows and returns `log_w = log(pos_weight).mean()`; `calibrated_predict` then
    predicts `sigmoid(logit - log_w) * magnitude`. Since `save_checkpoint` doesn't
    persist it, this reproduces it exactly -- but only if `train_mask` is the same
    split the model was trained with. Passing the WRONG `log_w` (or a raw
    `pos_weight` where a log was wanted -- an easy off-by-a-log to make, and worth
    ~0.2 R2 here) leaves ranking metrics untouched while wrecking R2, so a big
    R2/AUROC disagreement is the signature to check for.
    """
    labels = data["y_positive"] if train_mask is None else data["y_positive"][train_mask]
    n_pos = labels.sum(dim=0).clamp(min=1.0)
    n_neg = labels.shape[0] - labels.sum(dim=0)
    return float(torch.log((n_neg / n_pos).clamp(min=1.0)).mean().item())


def calibrated_forward(model, x: torch.Tensor, edge_index: torch.Tensor,
                       edge_weight: torch.Tensor, log_w: float,
                       target: str = "combined") -> torch.Tensor:
    """The differentiable version of `models.train.calibrated_predict`, with the
    hurdle head's two branches individually addressable.

    `target`:
      'combined'  -- `sigmoid(logit - log_w) * magnitude`, the scored prediction.
      'prob'      -- the calibrated probability branch alone ("is this cell
                     positive"), which is what AUROC is really measuring.
      'logit'     -- same branch before the sigmoid. Attributions on the logit are
                     the ones to read when the probability is saturated near 0/1,
                     where the sigmoid's vanishing gradient shrinks every
                     attribution toward 0 for reasons that are about the squashing
                     function, not about the feature.
      'magnitude' -- the "how bright, GIVEN positive" branch, only ever supervised
                     on truly-positive cells (see `TwoPartHead`), so its
                     attributions are only meaningful on/near that population.

    Splitting 'prob' from 'magnitude' is the point: this model can only score well
    by getting both right, and a feature can easily drive one branch and not the
    other -- reading only 'combined' hides that entirely.
    """
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
    _, logit, magnitude, _ = model.forward_two_part(x, edge_index, edge_weight)
    logit = logit - log_w
    if target == "logit":
        return logit
    if target == "magnitude":
        return magnitude
    prob = torch.sigmoid(logit)
    return prob if target == "prob" else prob * magnitude


@torch.no_grad()
def predict(model, data: dict, log_w: float, x: Optional[torch.Tensor] = None,
            target: str = "combined", device: str = "cpu") -> np.ndarray:
    """`calibrated_forward` as a plain numpy vector, optionally on a SUBSTITUTED
    `x` (same graph, same scaler, different feature values) -- the workhorse for
    every perturbation-based method below."""
    model = model.to(device).eval()
    x = data["x"] if x is None else x
    out = calibrated_forward(model, x.to(device), data["edge_index"].to(device),
                             data["edge_weight"].to(device), log_w, target)
    return out.cpu().numpy().ravel()


def check_reconstruction(model, data: dict, log_w: float, eval_mask: torch.Tensor,
                         device: str = "cpu") -> dict:
    """Score a LOADED checkpoint against the data dict you rebuilt for it.

    Run this before trusting anything else in this module. A checkpoint carries
    weights only, so a mismatched `train_idx` (different scaler fit), a different
    `x_cols` order, or a wrong `log_w` all load without error and quietly explain
    a model that was never trained. Compare the returned metrics against the
    training notebook's table for the same split.
    """
    pred = predict(model, data, log_w, device=device)
    m = eval_mask.cpu().numpy().astype(bool)
    return evaluate(data["y"].numpy().ravel()[m], pred[m],
                    data["y_positive"].numpy().ravel()[m])


# --------------------------------------------------------------------------
# Integrated gradients
# --------------------------------------------------------------------------


def _node_mask(nodes, n: int, device) -> torch.Tensor:
    if nodes is None:
        return torch.ones(n, dtype=torch.bool, device=device)
    t = torch.as_tensor(np.asarray(nodes))
    if t.dtype == torch.bool:
        return t.to(device)
    out = torch.zeros(n, dtype=torch.bool, device=device)
    out[t.long().to(device)] = True
    return out


def reference_baseline(data: dict, nodes, device: str = "cpu") -> torch.Tensor:
    """The mean scaled feature vector of a chosen REFERENCE population, as an IG
    baseline -- e.g. cells far outside the illuminated pattern, making every
    attribution read "relative to an unstimulated cell". This is the baseline to
    reach for here, and the total then answers a question worth asking: how much
    of the predicted signal in this well exists BECAUSE of the illumination?

    The default zeros baseline is not neutral in the same way: `build_data`
    standardizes every non-binary column, so 0 is that column's TRAIN MEAN (a
    typical cell of this well, not an unstimulated one), while the genuinely binary
    columns skip standardization, so 0 there really is "not illuminated". Mixed
    like that, it defines a cell that could not exist, and -- being an average --
    it predicts close to the average, so the whole attribution budget it explains
    is near zero (see `integrated_gradients`' completeness note). IG is well
    defined for any baseline; the sentence you can write about the result is not.
    """
    sel = _node_mask(nodes, data["x"].shape[0], "cpu")
    return data["x"][sel].mean(dim=0, keepdim=True).to(device)


def integrated_gradients(model, data: dict, log_w: float, target: str = "combined",
                         baseline: Optional[Union[torch.Tensor, np.ndarray]] = None,
                         steps: int = 32, nodes=None, device: str = "cpu",
                         verbose: bool = True) -> dict:
    """Integrated gradients over the node features, for a graph model.

    Returns `{'attr': (N, F) array, 'total', 'delta', 'error', ...}` where
    `attr[j, f]` is how much CELL j's feature f contributed to the summed
    prediction over the explained cells (`nodes`, default all).

    Why the sum over cells rather than one attribution per cell: message passing
    means cell i's prediction depends on its neighbours' features too, so a fully
    per-cell answer is an (N, N, F) object. Differentiating the SUM collapses that
    to the per-cell TOTAL INFLUENCE of each cell's features -- on its own
    prediction and on those of every cell it reaches. That is exact, not an
    approximation, and it costs one backward pass per integration step instead of
    N. It is also the right granularity here: with `radius=30` and 3 layers a
    cell's influence stops at ~90px, so `attr[j]` stays a local, mappable quantity
    (`plotting.interpret.plot_attribution_maps` plots it directly).

    The path integral uses the midpoint rule over `steps` points along
    `baseline -> x`. Attributions are in PREDICTION UNITS (the same scaled units
    `data['y']` is in, for 'combined'), so they add up: `attr.sum() ==
    f(x) - f(baseline)` summed over the explained cells, up to integration error.

    Completeness is reported two ways, and the difference matters:
      `error`      -- `|gap| / |delta|`, the textbook relative completeness.
      `error_mass` -- `|gap| / sum|attr|`, the gap against the total attribution
                      mass actually moved.
    A NEAR-NEUTRAL baseline makes the first one explode while the second stays
    tiny, and that is not a failure: with `baseline=None` (zeros = the train mean
    of every standardized column) the model predicts almost exactly the same TOTAL
    as it does on the real data, so `delta` is a small difference of two large
    numbers and dividing by it is meaningless. Measured here: zeros gives
    `delta = -22` against `f_x = 8260` (`error` >100%, `error_mass` 0.0002),
    while an unstimulated `reference_baseline` gives `delta = 7300` (`error` 0.02%)
    -- same attributions, honest denominator. Judge convergence on `error_mass`,
    and pick the baseline for the QUESTION, not for the completeness number.

    `steps=32` puts `error_mass` at ~5e-4 for this model; it falls roughly as
    1/steps, and each step is one forward + backward over the whole graph.
    """
    model = model.to(device).eval()
    x = data["x"].to(device)
    ei, ew = data["edge_index"].to(device), data["edge_weight"].to(device)
    sel = _node_mask(nodes, x.shape[0], device)

    if baseline is None:
        base = torch.zeros_like(x)
    else:
        base = torch.as_tensor(np.asarray(baseline), dtype=x.dtype, device=device)
        base = base.expand_as(x) if base.ndim == 2 else base.reshape(1, -1).expand_as(x)
        base = base.contiguous()

    grads = torch.zeros_like(x)
    alphas = (torch.arange(steps, dtype=x.dtype, device=device) + 0.5) / steps
    for alpha in tqdm(alphas, desc=f"IG:{target}", disable=not verbose, leave=False):
        xa = (base + alpha * (x - base)).requires_grad_(True)
        out = calibrated_forward(model, xa, ei, ew, log_w, target)
        grads += torch.autograd.grad(out[sel].sum(), xa)[0]
    attr = ((x - base) * grads / steps).detach()

    with torch.no_grad():
        f_x = calibrated_forward(model, x, ei, ew, log_w, target)[sel].sum().item()
        f_b = calibrated_forward(model, base, ei, ew, log_w, target)[sel].sum().item()
    total = float(attr.sum().item())
    delta = f_x - f_b
    gap = total - delta
    mass = float(attr.abs().sum().item())
    return dict(attr=attr.cpu().numpy(), total=total, delta=delta, gap=gap,
                error=abs(gap) / max(abs(delta), 1e-12),
                error_mass=abs(gap) / max(mass, 1e-12),
                f_x=f_x, f_baseline=f_b, target=target, steps=steps,
                x_cols=list(data["x_cols"]), n_explained=int(sel.sum().item()))


def attribution_table(ig: dict, nodes=None) -> pd.DataFrame:
    """Per-feature summary of an `integrated_gradients` result, sorted by influence.

    * `total`      -- summed signed attribution, in prediction units. These add up
                      to `ig['total']`, so a feature's row is literally "this much
                      of the predicted signal".
    * `mean`       -- signed mean per cell. Sign is the direction the feature
                      pushed the prediction ON AVERAGE; a feature that raises the
                      prediction for illuminated cells and lowers it elsewhere
                      cancels here, which is why `mean_abs` is reported too.
    * `mean_abs`   -- mean |attribution| per cell: how much the feature MOVED the
                      prediction, cancellation aside. This is the importance
                      ranking to read.
    * `share`      -- `mean_abs` as a fraction of all features' `mean_abs`.
    * `frac_positive` -- fraction of cells the feature pushed UP. Near 0.5 means a
                      genuinely two-sided feature; near 0 or 1 means one-directional.
    """
    attr = ig["attr"]
    if nodes is not None:
        attr = attr[_node_mask(nodes, attr.shape[0], "cpu").numpy()]
    mean_abs = np.abs(attr).mean(axis=0)
    table = pd.DataFrame({
        "feature": ig["x_cols"],
        "total": attr.sum(axis=0),
        "mean": attr.mean(axis=0),
        "mean_abs": mean_abs,
        "share": mean_abs / max(mean_abs.sum(), 1e-12),
        "frac_positive": (attr > 0).mean(axis=0),
    })
    return table.sort_values("mean_abs", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Perturbation-based importance
# --------------------------------------------------------------------------


def permutation_importance(model, data: dict, log_w: float, eval_mask: torch.Tensor,
                           n_repeats: int = 5, seed: int = 0, device: str = "cpu",
                           metrics: Sequence[str] = ("r2", "auroc", "spearman"),
                           verbose: bool = True) -> pd.DataFrame:
    """Score drop when one feature column is shuffled across cells.

    The column is permuted for EVERY node, not only the evaluated ones: a node's
    features reach its neighbours' predictions through the graph, so leaving the
    neighbours intact would measure something the model never experiences.
    Shuffling (rather than zeroing) keeps the marginal distribution intact, so the
    model stays on-distribution per-feature and only the feature's relationship to
    space -- and to the other features -- is destroyed.

    Returns one row per feature: `{metric}_drop` (baseline minus permuted, so
    LARGER = more important) and its std over `n_repeats`. The unpermuted baseline
    metrics are in `.attrs['baseline']`.

    Read it against `attribution_table`, not instead of it: strongly correlated
    features (which the pyramid scales are, by construction) can each show a drop
    near zero while jointly carrying the whole signal, because the model can
    recover a shuffled scale from its neighbouring scales.
    """
    x = data["x"]
    y = data["y"].numpy().ravel()
    ypos = data["y_positive"].numpy().ravel()
    m = eval_mask.cpu().numpy().astype(bool)
    base = evaluate(y[m], predict(model, data, log_w, device=device)[m], ypos[m])

    rng = np.random.default_rng(seed)
    rows = []
    cols = list(data["x_cols"])
    for j, col in enumerate(tqdm(cols, desc="permute", disable=not verbose, leave=False)):
        for _ in range(n_repeats):
            x2 = x.clone()
            x2[:, j] = x[torch.from_numpy(rng.permutation(x.shape[0])), j]
            met = evaluate(y[m], predict(model, data, log_w, x=x2, device=device)[m], ypos[m])
            rows.append(dict(feature=col, **{k: met.get(k, np.nan) for k in metrics}))

    raw = pd.DataFrame(rows)
    out = pd.DataFrame({"feature": cols})
    for k in metrics:
        agg = raw.groupby("feature", sort=False)[k].agg(["mean", "std"])
        out[f"{k}_drop"] = out["feature"].map(base.get(k, np.nan) - agg["mean"])
        out[f"{k}_drop_std"] = out["feature"].map(agg["std"])
    out = out.sort_values(f"{metrics[0]}_drop", ascending=False).reset_index(drop=True)
    out.attrs["baseline"] = base
    return out


def unscale_column(data: dict, feature: Union[str, int], values) -> np.ndarray:
    """Scaled feature values -> the units the dataframe column is in.

    `build_data`'s columns are already normalized before scaling (a fraction, a
    log-distance, a 0/1 flag, or `log1p(intensity - background)` -- see
    `recipe.normalize`), so this inverts the standardization ONLY: the result is
    still in those normalized units, which is what the axis label should say.
    """
    cols = list(data["x_cols"])
    j = cols.index(feature) if isinstance(feature, str) else int(feature)
    values = np.asarray(values, dtype=np.float32).ravel()
    grid = np.zeros((len(values), len(cols)), dtype=np.float32)
    grid[:, j] = values
    return data["x_scaler"].inverse_transform(grid)[:, j]


def partial_dependence(model, data: dict, log_w: float, feature: Union[str, int],
                       grid: Optional[Sequence[float]] = None, n_grid: int = 21,
                       quantiles: tuple = (0.01, 0.99), nodes=None, n_ice: int = 0,
                       target: str = "combined", seed: int = 0, device: str = "cpu",
                       verbose: bool = True) -> pd.DataFrame:
    """Sweep one feature across its observed range and watch the prediction move.

    The feature is overwritten for EVERY cell at each grid point -- the honest
    counterfactual for a graph model, and it reads as "what if the whole field had
    this much BMP4 at this scale", not "what if this one cell did". A per-cell-only
    sweep would leave each cell's neighbours at their real values and understate
    the effect by however much of it arrives through the graph.

    Everything else is left at its real value, so with correlated features this is
    an EXTRAPOLATION at the edges of the grid: setting `gauss_240` to its 99th
    percentile while `gauss_15` keeps its real value describes a field no well
    could produce. The middle of the curve is the trustworthy part, and the
    default 1st-99th percentile range keeps the ends from being pure fantasy.

    Returns a tidy frame (`x_scaled`, `x`, `pred_mean`, `pred_q25`, `pred_q75`,
    plus `ice_{k}` columns for `n_ice` randomly chosen individual cells, whose
    spread across the grid is what tells you whether the mean curve describes any
    real cell or is an average over two opposite behaviours).
    """
    cols = list(data["x_cols"])
    j = cols.index(feature) if isinstance(feature, str) else int(feature)
    x = data["x"]
    sel = _node_mask(nodes, x.shape[0], "cpu").numpy()
    if grid is None:
        lo, hi = np.quantile(x[:, j].numpy(), quantiles)
        grid = np.linspace(float(lo), float(hi), n_grid)
    grid = np.asarray(grid, dtype=np.float32)

    ice_idx = np.array([], dtype=int)
    if n_ice > 0:
        ice_idx = np.random.default_rng(seed).choice(np.where(sel)[0], size=n_ice, replace=False)

    rows = []
    for v in tqdm(grid, desc=f"PD:{cols[j]}", disable=not verbose, leave=False):
        x2 = x.clone()
        x2[:, j] = float(v)
        pred = predict(model, data, log_w, x=x2, target=target, device=device)
        row = dict(x_scaled=float(v), pred_mean=float(pred[sel].mean()),
                   pred_q25=float(np.quantile(pred[sel], 0.25)),
                   pred_q75=float(np.quantile(pred[sel], 0.75)))
        row.update({f"ice_{k}": float(pred[i]) for k, i in enumerate(ice_idx)})
        rows.append(row)

    out = pd.DataFrame(rows)
    out.insert(1, "x", unscale_column(data, j, out["x_scaled"].to_numpy()))
    out.attrs["feature"] = cols[j]
    out.attrs["target"] = target
    return out


# --------------------------------------------------------------------------
# What the graph itself contributes
# --------------------------------------------------------------------------


def edge_sensitivity(model, data: dict, centroids: np.ndarray,
                     bin_px: float = 5.0, device: str = "cpu") -> pd.DataFrame:
    """How much of the prediction arrives over edges of each LENGTH, per layer.

    Wraps `IFPredictor.message_sensitivity_by_layer` (grad x message per edge; see
    its docstring for why summing over destination cells before backpropagating is
    exact here) and bins the result by the physical distance between the two cells
    an edge connects. Distance 0 is the SELF-LOOP `build_radius_graph` adds for
    every node, so the `dist == 0` row is exactly "the cell's own features", and
    everything above it is "its neighbours'".

    That split is the question the feature-importance views can't answer: these
    inputs are already multi-scale neighbourhood summaries, so a model could in
    principle ignore message passing entirely and read everything off each cell's
    own columns. The self-loop share says whether it did. (The 3 x 30px layer
    stack reaches ~90px total, well short of the 240px top pyramid level -- so
    long-range context is expected to arrive as node FEATURES, and only short-range
    structure through the graph.)

    Returns a long frame: one row per (layer, distance bin) with the summed |grad x
    message| in that bin, its signed sum, the edge count, and `share` within the
    layer. Layer 0's messages are a direct function of the raw feature columns;
    later layers' are functions of the previous layer's embedding, so their
    distance profile is about information FLOW, not about any specific feature.
    """
    model = model.to(device).eval()
    sens = model.message_sensitivity_by_layer(
        data["x"].to(device), data["edge_index"].to(device), data["edge_weight"].to(device))
    src, dst = data["edge_index"].cpu().numpy()
    d = np.hypot(centroids[src, 0] - centroids[dst, 0], centroids[src, 1] - centroids[dst, 1])
    edges = np.concatenate([[-0.5, 0.5], np.arange(bin_px, d.max() + bin_px, bin_px)])
    which = np.digitize(d, edges) - 1
    centres = np.where(np.arange(len(edges) - 1) == 0, 0.0, (edges[:-1] + edges[1:]) / 2)

    rows = []
    for layer, s in enumerate(sens):
        s = s.cpu().numpy()
        abs_sum = np.bincount(which, weights=np.abs(s), minlength=len(centres))
        signed = np.bincount(which, weights=s, minlength=len(centres))
        count = np.bincount(which, minlength=len(centres))
        for k in np.where(count > 0)[0]:
            rows.append(dict(layer=layer, dist=float(centres[k]), self_loop=bool(k == 0),
                             abs_sum=float(abs_sum[k]), signed_sum=float(signed[k]),
                             n_edges=int(count[k])))
    out = pd.DataFrame(rows)
    out["share"] = out["abs_sum"] / out.groupby("layer")["abs_sum"].transform("sum")
    return out


# --------------------------------------------------------------------------
# Counterfactual illumination
# --------------------------------------------------------------------------


def counterfactual_prediction(df: pd.DataFrame, model, data: dict, log_w: float,
                              feature_fn: Callable[[pd.DataFrame], Sequence[str]],
                              device: str = "cpu") -> np.ndarray:
    """Predict on a MODIFIED dataframe, reusing the fitted scaler and the graph.

    `feature_fn(df)` recomputes the model's x_cols on the modified frame in place
    and returns their names -- e.g. `lambda d: recipe.mask_features(d, 'BMP4_bin')`
    after overwriting `BMP4_bin` with a synthetic pattern. Recomputing the pyramid
    is the whole point: nudging the scaled feature matrix directly would produce
    a feature combination no illumination pattern can generate (a mask fraction at
    240px that contradicts the one at 15px), whereas rebuilding from a mask keeps
    the counterfactual physically realisable.

    The graph and the scaler are deliberately REUSED: cell positions don't change,
    so `edge_index`/`edge_weight` are still exactly right, and the scaler must be
    the fitted one (refitting on the counterfactual would silently renormalize the
    thing being varied, hiding the effect being measured).

    Caveat worth stating whenever a number from this comes out: a synthetic pattern
    unlike anything in training is an extrapolation, and the model has no way to
    tell you so. Treat the SHAPE of the response (how far it extends, how sharply
    it falls) as the result, not the absolute level.
    """
    cols = list(feature_fn(df))
    expected = list(data["x_cols"])
    if cols != expected:
        raise ValueError(f"feature_fn returned {cols}, but the model expects {expected}")
    x = torch.from_numpy(
        data["x_scaler"].transform(df[cols].to_numpy(dtype=np.float32)).astype(np.float32))
    return predict(model, data, log_w, x=x, device=device)


def disc_mask(centroids: np.ndarray, centre: tuple[float, float], radius: float,
              inner_radius: float = 0.0, fill: float = 1.0, seed: int = 0) -> np.ndarray:
    """A synthetic illumination mask: 1.0 for cells in a disc (or an annulus, with
    `inner_radius`) around `centre`, 0.0 elsewhere. Float, matching `{channel}_bin`.
    Feed it through `counterfactual_prediction` to ask what the model predicts for a
    pattern of a size that was never imaged.

    `fill` < 1 marks only that random FRACTION of the cells inside the disc, which
    is the knob that matters for staying in-distribution: a real `{channel}_bin` is
    a per-cell threshold call, so even a fully illuminated region comes out as a
    scattering of positive cells, never a solid block. A `fill=1.0` disc is a
    pattern the model has never seen at any radius, and any sharp structure it
    predicts there could be extrapolation -- run a matched `fill` alongside it (the
    observed positive rate inside the real pattern) before believing the shape.
    """
    r = np.hypot(centroids[:, 0] - centre[0], centroids[:, 1] - centre[1])
    inside = (r <= radius) & (r >= inner_radius)
    if fill < 1.0:
        inside &= np.random.default_rng(seed).random(len(centroids)) < fill
    return inside.astype(np.float32)

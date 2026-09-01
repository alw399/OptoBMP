# Mask → marker prediction: the recommended model

Reference for `src/models/recipe.py`, `predictors.py`, `graph.py`, `scalers.py`,
`data.py`, `train.py` and `checkpoint.py`, plus the feature code in
`src/tools/spatial.py`/`dataset.py`, as actually used by
`notebooks/03_predict_from_mask.ipynb`, `05_train_and_cross_predict.ipynb`, and
`06_predict_from_intensity.ipynb`. Every number below is read directly from the
code — this is what the code does, including the effects of the defaults every
training run in this project actually uses, not a description of what the
parameters *could* be set to.

This model supersedes the one documented in [`gnn_gat_methods.md`](gnn_gat_methods.md)
(`src/models/gnn.py`/`gat.py`, now removed). The differences are called out
throughout, and summarized at the end.

## 1. Task

Predict a cell's own immunofluorescence level for one marker channel at a time
(`y_col`, e.g. `Sox17_mean` or `T_mean`) from **a single binary mask column**
(e.g. `BMP4_bin`) — both the cell's own mask value and a multiscale summary of its
neighbors', learned through message passing over a spatial graph.

Unlike the legacy model, there is no separate "global" pathway concatenated in at
the head. Every input, including the cell's own value, reaches the prediction only
through the graph's self-inclusive message passing (`§3`) — the self-loop *is* the
mechanism that used to require a separate `global_x_cols` concatenation.

## 2. Features (`x_cols`, `models.recipe.mask_features`)

```python
X_MASK = recipe.mask_features(df, 'BMP4_bin', max_sigma=recipe.MAX_SIGMA, geometry=True)
```

**What "pyramid" means here.** `tools.spatial.PYRAMID_SCALES = (15, 30, 60, 120,
240, 480, 960, 1920)` — a geometric sequence of Gaussian smoothing scales (σ, in
px). For a given per-cell signal, the code smooths it at every one of these scales
and keeps EACH scale as its own feature column (`_gauss_15`, `_gauss_30`, …) —
the same idea as the classic image-processing "Gaussian pyramid" (a stack of
increasingly blurred copies of the same signal), just evaluated per cell rather
than per pixel. The reason it exists at all: the graph (`§3`) only reaches ~90px
(`radius=30` × 3 hops), but the real spatial response extends to ~600px, and an
actual 600px-radius graph would need ~8×10⁸ edges. The pyramid supplies that
longer-range context as plain input features instead, computed by rasterizing
cells onto an 8px grid and Gaussian-filtering (`O(pixels)` per scale, regardless of
how large the scale is — not a KD-tree neighbor search, which would blow up at
large σ). `tools.spatial.add_mask_pyramid`/`add_density_pyramid`/
`add_intensity_pyramid` are three different SIGNALS run through this same
machinery — see the intensity variant below.

Calls `tools.spatial.add_mask_pyramid` (adds every column below, PLUS
`{mask_col}_signed_dist`, to `df` in place), then keeps only the pyramid scales
`<= max_sigma` and drops `signed_dist` (see the callout below). At the recommended
`MAX_SIGMA = 240`, the 8 columns actually used are:

| column | what it is | how it's computed |
|---|---|---|
| `BMP4_bin` | the raw binary mask itself | thresholded intensity, from `tools.dataset.load_well` |
| `BMP4_bin_gauss_15` … `_240` (5 cols) | local **positive fraction** of the mask at Gaussian scale σ | `multiscale_fraction`: rasterize cells to an 8px grid (`BIN_PX`), Gaussian-filter the positive-count grid and the all-cell grid separately at σ, divide — the local mean of the mask, not a raw smoothed count, so "few positive cells" isn't confused with "few cells" |
| `BMP4_bin_log_dist_pos` | `log1p(distance to nearest mask-positive cell)` | `distance_to` via a `cKDTree` query |
| `BMP4_bin_log_dist_neg` | `log1p(distance to nearest mask-negative cell)` | same, opposite class |

**Not used: `BMP4_bin_signed_dist`** — signed distance to the illuminated boundary
(**+** inside a positive region, **−** outside, magnitude = distance to the nearest
cell of the opposite class; `signed_boundary_distance`). `add_mask_pyramid` still
computes and attaches it to `df` — notebook 01's response-vs-geometry section uses
it directly as a standalone diagnostic (it's the dose-response coordinate, and the
only feature that separates "inside a ring with no positive cell nearby" from "far
outside the pattern with no positive cell nearby", two situations every
density/fraction feature above reports identically) — but `mask_features`
deliberately excludes it from the model's own `x_cols`. Reasoning: it's an
unbounded raw distance whose scale tracks the SIZE of the illuminated pattern in a
given well, so a `StandardScaler` fit on one well's pattern doesn't obviously
transfer to a well with a differently-sized one, and it's only ever been validated
against the one matched well pair (`W8_pattern1`↔`W9_pattern1`), which shares the
same ring geometry. `log_dist_pos`/`log_dist_neg` carry the same "how close is the
nearest signal source" information without the inside/outside sign, and — being
one-sided, log-compressed distances rather than a raw signed one — are kept.

**Why cap at σ = 240px, not the full pyramid (up to 1920px)?** From
`recipe.py`'s own docstring: a `radius=30` × 3-layer graph (`§3`) only reaches ~90px,
but 86% of cells have no mask-positive neighbor within 30px at all — their input
would otherwise be a constant. Scales above ~240px fix that gap but start acting as
**position proxies** instead of real context: at σ=960px a feature varies over
roughly the image's own scale, so it can encode "where am I in this particular
well", which is worthless (or actively harmful) in a well whose pattern sits
somewhere else. On the one directly comparable well pair (`W8_pattern1` ↔
`W9_pattern1` — same illumination power, same imaging session), capping at
120–240px maximizes *both* the within-well blocked score and the cross-well score;
extending to 1920px keeps nudging the within-well number up while cross-well T ρ
drops from 0.44 to 0.31.

**The control feature set** — `recipe.density_features(df, max_sigma)` — swaps the
mask pyramid for a smoothed **cell-density** pyramid (`tools.spatial.
add_density_pyramid`, no marker/mask information at all) at the same scales. Not
part of the recommended model; it's the baseline notebook 04 uses to check how much
of a mask-based score is actually attributable to the mask rather than to the local
cell-density field the mask happens to correlate with.

### 2.1 A continuous alternative — `recipe.intensity_features` (notebook 06)

Every feature set above is built from the **thresholded** `BMP4_bin` mask — a 0/1
call that throws away how bright a positive cell actually was. Notebook 06 asks
whether the underlying **continuous** channel, `BMP4_mean`, helps instead of (or
alongside) it:

```python
X_INTENSITY = recipe.intensity_features(df, 'BMP4_mean', max_sigma=recipe.MAX_SIGMA)
# -> ['BMP4_mean', 'BMP4_mean_mean_15', 'BMP4_mean_mean_30', 'BMP4_mean_mean_60',
#     'BMP4_mean_mean_120', 'BMP4_mean_mean_240']   (6 columns, vs. X_MASK's 8)
```

Calls `tools.spatial.add_intensity_pyramid`, which reuses `multiscale_mean` — the
same rasterize-then-Gaussian-filter machinery `multiscale_fraction`/
`multiscale_density` use, just averaging a CONTINUOUS value instead of a positive
fraction or a raw count (`smooth(Σ values) / smooth(cell count)` at each σ, i.e.
`tools.spatial.smoothed_field` evaluated at every cell's own position). Two
structural differences from `X_MASK`:

- **Fewer scales kept by default** — `multiscale_mean`/`add_intensity_pyramid` were
  added for this specific comparison and don't (yet) get the same `box=True`
  hard-disc variant `multiscale_fraction` optionally computes; not a limitation of
  the underlying scale sweep, just nothing downstream currently asks for it.
- **No geometry columns.** `distance_to` (the `_log_dist_*` columns in `X_MASK`)
  needs a binary class to define "inside" vs. "outside" of — a continuous channel
  has no such split on its own, so there is no intensity equivalent of them (nor of
  `signed_dist`, though that one is excluded from `X_MASK` too — see `§2`).

`notebooks/06_predict_from_intensity.ipynb` fits three feature sets side by side
under the otherwise-identical recipe: `X_MASK` alone, `X_INTENSITY` alone, and their
union (`X_MASK + X_INTENSITY`, 14 columns). See `§4`'s `intensity_cols` for how the
preprocessing pipeline treats `X_INTENSITY`'s columns differently from `X_MASK`'s.
The notebook deliberately asserts no winner — read its own results tables for
which feature set actually wins on your run, at `QUICK=False`.

## 3. Graph construction (`models.graph.build_radius_graph`)

Same underlying radius search as the legacy model (`tools.morphology.
radius_edge_index`, shared code — every pair of cells within `radius` px connected,
`min_neighbors=1` fallback to nearest-neighbor for an isolated cell so no node is
ever left with zero edges), plus one **self-loop** per node at distance 0:

```
edge_index = radius_pairs(centroids, radius) ∪ {(i, i) for every cell i}
w_e = exp(-dist_e / length_scale)          # length_scale defaults to radius itself; self-loop -> w = 1
```

**`normalize_data=False`** — the recommended model's key departure from the legacy
default (`normalize_weights=True` there). `WeightedRadiusConv` aggregates with
`aggr="add"`, so a **row-normalized** weight (sums to 1 per destination, the legacy
default) makes aggregation a weighted *average* — structurally insensitive to
neighbor *count*, which is why the legacy model needed `local_density` as a
separate explicit input. Leaving the raw `exp(-dist/length_scale)` weight
un-normalized makes aggregation a weighted **sum** instead: a cell with more/closer
neighbors accumulates a strictly larger raw signal, so density becomes implicitly
recoverable from the embedding's own magnitude — see `§5`'s extra `BatchNorm1d` for
how that magnitude signal survives the network rather than being normalized away.

`normalize_data` must be the *same* value passed to `build_prediction_data` (the
data pipeline) and to `IFPredictor`'s own constructor (the architecture) — one
decision shared by both, not two independently-set flags that happen to need to
agree (a prior version had it that way as `normalize_weights`, and a mismatch
silently broke training).

**Border exclusion** (`models.graph.border_mask`) — unchanged from the legacy
model: a cell within `radius` px of the image edge has an undercounted
neighborhood, so it's excluded from ever being a train/val/test *target*, but still
participates as a neighbor for cells further from the edge.

## 4. Preprocessing (`models.recipe.build_data` → `models.data.build_prediction_data`)

Two different scalers, one for `x`, one for `y` — both fit on **train rows only**.

### x_cols → `ColumnScaler`, with every per-column special case turned off

```python
build_prediction_data(..., scaling='log1p_standard',
    no_background_cols=x_cols, no_transform_cols=x_cols,
    no_scale_cols=[c for c in x_cols if df[c] is {0,1}-valued],   # BMP4_bin only
    normalize_data=False, two_part=True)
```

- `no_background_cols=x_cols` — skip background subtraction entirely. Every column
  here is a fraction, a distance, or a 0/1 flag; none of them has a camera-offset
  floor to subtract the way a raw intensity channel does.
- `no_transform_cols=x_cols` — skip the `log1p` variance-stabilizing step too, same
  reasoning: nothing here has the long multiplicative right tail that transform
  exists to compress.
- `no_scale_cols` — skip the final `StandardScaler` step, but **only** for the
  genuinely-binary `BMP4_bin` column: standardizing a skewed 0/1 flag divides by a
  small `std` and inflates the rarer class into an outlier-sized value purely from
  class imbalance, not real signal (`ColumnScaler`'s own documented rationale).

Net effect: every continuous feature (`_gauss_*`, `_signed_dist`, `_log_dist_*`)
gets `StandardScaler` alone; `BMP4_bin` passes through completely untouched.

**`intensity_cols`** (`recipe.build_data`/`recipe.fit`, default `()`) — an escape
hatch for feeding the model a *continuous* channel instead of (or alongside) the
binary mask, e.g. raw `BMP4_mean` rather than `BMP4_bin`
(`notebooks/06_predict_from_intensity.ipynb`). Any `x_col` named here is excluded
from the blanket exemption above, so it gets the *same* background-subtract +
`log1p` treatment `if_cols` get — because unlike a fraction/distance/flag, a real
intensity channel *does* have a camera-offset floor and a multiplicative right
tail. `recipe.intensity_features(df, value_col, max_sigma)` builds the matching
multiscale local-mean pyramid (`tools.spatial.add_intensity_pyramid` — the
continuous analog of `add_mask_pyramid`'s positive-fraction pyramid) and returns
exactly the column list to pass through as `intensity_cols`.

### y_col → `HurdleScaler`, since `two_part=True`

```
threshold = custom_background[y_col]                       # NOT the automatic bimodal estimate
y_positive = (raw > threshold)                              # same threshold, doubles as the classifier label
y = log1p(clip(raw - threshold, min=0)) / scale_            # scale_ = max over TRAIN positives
```
— exactly 0 at or below `threshold`, in `(0, ~1]` above it.

This project always supplies `custom_background=tools.dataset.IF_BACKGROUNDS`
(349.27 for `Sox17_mean`, 384.42 for `T_mean`) rather than letting
`tools.qc.get_bimodal_threshold` estimate it automatically. Per `dataset.py`'s own
docstring, that automatic estimate (log10-transform the positive values → Gaussian
KDE → valley between the two tallest peaks) is **not stable across wells** — run on
the same `Sox17` channel it returns anywhere from 240 to 3408 depending on the well,
even though the wells' median intensities all sit within 12% of each other.
`IF_BACKGROUNDS` are the thresholds verified **by eye** on `W8_pattern1`
specifically; using them elsewhere assumes that well's threshold transfers, which
`03`/`05` flag explicitly wherever they evaluate on a different well.

## 5. Model architecture (`IFPredictor(two_part=True)`)

### Encoder — `RadiusGNN`, a stack of `WeightedRadiusConv` layers

```python
dims = [len(x_cols)] + [HIDDEN]*(NUM_LAYERS-1) + [embedding_dim]   # [9, 64, 64, 32] at the recommended config
convs = [WeightedRadiusConv(dims[i], dims[i+1]) for i in range(NUM_LAYERS)]
```
Each `WeightedRadiusConv` (a `MessagePassing` subclass, `aggr="add"`):
```
x'_j   = Linear(x_j)                    # Kaiming/He init
msg_ij = x'_j * edge_weight_ij           # edge_weight from §3 -- SUMS to a total, not an average, since normalize_data=False
z_i    = Σ_{j ∈ N(i) ∪ {i}} msg_ij       # includes i itself, via the self-loop
```
Between **non-final** layers: `BatchNorm1d → ReLU → Dropout(p=0.1)`. After the
**last** layer: no ReLU/dropout (a regression embedding shouldn't be forced
non-negative right before the head) — but because `normalize_data=False`, an
**extra `BatchNorm1d(embedding_dim)`** is appended specifically here, to repair the
Kaiming-init "roughly unit-variance input" assumption that unnormalized (summed)
aggregation would otherwise violate. This BatchNorm rescales per-**channel** scale
only, not cross-**node** relative magnitude — so a denser cell's embedding stays
consistently larger than a sparser cell's after it, just consistently *scaled*.
This is the concrete mechanism by which the implicit density signal from `§3`
actually reaches the head instead of being normalized away.

**Recommended defaults**: `NUM_LAYERS=3` (`recipe.NUM_LAYERS`), `HIDDEN=64`
(`recipe.HIDDEN`), `embedding_dim=32` (`IFPredictor`'s own default, unchanged by
`recipe`), `dropout=0.1` (default).

**Contrast with the legacy model**: the legacy GNN used `num_layers=1`, which is
architecturally a fully linear encoder (no `BatchNorm`/`ReLU` branch ever fires with
only one layer) and a strict 1-hop function of neighbors alone. At `NUM_LAYERS=3`,
this encoder has **two real nonlinearities** (the two inter-layer `ReLU`s) and a
3-hop receptive field — `radius=30 × 3 layers ≈ 90px` reach through the graph
itself, which is exactly the gap the multiscale pyramid features (`§2`) exist to
cover. Because of the self-loop, a cell's own raw value is already inside its own
*first-layer* embedding directly (not concatenated separately, as the legacy
model's `global_x_cols` pathway did) — at `NUM_LAYERS ≥ 2` it can additionally leak
back into its own embedding a second time through a mutual neighbor, the same
"no longer a strict self-exclusion guarantee past 1 hop" caveat the legacy doc
notes for its own multi-layer case.

### Head — `TwoPartHead(embedding_dim=32, num_outputs=1)`

Two linear heads off the **same** embedding `z`:
```python
logit     = Linear(z)                     # BCE target: y_positive
magnitude = sigmoid(Linear(z))             # in (0,1); MSE target: y, but ONLY on y_positive rows
combined  = sigmoid(logit) * magnitude     # the raw (uncalibrated) prediction -- see §7
```
Both linear layers use **Xavier/Glorot** init (a sigmoid, not a ReLU, follows each —
same convention as the legacy model's own output head). `magnitude`'s target lives
in `(0, ~1]` specifically *because* `y` came from `HurdleScaler` (`§4`), not the
general-purpose `ColumnScaler` — a `ColumnScaler`-scaled bright positive can sit 5–7+
standard deviations above the mean, which this sigmoid-bounded head could never
reach.

**Why a hurdle head at all**: these marker channels are heavily zero-inflated
(~99% of cells at background, ~1% real positives for `Sox17_mean`) — a single MSE
regression over a target shaped like that is dominated by getting the majority
background right, with little gradient left to learn the sparse positive
population. Splitting "is this cell positive" from "how positive" fixes that.

### The graph-free control — `MLPPredictor`

Same `TwoPartHead`, but `encoder_mlp` ignores `edge_index`/`edge_weight` entirely —
same forward/`forward_two_part` signature, so it drops into the same training code
unchanged. This is the ablation for "does message passing add anything once each
node already carries a multiscale summary of its neighborhood via `§2`'s pyramid
features?" Measured on this data: GNN R² 0.184 vs. MLP R² 0.148 on identical
features — message passing is worth roughly a quarter of the achievable signal on
top of the pyramid features alone, not the whole story by itself.

## 6. Training (`models.recipe.fit` → `models.train.train_calibrated`)

Full-batch, transductive — every cell (train/val/test/border-excluded) always
participates in message passing; `train_mask`/`val_mask` only gate which cells'
rows contribute to the loss and to early stopping (same convention as the legacy
model).

**Optimizer**: `AdamW`, `lr=1e-2` (`recipe.LR`) — **100× the legacy default**
(`1e-3`/`1e-4`). `recipe.py` documents measuring that the original notebook's
`lr=1e-4` leaves validation R² still climbing after 1500 epochs; `weight_decay=1e-5`,
plus `ReduceLROnPlateau(mode='max', factor=0.5, patience=max(20, patience//4))`
scheduled on validation R².

**Loss** — three terms, computed from one shared forward pass every step:
```
pos_weight   = clamp(n_neg / n_pos, min=1)   per y_col, from TRAIN labels
log_w        = mean(log(pos_weight))
bce          = BCEWithLogits(logit, y_positive, pos_weight)
mag_loss     = MSE(magnitude[y_positive > 0.5], y[y_positive > 0.5])
combined_cal = sigmoid(logit - log_w) * magnitude            # the CALIBRATED output, see §7
loss         = bce + mag_loss + LAMBDA_COMBINED * MSE(combined_cal, y)
```
`LAMBDA_COMBINED = 10.0` (`recipe.LAMBDA_COMBINED`) — measured R² 0.184 at
`lambda_combined=10` vs. 0.168 at 1, vs. 0.101 for a plain single-output MSE head:
the hurdle structure is worth keeping, but only once it's actually optimized for
the quantity it gets scored on, not just the raw BCE+MSE sum.

**Early stopping**: on **validation R² of the calibrated combined output**, not the
raw loss — once the classifier is reweighted, raw loss and the scored quantity are
no longer monotonically related, so raw loss stops being a usable stopping signal.
`patience=400` (`recipe.PATIENCE`), best-val-R² weights restored at the end.
`epochs` up to `2500` (`recipe.EPOCHS`) — `QUICK=True` in the notebooks drops this
to 150 for a fast, explicitly provisional pass through the logic.

## 7. The calibration fix (`models.train.calibrated_predict`)

`auto_pos_weight` rebalances BCE toward the minority positive class — necessary to
learn anything about a ~1%-positive target at all, but it also means the trained
logit converges to
```
sigmoid(z) = w·p / (w·p + 1 - p),     not p
```
i.e. calibrated to a **re-balanced** prior, not the real one — so
`combined = sigmoid(z) · magnitude` is inflated by roughly `pos_weight`. Measured on
`W8_pattern1` `Sox17_mean`: true positive rate 0.165, mean raw predicted
probability 0.479 (`pos_weight ≈ 5.2`). Subtracting `log(pos_weight)` from an
**already-trained** model's logit, no retraining, moved R² from −0.55 to +0.08.

The correction `p = sigmoid(z − log_w)` is exact and monotone **in the logit**, so
the classifier's own ranking is essentially untouched (AUROC moves <0.01 in
practice) — the reported quantity is `prob × magnitude`, and rescaling `prob` alone
can reorder that product slightly, which is the whole reason AUROC moves at all. A
large R² swing alongside a negligible AUROC swing is the signature of a pure
calibration problem, not a ranking one.

`log_w` is a **scalar produced during training** (loss term 3 in `§6` already
trains against the calibrated output) — it is not a data-shape or model-architecture
fact, so it is **not** part of `save_model`'s payload (`§9`) and must be saved
separately alongside the checkpoint, then reused at inference via
`calibrated_predict(model, data, log_w)`.

## 8. Evaluation protocol (`tools.evaluation`)

- **`random_split` vs. `spatial_block_split`** — report both. `random_split` holds
  out individual cells (their physical neighbors stay in train, so a purely local
  smoother scores well); `spatial_block_split` holds out whole 800px tiles, which is
  what "generalizes to an unseen region" actually requires. The gap between the two
  is itself a measurement of how much a score depends on spatial autocorrelation —
  on this data, `normalize_data=False` (`§3`) additionally lets the graph read local
  density off its own aggregation regardless of `x_cols`, so the gap matters even
  more here than it would for a plain averaging GNN.
- **`oracle_ceiling`** — predicts each cell from its neighbors' *true* values,
  leave-one-out; the honest upper bound for any spatial model. Every R²/AUROC in
  this project is read as a fraction of this, not of 1.0 (on this data: Sox17
  ceiling R² 0.25, T ceiling R² 0.44).
- **`evaluate(y_true, y_pred, y_positive)`** returns `r2`, `pearson`, `spearman`
  always; `auroc`, `ap`, `base_rate` if `y_positive` is given and not degenerate;
  `r2_positives` if the slice has >10 positive rows.

## 9. Checkpointing & cross-image application (`models.checkpoint`)

`save_model` persists `state_dict()` (weights only), `model_kwargs` (exact
constructor args, to rebuild the architecture before loading weights), the
**fitted** `x_scaler`/`y_scaler` (reused as-is, never refit), `x_cols`/`y_cols`, and
the graph's `radius`/`min_neighbors`/`length_scale`/`density_radius`/
`normalize_data`.

`apply_prediction_data(df, checkpoint)` rebuilds a `predict_df`-ready data dict for
a **new** `df` — a different well/image — using the checkpoint's own graph radius
and its saved (not refit) scalers. This is what lets a model trained on one well be
scored on a different well's cells without that well's own statistics leaking into
the transform, i.e. what makes a cross-image score mean anything (see notebook 05's
cross-predict section, or the legacy `cross_predict.ipynb`/`combined.ipynb`
equivalent).

`normalize_data` must match between the training-time data pipeline, the model's
own constructor arg, and whatever `apply_prediction_data` uses to rebuild the eval
graph — it changes what the saved `edge_weight` actually *means* (weighted-average
vs. weighted-sum adjacency), not just its scale. The recommended config uses
`normalize_data=False` throughout, everywhere.

`log_w` (`§7`) is not part of this payload — save it separately (this project's
convention: a sibling `..._calibration.pt` file next to the main checkpoint) and
apply it via `calibrated_predict`, not `predict_df` (which has no calibration
concept at all).

## 10. Reusing checkpoints across notebooks (`models.recipe.load_or_fit`)

Notebooks 03, 04, 05 (at its default `TRAIN_SOURCE`), and 06 all fit some or all of
the identical `BMP4 mask + context` config on the identical well — run them in
sequence and that one ~2500-epoch fit would otherwise happen 3-4 times. `load_or_fit`
is `fit` (`§6`) with a checkpoint cache in front of it:

```python
model, data, log_w, history = recipe.load_or_fit(
    path, df, x_cols, y_col, train_mask, val_mask, train_idx, if_cols,
    custom_background=..., intensity_cols=..., **train_kw)
```

- **Cache HIT** (`path` and its `_calibration.pt` sibling both already exist):
  reloads the model (`load_model`) and rebuilds `data` by re-applying the
  checkpoint's OWN fitted scalers to `df` (`apply_prediction_data`, `§9`) — the same
  mechanism a genuine cross-well evaluation uses, just pointed back at the well it
  was trained on. `y_positive` isn't part of `apply_prediction_data`'s return (it
  has no two-part concept), so it's rebuilt directly from the checkpoint's own
  `HurdleScaler.threshold_` — the exact threshold it was originally defined from.
  **`history` comes back `None`** — no training happened, so there's no
  training-curve history to plot; every notebook that plots training curves checks
  for this and prints a note instead.
- **Cache MISS**: trains via `fit`, then saves the checkpoint + a `log_w`-only
  `_calibration.pt` sibling before returning — so whichever notebook runs next
  against the same `path` gets a hit.

**Path convention** (all four notebooks): `{RESULTS}/models/
{well}_{feature_slug}_{marker}_{split}{cache_suffix}_gnn.pt` — e.g.
`W8_pattern1_Sox17_Brachyury_mask_Sox17_blocked_gnn.pt`. `feature_slug` is `mask`
(03/05's default, and 06's first arm — same slug everywhere, deliberately, so they
address the same file), `density`/`shuffled` (04's controls), or
`intensity`/`maskintensity` (06's other two arms).

**`cache_suffix`**: `'_quick' if QUICK else ''`. The cache is keyed purely on file
path, with no notion of how many epochs produced it — without this, a `QUICK=True`
(150-epoch) checkpoint could be silently reused as if it were the real, fully
converged one the moment a later notebook runs with `QUICK=False`. Namespacing the
path on `QUICK` keeps a provisional run and the real run from ever colliding, in
either direction.

Deleting a checkpoint (and its `_calibration.pt` sibling) forces a refit next time
its path is requested — there's no other invalidation mechanism, since the cache
has no way to know if `df`, `x_cols`, or any other argument changed since it was
written.

## Summary table — the exact recommended configuration (`models.recipe`)

| | value | source |
|---|---|---|
| input mask | one binary channel (e.g. `BMP4_bin`) | `tools.dataset.load_well` |
| x_cols | mask + 5 pyramid scales (σ ≤ 240px) + 2 geometry cols = 8 total | `recipe.mask_features` |
| y_col | one marker at a time (`Sox17_mean` / `T_mean`) | |
| graph radius | 30px | `recipe.RADIUS` |
| encoder layers | 3 | `recipe.NUM_LAYERS` |
| hidden width | 64 | `recipe.HIDDEN` |
| embedding width | 32 | `IFPredictor` default |
| aggregation | fixed `exp(-dist/30)` decay, **unnormalized** (weighted sum) | `normalize_data=False` |
| head | two-part hurdle (classifier + magnitude) | `two_part=True` |
| x scaling | `StandardScaler` only for continuous cols; binary flag untouched | `recipe.build_data` |
| y scaling | `HurdleScaler`, threshold = eye-verified `IF_BACKGROUNDS` | `tools.dataset.IF_BACKGROUNDS` |
| optimizer | AdamW, lr=1e-2, weight_decay=1e-5, `ReduceLROnPlateau` | `recipe.LR` / `train_calibrated` |
| loss | BCE + MSE(positives) + 10 × MSE(calibrated combined) | `recipe.LAMBDA_COMBINED` |
| early stopping | on val R² of the calibrated output, patience 400 | `recipe.PATIENCE` |
| epoch budget | 2500 (150 under `QUICK=True`, provisional) | `recipe.EPOCHS` |
| calibration | `log_w = mean(log(pos_weight))`, subtracted from the logit at inference | `train_calibrated` / `calibrated_predict` |

## Where this differs from the legacy model

| | legacy (`gnn_gat_methods.md`) | recommended (this doc) |
|---|---|---|
| encoder depth | 1 layer — fully linear, no nonlinearity | 3 layers — 2 `ReLU`s, real nonlinearity |
| hop count | strict 1-hop (self value never reachable) | 3-hop (self can re-enter via a mutual neighbor past hop 1) |
| density signal | explicit `local_density` feature + row-normalized (averaged) aggregation | implicit, via unnormalized (summed) aggregation + an extra `BatchNorm1d` that preserves cross-node magnitude |
| context beyond the graph | none — receptive field = graph hops only | explicit multiscale mask pyramid (up to σ=240px), layered on top of the graph's own ~90px reach |
| own-value pathway | separate `global_x_cols`, concatenated at the head | none — reaches the head only via the graph's self-loop |
| head | plain linear regression | two-part hurdle (classifier + magnitude), matched to a zero-inflated target |
| calibration | none | explicit prior-correction (`log_w`), baked into both the loss and inference |
| optimizer | Adam, lr 1e-3, patience 20 | AdamW, lr 1e-2, patience 400, on a 2500-epoch budget |
| evaluation | R² in scaled units, informal | R² against the oracle ceiling (`§8`), random *and* blocked splits reported together |
| checkpoint reuse | none — every notebook/run retrains from scratch | `recipe.load_or_fit` (`§10`) — a shared checkpoint cache across notebooks 03-06, keyed by well/feature-set/marker/split |
| model input | thresholded mask only | mask (default) or the raw continuous channel (`§2.1`, notebook 06) |

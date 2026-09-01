# Neighbor → IF prediction: GNN/GAT methods

Reference for `src/models/gnn.py` and `src/models/gat.py`, as actually used by
`features.ipynb` / `sender_predict.ipynb` / `combined.ipynb`. Every number below is
read directly from the code (line references included) — this is not a summary from
memory, it's what the code does, including the effects of the default arguments that
every training run in this project actually uses.

## 1. Task

Predict a cell's own immunofluorescence (IF) level for one or more marker channels
(`y_cols`, e.g. `FoxA2_mean`) from two disjoint sources of information:

- **its own state** for a separate set of channels (`global_x_cols`, e.g.
  `Activin_mean`, `local_density`) — seen directly, no message passing
- **its spatial neighbors'** values for a set of channels (`neighbor_x_cols`,
  defaults to all IF channels) — seen ONLY via an aggregate of other cells, the
  cell's own value for these is never part of its own prediction path

`y_cols` defaults to every IF channel *not already in* `global_x_cols` — predicting a
channel the model already sees for itself directly would be trivial leakage.

## 2. Graph construction

### 2.1 Radius graph, not k-NN (`tools.morphology.radius_edge_index`)

Every pair of cells within `radius` pixels of each other is connected — the number of
edges per node is *not fixed*, so node degree carries a real density signal (a fixed
`k` graph would erase this).

Exact algorithm:
1. `sklearn.neighbors.NearestNeighbors(radius=radius).radius_neighbors(centroids)` —
   for every cell, find all other cells within `radius` px (this call includes the
   point itself at distance 0; that self-match is explicitly dropped: `idx = idx[idx
   != i]`).
2. **`min_neighbors` fallback** (default `1`): if a cell ends up with fewer than
   `min_neighbors` real neighbors within `radius` (an isolated cell in a sparse
   patch), it's instead connected to its `min_neighbors` *nearest* neighbors
   regardless of distance, via a separate `NearestNeighbors(n_neighbors=min_neighbors
   + 1)` k-NN query (the `+1`/`[1:]` drops the self-match, which k-NN queries always
   return as the closest point). This guarantees no node is ever left with zero
   edges.
3. Both directions of every pair are added to a `set` (`pairs.add((i,j))`,
   `pairs.add((j,i))`) — the resulting `edge_index` is symmetric by construction
   (Euclidean distance is symmetric), no separate "make mutual/undirected" step
   needed.
4. `edge_dist` = Euclidean distance between the two centroids, aligned 1:1 with
   `edge_index` columns.

Returns `edge_index: (2, E) int64`, `edge_dist: (E,) float32`.

### 2.2 Edge weights — GNN: fixed distance decay (`normalize_distance_weights`)

```
w_ij  = exp(-dist_ij / length_scale)            # length_scale defaults to radius itself
edge_weight_ij = w_ij / sum_k(w_kj)              # normalized over ALL incoming edges to j
```

i.e. row-normalized (per **destination** node) so every node's incoming edge weights
sum to exactly 1 — this is what turns "a decaying function of distance" into an
actual weighted-average adjacency, the spatial-graph analog of a GCN's degree
normalization. Implemented via `scatter_add_` over destination indices, denominator
clamped at `1e-8` to avoid divide-by-zero.

This `edge_weight` tensor is **fixed** — computed once from geometry, never updated
during training, used identically by both the GNN (as the literal aggregation weight)
and the GAT (as an *input feature* to the learned attention, see §4).

**`normalize_weights=False`** (`build_radius_graph`/`build_prediction_data`/
`build_multi_image_prediction_data`, all default `True`): skips the row-normalization
step, leaving the raw per-edge `exp(-dist/length_scale)` weight as-is. Since
`WeightedNeighborConv` aggregates with `aggr="add"`, a normalized weight (sums to 1)
makes that aggregation a weighted AVERAGE — structurally insensitive to neighbor
COUNT, which is exactly why `local_density` needed to be a separate explicit feature
in the first place (mean/attention aggregators normalize count information away, see
§2.3). Without normalization, aggregation becomes a weighted SUM
(GraphSAGE/GIN-style): a node with more/closer neighbors accumulates a strictly larger
total signal, so density becomes implicitly recoverable from the embedding's
magnitude, without `local_density` as an input at all. Only meaningfully changes GNN
behavior — GAT's attention is softmax-normalized by `GATConv` regardless of this flag
(that normalization is intrinsic to how attention scores are computed, not something
this edge weight controls for GAT); toggling it only changes the edge *feature* GAT
conditions its still-sum-to-1 attention on.

Practical caveat observed when verifying this: unnormalized aggregation can produce
much larger-magnitude embeddings early in training (loss starting orders of magnitude
higher than the normalized case, in one real test: ~680 vs. ~1.5), since Kaiming init
(§7) assumes roughly standardized input — training still converges (Adam adapts), but
expect a noisier/slower start than the normalized default.

**Checkpointing**: `normalize_weights` is saved in the checkpoint (`save_model`) and
restored by `apply_prediction_data` (`checkpoint.get("normalize_weights", True)`) —
it must match between training and evaluation, since it changes what the saved
`edge_weight` actually represents (weighted-average vs. weighted-sum adjacency), not
just its scale.

### 2.3 `local_density` — a separate, explicit density feature

```
local_density(cell) = (# other cells within DENSITY_RADIUS px) 
```

Computed via `NearestNeighbors(radius=DENSITY_RADIUS).radius_neighbors(...)`, minus 1
for the self-match. This exists *in addition to* graph degree because mean/attention
aggregation (§3, §4) tends to normalize raw neighbor **count** away — two
neighborhoods with similar per-cell marker values but very different density can
produce a similar aggregated message, so density needs to be an explicit input if the
model is meant to use it.

**`DENSITY_RADIUS` is a separate parameter from the graph's `radius` (`PRED_RADIUS`)**
— they do not have to match, and in the checkpoints trained so far in this project
they usually don't (e.g. `PRED_RADIUS=200`, `DENSITY_RADIUS=50`). This is saved
separately in the checkpoint (`density_radius` field, see §8) precisely so a later
`cross_predict.ipynb`/`combined.ipynb` run recomputes it correctly rather than
assuming it equals the graph radius.

### 2.4 Border exclusion (`border_mask`)

```
is_border(cell) = (y < R) | (y > H-R) | (x < R) | (x > W-R)     # R = max(PRED_RADIUS, DENSITY_RADIUS)
```

A cell within `R` px of the image edge has an undercounted neighborhood (part of its
disk was never imaged, not empty) — such cells are excluded from ever being a
train/val/test **target**, but they are NOT removed from the graph: they still act as
a neighbor/density contributor for cells further from the edge. `R` uses the max of
the graph radius and density radius since both features can be undercounted near an
edge.

## 3. Feature scaling (`ColumnScaler`)

Every column (in `global_x_cols`, `neighbor_x_cols`, and `y_cols` — each group gets
its own independently-fit `ColumnScaler` instance) goes through the same 3-step
pipeline, fit on `train_idx` rows only (never val/test, to avoid leakage) and then
applied to every row:

**Step 1 — background subtraction (per-column, optional, `subtract_background`,
default on).**
```
background_c = median(negative population)     # via get_bimodal_threshold, see below
x = clip(x - background_c, min=0)
```
Raw IF intensity is never truly zero even for a true-negative cell (camera offset,
autofluorescence, non-specific binding all add a shared baseline) — subtracting it
means the transform in step 2 sees actual signal-above-background starting from zero.

`background_c` is estimated by `tools.qc.get_bimodal_threshold`: log10-transform the
positive values, fit a Gaussian KDE, find local minima/maxima of the density curve,
and take the valley between the two tallest peaks (or the single lowest valley
overall as a fallback) as the negative/positive split point; `background_c` is the
**median of the population below that threshold**. If no bimodal split is found,
falls back to the median of the whole column.

**Step 2 — variance-stabilizing transform (`scaling` mode, per-column, optional via
`apply_transform`):**

| mode | formula | when it's appropriate |
|---|---|---|
| `log1p_standard` (default) | `log1p(x)` | roughly log-normal intensity (antibody binding/amplification is multiplicative) |
| `arcsinh` | `arcsinh(x / arcsinh_cofactor)` | same idea, but stays linear near zero — background-level cells don't get compressed the way log does; standard in flow/mass cytometry |
| `robust` | none (identity) | `RobustScaler` (step 3) alone handles centering/scaling |

**Step 3 — `StandardScaler`** (or `RobustScaler` if `scaling='robust'`), fit on the
transformed train rows: `(x - mean) / std` (or median/IQR for `RobustScaler`).

**Per-column overrides** (`no_background_cols`, `no_transform_cols`): both step 1 and
step 2 accept a per-column boolean mask instead of one shared flag for the whole
group — needed because `local_density` is fit in the *same* `ColumnScaler` call as
`Activin_mean` (both are in `global_x_cols`) but needs different treatment: it's a
bounded count with no real "background" concept and a roughly symmetric (sometimes
even left-skewed) distribution, so background-subtracting it floors a meaningful
fraction of cells to an identical value, and log1p-transforming it makes its skew
*worse*, not better (measured: raw skew ≈ 0.28 at `DENSITY_RADIUS=50`, log1p skew ≈
−0.53). Both are disabled for `local_density` in every notebook that uses it, leaving
it as identity → `StandardScaler` only.

`inverse_transform` reverses all three steps in order (`expm1`/`sinh` then `+
background_c`) — this is what lets `predict_df` report R²/MAE in original
fluorescence units instead of only the standardized space the model actually trains
in (see §6, and note **R² is NOT invariant under this nonlinear transform** — the
same predictions can show very different R² in scaled vs. original units).

## 4. GNN encoder (`WeightedNeighborConv` / `NeighborRadiusGNN`)

**One layer** (`WeightedNeighborConv`, a `torch_geometric.nn.MessagePassing` subclass,
`aggr="add"`):
```
x'_j  = W @ x_j + b                    # nn.Linear(in_channels, out_channels), Kaiming/He init
msg_ij = x'_j * edge_weight_ij          # edge_weight already sums to 1 over incoming edges to i
z_i    = sum_{j in N(i)} msg_ij         # = a weighted AVERAGE, since weights sum to 1
```
Node `i` never receives a message from itself — `edge_index` contains no `(i,i)`
pairs by construction (§2.1).

**Stacking** (`NeighborRadiusGNN`):
```python
dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
convs = [WeightedNeighborConv(dims[i], dims[i+1]) for i in range(num_layers)]
```
Between non-final layers only: `BatchNorm1d → ReLU → Dropout(p=dropout)`. **No
activation after the last layer** (the encoder's output is the raw embedding `z`).

### Exact dimensions at the defaults every notebook in this project actually uses

`hidden_channels=64`, `embedding_dim=32`, **`num_layers=1`**, `dropout=0.1`.

At `num_layers=1`: `[hidden_channels] * (1-1) = []`, so `dims = [in_channels,
embedding_dim]` — there is exactly **one** `WeightedNeighborConv(in_channels, 32)`.
`hidden_channels=64` is never instantiated. The "between non-final layers" branch
(`if i < len(convs)-1`) is never true with one layer, so **no BatchNorm/ReLU/dropout
ever runs** — the encoder is a single linear projection + weighted sum, with **no
nonlinearity at all**.

### Hop count

At `num_layers=1` (the default, used everywhere so far), node `i`'s embedding is
**strictly a 1-hop function** of its direct neighbors' raw `neighbor_x_cols` values —
`i`'s own value is guaranteed never to appear in it.

If `num_layers` is ever increased: at 2 hops, node `i`'s embedding aggregates its
neighbors' *layer-1* embeddings — and since the graph is symmetric, any neighbor `j`
of `i` also has `i` as one of *its* neighbors, so `j`'s layer-1 embedding already
contains a contribution from `i`'s own raw value. This means `i`'s own value can leak
back into its own 2-hop embedding indirectly, through a mutual neighbor. Not a bug —
just no longer the strict "self value never touches its own prediction" guarantee
that holds at 1 layer.

## 5. GAT encoder (`NeighborRadiusGAT`, wraps `torch_geometric.nn.GATConv`)

Same graph, same `global_x_cols`/`neighbor_x_cols` split, same final head as the GNN
— the only thing that changes is how the per-edge aggregation weight is computed:
**learned attention** instead of the fixed `exp(-dist/length_scale)` formula.

Standard GAT attention mechanism (as implemented by `GATConv`), per head:
```
e_ij = LeakyReLU( a^T [ W·x_i ‖ W·x_j ‖ W_edge·edge_attr_ij ] )
α_ij = softmax_j(e_ij)            # normalized over i's incoming edges -- sums to 1, same property as edge_weight
z_i  = sum_j α_ij · (W·x_j)       # concat (non-final layers) or average (final layer) over heads
```
`edge_attr_ij` here **is the same fixed distance-based `edge_weight` from §2.2**,
reshaped to `(E, 1)` and passed in via `edge_dim=1` — so geometric distance still
directly informs the attention score, it's just no longer the *only* signal (the
node-feature terms `W·x_i`, `W·x_j` let the score also depend on what's actually in
the neighborhood, e.g. its marker values).

**`add_self_loops=False`** — PyG's `GATConv` normally inserts a virtual `(i,i)` edge
by default so a node partially attends to itself; this is explicitly disabled here so
GAT preserves the exact same "neighbor only, self excluded" guarantee as the GNN (and
so `edge_index` order stays unchanged, which the attention-extraction/visualization
notebook cells rely on via `torch.equal(att_edge_index, pred_data['edge_index'])`).

**Layer construction** (`NeighborRadiusGAT.__init__`):
```python
is_last     = (i == num_layers - 1)
layer_out   = out_channels if is_last else hidden_channels
layer_heads = 1 if is_last else heads
GATConv(in_dim, layer_out, heads=layer_heads, concat=(not is_last), edge_dim=1, add_self_loops=False, dropout=dropout)
```
Non-final layers **concatenate** all heads' outputs (`concat=True`) → width
`hidden_channels * heads`; the final layer **averages** heads (`concat=False`,
`heads=1` forced) → width exactly `out_channels`, regardless of `heads`, so the
output shape matches the GNN's `embedding_dim` exactly.

### Exact dimensions at the defaults actually used

`hidden_channels=64`, `embedding_dim=32`, **`num_layers=1`**, `heads=4`, `dropout=0.1`.

At `num_layers=1`, the single layer is automatically `is_last=True`, which forces
`layer_heads = 1` regardless of the `heads=4` constructor default. **So every GAT
model actually trained in this project has single-head attention** — the multi-head,
`concat=True`, 256-wide (`64×4`) intermediate representation the constructor
*supports* only materializes if `num_layers ≥ 2`. At the current default: one
`GATConv(neighbor_in_channels, 32, heads=1, concat=False, edge_dim=1,
add_self_loops=False)`, straight to a 32-dim embedding. Same as the GNN, the
"between non-final layers" branch never fires with one layer, so there is no
BatchNorm/ReLU/dropout applied by `NeighborRadiusGAT.forward` either — though
`GATConv`'s own internal `LeakyReLU` (in the attention-score computation) and softmax
are still nonlinear regardless of layer count.

### 5.1 Extracting attention at `num_layers > 1` (`NeighborRadiusGAT.attention_by_layer`)

The interpretability notebook cells (`features.ipynb`/`sender_predict.ipynb`/
`combined.ipynb`, "Understanding the GAT: attention weights") call `GATConv(...,
return_attention_weights=True)` directly to pull out per-edge attention for plotting
against distance/marker value. At `num_layers=1` this is simple — calling
`encoder.convs[0]` directly returns `alpha` of shape `(E, 1)` (heads forced to 1, see
§5), which squeezes cleanly to `(E,)` matching `edge_dist`.

**At `num_layers ≥ 2` this breaks**: `encoder.convs[0]` is no longer the final layer,
so it uses the real `heads` value (default 4) with `concat=True` — `alpha` comes back
`(E, heads)`, not `(E, 1)`, and squeezing does nothing (there's no size-1 dimension to
remove). Plotting that directly against `edge_dist` (`E,`) raises `x and y must be the
same size` in `matplotlib`.

More fundamentally, with `num_layers ≥ 2` there is no longer a single "the" attention
value per edge — each layer has its own attention, over different inputs (layer 0
attends over the raw `neighbor_x_cols`; layer 1 attends over layer 0's *output*
embeddings, which may already contain 2-hop-mixed information, see §4's leakage
note).

`NeighborRadiusGAT.attention_by_layer(x, edge_index, edge_attr)` handles both issues:
it replays the exact same per-layer computation as `forward` (same intermediate `x`
fed to each layer, same `ReLU`/`BatchNorm`/dropout between non-final layers, dropout
explicitly forced off via `training=False` so the captured attention is deterministic
and matches what a real inference call would use, regardless of the model's actual
`.training` state), while also capturing each layer's attention and head-averaging it
down to `(E,)` (a no-op for the already-single-head final layer). Returns a
`list[Tensor]`, one `(E,)` tensor per layer. Notebook cells plot one row per entry
(distance/marker correlation for every layer), then use `layer_attentions[-1]` (the
final layer — the one that most directly determines what reaches the prediction
head) for downstream single-cell/global-marker attention maps, so those cells didn't
need to change.

## 6. Shared prediction head (both models)

```python
z      = encoder(x_neighbor, edge_index, edge_weight)     # (N, 32) -- GNN or GAT, from §4/§5
z_full = concat([z, x_global], dim=1)                     # (N, 32 + global_in_channels)
pred   = Linear(32 + global_in_channels, num_outputs)(z_full)     # Xavier/Glorot init, no activation after
return pred, z
```

`global_in_channels=0` (i.e. `global_x_cols=[]`) is a valid, explicit no-op case —
concatenating a zero-width tensor changes nothing. `z` is returned alongside the
prediction (used for embedding/interpretability inspection in notebooks, not by the
training loop itself).

**Net effect at the defaults used everywhere in this project (`num_layers=1`):**
the GNN is architecturally a **fully linear model** end-to-end (linear encoder +
linear head, no nonlinearity anywhere) — just structured as two separate linear paths
(neighbor-aggregated, self/global) concatenated before one more linear combination.
GAT is the same shape, except its single encoder layer's attention *weights* are
computed through a nonlinear (`LeakyReLU` + `softmax`) scoring function, even though
the feature transformation itself is linear.

## 7. Weight initialization

- `WeightedNeighborConv.lin` / `GATConv`'s internal linear projections: **Kaiming/He
  normal** (`nn.init.kaiming_normal_(..., mode='fan_in', nonlinearity='relu')`),
  biases zero — appropriate because a ReLU follows in the multi-layer case (and is a
  reasonable default even at 1 layer, where none does).
- Final prediction head (`nn.Linear(embedding_dim + global_in_channels,
  num_outputs)`): **Xavier/Glorot normal**, biases zero — used specifically because
  *no* activation follows this layer (it's a regression output).

## 8. Training procedure (`train_neighbor_predictor`)

- **Full-batch, transductive**: every cell (train/val/test/border-excluded) always
  participates in message passing every step — `train_mask`/`val_mask` only gate
  which cells' rows contribute to the loss and to early stopping. This means val/test
  cells' *features* are visible as neighbors during training (standard transductive
  GNN setting), but their own *labels* are never used for gradient updates.
- **Loss**: `F.mse_loss(pred[train_mask], y[train_mask])` — MSE on the **scaled**
  targets (`data['y']`), not original units.
- **Optimizer**: Adam, `lr=1e-3` default, `weight_decay=1e-5`.
- **Early stopping**: tracks best validation loss seen so far; if a new epoch's
  val loss doesn't improve by more than `1e-6`, `bad_epochs` increments; training
  stops once `bad_epochs >= patience` (default `20`). The model's weights are
  **rolled back to the best validation checkpoint** at the end (`best_state`), not
  left at the final epoch's weights.
- Same function, same signature, used for both GNN and GAT (`forward(x_neighbor,
  edge_index, edge_weight, x_global)` is identical for both).

## 9. Multi-image training (`build_multi_image_prediction_data`, `combined.ipynb`)

Builds an **independent radius graph per image** (§2.1, run separately per image's
own centroids) and concatenates them into one combined graph via an index offset —
cells in different images/wells are never spatially adjacent, so are never connected
to each other; `edge_index` for image 2 has every node id shifted by
`len(image_1_cells)`. `local_density`/border exclusion (§2.3, §2.4) are still computed
**per image** before concatenation, since each image has its own neighborhoods.

The `ColumnScaler`s (§3) are fit **once, across every image's train rows together** —
so a single checkpoint's scaler applies uniformly no matter which image a cell came
from. This assumes the images are on a comparable scale to begin with (checked
explicitly in `combined.ipynb`'s per-image dynamic-range histogram cell).

## 10. Inference & checkpointing

- **`predict_df(model, data, mask, index, y_scaler=<AUTO>)`**: runs the model over
  `mask`'d cells, returns `(pred_df, true_df)`. Default (`y_scaler` = the checkpoint's
  fitted `y_scaler`) inverse-transforms back to **original units**; passing
  `y_scaler=None` explicitly skips that, returning both in the model's raw
  **standardized/scaled** space instead. These are NOT comparable R² numbers (§3) —
  always check which one a given notebook cell is actually computing.
- **`save_model`** persists: `state_dict()` (weights only, not the whole `nn.Module`),
  `model_kwargs` (exact constructor args, to rebuild the architecture before loading
  weights), the fitted `global_scaler`/`neighbor_scaler`/`y_scaler` objects
  (reused as-is on new data, never refit), `neighbor_x_cols`/`global_x_cols`/`y_cols`,
  the graph `radius`, `min_neighbors`, `length_scale`, and the separate
  `density_radius` (§2.3).
- **`apply_prediction_data(df, checkpoint)`**: rebuilds a `predict_df`-ready data dict
  for a NEW `df` (a different image), using the checkpoint's saved graph radius to
  build the graph and its saved (not refit) scalers to transform columns — this is
  what `cross_predict.ipynb`/`combined.ipynb`'s cross-predict section use to evaluate
  a trained checkpoint against an image it never saw during training.

## Summary table — exact defaults used by every training run in this project

| | GNN | GAT |
|---|---|---|
| encoder layers (`num_layers`) | 1 | 1 |
| embedding width (`embedding_dim`) | 32 | 32 |
| hidden width (`hidden_channels`) | 64 (unused at 1 layer) | 64 (unused at 1 layer) |
| attention heads (`heads`) | n/a | 4 constructor default, **forced to 1** at 1 layer |
| dropout | 0.1 (unused at 1 layer — no between-layer step fires) | same |
| aggregation weight | fixed `exp(-dist/length_scale)`, row-normalized | learned, `LeakyReLU`+`softmax`, distance fed in as `edge_dim=1` |
| self-loops | never (edge_index excludes `(i,i)` by construction) | explicitly disabled (`add_self_loops=False`) |
| hop count (at defaults) | strict 1-hop (self value never reachable) | strict 1-hop (self value never reachable) |
| loss | MSE on scaled `y` | MSE on scaled `y` |
| optimizer | Adam, lr 1e-3, weight_decay 1e-5 | same |
| early stopping | patience 20, min-delta 1e-6, best-val weights restored | same |

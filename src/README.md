# `src/` layout

```
models/      predicting a cell's IF from its spatial neighbourhood
  graph        the self-inclusive radius graph + its distance-weighted convolution
  scalers      ColumnScaler (general), HurdleScaler (zero-inflated targets)
  data         graph + scaled x/y  ->  one training-ready dict
  predictors   RadiusGNN / IFPredictor, TwoPartHead, MLPPredictor (graph-free control)
  gat          learned-attention counterpart to RadiusGNN
  train        training loops, incl. train_calibrated (the prior-correction fix)
  checkpoint   save/load a model together with its fitted scalers
  recipe       the recommended end-to-end configuration

tools/       data handling and analysis
  dataset      loading a well; the verified background thresholds
  spatial      multi-scale neighbourhood features (mask pyramid, density, geometry)
  evaluation   random/blocked splits, hurdle targets, metrics, the oracle ceiling
  features     per-cell feature extraction from images
  qc           segmentation/stitching QC, bimodal thresholding
  morphology   graph utilities, older model classes, legacy plotting
  restitch_fovs, simulated

plotting/    figures
  diagnostics  before modelling: scale curves, receptive-field check, response curves
  fields       cell maps, smoothed field grids, input/measured/predicted panels
  evaluation   training curves, pred-vs-actual, calibration, metric comparisons
```

## Three things worth knowing before using any of this

**1. Read scores against the oracle ceiling, not against 1.0.**
`tools.evaluation.oracle_ceiling` predicts each cell from its neighbours' *true*
values, leave-one-out — an upper bound for any spatial model. On this data it is
R² 0.25 for Sox17 and 0.44 for T; the rest is single-cell noise. An R² of 0.33 for T
is three quarters of everything achievable, not a third.

**2. A `two_part` model trained with `auto_pos_weight` needs its prior corrected.**
The classifier converges to a probability calibrated to a *re-balanced* prior, so
`combined = prob × magnitude` is inflated by roughly `pos_weight`, and R² can go
strongly negative while AUROC barely moves (<0.01 — the correction is monotone in the
logit, though rescaling `prob` alone can reorder `prob × magnitude` slightly). Use
`models.train.train_calibrated` and `calibrated_predict`, and save the returned
`log_w` alongside the checkpoint.

**3. Report the blocked split, not just the random one.**
`tools.evaluation.spatial_block_split` holds out whole regions. On this data the
random split cannot distinguish the real BMP4 mask from a *shuffled* one, because the
graph's weighted-sum aggregation (`normalize_data=False`) supplies cell density on its
own and spatial autocorrelation does the rest. The blocked split separates them
cleanly, so it is the protocol of record here rather than a stricter afterthought.

## Removed

`models/cnn.py` and `models/gnn.py` are gone — `gnn.py`'s contents were split across
`graph`/`scalers`/`data`/`predictors`/`train`/`checkpoint`. Notebooks under
`notebooks/gnn_v1/`, `notebooks/morphology/` and `notebooks/local_density/` import the
old flat `from models.gnn import ...` API and will need their imports updated (several
of them already referenced symbols such as `NeighborIFPredictor` that no longer
existed).

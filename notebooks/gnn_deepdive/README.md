# Notebooks

Run in order — 01 saves the oracle ceilings that 03, 04 and (optionally) 05 read.

| notebook | question | runtime |
|---|---|---|
| `01_predictive_power.ipynb` | Is there anything to predict, at what spatial scale, and how much is reachable at all? | ~2 min |
| `02_predict_density.ipynb` | Does the BMP4 mask predict **cell density**? (the confound) | ~3 min |
| `03_predict_from_mask.ipynb` | Sox17 / T from the BMP4 mask | ~2 h |
| `04_predict_from_density.ipynb` | The controls: density-only and shuffled-mask | ~3 h |
| `05_train_and_cross_predict.ipynb` | Train 03's recommended model on any well/path, analyze it, cross-predict onto other images | ~2 h |
| `06_predict_from_intensity.ipynb` | Does the continuous BMP4 *intensity* beat the thresholded mask as model input? | ~4 h |

**05 is the reusable one.** It takes 03's recommended config (no ablation) and
parameterizes it: edit `TRAIN_SOURCE`/`EVAL_SOURCES` at the top of the config cell to
a well name or any `*_features.parquet` path, and everything downstream — training,
the feature/correlation/sensitivity analysis from `gnn_v1/mask_predict.ipynb`, and the
cross-image evaluation from `gnn_v1/cross_predict.ipynb` (now against the current
`models`/`tools` API) — runs unchanged. 01–04 answer *why* this config is the right
one; 05 is for *applying* it to new data.

**06 compares input feature sets, not architectures.** Every other notebook feeds
the model the thresholded `BMP4_bin` mask; 06 adds a multiscale local-mean-intensity
pyramid of the raw, continuous `BMP4_mean` channel (`tools.spatial.
add_intensity_pyramid`, `models.recipe.intensity_features`) and fits `mask`,
`intensity`, and `mask + intensity` side by side under the identical recipe, splits,
and held-out-well transfer check 03 uses. Unlike 01–04, it does not assert a
conclusion — read the numbers off its own results tables for your data.

**Kernel.** Select *Python (latte · torch_geometric)*. The default `python3` and
`mocha` kernels do not have `torch_geometric` and 02–06 will fail on them.

**`QUICK = False`** at the top of 03, 04, 05 and 06 is the real setting. `QUICK = True`
drops to 150 epochs for a fast pass through the logic — about a tenth of what these
models need, so the numbers it produces are provisional.

**Regenerating.** All six are generated from `_build.py`; edit there and re-run
`python _build.py` rather than editing the `.ipynb` files, or your changes will be
overwritten.

`gnn_v1/`, `morphology/`, `local_density/` and `analyses/` are the earlier work. They
import the pre-refactor `from models.gnn import ...` API and need their imports updated
before they will run — see `src/README.md`.

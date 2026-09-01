"""Generates the analysis notebooks. Edit here, re-run, notebooks regenerate."""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

SETUP = """import sys, os
sys.path.append('../src')

%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tools.dataset import load_well, marker_positive, PRIMARY_WELL, MATCHED_PAIR, RESULTS
from tools.spatial import PYRAMID_SCALES
import tools.evaluation as ev
import plotting as pl

pd.set_option('display.width', 160)
plt.rcParams['figure.dpi'] = 110"""


def nb(cells):
    return {
        "cells": [
            {"cell_type": "markdown" if k == "md" else "code", "metadata": {},
             "source": v.splitlines(keepends=True),
             **({} if k == "md" else {"execution_count": None, "outputs": []})}
            for k, v in cells
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python (latte \u00b7 torch_geometric)", "language": "python", "name": "latte"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


# ==========================================================================
# 01 -- predictive power
# ==========================================================================
N1 = [
    ("md", """# 01 · How much predictive power is actually there?

Before building anything, four questions, all answerable without a model:

1. **What does the illumination pattern look like, and what do the readouts do around it?**
2. **At what spatial scale** does the mask carry information — i.e. how far does a
   model need to be able to see?
3. **How much of the readout is predictable from space at all?** Much of the
   cell-to-cell variation is single-cell noise that no spatial model can reach.
4. **What is the density confound?** If illumination also changes cell density,
   then "predicted from BMP4" and "predicted from colony structure" are entangled.

The answers set every design choice in notebooks 02–04, so this notebook is the one
to re-run first if the data changes."""),

    ("code", SETUP),

    ("md", """## Load

The binary `BMP4_bin` mask is the model input throughout — not the raw intensity.
Positive/negative calls use the thresholds verified by eye (see `tools.dataset`).

Note the field is loaded **uncropped**. The crop previously in use kept only the
illuminated ring and discarded the second pattern (a filled square at x ≈ 4700–5250),
i.e. half the experiment."""),

    ("code", """df = load_well(PRIMARY_WELL)

df['Sox17_pos'] = marker_positive(df, 'Sox17')
df['T_pos'] = marker_positive(df, 'T')

centroids = df[['centroid_y', 'centroid_x']].to_numpy(np.float32)
print(f'{len(df):,} cells   field {centroids[:,0].max():.0f} x {centroids[:,1].max():.0f} px')
print(f"BMP4+ : {df['BMP4_bin'].mean():.1%}")
print(f"Sox17+: {df['Sox17_pos'].mean():.1%}   (threshold {IF_BACKGROUNDS['Sox17_mean']:.1f})")
print(f"T+    : {df['T_pos'].mean():.1%}   (threshold {IF_BACKGROUNDS['T_mean']:.1f})")"""),

    ("md", "## 1 · What does it look like?"),

    ("code", """fig = pl.plot_cell_maps(df, [
    ('BMP4+ (the model input)', 'BMP4_bin', 'gray_r'),
    ('Sox17 (log)', np.log1p(df['Sox17_mean']), 'magma'),
    ('T (log)', np.log1p(df['T_mean']), 'magma'),
    ('Hoechst (log)', np.log1p(df['Hoechst_mean']), 'viridis'),
], suptitle='per-cell maps')
plt.show()"""),

    ("md", """At ~100k cells the per-cell scatter is too dense to read. The smoothed fields
below are the single most useful picture in this notebook — and the **cell density**
row is included deliberately, because a marker field and the density field are very
easy to confuse by eye."""),

    ("code", """fig = pl.plot_field_grid(df, rows=[
    ('BMP4+', 'BMP4_bin'),
    ('Sox17+', 'Sox17_pos'),
    ('T+', 'T_pos'),
    ('cell density', None),
], sigmas=[30, 60, 120, 240])
plt.show()"""),

    ("md", """**What to look for.** The mask is an illuminated ring (radius ≈ 450px) plus
scattered isolated positives. T is strongly suppressed in a disc that fills *and
slightly exceeds* the ring. Sox17 shows the same suppression far more weakly, and is
dominated by patchy colony-scale structure unrelated to the illumination.

Crucially: **cell density is also low inside the ring**. Keep that in view — notebook
02 quantifies it and notebook 04 turns it into a control."""),

    ("md", "## 2 · Is there a spatial relationship at all?\\n\\nThe most generous possible test: correlate the *smoothed fields*, so per-cell noise is averaged away. A near-zero value here means there is genuinely nothing to find, not that a model was too weak."),

    ("code", """from tools.spatial import smoothed_field

shape = (int(centroids[:,0].max())+1, int(centroids[:,1].max())+1)
rows = []
for sig in [30, 60, 120, 240, 480]:
    fields = {n: smoothed_field(centroids, df[c].to_numpy(np.float32), sig, shape_px=shape)
              for n, c in [('BMP4', 'BMP4_bin'), ('Sox17', 'Sox17_pos'), ('T', 'T_pos')]}
    dens = smoothed_field(centroids, np.ones(len(df), np.float32), sig, shape_px=shape) * 0
    from tools.spatial import rasterize
    from scipy.ndimage import gaussian_filter
    g, _, _ = rasterize(centroids, np.ones(len(df), np.float32), shape_px=shape)
    fields['density'] = gaussian_filter(g, sigma=sig/8.0, mode='nearest')
    for a, b in [('BMP4', 'Sox17'), ('BMP4', 'T'), ('density', 'Sox17'), ('density', 'T')]:
        m = np.isfinite(fields[a]) & np.isfinite(fields[b])
        rows.append(dict(sigma=sig, pair=f'{a} ~ {b}', r=float(np.corrcoef(fields[a][m], fields[b][m])[0,1])))

field_corr = pd.DataFrame(rows)
pl.plot_field_correlation(rows)
plt.show()
field_corr.pivot_table(index='sigma', columns='pair', values='r').round(3)"""),

    ("md", """`BMP4 ~ T` reaches about **−0.76**: strong and negative, i.e. T is suppressed where
BMP4 is on. `BMP4 ~ Sox17` peaks around **−0.25** only.

But `density ~ T` reaches **+0.69** — nearly as strong as the BMP4 relationship, and
with the opposite sign. That is the confound, visible before any model exists."""),

    ("md", "## 3 · At what scale — and can the model even see that far?"),

    ("code", """from tools.spatial import add_mask_pyramid

mask_cols = add_mask_pyramid(df, 'BMP4_bin', geometry=True)
print('added:', mask_cols)

pl.plot_scale_auroc(df, 'BMP4_bin_gauss_', PYRAMID_SCALES,
                    targets=[('Sox17+', df['Sox17_pos']), ('T+', df['T_pos'])])
plt.show()"""),

    ("md", """The scale at which each curve departs from 0.5 is the length scale of the biology,
and it is what the graph radius has to be able to reach.

Now the blunt version of the same question — a cell with **no** BMP4+ cell anywhere
in its receptive field has a literally constant input, so nothing downstream can
distinguish it from any other such cell."""),

    ("code", """pl.plot_receptive_field_check(df, 'BMP4_bin', radii=[30, 60, 90, 120, 240, 480, 960])
plt.axvline(90, color='r', ls='--', lw=1)
plt.gca().annotate('radius=30px x 3 layers', xy=(90, 0.5), xytext=(120, 0.62),
                   arrowprops=dict(arrowstyle='->', color='r'), color='r', fontsize=9)
plt.show()"""),

    ("md", """**This is the headline design constraint.** A 30px radius with 3 message-passing
layers reaches ~90px, and the overwhelming majority of cells have no BMP4+ cell in
that range. Reaching ~1000px by stacking 30px hops would need ~30 layers; building an
explicit 960px-radius graph would need ~8×10⁸ edges. The workable route is a fixed
Gaussian **pyramid** of the mask as extra node features (`tools.spatial`) — long-range
context at O(pixels) per scale, with message passing still supplying fine structure."""),

    ("md", "## 4 · Response vs pattern geometry\\n\\nSigned distance to the illuminated boundary (+ inside, − outside) is the dose-response coordinate. It also distinguishes *inside a ring with no BMP4+ cell nearby* from *far outside the pattern with no BMP4+ cell nearby* — two situations every density feature reports identically, and the two the readouts differ most between."),

    ("code", """pl.plot_response_curve(df['BMP4_bin_signed_dist'], targets=[
    ('P(Sox17+)', df['Sox17_pos']), ('P(T+)', df['T_pos'])])
plt.show()

centre = pl.estimate_pattern_centre(df, 'BMP4_bin')
print('pattern centre (y, x) =', tuple(round(c) for c in centre))
pl.plot_radial_profile(df, centre, series=[
    ('BMP4+ fraction', 'BMP4_bin'), ('Sox17+ fraction', 'Sox17_pos'), ('T+ fraction', 'T_pos')])
plt.show()"""),

    ("md", """## 5 · The ceiling — how much is predictable at all?

Predict each cell from its **neighbours' true values**, leave-one-out. That is allowed
to peek at the answer everywhere except the cell being scored, so it is an upper bound
on any spatial model. Whatever fraction of the per-cell variation is independent
single-cell noise is unreachable, and this measures exactly that.

Every score in notebooks 02–04 should be read as a fraction of this, **not** of 1.0."""),

    ("code", """ceilings = {}
for marker in ['Sox17', 'T']:
    raw = df[f'{marker}_mean'].to_numpy(np.float32)
    y, ypos = ev.hurdle_target(raw, IF_BACKGROUNDS[f'{marker}_mean'])
    best = max((ev.oracle_ceiling(centroids, y, ypos, sigma=s) for s in [30, 60, 120]),
               key=lambda d: d['auroc'])
    ceilings[marker] = best
    print(f"{marker:6s} oracle ceiling:  R2 = {best['r2']:.3f}   AUROC = {best['auroc']:.3f}   (sigma={best['sigma']:g}px)")

pd.to_pickle(ceilings, f'{RESULTS}/oracle_ceilings.pkl')
print(f'\\nsaved -> {RESULTS}/oracle_ceilings.pkl')"""),

    ("md", """## What this implies

| finding | consequence |
|---|---|
| 86% of cells have no BMP4+ neighbour within 30px | a ~90px receptive field cannot work — **add multi-scale context features** |
| BMP4↔T field correlation −0.76; BMP4↔Sox17 only −0.25 | expect a good T model and a weak Sox17 one |
| density↔T correlation +0.69, and the ring is low-density | **density is a confound** — quantified in 02, controlled in 04 |
| Sox17 ceiling R² 0.25, T ceiling R² 0.44 | judge models against these, not against 1.0 |

Next: **02** measures how strongly the mask predicts density; **03** predicts the
markers from the mask; **04** repeats with density only, as the control."""),
]


# ==========================================================================
# 02 -- predicting density from the mask
# ==========================================================================
N2 = [
    ("md", """# 02 · Does the BMP4 mask predict *cell density*?

Notebook 01 found that the illuminated ring is also a **low cell-density** region,
and that smoothed cell density correlates with T at +0.69 — comparable in strength to
the BMP4 relationship itself.

That makes density a confound rather than a nuisance: if the mask predicts density,
and density predicts the markers, then a marker score obtained from mask features
cannot be attributed to BMP4 signalling without further work. This notebook measures
the first link directly, by making **density the prediction target**.

The result is what licenses (or forbids) the causal reading of notebooks 03 and 04."""),

    ("code", SETUP),

    ("code", """from tools.spatial import add_mask_pyramid, add_density_pyramid
from models.graph import border_mask
from models.recipe import RADIUS, MAX_SIGMA, build_data
from tools.morphology import indices_to_mask
from sklearn.ensemble import HistGradientBoostingRegressor

df = load_well(PRIMARY_WELL)
centroids = df[['centroid_y', 'centroid_x']].to_numpy(np.float32)
mask_cols = add_mask_pyramid(df, 'BMP4_bin', geometry=True)
dens_cols = add_density_pyramid(df)
print(f'{len(df):,} cells')
print('mask features   :', mask_cols)
print('density features:', dens_cols)"""),

    ("md", """## The two fields, side by side

If the illuminated region were density-neutral there would be no confound to worry
about. It is not."""),

    ("code", """fig = pl.plot_field_grid(df, rows=[('BMP4+', 'BMP4_bin'), ('cell density', None)],
                         sigmas=[60, 120, 240, 480])
plt.show()"""),

    ("md", """## Target: local cell density

Density at σ = 60px, log-transformed (it is a count, so its scale is multiplicative),
then rescaled to [0, 1] so R² is comparable with the marker notebooks."""),

    ("code", """TARGET_SIGMA = 60
raw_density = df[f'dens_{TARGET_SIGMA}'].to_numpy(np.float32)
y_density = np.log1p(raw_density)
y_density = ((y_density - y_density.min()) / (y_density.max() - y_density.min())).astype(np.float32)

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.hist(y_density, bins=80)
ax.set_xlabel(f'local cell density (sigma={TARGET_SIGMA}px, log, rescaled)')
ax.set_ylabel('# cells')
plt.tight_layout(); plt.show()"""),

    ("md", """## Splits

Both are reported throughout this project:

- **random** — held-out *cells*. Every test cell's own physical neighbours stay in
  train, so a purely local smoother scores well.
- **blocked** — held-out *regions* (800px tiles). This is what "generalises" means.

The gap between them measures how much a score depends on spatial autocorrelation."""),

    ("code", """shape = (int(centroids[:,0].max())+2, int(centroids[:,1].max())+2)
valid = np.where(~border_mask(centroids, shape, RADIUS))[0]

splits = {
    'random':  tuple(valid[i] for i in ev.random_split(len(valid), seed=0)),
    'blocked': tuple(valid[i] for i in ev.spatial_block_split(centroids[valid], block=800.0, seed=0)),
}
for k, (tr, va, te) in splits.items():
    print(f'{k:8s} train/val/test = {len(tr):,}/{len(va):,}/{len(te):,}')"""),

    ("md", """## Fit

Gradient boosting here rather than the GNN: the question is *how much information the
mask carries about density*, and a fast, strong tabular learner answers that without
the graph quietly re-introducing density through its own weighted-sum aggregation
(which is exactly what `normalize_data=False` does — see notebook 04)."""),

    ("code", """rows = []
# excludes 'BMP4_bin_signed_dist' -- same exclusion recipe.mask_features makes for
# the GNN (see its docstring): an unbounded raw distance whose scale tracks the
# SIZE of the illuminated pattern, not validated beyond the one matched well pair.
mask_cols_no_signed = [c for c in mask_cols if c != 'BMP4_bin_signed_dist']

FEATURE_SETS = {
    'mask, local only (sigma<=30)': ['BMP4_bin'] + [c for c in mask_cols if c.endswith(('_15','_30'))],
    'mask + context (sigma<=240)' : ['BMP4_bin'] + [c for c in mask_cols_no_signed
                                                     if not c.startswith('BMP4_bin_gauss_')
                                                     or int(c.rsplit('_',1)[1]) <= MAX_SIGMA],
    'mask + all scales'           : ['BMP4_bin'] + mask_cols_no_signed,
}

for split_name, (tr, va, te) in splits.items():
    for fname, cols in FEATURE_SETS.items():
        X = df[cols].to_numpy(np.float32)
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                                          l2_regularization=5.0, early_stopping=False,
                                          random_state=0).fit(X[tr], y_density[tr])
        met = ev.evaluate(y_density[te], m.predict(X[te]))
        rows.append(dict(split=split_name, features=fname, **met))
        print(f'{split_name:8s} {fname:30s} R2={met["r2"]:+.4f}  spearman={met["spearman"]:+.3f}')

density_results = pd.DataFrame(rows)
density_results.round(4)"""),

    ("code", """pl.plot_metric_comparison(density_results, metric='r2', group='features', hue='split',
                          title='predicting local cell density from the BMP4 mask')
plt.show()"""),

    ("md", "## Where the prediction is right, and where it is wrong"),

    ("code", """cols = FEATURE_SETS['mask + context (sigma<=240)']
X = df[cols].to_numpy(np.float32)
tr, va, te = splits['blocked']
best = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                                     l2_regularization=5.0, early_stopping=False,
                                     random_state=0).fit(X[tr], y_density[tr])
pred_density = best.predict(X)

pl.plot_prediction_panel(df, 'BMP4_bin', y_density, pred_density,
                         target_name=f'cell density (sigma={TARGET_SIGMA}px)',
                         mask_name='BMP4 mask (input)', smooth_sigma=60)
plt.show()"""),

    ("code", """pl.plot_response_curve(df['BMP4_bin_signed_dist'], targets=[
    ('measured density', y_density), ('predicted density', pred_density)],
    title='cell density vs distance to the illuminated boundary')
plt.show()"""),

    ("md", """## Read-out

The mask predicts density well **inside and around the illuminated pattern** and
poorly everywhere else — which is exactly what "the illumination depresses local cell
density" looks like. The response curve makes the effect explicit: density drops
inside the boundary.

**Consequence for notebooks 03 and 04.** Mask features and density features are not
independent inputs. Any marker score from the mask has a route through density, and
the honest baseline for "BMP4 predicts the marker" is therefore *not* zero — it is
whatever density alone achieves. Notebook 04 measures that baseline, and the two must
be read together."""),

    ("code", """density_results.to_csv(f'{RESULTS}/02_density_from_mask.csv', index=False)
print(f'saved -> {RESULTS}/02_density_from_mask.csv')"""),
]


# ==========================================================================
# 03 -- markers from the BMP4 mask
# ==========================================================================
COMMON_FIT = '''RUN_ABLATION = True     # set False to skip the A/B/C comparison and only fit the final model
QUICK = False           # True -> few epochs, for a fast pass through the notebook

EPOCHS = 150 if QUICK else recipe.EPOCHS
PATIENCE = 150 if QUICK else recipe.PATIENCE
# QUICK checkpoints live under a different name so a provisional 150-epoch run can
# never be silently reused as if it were the real, fully-trained one (or vice versa).
CACHE_SUFFIX = '_quick' if QUICK else ''
'''

N3 = [
    ("md", """# 03 · Predicting Sox17 and T from the BMP4 mask

Using what notebook 01 established:

- context features out to σ = 240px, because a 90px receptive field leaves most cells
  with a constant input — but no further, because larger scales become position
  proxies that do not transfer between wells;
- the **prior-corrected** two-part head (`models.train.train_calibrated`), because the
  re-balanced classifier prior otherwise inflates every prediction ~5× and drives R²
  negative while leaving AUROC untouched;
- scores read against the **oracle ceiling** from notebook 01, not against 1.0.

The notebook fits an A → B → C ablation so each fix can be attributed separately, then
evaluates the final model on a held-out region and a held-out well.

Config **C**'s fits (the one that matters) are cached to disk as they're trained
(`recipe.load_or_fit`, keyed by well/marker/split) -- notebooks 04, 05 and 06 default
to this exact well/feature-set, and will load these checkpoints instead of retraining
them if this notebook has already run. Delete a checkpoint under `results/models/` to
force a refit."""),

    ("code", SETUP),

    ("code", """import torch
from models import recipe
from models.graph import border_mask
from models.train import train_predictor
from models.predictors import IFPredictor
from tools.spatial import add_mask_pyramid
from tools.morphology import indices_to_mask

""" + COMMON_FIT),

    ("code", """df = load_well(PRIMARY_WELL)
centroids = df[['centroid_y', 'centroid_x']].to_numpy(np.float32)
IF_COLS = ['Sox17_mean', 'T_mean']

X_MASK = recipe.mask_features(df, 'BMP4_bin', max_sigma=recipe.MAX_SIGMA, geometry=True)
X_LOCAL = ['BMP4_bin']          # what a radius=30 x 3-layer graph effectively sees

print(f'{len(df):,} cells')
print(f'final feature set ({len(X_MASK)}):')
for c in X_MASK: print('   ', c)

ceilings = pd.read_pickle(f'{RESULTS}/oracle_ceilings.pkl')
print('\\noracle ceilings:', {k: {m: round(v[m],3) for m in ("r2","auroc")} for k, v in ceilings.items()})"""),

    ("code", """shape = (int(centroids[:,0].max())+2, int(centroids[:,1].max())+2)
valid = np.where(~border_mask(centroids, shape, recipe.RADIUS))[0]

SPLITS = {
    'random':  tuple(valid[i] for i in ev.random_split(len(valid), seed=0)),
    'blocked': tuple(valid[i] for i in ev.spatial_block_split(centroids[valid], block=800.0, seed=0)),
}
N = len(df)
MASKS = {k: tuple(indices_to_mask(i, N) for i in v) for k, v in SPLITS.items()}
for k, (tr, va, te) in SPLITS.items():
    print(f'{k:8s} train/val/test = {len(tr):,}/{len(va):,}/{len(te):,}')"""),

    ("md", """## The ablation

| | features | training |
|---|---|---|
| **A** | BMP4 mask only | `train_predictor`, `lr=1e-4`, uncorrected output — the original setup |
| **B** | BMP4 mask only | `train_calibrated` — isolates the calibration + learning-rate fix |
| **C** | mask + context (σ ≤ 240px) | `train_calibrated` — adds the receptive-field fix |

A → B and B → C are *not* clean single-variable steps (A also differs in learning rate
and epoch budget, so it is undertrained). The clean isolation of the calibration effect
is the cell further down that corrects an **already-trained** model's logit and
re-scores it without retraining."""),

    ("code", """def fit_config(config, x_cols, y_col, split_name, cache_path=None):
    tr, va, te = SPLITS[split_name]
    trm, vam, tem = MASKS[split_name]
    if config == 'A':
        data = recipe.build_data(df, x_cols, y_col, tr, IF_BACKGROUNDS)
        torch.manual_seed(0)
        model = IFPredictor(in_channels=len(x_cols), num_outputs=1, num_layers=recipe.NUM_LAYERS,
                            hidden_channels=recipe.HIDDEN, normalize_data=False, two_part=True)
        history = train_predictor(model, data, trm, vam, epochs=EPOCHS, patience=PATIENCE,
                                  lr=1e-4, device='cpu', desc=f'A {y_col}')
        model = model.to('cpu').eval()
        with torch.no_grad():
            pred, _ = model(data['x'], data['edge_index'], data['edge_weight'])
        pred, log_w = pred.numpy().ravel(), 0.0
    elif cache_path is not None:
        # config C only -- the recommended model, worth caching since 04/05/06 reuse it
        model, data, log_w, history = recipe.load_or_fit(
            cache_path, df, x_cols, y_col, trm, vam, tr, custom_background=epochs=EPOCHS, patience=PATIENCE, verbose=False)
        pred = recipe.predict(model, data, log_w)
    else:
        model, data, log_w, history = recipe.fit(
            df, x_cols, y_col, trm, vam, tr, custom_background=epochs=EPOCHS, patience=PATIENCE, verbose=False)
        pred = recipe.predict(model, data, log_w)
    y = data['y'].numpy().ravel(); ypos = data['y_positive'].numpy().ravel()
    met = ev.evaluate(y[tem.numpy()], pred[tem.numpy()], ypos[tem.numpy()])
    return dict(model=model, data=data, log_w=log_w, history=history, pred=pred,
                y=y, ypos=ypos, metrics=met)"""),

    ("code", """CONFIGS = [('A  notebook as-is', 'A', X_LOCAL),
           ('B  +calibration',   'B', X_LOCAL),
           ('C  +context',       'C', X_MASK)]

rows, final = [], {}
for marker in ['Sox17', 'T']:
    y_col = f'{marker}_mean'
    for split_name in SPLITS:
        for label, cfg, cols in ([CONFIGS[-1]] if not RUN_ABLATION else CONFIGS):
            cache_path = (f'{RESULTS}/models/{PRIMARY_WELL}_mask_{marker}_{split_name}{CACHE_SUFFIX}_gnn.pt'
                         if cfg == 'C' else None)
            out = fit_config(cfg, cols, y_col, split_name, cache_path=cache_path)
            rows.append(dict(target=marker, split=split_name, config=label, **out['metrics']))
            print(f"{marker:6s} {split_name:8s} {label:20s} "
                  f"R2={out['metrics']['r2']:+.4f}  AUROC={out['metrics']['auroc']:.3f}  "
                  f"AP={out['metrics']['ap']:.3f}")
            if cfg == 'C':
                final[(marker, split_name)] = out

mask_results = pd.DataFrame(rows)
mask_results.round(4)"""),

    ("code", """for marker in ['Sox17', 'T']:
    sub = mask_results[mask_results.target == marker]
    pl.plot_metric_comparison(sub, metric='r2', group='config', hue='split',
                              ceiling=ceilings[marker]['r2'],
                              title=f'{marker}: R2 vs oracle ceiling')
    plt.show()"""),

    ("md", """## The calibration effect, isolated

Take one **already-trained** model, change nothing but subtract `log(pos_weight)` from
the classifier logit, and re-score.

The correction is monotone *in the logit*, so the classifier's own ranking is
untouched. The reported output is `prob × magnitude` though, and rescaling `prob`
alone can reorder that product slightly — so AUROC moves a little rather than not at
all. It should move by <0.01 while R² moves by a lot; that asymmetry is the signature
of a pure calibration problem rather than a ranking one."""),

    ("code", """out = final[('Sox17', 'random')]
model, data = out['model'], out['data']
tem = MASKS['random'][2].numpy()

with torch.no_grad():
    _, logit, magnitude, _ = model.forward_two_part(data['x'], data['edge_index'], data['edge_weight'])
prob_raw = torch.sigmoid(logit).numpy().ravel()
prob_cal = torch.sigmoid(logit - out['log_w']).numpy().ravel()
mag = magnitude.numpy().ravel()

print(f"true positive rate      : {out['ypos'][tem].mean():.3f}")
print(f"mean predicted prob     : {prob_raw[tem].mean():.3f}   <- inflated by pos_weight")
print(f"  after -log(w) applied : {prob_cal[tem].mean():.3f}")
print(f"pos_weight = {np.exp(out['log_w']):.2f}   log(w) = {out['log_w']:.3f}\\n")
from sklearn.metrics import r2_score, roc_auc_score
for name, p in [('uncorrected', prob_raw * mag), ('corrected', prob_cal * mag)]:
    print(f'{name:12s} R2={r2_score(out["y"][tem], p[tem]):+.4f}   '
          f'AUROC={roc_auc_score(out["ypos"][tem], p[tem]):.4f}')

fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
pl.plot_calibration(out['ypos'][tem], prob_raw[tem], ax=axs[0], title='before correction')
pl.plot_calibration(out['ypos'][tem], prob_cal[tem], ax=axs[1], title='after correction')
plt.tight_layout(); plt.show()"""),

    ("md", "## The final model"),

    ("code", """for marker in ['Sox17', 'T']:
    out = final[(marker, 'random')]
    if out['history'] is not None:
        pl.plot_training_curves(out['history'], ceiling=ceilings[marker]['r2'], title=marker)
        plt.show()
    else:
        print(f'{marker}: loaded from a cached checkpoint -- no training-curve history to plot')

    tem = MASKS['random'][2].numpy()
    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    pl.plot_pred_vs_actual(out['y'][tem], out['pred'][tem], out['ypos'][tem], ax=ax,
                           title=f"{marker}: R2={out['metrics']['r2']:.3f}  AUROC={out['metrics']['auroc']:.3f}")
    plt.tight_layout(); plt.show()

    pl.plot_prediction_panel(df, 'BMP4_bin', out['y'], out['pred'], target_name=marker,
                             mask_name='BMP4 mask (input)', smooth_sigma=60)
    plt.show()"""),

    ("md", """## Held-out well

`W9_pattern1` was imaged at the same illumination power in the same session as
`W8_pattern1` — the only pair in `data/stitched` where a transfer score measures
biology rather than batch.

Scored with Spearman ρ against **raw** intensity, which is invariant to whatever
positive/negative threshold W9 would need. Only W8_pattern1's threshold is verified."""),

    ("code", """from scipy.stats import spearmanr

df9 = load_well(MATCHED_PAIR[1])
add_mask_pyramid(df9, 'BMP4_bin', geometry=True)

transfer = []
for marker in ['Sox17', 'T']:
    out = final[(marker, 'random')]
    d9 = recipe.build_data(df9, X_MASK, f'{marker}_mean', np.arange(len(df9)), IF_BACKGROUNDS)
    d9['x'] = torch.from_numpy(out['data']['x_scaler'].transform(df9[X_MASK].to_numpy(np.float32)))
    p9 = recipe.predict(out['model'], d9, out['log_w'])
    rho_self = spearmanr(out['pred'], df[f'{marker}_mean']).statistic
    rho_w9 = spearmanr(p9, df9[f'{marker}_mean']).statistic
    transfer.append(dict(target=marker, rho_train_well=rho_self, rho_heldout_well=rho_w9))
    print(f'{marker:6s} rho  W8_pattern1 (train) = {rho_self:+.3f}   W9_pattern1 (held out) = {rho_w9:+.3f}')

pd.DataFrame(transfer).round(3)"""),

    ("md", """## Save

The config-C checkpoints themselves were already saved during fitting above (one
per marker/split, via `recipe.load_or_fit`) -- only the results tables need saving
here."""),

    ("code", """mask_results.to_csv(f'{RESULTS}/03_markers_from_mask.csv', index=False)
pd.DataFrame(transfer).to_csv(f'{RESULTS}/03_transfer.csv', index=False)
print(f'saved -> {RESULTS}/03_markers_from_mask.csv')
print(f'saved -> {RESULTS}/03_transfer.csv')"""),

    ("md", """## Read-out

Both fixes matter and they are independent: the prior correction rescues R² while
leaving AUROC essentially where it was, and the context features move both.

With `QUICK = True` the models are trained for a tenth of the epochs they need, so
treat the absolute numbers as provisional — set `QUICK = False` for the real ones.

But these numbers **cannot yet be attributed to BMP4 signalling** — notebook 02 showed
the mask also predicts cell density, and the graph itself supplies density through its
weighted-sum aggregation. Notebook 04 runs the control that separates the two."""),
]


# ==========================================================================
# 04 -- markers from density only
# ==========================================================================
N4 = [
    ("md", """# 04 · The control: predicting Sox17 and T from *density alone*

Notebook 03 produced good-looking numbers from the BMP4 mask. Notebook 02 showed the
mask also predicts local cell density. So the obvious question is whether notebook 03
measured BMP4 signalling or colony structure.

Two controls, identical model, identical splits, identical everything else:

- **density only** — smoothed cell density at the same scales. No marker channel, no
  mask, nothing derived from BMP4. Whatever this reaches is available with *no BMP4
  information at all*, and it is therefore the real baseline for notebook 03.
- **shuffled mask** — the BMP4 labels permuted across cells. Destroys the mask's
  spatial arrangement while preserving the number of positives *and* the density field
  exactly. Isolates the contribution of the mask's arrangement specifically.

There is a further reason to run this: `normalize_data=False` makes
`WeightedRadiusConv` aggregate by weighted **sum**, which is local density by
construction, so the graph has access to density whether or not it appears in
`x_cols`.

All three feature sets are cached the same way 03 caches its config C
(`recipe.load_or_fit`) -- if 03 already ran, `BMP4 mask + context` loads straight
from its checkpoint instead of retraining; `density only`/`shuffled mask` are new
here and get cached for the first time, so a second run of this notebook (or one
that reuses these paths later) is fast too."""),

    ("code", SETUP),

    ("code", """import torch
from models import recipe
from models.graph import border_mask
from tools.spatial import add_mask_pyramid, add_density_pyramid, shuffle_mask
from tools.morphology import indices_to_mask

QUICK = False
EPOCHS = 150 if QUICK else recipe.EPOCHS
PATIENCE = 150 if QUICK else recipe.PATIENCE
CACHE_SUFFIX = '_quick' if QUICK else ''  # keeps a provisional run from being reused as the real one

df = load_well(PRIMARY_WELL)
centroids = df[['centroid_y', 'centroid_x']].to_numpy(np.float32)
IF_COLS = ['Sox17_mean', 'T_mean']

X_MASK = recipe.mask_features(df, 'BMP4_bin', max_sigma=recipe.MAX_SIGMA, geometry=True)
X_DENSITY = recipe.density_features(df, max_sigma=recipe.MAX_SIGMA)

shuffle_mask(df, 'BMP4_bin', seed=0, out_col='mask_shuffled')
X_SHUFFLED = recipe.mask_features(df, 'mask_shuffled', max_sigma=recipe.MAX_SIGMA, geometry=True)

FEATURE_SETS = {
    'BMP4 mask + context': X_MASK,
    'density only':        X_DENSITY,
    'shuffled mask':       X_SHUFFLED,
}
# same slug 03 uses for 'BMP4 mask + context' ('mask') -- so that arm's checkpoint
# path matches 03's exactly and this notebook loads it instead of retraining
SLUGS = {'BMP4 mask + context': 'mask', 'density only': 'density', 'shuffled mask': 'shuffled'}
for k, v in FEATURE_SETS.items():
    print(f'{k:22s} {len(v):2d} features')"""),

    ("md", "## What the density features look like"),

    ("code", """fig = pl.plot_cell_maps(df, [
    ('BMP4+ mask', 'BMP4_bin', 'gray_r'),
    ('shuffled mask', 'mask_shuffled', 'gray_r'),
    ('cell density (sigma=60)', 'dens_60', 'viridis'),
    ('cell density (sigma=240)', 'dens_240', 'viridis'),
], suptitle='the real mask, the shuffled control, and the density field')
plt.show()"""),

    ("code", """shape = (int(centroids[:,0].max())+2, int(centroids[:,1].max())+2)
valid = np.where(~border_mask(centroids, shape, recipe.RADIUS))[0]
SPLITS = {
    'random':  tuple(valid[i] for i in ev.random_split(len(valid), seed=0)),
    'blocked': tuple(valid[i] for i in ev.spatial_block_split(centroids[valid], block=800.0, seed=0)),
}
N = len(df)
MASKS = {k: tuple(indices_to_mask(i, N) for i in v) for k, v in SPLITS.items()}

ceilings = pd.read_pickle(f'{RESULTS}/oracle_ceilings.pkl')"""),

    ("code", """rows, fitted = [], {}
for marker in ['Sox17', 'T']:
    y_col = f'{marker}_mean'
    for split_name in SPLITS:
        tr, va, te = SPLITS[split_name]
        trm, vam, tem = MASKS[split_name]
        for fname, cols in FEATURE_SETS.items():
            cache_path = f'{RESULTS}/models/{PRIMARY_WELL}_{SLUGS[fname]}_{marker}_{split_name}{CACHE_SUFFIX}_gnn.pt'
            model, data, log_w, history = recipe.load_or_fit(
                cache_path, df, cols, y_col, trm, vam, tr, custom_background=epochs=EPOCHS, patience=PATIENCE, verbose=False)
            pred = recipe.predict(model, data, log_w)
            y = data['y'].numpy().ravel(); ypos = data['y_positive'].numpy().ravel()
            t = tem.numpy()
            met = ev.evaluate(y[t], pred[t], ypos[t])
            rows.append(dict(target=marker, split=split_name, features=fname, **met))
            fitted[(marker, split_name, fname)] = dict(model=model, data=data, pred=pred,
                                                       y=y, ypos=ypos, history=history)
            print(f'{marker:6s} {split_name:8s} {fname:22s} '
                  f'R2={met["r2"]:+.4f}  AUROC={met["auroc"]:.3f}')

control_results = pd.DataFrame(rows)
control_results.round(4)"""),

    ("code", """for marker in ['Sox17', 'T']:
    sub = control_results[control_results.target == marker]
    pl.plot_metric_comparison(sub, metric='auroc', group='features', hue='split',
                              ceiling=ceilings[marker]['auroc'],
                              title=f'{marker}: AUROC — does the BMP4 mask beat its controls?')
    plt.show()"""),

    ("md", """## Reading the result

The quantity of interest is **not** any single bar — it is the *gap* between the real
mask and its controls, computed within each split:

- `mask − density only` — what the BMP4 pattern adds beyond colony structure. Anything
  at or near zero means the score is available with no BMP4 information at all, and
  should not be described as predicting the marker from BMP4.
- `mask − shuffled` — what the mask's spatial *arrangement* adds beyond simply having
  that many positive cells scattered around. This is the tighter control of the two,
  because the shuffled mask preserves the density field exactly.

Expect the gaps to be **smaller under the random split than the blocked split**. Two
reasons push that way: spatial autocorrelation lets a purely local smoother score well
on held-out cells whose neighbours are in train, and `normalize_data=False` means the
graph aggregates by weighted sum, i.e. it can read local density off the graph itself
regardless of what is in `x_cols`. Where the gap is near zero, the blocked split is the
only one of the two that can tell the mask apart from its own controls.

> **On the numbers below.** With `QUICK = True` this notebook trains for 150 epochs,
> which is roughly a tenth of what these models need — validation R² is still climbing.
> The ordering of the bars is usually stable but the gaps are not; set `QUICK = False`
> before drawing any conclusion from a small gap. For reference, the fully-trained run
> on the *cropped* field (ring only, `crop=LEGACY_CROP`) gave T AUROC 0.800 for the
> mask against 0.737 density-only and 0.735 shuffled under the blocked split, and all
> three within 0.013 of each other under the random split."""),

    ("code", """summary = (control_results
           .pivot_table(index=['target', 'split'], columns='features', values=['r2', 'auroc'])
           .round(3))
display(summary)

gap = (control_results.pivot_table(index=['target','split'], columns='features', values='auroc')
       .assign(**{'mask - density': lambda d: d['BMP4 mask + context'] - d['density only'],
                  'mask - shuffled': lambda d: d['BMP4 mask + context'] - d['shuffled mask']})
       [['mask - density', 'mask - shuffled']].round(3))
print('\\nAUROC advantage of the real mask over its controls:')
gap"""),

    ("md", "## Where density succeeds and the mask does not, spatially"),

    ("code", """marker = 'T'
for fname in ['BMP4 mask + context', 'density only']:
    out = fitted[(marker, 'blocked', fname)]
    pl.plot_prediction_panel(df, 'BMP4_bin', out['y'], out['pred'],
                             target_name=f'{marker}  ({fname})',
                             mask_name='BMP4 mask', smooth_sigma=60)
    plt.show()"""),

    ("md", """## Conclusions across all four notebooks

1. **The original model could not work.** A 30px × 3-layer receptive field leaves most
   cells with a constant input, against a response region of radius ~600px (nb 01).
2. **The reported R² was a calibration artefact.** The re-balanced classifier prior
   inflates predictions ~5×; correcting the logit fixes R² without touching AUROC
   (nb 03).
3. **Illumination changes cell density**, so mask and density features are not
   independent (nb 02).
4. **Judge the mask against its controls, not against zero** (this notebook). How much
   of the score is attributable to BMP4 is the gap to density-only and shuffled-mask,
   and that gap depends on the split — check it rather than assuming it.
5. **T is predicted better than Sox17**, and T is the one that transfers to the
   held-out well (nb 03: ρ 0.40 vs 0.29). Both sit well below R² 1.0 but much closer to
   their oracle ceilings, which is the comparison that matters.

If Sox17 is the biological question, the productive next step is more wells at matched
illumination power — not more architecture."""),

    ("code", """control_results.to_csv(f'{RESULTS}/04_density_controls.csv', index=False)
print(f'saved -> {RESULTS}/04_density_controls.csv')"""),
]


# ==========================================================================
# 05 -- train + analyze + cross-predict, on whatever you point it at
# ==========================================================================
N5 = [
    ("md", """# 05 · Train, analyze, and cross-predict with the recommended model

Notebook 03 established the model worth using: mask + context features (σ ≤ 240px)
and the prior-corrected two-part head (`models.recipe`/`models.train.
train_calibrated`). This notebook applies exactly that configuration, but without
the ablation -- edit the **config cell** below to point it at any well or any
`*_features.parquet` path, train, then:

- run the same model-analysis views `gnn_v1/mask_predict.ipynb` used (spatial feature
  maps, intrinsic + spatial correlation, gradient-based neighbor sensitivity by
  layer), updated to the current `models`/`tools` API, and
- cross-predict onto other images the way `gnn_v1/cross_predict.ipynb` did -- load
  the saved checkpoint's fitted scalers (never refit) and score it against a
  *different* image's cells, which is the only way to tell "generalizes" from
  "memorized this well".

Only `TRAIN_SOURCE`/`EVAL_SOURCES` need to change to run this on new data; nothing
else in the notebook assumes `PRIMARY_WELL` specifically."""),

    ("code", SETUP),

    ("code", """import os, re
import torch
from models import recipe
from models.graph import border_mask
from models.train import calibrated_predict
from models.predictors import IFPredictor
from models.checkpoint import load_model, apply_prediction_data
from tools.morphology import indices_to_mask, plot_correlation_heatmap, plot_spatial_cross_correlation
from tools.dataset import STITCHED

# ---- what to run: edit these to point the notebook at new data, nothing else ----
TRAIN_SOURCE = PRIMARY_WELL          # a well name (looked up under data/stitched/) or a full path to a *_features.parquet
EVAL_SOURCES = [MATCHED_PAIR[1]]     # 0+ other images to cross-predict on once trained -- same name-or-path rule
MASK_CHANNEL = 'BMP4'                # f'{MASK_CHANNEL}_bin' is the model input; needs an f'{MASK_CHANNEL}+' column
TARGET_MARKERS = ['Sox17', 'T']      # y columns are f'{marker}_mean'
CUSTOM_BACKGROUNDS = IF_BACKGROUNDS  # {y_col: threshold} -- only W8_pattern1's thresholds are verified (see tools.dataset);
                                     # extend/override this dict before trusting AUROC/R2 on a different well

QUICK = False                # True -> few epochs, for a fast pass through the notebook
ANALYSIS_SPLIT = 'blocked'   # which trained model ('random' | 'blocked') the analysis/cross-predict sections use below

EPOCHS = 150 if QUICK else recipe.EPOCHS
PATIENCE = 150 if QUICK else recipe.PATIENCE
CACHE_SUFFIX = '_quick' if QUICK else ''  # keeps a provisional run from being reused as the real one"""),

    ("md", "## Load & features"),

    ("code", """def load_cell_table(source, mask_channel=MASK_CHANNEL):
    \"\"\"`source` is a well name under data/stitched/, or a full path to a
    *_features.parquet file -- this is the hook for pointing the notebook at a
    different image. Mirrors `tools.dataset.load_well` but accepts any path.\"\"\"
    path = source if str(source).endswith('.parquet') else os.path.join(STITCHED, f'{source}_features.parquet')
    out = pd.read_parquet(path)
    out[f'{mask_channel}_bin'] = out[f'{mask_channel}+'].astype(np.float32)
    return out

def _slug(source):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(source)).strip('_')

df = load_cell_table(TRAIN_SOURCE)
centroids = df[['centroid_y', 'centroid_x']].to_numpy(np.float32)
IF_COLS = [f'{m}_mean' for m in TARGET_MARKERS]
TRAIN_SLUG = _slug(TRAIN_SOURCE)  # e.g. PRIMARY_WELL itself, unchanged -- matches 03/04's checkpoint naming exactly

X_MASK = recipe.mask_features(df, f'{MASK_CHANNEL}_bin', max_sigma=recipe.MAX_SIGMA, geometry=True)
print(f'{len(df):,} cells from {TRAIN_SOURCE}')
print(f'feature set ({len(X_MASK)}):')
for c in X_MASK: print('   ', c)"""),

    ("code", """shape = (int(centroids[:,0].max())+2, int(centroids[:,1].max())+2)
valid = np.where(~border_mask(centroids, shape, recipe.RADIUS))[0]

SPLITS = {
    'random':  tuple(valid[i] for i in ev.random_split(len(valid), seed=0)),
    'blocked': tuple(valid[i] for i in ev.spatial_block_split(centroids[valid], block=800.0, seed=0)),
}
N = len(df)
MASKS = {k: tuple(indices_to_mask(i, N) for i in v) for k, v in SPLITS.items()}
for k, (tr, va, te) in SPLITS.items():
    print(f'{k:8s} train/val/test = {len(tr):,}/{len(va):,}/{len(te):,}')"""),

    ("md", """## Train

The recommended config directly -- no ablation here, 03 already made the case for
it. Both splits are fit for every marker so the random/blocked gap stays visible;
`ANALYSIS_SPLIT` picks which of the two the sections below dig into.

Cached the same way 03/04 cache theirs (`recipe.load_or_fit`): with the default
`TRAIN_SOURCE = PRIMARY_WELL`, this loads 03's checkpoints straight off disk instead
of retraining. Pointed at a different well, there's nothing to reuse, so it trains
and caches fresh -- and a second run against that same well would then hit its own
cache."""),

    ("code", """ceilings = None
try:
    ceilings = pd.read_pickle(f'{RESULTS}/oracle_ceilings.pkl')
except FileNotFoundError:
    pass  # only meaningful when TRAIN_SOURCE is the well notebook 01 was run on

rows, fitted = [], {}
for marker in TARGET_MARKERS:
    y_col = f'{marker}_mean'
    for split_name, (tr, va, te) in SPLITS.items():
        trm, vam, tem = MASKS[split_name]
        cache_path = f'{RESULTS}/models/{TRAIN_SLUG}_mask_{marker}_{split_name}{CACHE_SUFFIX}_gnn.pt'
        model, data, log_w, history = recipe.load_or_fit(
            cache_path, df, X_MASK, y_col, trm, vam, tr, custom_background=CUSTOM_BACKGROUNDS,
            epochs=EPOCHS, patience=PATIENCE, verbose=False)
        pred = recipe.predict(model, data, log_w)
        y = data['y'].numpy().ravel(); ypos = data['y_positive'].numpy().ravel()
        met = ev.evaluate(y[tem.numpy()], pred[tem.numpy()], ypos[tem.numpy()])
        rows.append(dict(target=marker, split=split_name, **met))
        fitted[(marker, split_name)] = dict(model=model, data=data, log_w=log_w, history=history,
                                            pred=pred, y=y, ypos=ypos, metrics=met)
        print(f"{marker:6s} {split_name:8s} R2={met['r2']:+.4f}  "
              f"AUROC={met.get('auroc', float('nan')):.3f}  AP={met.get('ap', float('nan')):.3f}")

results = pd.DataFrame(rows)
results.round(4)"""),

    ("code", """for marker in TARGET_MARKERS:
    sub = results[results.target == marker]
    ceiling = ceilings[marker]['r2'] if ceilings is not None and marker in ceilings else None
    pl.plot_metric_comparison(sub, metric='r2', group='target', hue='split',
                              ceiling=ceiling, title=f'{marker}: R2, random vs blocked split')
    plt.show()"""),

    ("md", "## The trained model"),

    ("code", """for marker in TARGET_MARKERS:
    out = fitted[(marker, ANALYSIS_SPLIT)]
    ceiling = ceilings[marker]['r2'] if ceilings is not None and marker in ceilings else None
    if out['history'] is not None:
        pl.plot_training_curves(out['history'], ceiling=ceiling, title=f'{marker} ({ANALYSIS_SPLIT} split)')
        plt.show()
    else:
        print(f'{marker}: loaded from a cached checkpoint -- no training-curve history to plot')

    tem = MASKS[ANALYSIS_SPLIT][2].numpy()
    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    pl.plot_pred_vs_actual(out['y'][tem], out['pred'][tem], out['ypos'][tem], ax=ax,
                           title=f"{marker}: R2={out['metrics']['r2']:.3f}  "
                                 f"AUROC={out['metrics'].get('auroc', float('nan')):.3f}")
    plt.tight_layout(); plt.show()

    pl.plot_prediction_panel(df, f'{MASK_CHANNEL}_bin', out['y'], out['pred'], target_name=marker,
                             mask_name=f'{MASK_CHANNEL} mask (input)', smooth_sigma=60)
    plt.show()"""),

    ("md", """## Model analysis

Same questions `gnn_v1/mask_predict.ipynb` asked of the earlier model, re-run against
the current `x_cols`/API: what do the scaled inputs look like in space, how do they
correlate with each other (same cell, and spatially with a neighbor), and -- since
`WeightedRadiusConv`'s edge weight is a fixed function of distance with nothing
learned to read off directly -- a gradient-based sensitivity in place of GAT-style
attention (`IFPredictor.message_sensitivity_by_layer`)."""),

    ("code", """ANALYSIS_MARKER = TARGET_MARKERS[0]
out = fitted[(ANALYSIS_MARKER, ANALYSIS_SPLIT)]
model, data = out['model'], out['data']
x_cols = data['x_cols']

feat = pd.DataFrame(data['x'].numpy(), columns=x_cols)
feat[f'{ANALYSIS_MARKER}_mean (scaled)'] = data['y'].numpy().ravel()
feat_cols = list(feat.columns)

ncols = 4
nrows = int(np.ceil(len(feat_cols) / ncols))
fig, axs = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 5.5 * nrows), squeeze=False)
axs = axs.ravel()
for ax, col in zip(axs, feat_cols):
    sc = ax.scatter(df['centroid_x'], df['centroid_y'], c=feat[col], cmap='coolwarm', s=2)
    ax.set_title(col, fontsize=10)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
for ax in axs[len(feat_cols):]:
    ax.set_visible(False)
fig.suptitle(f'{ANALYSIS_MARKER}: scaled model inputs + target, in space')
fig.tight_layout()
plt.show()"""),

    ("code", """fig, axs = plt.subplots(1, 2, figsize=(13, 6))
plot_correlation_heatmap(feat, feat_cols, title='Pearson correlation (intrinsic, same cell)',
                         corr_type='pearson', ax=axs[0])
plot_spatial_cross_correlation(feat, feat_cols, data['edge_index'], weights=data['edge_weight'],
                               title=\"Pearson spatial cross-corr. (extrinsic, neighbor)\",
                               corr_type='pearson', ax=axs[1])
plt.tight_layout(); plt.show()"""),

    ("md", """### Gradient-based neighbor sensitivity

For each layer, the per-edge message `edge_weight * lin(x_j)`, graded by how much
the *actual* prediction moved if that message changed (grad × message -- the
standard attribution choice, not the raw gradient alone). Correlated against edge
distance and a handful of representative features -- not every `x_col`, since
`X_MASK` can carry a dozen-plus pyramid scales and plotting all of them by layer
would be more panels than anyone reads."""),

    ("code", """layer_sensitivities = model.message_sensitivity_by_layer(data['x'], data['edge_index'], data['edge_weight'])

centroids_t = torch.from_numpy(centroids.copy())
src, dst = data['edge_index']
edge_dist = torch.norm(centroids_t[src] - centroids_t[dst], dim=1)

sample_cols = list(dict.fromkeys([x_cols[0]] + x_cols[1::max(1, len(x_cols) // 3)][:3]))

n_panels = 1 + len(sample_cols)
n_layers = len(layer_sensitivities)
fig, axes = plt.subplots(n_layers, n_panels, figsize=(5 * n_panels, 5 * n_layers), sharey=True, squeeze=False)

for layer_idx, layer_sens in enumerate(layer_sensitivities):
    ax = axes[layer_idx, 0]
    ax.scatter(edge_dist.numpy(), layer_sens.numpy(), s=2, alpha=0.15, linewidths=0)
    corr_d = np.corrcoef(edge_dist.numpy(), layer_sens.numpy())[0, 1]
    ax.set_xlabel('edge distance (px) -- 0 == self-loop')
    ax.set_ylabel(f'layer {layer_idx} sensitivity (grad x message)')
    ax.set_title(f'layer {layer_idx}: distance (r = {corr_d:.3f})')

    for j, col in enumerate(sample_cols):
        ax = axes[layer_idx, j + 1]
        values = data['x'][src, x_cols.index(col)].numpy()
        ax.scatter(values, layer_sens.numpy(), s=2, alpha=0.15, linewidths=0)
        corr_c = np.corrcoef(values, layer_sens.numpy())[0, 1]
        ax.set_xlabel(f'{col} (scaled)')
        ax.set_title(f'layer {layer_idx}: {col} (r = {corr_c:.3f})')

fig.suptitle(f'{ANALYSIS_MARKER}: sensitivity vs. distance and feature value, by layer', y=1.0)
fig.tight_layout()
plt.show()"""),

    ("code", """rng = np.random.default_rng(0)
example_cell_pos = rng.choice(SPLITS[ANALYSIS_SPLIT][2])  # a held-out test cell
mask_edges = (dst == example_cell_pos)
field_pos = src[mask_edges].numpy()  # includes the self-loop -- one of these IS example_cell_pos itself

n_layers = len(layer_sensitivities)
n_panels = n_layers + len(sample_cols)
fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))

for layer_idx, layer_sens in enumerate(layer_sensitivities):
    field_sens = layer_sens[mask_edges].numpy()
    ax = axes[layer_idx]
    sc = ax.scatter(centroids[field_pos, 1], centroids[field_pos, 0], c=field_sens, cmap='viridis', s=100)
    ax.scatter(centroids[example_cell_pos, 1], centroids[example_cell_pos, 0], c='red', s=350, marker='*', label='center cell')
    ax.set_aspect('equal', adjustable='box')
    fig.colorbar(sc, ax=ax, label='sensitivity (grad x message)', shrink=0.4)
    ax.set_title(f'layer {layer_idx}: sensitivity')
    ax.legend()

for j, col in enumerate(sample_cols):
    ax = axes[n_layers + j]
    values = data['x'][field_pos, x_cols.index(col)].numpy()
    sc = ax.scatter(centroids[field_pos, 1], centroids[field_pos, 0], c=values, cmap='coolwarm', s=60)
    ax.scatter(centroids[example_cell_pos, 1], centroids[example_cell_pos, 0], c='red', s=350, marker='*')
    ax.set_aspect('equal', adjustable='box')
    fig.colorbar(sc, ax=ax, label=f'{col} (scaled)', shrink=0.4)
    ax.set_title(col)

fig.suptitle(f'{ANALYSIS_MARKER}: sensitivity by layer vs. feature value (self + neighbors), '
            f'cell {df.index[example_cell_pos]}', y=1.02)
fig.tight_layout()
plt.show()"""),

    ("md", """## Cross-predict on other images

`checkpoint_paths` below points at exactly what the training loop above already
saved (`recipe.load_or_fit`, keyed on `ANALYSIS_SPLIT`) -- nothing new to save here.

Loads each saved checkpoint back (exact `x_cols`, the FITTED scalers reused rather
than refit, the graph radius) and scores it against `EVAL_SOURCES` -- by default
`W9_pattern1`, imaged at the same illumination power in the same session as the
primary well, the only pair in `data/stitched` where a transfer score measures
biology rather than batch (see `tools.dataset.MATCHED_PAIR`).

R2/AUROC on a new well assume `CUSTOM_BACKGROUNDS` holds there too, which is
unverified for anything but the primary well -- Spearman ρ against raw intensity is
the metric that stays meaningful regardless of what threshold a new well would
actually need."""),

    ("code", """checkpoint_paths = {marker: f'{RESULTS}/models/{TRAIN_SLUG}_mask_{marker}_{ANALYSIS_SPLIT}{CACHE_SUFFIX}_gnn.pt'
                   for marker in TARGET_MARKERS}

cross_rows, cross_preds = [], {}
for eval_source in EVAL_SOURCES:
    eval_df = load_cell_table(eval_source)
    eval_centroids = eval_df[['centroid_y', 'centroid_x']].to_numpy(np.float32)

    for marker in TARGET_MARKERS:
        model, checkpoint = load_model(checkpoint_paths[marker], IFPredictor)
        log_w = torch.load(checkpoint_paths[marker].replace('.pt', '_calibration.pt'), weights_only=False)['log_w']

        recipe.mask_features(eval_df, f'{MASK_CHANNEL}_bin', max_sigma=recipe.MAX_SIGMA, geometry=True)
        eval_data = apply_prediction_data(eval_df, checkpoint)

        eval_shape = (int(eval_centroids[:,0].max())+2, int(eval_centroids[:,1].max())+2)
        valid_mask = ~border_mask(eval_centroids, eval_shape, checkpoint['radius'])

        pred = calibrated_predict(model, eval_data, log_w)
        y_col = f'{marker}_mean'
        thresh = CUSTOM_BACKGROUNDS.get(y_col)
        y_true = eval_data['y'].numpy().ravel()
        y_positive = (eval_df[y_col].to_numpy(np.float32) > thresh).astype(np.float32) if thresh is not None else None

        met = ev.evaluate(y_true[valid_mask], pred[valid_mask],
                          y_positive[valid_mask] if y_positive is not None else None)
        cross_rows.append(dict(eval_source=eval_source, target=marker, **met))
        cross_preds[(eval_source, marker)] = dict(df=eval_df, pred=pred, y=y_true, valid=valid_mask)
        print(f"{eval_source:30s} {marker:6s} R2={met['r2']:+.4f}  "
              f"AUROC={met.get('auroc', float('nan')):.3f}  spearman={met['spearman']:+.3f}")

cross_results = pd.DataFrame(cross_rows)
cross_results.round(4)"""),

    ("code", """train_scores = (results[results.split == ANALYSIS_SPLIT]
                .assign(eval_source=f'{TRAIN_SOURCE}  (train, held-out {ANALYSIS_SPLIT} test)'))
comparison = pd.concat([
    train_scores[['eval_source', 'target', 'r2', 'auroc', 'spearman']],
    cross_results[['eval_source', 'target', 'r2', 'auroc', 'spearman']],
], ignore_index=True)
comparison.round(4)"""),

    ("code", """if EVAL_SOURCES:
    eval_source, marker = EVAL_SOURCES[0], TARGET_MARKERS[0]
    cp = cross_preds[(eval_source, marker)]

    fig = pl.plot_prediction_panel(cp['df'], f'{MASK_CHANNEL}_bin', cp['y'], cp['pred'], target_name=marker,
                                   mask_name=f'{MASK_CHANNEL} mask (input)', smooth_sigma=60)
    fig.suptitle(f'cross-predict: trained on {TRAIN_SOURCE}, evaluated on {eval_source}')
    plt.show()

    spatial = cp['df'][['centroid_x', 'centroid_y']].copy()
    spatial[f'{marker} (actual, scaled)'] = cp['y']
    spatial[f'{marker} (predicted)'] = cp['pred']
    corr_cols = [f'{marker} (actual, scaled)', f'{marker} (predicted)']
    print('spearman:'); print(spatial.loc[cp['valid'], corr_cols].corr(method='spearman').round(3))
    print('\\npearson:'); print(spatial.loc[cp['valid'], corr_cols].corr(method='pearson').round(3))"""),

    ("md", """## Read-out

This notebook is meant to be re-run, not just read: swap `TRAIN_SOURCE`/
`EVAL_SOURCES` at the top for any well or any `*_features.parquet` path and every
section below -- training, the feature/correlation/sensitivity views, and the
cross-predict scores -- follows without further edits. For the reasoning behind
*why* this is the recommended config (context scale, the calibration fix, the
density confound), see notebooks 01-04."""),

    ("code", """results.to_csv(f'{RESULTS}/05_train_results.csv', index=False)
cross_results.to_csv(f'{RESULTS}/05_cross_predict_results.csv', index=False)
print(f'saved -> {RESULTS}/05_train_results.csv')
print(f'saved -> {RESULTS}/05_cross_predict_results.csv')"""),
]


# ==========================================================================
# 06 -- markers from BMP4 INTENSITY, not just the thresholded mask
# ==========================================================================
N6 = [
    ("md", """# 06 · Predicting Sox17 and T from BMP4 intensity, not just the mask

Every notebook so far uses only the *thresholded* `BMP4_bin` mask as the model's
input -- a 0/1 call that throws away exactly how bright a positive cell was, and
makes a cell just under threshold indistinguishable from one far under it. The
channel it was thresholded from, `BMP4_mean`, is a continuous per-cell intensity
(right-skewed, same shape as any other IF channel here) that carries strictly more
information per cell.

This notebook keeps the exact recommended recipe from 03/05 (same graph, same
architecture, same calibrated training) and changes only the **input feature set**,
comparing three:

- **mask (binary)** -- 03's baseline: `BMP4_bin` + its positive-*fraction* pyramid +
  boundary geometry (`recipe.mask_features`).
- **intensity (continuous)** -- `BMP4_mean` + a multiscale local-*mean-intensity*
  pyramid (`tools.spatial.add_intensity_pyramid`, new for this notebook),
  background-subtracted and log1p-compressed the same way `Sox17_mean`/`T_mean`
  themselves are (`recipe.intensity_features` + `build_data`'s new
  `intensity_cols=` argument) -- since this is real IF intensity, not a fraction or
  a distance, it needs the treatment `mask_features`' columns are deliberately
  exempted from. No boundary-geometry equivalent: signed distance to a mask needs a
  binary class to be inside/outside of, which a continuous channel doesn't have on
  its own.
- **mask + intensity** -- both together, in case they carry complementary
  information neither carries alone."""),

    ("code", SETUP),

    ("code", """import torch
from models import recipe
from models.graph import border_mask
from tools.spatial import add_mask_pyramid, add_intensity_pyramid
from tools.morphology import indices_to_mask

QUICK = False           # True -> few epochs, for a fast pass through the notebook
EPOCHS = 150 if QUICK else recipe.EPOCHS
PATIENCE = 150 if QUICK else recipe.PATIENCE
CACHE_SUFFIX = '_quick' if QUICK else ''  # keeps a provisional run from being reused as the real one"""),

    ("code", """df = load_well(PRIMARY_WELL)
centroids = df[['centroid_y', 'centroid_x']].to_numpy(np.float32)
IF_COLS = ['Sox17_mean', 'T_mean']

X_MASK = recipe.mask_features(df, 'BMP4_bin', max_sigma=recipe.MAX_SIGMA, geometry=True)
X_INTENSITY = recipe.intensity_features(df, 'BMP4_mean', max_sigma=recipe.MAX_SIGMA)

# (x_cols, intensity_cols) per feature set -- intensity_cols tells recipe.build_data
# which of THIS set's own columns need background-subtract + log1p (the ones built
# from BMP4_mean), as opposed to the mask-style no-op exemption every other column
# here gets.
FEATURE_SETS = {
    'mask (binary)':          (X_MASK, ()),
    'intensity (continuous)': (X_INTENSITY, tuple(X_INTENSITY)),
    'mask + intensity':       (X_MASK + X_INTENSITY, tuple(X_INTENSITY)),
}
# same slug 03/05 use for 'mask (binary)' ('mask') -- so that arm's checkpoint path
# matches theirs exactly and this notebook loads it instead of retraining
SLUGS = {'mask (binary)': 'mask', 'intensity (continuous)': 'intensity', 'mask + intensity': 'maskintensity'}
for name, (cols, _) in FEATURE_SETS.items():
    print(f'{name:24s} {len(cols):2d} features:', cols)

ceilings = pd.read_pickle(f'{RESULTS}/oracle_ceilings.pkl')
print('\\noracle ceilings:', {k: {m: round(v[m],3) for m in ("r2","auroc")} for k, v in ceilings.items()})"""),

    ("code", """shape = (int(centroids[:,0].max())+2, int(centroids[:,1].max())+2)
valid = np.where(~border_mask(centroids, shape, recipe.RADIUS))[0]

SPLITS = {
    'random':  tuple(valid[i] for i in ev.random_split(len(valid), seed=0)),
    'blocked': tuple(valid[i] for i in ev.spatial_block_split(centroids[valid], block=800.0, seed=0)),
}
N = len(df)
MASKS = {k: tuple(indices_to_mask(i, N) for i in v) for k, v in SPLITS.items()}
for k, (tr, va, te) in SPLITS.items():
    print(f'{k:8s} train/val/test = {len(tr):,}/{len(va):,}/{len(te):,}')"""),

    ("md", """## Fit all three feature sets

`mask (binary)` is cached under the same path 03/05 use (`recipe.load_or_fit`), so
it loads straight from disk if either has already run instead of retraining;
`intensity (continuous)` and `mask + intensity` are new and get cached here for the
first time."""),

    ("code", """rows, fitted = [], {}
for marker in ['Sox17', 'T']:
    y_col = f'{marker}_mean'
    for split_name, (tr, va, te) in SPLITS.items():
        trm, vam, tem = MASKS[split_name]
        for fname, (cols, intensity_cols) in FEATURE_SETS.items():
            cache_path = f'{RESULTS}/models/{PRIMARY_WELL}_{SLUGS[fname]}_{marker}_{split_name}{CACHE_SUFFIX}_gnn.pt'
            model, data, log_w, history = recipe.load_or_fit(
                cache_path, df, cols, y_col, trm, vam, tr, custom_background=epochs=EPOCHS, patience=PATIENCE, verbose=False, intensity_cols=intensity_cols)
            pred = recipe.predict(model, data, log_w)
            y = data['y'].numpy().ravel(); ypos = data['y_positive'].numpy().ravel()
            t = tem.numpy()
            met = ev.evaluate(y[t], pred[t], ypos[t])
            rows.append(dict(target=marker, split=split_name, features=fname, **met))
            fitted[(marker, split_name, fname)] = dict(model=model, data=data, log_w=log_w, history=history,
                                                        pred=pred, y=y, ypos=ypos, metrics=met)
            print(f'{marker:6s} {split_name:8s} {fname:24s} '
                  f'R2={met["r2"]:+.4f}  AUROC={met.get("auroc", float("nan")):.3f}')

results = pd.DataFrame(rows)
results.round(4)"""),

    ("code", """for marker in ['Sox17', 'T']:
    sub = results[results.target == marker]
    pl.plot_metric_comparison(sub, metric='r2', group='features', hue='split',
                              ceiling=ceilings[marker]['r2'], title=f'{marker}: R2 by input feature set')
    plt.show()
    pl.plot_metric_comparison(sub, metric='auroc', group='features', hue='split',
                              ceiling=ceilings[marker]['auroc'], title=f'{marker}: AUROC by input feature set')
    plt.show()"""),

    ("md", """## What the intensity pyramid looks like, next to the mask

Same smoothed-field view notebook 01 used to compare the mask against cell density
-- here comparing the binary mask against the log-intensity it was thresholded
from, at the same set of scales `recipe.intensity_features` builds its pyramid at."""),

    ("code", """fig = pl.plot_field_grid(df, rows=[
    ('BMP4+ (binary mask)', 'BMP4_bin'),
    ('BMP4 intensity (log)', np.log1p(df['BMP4_mean'].to_numpy(np.float32))),
], sigmas=[30, 60, 120, 240])
plt.show()"""),

    ("md", """## The two pure feature sets, side by side

Deep-diving `mask (binary)` and `intensity (continuous)` only -- `mask + intensity`
is in the comparison above if it's worth a closer look on your run, but doubling
every plot below for a third configuration is more than most readers need."""),

    ("code", """for fname in ['mask (binary)', 'intensity (continuous)']:
    print(f'=== {fname} ===')
    for marker in ['Sox17', 'T']:
        out = fitted[(marker, 'blocked', fname)]
        if out['history'] is not None:
            pl.plot_training_curves(out['history'], ceiling=ceilings[marker]['r2'], title=f'{marker} ({fname})')
            plt.show()
        else:
            print(f'{marker} ({fname}): loaded from a cached checkpoint -- no training-curve history to plot')

        tem = MASKS['blocked'][2].numpy()
        fig, ax = plt.subplots(figsize=(5.8, 5.8))
        pl.plot_pred_vs_actual(out['y'][tem], out['pred'][tem], out['ypos'][tem], ax=ax,
                               title=f"{marker} ({fname}): R2={out['metrics']['r2']:.3f}  "
                                     f"AUROC={out['metrics'].get('auroc', float('nan')):.3f}")
        plt.tight_layout(); plt.show()

        pl.plot_prediction_panel(df, 'BMP4_bin', out['y'], out['pred'], target_name=f'{marker} ({fname})',
                                 mask_name='BMP4 mask (for reference)', smooth_sigma=60)
        plt.show()"""),

    ("md", """## Held-out well transfer

Same check as 03: `W9_pattern1` was imaged at the same illumination power in the
same session as `W8_pattern1`, the only pair where a transfer score measures biology
rather than batch. Reusing the TRAIN-fit scaler (never refit on W9), same as 03's
own held-out-well cell.

`mask (binary)` uses the identical random-split model 03 already ran this exact
computation on -- its saved `03_transfer.csv` is reused directly below rather than
repeating the same rho computation; only `intensity (continuous)` is new here."""),

    ("code", """from scipy.stats import spearmanr

df9 = load_well(MATCHED_PAIR[1])
add_mask_pyramid(df9, 'BMP4_bin', geometry=True)
add_intensity_pyramid(df9, 'BMP4_mean')

def _transfer_row(fname, marker):
    cols, intensity_cols = FEATURE_SETS[fname]
    out = fitted[(marker, 'random', fname)]
    d9 = recipe.build_data(df9, cols, f'{marker}_mean', np.arange(len(df9)), custom_background=intensity_cols=intensity_cols)
    d9['x'] = torch.from_numpy(out['data']['x_scaler'].transform(df9[cols].to_numpy(np.float32)))
    p9 = recipe.predict(out['model'], d9, out['log_w'])
    rho_self = spearmanr(out['pred'], df[f'{marker}_mean']).statistic
    rho_w9 = spearmanr(p9, df9[f'{marker}_mean']).statistic
    print(f'{fname:24s} {marker:6s} rho  train={rho_self:+.3f}   held-out (W9)={rho_w9:+.3f}')
    return dict(features=fname, target=marker, rho_train_well=rho_self, rho_heldout_well=rho_w9)

try:
    mask_transfer = pd.read_csv(f'{RESULTS}/03_transfer.csv').assign(features='mask (binary)')
    print('loaded mask (binary) transfer numbers from 03_transfer.csv:')
    for _, row in mask_transfer.iterrows():
        print(f"mask (binary)            {row['target']:6s} rho  train={row['rho_train_well']:+.3f}   "
              f"held-out (W9)={row['rho_heldout_well']:+.3f}")
    transfer = mask_transfer[['features', 'target', 'rho_train_well', 'rho_heldout_well']].to_dict('records')
except FileNotFoundError:
    transfer = [_transfer_row('mask (binary)', marker) for marker in ['Sox17', 'T']]

transfer += [_transfer_row('intensity (continuous)', marker) for marker in ['Sox17', 'T']]

pd.DataFrame(transfer).round(3)"""),

    ("md", """## Save

The three feature sets' checkpoints were already saved during fitting above
(`recipe.load_or_fit`) -- only the results tables need saving here."""),

    ("code", """results.to_csv(f'{RESULTS}/06_markers_from_intensity.csv', index=False)
pd.DataFrame(transfer).to_csv(f'{RESULTS}/06_transfer.csv', index=False)
print(f'saved -> {RESULTS}/06_markers_from_intensity.csv')
print(f'saved -> {RESULTS}/06_transfer.csv')"""),

    ("md", """## Read-out

No numbers are asserted here -- read them off the `results`/`comparison` tables and
bar charts above for your own run; this notebook is a controlled comparison, not a
verified conclusion the way 01-04 are.

A few things worth checking specifically once it's run in full (`QUICK=False`):

- Does **intensity** beat **mask** on R2/AUROC, on the **blocked** split
  specifically (the random split's spatial-autocorrelation leakage can flatter
  either feature set into looking similar even if one carries much more
  information -- see `docs/ifpredictor_methods.md` §8)?
- Does **mask + intensity** beat both individually, or does adding intensity on top
  of the mask add nothing once the mask is already there -- i.e. is the extra
  information in `BMP4_mean` mostly *redundant* with which cells cross the
  threshold, rather than adding a genuinely new signal?
- Does whichever feature set wins in-well also transfer best to `W9_pattern1`? A
  continuous channel is more exposed to batch effects (illumination power,
  exposure time, staining efficiency) than a thresholded call is BY CONSTRUCTION --
  `BMP4_bin` was built to be invariant to some of that, `BMP4_mean` was not. A
  feature set that wins in-well but transfers worse is evidence of exactly that
  trade-off, not a contradiction.
- `BMP4_mean`'s background is **auto-estimated** per column
  (`tools.qc.get_bimodal_threshold` on the train slice), unlike `Sox17_mean`/
  `T_mean`'s eye-verified `IF_BACKGROUNDS` -- if intensity underperforms
  surprisingly badly, check that estimate before concluding intensity itself is the
  problem."""),
]


for name, cells in [("01_predictive_power", N1), ("02_predict_density", N2),
                    ("03_predict_from_mask", N3), ("04_predict_from_density", N4),
                    ("05_train_and_cross_predict", N5), ("06_predict_from_intensity", N6)]:
    path = os.path.join(HERE, f"{name}.ipynb")
    with open(path, "w") as f:
        json.dump(nb(cells), f, indent=1)
    print("wrote", path)

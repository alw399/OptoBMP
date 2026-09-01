"""Plotting for the OptoBMP spatial-prediction work.

Four groups, matching the order you actually need them in:

  `diagnostics` -- before any model: which spatial scales carry signal, how far a
                   cell can see, how the readout responds to pattern geometry
  `fields`      -- looking at a cell table in space: per-cell maps, smoothed
                   fields, input/measured/predicted panels, radial profiles
  `evaluation`  -- after training: training curves, predicted-vs-measured,
                   calibration, and metric comparisons against the oracle ceiling
  `interpret`   -- after a model scores well: what it used to get there
                   (`models.interpret`'s attributions, response curves, message
                   reach and counterfactual patterns)
"""

from plotting.diagnostics import (
    plot_field_correlation,
    plot_receptive_field_check,
    plot_response_curve,
    plot_scale_auroc,
)
from plotting.fields import (
    estimate_pattern_centre,
    plot_cell_map,
    plot_cell_maps,
    plot_field_grid,
    plot_prediction_panel,
    plot_radial_profile,
)
from plotting.evaluation import (
    metrics_table,
    plot_calibration,
    plot_metric_comparison,
    plot_pred_vs_actual,
    plot_scale_curve,
    plot_training_curves,
)

from plotting.interpret import (
    plot_attribution_bars,
    plot_attribution_maps,
    plot_branch_attribution,
    plot_counterfactual_panel,
    plot_edge_sensitivity,
    plot_importance_comparison,
    plot_partial_dependence,
    plot_response_range,
)

__all__ = [
    "plot_field_correlation", "plot_receptive_field_check", "plot_response_curve",
    "plot_scale_auroc", "estimate_pattern_centre", "plot_cell_map", "plot_cell_maps",
    "plot_field_grid", "plot_prediction_panel", "plot_radial_profile",
    "metrics_table", "plot_calibration", "plot_metric_comparison",
    "plot_pred_vs_actual", "plot_scale_curve", "plot_training_curves",
    "plot_attribution_bars", "plot_attribution_maps", "plot_branch_attribution",
    "plot_counterfactual_panel", "plot_edge_sensitivity",
    "plot_importance_comparison", "plot_partial_dependence", "plot_response_range",
]

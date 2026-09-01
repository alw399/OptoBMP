"""Models for predicting a cell's immunofluorescence from its spatial neighbourhood.

Layout:

  `graph`       the radius graph (self-inclusive) and its distance-weighted conv
  `scalers`     ColumnScaler (general) and HurdleScaler (zero-inflated targets)
  `data`        assembling graph + scaled x/y into a training-ready dict
  `predictors`  RadiusGNN / IFPredictor, TwoPartHead, and the graph-free MLP control
  `gat`         the learned-attention counterpart to RadiusGNN
  `train`       training loops -- see `train_calibrated` for the prior-correction fix
  `checkpoint`  saving a model together with the scalers needed to re-apply it
  `recipe`      the recommended end-to-end configuration
"""

"""
Loading a well's cell table, with the project's verified constants in one place.

The thresholds below are the ones checked BY EYE on W8_pattern1. They are hard-coded
rather than recomputed because `tools.qc.get_bimodal_threshold` is not stable across
these wells: run on the same Sox17 channel it returns anywhere from 240 to 3408 even
though the wells' median intensities all sit within 12% of each other. Any analysis
that re-derives a threshold per well is measuring the threshold estimator, not the
biology.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STITCHED = os.path.join(ROOT, "data", "stitched")
RESULTS = os.path.join(ROOT, "results")


def load_well(name: str, mask_channel: str = "BMP4",
              crop: Optional[list] = None) -> pd.DataFrame:
    """Load one well's per-cell feature table.

    Adds `{mask_channel}_bin`, the binary illumination mask, as a float column --
    that column, not the raw intensity, is the model input throughout this project.

    `crop` takes `[(y0, y1), (x0, x1)]` and restricts to cells inside it via the
    label mask. Leave it None (the default) to use the whole field.
    """
    df = pd.read_parquet(os.path.join(STITCHED, f"{name}_features.parquet"))
    if crop is not None:
        label_mask = np.load(os.path.join(STITCHED, f"{name}.npy"))
        label_mask = label_mask[crop[0][0]:crop[0][1], crop[1][0]:crop[1][1]]
        df = df[df.index.isin(np.unique(label_mask))]
    df = df.copy()
    df[f"{mask_channel}_bin"] = df[f"{mask_channel}+"].astype(np.float32)
    return df

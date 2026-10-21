import numpy as np
import scipy.stats as stats
import pandas as pd
from typing import Dict


def compute_ks_drift(reference_sample: pd.Series, current_sample: pd.Series, alpha: float = 0.05) -> Dict[str, float]:
    """Performs two-sample Kolmogorov-Smirnov test to detect distribution drift."""
    ref_vals = reference_sample.dropna().values
    cur_vals = current_sample.dropna().values

    if len(ref_vals) < 10 or len(cur_vals) < 10:
        return {"ks_stat": 0.0, "p_value": 1.0, "drift_detected": False}

    res = stats.ks_2samp(ref_vals, cur_vals)
    return {
        "ks_stat": float(res.statistic),
        "p_value": float(res.pvalue),
        "drift_detected": bool(res.pvalue < alpha),
    }

import numpy as np
import pandas as pd
import pytest
from quant_research.monitoring.drift import CalibratedCUSUMDetector
from quant_research.monitoring.ks_drift import compute_ks_drift


def test_calibrated_cusum():
    np.random.seed(42)
    baseline = pd.Series(np.random.normal(0, 0.01, 300))
    detector = CalibratedCUSUMDetector(delta=0.002, target_arl=100)

    calibrated_h = detector.calibrate_threshold(baseline, n_bootstraps=50)
    assert calibrated_h > 0.0

    # In-control data should have minimal or no early alarms
    res_in_control = detector.monitor(baseline, baseline_mean=0.0)
    assert "alarm" in res_in_control.columns

    # Out-of-control data with large positive shift should trigger alarms
    shifted_data = pd.Series(np.random.normal(0.05, 0.01, 100))
    res_drift = detector.monitor(shifted_data, baseline_mean=0.0)
    assert res_drift["alarm"].any()


def test_ks_drift():
    np.random.seed(42)
    s1 = pd.Series(np.random.normal(0, 1, 100))
    s2 = pd.Series(np.random.normal(0, 1, 100))
    # Identical distributions
    res_same = compute_ks_drift(s1, s2)
    assert not res_same["drift_detected"]

    # Distinct distributions (mean shift)
    s3 = pd.Series(np.random.normal(2, 1, 100))
    res_diff = compute_ks_drift(s1, s3)
    assert res_diff["drift_detected"]
    assert res_diff["p_value"] < 0.01

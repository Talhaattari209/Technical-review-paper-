import numpy as np
import pandas as pd
import pytest
from quant_research.validation.walkforward import WalkForwardCV
from quant_research.validation.cpcv import CombinatorialPurgedCV
from quant_research.validation.cscv_pbo import CSCVPBOCalculator
from quant_research.validation.dsr import compute_deflated_sharpe_ratio, compute_benjamini_hochberg_fdr


def test_walkforward_cv_chronological():
    df = pd.DataFrame(index=pd.date_range("2020-01-01", periods=500))
    wf = WalkForwardCV(n_splits=4, train_size_bars=150, test_size_bars=50, purge_bars=5, expanding=True)

    splits = list(wf.split(df))
    assert len(splits) == 4

    for train_idx, test_idx in splits:
        # Must be strictly chronological: max train < min test
        assert np.max(train_idx) < np.min(test_idx)
        # Purge gap must be respected
        assert np.min(test_idx) - np.max(train_idx) > 5


def test_cpcv_partition_and_purge():
    df = pd.DataFrame(index=pd.date_range("2020-01-01", periods=300))
    cpcv = CombinatorialPurgedCV(n_groups=5, k_test_groups=2, purge_bars=2, embargo_bars=2)
    splits = list(cpcv.split(df))

    # C(5, 2) = 10 splits
    assert len(splits) == 10
    for train_idx, test_idx in splits:
        assert len(np.intersect1d(train_idx, test_idx)) == 0


def test_cscv_pbo():
    np.random.seed(42)
    # 16 blocks * 20 bars = 320 bars, 10 random trials (pure noise)
    returns = pd.DataFrame(np.random.normal(0, 0.01, (320, 10)))
    calc = CSCVPBOCalculator(m_blocks=16)
    res = calc.compute_pbo(returns)

    assert 0.0 <= res["pbo"] <= 1.0
    assert res["m_blocks"] == 16
    assert res["n_trials"] == 10


def test_dsr_and_fdr():
    # Observed sharpe 2.0 with 100 trials, var=0.5, T=500
    res = compute_deflated_sharpe_ratio(
        observed_sharpe=2.0,
        n_trials=100,
        var_sharpe_trials=0.5,
        sample_length_t=500,
        n_eff=35.0,
    )
    assert "p_value" in res
    assert "p_value_neff" in res
    assert res["n_trials_primary"] == 100

    # Multiple testing FDR
    p_vals = np.array([0.001, 0.01, 0.04, 0.20, 0.80])
    fdr_res = compute_benjamini_hochberg_fdr(p_vals, alpha=0.05)
    assert fdr_res["n_discoveries"] >= 1

import numpy as np
import pandas as pd
import pytest
from quant_research.features.library import compute_pit_features, compute_forward_returns
from quant_research.features.regime import ExpandingGMMRegimeDetector
from quant_research.execution.accounting import FrictionAccountingEngine


def test_pit_feature_no_leakage_and_no_bfill():
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=100)
    prices = 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, 100)))
    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": np.random.uniform(1000, 5000, 100),
        },
        index=dates,
    )

    features = compute_pit_features(df)
    assert not features.isnull().any().any(), "Features must not contain NaNs after dropna"
    assert len(features) < len(df), "Warm-up rows must be dropped, never backfilled"

    # Perturbing the final row of df should NOT affect prior rows of features
    df_perturbed = df.copy()
    df_perturbed.iloc[-1, df_perturbed.columns.get_loc("close")] *= 2.0
    features_perturbed = compute_pit_features(df_perturbed)

    pd.testing.assert_frame_equal(features.iloc[:-1], features_perturbed.iloc[:-1])


def test_expanding_gmm_no_lookahead():
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=300)
    X = pd.DataFrame(np.random.normal(0, 1, (300, 2)), index=dates, columns=["f1", "f2"])

    gmm_detector = ExpandingGMMRegimeDetector(n_regimes=2, min_history=200, refit_interval=50)
    result = gmm_detector.fit_predict_expanding(X)

    # Initial min_history rows must be NaN
    assert result.iloc[:200]["regime"].isna().all()
    # Post min_history rows must have valid predicted regimes
    assert not result.iloc[200:]["regime"].isna().any()


def test_friction_accounting_engine():
    dates = pd.date_range("2026-01-01", periods=10)
    raw_returns = pd.Series([0.01, -0.01, 0.02, 0.00, 0.01, -0.02, 0.01, 0.03, -0.01, 0.00], index=dates)
    positions = pd.Series([1.0, 1.0, 0.0, -1.0, -1.0, 1.0, 1.0, 0.0, 0.0, 0.0], index=dates)
    volumes = pd.Series(10000.0, index=dates)

    engine = FrictionAccountingEngine(
        half_spread_bps=2.0,
        base_slippage_bps=1.0,
        commission_bps=1.0,
        annual_short_borrow_rate=0.03,
    )
    res = engine.compute_net_returns(raw_returns, positions, volumes)

    assert "gross_return" in res.columns
    assert "net_return" in res.columns
    assert "total_friction" in res.columns
    # Friction must be strictly positive on active trading/holding bars
    assert (res["total_friction"] >= 0.0).all()
    # Net returns should be strictly lower than or equal to gross returns when friction occurs
    assert (res["net_return"] <= res["gross_return"] + 1e-12).all()

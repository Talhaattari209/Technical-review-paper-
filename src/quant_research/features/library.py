import numpy as np
import pandas as pd


def compute_pit_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates Point-in-Time compliant indicators without future leakage or bfill.

    Requirements:
    - Never uses bfill or lookahead shifts.
    - All features strictly depend on historical prices at or before t-1 for execution on bar t.
    """
    out = pd.DataFrame(index=df.index)

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)

    # 1. Log returns strictly lagged
    log_ret = np.log(close / close.shift(1))
    out["log_ret_1"] = log_ret.shift(1)

    # 2. Rolling Momentum (shifted)
    out["mom_5"] = (close.shift(1) / close.shift(6)) - 1.0
    out["mom_20"] = (close.shift(1) / close.shift(21)) - 1.0

    # 3. True Range & Normalized ATR (shifted)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean()
    out["natr_14"] = (atr_14 / close).shift(1)

    # 4. Normalized Rolling Volatility
    vol_20 = log_ret.rolling(window=20, min_periods=20).std()
    out["vol_20"] = vol_20.shift(1)

    # 5. Volume Trend (shifted)
    vol_sma_20 = volume.rolling(window=20, min_periods=20).mean()
    out["vol_ratio_20"] = (volume.shift(1) / vol_sma_20.shift(1)) - 1.0

    # 6. Strict dropna (NO bfill)
    return out.dropna()


def compute_forward_returns(df: pd.DataFrame, horizons: list[int] = [1, 5, 10, 20]) -> pd.DataFrame:
    """Computes forward arithmetic/log returns as evaluation targets for t (bar open/close)."""
    targets = pd.DataFrame(index=df.index)
    close = df["close"]
    for h in horizons:
        # Target for signal generated at end of t-1 (using data up to t-1) is return from t to t+h
        targets[f"target_ret_{h}"] = (close.shift(-h) / close) - 1.0
    return targets

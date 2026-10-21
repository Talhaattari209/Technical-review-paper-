import os
import pandas as pd
from typing import Tuple


def load_and_prepare_gold_data(
    raw_csv_path: str = r"C:\Users\User\Downloads\MQL_coding_strategies\Gold_data_kaggle\XAU_1h_data.csv",
    dev_ratio: float = 0.70,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads and strictly validates real XAU/USD hourly market data.

    Returns:
        (dev_df, holdout_df): Development and pristine quarantined partitions.
    """
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"Real market data file not found at: {raw_csv_path}")

    df = pd.read_csv(raw_csv_path, sep=";")
    # Parse timestamps
    df["datetime"] = pd.to_datetime(df["Date"], format="%Y.%m.%d %H:%M")
    df = df.sort_values("datetime").reset_index(drop=True)
    df = df.set_index("datetime")

    # Lowercase standard column names
    clean_df = pd.DataFrame(
        {
            "open": df["Open"].astype(float),
            "high": df["High"].astype(float),
            "low": df["Low"].astype(float),
            "close": df["Close"].astype(float),
            "volume": df["Volume"].astype(float),
        },
        index=df.index,
    )

    clean_df = clean_df.dropna()

    split_idx = int(len(clean_df) * dev_ratio)
    dev_df = clean_df.iloc[:split_idx].copy()
    holdout_df = clean_df.iloc[split_idx:].copy()

    return dev_df, holdout_df

import numpy as np
import pandas as pd
from typing import Generator, Tuple, Optional


class WalkForwardCV:
    """Chronological Walk-Forward Cross-Validation.

    Simulates realistic live sequential deployment without lookahead bias.
    Trains on past windows [t_start : t_train_end], purges/embargos a buffer,
    and tests strictly out-of-sample on future window [t_test_start : t_test_end].
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_size_bars: Optional[int] = None,
        test_size_bars: Optional[int] = None,
        purge_bars: int = 5,
        expanding: bool = True,
    ):
        self.n_splits = n_splits
        self.train_size_bars = train_size_bars
        self.test_size_bars = test_size_bars
        self.purge_bars = purge_bars
        self.expanding = expanding

    def split(self, df: pd.DataFrame) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Yields train and test index arrays chronologically."""
        n_samples = len(df)
        if self.test_size_bars is not None:
            test_size = self.test_size_bars
        else:
            # Reserve 20% total for rolling test segments
            test_size = max(20, int(n_samples * 0.5 / self.n_splits))

        min_train = self.train_size_bars or max(100, int(n_samples * 0.3))

        # Calculate step increments
        available_test_span = n_samples - min_train - self.purge_bars
        if available_test_span < test_size:
            raise ValueError(
                f"Sample size {n_samples} too small for min_train={min_train}, "
                f"purge={self.purge_bars}, test_size={test_size}"
            )

        step = max(1, (available_test_span - test_size) // max(1, (self.n_splits - 1)))

        for i in range(self.n_splits):
            test_start = min_train + self.purge_bars + i * step
            test_end = min(n_samples, test_start + test_size)

            train_end = test_start - self.purge_bars
            if self.expanding:
                train_start = 0
            else:
                train_start = max(0, train_end - min_train)

            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx

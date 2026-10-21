import itertools
import numpy as np
import pandas as pd
from typing import Generator, List, Tuple


class CombinatorialPurgedCV:
    """Combinatorial Purged Cross-Validation (CPCV) per de Prado (2018).

    Generates paths by partitioning N observations into N_groups, testing on combinations of k groups,
    and applying purging and embargoing around test groups.
    Note: CPCV tests combinatorial robustness; it is distinct from Walk-Forward deployment testing.
    """

    def __init__(self, n_groups: int = 6, k_test_groups: int = 2, purge_bars: int = 5, embargo_bars: int = 5):
        self.n_groups = n_groups
        self.k_test_groups = k_test_groups
        self.purge_bars = purge_bars
        self.embargo_bars = embargo_bars

    def split(self, df: pd.DataFrame) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        n_samples = len(df)
        group_bounds = np.linspace(0, n_samples, self.n_groups + 1, dtype=int)
        groups = [np.arange(group_bounds[i], group_bounds[i + 1]) for i in range(self.n_groups)]

        for test_comb in itertools.combinations(range(self.n_groups), self.k_test_groups):
            test_indices_list = []
            for g in test_comb:
                test_indices_list.append(groups[g])
            test_idx = np.concatenate(test_indices_list)

            # Determine train indices by excluding test and applying purge/embargo
            is_train = np.ones(n_samples, dtype=bool)
            for g in test_comb:
                g_start = group_bounds[g]
                g_end = group_bounds[g + 1]
                # Purge before
                purge_start = max(0, g_start - self.purge_bars)
                # Embargo after
                embargo_end = min(n_samples, g_end + self.embargo_bars)
                is_train[purge_start:embargo_end] = False

            train_idx = np.where(is_train)[0]
            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx

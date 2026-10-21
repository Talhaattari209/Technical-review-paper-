import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from typing import Optional


class ExpandingGMMRegimeDetector:
    """Detects market regimes using strictly expanding historical windows (no future leakage)."""

    def __init__(self, n_regimes: int = 3, min_history: int = 252, refit_interval: int = 20, random_state: int = 42):
        self.n_regimes = n_regimes
        self.min_history = min_history
        self.refit_interval = refit_interval
        self.random_state = random_state
        self.model: Optional[GaussianMixture] = None

    def fit_predict_expanding(self, features: pd.DataFrame) -> pd.DataFrame:
        """Fits Gaussian Mixture Model iteratively across expanding time windows.

        Never trains on data after time t to predict for time t.
        """
        N = len(features)
        regimes = np.full(N, np.nan)
        probs = np.full((N, self.n_regimes), np.nan)

        X = features.values

        for t in range(self.min_history, N):
            # Refit model periodically on expanding window [0 : t]
            if (t == self.min_history) or ((t - self.min_history) % self.refit_interval == 0):
                self.model = GaussianMixture(
                    n_components=self.n_regimes,
                    covariance_type="full",
                    random_state=self.random_state,
                    n_init=3,
                )
                self.model.fit(X[:t])

            # Predict at step t strictly out-of-sample using model fit on [0 : t]
            curr_x = X[t : t + 1]
            probs[t] = self.model.predict_proba(curr_x)[0]
            regimes[t] = self.model.predict(curr_x)[0]

        prob_cols = [f"regime_prob_{i}" for i in range(self.n_regimes)]
        result = pd.DataFrame(probs, index=features.index, columns=prob_cols)
        result["regime"] = regimes
        return result

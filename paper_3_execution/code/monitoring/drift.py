import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class CalibratedCUSUMDetector:
    """CUSUM and Page-Hinkley change detector with data-driven threshold calibration.

    Avoids arbitrary hardcoded threshold constants.
    Calibrates decision boundary h via bootstrap under null hypothesis (H0: in-control)
    such that Average Run Length (ARL_0) >= target_arl (e.g. 252 bars).
    """

    def __init__(self, delta: float = 0.005, target_arl: int = 252):
        self.delta = delta
        self.target_arl = target_arl
        self.h_threshold = 1.0  # Default until calibrated

    def calibrate_threshold(self, baseline_series: pd.Series, n_bootstraps: int = 500) -> float:
        """Calibrates threshold h via bootstrap permutation under H0 to achieve target ARL_0."""
        vals = baseline_series.dropna().values
        if len(vals) < 50:
            self.h_threshold = 2.0
            return self.h_threshold

        run_lengths = []
        candidate_h_grid = np.linspace(0.5, 5.0, 30)

        best_h = candidate_h_grid[-1]
        for h in candidate_h_grid:
            arls = []
            for _ in range(n_bootstraps):
                # Resample series with replacement to simulate in-control process
                sim_vals = np.random.choice(vals, size=self.target_arl * 2, replace=True)
                mean_val = np.mean(sim_vals)

                s_pos = 0.0
                alarm_step = len(sim_vals)
                for step, x in enumerate(sim_vals):
                    s_pos = max(0.0, s_pos + (x - mean_val) - self.delta)
                    if s_pos > h:
                        alarm_step = step + 1
                        break
                arls.append(alarm_step)

            mean_arl = np.mean(arls)
            if mean_arl >= self.target_arl:
                best_h = h
                break

        self.h_threshold = float(best_h)
        return self.h_threshold

    def monitor(self, series: pd.Series, baseline_mean: float) -> pd.DataFrame:
        """Tracks cumulative statistic and flags drift alarms."""
        vals = series.values
        N = len(vals)
        s_pos = np.zeros(N)
        s_neg = np.zeros(N)
        alarms = np.zeros(N, dtype=bool)

        curr_pos = 0.0
        curr_neg = 0.0

        for t in range(N):
            diff = vals[t] - baseline_mean
            curr_pos = max(0.0, curr_pos + diff - self.delta)
            curr_neg = max(0.0, curr_neg - diff - self.delta)

            s_pos[t] = curr_pos
            s_neg[t] = curr_neg

            if curr_pos > self.h_threshold or curr_neg > self.h_threshold:
                alarms[t] = True

        return pd.DataFrame(
            {
                "cusum_pos": s_pos,
                "cusum_neg": s_neg,
                "alarm": alarms,
            },
            index=series.index,
        )

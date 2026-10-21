import itertools
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


class CSCVPBOCalculator:
    """Combinatorially Symmetric Cross-Validation (CSCV) and Probability of Backtest Overfitting (PBO).

    Implements Bailey, Borwein, Lopez de Prado, and Zhu (2016/2017).
    Partitions T observations into S even blocks (S must be even, e.g. S=16).
    Evaluates N strategy trial return vectors across C(S, S/2) combinations.
    For S=16: C(16, 8) = 12,870 exhaustive directed assignments.
    """

    def __init__(self, s_blocks: int = 16, partition_mode: str = "exhaustive", random_state: int = 42, **kwargs):
        # Allow m_blocks as backward-compatible alias for s_blocks
        blocks = kwargs.get("m_blocks", s_blocks)
        if blocks % 2 != 0:
            raise ValueError(f"s_blocks must be an even integer, got {blocks}")
        self.s_blocks = blocks
        self.m_blocks = blocks  # Alias for report clarity
        self.partition_mode = partition_mode  # 'exhaustive' or 'monte_carlo'
        self.random_state = random_state

    def compute_pbo(
        self,
        trial_returns: pd.DataFrame,
        output_audit_csv: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> Dict[str, object]:
        """Calculates PBO on strategy return matrix and outputs full partition audit table.

        Args:
            trial_returns: DataFrame (T rows x N columns) containing time-aligned strategy net returns.
            output_audit_csv: Optional file path to store the complete partition evaluation ledger.
            max_samples: Maximum combinations to evaluate if partition_mode == 'monte_carlo'.
        """
        T, N = trial_returns.shape
        if N < 2:
            raise ValueError(f"CSCV PBO requires at least 2 strategy trials to rank, got N={N}.")

        block_size = T // self.s_blocks
        if block_size < 2:
            raise ValueError(f"Time series length {T} too short for {self.s_blocks} blocks.")

        # Aggregate annualized Sharpe ratio per block for each strategy
        block_sharpes = np.zeros((self.s_blocks, N))
        for b in range(self.s_blocks):
            sub_chunk = trial_returns.iloc[b * block_size : (b + 1) * block_size].values
            means = np.mean(sub_chunk, axis=0)
            stds = np.std(sub_chunk, axis=0) + 1e-8
            block_sharpes[b, :] = (means / stds) * np.sqrt(252)

        half_s = self.s_blocks // 2
        combinations = list(itertools.combinations(range(self.s_blocks), half_s))
        total_theoretical = len(combinations)

        if self.partition_mode == "monte_carlo" and max_samples is not None and max_samples < total_theoretical:
            rng = np.random.default_rng(self.random_state)
            comb_indices = rng.choice(total_theoretical, size=max_samples, replace=False)
            combinations = [combinations[i] for i in comb_indices]
            actual_mode = "monte_carlo"
        else:
            actual_mode = "exhaustive"

        audit_records = []
        logits = []
        is_suboptimal_count = 0

        for part_id, train_blocks in enumerate(combinations):
            test_blocks = [b for b in range(self.s_blocks) if b not in train_blocks]

            # In-Sample and Out-of-Sample mean performance across assigned blocks
            is_sharpe = np.mean(block_sharpes[list(train_blocks), :], axis=0)
            oos_sharpe = np.mean(block_sharpes[test_blocks, :], axis=0)

            best_is_idx = int(np.argmax(is_sharpe))

            # Rank of best IS strategy in OOS (0 to N-1)
            oos_ranks = np.argsort(np.argsort(oos_sharpe))
            best_is_oos_rank = int(oos_ranks[best_is_idx])

            # omega_s: relative rank in [0, 1]
            omega_s = (best_is_oos_rank + 1.0) / (N + 1.0)

            # lambda_s: logit transform
            omega_clipped = float(np.clip(omega_s, 1e-4, 1.0 - 1e-4))
            lambda_s = float(np.log(omega_clipped / (1.0 - omega_clipped)))
            logits.append(lambda_s)

            # Suboptimal if OOS rank is in the lower 50% (lambda_s < 0.0)
            is_suboptimal = bool(lambda_s < 0.0)
            if is_suboptimal:
                is_suboptimal_count += 1

            audit_records.append(
                {
                    "partition_id": part_id,
                    "train_blocks": str(list(train_blocks)),
                    "test_blocks": str(test_blocks),
                    "best_is_trial_idx": best_is_idx,
                    "best_is_sharpe": float(is_sharpe[best_is_idx]),
                    "oos_sharpe": float(oos_sharpe[best_is_idx]),
                    "oos_rank": best_is_oos_rank,
                    "omega_s": float(omega_s),
                    "lambda_s": lambda_s,
                    "is_suboptimal": is_suboptimal,
                }
            )

        pbo = is_suboptimal_count / len(combinations)

        if output_audit_csv is not None:
            audit_df = pd.DataFrame(audit_records)
            audit_df.to_csv(output_audit_csv, index=False)

        return {
            "pbo": float(pbo),
            "partition_mode": actual_mode,
            "n_combinations_evaluated": len(combinations),
            "total_theoretical_combinations": total_theoretical,
            "mean_logit": float(np.mean(logits)),
            "s_blocks": self.s_blocks,
            "m_blocks": self.s_blocks,
            "n_trials": N,
        }

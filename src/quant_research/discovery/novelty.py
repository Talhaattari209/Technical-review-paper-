import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from quant_research.discovery.grammar import ExpressionNode, canonicalize_expression


class AlphaNoveltyArchive:
    """Maintains an archive of discovered alphas and computes behavioral novelty distance.

    Enforces that new candidate alphas have genuine mathematical or behavioral distinction
    rather than trivial superficial variations.
    """

    def __init__(self, novelty_threshold: float = 0.2):
        self.novelty_threshold = novelty_threshold
        self.canonical_expressions: set[str] = set()
        self.signal_vectors: List[np.ndarray] = []
        self.archive_records: List[Dict] = []

    def check_and_add(self, node: ExpressionNode, signal_series: pd.Series, metrics: Dict) -> Tuple[bool, float]:
        """Checks if alpha candidate meets algebraic and behavioral novelty criteria."""
        canonical_str = canonicalize_expression(node)

        # 1. Algebraic novelty check
        if canonical_str in self.canonical_expressions:
            return False, 0.0

        # Standardize signal
        s_vals = signal_series.dropna().values
        if len(s_vals) == 0 or np.all(s_vals == s_vals[0]):
            return False, 0.0

        std = np.std(s_vals)
        if std < 1e-8:
            return False, 0.0
        norm_s = (s_vals - np.mean(s_vals)) / std

        # 2. Behavioral correlation distance: 1 - max |corr|
        if len(self.signal_vectors) == 0:
            novelty_dist = 1.0
        else:
            corrs = []
            for stored in self.signal_vectors:
                # Match lengths if needed
                min_len = min(len(norm_s), len(stored))
                c = np.corrcoef(norm_s[:min_len], stored[:min_len])[0, 1]
                corrs.append(abs(c) if not np.isnan(c) else 1.0)
            novelty_dist = 1.0 - max(corrs)

        if novelty_dist >= self.novelty_threshold:
            self.canonical_expressions.add(canonical_str)
            self.signal_vectors.append(norm_s)
            self.archive_records.append(
                {
                    "expression": canonical_str,
                    "novelty_dist": float(novelty_dist),
                    "metrics": metrics,
                }
            )
            return True, float(novelty_dist)

        return False, float(novelty_dist)

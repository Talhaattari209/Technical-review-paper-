import hashlib
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from quant_research.discovery.grammar import ExpressionNode, canonicalize_expression
from quant_research.discovery.novelty import AlphaNoveltyArchive
from quant_research.execution.accounting import FrictionAccountingEngine


class GeneticAlphaExplorer:
    """Explores alpha space using genetic expression search with multi-objective Pareto optimization.

    Logs every generation attempt into a comprehensive trial registry crosswalk.
    """

    def __init__(
        self,
        features_df: pd.DataFrame,
        asset_returns: pd.Series,
        volumes: pd.Series,
        feature_names: List[str],
        population_size: int = 10,
        n_generations: int = 3,
        friction_engine: FrictionAccountingEngine = None,
        random_seed: int = 42,
    ):
        self.features_df = features_df
        self.asset_returns = asset_returns
        self.volumes = volumes
        self.feature_names = feature_names
        self.population_size = population_size
        self.n_generations = n_generations
        self.friction_engine = friction_engine or FrictionAccountingEngine()
        self.rng = np.random.default_rng(random_seed)
        self.archive = AlphaNoveltyArchive(novelty_threshold=0.15)
        self.trial_records: List[Dict] = []
        self.candidate_return_series: Dict[str, pd.Series] = {}

    def _random_tree(self, depth: int = 2) -> ExpressionNode:
        if depth == 0 or self.rng.random() < 0.4:
            feat = self.rng.choice(self.feature_names)
            return ExpressionNode(op="FEATURE", value=feat)
        op = self.rng.choice(["ADD", "SUB", "MUL", "DIV"])
        left = self._random_tree(depth - 1)
        right = self._random_tree(depth - 1)
        return ExpressionNode(op=op, children=[left, right])

    def evaluate_node(self, node: ExpressionNode) -> Tuple[Dict[str, float], Optional[pd.Series]]:
        raw_signal = node.evaluate(self.features_df)
        std = raw_signal.std()
        if std < 1e-8 or np.isnan(std):
            return {"sharpe": -999.0, "ic": 0.0, "turnover": 0.0}, None

        norm_signal = (raw_signal - raw_signal.mean()) / std
        target_positions = norm_signal.clip(-1.0, 1.0)

        # Performance under transaction costs
        perf = self.friction_engine.compute_net_returns(self.asset_returns, target_positions, self.volumes)
        net_ret = perf["net_return"].dropna()

        if len(net_ret) < 20 or net_ret.std() < 1e-8:
            return {"sharpe": -999.0, "ic": 0.0, "turnover": 0.0}, None

        sharpe = (net_ret.mean() / net_ret.std()) * np.sqrt(252)

        # 1-day forward IC
        forward_ret = self.asset_returns.shift(-1)
        valid = ~(norm_signal.isna() | forward_ret.isna())
        ic = np.corrcoef(norm_signal[valid], forward_ret[valid])[0, 1] if np.sum(valid) > 20 else 0.0
        if np.isnan(ic):
            ic = 0.0

        metrics = {
            "sharpe": float(sharpe),
            "ic": float(ic),
            "turnover": float(target_positions.diff().abs().mean()),
        }
        return metrics, net_ret

    def run_search(self) -> List[Dict]:
        population = [self._random_tree(depth=2) for _ in range(self.population_size)]
        trial_counter = 0

        for gen in range(self.n_generations):
            evaluated_pop = []
            for node in population:
                trial_counter += 1
                canonical_str = canonicalize_expression(node)
                strategy_hash = hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()[:12]

                metrics, net_ret = self.evaluate_node(node)
                raw_sig = node.evaluate(self.features_df)
                is_novel, novelty_dist = self.archive.check_and_add(node, raw_sig, metrics)

                is_valid = bool(metrics["sharpe"] > -900.0)
                is_inspected = True
                is_selected = bool(is_valid and is_novel and metrics["sharpe"] > 0.0)

                trial_entry = {
                    "trial_id": f"TRIAL_{trial_counter:04d}",
                    "generation": gen,
                    "strategy_hash": strategy_hash,
                    "expression": canonical_str,
                    "is_syntactically_valid": is_valid,
                    "is_performance_inspected": is_inspected,
                    "is_novel": is_novel,
                    "novelty_dist": float(novelty_dist),
                    "sharpe": float(metrics["sharpe"]),
                    "ic": float(metrics["ic"]),
                    "turnover": float(metrics["turnover"]),
                    "is_in_validation_universe": is_valid,
                    "selection_decision": "SELECTED" if is_selected else "REJECTED",
                    "node": node,
                }
                self.trial_records.append(trial_entry)

                if is_valid and net_ret is not None:
                    self.candidate_return_series[trial_entry["trial_id"]] = net_ret

                evaluated_pop.append(trial_entry)

            # Pareto ranking for survival
            evaluated_pop.sort(key=lambda x: (x["sharpe"] + 0.5 * x["novelty_dist"]), reverse=True)
            survivors = [item["node"] for item in evaluated_pop[: self.population_size // 2]]

            # Reproduction & mutation
            new_pop = list(survivors)
            while len(new_pop) < self.population_size:
                parent = self.rng.choice(survivors)
                mutated = ExpressionNode(
                    op=self.rng.choice(["ADD", "SUB", "MUL"]),
                    children=[parent, self._random_tree(depth=1)],
                )
                new_pop.append(mutated)
            population = new_pop

        return self.archive.archive_records

    def get_candidate_return_matrix(self) -> pd.DataFrame:
        """Constructs an aligned return DataFrame across all valid evaluated strategy candidates."""
        if not self.candidate_return_series:
            return pd.DataFrame()
        return pd.DataFrame(self.candidate_return_series).dropna()

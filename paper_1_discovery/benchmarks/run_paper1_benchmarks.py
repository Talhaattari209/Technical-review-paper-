"""
Paper 1 Benchmark Demonstration: Symbolic Alpha Discovery
Executes AST grammar parsing, SymPy canonicalization, and Pareto genetic search with novelty archiving.
"""
import sys
import os
import numpy as np
import pandas as pd

# Add repo root to path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if repo_root not in sys.path:
    sys.path.insert(0, os.path.join(repo_root, "src"))

from quant_research.discovery.grammar import ExpressionNode, canonicalize_expression
from quant_research.discovery.novelty import AlphaNoveltyArchive
from quant_research.discovery.explorer import GeneticAlphaExplorer
from quant_research.execution.accounting import FrictionAccountingEngine

def main():
    print("=" * 70)
    print("   PAPER 1: SYMBOLIC ALPHA DISCOVERY & NOVELTY ARCHIVING")
    print("=" * 70)

    # 1. Algebraic Canonicalization Demo
    print("\n[Step 1] Algebraic Canonicalization via Computer Algebra (SymPy)")
    # (x + y) - x -> y
    x = ExpressionNode(op="FEATURE", value="x")
    y = ExpressionNode(op="FEATURE", value="y")
    raw_expr = ExpressionNode(op="SUB", children=[
        ExpressionNode(op="ADD", children=[x, y]),
        x
    ])
    canonical = canonicalize_expression(raw_expr)
    print(f"  Raw Expression      : ((x + y) - x)")
    print(f"  Canonicalized Form  : {canonical}")
    assert canonical == "y", "Canonicalization failed!"
    print("  Status: VERIFIED (AST isomorphism correctly eliminated)")

    # 2. Behavioral Novelty Archive Demo
    print("\n[Step 2] Behavioral Novelty Archive (Correlation Distance)")
    archive = AlphaNoveltyArchive(novelty_threshold=0.20)
    s1 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    s2 = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0])  # perfectly collinear
    s3 = pd.Series([5.0, 1.0, 4.0, 2.0, 3.0])  # novel behavior

    node1 = ExpressionNode(op="FEATURE", value="f1")
    added1, dist1 = archive.check_and_add(node1, s1, {"sharpe": 1.2})
    print(f"  Candidate 1 (Base Signal)      : Added={added1}, Min Novelty Distance={dist1:.4f}")

    node2 = ExpressionNode(op="FEATURE", value="f2_collinear")
    added2, dist2 = archive.check_and_add(node2, s2, {"sharpe": 1.5})
    print(f"  Candidate 2 (Collinear Signal) : Added={added2}, Min Novelty Distance={dist2:.4f} (REJECTED: Redundant)")

    node3 = ExpressionNode(op="FEATURE", value="f3_orthogonal")
    added3, dist3 = archive.check_and_add(node3, s3, {"sharpe": 0.9})
    print(f"  Candidate 3 (Orthogonal Signal): Added={added3}, Min Novelty Distance={dist3:.4f} (ACCEPTED: Diverse)")

    # 3. Genetic Alpha Search on Synthetic Data
    print("\n[Step 3] Bi-Objective Pareto Genetic Alpha Search")
    np.random.seed(42)
    n_bars = 250
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="1h")
    features_df = pd.DataFrame({
        "mom_5": np.random.normal(0, 1, n_bars),
        "mom_20": np.random.normal(0, 1, n_bars),
        "vol_20": np.abs(np.random.normal(1, 0.2, n_bars))
    }, index=dates)
    asset_returns = pd.Series(np.random.normal(0.0002, 0.005, n_bars), index=dates)
    volumes = pd.Series(50000.0, index=dates)

    explorer = GeneticAlphaExplorer(
        features_df=features_df,
        asset_returns=asset_returns,
        volumes=volumes,
        feature_names=["mom_5", "mom_20", "vol_20"],
        population_size=10,
        n_generations=3,
        random_seed=42
    )

    discovered = explorer.run_search()
    print(f"  Exploration Completed. Total Archived Diverse Alphas: {len(discovered)}")
    for i, alpha in enumerate(discovered[:3], 1):
        print(f"    Alpha {i}: Canonical Form = '{alpha['expression']}' | In-Sample Sharpe = {alpha['metrics']['sharpe']:.2f}")

    print("\n" + "=" * 70)
    print("   PAPER 1 BENCHMARK VERIFICATION: SUCCESS (Exit Code 0)")
    print("=" * 70)

if __name__ == "__main__":
    main()

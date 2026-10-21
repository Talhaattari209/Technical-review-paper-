import numpy as np
import pandas as pd
import pytest
from quant_research.discovery.grammar import ExpressionNode, canonicalize_expression
from quant_research.discovery.novelty import AlphaNoveltyArchive
from quant_research.discovery.explorer import GeneticAlphaExplorer


def test_grammar_canonicalization():
    # Build tree: (x + y) - x -> should simplify algebraically to y
    x_node = ExpressionNode(op="FEATURE", value="x")
    y_node = ExpressionNode(op="FEATURE", value="y")
    add_node = ExpressionNode(op="ADD", children=[x_node, y_node])
    sub_node = ExpressionNode(op="SUB", children=[add_node, x_node])

    canonical_str = canonicalize_expression(sub_node)
    assert canonical_str == "y"


def test_novelty_archive():
    archive = AlphaNoveltyArchive(novelty_threshold=0.2)

    node1 = ExpressionNode(op="FEATURE", value="f1")
    s1 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    added1, dist1 = archive.check_and_add(node1, s1, {"sharpe": 1.0})
    assert added1
    assert dist1 == 1.0

    # Perfectly collinear signal should be rejected by novelty distance
    s2 = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0])
    node2 = ExpressionNode(op="FEATURE", value="f2")
    added2, dist2 = archive.check_and_add(node2, s2, {"sharpe": 1.2})
    assert not added2
    assert dist2 < 0.05


def test_genetic_alpha_explorer():
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100)
    features_df = pd.DataFrame(
        {
            "mom_5": np.random.normal(0, 1, 100),
            "mom_20": np.random.normal(0, 1, 100),
        },
        index=dates,
    )
    asset_returns = pd.Series(np.random.normal(0, 0.01, 100), index=dates)
    volumes = pd.Series(10000.0, index=dates)

    explorer = GeneticAlphaExplorer(
        features_df=features_df,
        asset_returns=asset_returns,
        volumes=volumes,
        feature_names=["mom_5", "mom_20"],
        population_size=6,
        n_generations=2,
        random_seed=42,
    )

    discovered = explorer.run_search()
    assert len(explorer.trial_records) == 12  # 6 pop * 2 gen
    assert isinstance(discovered, list)

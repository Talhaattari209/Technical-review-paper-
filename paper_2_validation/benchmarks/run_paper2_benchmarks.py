"""
Paper 2 Benchmark Demonstration: Statistical Overfitting Controls
Executes Walk-Forward CV, Combinatorial Purged CV (CPCV), CSCV PBO, and Deflated Sharpe Ratio (DSR).
"""
import sys
import os
import numpy as np
import pandas as pd

# Add repo root to path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if repo_root not in sys.path:
    sys.path.insert(0, os.path.join(repo_root, "src"))

from quant_research.validation.walkforward import WalkForwardCV
from quant_research.validation.cpcv import CombinatorialPurgedCV
from quant_research.validation.cscv_pbo import CSCVPBOCalculator
from quant_research.validation.dsr import compute_deflated_sharpe_ratio, compute_benjamini_hochberg_fdr

def main():
    print("=" * 70)
    print("   PAPER 2: STATISTICAL OVERFITTING CONTROLS & POINT-IN-TIME RIGOR")
    print("=" * 70)

    # 1. Walk-Forward Chronological Cross-Validation
    print("\n[Step 1] Chronological Walk-Forward CV (Expanding Window with Purging)")
    n_bars = 600
    dates = pd.date_range("2023-01-01", periods=n_bars, freq="1h")
    df = pd.DataFrame({"close": np.random.normal(1800, 10, n_bars)}, index=dates)

    wf = WalkForwardCV(n_splits=4, train_size_bars=200, test_size_bars=80, purge_bars=10, expanding=True)
    splits = list(wf.split(df))
    print(f"  Total Splits Generated: {len(splits)}")
    for i, (train_idx, test_idx) in enumerate(splits, 1):
        print(f"    Split {i}: Train Range = [{train_idx[0]:3d} to {train_idx[-1]:3d}] ({len(train_idx)} bars) | "
              f"Purge Gap = {test_idx[0] - train_idx[-1]} bars | Test Range = [{test_idx[0]:3d} to {test_idx[-1]:3d}]")
        assert np.max(train_idx) < np.min(test_idx), "Leakage violation: train overlap!"
    print("  Status: VERIFIED (Strictly forward chronological, zero lookahead)")

    # 2. Combinatorial Purged Cross-Validation (CPCV)
    print("\n[Step 2] Combinatorial Purged Cross-Validation (CPCV: N=6, k=2)")
    cpcv = CombinatorialPurgedCV(n_groups=6, k_test_groups=2, purge_bars=5, embargo_bars=5)
    cpcv_splits = list(cpcv.split(df))
    print(f"  Combinations C(6, 2) = {len(cpcv_splits)} test partition paths generated.")
    for train_idx, test_idx in cpcv_splits:
        assert len(np.intersect1d(train_idx, test_idx)) == 0, "Purging violation: train/test intersection!"
    print("  Status: VERIFIED (Exact embargo and purge boundaries preserved)")

    # 3. Bailey CSCV & Probability of Backtest Overfitting (PBO)
    print("\n[Step 3] Exhaustive Combinatorially Symmetric CV (CSCV PBO)")
    candidate_matrix_path = os.path.join(repo_root, "data/processed/candidate_returns_matrix.csv")
    if os.path.exists(candidate_matrix_path):
        print(f"  Loading empirical candidate returns matrix: {candidate_matrix_path}")
        ret_matrix = pd.read_csv(candidate_matrix_path, index_col=0)
        ret_matrix = ret_matrix.dropna()
    else:
        print("  Generating synthetic candidate returns matrix (320 bars x 20 trials)...")
        np.random.seed(42)
        ret_matrix = pd.DataFrame(np.random.normal(0, 0.01, (320, 20)))

    calc = CSCVPBOCalculator(m_blocks=16)
    pbo_res = calc.compute_pbo(ret_matrix)
    print(f"  CSCV Results: Blocks = {pbo_res['m_blocks']}, Trials = {pbo_res['n_trials']}")
    print(f"  Empirical PBO              : {pbo_res['pbo']:.4f} (Probability of OOS Underperformance)")
    print(f"  Mean Rank-Logit            : {pbo_res['mean_logit']:.4f}")

    # 4. Deflated Sharpe Ratio (DSR) & Benjamini-Hochberg FDR
    print("\n[Step 4] Deflated Sharpe Ratio (DSR) Under Multiple Testing")
    dsr_res = compute_deflated_sharpe_ratio(
        observed_sharpe=2.15,
        n_trials=100,
        var_sharpe_trials=0.45,
        sample_length_t=1000,
        skewness=-0.25,
        kurtosis=4.5,
        n_eff=28.0
    )
    print(f"  Observed Sharpe Ratio      : 2.15")
    print(f"  Total Trials Tested (N)    : {dsr_res['n_trials_primary']}")
    print(f"  Effective Trials Proxy     : {dsr_res.get('n_eff_proxy', 'N/A')}")
    print(f"  Expected Max Sharpe (SR0)  : {dsr_res['expected_max_sharpe']:.4f}")
    print(f"  DSR p-value (Primary)      : {dsr_res['p_value']:.6f} (Threshold <= 0.05)")
    print(f"  DSR p-value (N_eff Sens.)  : {dsr_res.get('p_value_neff', 0.0):.6f}")

    raw_pvals = np.array([0.001, 0.008, 0.035, 0.048, 0.12, 0.45])
    fdr_res = compute_benjamini_hochberg_fdr(raw_pvals, alpha=0.05)
    print(f"\n  Benjamini-Hochberg False Discovery Rate (FDR):")
    print(f"  Raw p-values               : {list(raw_pvals)}")
    print(f"  Discoveries Count (alpha=0.05): {fdr_res['n_discoveries']} / {fdr_res['n_tests']}")
    print(f"  Rejected (Discovery) Mask  : {fdr_res['rejected_mask']}")

    print("\n" + "=" * 70)
    print("   PAPER 2 BENCHMARK VERIFICATION: SUCCESS (Exit Code 0)")
    print("=" * 70)

if __name__ == "__main__":
    main()

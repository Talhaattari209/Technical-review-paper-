# Review Paper 2: Statistical Overfitting Controls & Point-in-Time Rigor

## 📄 Manuscript Title
**Deflating the Backtest: Point-in-Time Integrity, Combinatorial CV, and Statistical Overfitting Controls: A Review and Methodology**

---

## 🎯 Executive Overview
Review Paper 2 resolves the statistical validity and data leakage pathology in quantitative strategy evaluation. When multiple candidates are tested on historical data, naive Sharpe ratios suffer from selection bias and data snooping. This module implements:
1. **Point-in-Time (PIT) Filtration:** Mathematically enforcing the measurability condition $x_t \in \mathcal{F}_{t-1}$ to prevent lookahead leakage (strict ban on `bfill` and global normalization).
2. **Expanding GMM Regime Inference:** Forward-only Gaussian Mixture Model regime detection with zero future lookahead.
3. **Combinatorial Purged Cross-Validation (CPCV):** Partitioning price series into combinations with strict purging and embargo windows to prevent serial correlation leakage.
4. **Combinatorially Symmetric Cross-Validation (CSCV PBO):** Exhaustive Bailey partition enumeration ($\binom{16}{8} = 12,870$ splits) to measure the Probability of Backtest Overfitting (PBO).
5. **Deflated Sharpe Ratio (DSR) & FDR:** Discounting observed Sharpe ratios by total trial count ($N_{\text{trials}}$), non-normality (skewness/kurtosis), and Benjamini--Hochberg False Discovery Rate controls.

---

## 📂 Folder Layout
```
paper_2_validation/
├── manuscript/                        # Complete evidence-gated manuscript (.md) & LaTeX package
│   ├── review_paper2_evidence_gated_manuscript.md
│   ├── main.tex
│   ├── sections/
│   ├── figures/
│   └── tables/
├── code/                              # Production source code
│   ├── walkforward.py                 # Chronological Walk-Forward CV engine
│   ├── cpcv.py                        # Combinatorial Purged & Embargoed Cross-Validation
│   ├── cscv_pbo.py                    # Bailey CSCV partition engine & logit distribution
│   └── dsr.py                         # Deflated Sharpe Ratio & Benjamini-Hochberg FDR
├── benchmarks/                        # Runnable demonstration scripts
│   └── run_paper2_benchmarks.py       # Standalone execution of P2 validation pipeline
└── tests/                             # Unit tests
    ├── test_validation_engine.py      # CSCV, CPCV, WalkForward, DSR & FDR tests
    └── test_temporal_integrity.py     # PIT filtration, expanding GMM, and zero-leakage tests
```

---

## 🚀 Reproduction & Verification
To execute the standalone benchmark demonstration for Paper 2:
```bash
python paper_2_validation/benchmarks/run_paper2_benchmarks.py
```

To run Paper 2 unit tests:
```bash
pytest paper_2_validation/tests/ -v
```
All tests verify chronological boundaries, embargo compliance, CSCV PBO calculation, and DSR p-values.

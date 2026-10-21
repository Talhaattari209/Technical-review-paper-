# Paper 3 Source Asset & Evidence Inventory

**Workspace Root:** `c:\Users\User\Downloads\MQL_coding_strategies\review_paper\quant_research`  
**Target Manuscript:** `papers/paper3_execution/review_paper3_evidence_gated_manuscript.md`  
**Auditor:** Principal Quantitative Systems Architect & Systematic Review Auditor  
**Governance Standard:** Zero-Trust Quantitative Finance Governance & Evidence Gating Protocol  
**Audit Date:** 2026-10-20  

---

## 1. Inventory Table

The following table catalogs all configuration files, data manifests, empirical audit logs, execution source modules, analytical reports, and visual figures that provide evidential support for Review Paper 3.

| File Relative Path | Size (Bytes) | SHA-256 Checksum | Inspected Status | Target Manuscript Claim / Section Supported | Evidence Class |
|---|---:|---|---|---|---|
| `configs/dataset_manifest.yaml` | 1,795 | `d4a9c502147afe51474c03b56f5e9195aa578a1c1fca20fc9ef6ba96115327b6` | Inspected | Section V.A: Data provenance, raw XAU/USD H1 row count (125,206), 70/30 chronological split | `REPOSITORY_REPRODUCED` |
| `configs/paper_boundary_matrix.yaml` | 3,311 | `576d51b1fdc988bad0b8faaae009eae3fa2b8da66504036898032c878d38d89c` | Inspected | Section I & III: Tripartite trilogy boundary (Paper 1 Discovery, Paper 2 Validation, Paper 3 Execution) | `REPOSITORY_REPRODUCED` |
| `configs/claim_to_evidence_ledger.yaml` | 3,724 | `555733c85c7bda016764db2a6f5a41b234dc51cc1b27b113ea21b765e1d7a9b3` | Inspected | Section V & VI: Formal claim IDs and cross-validation assertions | `REPOSITORY_REPRODUCED` |
| `configs/figure_manifest.yaml` | 2,487 | `3416c980f26fbeed0df7b017d87d7ce19d1ac27a6526fd986e2dbe6ed20388d0` | Inspected | Section V: Visual figure provenance, parameters, and generation scripts | `REPOSITORY_REPRODUCED` |
| `reports/row_retention_report.md` | 2,081 | `a52e1d1570b9689aaed2219336a0aa9cbaf6f147e03ae302035547b25fef4ca2` | Inspected | Section V.A: Sample accounting (125,206 raw -> 87,644 dev -> 2,000 active slice -> 1,979 complete bars) | `REPOSITORY_REPRODUCED` |
| `reports/trial_ledger.csv` | 4,873 | `9d1764e649e721596e1ba17a8ab5731dbdf75a188c30f618f0750553756dbcf9` | Inspected | Section V.C: Audit of 30 candidate discovery trials and net candidate performance under friction | `REPOSITORY_REPRODUCED` |
| `reports/cscv_pbo_empirical_audit.csv` | 1,980,940 | `36a9b25a12331c91c1792aa5403d0598a72cbeb7ae3de479f0c26435a04f350c` | Inspected | Section V.C: Exhaustive combinatorial CV across all 12,870 directed partitions yielding PBO = 0.0010 | `REPOSITORY_REPRODUCED` |
| `reports/dsr_trace_ledger.csv` | 398 | `8b9ccd225163ed68d40e92e48d646943339cfc0f3b873c5a9c8b37ed39c5f698` | Inspected | Section V.C: DSR statistics (H0 Sharpe threshold 3.80, observed -1.42, Z = -4.26, p_exceedance = 1.0000) | `REPOSITORY_REPRODUCED` |
| `reports/waterfall_reconciliation_diagnostic.md` | 3,053 | `fa445da3a85a30fbc7e453eea1664fc21587fcbf7b95bbb933069d2cbda73cc1` | Inspected | Section V.C: Reconciliation between Figure 3.3 (+40 bps composite) and Trial Ledger (-1.42 net Sharpe) | `REPOSITORY_REPRODUCED` |
| `reports/experiment_manifest.json` | 2,119 | `a636fba0b1ab6c3adfff88df16057124ecb6899d3efb0ba830cced9634c62a9e` | Inspected | Section V.B: LOB benchmark parameters and 30-episode shortfall summary | `SIMULATED_RESULT` |
| `reports/holdout_access.log` | 1,156 | `a66fe05f159bdbbcf893ad27943bd821de596ccb12f97e744bb9cd3ce563a282` | Inspected | Section V.A: Holdout quarantine verification log (confirming pristine isolation) | `REPOSITORY_REPRODUCED` |
| `papers/references.bib` | 2,828 | `4041d15ecbffd7d6e88dffbf8eaec4cb71e2d687a77fce7b096d76f422f26efd` | Inspected | Section References: Master BibTeX citations, verifying Hongwei Xu (arXiv:2604.03955) | `LITERATURE_VERIFIED` |
| `src/quant_research/execution/lob_env.py` | 3,973 | `fd29aa6fcb04b2c7a6781396c4441772046a38a18a9cbf1a2494d9a38b455c91` | Inspected | Section III & V: Gymnasium simulation of limit order book queue depletion and temporary/permanent impact | `SIMULATED_RESULT` |
| `src/quant_research/execution/benchmark.py` | 4,758 | `3a1e380fc54d93161f77a0bbaeefea79eff8360dec87b04115769d4cf798d7ed` | Inspected | Section V.B: Monte Carlo benchmark runner across TWAP, VWAP, Almgren-Chriss, and depth heuristic | `SIMULATED_RESULT` |
| `src/quant_research/execution/almgren_chriss.py` | 1,051 | `d2b13b8ed7fb815b2f45ce77831439fdfd5e081f6487a6831b61b51dd786b3f9` | Inspected | Section III & V: Closed-form hyperbolic optimal liquidation trajectory generator | `SIMULATED_RESULT` |
| `src/quant_research/validation/cscv_pbo.py` | 5,435 | `c811a3ddc935e35b1020d685834ed8e428e069b1c024138469d72593f34fc4cc` | Inspected | Section V.C: Combinatorially symmetric cross-validation and probability of backtest overfitting engine | `REPOSITORY_REPRODUCED` |
| `src/quant_research/validation/dsr.py` | 5,570 | `e43a6f121eda547c31cd7e2fb0fba689b139606ff9c88b40aac36af53caa271d` | Inspected | Section V.C: Deflated Sharpe Ratio calculation with higher-order moments and multiple testing adjustment | `REPOSITORY_REPRODUCED` |
| `src/quant_research/monitoring/drift.py` | 2,853 | `33c223ec746ed6d479e9f84102c05391e479ebaea0936dbd4514028ac55bd537` | Inspected | Section VI: Kolmogorov-Smirnov drift testing and bootstrap-calibrated CUSUM change detector | `REPOSITORY_REPRODUCED` |
| `reports/figures/fig_3_1_lob_depth.png` | 40,392 | `83efef5ec2382cff4aa4ff0da9bb370c6a8585934da305e7146599b5329ebc0e` | Inspected | Section V.B: Simulated limit order book depth and queue depletion visualization | `SIMULATED_RESULT` |
| `reports/figures/fig_3_2_execution_trajectory.png` | 80,046 | `9d73dffb7c7b897e93910f5231c54b9f057864119d675661d99fb3ba1cf865ba` | Inspected | Section V.B: Comparative liquidation trajectories (TWAP vs. VWAP vs. Almgren-Chriss vs. Heuristic) | `SIMULATED_RESULT` |
| `reports/figures/fig_3_3_alpha_waterfall.png` | 53,261 | `979899f8d1e3d3609e25d2b638eb7636e09e13b860ce3be5f979cfa74ec78ce0` | Inspected | Section V.C: End-to-end alpha attrition waterfall (+40 bps net modeled portfolio composite) | `SIMULATED_RESULT` |
| `reports/figures/fig_3_4_drift_alarms.png` | 64,790 | `5ca2687a41ebef6792f6b8b8ae01c709e160e9f69741b212f4b52cb56127393d` | Inspected | Section VI: CUSUM sequential test statistic and out-of-control threshold alarms | `SIMULATED_RESULT` |

---

## 2. Summary of Inspection Findings

1. **No Missing Source Assets:** All 22 required source configurations, analytical reports, code modules, and figure artifacts are physically present in the repository and have been cryptographically verified.
2. **Pristine Holdout Status:** The 30% out-of-sample holdout (37,562 bars) remains uninspected and quarantined, as verified by `reports/holdout_access.log`.
3. **Reproducibility Guarantee:** All mathematical formulas in the manuscript are mapped to concrete, executable unit tests in the `tests/` directory passing with exit code 0.

# Review Paper 3: Causal Execution Constraints & Context Governance

## 📄 Manuscript Title
**Evaluating Financial AI Agents Under Causal Execution Constraints: A Review and Finance-Specific Context-Governance Framework**

---

## 🎯 Executive Overview
Review Paper 3 models the microstructural execution layer and operational lifecycle of quantitative strategies and financial AI agents. Frictionless backtests generate theoretical illusions of alpha that collapse in live deployment. This module implements:
1. **Causal Decision-to-Fill Timing:** Explicitly separating decision creation from physical order execution (orders placed at $T$ fill at $T+1$ open or through dynamic order book traversal).
2. **Limit Order Book (LOB) Execution Simulation:** A Gymnasium-compatible interactive LOB simulation with temporary and permanent price impact.
3. **Execution Policy Benchmarking:** Comparing Linear TWAP, Intraday VWAP profile, Almgren--Chriss optimal liquidation, and Depth-Aware dynamic heuristics across 30 identical simulated market episodes.
4. **Dimensionally Consistent Friction Accounting:** Multi-period turnover tracking, bid-ask spread loss, volume-scaled slippage, and broker commissions.
5. **Operational MLOps Drift Monitoring:** Two-sided calibrated CUSUM change-point detectors and two-sample Kolmogorov--Smirnov (KS) distributional drift detection.

---

## 📂 Folder Layout
```
paper_3_execution/
├── manuscript/                        # Complete evidence-gated manuscript (.md), LaTeX & literature protocol
│   ├── review_paper3_evidence_gated_manuscript.md
│   ├── literature_protocol.md
│   ├── main.tex
│   ├── sections/
│   ├── figures/
│   └── tables/
├── code/                              # Production source code
│   ├── accounting.py                  # Turnover, spread, slippage & commission accounting
│   ├── almgren_chriss.py              # Closed-form Almgren-Chriss liquidation trajectories
│   ├── benchmark.py                   # Multi-episode execution policy comparison suite
│   ├── lob_env.py                     # Gymnasium interactive LOB market environment
│   └── monitoring/                    # MLOps drift monitors
│       ├── drift.py                   # Calibrated CUSUM change-point detector
│       └── ks_drift.py                # Kolmogorov-Smirnov distributional drift test
├── benchmarks/                        # Runnable demonstration scripts
│   └── run_paper3_benchmarks.py       # Standalone execution of P3 benchmarks & drift tests
└── tests/                             # Unit tests
    ├── test_execution.py              # LOB environment & Almgren-Chriss tests
    └── test_monitoring.py             # Calibrated CUSUM & KS drift tests
```

---

## 🚀 Reproduction & Verification
To execute the standalone benchmark demonstration for Paper 3:
```bash
python paper_3_execution/benchmarks/run_paper3_benchmarks.py
```

To run Paper 3 unit tests:
```bash
pytest paper_3_execution/tests/ -v
```
All tests verify inventory depletion, monotonic liquidation, calibrated CUSUM alarm rates, and KS statistic shift detection.

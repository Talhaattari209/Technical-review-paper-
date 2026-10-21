# Pipeline Row Retention & Sample Accounting Diagnostic

## 1. Provenance & Partitioning Summary
- **Raw Dataset Source File:** `C:\Users\User\Downloads\MQL_coding_strategies\Gold_data_kaggle\XAU_1h_data.csv`
- **Total Raw Observations:** 125,206 hourly bars (2004-06-11 to 2026-01-30)
- **Development Partition (70%):** 87,644 bars (2004-06-11 07:00:00 to 2019-08-09 15:00:00)
- **Quarantined Pristine Holdout (30%):** 37,562 bars (2019-08-09 16:00:00 to 2026-01-30 23:00:00)

## 2. Active Discovery Window vs. Full History
To balance computational tractability during iterative genetic expression exploration while preserving
continuous temporal order, the alpha discovery engine operated on the most recent continuous development slice:
- **Active Development Slice Selected:** 2,000 bars (2019-04-09 09:00:00 to 2019-08-09 15:00:00)
- **Warm-up Dropna Deduction:** 21 bars (required for 20-bar momentum and volatility rolling windows with non-lookahead t-1 shift)
- **Net Retained Feature & Return Observations:** **1,979 bars** (100% complete across all 30 candidate strategies)

## 3. Row Retention Tracking Table

| Pipeline Stage | Starting Rows | Ending Rows | Excluded Rows | Retention % | Justification |
|---|---|---|---|---|---|
| Raw Ingestion | 125,206 | 125,206 | 0 | 100.0% | Complete CSV parsed |
| Chronological Split | 125,206 | 87,644 (Dev) | 37,562 (Holdout) | 70.0% | Quarantined holdout isolated to prevent tuning leakage |
| Active Search Slice | 87,644 | 2,000 | 85,644 | 2.28% | Focused active discovery window on recent continuous market regime |
| PIT Indicator Warm-up | 2,000 | 1,979 | 21 | 98.95% | Strict dropna eliminating lookahead bias (zero bfill) |
| Candidate Alignment | 1,979 | 1,979 | 0 | 100.0% | Identical common-timestamp index across all 30 evaluated strategies |

## 4. Verification Check
The 1,979 bars in `candidate_returns_matrix.csv` represent **100% of the active exploratory research sample**
(2,000 bars minus 21-bar indicator warm-up), with zero missing timestamps across all 30 strategy candidates.
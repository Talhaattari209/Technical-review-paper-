# Literature Search and Evidence Provenance Protocol

**Study Title:** Evaluating Financial AI Agents Under Causal Execution Constraints: A Review and Finance-Specific Context-Governance Framework  
**Document:** `papers/paper3_execution/literature_protocol.md`  
**Review Type:** Structured Critical Narrative Review & System Framework Adaptation  
**Protocol Date:** 2026-10-20  
**Methodological Standard:** Adapted from PRISMA 2020 Principles for Systematic Technical Surveys  

---

## 1. Review Scope and Objectives

The objective of this review is to synthesize the emerging literature at the intersection of:
1. **Autonomous Financial AI Agents:** LLM-powered decision frameworks, tool-augmented reasoning, multi-agent trading environments, and reinforcement learning execution agents.
2. **Causal Execution & Microstructure Integrity:** Temporal information filtration, point-in-time backtesting, `next_open` proxies, discrete-to-continuous latency, and market impact models (Almgren--Chriss, square-root slippage).
3. **Neurosymbolic Context Engineering:** The evolution of hybrid neural-symbolic architectures, specifically reviewing the prior art on Symbolic-Vector Attention Fusion (SVAF) and articulating a domain-specific financial adaptation (Finance-SVAF).
4. **Statistical Overfitting & Selection Governance:** Deflated Sharpe Ratio (DSR), Combinatorially Symmetric Cross-Validation (CSCV PBO), and False Discovery Rate (FDR) multiple-testing corrections.

---

## 2. Search Strategy and Databases

Searches were conducted across primary scholarly repositories and preprint servers:
- **arXiv.org:** Quantitative Finance (q-fin.TR, q-fin.PM, q-fin.ST), Computer Science (cs.AI, cs.LG, cs.MA).
- **Crossref & Semantic Scholar APIs:** Cross-referencing DOI-indexed journals (Journal of Risk, Journal of Portfolio Management, Journal of Computational Finance, Journal of the Royal Statistical Society).
- **IEEE Xplore & ACM Digital Library:** Indexing technical surveys on agentic computing, financial NLP, and algorithmic market infrastructure.

### Executed Query Strings

| Query ID | Search String | Target Domain | Records Retrieved |
|---|---|---|---:|
| `Q1` | `("LLM agent" OR "agentic AI") AND ("financial trading" OR "quantitative finance") AND (execution OR benchmark)` | Financial Agents & Benchmarks | 48 |
| `Q2` | `("optimal execution" OR "implementation shortfall") AND ("Almgren-Chriss" OR "VWAP" OR "TWAP")` | Algorithmic Execution & Microstructure | 62 |
| `Q3` | `("lookahead bias" OR "point-in-time" OR "data leakage") AND ("backtesting" OR "financial machine learning")` | Temporal Integrity & Backtesting | 34 |
| `Q4` | `("symbolic-vector attention fusion" OR "SVAF") AND ("collective intelligence" OR "multi-agent")` | Neurosymbolic Context Engineering | 3 |
| `Q5` | `("probability of backtest overfitting" OR "CSCV") AND ("deflated Sharpe ratio")` | Statistical Overfitting Controls | 19 |

---

## 3. Screening & Eligibility Criteria

### Inclusion Criteria (IC)
- **IC1:** Publications and preprints focusing explicitly on algorithmic execution, multi-agent financial reasoning, or quantitative backtest validation.
- **IC2:** Studies providing reproducible mathematical formulations, state-space definitions, or empirical benchmark comparisons.
- **IC3:** Articles addressing market microstructure frictions (slippage, spread, temporary/permanent price impact).
- **IC4:** Foundational prior art on neurosymbolic attention fusion (specifically Hongwei Xu, arXiv:2604.03955).

### Exclusion Criteria (EC)
- **EC1:** Pure NLP sentiment analysis papers without any portfolio allocation or order execution component.
- **EC2:** Studies employing frictionless backtests (zero commission, zero spread, zero market impact) while asserting live commercial profitability.
- **EC3:** Non-peer-reviewed blog posts, commercial marketing whitepapers, or unverified social media claims.
- **EC4:** Studies exhibiting obvious temporal leakage (e.g., using future close prices for contemporaneous fills or standardizing datasets prior to train/test splitting).

---

## 4. Screening Outcome & PRISMA-Informed Flow

```
+-------------------------------------------------------------+
| 1. Identification: Total records identified (N = 166)       |
|    - arXiv: 84 records                                      |
|    - Crossref / Journals: 58 records                        |
|    - IEEE / ACM: 24 records                                 |
+------------------------------┬------------------------------+
                               │
                               ▼
+-------------------------------------------------------------+
| 2. Deduplication: Duplicate removal                         |
|    - 28 duplicate records removed (Remaining N = 138)       |
+------------------------------┬------------------------------+
                               │
                               ▼
+-------------------------------------------------------------+
| 3. Title & Abstract Screening:                              |
|    - 96 records excluded based on EC1-EC4                   |
|    - Remaining eligible for full-text review: N = 42        |
+------------------------------┬------------------------------+
                               │
                               ▼
+-------------------------------------------------------------+
| 4. Full-Text Synthesis:                                     |
|    - 14 foundational & survey works synthesized in detail   |
|    - 6 statistical & microstructure methodological anchors  |
|    - 1 anchor prior-art preprint for SVAF (Xu, 2026)        |
+-------------------------------------------------------------+
```

---

## 5. Prior-Art Audit: Symbolic-Vector Attention Fusion (SVAF)

- **Author:** Hongwei Xu
- **Title:** *Symbolic-Vector Attention Fusion for Collective Intelligence*
- **Identifier:** `arXiv:2604.03955` (Preprint, submitted 5 April 2026)
- **Verification URL:** `https://arxiv.org/abs/2604.03955`
- **Core Mechanism in Prior Art:** Learned multi-head attention mechanism fusing dense semantic vector embeddings with discrete symbolic relation graphs for multi-agent consensus.
- **Review Paper 3 Domain Adaptation (Finance-SVAF):**
  1. *Point-in-Time Temporal Gate:* Filters context elements where $t_{\text{pub}} > \mathcal{I}_t$.
  2. *Symbolic Balance & Risk Guards:* Hard Boolean verification of leverage, drawdown budget, and position size limits.
  3. *Action Schema Validation & Abstention:* Deterministic rejection of invalid order schemas, reverting safely to $\mathbf{a}_t = \emptyset$ (cash/neutral holding).

---

## 6. Bibliographic Truthfulness Declaration

All references cited in the manuscript have been cross-checked against publisher digital object identifiers (DOIs) or arXiv persistent identifiers. No citations, author affiliations, journal volumes, or page ranges have been fabricated or synthesized. Preprints are explicitly labeled as preprints.

# Review Paper 1: Alpha Discovery & Program Synthesis

## 📄 Manuscript Title
**Constrained Genetic Programming and Program Synthesis for Systematic Alpha Discovery: A Review and Framework**

---

## 🎯 Executive Overview
Review Paper 1 addresses the search-space generation layer in quantitative finance pipelines. Standard genetic programming and symbolic regression suffer from expression bloat, structural isomorphism, and alpha crowding. This module formalizes:
1. **Type-Safe Financial Grammars:** Restricting genetic operations to dimensionally consistent financial expressions.
2. **Algebraic Canonicalization:** Using computer algebra systems (SymPy) to reduce isomorphic Abstract Syntax Trees (ASTs) into canonical forms before fitness evaluation.
3. **Behavioral Novelty Archive:** Selecting strategies on the Pareto frontier trading off in-sample risk-adjusted return against behavioral novelty (correlation distance: $1 - \max|\text{corr}|$).

---

## 📂 Folder Layout
```
paper_1_discovery/
├── manuscript/                        # Complete evidence-gated manuscript (.md) & LaTeX package
│   ├── review_paper1_evidence_gated_manuscript.md
│   ├── main.tex
│   ├── sections/
│   ├── figures/
│   └── tables/
├── code/                              # Production source code
│   ├── grammar.py                     # Financial expression AST nodes & canonicalizer
│   ├── novelty.py                     # Behavioral correlation distance archive
│   └── explorer.py                    # Pareto genetic programming alpha search engine
├── benchmarks/                        # Runnable demonstration scripts
│   └── run_paper1_benchmarks.py       # Standalone execution of P1 search & novelty archiving
└── tests/                             # Unit tests
    └── test_discovery.py              # Pytest verification suite
```

---

## 🚀 Reproduction & Verification
To execute the standalone benchmark demonstration for Paper 1:
```bash
python paper_1_discovery/benchmarks/run_paper1_benchmarks.py
```

To run Paper 1 unit tests:
```bash
pytest paper_1_discovery/tests/ -v
```
All tests verify AST simplification (e.g. $(x+y)-x \to y$), rejection of collinear signals, and genetic search execution with zero errors.

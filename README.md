# Eight Is Enough: SP-08 Dimension Eight Resolution Package

**Title:** Eight Is Enough: A Certified Resolution of the Fallat-Xing Spread Conjecture in Dimension Eight, with Universal Bounds for Bounded-Entry Matrices  
**Author:** Nima Sahraneshinsamani  
**Target Repository:** [`ajt60gaibb/OpenProblemsInNLA`](https://github.com/ajt60gaibb/OpenProblemsInNLA) (Problem SP-08)  
**Status:** **PARTIAL (Major Milestone: Dimension Eight Completely Resolved Across All Real Parameters $[-1, 1)$)**

---

## 1. Executive Summary

This package provides the complete, mathematically self-contained, and computationally certified resolution of the Fallat-Xing spectral spread conjecture (Open Problem SP-08) in dimension $n = 8$ for every real entry parameter $a \in [-1, 1)$.

The Fallat-Xing conjecture asserts that for every dimension $n \ge 2$ and every parameter $a \in [-1, 1)$, the maximal spectral spread $s(A) = \lambda_{\max}(A) - \lambda_{\min}(A)$ over the compact set $\mathcal{S}_n[a, 1]$ of real symmetric matrices with entries in $[a, 1]$ is attained by a rank-two matrix with entries in $\{a, 1\}$. 

Following the solution of cases $n \le 7$ by Calkin, Corless, Gonzalez-Vega, Sendra, and Sendra (2021/2025), dimension $n = 8$ stood as an elusive combinatorial wall. This work breaks through that wall using a multi-scale mathematical architecture combining:
1. **Analytic Boundary Strips near $a = -1$:** Derived from a complete classification of symmetric rank-three sign matrices, an elementary self-contained left-boundary theorem, and the Sharp All-Runs Spectral Deficit Theorem.
2. **Convexity Chord-Tangent Enclosures across the Compact Middle Core:** Trapping all $32{,}768$ signed-threshold candidates strictly below the optimal two-block curve via 16 exact rational node certificates and 15 closed intervals.
3. **Exact Fraction-Free Bareiss Integer Elimination:** Checking positive definiteness of integer secular pencils across all candidate patterns, verified independently at nodes and near $a \to 1^-$.
4. **Universal Second-Order Dominance over the Tied Family:** An exact algebraic discriminant identity proving that the two-block family $B_{n, k}$ strictly defeats the competing two-cluster family $H^\star_{n, k} = J_k \oplus J_k \oplus 0_{n-2k}$ for all $a < 1$ across all dimensions $n \ge 2k$.
5. **Spherical Variational Grounding:** Formulation of the continuous Spherical Rayleigh Spread Maximization conjecture, proof that active support graphs are threshold graphs, and an exact analytical proof of the conjecture for dimensions $n \in \{3, 4\}$.

---

## 2. Package Contents

| File | Role & Description |
|---|---|
| [`paper.pdf`](paper.pdf) | Complete 31-page typeset manuscript (compiled cleanly with pdfTeX, 0 errors, 0 warnings, 4 vector figures embedded). |
| [`paper.tex`](paper.tex) | Complete LaTeX source code with all theorems, proofs, and references. |
| [`figures/`](figures/) | Directory containing the 4 publication-quality vector figures (`fig1.pdf` through `fig4.pdf`). |
| [`check_n8_standalone.py`](check_n8_standalone.py) | **Minimal, Zero-Dependency Verifier (Layer 2).** Pure Python standard library (no NumPy, no SciPy, zero floating point operations). Uses exact rational arithmetic (`fractions.Fraction`) to verify all 30 chord-tangent inequalities in approximately 0.02 seconds. |
| [`sp08_independent_psd_audit.py`](sp08_independent_psd_audit.py) | **Independent Integer Bareiss Audit.** Implements pure integer Bareiss fraction-free Gaussian elimination to verify positive-definiteness on all rational nodes and chord-tangent margins without floating-point arithmetic. |
| [`n8_certified_data.json`](n8_certified_data.json) | Exported exact rational certificate containing all 16 node locations, verified rational bounds, and positive quadratic margins. |
| [`verify_sp08_n8_complete.py`](verify_sp08_n8_complete.py) | **Layer 1 Foundational Certificate Pipeline.** Deduplicates 32,768 patterns to 5,643 bivariate characteristic pencils and computes rational node bounds via exact shifted polynomial nonnegativity. |
| [`verify_sp08_n8_near_one.py`](verify_sp08_n8_near_one.py) | **Endpoint Verifier near $a = 1$.** Exact fraction-free Bareiss integer Gaussian elimination checking Sylvester's criterion on integer matrices for all 32,768 patterns (~6s). |
| [`verify_sp08.py`](verify_sp08.py) | **Comprehensive Master Suite.** Verifies rank-two closed forms, rank-three classification cubics, all-runs deficit $\Delta_r$, universal dominance over $H^\star_{n, k}$, and numerical spherical optimization up to $n = 16$ (~5s). |
| [`verify_sp08_intervals.py`](verify_sp08_intervals.py) | Verifies the five certified open parameter intervals for $n \in \{8, 9\}$ via rational shifted polynomials (~3s). |
| [`verify_sp08_general_gap.py`](verify_sp08_general_gap.py) | Validates the tridiagonal Jacobi inverse and uniform sharpness of the all-runs deficit $\Delta_r$ (~1s). |
| [`sp08_fixed_d_run_reduction_audit.py`](sp08_fixed_d_run_reduction_audit.py) | **Fixed-$d$ Run Reduction Audit.** Audits the two counterexamples on non-local run reduction ($n=4$) and impossibility of fixed-$d$ reduction to $\le 2$ runs ($n=3$) (~0.1s). |

---

## 3. How to Verify

### Quick Verification (0.02 seconds, Zero Dependencies)
To verify the complete dimension-eight certificate using only the Python standard library:
```bash
python3 -O check_n8_standalone.py
```
This runs 100% exact rational arithmetic, checks interval contiguity across $[-0.97, 499/500]$, verifies containment in the boundary strips, and certifies all 30 chord-tangent algebraic inequalities with certified separation $> 10^{-7}$.

### Complete Verification Suite (All Scripts)
To run the full suite covering foundational pattern certification, the near-one endpoint, independent Bareiss audit, the master algebraic identities, open intervals, and the general deficit theorem:
```bash
python3 verify_sp08_n8_complete.py
python3 sp08_independent_psd_audit.py all
python3 sp08_fixed_d_run_reduction_audit.py
python3 verify_sp08_n8_near_one.py
python3 verify_sp08.py
python3 verify_sp08_intervals.py
python3 verify_sp08_general_gap.py
```

### Building the Manuscript
To compile the 31-page article from source:
```bash
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex
```
Output: `paper.pdf` (31 pages, 0 errors, 0 warnings).

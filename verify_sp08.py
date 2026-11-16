#!/usr/bin/env python3
"""
Exact and Analytic Verification Suite for SP-08 (The Fallat-Xing Conjecture)

Verifies all theorems, identities, bounds, Appendix A identities, and exact
finite certificates in the revised manuscript:
1. Closed-form Fallat-Xing rank-2 spectrum and unification with literature
   benchmarks (Zhan 2005 even/odd, Calkin et al. 2025, Holden 2026a).
2. Complete structural classification of rank-3 symmetric sign matrices (Type I T_-
   and Type II T_+ 3-block blow-ups), their exact characteristic cubics P^-_{p,q,r}
   and P^+_{p,q,r}, the strict rank >= 3 spectral gaps:
     - spn(S)^2 < 2n^2 - 3                       (n >= 4)
     - spn(S)^2 <= 2n^2 - (16 - 8*sqrt(2))       (n >= 7, sharp 4x4 tail bound)
   and the corresponding widened unrestricted boundary strips for n >= 4 and n >= 7.
3. Exact Frobenius-tail identity s(A)^2 = F(d~_H) - 2(1-a^2)(m_H - d~_H^2) - 2 Q_int
   and s(A)^2 = 2(||A||_F^2 - Q_int) - (tr(A) - T_int)^2.
4. Appendix A negative-parameter centered-rank-two theorem (rank(S) <= 2, -1 <= a <= 0):
   exact polynomial remainder R >= 0, eigenvalue bound eta <= (1-a)(1+a)z, and
   exact comparison identities for both t >= 0 and t < 0.
5. Five new exact integer/rational polynomial-shift finite-case certificates:
   (8, -1/2), (8, -1/3), (8, 1/3), (9, 1/2), and (9, -1/2) across all 2^(2n-1)
   signed-threshold patterns using checked int64 Newton identities, scalar
   arbitrary-precision Faddeev-LeVerrier cross-checks, and exact integer Taylor-shift
   positivity tests, explicitly separating:
     - indefinite rank-2 polynomials (c_3=...=c_n=0, c_2 < 0)
     - definite rank-2 polynomials (c_3=...=c_n=0, c_2 > 0)
     - rank-1 polynomials (c_2=...=c_n=0, c_1 != 0)
     - rank >= 3 polynomials enclosed via Lemma 5.2.
"""

from __future__ import annotations

import itertools
import math
import time
import numpy as np


# ---------------------------------------------------------------------------
# 1. Rank-2 Fallat-Xing Formula & Literature Benchmarks
# ---------------------------------------------------------------------------

def theoretical_spread_squared(n: int, k: int, a: float) -> float:
    """Exact squared spread of the rank-2 extremal block matrix B_{n,k}(a)."""
    return n**2 + 2.0 * n * (1.0 - a) * k - (1.0 - a) * (3.0 + a) * (k**2)


def optimal_k(n: int, a: float) -> int:
    """Discrete cluster size k* in {1, ..., n-1} maximizing s(B_{n,k}(a))."""
    k_cont = n / (3.0 + a)
    k1 = max(1, min(n - 1, math.floor(k_cont)))
    k2 = max(1, min(n - 1, math.ceil(k_cont)))
    if theoretical_spread_squared(n, k1, a) >= theoretical_spread_squared(n, k2, a):
        return k1
    return k2


def construct_B(n: int, k: int, a: float) -> np.ndarray:
    """Constructs B_{n,k}(a) with upper-left k x k block equal to a."""
    B = np.ones((n, n), dtype=np.float64)
    B[:k, :k] = a
    return B


def check_closed_form_and_literature() -> None:
    print("=== 1. Rank-2 Closed-Form Spectrum & Literature Benchmarks ===")
    sample_cases = [
        (4, 0.0), (5, 0.0), (6, 0.0), (7, 0.0), (8, 0.0),
        (8, 0.5), (10, 0.0), (11, 0.0), (12, -0.5), (16, -1.0),
    ]
    for n, a in sample_cases:
        k = optimal_k(n, a)
        B = construct_B(n, k, a)
        eigs = np.linalg.eigvalsh(B)
        s_num = float(eigs[-1] - eigs[0])
        s_exact = math.sqrt(theoretical_spread_squared(n, k, a))
        err = abs(s_num - s_exact)
        assert err < 1e-11, f"Spectrum discrepancy at n={n}, a={a}: {err}"
        print(f"  (n={n:2d}, a={a:5.2f}, k*={k:2d}): s = {s_exact:.8f} (err={err:.2e})")

    # Holden (2026a) exact rational cases
    for n, a, k_exp, s2_exp in [(8, 0.5, 2, 73), (10, 0.0, 3, 133), (11, 0.0, 4, 161)]:
        k = optimal_k(n, a)
        assert k == k_exp
        s2 = theoretical_spread_squared(n, k, a)
        assert abs(s2 - s2_exp) < 1e-11
        print(f"  Holden (2026a) ({n:2d}, {a:3.1f}): k*={k}, s^2={s2:.0f} == {s2_exp} [OK]")

    # Zhan (2005) a = -1 even/odd
    for n in [4, 6, 8, 10, 12]:
        k = optimal_k(n, -1.0)
        assert k == n // 2
        assert abs(theoretical_spread_squared(n, k, -1.0) - 2.0 * n**2) < 1e-11
    for n in [3, 5, 7, 9, 11]:
        k = optimal_k(n, -1.0)
        assert k == (n - 1) // 2
        assert abs(theoretical_spread_squared(n, k, -1.0) - (2.0 * n**2 - 1.0)) < 1e-11
    print("  Zhan (2005) a=-1 even/odd formulas match exactly [OK]")

    # Calkin et al. (2025) a = 0, 3 | n
    for n in [3, 6, 9, 12, 15]:
        k = optimal_k(n, 0.0)
        assert k == n // 3
        assert abs(theoretical_spread_squared(n, k, 0.0) - (4.0 / 3.0) * n**2) < 1e-11
    print("  Calkin et al. (2025) a=0, 3|n formula matches exactly [OK]\n")


# ---------------------------------------------------------------------------
# 2. Rank-3 Classification, Spectral Gaps (< 2n^2-3 & <= 2n^2-(16-8sqrt(2)))
# ---------------------------------------------------------------------------

def _all_sym_sign_matrices(n: int):
    """Generates all 2^(n(n+1)/2) symmetric {-1, +1} matrices of order n."""
    num_vars = n * (n + 1) // 2
    tri_r, tri_c = np.triu_indices(n)
    M = np.empty((n, n), dtype=np.float64)
    for bits in range(1 << num_vars):
        vals = [1.0 if ((bits >> idx) & 1) == 0 else -1.0 for idx in range(num_vars)]
        M[tri_r, tri_c] = vals
        M[tri_c, tri_r] = vals
        yield M


def check_rank3_sign_classification_and_gap() -> None:
    print("=== 2. Rank-3 Classification, Spectral Gaps & Widened Boundary Strips ===")

    # 2a. Classify all 64 symmetric 3x3 sign matrices and check Schur admissibility
    Tm = np.array([[1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype=np.float64)
    Tp = np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1]], dtype=np.float64)
    spec_Tm = tuple(np.round(np.linalg.eigvalsh(Tm), 8))
    spec_Tp = tuple(np.round(np.linalg.eigvalsh(Tp), 8))

    nonsing_3x3 = 0
    for M in _all_sym_sign_matrices(3):
        det = round(float(np.linalg.det(M)))
        if det != 0:
            nonsing_3x3 += 1
            assert abs(det) == 4
            ev_pos = tuple(np.round(np.linalg.eigvalsh(M), 8))
            ev_neg = tuple(np.round(np.linalg.eigvalsh(-M), 8))
            assert (ev_pos == spec_Tm or ev_neg == spec_Tm or
                    ev_pos == spec_Tp or ev_neg == spec_Tp)
    assert nonsing_3x3 == 32
    print(f"  3x3 classification: {nonsing_3x3}/64 nonsingular; all have |det|=4 & match +/-Tm or +/-Tp [OK]")

    Tm_inv = np.linalg.inv(Tm)
    Tp_inv = np.linalg.inv(Tp)
    for b_tup in itertools.product([-1, 1], repeat=3):
        b = np.array(b_tup, dtype=np.float64)
        val_m = round(float(b @ Tm_inv @ b))
        val_p = round(float(b @ Tp_inv @ b))
        expected_m = round(b[0] * (b[1] + b[2]) - 1)
        expected_p = round(b[0] * (b[1] + b[2]) - b[1] * b[2])
        assert val_m == expected_m and val_p == expected_p
        assert (val_m in (-1, 1)) == (b_tup not in ((-1, 1, 1), (1, -1, -1)))
        assert (val_p in (-1, 1)) == (b_tup not in ((1, -1, -1), (-1, 1, 1)))
    print("  Schur complement admissibility b^T T^{-1} b in {-1,1} verified for Tm & Tp [OK]")

    # 2b. Verify cubics P^- and P^+ and bounds:
    #     - n >= 4: s(S)^2 < 2n^2 - 3
    #     - n >= 7: s(S)^2 < 2n^2 - 19/4 < 2n^2 - (16 - 8*sqrt(2))
    delta_star = 16.0 - 8.0 * math.sqrt(2.0)
    worst_n4 = 0.0
    worst_n7_gap = float("inf")
    for n in range(3, 31):
        for p in range(1, n - 1):
            for q in range(1, n - p):
                r = n - p - q
                w = q + r
                tau = p - w
                roots_1 = np.sort(np.roots([1.0, -tau, -2.0 * p * w, -4.0 * p * q * r]).real)
                roots_2 = np.sort(np.roots([1.0, -float(n), 0.0, 4.0 * p * q * r]).real)

                if n <= 12:
                    B1 = np.ones((n, n))
                    B1[p:p + q, p:p + q] = -1.0
                    B1[p + q:, p + q:] = -1.0
                    nz1 = np.sort([x for x in np.linalg.eigvalsh(B1) if abs(x) > 1e-8])
                    assert len(nz1) == 3 and np.allclose(nz1, roots_1, atol=1e-9)

                    B2 = np.ones((n, n))
                    B2[p:p + q, p + q:] = -1.0
                    B2[p + q:, p:p + q] = -1.0
                    nz2 = np.sort([x for x in np.linalg.eigvalsh(B2) if abs(x) > 1e-8])
                    assert len(nz2) == 3 and np.allclose(nz2, roots_2, atol=1e-9)

                sp1_sq = float((roots_1[-1] - roots_1[0]) ** 2)
                sp2_sq = float((roots_2[-1] - roots_2[0]) ** 2)
                if n == 4:
                    worst_n4 = max(worst_n4, sp1_sq)
                if n >= 4:
                    assert abs(roots_1[1]) > 1.0 + 1e-12
                    assert sp1_sq < 2 * n**2 - 3.0 - 1e-9
                    assert sp2_sq <= (4.0 / 3.0) * n**2 + 1e-9
                    assert sp2_sq < 2 * n**2 - 3.0 - 1e-9
                if n >= 7:
                    if tau in (0, -1, -2):
                        # Q_{p,q,r}(3/2) >= 9/8 > 0 forces -roots_1[1] > 3/2
                        Q_val = 27.0 / 8.0 + (9.0 / 4.0) * tau - 3.0 * p * w + 4.0 * p * q * r
                        assert Q_val >= 9.0 / 8.0 - 1e-12
                        assert -roots_1[1] > 1.5
                    assert sp1_sq < 2 * n**2 - 4.75 - 1e-9
                    assert sp1_sq < 2 * n**2 - delta_star - 1e-9
                    assert sp2_sq < 2 * n**2 - delta_star - 1e-9
                    worst_n7_gap = min(worst_n7_gap, 2 * n**2 - sp1_sq)

    print(f"  3-block cubics P^- and P^+ verified for n=3..30 "
          f"(n=4 max s^2={worst_n4:.6f}<29; n>=7 min gap={worst_n7_gap:.6f}>4.75>{delta_star:.4f}) [OK]")

    # 2c. Exhaustive check of 4x4 normalized sign matrices: min(s3^2 + s4^2) = 8 - 4*sqrt(2)
    min_s34 = float("inf")
    M4 = np.ones((4, 4), dtype=np.float64)
    for bits in range(1 << 9):
        for r_idx in range(3):
            for c_idx in range(3):
                M4[r_idx + 1, c_idx + 1] = -1.0 if ((bits >> (3 * r_idx + c_idx)) & 1) else 1.0
        det = round(float(np.linalg.det(M4)))
        if det != 0:
            assert det % 8 == 0
            sv = np.linalg.svd(M4, compute_uv=False)
            s34 = float(sv[2]**2 + sv[3]**2)
            if s34 < min_s34:
                min_s34 = s34
    target_s34 = 8.0 - 4.0 * math.sqrt(2.0)
    assert abs(min_s34 - target_s34) < 1e-11
    assert abs(2.0 * min_s34 - delta_star) < 1e-11
    print(f"  4x4 nonsingular tail min(sigma_3^2+sigma_4^2) = {min_s34:.6f} == 8-4sqrt(2) [OK]")

    # 2d. Exhaustive symmetric sign matrix check for n=4 (1,024) and n=5 (32,768)
    for n in (4, 5):
        pqr_specs = set()
        for p in range(1, n - 1):
            for q in range(1, n - p):
                r = n - p - q
                for xi in (1.0, -1.0):
                    r1 = xi * np.roots([1.0, -(p - q - r), -2.0 * p * (q + r), -4.0 * p * q * r]).real
                    r2 = xi * np.roots([1.0, -float(n), 0.0, 4.0 * p * q * r]).real
                    zeros = [0.0] * (n - 3)
                    pqr_specs.add(tuple(np.round(np.sort(np.concatenate([r1, zeros])), 6)))
                    pqr_specs.add(tuple(np.round(np.sort(np.concatenate([r2, zeros])), 6)))

        count_r3 = 0
        max_r3_ge_sp2 = 0.0
        for M in _all_sym_sign_matrices(n):
            rk = int(np.linalg.matrix_rank(M, tol=1e-9))
            if rk >= 3:
                eigs = np.linalg.eigvalsh(M)
                sp2 = float((eigs[-1] - eigs[0]) ** 2)
                if sp2 > max_r3_ge_sp2:
                    max_r3_ge_sp2 = sp2
                if rk == 3:
                    count_r3 += 1
                    ev = tuple(np.round(eigs, 6))
                    assert ev in pqr_specs, f"Unclassified rank-3 matrix at n={n}: {ev}"
        assert max_r3_ge_sp2 < 2 * n**2 - 3.0
        print(f"  Exhaustive n={n} symmetric sign matrices: {count_r3} rank-3 matched, "
              f"max rank>=3 s^2={max_r3_ge_sp2:.6f} < {2*n**2 - 3} [OK]")

    # 2e. Verify boundary strips for (delta=3, n=4..50) and (delta=delta_*, n=7..50)
    sq2 = math.sqrt(2.0)
    for label, delta_val, n_min, mu_uni_fn in [
        ("n>=4, delta=3", 3.0, 4, lambda n: (1.0 + sq2) / (n**2)),
        ("n>=7, delta=16-8sqrt(2)", delta_star, 7, lambda n: (7.0 * sq2 - 1.0) / (2.0 * n**2)),
    ]:
        for n in range(n_min, 51):
            L = math.sqrt(2 * n**2 - delta_val)
            # Check 2n*L - 2n^2 > 2*(sqrt(2)-1)*n^2 - delta_val
            assert 2 * n * L - 2 * n**2 > 2 * (sq2 - 1.0) * n**2 - delta_val
            eps = n % 2
            if eps == 0:
                C_eps = 2 * n * L - 2 * n**2 + delta_val
                mu_sharp = delta_val / C_eps
                assert C_eps > 2 * (sq2 - 1.0) * n**2
            else:
                C_eps = 2 * n * L - 2 * n**2 - 2 * n + delta_val + 1.0
                mu_sharp = (delta_val - 1.0) / C_eps
                assert C_eps > 2 * (sq2 - 1.0) * n**2 - 2 * n + 1.0

            assert n * mu_sharp < 1.0, f"n*mu_sharp >= 1 at ({label}, n={n})"
            mu_uni = mu_uni_fn(n)
            assert mu_uni < mu_sharp, f"mu_uni >= mu_sharp at ({label}, n={n})"

            a_sharp = -1.0 + 2.0 * mu_sharp
            assert a_sharp < 0.0
            k_floor = n // 2
            assert abs(k_floor - n / (3.0 + a_sharp)) < 0.5
            sp_Bk = math.sqrt(theoretical_spread_squared(n, k_floor, a_sharp))
            rho_sharp = 1.0 - mu_sharp
            assert sp_Bk + 1e-11 >= mu_sharp * n + rho_sharp * L
        print(f"  Widened boundary strip ({label}) verified for n={n_min}..50 [OK]")
    print()


# ---------------------------------------------------------------------------
# 3. Frobenius-Tail Identity & Appendix A Analytic Identities
# ---------------------------------------------------------------------------

def check_frobenius_tail_identity() -> None:
    print("=== 3. Exact Frobenius-Tail Identity Check ===")
    rng = np.random.default_rng(42)
    for n in (5, 7, 10):
        for a in (-0.5, 0.0, 0.5):
            delta = 1.0 - a
            H = rng.integers(0, 2, size=(n, n)).astype(np.float64)
            H = np.triu(H) + np.triu(H, 1).T
            A = np.ones((n, n)) - delta * H
            eigs = np.sort(np.linalg.eigvalsh(A))
            s_sq = float((eigs[-1] - eigs[0]) ** 2)

            d_H = float(np.trace(H))
            m_H = float(np.sum(H * H))
            T_int = float(np.sum(eigs[1:-1]))
            Q_int = float(np.sum(eigs[1:-1] ** 2))
            d_tilde = d_H + T_int / delta
            F_val = n**2 + 2.0 * n * delta * d_tilde - delta * (4.0 - delta) * (d_tilde**2)

            rhs_tail = 2.0 * (float(np.sum(A * A)) - Q_int) - (float(np.trace(A)) - T_int) ** 2
            rhs_Fh = F_val - 2.0 * (1.0 - a**2) * (m_H - d_tilde**2) - 2.0 * Q_int
            assert abs(s_sq - rhs_tail) < 1e-9
            assert abs(s_sq - rhs_Fh) < 1e-9
    print("  Exact identities s(A)^2 = 2(||A||_F^2-Q_int)-(tr(A)-T_int)^2 and F(d~)-2(1-a^2)(m_H-d~^2)-2Q_int [OK]\n")


def check_appendix_a_identities() -> None:
    print("=== 4. Appendix A (Holden 2026b rank(S)<=2, a<=0) Identities ===")
    rng = np.random.default_rng(7)
    # Test Lemma A.1 polynomial identity n*N^2*h(g_0) = -R
    for _ in range(50):
        P = float(rng.integers(1, 10))
        Q = float(rng.integers(1, 10))
        x = float(rng.uniform(0, 8))
        y = float(rng.uniform(0, 8))
        p, q = P + x, Q + y
        n, N = p + q, P + Q
        R0 = n - P**2 / p - Q**2 / q
        K = P**2 + 2.0 * P * Q - Q**2
        g0 = q * P / N
        h_g0 = g0**2 + (p - q - K / n) * g0 - 2.0 * p * q * R0 / n
        R_poly = (
            P**2 * N * Q**2
            + P * Q * (2 * P**2 + 5 * P * Q + 4 * Q**2) * x
            + P * Q * (4 * P**2 + 9 * P * Q + 6 * Q**2) * y
            + N * Q * (P + 2 * Q) * x**2
            + P * (P**2 + 4 * P * Q + 5 * Q**2) * y**2
            + 2.0 * (P**3 + 4 * P**2 * Q + 6 * P * Q**2 + 2 * Q**3) * x * y
            + N * (P + 2 * Q) * x**2 * y
            + (P**2 + 4 * P * Q + 2 * Q**2) * x * y**2
            + P * Q * y**3
        )
        assert abs(n * N**2 * h_g0 + R_poly) < 1e-7

    # Test Type-A 3x3 compression eigenvalues, Lemma A.3, and t>=0 / t<0 comparisons
    for p in range(1, 5):
        for y in range(0, 4):
            for z in range(1, 4):
                n = p + y + z
                q = y + z
                for a in (-1.0, -0.75, -0.5, -0.25, 0.0):
                    # Construct explicit n x n Type-A matrix
                    A = np.ones((n, n), dtype=np.float64)
                    # Groups: 0..p-1, p..p+y-1, p+y..n-1 with base [[1,1,a],[1,a,1],[a,1,a]]
                    A[:p, p + y:] = a
                    A[p + y:, :p] = a
                    A[p:p + y, p:p + y] = a
                    A[p + y:, p + y:] = a
                    eigs_A = np.sort(np.linalg.eigvalsh(A))
                    sp_A = float(eigs_A[-1] - eigs_A[0])

                    # 3x3 compression T_A
                    TA = np.array([
                        [float(p), math.sqrt(p * y), a * math.sqrt(p * z)],
                        [math.sqrt(p * y), a * y, math.sqrt(y * z)],
                        [a * math.sqrt(p * z), math.sqrt(y * z), a * z],
                    ], dtype=np.float64)
                    eigs_T = np.sort(np.linalg.eigvalsh(TA))
                    sp_T = float(eigs_T[-1] - eigs_T[0])
                    assert abs(sp_A - sp_T) < 1e-10
                    lam, eta, neg_mu = eigs_T[2], max(0.0, eigs_T[1]), eigs_T[0]
                    mu = -neg_mu
                    s = lam + mu
                    delta, b_param = 1.0 - a, 1.0 + a
                    t_tr = p + a * q
                    # Identity s^2 = F(q) - 4*delta*b_param*z*(p - q + z) + 2*t*eta - 3*eta^2
                    Fq = theoretical_spread_squared(n, q, a)
                    rhs = Fq - 4.0 * delta * b_param * z * (p - q + z) + 2.0 * t_tr * eta - 3.0 * eta**2
                    assert abs(s**2 - rhs) < 1e-8
                    assert eta <= delta * b_param * z + 1e-10

                    M2 = max(theoretical_spread_squared(n, k, a) for k in range(1, n))
                    assert sp_A**2 <= M2 + 1e-9
    print("  Appendix A algebraic identities, eta <= (1-a)(1+a)z, and Type-A <= M_n(a) verified [OK]\n")


# ---------------------------------------------------------------------------
# 4. Exact Finite-Case Polynomial-Shift Certificates
# ---------------------------------------------------------------------------

def integer_safety_bound(n: int, max_abs_entry: int) -> int:
    """Upper bound on intermediate sums in int64 Newton identities."""
    return n * (2**n) * (n * max_abs_entry) ** n


def signed_threshold_batches(n: int, low: int, high: int, batch: int = 2048):
    """Yields all 2^(2n-1) signed-threshold endpoint matrices as int64 batches."""
    total = 1 << (2 * n - 1)
    idx = np.minimum.outer(np.arange(n), np.arange(n))
    shifts = np.arange(n, dtype=np.uint64)
    for start in range(0, total, batch):
        ids = np.arange(start, min(start + batch, total), dtype=np.uint64)
        ecode = ids >> np.uint64(n - 1)
        eps = 1 - 2 * ((ecode[:, None] >> shifts) & 1).astype(np.int64)
        d = np.ones((len(ids), n), dtype=np.int64)
        if n > 1:
            d[:, 1:] = 1 - 2 * ((ids[:, None] >> shifts[: n - 1]) & 1).astype(np.int64)
        S = eps[:, idx] * d[:, :, None] * d[:, None, :]
        A = np.where(S > 0, high, low).astype(np.int64)
        yield ids, A


def charpolys_newton_int64(A: np.ndarray) -> np.ndarray:
    """Computes exact descending characteristic polynomial coefficients via Newton identities."""
    count, n, _ = A.shape
    C = np.zeros((count, n + 1), dtype=np.int64)
    C[:, 0] = 1
    traces = np.empty((count, n), dtype=np.int64)
    power = A.copy()
    for k in range(1, n + 1):
        traces[:, k - 1] = np.trace(power, axis1=1, axis2=2)
        sums = np.zeros(count, dtype=np.int64)
        for i in range(1, k + 1):
            sums += C[:, k - i] * traces[:, i - 1]
        if np.any(sums % k):
            raise ArithmeticError("Newton identity divisibility failure")
        C[:, k] = -(sums // k)
        if k < n:
            power = power @ A
    return C


def faddeev_leverrier_bigint(A: list[list[int]]) -> tuple[int, ...]:
    """Independent arbitrary-precision Python integer characteristic polynomial."""
    n = len(A)
    B = [[int(i == j) for j in range(n)] for i in range(n)]
    coeffs = [1]
    for k in range(1, n + 1):
        AB = [[sum(A[i][r] * B[r][j] for r in range(n)) for j in range(n)] for i in range(n)]
        tr = sum(AB[i][i] for i in range(n))
        if tr % k != 0:
            raise ArithmeticError("Non-integral Faddeev-LeVerrier coefficient")
        c = -tr // k
        coeffs.append(c)
        for i in range(n):
            AB[i][i] += c
        B = AB
    return tuple(coeffs)


def shifted_poly_nonnegative(coeffs: tuple[int, ...], u: int, D: int) -> bool:
    """Checks via exact Python bigints whether all coefficients of D^n p((z+u)/D) are >= 0."""
    if D <= 0 or coeffs[0] != 1:
        raise ValueError("Monic polynomial and positive denominator D required")
    b = [1]
    Dpow = 1
    for c_j in coeffs[1:]:
        Dpow *= D
        new = [u * b[0] + c_j * Dpow]
        for j in range(1, len(b)):
            new.append(u * b[j] + b[j - 1])
        new.append(b[-1])
        b = new
    return all(x >= 0 for x in b)


def negated_poly(coeffs: tuple[int, ...]) -> tuple[int, ...]:
    """Returns (-1)^n p(-z), reversing eigenvalue signs."""
    return tuple(((-1) ** j) * c for j, c in enumerate(coeffs))


def certify_exact_finite_case(n: int, low: int, high: int) -> dict:
    """Enumerates all 2^(2n-1) patterns and classifies & certifies every distinct polynomial."""
    t0 = time.monotonic()
    bound = integer_safety_bound(n, max(abs(low), abs(high)))
    assert bound < (1 << 63), f"Overflow guard triggered: {bound}"

    best_num = -1
    best_ks = []
    for k in range(1, n):
        val = (low * k - high * (n - k)) ** 2 + 4 * (high**2) * k * (n - k)
        if val > best_num:
            best_num = val
            best_ks = [k]
        elif val == best_num:
            best_ks.append(k)

    distinct: dict[tuple[int, ...], np.ndarray] = {}
    for ids, batch_A in signed_threshold_batches(n, low, high):
        C = charpolys_newton_int64(batch_A)
        for idx, row in enumerate(C):
            c_tup = tuple(map(int, row))
            if c_tup not in distinct:
                distinct[c_tup] = batch_A[idx].copy()
        probe = (int(ids[0]) // 2048) % len(ids)
        assert faddeev_leverrier_bigint(batch_A[probe].tolist()) == tuple(map(int, C[probe]))

    indef_rank2 = 0
    def_rank2 = 0
    rank1 = 0
    rank_ge3 = 0
    enclosure_all_non_indef = 0
    uncertified = 0

    for c_tup, A in distinct.items():
        if all(x == 0 for x in c_tup[2:]) and c_tup[1] != 0:
            # Rank 1: s(A) = |c_1|
            rank1 += 1
            assert c_tup[1] ** 2 <= best_num
        elif all(x == 0 for x in c_tup[3:]) and c_tup[2] < 0:
            # Indefinite rank 2: opposite-sign roots
            assert c_tup[1] ** 2 - 4 * c_tup[2] <= best_num
            indef_rank2 += 1
            continue
        elif all(x == 0 for x in c_tup[3:]) and c_tup[2] > 0:
            # Definite rank 2: two nonzero roots of same sign plus n-2 >= 1 zeros
            disc = c_tup[1] ** 2 - 4 * c_tup[2]
            assert disc >= 0 and c_tup[1] ** 2 <= 4 * best_num
            # Exact check: (|c1| + sqrt(disc))/2 <= sqrt(best_num)
            abs_c1 = abs(c_tup[1])
            assert c_tup[1] ** 2 <= best_num  # guarantees s <= |c1| <= M
            def_rank2 += 1
        else:
            rank_ge3 += 1

        # Also certify every polynomial outside indef_rank2 with rational Taylor shifts
        eigs = np.linalg.eigvalsh(A.astype(np.float64))
        ok = False
        for D in (10**4, 10**6, 10**8, 10**10):
            u_pos = math.ceil(float(eigs[-1]) * D) + 3
            u_neg = math.ceil(float(-eigs[0]) * D) + 3
            if u_pos + u_neg >= 0 and (u_pos + u_neg) ** 2 <= best_num * (D**2):
                if shifted_poly_nonnegative(c_tup, u_pos, D) and shifted_poly_nonnegative(
                    negated_poly(c_tup), u_neg, D
                ):
                    enclosure_all_non_indef += 1
                    ok = True
                    break
        if not ok:
            uncertified += 1

    return {
        "n": n,
        "low": low,
        "high": high,
        "patterns": 1 << (2 * n - 1),
        "distinct": len(distinct),
        "indef_rank2": indef_rank2,
        "def_rank2": def_rank2,
        "total_rank2": indef_rank2 + def_rank2,
        "rank1": rank1,
        "rank_ge3": rank_ge3,
        "enclosure_non_indef": enclosure_all_non_indef,
        "uncertified": uncertified,
        "best_num": best_num,
        "best_ks": best_ks,
        "bound": bound,
        "seconds": time.monotonic() - t0,
    }


def check_new_exact_finite_certificates() -> None:
    print("=== 5. Five New Exact Finite Certificates (n in {8, 9}) ===")
    # (n, low, high, a_label, distinct, indef_r2, def_r2, total_r2, r1, rank_ge3, enc_non_indef, best_num, best_ks)
    cases = [
        (8, -1, 2, "-1/2",  5629, 30,  4, 34, 2,  5593,  5599, 409, [3]),
        (8, -1, 3, "-1/3",  5641, 18,  4, 22, 2,  5617,  5623, 864, [3]),
        (8,  1, 3,  "1/3",  5647, 11, 11, 22, 2,  5623,  5636, 688, [2]),
        (9,  1, 2,  "1/2", 19138, 12, 12, 24, 2, 19112, 19126, 369, [3]),
        (9, -1, 2, "-1/2", 19118, 36,  4, 40, 2, 19076, 19082, 516, [4]),
    ]
    for n, lo, hi, a_lbl, e_dist, e_ir2, e_dr2, e_tr2, e_r1, e_ge3, e_enc, e_best, e_ks in cases:
        res = certify_exact_finite_case(n, lo, hi)
        assert res["uncertified"] == 0
        assert res["distinct"] == e_dist
        assert res["indef_rank2"] == e_ir2
        assert res["def_rank2"] == e_dr2
        assert res["total_rank2"] == e_tr2
        assert res["rank1"] == e_r1
        assert res["rank_ge3"] == e_ge3
        assert res["enclosure_non_indef"] == e_enc
        assert res["best_num"] == e_best
        assert res["best_ks"] == e_ks
        s_val = math.sqrt(e_best) / hi
        print(
            f"  PASS (n={n}, a={a_lbl:>4s}, {{{lo:2d},{hi}}}): {res['patterns']:6d} patterns -> "
            f"{e_dist:5d} polys [r1={e_r1}, r2={e_tr2} ({e_ir2} indef + {e_dr2} def), "
            f"r>=3={e_ge3}, 0 fails], k*={e_ks[0]}, max s = {s_val:.8f} [{res['seconds']:.2f}s]"
        )
    print()




def check_universal_tied_family_and_spherical_conjecture() -> None:
    print("=== 6. Universal Tied Family H*_{n,k} and Spherical Rayleigh Verification ===")
    import sympy as sp
    import numpy as np
    from scipy.optimize import minimize

    # Part A: Exact algebraic identity for H*_{n,k}
    n, k, delta = sp.symbols("n k delta", positive=True)
    disc = n**2 + 2*k*(n - 4*k)*delta + k**2 * delta**2
    X = n + delta*k
    Y = sp.sqrt(disc)
    s_B_sq = n**2 + 2*k*(n - 2*k)*delta + k**2 * delta**2
    s_B = sp.sqrt(s_B_sq)
    s_Hstar = (X + Y) / 2

    # Verify 2 s(B)^2 = X^2 + Y^2
    assert sp.simplify(2*s_B_sq - (X**2 + Y**2)) == 0, "2*s(B)^2 != X^2 + Y^2"
    # Verify X^2 - Y^2 = 8 k^2 delta
    assert sp.simplify(X**2 - Y**2 - 8*delta*k**2) == 0, "X^2 - Y^2 != 8 k^2 delta"

    # Verify Taylor series expansion near delta = 0
    series = sp.series(s_B - s_Hstar, delta, 0, 4)
    c1 = sp.simplify(series.coeff(delta, 1))
    c2 = sp.simplify(series.coeff(delta, 2))
    c3 = sp.simplify(series.coeff(delta, 3))
    assert c1 == 0, f"c1 is {c1}, expected 0 (first-order tie)"
    assert sp.simplify(c2 - 2*k**4 / n**3) == 0, f"c2 is {c2}, expected 2k^4/n^3"
    assert sp.simplify(c3 - (-6*k**5*(n - 2*k) / n**5)) == 0, f"c3 is {c3}, expected -6k^5(n-2k)/n^5"
    print("  PASS: Exact algebraic identities and series expansion for H*_{n,k} verified.")
    print("        s(B_{n,k})^2 - s(J_n - delta H*_{n,k})^2 = (X - Y)^2 / 4 > 0 (for all n >= 2k, delta > 0)")
    print("        Gap = (2k^4 / n^3) * (1-a)^2 - (6k^5(n-2k)/n^5) * (1-a)^3 + O((1-a)^4)")

    # Part B: Numerical multi-start spherical optimization for n = 3..16
    np.random.seed(42)
    for dim in range(3, 17):
        k_vals = np.arange(1, dim)
        c1_vals = k_vals - 2.0 * k_vals**2 / dim
        max_two_block = np.max(c1_vals)
        best_k = k_vals[np.argmax(c1_vals)]

        def loss(z):
            z = z - np.mean(z)
            norm = np.linalg.norm(z)
            if norm < 1e-12:
                return 0.0
            z = z / norm
            M = np.outer(z, z) - 1.0 / dim
            return -np.sum(np.maximum(M, 0.0))

        best_val = 0.0
        for trial in range(15):
            z0 = np.random.randn(dim)
            res = minimize(loss, z0, method="BFGS")
            val = -res.fun
            if val > best_val:
                best_val = val
        assert abs(best_val - max_two_block) < 1e-5, f"Sphere check failed at dim {dim}: {best_val} vs {max_two_block}"

    print("  PASS: Spherical Rayleigh optimization matches two-block bound for all n in [3, 16] (< 1e-5).")
    print()
if __name__ == "__main__":
    print("=" * 80)
    print("  EXACT & ANALYTIC VERIFIER FOR SP-08 (REVISED WITH REFEREE CORRECTIONS)")
    print("=" * 80)
    check_closed_form_and_literature()
    check_rank3_sign_classification_and_gap()
    check_frobenius_tail_identity()
    check_appendix_a_identities()
    check_new_exact_finite_certificates()
    check_universal_tied_family_and_spherical_conjecture()
    print("=" * 80)
    print("  ALL ANALYTIC, CLASSIFICATION, APPENDIX A & CERTIFICATE CHECKS PASSED (100%)")
    print("=" * 80)

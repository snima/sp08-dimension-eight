#!/usr/bin/env python3
"""Certified parameter-wide proof of Fallat–Xing SP-08 for n=8.

Mathematical prerequisites:
  (i) Holden signed-threshold reduction, proved in the accompanying paper;
  (ii) Holden 2026b original all-dimensional analytic strip near a=-1;
  (iii) exact analytic/Bareiss near-one certificate, using
        verify_sp08_n8_near_one.py (run separately).

The remaining compact interval is partitioned into 15 rational subintervals.
For each endpoint alpha, this program generates an EXACT rational upper bound
U(alpha) for every non-B signed-threshold candidate using exact integer
characteristic polynomials and shifted-polynomial nonnegativity. Floating
point eigenvalues merely *suggest* rational root bounds; exact integer
arithmetic verifies EVERY bound. Across each interval, convexity gives an
upper chord through the two endpoint U's. A tangent to the appropriate
rank-two spread b_k gives an exact algebraic lower bound. The final two
chord-vs-tangent inequalities are tested over the rationals (squaring only
strictly positive quantities). No floating-point operation contributes to
accept/reject decisions.

Dependencies: Python3, NumPy and verify_sp08.py in the same directory.
Run: OPENBLAS_NUM_THREADS=1 python3 verify_sp08_n8_complete.py

Output: positive algebraic inequalities for all 15 intervals or failure.
Also exports `n8_certified_data.json` for independent zero-dependency checking.
"""
from __future__ import annotations
import math
import time
import json
from fractions import Fraction as F
import numpy as np
from verify_sp08 import (
    signed_threshold_batches,
    charpolys_newton_int64,
    shifted_poly_nonnegative,
    negated_poly,
    faddeev_leverrier_bigint,
    integer_safety_bound,
)
from verify_sp08_intervals import is_B_shape

N = 8
# a=-1+ (7 sqrt(2)-1)/64 is greater than -17219/20000.
# The first node -97/100 is strictly inside Holden's original strip.
NODES = [
    F(-97, 100), F(-36619, 40000), F(-17219, 20000), F(-5, 7), F(-9, 35), F(1, 5),
    F(599, 1000), F(1597, 2000), F(3593, 4000), F(1517, 1600),
    F(15569, 16000), F(31537, 32000), F(63473, 64000),
    F(25469, 25600), F(255089, 256000), F(499, 500),
]
if len(NODES) != 16 or not all(x < y for x, y in zip(NODES, NODES[1:])):
    raise ValueError("Invalid node partition")

# Eigenvalue rational denominator for the SCALED matrix q A, not A itself.
ROOT_DEN = 10**6
# A globally certified minimum gap on the compact middle interval.
EXACT_MARGIN = F(1, 10**7)


def enumerate_spectral_pencils():
    """Dedup by an exact invariant defining the full characteristic polynomial.

    Write delta=(q-p)/q, and let c_k^-(H) denote coefficient of x^(8-k)
    in det(xI+H); and d_k(H) = c_k(J-H)-c_k(-H).
    Rank-one-perturbation determinant identity yields the coefficient in
    det(xI-qA(a)) for a=p/q, t=q-p:
      c_k(qA(a)) = t^k c_k^-(H) + q t^(k-1) d_k(H).
    Thus the pair of vectors (c^-,d) is a complete characteristic-pencil
    invariant, for every value of a, not a floating-point fingerprint.
    """
    if integer_safety_bound(N, 1) >= (1 << 63):
        raise RuntimeError("int64 overflow bound violated")
    all_count = 0
    b_count = 0
    rep = {}
    for ids, S in signed_threshold_batches(N, -1, 1):
        is_b = is_B_shape(S, -1, 1)
        b_count += int(is_b.sum())
        all_count += len(ids)
        H = ((1 - S) // 2).astype(np.int64)
        c_minus = charpolys_newton_int64(-H)
        c_one = charpolys_newton_int64(1 - H)
        qidx = min(13, len(ids) - 1)
        for sample, all_c in ((-H, c_minus), (1 - H, c_one)):
            if faddeev_leverrier_bigint(sample[qidx].tolist()) != tuple(map(int, all_c[qidx])):
                raise RuntimeError("Faddeev-LeVerrier cross-check failed")
        for i in np.flatnonzero(~is_b):
            cm = tuple(map(int, c_minus[i]))
            d = tuple(int(c_one[i, k]) - cm[k] for k in range(9))
            key = cm, d
            if key not in rep:
                rep[key] = H[i].astype(np.int8)
    if all_count != 32768:
        raise RuntimeError(f"Expected 32768 patterns, got {all_count}")
    print(
        f"Exact enumeration: {all_count} threshold patterns; "
        f"{b_count} B-shaped; {len(rep)} distinct non-B characteristic pencils.",
        flush=True,
    )
    return rep


def scaled_charpoly(cm, d, a):
    q = a.denominator
    t = q - a.numerator
    if t <= 0:
        raise ValueError(f"t must be positive, got {t}")
    p = [1]
    tk = 1
    for k in range(1, N + 1):
        tkprev = tk
        tk *= t
        p.append(tk * cm[k] + q * tkprev * d[k])
    return tuple(p)


def upper_spread_exact(poly, a, H):
    """Exact rational upper spread via per-root rational witnesses."""
    qa = a.denominator
    t = qa - a.numerator
    AA = np.ones((N, N), dtype=np.float64) - float(F(t, qa)) * H.astype(float)
    ev = np.linalg.eigvalsh(AA)
    R = ROOT_DEN
    plus = math.ceil(float(ev[-1]) * qa * R) + 4
    minus = math.ceil(float(-ev[0]) * qa * R) + 4
    pminus = negated_poly(poly)
    for _ in range(1000):
        if shifted_poly_nonnegative(poly, plus, R):
            break
        plus += 4
    else:
        raise RuntimeError("No exact upper positive-eigenvalue enclosure")
    for _ in range(1000):
        if shifted_poly_nonnegative(pminus, minus, R):
            break
        minus += 4
    else:
        raise RuntimeError("No exact upper negative-eigenvalue enclosure")
    if plus + minus < 0:
        raise RuntimeError("Spread width cannot be negative")
    return F(plus + minus, qa * R)


def b2(k, a):
    # Exact squared spread of the rank-two family B_{8,k}(a).
    return F(N * N) + F(2 * N * k) * (1 - a) - (1 - a) * (3 + a) * k * k


def tangent_dominates_at_endpoint(k, m, a, U):
    """Check exactly: U < b_k(m)+b'_k(m)*(a-m)."""
    B = b2(k, m)
    C = B + k * (k * m - (N - k)) * (a - m)
    if U < 0 or B <= 0 or C <= 0:
        raise ValueError(f"Positivity violation in tangent test: U={U}, B={B}, C={C}")
    gap_square = C * C - (U + EXACT_MARGIN) ** 2 * B
    if gap_square <= 0:
        raise RuntimeError(f"Failed exact chord-tangent margin certificate: a={a}, k={k}, gap_sq={gap_square}")
    approx_margin = float(C) / math.sqrt(float(B)) - float(U)
    return approx_margin, B, C, gap_square


def run():
    start = time.monotonic()
    a0 = NODES[0]
    if not (a0 == F(-97, 100) and 64 * (a0 + 1) == F(48, 25) and F(48, 25) < F(2)):
        raise RuntimeError("a0 Holden strip containment check failed")
    rep = enumerate_spectral_pencils()

    # Spot checks against direct integer chars.
    sample = list(rep.items())[:3]
    for (cm, d), H in sample:
        for a in (F(0), F(1, 2)):
            poly = scaled_charpoly(cm, d, a)
            qa = a.denominator
            t = qa - a.numerator
            direct = charpolys_newton_int64(np.array([(qa - t * H).astype(np.int64)]))[0]
            if tuple(map(int, direct)) != poly:
                raise RuntimeError(f"Spot check mismatch at a={a}")

    upper = []
    for j, a in enumerate(NODES):
        best = F(-1)
        for (cm, d), H in rep.items():
            poly = scaled_charpoly(cm, d, a)
            U = upper_spread_exact(poly, a, H)
            if U > best:
                best = U
        upper.append(best)
        print(f"node {j:2d}: a={str(a):>16s}, certified max non-B upper <={float(best):.12f}", flush=True)

    interval_records = []
    for i, (left, right) in enumerate(zip(NODES, NODES[1:]), 1):
        m = (left + right) / 2
        k = 4 if m < F(-5, 7) else (3 if m < F(1, 5) else 2)
        valid_k = (
            (right <= F(-5, 7)) if k == 4 else
            (left >= F(-5, 7) and right <= F(1, 5)) if k == 3 else
            (left >= F(1, 5))
        )
        if not valid_k:
            raise RuntimeError(f"Interval crosses switch point improperly: [{left}, {right}], k={k}")

        gl, B, Cl, gap_sq_l = tangent_dominates_at_endpoint(k, m, left, upper[i - 1])
        gr, B, Cr, gap_sq_r = tangent_dominates_at_endpoint(k, m, right, upper[i])
        min_margin = min(gl, gr)
        print(
            f"interval {i:2d}: [{left},{right}] k={k} "
            f"certified approx minimum margin={min_margin:.10g}",
            flush=True,
        )
        interval_records.append({
            "index": i,
            "left": str(left),
            "right": str(right),
            "k": k,
            "midpoint": str(m),
            "B": str(B),
            "C_left": str(Cl),
            "C_right": str(Cr),
            "gap_sq_left": str(gap_sq_l),
            "gap_sq_right": str(gap_sq_r),
            "approx_margin": min_margin,
        })

    print(f"All 15 intervals have a UNIFORM exact gap > {EXACT_MARGIN} over all non-B threshold patterns.")
    print(
        "EXACT 15-INTERVAL CERTIFICATION PASS; "
        "with the original Holden left strip and the near-one right interval, "
        "Fallat-Xing conjecture is proved for n=8 and all -1<=a<1.",
        flush=True,
    )
    print(f"Elapsed: {time.monotonic()-start:.1f}s")

    # Export certified certificate for independent zero-dependency checking
    cert_data = {
        "problem": "SP-08",
        "dimension": 8,
        "nodes": [str(x) for x in NODES],
        "upper_bounds": [str(u) for u in upper],
        "exact_margin": str(EXACT_MARGIN),
        "intervals": interval_records,
        "holden_left_boundary": "-97/100",
        "near_one_right_boundary": "499/500",
        "verified_domain": "[-1, 1)",
    }
    with open("n8_certified_data.json", "w") as f:
        json.dump(cert_data, f, indent=2)
    print("Exported exact certificate to n8_certified_data.json")


if __name__ == "__main__":
    run()

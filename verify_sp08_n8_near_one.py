#!/usr/bin/env python3
"""SP-08: standalone exact integer certificate for all a in [499/500, 1), n=8.

Only the signed-threshold reduction (Holden, 2026) is assumed.  The code
runs Sylvester's positive-definiteness test on 7x7 INTEGER matrices for all
2^(2*8-1) signed-threshold candidates; no floating-point arithmetic is used.
The mathematical uniform remainder inequality and three exceptional patterns
are explained in SP08_n8_near_one_proof.md.

No external dependencies: Python 3 standard library only.
Hardened against python3 -O: uses explicit exceptions instead of assert.
"""
from fractions import Fraction
from collections import Counter

N = 8


def det_bareiss(matrix):
    """Exact integer determinant via fraction-free Gaussian elimination."""
    a = [list(map(int, row)) for row in matrix]
    n = len(a)
    previous = 1
    swaps = 0
    for k in range(n - 1):
        if a[k][k] == 0:
            pivot_row = next((i for i in range(k + 1, n) if a[i][k] != 0), None)
            if pivot_row is None:
                return 0
            a[k], a[pivot_row] = a[pivot_row], a[k]
            swaps += 1
        pivot = a[k][k]
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                numerator = pivot * a[i][j] - a[i][k] * a[k][j]
                if numerator % previous != 0:
                    raise RuntimeError("Bareiss division was not exact")
                a[i][j] = numerator // previous
        for i in range(k + 1, n):
            a[i][k] = 0
        previous = pivot
    return (-1 if swaps % 2 else 1) * a[-1][-1]


def leading_minor_test(a):
    """Return (positive_definite, first_nonpositive_leading_minor_size, value)."""
    for k in range(1, len(a) + 1):
        determinant = det_bareiss([row[:k] for row in a[:k]])
        if determinant <= 0:
            return (False, k, determinant)
    return (True, 0, 0)


def threshold_H(eps, d):
    # H_{ij}=(1-d_i*d_j*eps_min(i,j))/2, so J-delta H is the endpoint matrix.
    return [[(1 - d[i] * d[j] * eps[min(i, j)]) // 2 for j in range(N)] for i in range(N)]


def first_order_matrix(H, frac):
    """Integer multiple of E^T((h+c)I-H)E, h=sum(H)/8, c=frac.

    E is the 8x7 matrix with columns e_i-e_7. Thus the leading principal
    minors can be checked in Z. The matrix is PD iff
    lambda_max(PHP restricted to 1^perp)-h < c.
    """
    p, q = frac.numerator, frac.denominator
    s = sum(sum(row) for row in H)
    return [
        [
            (q * s + 8 * p) * (1 + int(i == j)) - 8 * q * (H[i][j] - H[i][7] - H[7][j] + H[7][7])
            for j in range(7)
        ]
        for i in range(7)
    ]


def block_structure(H):
    # For the five exceptional patterns: classify 2x2 / 1x1 binary diagonal blocks.
    indices = [i for i in range(N) if H[i][i]]
    if len(indices) == 2:
        e = [int(i in indices) for i in range(N)]
        if all(H[i][j] == e[i] * e[j] for i in range(N) for j in range(N)):
            return "B_8,2"
    if len(indices) == 4:
        a, b, c, d = indices
        e = [int(i in (a, b)) for i in range(N)]
        f = [int(i in (c, d)) for i in range(N)]
        if all(H[i][j] == e[i] * e[j] + f[i] * f[j] for i in range(N) for j in range(N)):
            return "J2+J2"
    if len(indices) == 3:
        for singleton in indices:
            pair = [i for i in indices if i != singleton]
            e = [int(i in pair) for i in range(N)]
            f = [int(i == singleton) for i in range(N)]
            if all(H[i][j] == e[i] * e[j] + f[i] * f[j] for i in range(N) for j in range(N)):
                return "J2+J1"
    return "OTHER"


def run():
    total = 0
    tie = []
    high = []
    certified_095 = 0
    certified_0963 = 0
    for e_mask in range(1 << N):
        eps = [-1 if e_mask >> i & 1 else 1 for i in range(N)]
        for d_mask in range(1 << (N - 1)):
            d = [1] + [-1 if d_mask >> (i - 1) & 1 else 1 for i in range(1, N)]
            H = threshold_H(eps, d)
            total += 1
            pass_1, j, val = leading_minor_test(first_order_matrix(H, Fraction(1)))
            if not pass_1:
                if (j, val) != (7, 0):
                    raise RuntimeError(f"Unexpected equality/non-PSD: {(e_mask, d_mask, j, val)}")
                tie.append((e_mask, d_mask, block_structure(H)))
                continue
            pass_0963, j, val = leading_minor_test(first_order_matrix(H, Fraction(963, 1000)))
            if not pass_0963:
                raise RuntimeError(f"c>0.963 not excluded for {(e_mask, d_mask, j, val)}")
            certified_0963 += 1
            pass_095, j, val = leading_minor_test(first_order_matrix(H, Fraction(19, 20)))
            if pass_095:
                certified_095 += 1
            else:
                high.append((e_mask, d_mask, block_structure(H)))

    if total != 32768:
        raise RuntimeError(f"Expected 32768 patterns, got {total}")
    if len(tie) != 3 or Counter(x[-1] for x in tie) != {"B_8,2": 2, "J2+J2": 1}:
        raise RuntimeError(f"Tie set mismatch: {tie}")
    if len(high) != 2 or not all(x[-1] == "J2+J1" for x in high):
        raise RuntimeError(f"High coefficient set mismatch: {high}")
    if certified_0963 != 32765 or certified_095 != 32763:
        raise RuntimeError("Certification counts mismatch")

    # Analytic inequality for 0 < delta <= 1/500:
    # spread(J-δH) <= 8 + δ(c1 + 16δ/(1-2δ)); for the 32765 non-ties
    # c1 < 963/1000. Then 963/1000 + 16/498 < 1.
    if Fraction(963, 1000) + Fraction(16, 498) >= 1:
        raise RuntimeError("Analytical bound 963/1000 + 16/498 < 1 failed")

    print(f"EXACT CERTIFIED: all {total} signed-threshold patterns for n=8.")
    print(f"  c1 < 19/20 : {certified_095} patterns")
    print(f"  19/20 <= c1 < 963/1000: {len(high)} patterns, both J2+J1")
    print(f"  c1 = 1 : {len(tie)} patterns: 2 x B_8,2 and 1 x J2+J2")
    print("  no patterns with 963/1000 <= c1 < 1; no patterns c1 > 1")
    print("  second-best exact coefficient = (9+sqrt(41))/16 (J2+J1)")
    print("  near-one sufficient interval: a in [499/500, 1)")
    print("  all conditions verified using integer arithmetic only")


if __name__ == "__main__":
    run()

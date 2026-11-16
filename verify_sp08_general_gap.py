#!/usr/bin/env python3
"""Independent checks for the all-runs signed-threshold spectral-gap theorem.
Analytic proof is in SP08_general_oscillation_theorem.md; this is diagnostic.
"""
import math
import numpy as np


def alternating_core(r):
    eps = np.array([1 if j % 2 == 0 else -1 for j in range(r)])
    return eps[np.minimum.outer(np.arange(r),np.arange(r))].astype(float)


def exact_spectrum(r):
    if r % 2:
        angles = [k * math.pi / r for k in range(r)]
    else:
        angles = [(2*k - 1) * math.pi / (2*r) for k in range(1, r+1)]
    return np.sort([1.0 / math.cos(t) for t in angles])


def delta(r):
    return 2*r*r - 4/(math.sin(math.pi/(2*r))**2)


def runs(eps):
    return 1 + sum(eps[j] != eps[j-1] for j in range(1,len(eps)))


def check():
    for r in range(2, 31):
        T = alternating_core(r)
        eigs = np.linalg.eigvalsh(T)
        expected = exact_spectrum(r)
        assert np.allclose(eigs, expected, rtol=2e-12, atol=2e-12), r
        q = sum(eigs[1:-1]**2)
        qformula = r*r - 2/(math.sin(math.pi/(2*r))**2)
        assert abs(q - qformula) < 2e-10 * r*r
        spread2 = (eigs[-1]-eigs[0])**2
        assert abs(spread2-(2*r*r-delta(r))) < 2e-10 * r*r
    rng = np.random.default_rng(20261115)
    for n in range(3, 31):
        for _ in range(150):
            eps = rng.choice([-1,1], size=n)
            signs = rng.choice([-1,1], size=n)
            S = eps[np.minimum.outer(np.arange(n),np.arange(n))]*np.outer(signs, signs)
            r = runs(eps)
            if r >= 3:
                ev = np.linalg.eigvalsh(S.astype(float))
                sp2 = (ev[-1]-ev[0])**2
                assert sp2 <= 2*n*n-delta(r)+1e-9*n*n, (n,r,sp2)
    print('PASS: exact core spectra, sharpness, and randomized threshold matrices (r=2..30, n=3..30).')
    print('r, delta_r')
    for r in range(3,11): print(r, f'{delta(r):.10f}')


if __name__ == '__main__':
    check()

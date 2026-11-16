#!/usr/bin/env python3
"""Independent extension of finite SP-08 certificates to neighborhoods in a.
Requires verify_sp08.py for the exact arithmetic kernels.
The classification of B-shaped patterns is from their sign-matrix entries.
Every tested inequality is exact integer arithmetic (float eig only proposes shifts).
"""
from __future__ import annotations
from fractions import Fraction
import math
import numpy as np
from verify_sp08 import signed_threshold_batches, charpolys_newton_int64, shifted_poly_nonnegative, negated_poly

CASES = [
    (8, -1, 2, Fraction(10), 409),
    (8, -1, 3, Fraction(97,10), 864),
    (8,  1, 3, Fraction(873,100), 688),
    (9,  1, 2, Fraction(959,100), 369),
    (9, -1, 2, Fraction(45,4), 516),
]

def is_B_shape(A: np.ndarray, lo:int,hi:int)->np.ndarray:
    n=A.shape[1]
    diagonal=np.arange(n)
    neg=A[:,diagonal,diagonal] == lo
    mask=(neg[:,:,None]&neg[:,None,:])
    return np.all((A==lo)==mask,axis=(1,2)) & np.any(neg,axis=1) & np.any(~neg,axis=1)

def certified_upper_bound(coeffs:tuple[int,...], A:np.ndarray, U:Fraction, D_values=(10**4,10**6,10**8,10**10)):
    c=coeffs; n=len(c)-1
    if all(z==0 for z in c[2:]):
        return c[1]*c[1]*U.denominator*U.denominator <= U.numerator*U.numerator
    if all(z==0 for z in c[3:]):
        if c[2]<0:
            return (c[1]**2-4*c[2])*U.denominator**2 <= U.numerator**2
        if c[2]>0:
            return c[1]*c[1]*U.denominator**2 <= U.numerator**2
    ev=np.linalg.eigvalsh(A.astype(float))
    for D in D_values:
        u_pos=math.ceil(float(ev[-1])*D)+3
        u_neg=math.ceil(float(-ev[0])*D)+3
        if (u_pos+u_neg)>=0 and (u_pos+u_neg)*U.denominator <= U.numerator*D:
            if shifted_poly_nonnegative(c,u_pos,D) and shifted_poly_nonnegative(negated_poly(c),u_neg,D):
                return True
    return False

def verify(n:int,lo:int,hi:int,U:Fraction,best_num:int):
    distinct={}
    all_count=0; B_count=0
    for ids, A in signed_threshold_batches(n,lo,hi):
        is_B=is_B_shape(A,lo,hi)
        B_count += int(is_B.sum())
        all_count += len(A)
        coeffs=charpolys_newton_int64(A)
        for i in np.flatnonzero(~is_B):
            key=tuple(map(int,coeffs[i]))
            if key not in distinct:
                distinct[key]=A[i].copy()
    assert (U.numerator*hi)**2 < best_num*U.denominator**2
    fail=[]
    for c,A in distinct.items():
        if not certified_upper_bound(c,A,U*hi): fail.append(c)
    assert not fail, f'Failed n={n}, ratio={lo}/{hi}, U={U}: {len(fail)} distinct counterexamples.'
    assert U.numerator**2*hi**2 < best_num*U.denominator**2
    print(f'CERTIFIED: (n={n}, a={lo}/{hi}): {all_count} threshold patterns, B-shape={B_count}, non-B distinct polynomials={len(distinct)}, U={U} ; every non-B spread <= {U}; M=sqrt({best_num})/{hi} > U; 0 failures.')
    return len(distinct)

if __name__=='__main__':
    for case in CASES: verify(*case)
    print('All five strict separation certificates passed.')

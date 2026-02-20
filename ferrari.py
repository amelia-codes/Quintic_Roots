#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 16:10:08 2024
"""

import math, cmath
import numpy as np
from numba import jit #A - precompiling!

@jit(nopython=True)
def single_quadratic(a0, b0, c0):

    a, b = b0 / a0, c0 / a0

    # Some repating variables
    a0 = -0.5*a
    delta = a0*a0 - b
    sqrt_delta = cmath.sqrt(delta)

    # Roots
    r1 = a0 - sqrt_delta
    r2 = a0 + sqrt_delta

    return r1, r2


@jit(nopython=True)
def single_cubic(a0, b0, c0, d0):

    a, b, c = b0 / a0, c0 / a0, d0 / a0

    # Some repeating constants and variables
    third = 1./3.
    a13 = a*third
    a2 = a13*a13
    sqr3 = math.sqrt(3)

    # Additional intermediate variables
    f = third*b - a2
    g = a13 * (2*a2 - b) + c
    h = 0.25*g*g + f*f*f

    def cubic_root(x):
        ''' Compute cubic root of a number while maintaining its sign'''
        if x.real >= 0:
            return x**third
        else:
            return -(-x)**third

    if f == g == h == 0:
        r1 = -cubic_root(c)
        return r1, r1, r1

    elif h <= 0:
        j = math.sqrt(-f)
        k = math.acos(-0.5*g / (j*j*j))
        m = math.cos(third*k)
        n = sqr3 * math.sin(third*k)
        r1 = 2*j*m - a13
        r2 = -j * (m + n) - a13
        r3 = -j * (m - n) - a13
        return r1, r2, r3

    else:
        sqrt_h = cmath.sqrt(h)
        S = cubic_root(-0.5*g + sqrt_h)
        U = cubic_root(-0.5*g - sqrt_h)
        S_plus_U = S + U
        S_minus_U = S - U
        r1 = S_plus_U - a13
        r2 = -0.5*S_plus_U - a13 + S_minus_U*sqr3*0.5j
        r3 = -0.5*S_plus_U - a13 - S_minus_U*sqr3*0.5j
        return r1, r2, r3


@jit(nopython=True)
def single_cubic_one(a0, b0, c0, d0):

    a, b, c = b0 / a0, c0 / a0, d0 / a0

    # Some repeating constants and variables
    third = 1./3.
    a13 = a*third
    a2 = a13*a13

    # Additional intermediate variables
    f = third*b - a2
    g = a13 * (2*a2 - b) + c
    h = 0.25*g*g + f*f*f

    def cubic_root(x):
        ''' Compute cubic root of a number while maintaining its sign
        '''
        if x.real >= 0:
            return x**third
        else:
            return -(-x)**third

    if f == g == h == 0:
        return -cubic_root(c)

    elif h <= 0:
        j = math.sqrt(-f)
        k = math.acos(-0.5*g / (j*j*j))
        m = math.cos(third*k)
        return 2*j*m - a13

    else:
        sqrt_h = cmath.sqrt(h)
        S = cubic_root(-0.5*g + sqrt_h)
        U = cubic_root(-0.5*g - sqrt_h)
        S_plus_U = S + U
        return S_plus_U - a13


@jit(nopython=True)
def single_quartic(a0, b0, c0, d0, e0):

    a, b, c, d = b0/a0, c0/a0, d0/a0, e0/a0

    # Some repeating variables
    a0 = 0.25*a
    a02 = a0*a0

    # Coefficients of subsidiary cubic euqtion
    p = 3*a02 - 0.5*b
    q = a*a02 - b*a0 + 0.5*c
    r = 3*a02*a02 - b*a02 + c*a0 - d

    # One root of the cubic equation
    z0 = single_cubic_one(1, p, r, p*r - 0.5*q*q)

    # Additional variables
    s = cmath.sqrt(2*p + 2*z0.real + 0j)
    if s == 0:
        t = z0*z0 + r
    else:
        t = -q / s

    # Compute roots by quadratic equations
    r0, r1 = single_quadratic(1, s, z0 + t)
    r2, r3 = single_quadratic(1, -s, z0 - t)

    return r0 - a0, r1 - a0, r2 - a0, r3 - a0


def multi_quadratic(a0, b0, c0):

    a, b = b0 / a0, c0 / a0

    # Some repating variables
    a0 = -0.5*a
    delta = a0*a0 - b
    sqrt_delta = np.sqrt(delta + 0j)

    # Roots
    r1 = a0 - sqrt_delta
    r2 = a0 + sqrt_delta

    return r1, r2


def multi_cubic(a0, b0, c0, d0, all_roots=True):

    a, b, c = b0 / a0, c0 / a0, d0 / a0

    # Some repeating constants and variables
    third = 1./3.
    a13 = a*third
    a2 = a13*a13
    sqr3 = math.sqrt(3)

    # Additional intermediate variables
    f = third*b - a2
    g = a13 * (2*a2 - b) + c
    h = 0.25*g*g + f*f*f

    # Masks for different combinations of roots
    m1 = (f == 0) & (g == 0) & (h == 0)     # roots are real and equal
    m2 = (~m1) & (h <= 0)                   # roots are real and distinct
    m3 = (~m1) & (~m2)                      # one real root and two complex

    def cubic_root(x):
        ''' Compute cubic root of a number while maintaining its sign
        '''
        root = np.zeros_like(x)
        positive = (x >= 0)
        negative = ~positive
        root[positive] = x[positive]**third
        root[negative] = -(-x[negative])**third
        return root

    def roots_all_real_equal(c):
        ''' Compute cubic roots if all roots are real and equal
        '''
        r1 = -cubic_root(c)
        if all_roots:
            return r1, r1, r1
        else:
            return r1

    def roots_all_real_distinct(a13, f, g, h):
        ''' Compute cubic roots if all roots are real and distinct
        '''
        j = np.sqrt(-f)
        k = np.arccos(-0.5*g / (j*j*j))
        m = np.cos(third*k)
        r1 = 2*j*m - a13
        if all_roots:
            n = sqr3 * np.sin(third*k)
            r2 = -j * (m + n) - a13
            r3 = -j * (m - n) - a13
            return r1, r2, r3
        else:
            return r1

    def roots_one_real(a13, g, h):
        ''' Compute cubic roots if one root is real and other two are complex
        '''
        sqrt_h = np.sqrt(h)
        S = cubic_root(-0.5*g + sqrt_h)
        U = cubic_root(-0.5*g - sqrt_h)
        S_plus_U = S + U
        r1 = S_plus_U - a13
        if all_roots:
            S_minus_U = S - U
            r2 = -0.5*S_plus_U - a13 + S_minus_U*sqr3*0.5j
            r3 = -0.5*S_plus_U - a13 - S_minus_U*sqr3*0.5j
            return r1, r2, r3
        else:
            return r1

    # Compute roots
    if all_roots:
        roots = np.zeros((3, len(a))).astype(complex)
        roots[:, m1] = roots_all_real_equal(c[m1])
        roots[:, m2] = roots_all_real_distinct(a13[m2], f[m2], g[m2], h[m2])
        roots[:, m3] = roots_one_real(a13[m3], g[m3], h[m3])
    else:
        roots = np.zeros(len(a))  # .astype(complex)
        roots[m1] = roots_all_real_equal(c[m1])
        roots[m2] = roots_all_real_distinct(a13[m2], f[m2], g[m2], h[m2])
        roots[m3] = roots_one_real(a13[m3], g[m3], h[m3])

    return roots


def multi_quartic(a0, b0, c0, d0, e0):

    a, b, c, d = b0/a0, c0/a0, d0/a0, e0/a0

    # Some repeating variables
    a0 = 0.25*a
    a02 = a0*a0

    # Coefficients of subsidiary cubic euqtion
    p = 3*a02 - 0.5*b
    q = a*a02 - b*a0 + 0.5*c
    r = 3*a02*a02 - b*a02 + c*a0 - d

    # One root of the cubic equation
    z0 = multi_cubic(1, p, r, p*r - 0.5*q*q, all_roots=False)

    # Additional variables
    s = np.sqrt(2*p + 2*z0.real + 0j)
    t = np.zeros_like(s)
    mask = (s == 0)
    t[mask] = z0[mask]*z0[mask] + r[mask]
    t[~mask] = -q[~mask] / s[~mask]

    # Compute roots by quadratic equations
    r0, r1 = multi_quadratic(1, s, z0 + t) - a0
    r2, r3 = multi_quadratic(1, -s, z0 - t) - a0

    return r0, r1, r2, r3


def cubic_roots(p):
    
    if p.ndim < 2:
        p = p[np.newaxis, :]

    if p.shape[0] < 100:
        roots = [single_cubic(*pi) for pi in p]
        return np.array(roots)
    else:
        roots = multi_cubic(*p.T)
        return np.array(roots).T

def quartic_roots(p):

    # Convert input to an array (if input is a list or tuple)
    p = np.asarray(p)

    # If only one set of coefficients is given, add axis
    if p.ndim < 2:
        p = p[np.newaxis, :]

    # Check if all five coefficients are given
    #if p.shape[1] != 5:
       # raise ValueError('Expected 4th order polynomial with 5 '
              #           'coefficients, got {:d}.'.format(p.shape[1]))

    if p.shape[0] < 100:
        roots = [single_quartic(*pi) for pi in p]
        return np.array(roots)
    else:
        roots = multi_quartic(*p.T)
        return np.array(roots).T

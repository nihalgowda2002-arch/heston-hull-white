"""Heston characteristic function and call price by Gil-Pelaez inversion.

Implements Proposition 4.5 (characteristic function), Remark 4.13 (the
corrected form that avoids the branch cut of the complex logarithm) and
equation (5.2) (Gauss-Legendre quadrature of the two exercise
probabilities) of the dissertation.
"""

from functools import lru_cache

import numpy as np

_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # NumPy 2 renamed trapz


def heston_cf(u, tau, kappa, nubar, eta, rho, nu0, S0, r, corrected=True):
    """Characteristic function of ln S_T under the risk-neutral measure.

    u may be complex (u - i gives the share-measure function). With
    corrected=False the original published form is used: it agrees with the
    corrected form at short maturities but jumps when the complex logarithm
    leaves its principal branch at long maturities.
    """
    u = np.asarray(u, complex)
    b = rho * eta * 1j * u - kappa
    d = np.sqrt(b * b + eta * eta * (u * u + 1j * u))
    d_plus, d_minus = (-b + d) / eta**2, (-b - d) / eta**2
    if corrected:
        g, e, d0 = d_minus / d_plus, np.exp(-d * tau), d_minus
    else:
        g, e, d0 = d_plus / d_minus, np.exp(d * tau), d_plus
    D = d0 * (1 - e) / (1 - g * e)
    C = r * 1j * u * tau + (kappa * nubar / eta**2) * (
        eta**2 * d0 * tau - 2 * np.log((1 - g * e) / (1 - g))
    )
    return np.exp(C + D * nu0 + 1j * u * np.log(S0))


@lru_cache(maxsize=32)
def _legendre(N):
    """Gauss-Legendre nodes and weights on [-1, 1], computed once per N."""
    x, w = np.polynomial.legendre.leggauss(int(N))
    x.setflags(write=False)
    w.setflags(write=False)
    return x, w


def _nodes(N, U):
    """Nodes and weights mapped from [-1, 1] to (0, U]."""
    x, w = _legendre(N)
    return 0.5 * U * (x + 1.0), 0.5 * U * w


def heston_probabilities(K, tau, kappa, nubar, eta, rho, nu0, S0, r,
                         N=160, U=200.0, corrected=True):
    """Exercise probabilities P1 (share measure) and P2 (risk-neutral)."""
    u, w = _nodes(N, U)
    K = np.atleast_1d(np.asarray(K, float))
    p2 = heston_cf(u, tau, kappa, nubar, eta, rho, nu0, S0, r, corrected)
    p1 = heston_cf(u - 1j, tau, kappa, nubar, eta, rho, nu0, S0, r, corrected) / (
        S0 * np.exp(r * tau)
    )
    kernel = np.exp(-1j * u[None, :] * np.log(K)[:, None]) / (1j * u[None, :])
    P1 = 0.5 + (w * (kernel * p1[None, :]).real).sum(axis=1) / np.pi
    P2 = 0.5 + (w * (kernel * p2[None, :]).real).sum(axis=1) / np.pi
    return P1, P2


def heston_call(K, tau, kappa, nubar, eta, rho, nu0, S0, r,
                N=160, U=200.0, corrected=True):
    """Heston call prices for an array of strikes: C = S P1 - K e^{-r tau} P2."""
    K = np.atleast_1d(np.asarray(K, float))
    P1, P2 = heston_probabilities(K, tau, kappa, nubar, eta, rho, nu0, S0, r,
                                  N, U, corrected)
    return S0 * P1 - K * np.exp(-r * tau) * P2


def heston_put(K, tau, kappa, nubar, eta, rho, nu0, S0, r,
               N=160, U=200.0, corrected=True):
    """Heston put from its own pair of inversions (not from put-call parity)."""
    K = np.atleast_1d(np.asarray(K, float))
    P1, P2 = heston_probabilities(K, tau, kappa, nubar, eta, rho, nu0, S0, r,
                                  N, U, corrected)
    return K * np.exp(-r * tau) * (1 - P2) - S0 * (1 - P1)


def heston_call_fine(K, tau, kappa, nubar, eta, rho, nu0, S0, r,
                     U=200.0, M=400_000, corrected=True):
    """One call price by the trapezoidal rule on a fine uniform grid.

    Used to show where the original form actually lands. Its integrand jumps
    wherever the logarithm crosses a branch, and Gauss-Legendre relies on a
    smooth integrand, so no fixed-order rule converges there; a dense uniform
    grid does. Settings follow Section A.7 of the dissertation.
    """
    u = np.linspace(1e-10, U, M)
    with np.errstate(all="ignore"):
        p2 = heston_cf(u, tau, kappa, nubar, eta, rho, nu0, S0, r, corrected)
        p1 = heston_cf(u - 1j, tau, kappa, nubar, eta, rho, nu0, S0, r, corrected) / (
            S0 * np.exp(r * tau)
        )
        kernel = np.exp(-1j * u * np.log(K)) / (1j * u)
        f1 = np.nan_to_num((kernel * p1).real)
        f2 = np.nan_to_num((kernel * p2).real)
    P1 = 0.5 + _trapezoid(f1, u) / np.pi
    P2 = 0.5 + _trapezoid(f2, u) / np.pi
    return float(S0 * P1 - K * np.exp(-r * tau) * P2)

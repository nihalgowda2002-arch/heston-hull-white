"""Monte Carlo simulation of the square-root variance and of the Heston model.

Section 5.3 of the dissertation: the full-truncation Euler scheme of Lord,
Koekkoek and van Dijk, and the quadratic-exponential (QE) scheme of Andersen.
Both stay non-negative when the Feller condition 2 kappa nubar >= eta^2 fails.
"""

import numpy as np
from scipy.special import ndtri


def cir_euler_ft(nu0, kappa, nubar, eta, T, n, steps, rng):
    """Full-truncation Euler: drift and diffusion see max(nu, 0)."""
    dt = T / steps
    nu = np.full(n, float(nu0))
    for _ in range(steps):
        nup = np.maximum(nu, 0.0)
        nu = nu + kappa * (nubar - nup) * dt + eta * np.sqrt(nup * dt) * rng.standard_normal(n)
    return np.maximum(nu, 0.0)


def qe_step(nu, kappa, nubar, eta, dt, uniforms, psi_c=1.5):
    """One step of Andersen's quadratic-exponential scheme.

    Matches the conditional mean m and variance s2 of the exact transition,
    sampling a scaled non-central square when psi = s2/m^2 <= psi_c and an
    exponential law with an atom at zero otherwise.
    """
    E = np.exp(-kappa * dt)
    m = nubar + (nu - nubar) * E
    s2 = nu * eta**2 * E * (1 - E) / kappa + nubar * eta**2 * (1 - E) ** 2 / (2 * kappa)
    psi = s2 / np.maximum(m * m, 1e-300)
    out = np.empty_like(nu)
    lo = psi <= psi_c
    if lo.any():
        binv = 2.0 / psi[lo] - 1.0
        b2 = binv + np.sqrt(binv * (binv + 1.0))
        a = m[lo] / (1.0 + b2)
        z = ndtri(np.clip(uniforms[lo], 1e-12, 1 - 1e-12))
        out[lo] = a * (np.sqrt(b2) + z) ** 2
    hi = ~lo
    if hi.any():
        p = (psi[hi] - 1.0) / (psi[hi] + 1.0)
        beta = (1.0 - p) / np.maximum(m[hi], 1e-300)
        u = uniforms[hi]
        out[hi] = np.where(u <= p, 0.0,
                           np.log((1.0 - p) / np.maximum(1.0 - u, 1e-300)) / beta)
    return out


def cir_qe(nu0, kappa, nubar, eta, T, n, steps, rng, psi_c=1.5):
    """Andersen QE scheme for the square-root variance."""
    dt = T / steps
    nu = np.full(n, float(nu0))
    for _ in range(steps):
        nu = qe_step(nu, kappa, nubar, eta, dt, rng.random(n), psi_c)
    return nu


def cir_exact_moments(nu0, kappa, nubar, eta, T):
    """Exact mean and variance of the square-root diffusion at time T."""
    E = np.exp(-kappa * T)
    mean = nubar + (nu0 - nubar) * E
    var = nu0 * eta**2 * E * (1 - E) / kappa + nubar * eta**2 * (1 - E) ** 2 / (2 * kappa)
    return mean, var


def heston_mc_call(S0, K, r, tau, kappa, nubar, eta, rho, nu0,
                   n=200_000, steps=400, seed=11):
    """Heston call by Monte Carlo: log-Euler for the price, full-truncation
    Euler for the variance, correlated Gaussian increments.

    Returns the price and its standard error.
    """
    rng = np.random.default_rng(seed)
    dt = tau / steps
    x = np.full(n, np.log(S0))
    nu = np.full(n, float(nu0))
    sq = np.sqrt(1.0 - rho * rho)
    for _ in range(steps):
        z1 = rng.standard_normal(n)
        z2 = rho * z1 + sq * rng.standard_normal(n)
        nup = np.maximum(nu, 0.0)
        x = x + (r - 0.5 * nup) * dt + np.sqrt(nup * dt) * z1
        nu = nu + kappa * (nubar - nup) * dt + eta * np.sqrt(nup * dt) * z2
    payoff = np.exp(-r * tau) * np.maximum(np.exp(x) - K, 0.0)
    return payoff.mean(), payoff.std(ddof=1) / np.sqrt(n)

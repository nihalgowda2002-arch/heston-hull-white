"""Synthetic reference surface and calibration of both models (Sections 5.4, 5.5).

The reference surface is generated from the Heston model at known parameters
and perturbed by Gaussian noise of 0.2 volatility points, so the calibration
tests the implementation and the identifiability of the parameters. It does
not test whether Heston fits real market data.
"""

import numpy as np
from scipy.optimize import differential_evolution, least_squares

from .blackscholes import implied_vol
from .heston import heston_call
from .hullwhite import hw_implied_vols, hw_normals

S0, R = 100.0, 0.02
GENERATING = dict(kappa=1.5, nubar=0.045, eta=0.55, rho=-0.75, nu0=0.032)
MATURITIES = (0.25, 1.0, 2.0)
LOG_MONEYNESS = np.array([-0.25, -0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15])
STRIKES = S0 * np.exp(LOG_MONEYNESS)
HESTON_NAMES = ("kappa", "nubar", "eta", "rho", "nu0")
HESTON_BOUNDS = [(0.2, 8.0), (0.01, 0.20), (0.05, 1.50), (-0.95, 0.20), (0.005, 0.20)]
HW_NAMES = ("nu0", "mbar", "eta")
HW_BOUNDS = [(0.005, 0.20), (-2.0, 2.0), (0.05, 2.0)]

# Calibrated values as printed in the dissertation (Table 5.1 and Section A.3).
# The later tables and figures use them, exactly as the dissertation's own
# listings do, so every printed number can be checked digit for digit.
HESTON_CALIBRATED = dict(kappa=1.614449, nubar=0.043346, eta=0.543077,
                         rho=-0.755296, nu0=0.032122)
HW_CALIBRATED = dict(nu0=0.030544, mbar=0.764633, eta=2.0)


def reference_surface(params=GENERATING, sd=0.002, seed=20240905):
    """Implied volatilities from Heston at `params`, plus N(0, sd^2) noise."""
    rng = np.random.default_rng(seed)
    quotes = {}
    for tau in MATURITIES:
        prices = heston_call(STRIKES, tau, **params, S0=S0, r=R)
        quotes[tau] = implied_vol(prices, S0, STRIKES, R, tau) + rng.normal(0.0, sd, STRIKES.size)
    return quotes


def heston_vols(params, tau):
    """Heston implied volatilities at the nine reference strikes."""
    prices = heston_call(STRIKES, tau, **params, S0=S0, r=R)
    return implied_vol(np.maximum(prices, 1e-12), S0, STRIKES, R, tau)


def heston_residuals(theta, quotes):
    """Model minus reference implied volatility, quote by quote (27 values)."""
    params = dict(zip(HESTON_NAMES, theta))
    out = [heston_vols(params, tau) - quotes[tau] for tau in MATURITIES]
    return np.nan_to_num(np.concatenate(out), nan=10.0)


def _two_stage(residuals, bounds, seed, maxiter, popsize, tol):
    """Differential evolution for the basin, then trust-region least squares."""
    de = differential_evolution(lambda th: float(np.sum(residuals(th) ** 2)), bounds,
                                seed=seed, maxiter=maxiter, popsize=popsize,
                                tol=1e-10, polish=False)
    lower, upper = zip(*bounds)
    fit = least_squares(residuals, de.x, bounds=(lower, upper), xtol=tol, ftol=tol)
    return fit.x, float(np.sqrt(np.mean(fit.fun ** 2)))


def calibrate_heston(quotes, seed=20240905, maxiter=45, popsize=14):
    """Five Heston parameters fitted to the 27 quotes (Table 5.1)."""
    x, rmse = _two_stage(lambda th: heston_residuals(th, quotes), HESTON_BOUNDS,
                         seed, maxiter, popsize, 1e-14)
    return dict(zip(HESTON_NAMES, x)), rmse


def hw_residuals(theta, quotes, normals):
    """Hull-White minus reference implied volatility, quote by quote."""
    nu0, mbar, eta = theta
    out = [hw_implied_vols(S0, STRIKES, R, tau, nu0, mbar, eta, normals) - quotes[tau]
           for tau in MATURITIES]
    return np.nan_to_num(np.concatenate(out), nan=10.0)


def calibrate_hull_white(quotes, normals=None, seed=20240905, maxiter=30, popsize=12):
    """Hull-White (nu0, mbar, eta) fitted to the same quotes by the same criterion.

    The same normals are reused at every parameter set (common random numbers).
    """
    normals = hw_normals() if normals is None else normals
    x, rmse = _two_stage(lambda th: hw_residuals(th, quotes, normals), HW_BOUNDS,
                         seed, maxiter, popsize, 1e-12)
    return dict(zip(HW_NAMES, x)), rmse

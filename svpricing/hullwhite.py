"""Hull-White (1987) stochastic volatility call price, independent noises.

Proposition 3.9 of the dissertation: conditioning on the volatility path makes
the asset lognormal, so the call price is the Black-Scholes price averaged
over the law of the average variance V. The variance is a geometric Brownian
motion, stepped by its exact solution; only the time integral is discretised
(left-point rule, bias of order dt).
"""

import numpy as np

from .blackscholes import bs_call, implied_vol


def average_variance(nu0, mbar, eta, tau, steps, normal_rows):
    """Average variance over [0, tau] on each path.

    normal_rows yields one array of standard normals per time step, in time
    order: a (steps, paths) array, or a generator that draws as it goes.
    """
    dt = tau / steps
    drift, vol = (mbar - eta * eta / 2) * dt, eta * np.sqrt(dt)
    v = acc = None
    for z in normal_rows:
        if v is None:
            v, acc = np.full(z.shape, float(nu0)), np.zeros(z.shape)
        acc += v * dt
        v = v * np.exp(drift + vol * z)
    return acc / tau


def hw_call(S0, K, r, tau, nu0, mbar, eta, n=200_000, steps=200, seed=20240905):
    """Hull-White call, E[ C_BS(S0, K, r, tau, sqrt(V)) ], by Monte Carlo.

    Returns (price, standard error); K may be an array (common paths).
    """
    rng = np.random.default_rng(seed)
    vbar = average_variance(nu0, mbar, eta, tau, steps,
                            (rng.standard_normal(n) for _ in range(steps)))
    K = np.atleast_1d(np.asarray(K, float))
    px = bs_call(S0, K[None, :], r, tau, np.sqrt(np.maximum(vbar, 1e-14))[:, None])
    return px.mean(axis=0), px.std(axis=0, ddof=1) / np.sqrt(px.shape[0])


def hw_normals(n=40_000, steps=160, seed=4321):
    """The fixed draws behind every Hull-White implied volatility (Section A.3)."""
    return np.random.default_rng(seed).standard_normal((steps, n))


def hw_implied_vols(S0, K, r, tau, nu0, mbar, eta, normals=None, bins=400):
    """Hull-White implied volatilities on common random numbers.

    The simulated average variances are sorted and averaged within `bins`
    equal groups, so each strike needs `bins` Black-Scholes evaluations rather
    than one per path. Reusing the same normals at every parameter set keeps
    the calibration objective a smooth function of the parameters.
    """
    if normals is None:
        normals = hw_normals()
    steps = normals.shape[0]
    vbar = average_variance(nu0, mbar, eta, tau, steps, normals)
    sig = np.sqrt(np.sort(vbar).reshape(bins, -1).mean(axis=1))
    K = np.atleast_1d(np.asarray(K, float))
    price = bs_call(S0, K[:, None], r, tau, sig[None, :]).mean(axis=1)
    return implied_vol(price, S0, K, r, tau)

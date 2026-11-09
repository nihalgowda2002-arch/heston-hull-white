"""Black-Scholes call price and implied volatility (Definition 2.4 of the dissertation)."""

import numpy as np
from scipy.special import ndtr


def bs_call(S, K, r, tau, sigma):
    """Black-Scholes price of a European call.

    S, K: spot and strike; r: continuously compounded rate; tau: time to
    maturity in years; sigma: volatility. Arrays broadcast.
    """
    S, K, tau = np.asarray(S, float), np.asarray(K, float), np.asarray(tau, float)
    sigma = np.maximum(np.asarray(sigma, float), 1e-12)
    srt = sigma * np.sqrt(tau)
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * tau) / srt
    return S * ndtr(d1) - K * np.exp(-r * tau) * ndtr(d1 - srt)


def implied_vol(price, S, K, r, tau, lo=1e-4, hi=3.0, iters=80):
    """Implied volatility by bisection.

    The call price is increasing in volatility, so bisection on [lo, hi]
    always converges. Vectorised over arrays of prices and strikes.
    """
    price = np.asarray(price, float)
    lo = np.full(np.broadcast(price, K).shape, lo)
    hi = np.full_like(lo, hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        below = bs_call(S, K, r, tau, mid) < price
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi)

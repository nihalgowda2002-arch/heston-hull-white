"""Fast checks of the pricing chain. Run with: pytest -q"""

import numpy as np
import pytest
from scipy.optimize import least_squares

from svpricing.blackscholes import bs_call, implied_vol
from svpricing.calibration import (
    GENERATING, HESTON_BOUNDS, HESTON_CALIBRATED, HESTON_NAMES, R, S0,
    heston_residuals, reference_surface,
)
from svpricing.heston import heston_call, heston_call_fine, heston_cf, heston_put
from svpricing.hullwhite import hw_call
from svpricing.simulation import cir_euler_ft, cir_exact_moments, cir_qe, heston_mc_call

TAB = dict(kappa=2.0, nubar=0.04, eta=0.5, rho=-0.7, nu0=0.03)


# Black-Scholes building blocks

def test_black_scholes_reference_value():
    assert bs_call(100.0, 100.0, 0.02, 1.0, 0.2) == pytest.approx(8.916037, abs=1e-6)


def test_implied_vol_inverts_black_scholes():
    K = np.array([70.0, 90.0, 100.0, 115.0, 140.0])
    sigma = np.array([0.35, 0.25, 0.2, 0.18, 0.3])
    price = bs_call(S0, K, R, 1.5, sigma)
    assert np.allclose(implied_vol(price, S0, K, R, 1.5), sigma, atol=1e-10)


# Heston characteristic function and Fourier inversion

def test_characteristic_function_normalisation_and_martingale():
    tau = 1.3
    assert heston_cf(0.0, tau, **TAB, S0=S0, r=R) == pytest.approx(1.0, abs=1e-14)
    # u = -i gives E[S_T], which must equal the forward S0 e^{r tau}
    forward = heston_cf(-1j, tau, **TAB, S0=S0, r=R)
    assert forward.real == pytest.approx(S0 * np.exp(R * tau), rel=1e-12)
    assert abs(forward.imag) < 1e-10


def test_heston_tends_to_black_scholes_as_vol_of_vol_vanishes():
    k, nb, v0, tau = TAB["kappa"], TAB["nubar"], TAB["nu0"], 1.0
    vbar = nb + (v0 - nb) * (1 - np.exp(-k * tau)) / (k * tau)
    target = bs_call(S0, 100.0, R, tau, np.sqrt(vbar))
    price = heston_call([100.0], tau, k, nb, 1e-4, TAB["rho"], v0, S0, R)[0]
    assert price == pytest.approx(target, abs=1e-5)


def test_put_call_parity_from_independent_inversions():
    K = np.arange(70.0, 131.0, 5.0)
    c = heston_call(K, 1.5, **TAB, S0=S0, r=R)
    p = heston_put(K, 1.5, **TAB, S0=S0, r=R)
    assert np.max(np.abs(c - p - (S0 - K * np.exp(-R * 1.5)))) < 1e-12


def test_table_5_5_fourier_column():
    expected = {(90, 1.0): 14.7816, (100, 1.0): 7.9222, (110, 1.0): 3.1676,
                (100, 2.0): 12.0753, (120, 2.0): 3.6489}
    for (K, tau), value in expected.items():
        assert heston_call([K], tau, **TAB, S0=S0, r=R)[0] == pytest.approx(value, abs=5e-5)


def test_original_form_agrees_at_short_maturity():
    K = np.array([90.0, 100.0, 110.0])
    corrected = heston_call(K, 0.5, **TAB, S0=S0, r=R)
    original = heston_call(K, 0.5, **TAB, S0=S0, r=R, corrected=False)
    assert np.max(np.abs(corrected - original)) < 1e-8


def test_original_form_fails_past_the_branch_cut():
    # Table 5.4: at K = 100, tau = 2 the original form returns 17.9913, the true price is 12.0753
    original = heston_call_fine(100.0, 2.0, **TAB, S0=S0, r=R, corrected=False)
    corrected = heston_call([100.0], 2.0, **TAB, S0=S0, r=R)[0]
    assert original == pytest.approx(17.9913, abs=5e-4)
    assert corrected == pytest.approx(12.0753, abs=5e-5)


# Simulation

def test_variance_schemes_match_exact_moments_and_stay_non_negative():
    mean, var = cir_exact_moments(0.03, 2.0, 0.04, 0.5, 1.0)
    for scheme in (cir_euler_ft, cir_qe):
        nu = scheme(0.03, 2.0, 0.04, 0.5, 1.0, 100_000, 100, np.random.default_rng(7))
        assert nu.min() >= 0.0
        assert abs(nu.mean() - mean) < 4 * np.sqrt(var / nu.size) + 2e-4
        assert nu.var() == pytest.approx(var, rel=0.05)


def test_monte_carlo_agrees_with_fourier():
    fourier = heston_call([100.0], 1.0, **TAB, S0=S0, r=R)[0]
    mc, se = heston_mc_call(S0, 100.0, R, 1.0, **TAB, n=50_000, steps=100, seed=3)
    assert abs(mc - fourier) < 3 * se + 0.02  # 0.02 allows for the Euler bias at 100 steps


def test_hull_white_tends_to_black_scholes_as_vol_of_vol_vanishes():
    price, se = hw_call(S0, 100.0, R, 1.0, 0.04, 0.0, 1e-3, n=20_000, steps=50, seed=1)
    assert price[0] == pytest.approx(bs_call(S0, 100.0, R, 1.0, 0.2), abs=1e-3)


# Calibration

def test_calibration_recovers_noise_free_parameters():
    quotes = reference_surface(sd=0.0)
    lower, upper = zip(*HESTON_BOUNDS)
    start = [2.5, 0.06, 0.4, -0.5, 0.04]
    fit = least_squares(heston_residuals, start, args=(quotes,), bounds=(lower, upper),
                        xtol=1e-14, ftol=1e-14)
    assert np.sqrt(np.mean(fit.fun ** 2)) < 1e-7
    assert np.allclose(fit.x, [GENERATING[k] for k in HESTON_NAMES], rtol=1e-4)


def test_dissertation_parameters_fit_the_noisy_surface():
    quotes = reference_surface()
    theta = [HESTON_CALIBRATED[k] for k in HESTON_NAMES]
    rmse = np.sqrt(np.mean(heston_residuals(theta, quotes) ** 2))
    assert 100 * rmse == pytest.approx(0.235, abs=5e-4)

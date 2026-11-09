"""Reproduce every table and figure of Chapter 5 of the dissertation.

Run from the repository root:

    python scripts/reproduce.py

Tables are written to results/ as Markdown and figures to figures/ as PNG.
Every random number comes from a fixed seed, so each run prints the same
values (timings aside, which depend on the machine). A full run takes a few
minutes, most of it in the two calibrations.
"""

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from svpricing.blackscholes import bs_call, implied_vol  # noqa: E402
from svpricing.calibration import (  # noqa: E402
    GENERATING, HESTON_CALIBRATED, HESTON_NAMES, HW_CALIBRATED, LOG_MONEYNESS,
    MATURITIES, R, S0, STRIKES, calibrate_heston, calibrate_hull_white,
    heston_vols, hw_residuals, reference_surface,
)
from svpricing.heston import heston_call, heston_call_fine, heston_put  # noqa: E402
from svpricing.hullwhite import hw_call, hw_implied_vols, hw_normals  # noqa: E402
from svpricing.simulation import (  # noqa: E402
    cir_euler_ft, cir_exact_moments, cir_qe, heston_mc_call,
)

RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

TAB = dict(kappa=2.0, nubar=0.04, eta=0.5, rho=-0.7, nu0=0.03)  # Tables 5.4 and 5.5
HES, HW = HESTON_CALIBRATED, HW_CALIBRATED

BLUE, ORANGE, INK = "#0072B2", "#D55E00", "#333333"  # colour-blind safe pair
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": "#888888", "axes.grid": True,
    "grid.color": "#DDDDDD", "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "savefig.dpi": 200, "legend.frameon": False,
})


def write(name, title, lines):
    text = "\n".join([f"## {title}", ""] + lines) + "\n"
    (RESULTS / name).write_text(text)
    print(text)


def validation():
    """Section 5.2 and 5.3: the degenerate cases and the variance schemes."""
    k, nb, v0, t = TAB["kappa"], TAB["nubar"], TAB["nu0"], 1.0
    vbar = nb + (v0 - nb) * (1 - np.exp(-k * t)) / (k * t)  # average expected variance
    lines = ["| Check | Value |", "|---|---|",
             f"| Black-Scholes at the average variance, K = 100, tau = 1 | {float(bs_call(S0, 100.0, R, t, np.sqrt(vbar))):.6f} |"]
    for eta in (1e-2, 1e-3, 1e-4):
        h = heston_call([100.0], t, k, nb, eta, TAB["rho"], v0, S0, R)[0]
        lines.append(f"| Heston Fourier price, eta = {eta:g} | {h:.6f} |")
    hw, se = hw_call(S0, 100.0, R, 1.0, 0.04, 0.0, 1e-2)
    lines.append(f"| Hull-White Monte Carlo, eta = 0.01 (Black-Scholes at 20%: "
                 f"{float(bs_call(S0, 100.0, R, 1.0, 0.2)):.6f}) | {hw[0]:.6f} (s.e. {se[0]:.1e}) |")
    strikes = np.arange(70.0, 131.0, 5.0)
    c = heston_call(strikes, 1.5, **TAB, S0=S0, r=R)
    p = heston_put(strikes, 1.5, **TAB, S0=S0, r=R)
    parity = np.max(np.abs(c - p - (S0 - strikes * np.exp(-R * 1.5))))
    lines.append(f"| Put-call parity, strikes 70 to 130, tau = 1.5, put from its own inversion | max error {parity:.1e} |")
    m, v = cir_exact_moments(0.03, 2.0, 0.04, 0.5, 1.0)
    ft = cir_euler_ft(0.03, 2.0, 0.04, 0.5, 1.0, 400_000, 200, np.random.default_rng(4242))
    qe = cir_qe(0.03, 2.0, 0.04, 0.5, 1.0, 400_000, 200, np.random.default_rng(4242))
    lines += [f"| Variance at T = 1, exact mean and variance (Feller fails: 0.16 < 0.25) | {m:.6f}, {v:.6f} |",
              f"| Full-truncation Euler, 400,000 paths, 200 steps, seed 4242 | {ft.mean():.6f}, {ft.var():.6f} |",
              f"| Andersen QE, 400,000 paths, 200 steps, seed 4242 | {qe.mean():.6f}, {qe.var():.6f} |",
              f"| Negative variances produced (Euler, QE) | {int((ft < 0).sum())}, {int((qe < 0).sum())} |"]
    write("validation.md", "Validation checks (Sections 5.2 and 5.3)", lines)


def table_5_1(quotes):
    t0 = time.perf_counter()
    fit, rmse = calibrate_heston(quotes)
    secs = time.perf_counter() - t0
    fmt = dict(kappa="{:.3f}", nubar="{:.4f}", eta="{:.3f}", rho="{:.3f}", nu0="{:.4f}")
    row = lambda p: " | ".join(fmt[k].format(p[k]) for k in HESTON_NAMES)  # noqa: E731
    gap = max(abs(fit[k] - HES[k]) for k in HESTON_NAMES)
    write("table_5_1_calibration.md", "Table 5.1: Heston parameters recovered from the reference surface", [
        "| | kappa | nubar | eta | rho | nu0 |", "|---|---|---|---|---|---|",
        f"| generating value | {row(GENERATING)} |", f"| calibrated value | {row(fit)} |", "",
        f"RMSE on implied volatilities: {100 * rmse:.3f} volatility points (noise added: 0.200).",
        f"Feller check: 2 kappa nubar = {2 * fit['kappa'] * fit['nubar']:.3f} against eta^2 = {fit['eta'] ** 2:.3f}.",
        f"Largest gap to the six-decimal values printed in the dissertation: {gap:.1e}.",
        f"Differential evolution (45 generations, population factor 14) then least squares: {secs:.0f} s."])


def hull_white_fit(quotes, normals):
    t0 = time.perf_counter()
    fit, rmse = calibrate_hull_white(quotes, normals)
    secs = time.perf_counter() - t0
    stored = np.sqrt(np.mean(hw_residuals(tuple(HW.values()), quotes, normals) ** 2))
    write("hull_white_calibration.md", "Hull-White calibration (Section 5.5)", [
        "| | nu0 | mbar | eta | RMSE (vol. points) |", "|---|---|---|---|---|",
        f"| this run | {fit['nu0']:.6f} | {fit['mbar']:.6f} | {fit['eta']:.4f} | {100 * rmse:.4f} |",
        f"| dissertation, Section A.3 | {HW['nu0']:.6f} | {HW['mbar']:.6f} | {HW['eta']:.4f} | {100 * stored:.4f} |", "",
        "In both, the volatility of variance sits on the upper bound of the search region (2.0).",
        "The two parameter sets differ along a flat direction of the objective, where a higher",
        "initial variance offsets a lower drift, and their fits agree to within 0.001 volatility points.",
        f"Calibration time: {secs:.0f} s on 40,000 paths and 160 steps (common random numbers, seed 4321)."])


def table_5_2(quotes, normals):
    rows = []
    for tau in MATURITIES:
        h = 100 * (heston_vols(HES, tau) - quotes[tau])
        w = 100 * (hw_implied_vols(S0, STRIKES, R, tau, **HW, normals=normals) - quotes[tau])
        rows += list(zip(LOG_MONEYNESS, h, w))
    lm, h, w = map(np.array, zip(*rows))
    buckets = [("In the money", lm < -0.02), ("At the money", np.abs(lm) <= 0.02),
               ("Out of the money", lm > 0.02)]
    rms = lambda x: np.sqrt(np.mean(x ** 2))  # noqa: E731
    lines = ["| Moneyness | Quotes | Heston RMSE | Heston bias | Hull-White RMSE | Hull-White bias |",
             "|---|---|---|---|---|---|"]
    for name, m in buckets:
        lines.append(f"| {name} | {m.sum()} | {rms(h[m]):.3f} | {h[m].mean():+.3f} | {rms(w[m]):.3f} | {w[m].mean():+.3f} |")
    lines.append(f"| All | {lm.size} | {rms(h):.3f} | | {rms(w):.3f} | |")
    write("table_5_2_fit_by_moneyness.md",
          "Table 5.2: implied-volatility error by moneyness, in volatility points", lines)
    return rms(h), rms(w)


def figure_5_1(quotes, normals):
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), sharey=True)
    lm = np.linspace(-0.27, 0.17, 45)
    strikes = S0 * np.exp(lm)
    labels = {0.25: "0.25 years", 1.0: "1 year", 2.0: "2 years"}
    for ax, tau in zip(axes, MATURITIES):
        h_iv = implied_vol(heston_call(strikes, tau, **HES, S0=S0, r=R), S0, strikes, R, tau)
        w_iv = hw_implied_vols(S0, strikes, R, tau, **HW, normals=normals)
        ax.plot(lm, 100 * h_iv, color=BLUE, lw=1.8, label="Heston (calibrated)")
        ax.plot(lm, 100 * w_iv, color=ORANGE, lw=1.8, ls="--", label="Hull-White (calibrated)")
        ax.plot(LOG_MONEYNESS, 100 * quotes[tau], "o", ms=4.5, mfc="white", mec=INK,
                label="reference quotes")
        ax.set_title(f"maturity {labels[tau]}", fontsize=9)
        ax.set_xlabel("log-moneyness ln(K/S)")
    axes[0].set_ylabel("implied volatility (%)")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_5_1_smile.png")
    plt.close(fig)


def table_5_3(h_rmse, w_rmse):
    heston_call([100.0], 1.0, **TAB, S0=S0, r=R)  # warm the node cache
    t0 = time.perf_counter()
    for _ in range(500):
        heston_call([100.0], 1.0, **TAB, S0=S0, r=R)
    t_fourier = (time.perf_counter() - t0) / 500
    t0 = time.perf_counter()
    _, se = heston_mc_call(S0, 100.0, R, 1.0, **TAB)
    t_mc = time.perf_counter() - t0
    t0 = time.perf_counter()
    hw_call(S0, 100.0, R, 1.0, **HW)
    t_hw = time.perf_counter() - t0
    write("table_5_3_fit_and_cost.md", "Table 5.3: fit quality and cost of one call price", [
        "| Method | Fit RMSE (vol. points) | Time per price on this machine |", "|---|---|---|",
        f"| Heston, Fourier inversion (5.2) | {h_rmse:.3f} | {1e3 * t_fourier:.2f} ms |",
        f"| Heston, Monte Carlo (200,000 paths, 400 steps) | (as above) | {t_mc:.1f} s |",
        f"| Hull-White, Monte Carlo (200,000 paths, 200 steps) | {w_rmse:.3f} | {t_hw:.1f} s |", "",
        f"Monte Carlo standard error: {se:.3f}. Ratio, Monte Carlo to Fourier: {t_mc / t_fourier:,.0f}.",
        "Absolute timings depend on the hardware; the dissertation's run on Google Colab gave",
        "0.37 ms, 6.1 s and 2.8 s."])


def figure_5_2():
    ref = heston_call([100.0], 2.0, **HES, S0=S0, r=R, N=2000, U=600.0)[0]
    Us, Ns = np.arange(10, 201, 10), np.arange(8, 161, 8)
    err_u = [abs(heston_call([100.0], 2.0, **HES, S0=S0, r=R, N=400, U=u)[0] - ref) for u in Us]
    err_n = [abs(heston_call([100.0], 2.0, **HES, S0=S0, r=R, N=n, U=200.0)[0] - ref) for n in Ns]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    axes[0].semilogy(Us, np.maximum(err_u, 1e-16), "o-", ms=3, color=BLUE)
    axes[0].set_xlabel("truncation bound U (400 nodes)")
    axes[1].semilogy(Ns, np.maximum(err_n, 1e-16), "s-", ms=3, color=BLUE)
    axes[1].set_xlabel("quadrature nodes N (U = 200)")
    for ax in axes:
        ax.set_ylabel("absolute price error")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_5_2_convergence.png")
    plt.close(fig)


def table_5_4():
    lines = ["| K | tau | Original form | Corrected form | Monte Carlo (s.e.) |", "|---|---|---|---|---|"]
    for K, tau in [(100, 0.5), (100, 1.0), (100, 2.0), (100, 3.0), (100, 5.0), (120, 3.0), (80, 5.0)]:
        orig = heston_call_fine(K, tau, **TAB, S0=S0, r=R, corrected=False)
        corr = heston_call_fine(K, tau, **TAB, S0=S0, r=R, corrected=True)
        mc, se = heston_mc_call(S0, K, R, tau, **TAB, steps=int(400 * max(tau, 1)))
        lines.append(f"| {K} | {tau:.1f} | {orig:.4f} | {corr:.4f} | {mc:.4f} ({se:.4f}) |")
    lines += ["", "Monte Carlo: 200,000 paths, full-truncation Euler, seed 11; the step count",
              "scales with maturity (400 per year, at least 400)."]
    write("table_5_4_branches.md", "Table 5.4: original against corrected characteristic function", lines)


def figure_5_3():
    taus = np.linspace(0.1, 6.0, 80)
    orig = [heston_call_fine(100.0, t, **HES, S0=S0, r=R, corrected=False) for t in taus]
    corr = [heston_call([100.0], t, **HES, S0=S0, r=R)[0] for t in taus]
    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    ax.plot(taus, corr, color=BLUE, lw=1.8, label="corrected form")
    ax.plot(taus, orig, color=ORANGE, lw=1.4, label="original form, principal branch")
    ax.set_xlabel("maturity (years)")
    ax.set_ylabel("call price, K = 100")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_5_3_branch_cut.png")
    plt.close(fig)


def table_5_5():
    lines = ["| K | tau | Fourier | Monte Carlo | Standard error | (MC - Fourier) / s.e. |",
             "|---|---|---|---|---|---|"]
    for K, tau in [(90, 1.0), (100, 1.0), (110, 1.0), (100, 2.0), (120, 2.0)]:
        f = heston_call([K], tau, **TAB, S0=S0, r=R)[0]
        mc, se = heston_mc_call(S0, K, R, tau, **TAB, steps=400)
        lines.append(f"| {K} | {tau:.1f} | {f:.4f} | {mc:.4f} | {se:.4f} | {(mc - f) / se:+.2f} |")
    write("table_5_5_fourier_vs_monte_carlo.md", "Table 5.5: Fourier inversion against Monte Carlo", lines)


if __name__ == "__main__":
    start = time.perf_counter()
    quotes = reference_surface()
    normals = hw_normals()
    validation()
    table_5_1(quotes)
    hull_white_fit(quotes, normals)
    h_rmse, w_rmse = table_5_2(quotes, normals)
    figure_5_1(quotes, normals)
    table_5_3(h_rmse, w_rmse)
    figure_5_2()
    table_5_4()
    figure_5_3()
    table_5_5()
    print(f"Finished in {time.perf_counter() - start:.0f} s.")

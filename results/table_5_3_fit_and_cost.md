## Table 5.3: fit quality and cost of one call price

| Method | Fit RMSE (vol. points) | Time per price on this machine |
|---|---|---|
| Heston, Fourier inversion (5.2) | 0.235 | 0.15 ms |
| Heston, Monte Carlo (200,000 paths, 400 steps) | (as above) | 2.9 s |
| Hull-White, Monte Carlo (200,000 paths, 200 steps) | 2.991 | 0.7 s |

Monte Carlo standard error: 0.021. Ratio, Monte Carlo to Fourier: 19,503.
Absolute timings depend on the hardware; the dissertation's run on Google Colab gave
0.37 ms, 6.1 s and 2.8 s.

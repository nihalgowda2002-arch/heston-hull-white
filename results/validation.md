## Validation checks (Sections 5.2 and 5.3)

| Check | Value |
|---|---|
| Black-Scholes at the average variance, K = 100, tau = 1 | 8.481313 |
| Heston Fourier price, eta = 0.01 | 8.481474 |
| Heston Fourier price, eta = 0.001 | 8.481354 |
| Heston Fourier price, eta = 0.0001 | 8.481317 |
| Hull-White Monte Carlo, eta = 0.01 (Black-Scholes at 20%: 8.916037) | 8.915968 (s.e. 5.0e-05) |
| Put-call parity, strikes 70 to 130, tau = 1.5, put from its own inversion | max error 1.8e-14 |
| Variance at T = 1, exact mean and variance (Feller fails: 0.16 < 0.25) | 0.038647, 0.002308 |
| Full-truncation Euler, 400,000 paths, 200 steps, seed 4242 | 0.038685, 0.002328 |
| Andersen QE, 400,000 paths, 200 steps, seed 4242 | 0.038656, 0.002310 |
| Negative variances produced (Euler, QE) | 0, 0 |

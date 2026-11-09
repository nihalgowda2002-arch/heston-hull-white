## Hull-White calibration (Section 5.5)

| | nu0 | mbar | eta | RMSE (vol. points) |
|---|---|---|---|---|
| this run | 0.030589 | 0.748524 | 2.0000 | 2.9905 |
| dissertation, Section A.3 | 0.030544 | 0.764633 | 2.0000 | 2.9914 |

In both, the volatility of variance sits on the upper bound of the search region (2.0).
The two parameter sets differ along a flat direction of the objective, where a higher
initial variance offsets a lower drift, and their fits agree to within 0.001 volatility points.
Calibration time: 62 s on 40,000 paths and 160 steps (common random numbers, seed 4321).

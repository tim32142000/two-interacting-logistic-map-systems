Two interacting Logistic maps: independent exploratory experiment, 2026-10-18

Open report.html first. Readable Chinese report, figures and all data are included.
The original model is unchanged. This experiment is stored in its own directory.

Reproduce with Python 3.11+:
  python -m pip install -r requirements.txt
  python run_experiment.py
  python refine.py

run_experiment.py regenerates scan, long checks, finite perturbations, figures,
Floquet checks and metadata. refine.py regenerates phase-grid checks, numerical
validation and continuation of the period-2 orbit. Approximate runtime: minutes.
report.html is a fixed human-readable report of this run, not auto-generated.

Sources:
https://github.com/tim32142000/two-interacting-logistic-map-systems
Commit: 3366589902cf4a312f73dd8d7a0a1df2643149c0
log2inter.py Git blob SHA: 392986eca08c5b4ee25574ba2b30f65eb82f503c
source/ contains snapshots (line endings normalized), original attribution HTW,
and the upstream Apache 2.0 LICENSE. All experimental code is separate.

CSV conventions:
px, py = smallest scalar recurrence lag <=64 with max residual <1e-8 over
1024 tail points; 0 = undetected, NEVER an assertion of chaos.
original_400_steps.csv instead uses tail steps 300..399 and maximum lag 20.
lambda_max = largest exponent from analytic 2x2 Jacobian and normalized tangent.
block_min/max = extreme 1024-step block estimates; NOT confidence intervals.
mean_abs_diff/max_abs_diff measure simultaneous |x-y| after transient.
Seeds are stored as x0,y0 in each CSV. Floating-point k strings retain binary
representation; e.g. 0.10200000000000001 means the nominal grid value 0.102.
scan_tail.npz: only original-seed last 128 pairs; array axes time,k,coordinate.
long_tail.npz: last 1024 pairs; axes time,k,seed,coordinate.
validation.json: recurrence tolerance checks at 1e-6,1e-8,1e-10; orbit distances
ignore cyclic phase and use the joint least-common-multiple period.
Period-2 branch root residuals and multipliers are in period2_branch.csv.

This study samples initial conditions and finite time; it does not prove global
attractor uniqueness, exhaustive basins, universal behavior or rigorous chaos.

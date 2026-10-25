# Outputs

`python run_experiment.py` writes:

- `full_up.csv` and `full_down.csv`: directional 551-point full-range scans.
- `zoom_up.csv` and `zoom_down.csv`: directional 501-point local scans (k=0.135..0.16).
- `bifurcation.png` and `bifurcation.svg`: four-panel x/y late-state scatter plots.
- `diagnostics.png`: local largest Lyapunov exponent and period-2 indicator.
- `summary.json`: metadata, sample-period ranges, and start/end continuity checks.

Executed in the ChatGPT working environment with Python 3.13.5, NumPy 2.3.5, Matplotlib 3.10.8, Numba 0.65.1.

Verified outcomes: 551 full k values in each direction; 501 zoom k values in each direction; period-2 up to k=0.14530 in both zoom scans, no detected period at k=0.14535. All adjacent start/end state comparisons yielded exact zero difference.

The full CSV files and generated figures were produced in the working environment. See conversation downloads for those specific execution artifacts. To regenerate in your own checkout, run the included script. Do not mistake a detected period of 0 for proof of chaos.

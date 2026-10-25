# State-continuation bifurcation experiment (2026-10-25)

This independent experiment tests whether using the **last state at coupling strength k** as the **initial state at the next k** changes the period-2 right-hand boundary. It does not overwrite the 2026-10-18 experiment.

## Run

Python 3.11+:

```bash
python -m pip install -r requirements.txt
python run_experiment.py
```

The script generates four CSV tables, two PNG figures, a vector SVG version of the bifurcation diagram, and a machine-readable summary.json in `results/`. NumPy and Matplotlib are required; Numba is optional but accelerates this large scan.

## Experiment design

Model: x'=a+k(b-a), y'=b+k(a-b), with a=3.8318741*x*(1-x), b=3.7389149*y*(1-y).

Initial seed (x,y)=(0.12924,0.2304). Before the very first k only, apply 249 uncoupled steps to reproduce the original setup. For each k, run 12,000 coupled transient steps plus 8,192 measurement steps, then *pass the final (x,y) to the next k*. The state is never reset between k values in one scan. Each scan computes its largest finite-time Lyapunov exponent and checks x/y recurrence over 1024 late samples for p<=64 with tolerance 1e-8. Each bifurcation panel draws the last 64 states per k.

There are four scans: full increasing k (551 values, 0..0.5); full decreasing k (same values reversed); local increasing k (501 values, 0.135..0.16 by 0.00005); and local decreasing k (reversed). Full scans independently start from the same post-prehistory seed. Local scans start from the completed full scan's end state at k=0.135 or 0.16, respectively. Thus local scans inherit a settled directional state but are separate refinement passes.

CSV columns include start_x,start_y,end_x,end_y for direct independent verification. The script asserts that each k's start state equals the previous k's end state.

## Results

See [REPORT.md](REPORT.md) for conclusions and caveats. Representative output and actual generated figures are available as downloadable local attachments in the ChatGPT conversation. GitHub contains the reproducible generator and report.

No GitHub Actions or existing model files are changed.

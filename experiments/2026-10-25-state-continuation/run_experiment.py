"""State-continuation experiment for two output-coupled logistic maps.

Run: python run_experiment.py
Dependencies: numpy, matplotlib; numba is optional (faster).
Each successive k receives the *last state* from the preceding k.
No initial-condition reset or uncoupled prehistory between k values.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from numba import njit
except ImportError:
    def njit(func):
        return func

RX, RY = 3.8318741, 3.7389149
INITIAL = (0.12924, 0.2304)
BURN, MEASURE, TAIL = 12000, 8192, 1024
DRAW = 64
TOL, MAX_PERIOD = 1e-8, 64
OUT = Path(__file__).resolve().parent / 'results'


@njit
def advance(x, y, k):
    a = RX * x * (1.0 - x)
    b = RY * y * (1.0 - y)
    return a + k * (b - a), b + k * (a - b)


@njit
def prehistory():
    x, y = INITIAL
    for _ in range(249):
        x, y = advance(x, y, 0.0)
    return x, y


@njit
def scan(ks, x0, y0, burn=BURN, measure=MEASURE):
    """Per-k state-continuation scan.

    Columns: k,start_x,start_y,end_x,end_y,lambda_max,period_x,period_y,
    mean_abs_xy,x_min,x_max,y_min,y_max. Period 0 means not detected.
    """
    nk = len(ks)
    summary = np.empty((nk, 13))
    points = np.empty((nk, DRAW, 2))
    tail = np.empty((TAIL, 2))
    x, y = x0, y0
    for i in range(nk):
        k = ks[i]
        startx, starty = x, y
        vx, vy = 1.0, 0.37
        length = np.sqrt(vx * vx + vy * vy)
        vx, vy = vx / length, vy / length
        lsum = 0.0
        dsum = 0.0
        xmin, xmax, ymin, ymax = 2.0, -1.0, 2.0, -1.0
        for it in range(burn + measure):
            ax = RX * (1.0 - 2.0 * x)
            by = RY * (1.0 - 2.0 * y)
            wx = (1.0 - k) * ax * vx + k * by * vy
            wy = k * ax * vx + (1.0 - k) * by * vy
            norm = np.sqrt(wx * wx + wy * wy)
            if norm < 1e-300:
                norm = 1e-300
            vx, vy = wx / norm, wy / norm
            x, y = advance(x, y, k)
            if it >= burn:
                lsum += np.log(norm)
                dsum += np.abs(x - y)
                xmin, xmax = min(xmin, x), max(xmax, x)
                ymin, ymax = min(ymin, y), max(ymax, y)
                if it >= burn + measure - TAIL:
                    j = it - (burn + measure - TAIL)
                    tail[j, 0], tail[j, 1] = x, y
                if it >= burn + measure - DRAW:
                    j = it - (burn + measure - DRAW)
                    points[i, j, 0], points[i, j, 1] = x, y
        px, py = 0, 0
        for p in range(1, MAX_PERIOD + 1):
            goodx, goody = True, True
            for t in range(TAIL - p):
                if np.abs(tail[t + p, 0] - tail[t, 0]) >= TOL:
                    goodx = False
                if np.abs(tail[t + p, 1] - tail[t, 1]) >= TOL:
                    goody = False
                if not goodx and not goody:
                    break
            if px == 0 and goodx:
                px = p
            if py == 0 and goody:
                py = p
            if px != 0 and py != 0:
                break
        summary[i, 0] = k
        summary[i, 1] = startx
        summary[i, 2] = starty
        summary[i, 3] = x
        summary[i, 4] = y
        summary[i, 5] = lsum / measure
        summary[i, 6] = px
        summary[i, 7] = py
        summary[i, 8] = dsum / measure
        summary[i, 9] = xmin
        summary[i, 10] = xmax
        summary[i, 11] = ymin
        summary[i, 12] = ymax
    return summary, points


FIELDS = ['k', 'start_x', 'start_y', 'end_x', 'end_y', 'lambda_max',
          'period_x', 'period_y', 'mean_abs_xy', 'x_min', 'x_max', 'y_min', 'y_max']


def write_csv(path, a):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        for row in a:
            w.writerow([f'{v:.17g}' if j not in (6, 7) else int(v)
                        for j, v in enumerate(row)])


def figure(results):
    fig, axs = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for ax, key, label in [
        (axs[0, 0], 'full_up', 'Full range / increasing k'),
        (axs[0, 1], 'full_down', 'Full range / decreasing k'),
        (axs[1, 0], 'zoom_up', 'Near period-2 boundary / increasing k'),
        (axs[1, 1], 'zoom_down', 'Near period-2 boundary / decreasing k'),
    ]:
        d, pts = results[key]
        for col, color, name in [(0, 'tab:blue', 'x'), (1, 'tab:orange', 'y')]:
            ax.plot(np.repeat(d[:, 0], DRAW), pts[:, :, col].ravel(), '.',
                    markersize=0.55, alpha=0.5, color=color, label=name)
        ax.set(title=label, xlabel='Symmetric coupling k', ylabel='Late state')
        ax.legend(loc='lower right', markerscale=6)
        if key.startswith('zoom'):
            ax.set_xlim(0.135, 0.16)
    fig.suptitle('Output-coupled logistic maps | state continuation in k')
    fig.savefig(OUT / 'bifurcation.png', dpi=160)
    fig.savefig(OUT / 'bifurcation.svg')
    plt.close(fig)
    fig, axs = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    for key, name, color in [('zoom_up', 'Increasing k', 'tab:blue'),
                             ('zoom_down', 'Decreasing k', 'tab:orange')]:
        d = results[key][0]
        axs[0].plot(d[:, 0], d[:, 5], lw=0.8, label=name, color=color)
        valid = ((d[:, 6] == 2) & (d[:, 7] == 2)).astype(float)
        axs[1].plot(d[:, 0], valid, lw=0.85, label=name, color=color)
    axs[0].axhline(0, color='black', lw=0.6)
    axs[0].set(ylabel='Largest Lyapunov exponent', title='Local sweep diagnostics')
    axs[1].set(xlabel='Symmetric coupling k', ylabel='Both detected period-2')
    for ax in axs:
        ax.legend()
        ax.set_xlim(0.135, 0.16)
    fig.savefig(OUT / 'diagnostics.png', dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    full_up_k = np.r_[np.arange(401) * 0.0005,
                      np.arange(0.202, 0.50001, 0.002)]
    full_down_k = full_up_k[::-1].copy()
    z0 = prehistory()
    results = {}
    results['full_up'] = scan(full_up_k, *z0)
    results['full_down'] = scan(full_down_k, *z0)
    zoom_up_k = np.linspace(0.135, 0.16, 501)
    zoom_down_k = zoom_up_k[::-1].copy()
    up = results['full_up'][0]
    down = results['full_down'][0]
    up_idx = np.argmin(np.abs(up[:, 0] - zoom_up_k[0]))
    down_idx = np.argmin(np.abs(down[:, 0] - zoom_down_k[0]))
    assert abs(up[up_idx, 0] - zoom_up_k[0]) < 1e-12
    assert abs(down[down_idx, 0] - zoom_down_k[0]) < 1e-12
    results['zoom_up'] = scan(zoom_up_k, up[up_idx, 3], up[up_idx, 4])
    results['zoom_down'] = scan(zoom_down_k, down[down_idx, 3], down[down_idx, 4])
    for name, (rows, _) in results.items():
        write_csv(OUT / f'{name}.csv', rows)
    figure(results)
    stats = {'model': {'rx': RX, 'ry': RY, 'initial': INITIAL},
             'steps': {'uncoupled_prehistory_first_k_only': 249, 'burn': BURN,
                       'measure': MEASURE, 'tail': TAIL, 'draw': DRAW},
             'period': {'tolerance': TOL, 'max_period': MAX_PERIOD},
             'scan': {}, 'end_state_consistency': {}, 'zoom_seed_source': {
                 'up': {'k': float(up[up_idx, 0]), 'end_state': up[up_idx, 3:5].tolist()},
                 'down': {'k': float(down[down_idx, 0]), 'end_state': down[down_idx, 3:5].tolist()}
             }}
    for name, (d, pts) in results.items():
        period2 = (d[:, 6] == 2) & (d[:, 7] == 2)
        good = d[period2]
        stats['scan'][name] = {
            'n_k': len(d), 'period2_count': int(np.count_nonzero(period2)),
            'period2_min_k': float(np.min(good[:, 0])) if len(good) else None,
            'period2_max_k': float(np.max(good[:, 0])) if len(good) else None,
            'positive_lyapunov_count': int(np.count_nonzero(d[:, 5] > 0)),
            'sample_near_boundary': [
                {'k': float(row[0]), 'period_x': int(row[6]),
                 'period_y': int(row[7]), 'lambda_max': float(row[5])}
                for row in d if 0.1449 <= row[0] <= 0.1461
            ],
        }
        stats['end_state_consistency'][name] = float(np.max(
            np.abs(d[1:, 1:3] - d[:-1, 3:5])))
        assert np.isfinite(d).all() and np.isfinite(pts).all()
        assert stats['end_state_consistency'][name] == 0.0
    assert results['full_up'][0][0, 1] == z0[0]
    assert results['full_down'][0][0, 2] == z0[1]
    (OUT / 'summary.json').write_text(json.dumps(stats, ensure_ascii=False,
                                   indent=2), encoding='utf-8')
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

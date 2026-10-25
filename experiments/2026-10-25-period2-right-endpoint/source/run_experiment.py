"""Independent experiment based on HTW's log2inter.py. Original repo is untouched.
Run: python run_experiment.py (NumPy, SciPy, Matplotlib).
Lyapunov estimates use natural logarithms and one map iteration as time unit.
"""

from pathlib import Path
import csv, json, platform
import numpy as np
import scipy
from scipy.optimize import root
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
RX, RY = 3.8318741, 3.7389149
DEFAULT = np.array([0.12924, 0.2304])


def advance(z, k):
    """將輸出耦合 logistic map 推進一步，回傳新的狀態陣列。

    z 的形狀為 (..., 2)，最後一軸依序是 x、y；k 可為純量或可與
    z[..., 0] 廣播的陣列。回傳形狀為廣播後的 (..., 2)。RX、RY 固定，
    兩個 logistic 輸出由同一個舊狀態計算，再做對稱耦合。
    實驗使用雙精度、初值位於 (0, 1)、0 <= k <= 0.5；此函式不裁切狀態，
    也不檢查範圍，以保留原模型的浮點運算行為。
    """
    a = RX * z[..., 0] * (1 - z[..., 0])
    b = RY * z[..., 1] * (1 - z[..., 1])
    # Preserve the arithmetic ordering in the original program.
    return np.stack((a + k * (b - a), b + k * (a - b)), axis=-1)


def jac(z, k):
    """回傳 advance 對舊狀態 (x, y) 的解析 Jacobian。

    z、k 的形狀與廣播規則同 advance；輸出形狀為 (..., 2, 2)，
    倒數第二軸是輸出座標，最後一軸是微分的輸入座標。
    k 被視為固定參數；這不是對耦合強度的導數。
    """
    # 鏈式法則：先對各自的 logistic map 微分，再乘上耦合矩陣。
    a = RX * (1 - 2 * z[..., 0])
    b = RY * (1 - 2 * z[..., 1])
    return np.stack(
        (np.stack(((1 - k) * a, k * b), -1), np.stack((k * a, (1 - k) * b), -1)), -2
    )


def period(tail, tol=1e-8, maxp=64):
    """以絕對誤差門檻搜尋每個純量座標最小的重現延遲。

    tail 的形狀為 (時間, ..., 座標)，回傳整數陣列 tail.shape[1:]。
    對每個 lag p，比較所有可配對時間點，最大差值嚴格小於 tol 才接受。
    maxp 必須小於時間長度；tol 應為正值，資料應為有限數值。
    回傳 0 表示在 1..maxp 之間未檢出，不能據此斷言混沌。
    x、y 分別判定；聯合軌道週期需另取兩者最小公倍數。
    """
    # Every retained pair must recur; use smallest accepted lag.
    out = np.zeros(tail.shape[1:], dtype=int)
    for p in range(1, maxp + 1):
        good = np.max(np.abs(tail[p:] - tail[:-p]), axis=0) < tol
        out[(out == 0) & good] = p
    return out


def batch(ks, seeds, burn=12000, measure=8192, tailn=1024):
    """批次估計不同耦合強度與初始條件的長時間行為。

    ks 為一維 K 個耦合強度，seeds 為 (S, 2) 的雙精度初值。
    先跑 249 步無耦合前史，再跑 burn 步耦合暫態，最後量測 measure 步。
    burn 應為非負整數；measure >= 1024；64 < tailn <= measure。
    呼叫端負責這些條件與 advance 的數值範圍，本函式不做驗證。

    回傳 (rows, tail)：rows 依 k 優先、seed 次之排列，包含逐座標週期、
    每步最大 Lyapunov 指數估計及同時刻 |x-y| 的平均／最大值；
    tail 形狀為 (tailn, K, S, 2)，保存每次推進後的最後狀態。
    block_min/max 是完整 1024 步區塊的指數極值，並非信賴區間；不足
    一個區塊的餘數仍計入總平均，但不納入區塊極值。
    指數使用自然對數，單位是一個 map iteration。
    """
    ks = np.asarray(ks)
    z = np.broadcast_to(seeds, (len(ks), len(seeds), 2)).copy()
    k = ks[:, None]
    # 與原程式 m=1..249 的無耦合前史對齊；seeds 並非耦合啟動時的狀態。
    for _ in range(1, 250):
        z = advance(z, 0.0)
    # 單一切向量反覆正規化，通常會對齊最大成長方向；不是完整 Lyapunov 譜。
    v = np.broadcast_to([1.0, 0.37], z.shape).copy()
    v /= np.linalg.norm(v, axis=-1)[..., None]
    sums = np.zeros(z.shape[:-1])
    blocks = []
    buf = []
    tail = []
    dsum = np.zeros_like(sums)
    dmax = np.zeros_like(sums)
    # 暫態期間也演化切向量，使方向先收斂；量測時才累加對數伸長率。
    for i in range(burn + measure):
        # Jacobian 必須在推進前的 z 評估，對應 v_{t+1} = DF(z_t) v_t。
        w = np.einsum("...ij,...j->...i", jac(z, k), v)
        norm = np.linalg.norm(w, axis=-1)
        # 下限避免除零與 log(0)；若切向量完全退化，此有限值不是嚴格的指數。
        v = w / np.maximum(norm, 1e-300)[..., None]
        z = advance(z, k)
        if i >= burn:
            ln = np.log(np.maximum(norm, 1e-300))
            sums += ln
            buf.append(ln)
            d = np.abs(z[..., 0] - z[..., 1])
            dsum += d
            dmax = np.maximum(dmax, d)
            if len(buf) == 1024:
                blocks.append(np.mean(buf, axis=0))
                buf = []
        if i >= burn + measure - tailn:
            tail.append(z.copy())
    tail = np.array(tail)
    ps = period(tail)
    rows = []
    for a, kv in enumerate(ks):
        for b, seed in enumerate(seeds):
            rows.append(
                dict(
                    k=float(kv),
                    seed=b,
                    x0=float(seed[0]),
                    y0=float(seed[1]),
                    px=int(ps[a, b, 0]),
                    py=int(ps[a, b, 1]),
                    lambda_max=float(sums[a, b] / measure),
                    block_min=float(np.min(blocks, axis=0)[a, b]),
                    block_max=float(np.max(blocks, axis=0)[a, b]),
                    mean_abs_diff=float(dsum[a, b] / measure),
                    max_abs_diff=float(dmax[a, b]),
                )
            )
    return rows, tail


def writecsv(name, rows):
    """將非空、欄位一致的字典列寫入 OUT/name，覆寫同名結果。

    name 為相對結果檔名；rows 為非空字典序列，首列的鍵順序決定 CSV 欄序。
    此函式沒有回傳值。
    """
    with (OUT / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def trajectory(k, n=400, seed=DEFAULT):
    """重現原模型的單條軌跡，回傳形狀 (n, 2) 的雙精度陣列。

    k 為純量耦合強度，seed 為 (2,) 初值，n 為至少 1 的整數。
    第 0 列是初值；m=1..249 無耦合，m=250 起才使用 k。
    因此 n<=250 時不會執行耦合更新；不丟棄暫態。
    """
    z = np.empty((n, 2))
    z[0] = seed
    for m in range(1, n):
        z[m] = advance(z[m - 1], 0 if m < 250 else k)
    return z


def main():
    """執行固定亂數種子的完整掃描與複查，覆寫 OUT 下的數據及圖表。

    包含多初值掃描、長時間檢查、週期軌道局部穩定性與有限微擾比較；
    不回傳值，也不自動更新固定的實驗報告。
    """
    rng = np.random.default_rng(20261018)
    seeds = np.vstack((DEFAULT, rng.uniform(0.001, 0.999, (15, 2))))
    ks = np.r_[np.arange(401) * 0.0005, np.arange(0.202, 0.50001, 0.002)]
    rows, tail = batch(ks, seeds)
    writecsv("scan.csv", rows)
    np.savez_compressed(
        OUT / "scan_tail.npz", ks=ks, seeds=seeds, tail=tail[-128:, :, 0, :]
    )
    selected = [0, 0.05, 0.08, 0.1, 0.12, 0.15, 0.3, 0.5]
    long, lt = batch(selected, seeds, burn=50000, measure=65536)
    writecsv("long_checks.csv", long)
    np.savez_compressed(OUT / "long_tail.npz", ks=selected, seeds=seeds, tail=lt)
    short = []
    for k in selected:
        z = trajectory(k)
        p = period(z[300:, None, :], maxp=20)[0]
        short.append(
            dict(
                k=k,
                px=int(p[0]),
                py=int(p[1]),
                lag4_residual=float(np.max(np.abs(z[378:] - z[374:396]))),
            )
        )
    writecsv("original_400_steps.csv", short)
    # Local stability of the orbit: eigenvalues of D F^p, independently of tangent estimate.
    floquet = []
    for a, k in enumerate(selected):
        r = long[a * len(seeds)]
        p = int(np.lcm(r["px"], r["py"])) if r["px"] * r["py"] else 0
        if not p:
            continue

        def fp(z):
            """固定目前 k、p，將形狀 (2,) 的狀態推進 p 步。"""
            for _ in range(p):
                z = advance(z, k)
            return z

        # 尾端狀態提供 root 的局部初猜；殘差會寫入 CSV，供檢查根的精度。
        # F^p(z)=z 也包含較短週期，這裡的 p 由尾端重現檢查提供。
        sol = root(lambda z: fp(z) - z, lt[-1, a, 0])
        z = sol.x.copy()
        M = np.eye(2)
        # 沿軌道依時間順序左乘：DF(z_{p-1}) ... DF(z_0)。其特徵值為
        # Floquet 乘子；rho<1 表示局部吸引，與每步 Lyapunov 指數量綱不同。
        for _ in range(p):
            M = jac(z, k) @ M
            z = advance(z, k)
        eig = np.linalg.eigvals(M)
        floquet.append(
            dict(
                k=k,
                p=p,
                residual=float(np.max(np.abs(fp(sol.x) - sol.x))),
                rho=float(max(abs(eig))),
                eig1=str(eig[0]),
                eig2=str(eig[1]),
            )
        )
    writecsv("floquet.csv", floquet)
    # Sensitive dependence: perturb an already settled state; coupling-onset states happen to be near a critical point.
    fig, axs = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)
    for col, label in [(0, "x"), (1, "y")]:
        axs[0].plot(
            np.repeat(ks, 128),
            tail[-128:, :, 0, col].T.ravel(),
            ".",
            ms=0.45,
            alpha=0.6,
            label=label,
        )
    axs[0].set(
        ylabel="Late state",
        title="Output-coupled logistic maps | original initial condition",
    )
    axs[0].legend()
    rr = np.array([r["lambda_max"] for r in rows]).reshape(len(ks), len(seeds))
    axs[1].plot(ks, rr[:, 0], lw=0.9, label="original seed")
    axs[1].fill_between(
        ks, rr.min(1), rr.max(1), alpha=0.25, label="range over 16 initial conditions"
    )
    axs[1].axhline(0, c="k", lw=0.7)
    axs[1].set(ylabel="Largest Lyapunov exponent")
    axs[1].legend()
    for target, marker in [(4, "s"), (2, "o")]:
        pp = (
            np.array([r["px"] == target and r["py"] == target for r in rows])
            .reshape(len(ks), len(seeds))
            .mean(1)
        )
        axs[2].plot(ks, pp, marker, ms=2, label=f"both period {target}")
    axs[2].set(
        xlabel="Symmetric coupling k", ylabel="Fraction of 16 seeds", ylim=(-0.05, 1.05)
    )
    axs[2].legend()
    fig.savefig(OUT / "overview.png", dpi=160)
    plt.close(fig)
    fig, axs = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    sensitivity = []
    for k in [0.05, 0.08, 0.1, 0.12, 0.15, 0.5]:
        z = trajectory(k, 20251)[-1]
        zp = z + np.array([1e-10, 0.0])
        ds = []
        for t in range(400):
            ds.append(float(np.linalg.norm(z - zp)))
            z = advance(z, k)
            zp = advance(zp, k)
        axs[1, 1].semilogy(np.maximum(ds, 1e-17), label=str(k))
        sensitivity.extend(dict(k=k, t=t, distance=d) for t, d in enumerate(ds))
    writecsv("sensitivity.csv", sensitivity)
    for a, k in enumerate([0.08, 0.1, 0.12]):
        z = trajectory(k, 10000)[-24:]
        axs[0, 0].plot(z[:, 0], ".-", label=f"x k={k}")
        axs[0, 1].plot(z[:, 1], ".-", label=f"y k={k}")
        axs[1, 0].plot(z[:, 0], z[:, 1], "o", label=f"k={k}")
    axs[0, 0].set(title="Late x: period 4 / 4 / 2", xlabel="Late iteration")
    axs[0, 1].set(title="Late y", xlabel="Late iteration")
    axs[1, 0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axs[1, 0].set(xlabel="x", ylabel="y", title="Equal period does not imply x = y")
    axs[1, 1].set(
        xlabel="Steps after perturbation at m=20250",
        ylabel="Euclidean separation",
        title="Finite perturbation (roundoff floor shown)",
    )
    for ax in axs.flat:
        ax.legend(fontsize=7)
    fig.savefig(OUT / "examples.png", dpi=160)
    plt.close(fig)
    metadata = dict(
        source_commit="3366589902cf4a312f73dd8d7a0a1df2643149c0",
        source_blob="392986eca08c5b4ee25574ba2b30f65eb82f503c",
        rx=RX,
        ry=RY,
        seed=20261018,
        burn=12000,
        measure=8192,
        tail=1024,
        period_max=64,
        period_tol=1e-8,
        long_burn=50000,
        long_measure=65536,
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        matplotlib=matplotlib.__version__,
    )
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print("DEFAULT LONG CHECKS")
    print(json.dumps(long[:: len(seeds)], indent=2))
    print("FLOQUET", floquet)
    print("SCAN DEFAULT PERIOD RUNS")
    p = np.array([(r["px"], r["py"]) for r in rows[:: len(seeds)]])
    start = 0
    for i in range(1, len(ks) + 1):
        if i == len(ks) or np.any(p[i] != p[start]):
            print(ks[start], ks[i - 1], p[start].tolist())
            start = i


if __name__ == "__main__":
    main()


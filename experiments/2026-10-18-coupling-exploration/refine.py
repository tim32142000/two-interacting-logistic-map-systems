"""Targeted follow-up and independent checks; run after run_experiment.py."""

from run_experiment import *
from scipy.optimize import root


def main():
    """讀取 long_tail.npz，驗證模型並覆寫精查結果，不回傳值。

    輸入由 run_experiment.py 產生：tail 軸序為 (時間, k, seed, 座標)，
    seeds 為 (S, 2)，ks 為 (K,)。延續初猜假定第 5 個 k 是 0.12，
    第 1 個 seed 是原始初值；更改上游取樣時必須一併檢查這些位置。
    週期容差檢查、軌道距離與 Floquet 乘子都是有限精度的數值證據。
    """
    # Numerical equivalence with original code, without invoking its GUI.
    # 只執行來源快照的模型區段，避免開啟 GUI；保持快照本身原樣以供比對。
    ns = {"__name__": "__check__"}
    src = (
        (Path(__file__).parent / "source/log2inter.py")
        .read_text()
        .split("plt.figure()")[0]
    )
    exec(compile(src, "source/log2inter.py", "exec"), ns)
    original = np.column_stack((ns["x"], ns["y"]))
    assert np.array_equal(original, trajectory(0.1)), "Original trace mismatch"
    # Analytic Jacobian versus central finite difference.
    z = np.array([0.321, 0.745])
    h = 1e-6
    # 每欄擾動一個輸入座標；中心差分只用於獨立驗證，不取代解析導數。
    fd = np.column_stack(
        [
            (advance(z + np.eye(2)[i] * h, 0.1) - advance(z - np.eye(2)[i] * h, 0.1))
            / (2 * h)
            for i in range(2)
        ]
    )
    assert np.max(abs(fd - jac(z, 0.1))) < 1e-8
    a = np.load(OUT / "long_tail.npz")
    t = a["tail"]
    seeds = a["seeds"]
    check = []
    for i, k in enumerate(a["ks"]):
        ps = {
            str(tol): period(t[:, i], tol=tol).tolist() for tol in [1e-6, 1e-8, 1e-10]
        }
        # 先求聯合週期；任一座標未檢出時，不做週期軌道的距離比較。
        p = int(np.lcm(*ps["1e-08"][0])) if all(ps["1e-08"][0]) else 0
        dist = None
        if p:
            # 最近點配對忽略循環相位；取原始種子軌道到各種子軌道的
            # 最大單向距離，不是逐時刻同步誤差，也不是吸引子唯一性的證明。
            base = t[-p:, i, 0]
            dist = max(
                float(
                    np.max(
                        np.min(
                            np.linalg.norm(
                                base[:, None, :] - t[-p:, i, s][None, :, :], axis=-1
                            ),
                            axis=1,
                        )
                    )
                )
                for s in range(len(seeds))
            )
        check.append(
            dict(k=float(k), periods_by_tolerance=ps, orbit_distance_across_seeds=dist)
        )
    (OUT / "validation.json").write_text(
        json.dumps(
            dict(
                original_trace_bitwise_equal=True,
                jacobian_max_error=float(np.max(abs(fd - jac(z, 0.1)))),
                checks=check,
            ),
            indent=2,
        )
    )
    # Continue the period-2 branch through its loss of stability as k decreases.
    # 自然參數延續：從已穩定的 k=0.12 軌道出發，以上一步的根作下一步初猜。
    # 直接求 F²(z)-z=0 可追蹤失穩後的分支，單純向前迭代則會離開不穩定軌道。
    z = t[-1, 4, 0]
    branch = []
    for k in np.arange(0.12, 0.09999, -0.00005):
        sol = root(lambda u: advance(advance(u, k), k) - u, z, tol=1e-11)
        z = sol.x
        # 二步鏈式導數的乘法次序不可交換；rho 是兩步映射的譜半徑。
        # 方程也容許固定點，殘差與乘子需合看；此法未做弧長延續或分支切換。
        M = jac(advance(z, k), k) @ jac(z, k)
        eig = np.linalg.eigvals(M)
        branch.append(
            dict(
                k=float(k),
                residual=float(np.max(abs(advance(advance(z, k), k) - z))),
                eig_min_real=float(np.min(eig.real)),
                eig_max_real=float(np.max(eig.real)),
                rho=float(max(abs(eig))),
            )
        )
    writecsv("period2_branch.csv", branch)
    # 乘子越過 -1 是倍週期失穩的局部指標；輸出網格夾住的區間而非精確分岔點。
    print(
        "period2 -1 crossings",
        [
            (branch[i - 1]["k"], r["k"])
            for i, r in enumerate(branch)
            if i and (r["eig_min_real"] + 1) * (branch[i - 1]["eig_min_real"] + 1) < 0
        ],
    )
    print("orbit distance", [(x["k"], x["orbit_distance_across_seeds"]) for x in check])
    # Near-zero coupling is unsampled by the coarse grid. Include an explicit 5x3 phase grid.
    tr = trajectory(0, 300)
    phase = np.array(
        [[tr[249 + i, 0], tr[249 + j, 1]] for i in range(3) for j in range(5)]
    )
    # These are treated as prehistory initial conditions; the same 249 uncoupled steps follow.
    ks = [
        0,
        1e-7,
        1e-6,
        1e-5,
        1e-4,
        0.0002,
        0.0003,
        0.0004,
        0.0735,
        0.074,
        0.0745,
        0.075,
        0.0795,
        0.08,
        0.102,
        0.1025,
        0.145,
        0.1455,
    ]
    rr, _ = batch(ks, np.vstack((DEFAULT, phase)), burn=20000, measure=16384)
    writecsv("phase_and_boundary_checks.csv", rr)
    for k in ks:
        b = [r for r in rr if r["k"] == k]
        print(
            k,
            sorted(set((r["px"], r["py"]) for r in b)),
            min(r["lambda_max"] for r in b),
            max(r["lambda_max"] for r in b),
        )


if __name__ == "__main__":
    main()

# 耦合 Logistic maps 探索實驗（2026-10-18）

以根目錄 `log2inter.py` 的既有模型，重現共同週期 4，掃描耦合強度、比較初始條件，並區分週期鎖定、完全狀態同步與混沌。

**先閱讀 [中文實驗報告](REPORT.md)**；[HTML 版本](report.html) 可下載後在瀏覽器開啟。

## 主要發現

- 重現 `k=0.08、0.10` 的共同週期 4，及 `k=0.12` 的共同週期 2；原程式 400 步尚不足以確認漸近週期。
- 掃描找到共同週期 `16 → 8 → 4 → 2` 的區域，並以週期 2 軌道乘子支持局部反向倍週期分岔。
- `k=0.05、0.15` 有長時間正最大 Lyapunov 指數及微擾成長的混沌數值證據。
- `k=0.5` 由模型代數保證完全同步，但共同軌道仍有混沌數值特徵。
- 極弱耦合下可見共同週期 15，提醒我們相同週期本身並不足以推論同步機制。

所有數值結果限於本次參數、取樣、時間與雙精度運算，未證明普遍規律或吸引子唯一性。

## 重現

需要 Python 3.11+，在此資料夾執行：

```bash
python -m pip install -r requirements.txt
python run_experiment.py
python refine.py
```

運算會重新產生 `results/` 中的數據與圖表，可能需要數分鐘。`REPORT.md` 與 `report.html` 為本次結果的固定報告，不會自動重寫。

## 檔案

| 路徑 | 內容 |
| --- | --- |
| [REPORT.md](REPORT.md) | GitHub 可直接閱讀的完整中文報告 |
| [run_experiment.py](run_experiment.py) | 主掃描、長時間複查、Lyapunov 指數、Floquet 乘子與圖表 |
| [refine.py](refine.py) | 相位取樣、容差複查與週期 2 軌道延續 |
| [results/](results/) | CSV、圖表、尾端軌道與重現環境 |
| [source/](source/) | 原始模型、README 與授權的來源快照 |
| [README.txt](README.txt) | 英文資料欄位與執行細節 |

原始模型作者：HTW。來源 commit：`3366589902cf4a312f73dd8d7a0a1df2643149c0`；來源快照保留原署名與 Apache 2.0 授權。所有新增實驗程式獨立於根目錄的原始模型。

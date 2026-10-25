兩個耦合 Logistic maps：二週期右端研究
日期：2026-10-25

先開啟 right_endpoint_report.html。圖片已內嵌，可以離線閱讀。
Markdown 原始報告：REPORT.md。高解析度圖：results/*.png。

原始 scan.npz 以 scan.npz.parts 分片保存。執行 python assemble_data.py
可還原原檔並核對 SHA-256；完整重跑 run_all.py 亦會產生 scan.npz。

重現：
python -m pip install -r requirements.txt
python run_all.py

這些程式只在本資料夾內產生結果，不修改 GitHub、不需要 GitHub 權杖。
完整運算包含密集初值掃描，需要等待。可分別執行單一腳本。
浮點平台差異可能改變混沌軌跡或細碎吸引域分類；分岔根與穩定週期應可比較。
正式結果和限制請閱讀 HTML 報告。

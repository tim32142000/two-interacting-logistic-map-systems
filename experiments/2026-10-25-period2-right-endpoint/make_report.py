"""Build a self-contained offline HTML from REPORT.md and measured PNGs."""
from pathlib import Path
import base64,re
import markdown

ROOT=Path(__file__).parent
def main():
    raw=(ROOT/'REPORT.md').read_text(encoding='utf-8')
    html=markdown.markdown(raw,extensions=['tables','fenced_code','toc'])
    def embed(m):
        path=ROOT/m.group(1)
        return 'src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'"'
    html=re.sub(r'src="(results/[^"]+\.png)"',embed,html)
    css='''body{margin:0;background:#f4f6f8;color:#182a36;font:17px/1.85 system-ui,-apple-system,"Noto Sans TC","Microsoft JhengHei",sans-serif}main{max-width:1080px;margin:36px auto;padding:42px 5%;background:white;box-shadow:0 4px 35px #182a3610}h1{font-size:2.1rem;line-height:1.4;color:#123e59}h2{font-size:1.4rem;margin-top:2.8rem;padding-top:1rem;border-top:2px solid #dce7ed}img{display:block;max-width:100%;height:auto;margin:24px auto;border:1px solid #dce5eb}table{border-collapse:collapse;width:100%;font-size:.9rem;line-height:1.65;margin:20px 0}td,th{padding:10px 12px;border:1px solid #dce5eb;text-align:left}th{background:#edf4f8}tr:nth-child(even){background:#fafcfd}pre{overflow:auto;white-space:pre;background:#f1f5f8;border-left:4px solid #146a9a;padding:16px;font:14px/1.8 ui-monospace,Consolas,monospace}code{font-family:ui-monospace,Consolas,monospace}a{color:#146a9a}strong{color:#12415d}p{margin:16px 0}@media(max-width:700px){main{margin:0;padding:24px 5%}body{font-size:16px}h1{font-size:1.6rem}table{display:block;overflow-x:auto}}@media print{body{background:white}main{box-shadow:none;margin:0;padding:0}h2{break-after:avoid}img,table{break-inside:avoid}}'''
    full='<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>二週期右端分岔研究</title><style>'+css+'</style></head><body><main>'+html+'</main></body></html>'
    out=ROOT/'right_endpoint_report.html';out.write_text(full,encoding='utf-8')
    assert full.count('data:image/png;base64,')==6
    assert 'src="results/' not in full
    print(out, out.stat().st_size)

if __name__=='__main__':main()

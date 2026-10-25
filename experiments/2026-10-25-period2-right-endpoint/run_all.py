"""Run the independent experiment and regenerate all deliverables."""
import subprocess,sys
from pathlib import Path
root=Path(__file__).parent
for script in ['research.py','precision.py','analyze.py','simulate.py','coexistence.py','plots.py','make_report.py']:
    print('\nRunning '+script,flush=True)
    subprocess.run([sys.executable,str(root/script)],check=True,cwd=root)

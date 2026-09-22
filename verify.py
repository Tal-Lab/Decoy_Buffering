"""Compare regenerated data/*.csv against the published reference/*.csv.

The deterministic solvers (adiabatic master equation) should agree to machine
precision. The stochastic components are seeded, but NumPy/numba version
differences can perturb the random stream, so a tolerance is applied and the
worst deviation per file is reported.

Run from the repository root:  python verify.py
Exit status is non-zero if any file is missing or exceeds tolerance.
"""
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
RTOL = 0.02          # 2 % on stochastic quantities
ok = True
print(f"{'file':28s} {'rows':>6s} {'max rel dev':>12s}  status")
for fn in sorted(os.listdir("reference")):
    ref = pd.read_csv(f"reference/{fn}")
    new_path = f"data/{fn}"
    if not os.path.exists(new_path):
        print(f"{fn:28s} {'-':>6s} {'-':>12s}  MISSING (run make_all.py)")
        ok = False
        continue
    new = pd.read_csv(new_path)
    if list(new.columns) != list(ref.columns) or len(new) != len(ref):
        print(f"{fn:28s} {len(new):6d} {'-':>12s}  SHAPE/COLUMN MISMATCH")
        ok = False
        continue
    num = ref.select_dtypes(include=[np.number]).columns
    a, b = ref[num].to_numpy(float), new[num].to_numpy(float)
    denom = np.maximum(np.abs(a), 1e-12)
    dev = np.nanmax(np.abs(b - a) / denom) if a.size else 0.0
    good = dev <= RTOL
    ok &= good
    print(f"{fn:28s} {len(new):6d} {dev:12.3e}  {'OK' if good else 'EXCEEDS TOLERANCE'}")
print("\nall files reproduce within tolerance" if ok else "\nDISCREPANCIES FOUND")
sys.exit(0 if ok else 1)

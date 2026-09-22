"""13_export_age_profiles: age-resolved mean free protein over the cell cycle.

These are the curves behind Fig 3a. They are trajectories rather than summary
statistics, so no sweep tabulates them; this script writes them out so the
figure script can read them instead of re-running the simulation.

Generated output: data/fig3a_age_profiles.csv
Requires: data/needsim4_isolated.csv (for the tuned burst rates)
Run from the repository root:  python scripts/13_export_age_profiles.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import decoy_sim as ds

TGT, B, KD = 50.0, 5.0, 200.0
N_CYCLES, N_BINS, SEED = 60000, 40, 101

if not os.path.exists("data/needsim4_isolated.csv"):
    raise SystemExit("run scripts/04_sweep_replication.py first")
diso = pd.read_csv("data/needsim4_isolated.csv")

# dose 3 = both the TF gene and the target gene double at replication, which is
# the biological case; the static control never replicates, so dose is moot there.
CONFIGS = [("M = 0 (no reservoir)", 0, 0, 0.5, 0, 3),
           ("M = 141, replicating", 141, 282, 0.5, 141, 3),
           ("M = 212 static (matched buffer)", 212, 212, 1.0, 141, 0),
           ("M = 141, reservoir only", 141, 282, 0.5, 141, 0)]

rows = []
for label, M1, M2, aR, kx_from, dose in CONFIGS:
    sel = diso[(diso.regime == "dilution") & (diso.kd == KD) & (diso.M == kx_from)
               & (diso.dosage == dose)]
    assert len(sel) == 1, (label, kx_from, len(sel))
    kx = float(sel.kx.iloc[0])
    nmax = int(4 * (TGT + 2 * M2 * TGT / (KD + TGT)) + 500)
    E1, V1 = ds.cond_tables(M1, KD, nmax)
    E2, V2 = ds.cond_tables(M2, KD, nmax)
    n = np.arange(nmax + 1)
    st, amf, avf, aw = ds.ssa_cellcycle(
        kx, B, 0.0, 1.0, KD, (n - E1).astype(float), (n - E2).astype(float),
        V1.astype(float), V2.astype(float), nmax, 1.0, aR, int(dose),
        0.06, 1.0, 0.0, N_CYCLES, 300, N_BINS, SEED)
    mean = amf.mean()
    for k, v in enumerate(amf):
        rows.append(dict(series=label, dosage=dose, M_pre=M1, M_post=M2, replication_age=aR,
                         age_bin=k, age_over_T=(k + 0.5) / len(amf),
                         mean_xf=v, mean_xf_over_mean=v / mean, weight=aw[k]))

df = pd.DataFrame(rows)
df.to_csv("data/fig3a_age_profiles.csv", index=False)
for lab, g in df.groupby("series", sort=False):
    print(f"{lab:34s} n={len(g)}  excursion={(g.mean_xf.max()-g.mean_xf.min())/g.mean_xf.mean():.3f}")

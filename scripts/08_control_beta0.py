"""08_control_beta0

Generated output: data/needsim5_beta0.csv
Extracted verbatim from the analysis session that produced the published
numbers; do not edit without re-running verify.py.
Run from the repository root:  python scripts/08_control_beta0.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.getcwd())
import decoy_sim as ds

TGT = 50.0
LN2 = np.log(2.0)
B = 5.0
gf = 5 * LN2

rows5 = []
for kd in (200.0,):
    for M in (0, 100, 400, 1600):
        p0 = TGT / (kd + TGT)
        nmax = int(4 * (TGT + 2 * M) + 500)
        kx = gf * TGT / B
        Eb1, Vb1 = ds.cond_tables(M, kd, nmax)
        Eb2, Vb2 = ds.cond_tables(2 * M, kd, nmax)
        n = np.arange(nmax + 1)
        Ef1 = (n - Eb1).astype(float)
        Ef2 = (n - Eb2).astype(float)
        out = {}
        for lab, aR in (("repl", 0.5), ("static", 1.0)):
            st, amf, avf, aw = ds.ssa_cellcycle(float(kx), B, float(gf), 0.0, float(kd),
                Ef1, Ef2, Vb1.astype(float), Vb2.astype(float), int(nmax), 1.0, aR, 0,
                0.06, 1.0, float(gf), 30000, 200, 40, 101)
            Fage = np.where(amf > 0, avf / np.maximum(amf, 1e-9), np.nan)
            out[lab] = dict(mf=st[0], Ff=ds.fano(st[0], st[1]), CV2y=ds.cv2(st[2], st[3]),
                            Fmin=np.nanmin(Fage), Fmax=np.nanmax(Fage), Fage=Fage)
        r, t = out['repl'], out['static']
        rows5.append(dict(M=M, kd=kd, mf=r['mf'], mf_static=t['mf'], Ff=r['Ff'], Ff_static=t['Ff'],
                          CV2y=r['CV2y'], CV2y_static=t['CV2y'], Fmin=r['Fmin'], Fmax=r['Fmax'],
                          Fage=";".join(f"{v:.4f}" for v in r['Fage'])))
        print(f" {M:5d} {kd:5.0f}  {r['mf']:7.2f}   {r['Ff']:8.3f}   {t['Ff']:9.3f}   "
              f"{r['CV2y']:9.5f}  {t['CV2y']:11.5f}   {r['Fmin']:.2f} - {r['Fmax']:.2f}")

pd.DataFrame(rows5).to_csv("data/needsim5_beta0.csv", index=False)
"""07_scan_cancellation

Generated output: data/cancellation_optimum.csv
Extracted verbatim from the analysis session that produced the published
numbers; do not edit without re-running verify.py.
Run from the repository root:  python scripts/07_scan_cancellation.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.optimize import brentq
import sys
import os

sys.path.insert(0, os.getcwd())
import decoy_sim as ds

TGT = 50.0
LN2 = np.log(2.0)
B = 5.0

_cache = {}

def tab(M, kd, nmax):
    key = (M, kd, nmax)
    if key not in _cache:
        Eb, Vb = ds.cond_tables(M, kd, nmax)
        n = np.arange(nmax + 1)
        _cache[key] = ((n - Eb).astype(float), Vb.astype(float))
    return _cache[key]


def profile(kx, M1, M2, kd, gf, aR, nmax, seed=101, ncyc=60000, dose=0):
    Ef1, Vf1 = tab(M1, kd, nmax)
    Ef2, Vf2 = tab(M2, kd, nmax)
    st, amf, avf, aw = ds.ssa_cellcycle(float(kx), B, float(gf), 1.0, float(kd),
        Ef1, Ef2, Vf1, Vf2, int(nmax), 1.0, float(aR), int(dose), 0.06, 1.0, float(gf),
        int(ncyc), 300, 40, int(seed))
    w = aw / aw.sum()
    mu = (w * amf).sum()
    var_between = (w * (amf - mu) ** 2).sum()
    var_within = (w * avf).sum()
    return dict(mf=st[0], amf=amf, w=w, CV2_cc=var_between / mu ** 2,
                CV2_within=var_within / mu ** 2, CV2f=ds.cv2(st[0], st[1]),
                CV2y=ds.cv2(st[2], st[3]))


rows = []
for dose in (0, 3):
  for kd in (100.0, 200.0, 500.0, 1000.0):
      p0 = TGT / (kd + TGT)
      Mpred = (kd + TGT) / 2
      grid = np.unique(np.round(Mpred * np.array([0.2, 0.4, 0.7, 1.0, 1.5, 2.5, 4.0])).astype(int))
      for M in grid:
          nmax = int(4 * (TGT + 2 * M * p0) + 500)
          f = lambda lk: profile(np.exp(lk), M, 2 * M, kd, 0.0, 0.5, nmax,
                                 ncyc=3000, dose=dose)['mf'] - TGT
          kx = float(np.exp(brentq(f, np.log(0.5), np.log(3e4), xtol=3e-3, maxiter=40)))
          q = profile(kx, M, 2 * M, kd, 0.0, 0.5, nmax, ncyc=60000, dose=dose)
          rows.append(dict(dosage=dose, kd=kd, M=int(M), Mpred=Mpred,
                           mf=q['mf'], cc=q['CV2_cc']))

dv = pd.DataFrame(rows)
dv.to_csv("data/cancellation_optimum.csv", index=False)
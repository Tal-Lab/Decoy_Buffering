"""04_sweep_replication

Generated output: data/needsim4_isolated.csv
Extracted verbatim from the analysis session that produced the published
numbers; do not edit without re-running verify.py.
Run from the repository root:  python scripts/04_sweep_replication.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.optimize import brentq

import sys, os
sys.path.insert(0, os.getcwd())
import decoy_sim as ds

B, Tcyc, By, TGT = 5.0, 1.0, 1.0, 50.0
LN2 = np.log(2.0)
_cache = {}

def tables(M, kd, nmax):
    key = (M, kd, nmax)
    if key not in _cache:
        Eb1, Vb1 = ds.cond_tables(M, kd, nmax)
        Eb2, Vb2 = ds.cond_tables(2*M, kd, nmax)
        n = np.arange(nmax+1)
        _cache[key] = ((n-Eb1).astype(float), (n-Eb2).astype(float),
                       Vb1.astype(float), Vb2.astype(float))
    return _cache[key]

def one(kx, M, kd, gf, aR, ncyc, nmax, ky, seed=7, dose=0):
    Ef1, Ef2, Vf1, Vf2 = tables(M, kd, nmax)
    st, amf, avf, aw = ds.ssa_cellcycle(float(kx), B, float(gf), 1.0, float(kd),
        Ef1, Ef2, Vf1, Vf2, int(nmax), Tcyc, float(aR), int(dose), float(ky), By, float(gf),
        int(ncyc), 200, 40, int(seed))
    out = dict(mf=st[0], CV2f=ds.cv2(st[0],st[1]), my=st[2], CV2y=ds.cv2(st[2],st[3]),
               Ff=ds.fano(st[0],st[1]), age_mf=amf)
    return out

rows = []
Ms = np.unique(np.round(np.logspace(1, 3.3, 9)).astype(int))
for tag, gf in (("dilution", 0.0), ("active-deg", 5*LN2)):
    for kd in (200.0, 500.0):
        p0 = TGT/(kd+TGT)
        Mstar = (kd*B*TGT/(2*0.25*p0**2))**(1/3)      # draft Eq (24)
        for dose, M in [(d, int(m)) for d in (0, 3) for m in np.r_[0, Ms]]:
            nmax = int(4*(TGT + 2*M*p0) + 500)
            ky = 0.06
            f = lambda lk: one(np.exp(lk), M, kd, gf, 0.5, 2500, nmax, ky,
                               dose=dose)['mf'] - TGT
            lk = brentq(f, np.log(0.5), np.log(3e4), xtol=2e-3, maxiter=40)
            kx = float(np.exp(lk))
            rep = one(kx, M, kd, gf, 0.5, 30000, nmax, ky, seed=101, dose=dose)
            # static reference: no replication, so no dosage step either
            sta = one(kx, M, kd, gf, 1.0, 30000, nmax, ky, seed=101, dose=0)
            amf = rep['age_mf']; h = len(amf)//2
            rows.append(dict(regime=tag, kd=kd, dosage=dose, Mstar_pred=Mstar, M=M, kx=kx,
                             mf=rep['mf'], mf_static=sta['mf'],
                             CV2f=rep['CV2f'], CV2f_static=sta['CV2f'],
                             CV2y=rep['CV2y'], CV2y_static=sta['CV2y'],
                             step=(amf[h]-amf[h-1])/amf.mean(),
                             ramp=(amf[h-1]-amf[0])/amf.mean()))
d6 = pd.DataFrame(rows)
d6['dCV2y_repl'] = d6.CV2y - d6.CV2y_static
d6['dCV2f_repl'] = d6.CV2f - d6.CV2f_static
d6.to_csv("data/needsim4_isolated.csv", index=False)
"""02_sweep_extrinsic

Generated output: data/needsim2_matchedmean.csv
Extracted verbatim from the analysis session that produced the published
numbers; do not edit without re-running verify.py.
Run from the repository root:  python scripts/02_sweep_extrinsic.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
import decoy_sim as ds
from scipy.optimize import brentq

B, gf, CV2eta, TARGET = 5.0, 1.0, 0.01, 10.0

def kx_for_mean(M, kd, beta, target=TARGET):
    f = lambda lk: ds.adiabatic_cme(kx=np.exp(lk), B=B, gf=gf, beta=beta, M=int(M), kd=kd)['mf'] - target
    lo, hi = np.log(0.05), np.log(5000.0)
    return float(np.exp(brentq(f, lo, hi, xtol=1e-4)))

rows = []
Ms = np.unique(np.round(np.logspace(0, 3, 13)).astype(int))
for beta in (0.0, 1.0):
    for M in Ms:
        kxm = kx_for_mean(M, 10.0, beta)
        base = ds.adiabatic_cme(kx=kxm, B=B, gf=gf, beta=beta, M=int(M), kd=10.0)
        mix = ds.extrinsic_mix(kxm, B, gf, beta, int(M), 10.0, CV2eta, n_eta=11)
        xf, mb = base['mf'], base['mb']; p = xf/(10.0+xf); chi = M*p*(1-p)
        rows.append(dict(M=int(M), beta=beta, kx=kxm, mf=xf, p=base['p_occ'],
                         eps=base['eps_lna'], CV2_int=mix['CV2f_int'],
                         CV2_ext=mix['CV2f_ext'], CV2_tot=mix['CV2f_tot'],
                         gain=mix['gain'], g_th=(xf+beta*mb)/(xf+beta*chi)))
dm = pd.DataFrame(rows)
dm.to_csv("data/needsim2_matchedmean.csv", index=False)
print("matched mean free TF = 10 for all M:", dm.mf.round(4).unique())
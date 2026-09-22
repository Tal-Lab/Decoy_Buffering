"""01_sweep_thinning

Generated output: data/needsim1.csv
Extracted verbatim from the analysis session that produced the published
numbers; do not edit without re-running verify.py.
Run from the repository root:  python scripts/01_sweep_thinning.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
import decoy_sim as ds

B, kx, gf, F0 = 5.0, 2.0, 1.0, 5.0
rows = []
Ms = np.unique(np.round(np.logspace(0.3, 3.3, 22)).astype(int))
for kd in (3.0, 10.0, 30.0, 100.0, 300.0, 1000.0):
    for beta in (0.0, 1.0):
        for M in Ms:
            r = ds.adiabatic_cme(kx=kx, B=B, gf=gf, beta=beta, M=int(M), kd=kd)
            if r['trunc_tail'] > 1e-9:   # reject untrustworthy truncation
                continue
            rows.append(dict(M=int(M), kd=kd, beta=beta, p=r['p_occ'],
                             Ff=r['Ff'], FT=r['FT'], mf=r['mf'], mT=r['mT'],
                             eps_thin=r['eps_thin'], eps_lna=r['eps_lna'],
                             thin_pred=1+r['eps_thin']*(r['FT']-1),
                             lna_pred=1+r['eps_lna']*(F0-1),
                             CV2f=r['CV2f']))
df = pd.DataFrame(rows)
df['thin_err'] = (df.thin_pred-1)/(df.Ff-1) - 1
df['lna_err']  = (df.lna_pred-1)/(df.Ff-1) - 1
df['hybrid_pred'] = 1 + df.eps_lna*(df.FT-1)
df['hyb_err'] = (df.hybrid_pred-1)/(df.Ff-1) - 1
df.to_csv("data/needsim1.csv", index=False)
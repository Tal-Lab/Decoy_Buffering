"""12_scan_iteron: extrinsic gain when the binding reservoir co-scales with the source.

Two architectures, same reservoir, slow modulation of the driving quantity:
  decoy  -- reservoir size M fixed; synthesis alone is modulated (chromosomal TF
            with genomic decoy sites)
  iteron -- reservoir size and synthesis both scale with the modulating quantity
            (plasmid iterons: the rep gene and the iterons sit on one replicon,
            so a copy-number fluctuation scales both)

Mean-field predictions:  g_decoy  = (x_f + beta*x_b)/(x_f + beta*chi) >= 1
                         g_iteron =  x_f/(x_f + beta*chi)            <= 1
                                  =  epsilon exactly at beta = 1

Generated output: data/iteron_coscale.csv
Run from the repository root:  python scripts/12_scan_iteron.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import norm
import decoy_sim as ds

B, GF, BETA, CV2ETA = 5.0, 1.0, 1.0, 0.0025
KX = 40.0                      # x_f0 = KX*B/GF = 200 free molecules with no reservoir
XF0 = KX * B / GF
KDS = (30.0, 100.0, 300.0)
N_ETA = 13


def meanfield(M, kd, beta=BETA, xf0=XF0):
    """Quasi-static free level and the two gains, from the mean-field balance."""
    c = xf0 - beta * M - kd
    xf = 0.5 * (c + np.sqrt(c * c + 4.0 * kd * xf0))
    p = xf / (kd + xf)
    xb = M * p
    chi = M * p * (1.0 - p)
    return xf, p, (xf + beta * xb) / (xf + beta * chi), xf / (xf + beta * chi)


def measured_gain(M, kd, coscale, n=N_ETA):
    """Exact adiabatic CME, mixed over a slow lognormal modulation of copy number."""
    s = np.sqrt(np.log1p(CV2ETA))
    q = np.linspace(0.5 / n, 1.0 - 0.5 / n, n)
    eta = np.exp(-0.5 * s * s + s * norm.ppf(q))
    w = np.ones(n) / n
    eta = eta / (w * eta).sum()
    means = []
    for e in eta:
        Me = int(round(M * e)) if coscale else int(M)
        r = ds.adiabatic_cme(kx=KX * e, B=B, gf=GF, beta=BETA, M=Me, kd=kd)
        assert r["trunc_tail"] < 1e-9, (M, kd, e, r["trunc_tail"])
        means.append(r["mf"])
    means = np.asarray(means)
    m = (w * means).sum()
    return np.sqrt((w * (means - m) ** 2).sum()) / m / np.sqrt(CV2ETA)


rows = []
Mgrid = np.unique(np.round(np.logspace(0.7, 3.5, 60)).astype(int))
for kd in KDS:
    for M in Mgrid:
        xf, p, gA, gB = meanfield(M, kd)
        rows.append(dict(kind="meanfield", case="decoy", kd=kd, M=int(M), p=p, xf=xf, g=gA))
        rows.append(dict(kind="meanfield", case="iteron", kd=kd, M=int(M), p=p, xf=xf, g=gB))
for kd in KDS:
    for M in (50, 150, 400, 1000):
        xf, p, gA, gB = meanfield(M, kd)
        rows.append(dict(kind="exact", case="decoy", kd=kd, M=M, p=p, xf=xf,
                         g=measured_gain(M, kd, coscale=False)))
        rows.append(dict(kind="exact", case="iteron", kd=kd, M=M, p=p, xf=xf,
                         g=measured_gain(M, kd, coscale=True)))

df = pd.DataFrame(rows)
df.to_csv("data/iteron_coscale.csv", index=False)

ex = df[df.kind == "exact"]
mf = df[df.kind == "meanfield"]
err = []
for _, r in ex.iterrows():
    pred = mf[(mf.case == r.case) & (mf.kd == r.kd) & (mf.M == r.M)]
    if len(pred):
        err.append(abs(r.g / pred.g.iloc[0] - 1))
print(f"rows {len(df)} | exact points {len(ex)} | max |exact/mean-field - 1| = {max(err):.3f}")
print(f"decoy  gain range {ex[ex.case=='decoy'].g.min():.3f}-{ex[ex.case=='decoy'].g.max():.3f}")
print(f"iteron gain range {ex[ex.case=='iteron'].g.min():.3f}-{ex[ex.case=='iteron'].g.max():.3f}")

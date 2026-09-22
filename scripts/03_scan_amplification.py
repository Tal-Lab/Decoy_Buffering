"""03_scan_amplification

Generated output: data/amplification_scan.csv
Extracted verbatim from the analysis session that produced the published
numbers; do not edit without re-running verify.py.
Run from the repository root:  python scripts/03_scan_amplification.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
import decoy_sim as ds

kx, B, gf, CV2eta = 2.0, 5.0, 1.0, 0.0025
rows = []
for kd in (0.3, 1.0, 3.0, 10.0, 30.0):
    for M in np.unique(np.round(np.logspace(0, 2.7, 12)).astype(int)):
        base = ds.adiabatic_cme(kx=kx, B=B, gf=gf, beta=1.0, M=int(M), kd=kd)
        if base['trunc_tail'] > 1e-9:
            continue
        xf, mb = base['mf'], base['mb']
        p = xf / (kd + xf)
        chi = M * p * (1 - p)
        g_th = (xf + mb) / (xf + chi)
        mix = ds.extrinsic_mix(kx, B, gf, 1.0, int(M), kd, CV2eta, n_eta=9)
        rows.append(dict(kd=kd, M=int(M), p=base['p_occ'], xf=xf, g_meas=mix['gain'],
                         g_th=g_th, CV2_ext_ratio=mix['CV2f_ext'] / CV2eta,
                         CV2_int=mix['CV2f_int'], eps=base['eps_lna']))
d3 = pd.DataFrame(rows)
d3.to_csv("data/amplification_scan.csv", index=False)
print("peak amplification per k_d (beta=1):")
print("   kd   M*   p      g_meas  g_th   CV2_ext/CV2_eta")
for kd, s in d3.groupby('kd'):
    r = s.loc[s.g_meas.idxmax()]
    print(f" {kd:5.1f} {r.M:4.0f}  {r.p:.3f}  {r.g_meas:6.3f} {r.g_th:6.3f}   {r.CV2_ext_ratio:6.3f}")
print(f"\nglobal max measured gain: {d3.g_meas.max():.4f} at kd={d3.loc[d3.g_meas.idxmax(), 'kd']}, "
      f"M={d3.loc[d3.g_meas.idxmax(), 'M']:.0f}  -> extrinsic CV^2 inflated {d3.CV2_ext_ratio.max():.3f}x")
print(f"max |g_meas/g_th - 1| = {(d3.g_meas / d3.g_th - 1).abs().max():.3f}")
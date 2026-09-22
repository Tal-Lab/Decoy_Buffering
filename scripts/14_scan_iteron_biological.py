"""14_scan_iteron_biological: the co-scaling result at real iteron numbers.

Iteron arrays carry single-digit numbers of sites. The reservoir available to
the initiator is nevertheless (copy number) x (iterons per plasmid), and both
factors scale with copy number, which is the co-scaling condition. This script
varies the iterons per plasmid m at fixed mean copy number and computes the
transmitted copy-number noise exactly, for a co-scaling reservoir (M = n*m,
the plasmid architecture) and for a fixed reservoir of the same mean size
(chromosomal sites for a plasmid-encoded protein).

Copy number is an integer throughout: n is drawn from a discretised Gaussian
and both the reservoir size and the initiator supply are set by it, so no
rounding of a continuous M is involved.

Generated output: data/iteron_biological.csv
Run from the repository root:  python scripts/14_scan_iteron_biological.py
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import pandas as pd
import decoy_sim as ds

B, GF, BETA = 5.0, 1.0, 1.0
NBAR, CV_N = 10, 0.20          # mean plasmid copy number and its coefficient of variation
S_REP = 5.0                    # initiator molecules supplied per plasmid
M_ITER = range(1, 13)          # iterons per plasmid
KDS = (1.0, 3.0, 10.0, 30.0)
TYPICAL = (3, 9)               # iterons per array in characterised iteron plasmids


def cn_dist(nbar=NBAR, cv=CV_N):
    n = np.arange(1, int(nbar + 5 * cv * nbar + 4) + 1)
    w = np.exp(-0.5 * ((n - nbar) / (cv * nbar)) ** 2)
    return n, w / w.sum()


def transmitted(m_iter, kd, coscale):
    """Exact adiabatic CME per copy-number state, then quasi-static mixing."""
    n, w = cn_dist()
    M_fixed = int(round(NBAR * m_iter))
    means, frees = [], []
    for nn in n:
        M = int(nn * m_iter) if coscale else M_fixed
        r = ds.adiabatic_cme(kx=nn * S_REP * GF / B, B=B, gf=GF, beta=BETA, M=M, kd=kd)
        assert r["trunc_tail"] < 1e-9, (m_iter, kd, nn, r["trunc_tail"])
        means.append(r["mf"])
        frees.append(r["p_occ"])
    means = np.asarray(means)
    mu = (w * means).sum()
    sd = np.sqrt((w * (means - mu) ** 2).sum())
    nmu = (w * n).sum()
    nsd = np.sqrt((w * (n - nmu) ** 2).sum())
    return mu, (sd / mu) / (nsd / nmu), float(np.dot(w, frees)), M_fixed


def meanfield(M, kd, xf0, beta=BETA):
    c = xf0 - beta * M - kd
    xf = 0.5 * (c + np.sqrt(c * c + 4 * kd * xf0))
    p = xf / (kd + xf)
    chi = M * p * (1 - p)
    return xf, p, chi, xf / (xf + beta * chi)


rows = []
for m_iter in M_ITER:
    for kd in KDS:
        mu_c, g_c, p_c, M_fixed = transmitted(m_iter, kd, coscale=True)
        mu_f, g_f, _, _ = transmitted(m_iter, kd, coscale=False)
        xf_mf, p_mf, chi_mf, g_mf = meanfield(M_fixed, kd, NBAR * S_REP)
        rows.append(dict(iterons_per_plasmid=m_iter, copy_number=NBAR, M_mean=M_fixed,
                         rep_per_plasmid=S_REP, kd=kd,
                         mean_free_initiator=mu_c, occupancy=p_c,
                         g_coscaling=g_c, g_fixed=g_f, ratio=g_f / g_c,
                         g_coscaling_meanfield=g_mf, chi_over_xf=chi_mf / xf_mf,
                         typical_array=(TYPICAL[0] <= m_iter <= TYPICAL[1])))

df = pd.DataFrame(rows)
df.to_csv("data/iteron_biological.csv", index=False)

ok = df[df.mean_free_initiator >= 1.5]
typ = ok[ok.typical_array]
print(f"rows {len(df)} | {len(ok)} with mean free initiator >= 1.5 molecules")
print(f"typical arrays (m={TYPICAL[0]}-{TYPICAL[1]}): g_coscaling "
      f"{typ.g_coscaling.min():.3f}-{typ.g_coscaling.max():.3f}, "
      f"g_fixed {typ.g_fixed.min():.3f}-{typ.g_fixed.max():.3f}")
print(f"best attenuation in that range: {typ.g_coscaling.min():.3f} at m="
      f"{int(typ.loc[typ.g_coscaling.idxmin(), 'iterons_per_plasmid'])}, "
      f"k_d={typ.loc[typ.g_coscaling.idxmin(), 'kd']:g}")
print(f"mean-field / exact ratio over the usable set: "
      f"{(ok.g_coscaling_meanfield / ok.g_coscaling).min():.2f}-"
      f"{(ok.g_coscaling_meanfield / ok.g_coscaling).max():.2f}")

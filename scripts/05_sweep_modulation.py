"""05_sweep_modulation

Generated output: data/needsim4_modulation.csv
Extracted verbatim from the analysis session that produced the published
numbers; do not edit without re-running verify.py.
Run from the repository root:  python scripts/05_sweep_modulation.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import sys
import os

# inline decoy_sim module (no numba fallback)
try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False
    def njit(*a, **k):
        if len(a) == 1 and callable(a[0]):
            return a[0]
        return lambda f: f

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


def cond_tables(M, kd, nmax):
    M = int(M); nmax = int(nmax)
    Eb = np.zeros(nmax + 1); Vb = np.zeros(nmax + 1)
    logkd = np.log(kd)
    for n in range(1, nmax + 1):
        bmax = min(n, M)
        b = np.arange(bmax + 1)
        lg = np.zeros(bmax + 1)
        if bmax > 0:
            i = np.arange(bmax)
            lg[1:] = (np.cumsum(np.log(n - i)) + np.cumsum(np.log(M - i))
                      - np.cumsum(np.log(i + 1.0)) - (i + 1.0) * logkd)
        lg -= lg.max()
        p = np.exp(lg); p /= p.sum()
        m1 = p @ b
        Eb[n] = m1
        Vb[n] = p @ (b * b) - m1 * m1
    return Eb, Vb


@njit(cache=True)
def _geom(B):
    if B <= 1.0:
        return 1
    q = 1.0 / B
    u = np.random.random()
    return 1 + int(np.log(1.0 - u) / np.log(1.0 - q))


@njit(cache=True)
def ssa_cellcycle(kx, B, gf, beta, kd, Ef1, Ef2, Vf1, Vf2, nmax,
                  Tcyc, aR, tf_dosage, ky, By, gy, n_cycles, burn_cycles,
                  n_age_bins, seed):
    np.random.seed(seed)
    n = 0
    y = 0
    t = 0.0
    a = 0.0
    cyc = 0
    sf = 0.0; sf2 = 0.0; sy = 0.0; sy2 = 0.0; sT = 0.0; sT2 = 0.0; sb = 0.0; tt = 0.0
    age_mf = np.zeros(n_age_bins); age_vf = np.zeros(n_age_bins); age_n = np.zeros(n_age_bins)
    while cyc < n_cycles:
        post = 1 if a >= aR * Tcyc else 0
        if post == 1:
            Ef = Ef2; Vf = Vf2
        else:
            Ef = Ef1; Vf = Vf1
        nn = n if n <= nmax else nmax
        ef = Ef[nn]; vf = Vf[nn]
        eb = nn - ef
        # tf_dosage is a bitmask: bit 0 = TF gene doubles at replication,
        # bit 1 = the downstream target gene doubles at replication.
        kxe = kx * 2.0 if (post == 1 and (tf_dosage & 1) != 0) else kx
        kye = ky * 2.0 if (post == 1 and (tf_dosage & 2) != 0) else ky
        a0 = kxe
        a1 = gf * ef
        a2 = gf * beta * eb
        a3 = kye * ef
        a4 = gy * y
        A = a0 + a1 + a2 + a3 + a4
        dt = 1e9 if A <= 0.0 else -np.log(np.random.random()) / A
        if post == 0:
            t_edge = aR * Tcyc - a
        else:
            t_edge = Tcyc - a
        if dt > t_edge:
            dt = t_edge
            fire = -1
        else:
            fire = 1
        if cyc >= burn_cycles:
            w = dt
            sf += w * ef; sf2 += w * (vf + ef * ef)
            sy += w * y;  sy2 += w * y * y
            sT += w * nn; sT2 += w * nn * nn
            sb += w * eb; tt += w
            ib = int(a / Tcyc * n_age_bins)
            if ib < 0:
                ib = 0
            if ib >= n_age_bins:
                ib = n_age_bins - 1
            age_mf[ib] += w * ef
            age_vf[ib] += w * (vf + ef * ef)
            age_n[ib] += w
        t += dt; a += dt
        if fire == -1:
            if post == 1 or a >= Tcyc - 1e-12:
                if a >= Tcyc - 1e-12:
                    nk = 0
                    for _ in range(n):
                        if np.random.random() < 0.5:
                            nk += 1
                    n = nk
                    yk = 0
                    for _ in range(y):
                        if np.random.random() < 0.5:
                            yk += 1
                    y = yk
                    a = 0.0; cyc += 1
            continue
        u = np.random.random() * A
        if u < a0:
            n += _geom(B)
            if n > nmax:
                n = nmax
        elif u < a0 + a1:
            n -= 1
        elif u < a0 + a1 + a2:
            n -= 1
        elif u < a0 + a1 + a2 + a3:
            y += _geom(By)
        else:
            y -= 1
        if n < 0:
            n = 0
    stats = np.empty(8)
    stats[0] = sf / tt; stats[1] = sf2 / tt; stats[2] = sy / tt; stats[3] = sy2 / tt
    stats[4] = sT / tt; stats[5] = sT2 / tt; stats[6] = sb / tt; stats[7] = tt
    for i in range(n_age_bins):
        if age_n[i] > 0:
            age_mf[i] /= age_n[i]
            age_vf[i] = age_vf[i] / age_n[i] - age_mf[i] * age_mf[i]
    return stats, age_mf, age_vf, age_n


def cv2(m1, m2):
    return (m2 - m1 ** 2) / m1 ** 2


def fano(m1, m2):
    return (m2 - m1 ** 2) / m1


B, TGT, LN2 = 5.0, 50.0, np.log(2.0)
GF = {"dilution": 0.0, "active-deg": 5 * LN2}

_c = {}

def tab(M, kd, nmax):
    key = (M, kd, nmax)
    if key not in _c:
        Eb1, Vb1 = cond_tables(M, kd, nmax)
        Eb2, Vb2 = cond_tables(2 * M, kd, nmax)
        n = np.arange(nmax + 1)
        _c[key] = ((n - Eb1).astype(float), (n - Eb2).astype(float),
                   Vb1.astype(float), Vb2.astype(float))
    return _c[key]


def run(kx, M1, M2, kd, gf, aR, nmax, seed, ncyc=40000):
    Ef1, Vf1 = tab(M1, kd, nmax)[0], tab(M1, kd, nmax)[2]
    Ef2, Vf2 = tab(M2, kd, nmax)[0], tab(M2, kd, nmax)[2]
    st, amf, avf, aw = ssa_cellcycle(float(kx), B, float(gf), 1.0, float(kd),
        Ef1, Ef2, Vf1, Vf2, int(nmax), 1.0, float(aR), 0, 0.06, 1.0, float(gf),
        int(ncyc), 200, 40, int(seed))
    return dict(mf=st[0], CV2f=cv2(st[0], st[1]), my=st[2], CV2y=cv2(st[2], st[3]))


# Upstream dependency: the burst rates kx were tuned in 04_sweep_replication so that
# the REPLICATING run has mean free TF = 50. Read them rather than re-deriving: an
# independent re-derivation tunes a different configuration and shifts the mean.
if not os.path.exists("data/needsim4_isolated.csv"):
    raise SystemExit("run scripts/04_sweep_replication.py first")
d6 = pd.read_csv("data/needsim4_isolated.csv")

# Now produce needsim4_modulation.csv
rows = []
for _, r in d6[d6.M > 0].iterrows():
    M = int(r.M); kd = float(r.kd); gf = GF[r.regime]; p0 = TGT / (kd + TGT)
    nmax = int(4 * (TGT + 2 * M * p0) + 500); Me = int(round(1.5 * M))
    reps = [run(r.kx, M, 2 * M, kd, gf, 0.5, nmax, s) for s in (101, 202)]
    ctls = [run(r.kx, Me, Me,   kd, gf, 1.0, nmax, s) for s in (101, 202)]
    cy_r = np.mean([x['CV2y'] for x in reps]); cy_c = np.mean([x['CV2y'] for x in ctls])
    cf_r = np.mean([x['CV2f'] for x in reps]); cf_c = np.mean([x['CV2f'] for x in ctls])
    rows.append(dict(regime=r.regime, kd=kd, M=M, Meff=Me,
                     mf=np.mean([x['mf'] for x in reps]),
                     mf_ctl=np.mean([x['mf'] for x in ctls]),
                     CV2y=cy_r, CV2y_ctl=cy_c, dCV2y=cy_r - cy_c,
                     CV2f=cf_r, CV2f_ctl=cf_c, dCV2f=cf_r - cf_c,
                     spread=np.std([x['CV2y'] for x in reps]) + np.std([x['CV2y'] for x in ctls]),
                     CV2cc_pred=0.25 * (p0 * M / TGT) ** 2))

d7 = pd.DataFrame(rows)
d7.to_csv("data/needsim4_modulation.csv", index=False)
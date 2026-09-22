"""06_decompose_cellcycle

Generated output: data/needsim4_cc_decomp.csv
Extracted verbatim from the analysis session that produced the published
numbers; do not edit without re-running verify.py.
Run from the repository root:  python scripts/06_decompose_cellcycle.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from numba import njit
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# ---------------------------------------------------------------- conditional law
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


def _burst_pmf(B, tol=1e-15):
    q = 1.0 / B
    kmax = max(1, int(np.ceil(np.log(tol) / np.log(1 - q)))) if B > 1 else 1
    k = np.arange(1, kmax + 1)
    p = q * (1 - q) ** (k - 1)
    p = p / p.sum()
    return k, p


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


B = 5.0
TGT = 50.0
LN2 = np.log(2.0)
_c = {}


def tab(M, kd, nmax):
    k = (M, kd, nmax)
    if k not in _c:
        Eb, Vb = cond_tables(M, kd, nmax)
        n = np.arange(nmax + 1)
        _c[k] = ((n - Eb).astype(float), Vb.astype(float))
    return _c[k]


def profile(kx, M1, M2, kd, gf, aR, nmax, seed=101, ncyc=60000, dose=0):
    Ef1, Vf1 = tab(M1, kd, nmax)
    Ef2, Vf2 = tab(M2, kd, nmax)
    st, amf, avf, aw = ssa_cellcycle(float(kx), B, float(gf), 1.0, float(kd),
        Ef1, Ef2, Vf1, Vf2, int(nmax), 1.0, float(aR), int(dose), 0.06, 1.0, float(gf),
        int(ncyc), 300, 40, int(seed))
    w = aw / aw.sum()
    mu = (w * amf).sum()
    var_between = (w * (amf - mu) ** 2).sum()
    var_within = (w * avf).sum()
    return dict(mf=st[0], amf=amf, w=w, CV2_cc=var_between / mu ** 2,
                CV2_within=var_within / mu ** 2, CV2f=cv2(st[0], st[1]),
                CV2y=cv2(st[2], st[3]))


# Load d6 data - we need to reconstruct it
# First reconstruct the run function and d6

def adiabatic_cme(kx, B, gf, beta, M, kd, nmax=None, tol=1e-15):
    xf0 = kx * B / gf if gf > 0 else kx * B
    if nmax is None:
        if gf > 0:
            c = xf0 - beta * M - kd
            xfm = 0.5 * (c + np.sqrt(c * c + 4 * kd * xf0))
            mT_est = xfm + M * xfm / (kd + xfm)
            nmax = int(60 + 1.6 * mT_est + 14 * np.sqrt(max(mT_est, 1.0) * max(B, 1.0)))
        else:
            nmax = int(60 + 4 * (xf0 + beta * M) + 25 * np.sqrt(max(xf0, 1.0) * max(B, 1.0)))
    Eb, Vb = cond_tables(M, kd, nmax)
    n = np.arange(nmax + 1)
    Ef = n - Eb
    death = gf * Ef + gf * beta * Eb
    ks, kp = _burst_pmf(B, tol)

    rows_l, cols_l, vals_l = [], [], []
    for i in range(nmax + 1):
        tot = 0.0
        if i > 0 and death[i] > 0:
            rows_l.append(i); cols_l.append(i - 1); vals_l.append(death[i]); tot += death[i]
        for k, pk in zip(ks, kp):
            j = i + k
            if j > nmax:
                j = nmax
            if j == i:
                continue
            r = kx * pk
            rows_l.append(i); cols_l.append(j); vals_l.append(r); tot += r
        rows_l.append(i); cols_l.append(i); vals_l.append(-tot)
    Q = csr_matrix((vals_l, (rows_l, cols_l)), shape=(nmax + 1, nmax + 1))

    A = Q.T.tolil()
    A[0, :] = 1.0
    b = np.zeros(nmax + 1); b[0] = 1.0
    pi = spsolve(A.tocsr(), b)
    pi = np.clip(pi, 0, None); pi /= pi.sum()

    mT = pi @ n; vT = pi @ (n * n) - mT ** 2
    mf = pi @ Ef; vf = pi @ (Vb + Ef ** 2) - mf ** 2
    mb = pi @ Eb
    tail = pi[-1]
    p_occ = mb / M if M > 0 else 0.0
    chi = M * p_occ * (1 - p_occ)
    return dict(mT=mT, vT=vT, FT=vT / mT if mT > 0 else 0,
                mf=mf, vf=vf, Ff=vf / mf if mf > 0 else 0,
                CV2f=vf / mf ** 2 if mf > 0 else 0, mb=mb, p_occ=p_occ, chi=chi,
                eps_lna=mf / (mf + chi) if (mf + chi) > 0 else 0,
                eps_thin=kd / (kd + M) if M > 0 else 1.0,
                xf0=xf0, trunc_tail=tail, nmax=nmax)


GF = {"dilution": 0.0, "active-deg": 5 * LN2}

# Upstream dependency: burst rates kx tuned in 04_sweep_replication. Re-deriving them
# here with brentq(xtol=2e-3) reproduces cc only to ~1.5%; reading the tuned values
# makes this script bit-exact.
if not os.path.exists("data/needsim4_isolated.csv"):
    raise SystemExit("run scripts/04_sweep_replication.py first")
d6 = pd.read_csv("data/needsim4_isolated.csv")

# Now compute needsim4_cc_decomp.csv
prof_data = {}
ccrows = []
for tag in ("dilution", "active-deg"):
    gf = GF[tag]
    for dose, M in [(d, m) for d in (0, 3) for m in
                    (0, 19, 38, 73, 141, 274, 531, 1029, 1995)]:
        kx = float(d6[(d6.regime == tag) & (d6.kd == 200.0) & (d6.M == M)
                      & (d6.dosage == dose)].kx.iloc[0])
        p0 = TGT / 250.0
        nmax = int(4 * (TGT + 2 * M * p0) + 500)
        Me = int(round(1.5 * M))
        q = profile(kx, M, 2 * M, 200.0, gf, 0.5, nmax, dose=dose)
        row = dict(regime=tag, dosage=dose, M=M, cc_rep=q['CV2_cc'])
        if M in (0, 141, 531):
            prof_data[(tag, M, 'rep', dose)] = q
        if M > 0:
            c = profile(kx, Me, Me, 200.0, gf, 1.0, nmax)
            row['cc_ctl'] = c['CV2_cc']
            if M in (141, 531) and dose == 0:
                prof_data[(tag, M, 'ctl')] = c
        ccrows.append(row)
dcc = pd.DataFrame(ccrows)
dcc.to_csv("data/needsim4_cc_decomp.csv", index=False)
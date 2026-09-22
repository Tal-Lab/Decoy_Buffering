"""
Stochastic models of transcription-factor decoy binding.

Model (Soltani 2015 / Dey 2020 convention)
------------------------------------------
  bursts            : rate k_x, size B_x ~ Geometric{1,2,...} with mean B  (so F_0 = B)
  binding           : k_b * x_f * (M - x_b)
  unbinding         : k_u * x_b                  (k_d = k_u / k_b)
  degradation       : gamma_f * x_f ,  gamma_b * x_b = beta * gamma_f * x_b
  target protein y  : bursts at rate k_y * x_f, size ~ Geom mean B_y, decay gamma_y

Three solvers
-------------
 1. exact adiabatic CME  (`adiabatic_cme`)  -- no Monte Carlo error. In the fast
    binding/unbinding limit x_b|x_T is at conditional equilibrium with the exact
    distribution  P(b|n) ~ C(n,b) C(M,b) b! / k_d^b  (detailed balance of the binding
    reaction), so x_T is a Markov jump process with death rate
    gamma_f*E[x_f|n] + gamma_b*E[x_b|n]. Its stationary law is obtained by solving
    pi Q = 0 on a truncated state space.
 2. exact SSA with explicit binding (`ssa_full`) -- no adiabatic assumption; used to
    check 1 and to probe the non-adiabatic regime. Optional OU-modulated k_x.
 3. age-structured cycling cell (`ssa_cellcycle`) -- M doubles at replication, division
    with binomial partitioning, optional simultaneous TF-gene dosage doubling.
"""
import numpy as np
try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:                      # CME/quasi-static solvers work without numba;
    HAVE_NUMBA = False                   # only the SSA kernels need it.
    def njit(*a, **k):
        if len(a) == 1 and callable(a[0]):
            return a[0]
        return lambda f: f
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# ---------------------------------------------------------------- conditional law
def cond_tables(M, kd, nmax):
    """Exact adiabatic conditional moments of x_b given x_T=n, for n=0..nmax.
    Returns (Eb, Vb) with Eb[n]=E[x_b|n], Vb[n]=Var(x_b|n).  Var(x_f|n)=Var(x_b|n)."""
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


# ---------------------------------------------------------------- exact adiabatic CME
def _burst_pmf(B, tol=1e-15):
    """Geometric on {1,2,...} with mean B. Returns (sizes, probs) truncated."""
    q = 1.0 / B
    kmax = max(1, int(np.ceil(np.log(tol) / np.log(1 - q)))) if B > 1 else 1
    k = np.arange(1, kmax + 1)
    p = q * (1 - q) ** (k - 1)
    p = p / p.sum()
    return k, p


def adiabatic_cme(kx, B, gf, beta, M, kd, nmax=None, tol=1e-15):
    """Stationary moments in the adiabatic limit, exact up to state-space truncation."""
    xf0 = kx * B / gf                                    # no-decoy mean free level
    if nmax is None:
        # mean-field free level solves xf0 = x + beta*M*x/(kd+x)
        c = xf0 - beta * M - kd
        xfm = 0.5 * (c + np.sqrt(c * c + 4 * kd * xf0))
        mT_est = xfm + M * xfm / (kd + xfm)               # mean total pool
        nmax = int(60 + 1.6 * mT_est + 14 * np.sqrt(max(mT_est, 1.0) * max(B, 1.0)))
    Eb, Vb = cond_tables(M, kd, nmax)
    n = np.arange(nmax + 1)
    Ef = n - Eb                                          # E[x_f|n]
    death = gf * Ef + gf * beta * Eb                     # slow death rate of x_T
    ks, kp = _burst_pmf(B, tol)

    rows, cols, vals = [], [], []
    for i in range(nmax + 1):
        tot = 0.0
        if death[i] > 0:
            rows.append(i); cols.append(i - 1); vals.append(death[i]); tot += death[i]
        for k, pk in zip(ks, kp):
            j = i + k
            if j > nmax:
                j = nmax                                  # lump tail at the boundary
            if j == i:
                continue
            r = kx * pk
            rows.append(i); cols.append(j); vals.append(r); tot += r
        rows.append(i); cols.append(i); vals.append(-tot)
    Q = csr_matrix((vals, (rows, cols)), shape=(nmax + 1, nmax + 1))

    A = Q.T.tolil()
    A[0, :] = 1.0                                         # replace one eqn by normalisation
    b = np.zeros(nmax + 1); b[0] = 1.0
    pi = spsolve(A.tocsr(), b)
    pi = np.clip(pi, 0, None); pi /= pi.sum()

    mT = pi @ n;            vT = pi @ (n * n) - mT ** 2
    mf = pi @ Ef;           vf = pi @ (Vb + Ef ** 2) - mf ** 2
    mb = pi @ Eb
    tail = pi[-1]
    p_occ = mb / M if M > 0 else 0.0
    chi = M * p_occ * (1 - p_occ)
    return dict(mT=mT, vT=vT, FT=vT / mT, mf=mf, vf=vf, Ff=vf / mf,
                CV2f=vf / mf ** 2, mb=mb, p_occ=p_occ, chi=chi,
                eps_lna=mf / (mf + chi), eps_thin=kd / (kd + M),
                xf0=xf0, trunc_tail=tail, nmax=nmax)


# ---------------------------------------------------------------- quasi-static extrinsic mixing
def extrinsic_mix(kx, B, gf, beta, M, kd, CV2_eta, n_eta=25, dist="lognormal", **kw):
    """Slow multiplicative extrinsic noise on k_x: eta with mean 1, relative variance CV2_eta.
    In the slow-eta limit the system is quasi-static, so
      E[x]   = E_eta[m(eta)]
      Var(x) = E_eta[v(eta)] + Var_eta(m(eta))
    Returns total and decomposed statistics for the free pool."""
    if CV2_eta <= 0:
        r = adiabatic_cme(kx, B, gf, beta, M, kd, **kw)
        return dict(CV2f_tot=r["CV2f"], CV2f_int=r["CV2f"], CV2f_ext=0.0, mf=r["mf"], **r)
    s2 = np.log1p(CV2_eta)
    # Gauss-Hermite nodes for lognormal eta
    x, w = np.polynomial.hermite_e.hermegauss(n_eta)
    w = w / w.sum()
    eta = np.exp(np.sqrt(s2) * x - 0.5 * s2)
    ms, vs = np.zeros(n_eta), np.zeros(n_eta)
    for i, e in enumerate(eta):
        r = adiabatic_cme(kx * e, B, gf, beta, M, kd, **kw)
        ms[i], vs[i] = r["mf"], r["vf"]
    mf = w @ ms
    v_within = w @ vs                      # intrinsic
    v_between = w @ (ms ** 2) - mf ** 2    # extrinsic
    return dict(mf=mf, CV2f_tot=(v_within + v_between) / mf ** 2,
                CV2f_int=v_within / mf ** 2, CV2f_ext=v_between / mf ** 2,
                gain=np.sqrt(v_between / mf ** 2 / CV2_eta))


# ---------------------------------------------------------------- SSA helpers
@njit(cache=True)
def _geom(B):
    """Geometric on {1,2,...} with mean B."""
    if B <= 1.0:
        return 1
    q = 1.0 / B
    u = np.random.random()
    return 1 + int(np.log(1.0 - u) / np.log(1.0 - q))


# ---------------------------------------------------------------- exact SSA, explicit binding
@njit(cache=True)
def ssa_full(kx, B, gf, beta, M, kd, kb, T, burn,
             ky, By, gy, tau_eta, cv2_eta, seed):
    """Exact SSA with explicit binding/unbinding (k_u = kd*kb).
    Optional target protein y (set ky>0) and OU-modulated burst rate
    (set tau_eta>0, cv2_eta>0; eta held as a piecewise-constant telegraph-free
    Ornstein-Uhlenbeck sampled on a fine grid).
    Returns time-averaged [mf, mf2, mT, mT2, my, my2, mb, t_total]."""
    np.random.seed(seed)
    ku = kd * kb
    gb = beta * gf
    xf = 0; xb = 0; y = 0
    eta = 1.0
    sig = np.sqrt(cv2_eta)
    t = 0.0
    # accumulators (time-weighted)
    sf = 0.0; sf2 = 0.0; sT = 0.0; sT2 = 0.0; sy = 0.0; sy2 = 0.0; sb = 0.0; tt = 0.0
    dt_eta = 0.0 if tau_eta <= 0 else tau_eta / 50.0
    t_next_eta = dt_eta
    while t < T:
        kx_e = kx * eta
        a0 = kx_e
        a1 = kb * xf * (M - xb)
        a2 = ku * xb
        a3 = gf * xf
        a4 = gb * xb
        a5 = ky * xf
        a6 = gy * y
        A = a0 + a1 + a2 + a3 + a4 + a5 + a6
        if A <= 0.0:
            break
        dt = -np.log(np.random.random()) / A
        # OU update on a fine grid, integrating observables in between
        if dt_eta > 0.0:
            while t + dt > t_next_eta:
                step = t_next_eta - t
                if t >= burn:
                    w = step
                    sf += w * xf; sf2 += w * xf * xf
                    xT = xf + xb
                    sT += w * xT; sT2 += w * xT * xT
                    sy += w * y;  sy2 += w * y * y
                    sb += w * xb; tt += w
                t = t_next_eta
                # exact OU step for log-normal-ish multiplicative factor (on log eta)
                lam = np.exp(-dt_eta / tau_eta)
                s2 = np.log(1.0 + cv2_eta)
                mu = -0.5 * s2
                le = np.log(eta)
                le = mu + lam * (le - mu) + np.sqrt(s2 * (1.0 - lam * lam)) * np.random.normal()
                eta = np.exp(le)
                t_next_eta += dt_eta
                kx_e = kx * eta
                a0 = kx_e
                A = a0 + a1 + a2 + a3 + a4 + a5 + a6
                dt = -np.log(np.random.random()) / A
        if t >= burn:
            w = dt
            sf += w * xf; sf2 += w * xf * xf
            xT = xf + xb
            sT += w * xT; sT2 += w * xT * xT
            sy += w * y;  sy2 += w * y * y
            sb += w * xb; tt += w
        t += dt
        u = np.random.random() * A
        if u < a0:
            xf += _geom(B)
        elif u < a0 + a1:
            xf -= 1; xb += 1
        elif u < a0 + a1 + a2:
            xf += 1; xb -= 1
        elif u < a0 + a1 + a2 + a3:
            xf -= 1
        elif u < a0 + a1 + a2 + a3 + a4:
            xb -= 1
        elif u < a0 + a1 + a2 + a3 + a4 + a5:
            y += _geom(By)
        else:
            y -= 1
    out = np.empty(8)
    out[0] = sf / tt; out[1] = sf2 / tt; out[2] = sT / tt; out[3] = sT2 / tt
    out[4] = sy / tt; out[5] = sy2 / tt; out[6] = sb / tt; out[7] = tt
    return out


# ---------------------------------------------------------------- cycling cell, adiabatic binding
@njit(cache=True)
def ssa_cellcycle(kx, B, gf, beta, kd, Ef1, Ef2, Vf1, Vf2, nmax,
                  Tcyc, aR, tf_dosage, ky, By, gy, n_cycles, burn_cycles,
                  n_age_bins, seed):
    """Age-structured cycling cell, adiabatic binding, M -> 2M at age aR*Tcyc.

    Ef1/Vf1 : tables E[x_f|n], Var(x_f|n) for M sites   (pre-replication)
    Ef2/Vf2 : same for 2M sites                          (post-replication)
    tf_dosage : bitmask. 1 = TF gene doubles at aR (kx -> 2kx); 2 = target
        gene doubles (ky -> 2ky); 3 = both. 0 = neither (dosage held fixed).
    Turnover: active degradation gf (free) and beta*gf (bound) PLUS dilution by
    binomial partitioning at division. Set gf=0 for purely dilution-dominated turnover.

    Returns (stats, age_mf, age_vf, age_n) where stats =
      [mf, mf2, my, my2, mT, mT2, mb, t_total] as lineage time averages.
    """
    np.random.seed(seed)
    n = 0            # total TF (free+bound)
    y = 0
    t = 0.0
    a = 0.0          # age within cycle
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
        # time to next scheduled event (replication or division)
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
                    # division: binomial partitioning of TF and target
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


# ---------------------------------------------------------------- convenience wrappers
def cv2(m1, m2):
    return (m2 - m1 ** 2) / m1 ** 2


def fano(m1, m2):
    return (m2 - m1 ** 2) / m1


def run_cellcycle(kx, B, M, kd, beta=1.0, gf=0.0, Tcyc=1.0, aR=0.5, tf_dosage=False,
                  ky=0.0, By=1.0, gy=0.0, n_cycles=6000, burn_cycles=200,
                  n_age_bins=40, seed=1, nmax=None):
    """Wrapper: builds the two conditional-moment tables then runs the lineage SSA."""
    xf0 = kx * B / (gf if gf > 0 else np.log(2.0) / Tcyc)
    if nmax is None:
        nmax = int(60 + 6 * (xf0 + beta * 2 * M) + 30 * np.sqrt(max(xf0, 1.0) * max(B, 1.0)))
    Eb1, Vb1 = cond_tables(M, kd, nmax)
    Eb2, Vb2 = cond_tables(2 * M, kd, nmax)
    n = np.arange(nmax + 1)
    Ef1 = (n - Eb1).astype(np.float64); Ef2 = (n - Eb2).astype(np.float64)
    stats, age_mf, age_vf, age_n = ssa_cellcycle(
        float(kx), float(B), float(gf), float(beta), float(kd),
        Ef1, Ef2, Vb1.astype(np.float64), Vb2.astype(np.float64), int(nmax),
        float(Tcyc), float(aR), int(tf_dosage),
        float(ky), float(By), float(gy), int(n_cycles), int(burn_cycles),
        int(n_age_bins), int(seed))
    res = dict(mf=stats[0], CV2f=cv2(stats[0], stats[1]), Ff=fano(stats[0], stats[1]),
               mT=stats[4], FT=fano(stats[4], stats[5]), mb=stats[6],
               age_mf=age_mf, age_vf=age_vf, age_w=age_n, nmax=nmax)
    if ky > 0:
        res.update(my=stats[2], CV2y=cv2(stats[2], stats[3]))
    return res

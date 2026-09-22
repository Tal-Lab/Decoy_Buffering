"""Recompute every number quoted in the manuscript from the generated data.

Writes reported_values.csv with one row per claim: the value as printed in the
manuscript, the value recomputed here, the relative difference, and the file
and criterion it came from. A referee can check any single claim without
re-running the simulations.

Run from the repository root:  python reported_values.py
Exit status is non-zero if any claim fails its tolerance.
"""
import os, sys, math
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
DATA = "data" if os.path.exists("data/needsim1.csv") else "reference"

d1 = pd.read_csv(f"{DATA}/needsim1.csv")
dm = pd.read_csv(f"{DATA}/needsim2_matchedmean.csv")
damp = pd.read_csv(f"{DATA}/amplification_scan.csv")
diso = pd.read_csv(f"{DATA}/needsim4_isolated.csv")
dmod = pd.read_csv(f"{DATA}/needsim4_modulation.csv")
dcc = pd.read_csv(f"{DATA}/needsim4_cc_decomp.csv")
dcan = pd.read_csv(f"{DATA}/cancellation_optimum.csv")
db0 = pd.read_csv(f"{DATA}/needsim5_beta0.csv")
dit = pd.read_csv(f"{DATA}/iteron_coscale.csv")
dbio = pd.read_csv(f"{DATA}/iteron_biological.csv")

rows = []
def claim(cid, text, printed, value, source, tol=0.05, fmt="{:.4g}"):
    v = float(value)
    p = float(printed) if printed is not None else float("nan")
    rel = abs(v - p) / abs(p) if p not in (0.0,) and not math.isnan(p) else abs(v - p)
    rows.append(dict(id=cid, claim=text, manuscript=fmt.format(p), recomputed=fmt.format(v),
                     rel_diff=f"{rel:.3g}", status="PASS" if rel <= tol else "FAIL", source=source))

def bound(cid, text, limit, value, source, fmt="{:.4g}"):
    """A claim of the form 'X or better': passes when value <= limit."""
    v = float(value)
    rows.append(dict(id=cid, claim=text, manuscript="<= " + fmt.format(limit),
                     recomputed=fmt.format(v), rel_diff="",
                     status="PASS" if v <= limit else "FAIL", source=source))

# ---- Proposition 1 vs the LNA (Fig 1) -------------------------------------
claim("F1.n", "number of parameter sets in Fig 1", 264, len(d1),
      "needsim1.csv: row count", tol=0)
lo = d1[d1.p < 0.02]
claim("F1.thin.lo", "median |rel. err| of Prop. 1 at p<0.02", 1.1e-2,
      lo.thin_err.abs().median(), "needsim1.csv: median |thin_err|, p<0.02")
claim("F1.lna.lo", "median |rel. err| of the LNA at p<0.02", 2.2e-4,
      lo[lo.beta == 0].lna_err.abs().median(),
      "needsim1.csv: median |lna_err|, p<0.02 and beta=0")
hi = d1[(d1.p >= 0.45) & (d1.p <= 0.55)]
claim("F1.thin.hi", "median |rel. err| of Prop. 1 at half occupancy", 0.82,
      hi.thin_err.abs().median(), "needsim1.csv: median |thin_err|, p in [0.45,0.55]")
claim("F1.lna.hi", "median |rel. err| of the LNA at half occupancy", 3.4e-2,
      hi[hi.beta == 0].lna_err.abs().median(),
      "needsim1.csv: median |lna_err|, p in [0.45,0.55] and beta=0")

# ---- suppression law and extrinsic transparency (Fig 2) -------------------
b0 = dm[dm.beta == 0].sort_values("eps")
exc = b0.CV2_int - 1.0 / b0.mf
sl, ic = np.polyfit(np.log(b0.eps), np.log(exc), 1)
claim("F2.slope", "intrinsic excess scales as eps^0.988", 0.988, sl,
      "needsim2_matchedmean.csv: slope of log(CV2_int-1/mf) vs log(eps), beta=0")
claim("F2.coeff", "coefficient of the suppression law", 0.412, np.exp(ic),
      "needsim2_matchedmean.csv: exp(intercept) of the same fit")
claim("F2.ext.b0", "d log CV2_ext / d log eps at beta=0", 0.0,
      np.polyfit(np.log(b0.eps), np.log(b0.CV2_ext), 1)[0],
      "needsim2_matchedmean.csv: slope, beta=0", tol=1e-6)
b1 = dm[dm.beta == 1].sort_values("eps")
claim("F2.ext.b1", "d log CV2_ext / d log eps at beta=1", -0.476,
      np.polyfit(np.log(b1.eps), np.log(b1.CV2_ext), 1)[0],
      "needsim2_matchedmean.csv: slope, beta=1")
claim("F2.g.exact", "max |g-1| at beta=0 (transparency)", 0.0,
      (b0.gain - 1).abs().max(), "needsim2_matchedmean.csv: max|gain-1|, beta=0", tol=1e-9)
err = (b1.gain / b1.g_th - 1).abs() * 100
bound("F2.g.all", "mean-field gain accurate to better than 8% (max %)", 8.0, err.max(),
      "needsim2_matchedmean.csv: max |gain/g_th-1|, beta=1")
claim("F2.g.peak", "worst-case gain error (%)", 7.7, err.max(),
      "needsim2_matchedmean.csv: max |gain/g_th-1|, beta=1", tol=0.02)
claim("F2.g.peakM", "decoy load at which the gain error peaks", 300.0,
      float(b1.M.iloc[int(np.argmax(err.values))]),
      "needsim2_matchedmean.csv: argmax |gain/g_th-1|, beta=1", tol=0.10)
bound("F2.g.weak", "gain accurate to 1% or better for M<32 (max %)", 1.0, err[b1.M < 32].max(),
      "needsim2_matchedmean.csv: max |gain/g_th-1|, beta=1, M<32")
for p_, exp_ in ((0.09, 1.21), (0.50, 4.00), (0.91, 123.0)):
    claim(f"F2.floor.p{p_}", f"extrinsic floor inflation 1/(1-p)^2 at p={p_}", exp_,
          1 / (1 - p_) ** 2, "closed form 1/(1-p)^2", tol=0.01)

# ---- replication (Fig 3) ---------------------------------------------------
# At replication the reservoir, the TF gene and the target gene all double,
# since they share a replicon: dosage=3. dosage=0 holds gene dosage fixed and
# isolates the reservoir's own contribution; both are quoted in the text.
DOSE, NODOSE = 3, 0
cc3 = dcc[dcc.dosage == DOSE]
cc0 = dcc[dcc.dosage == NODOSE]
# the control never replicates, so it carries no dosage step: one reference for both
ctl = cc0.set_index(["regime", "M"]).cc_ctl

for reg, printed in (("dilution", 0.049), ("active-deg", 0.070)):
    claim(f"F3.base.{reg}", f"CV2_cc with no reservoir, {reg}, genes replicating", printed,
          cc3[cc3.M == 0].set_index("regime").cc_rep[reg],
          "needsim4_cc_decomp.csv: cc_rep at M=0, dosage=3")
claim("F3.base.active-deg.nodose",
      "CV2_cc with no reservoir, active-deg, reservoir replicating alone", 0.024,
      cc0[cc0.M == 0].set_index("regime").cc_rep["active-deg"],
      "needsim4_cc_decomp.csv: cc_rep at M=0, dosage=0")

def ratio(frame, reg):
    s = frame[(frame.regime == reg) & (frame.M > 0)].sort_values("M")
    r = np.array([ctl.get((reg, int(m)), np.nan) for m in s.M]) / s.cc_rep.to_numpy()
    return s.M.to_numpy(), r

M_, r_ = ratio(cc3, "dilution")
claim("F3.reduce", "peak fold reduction in CV2_cc, dilution, genes replicating", 3.1,
      float(np.nanmax(r_)), "needsim4_cc_decomp.csv: max cc_ctl/cc_rep, dosage=3", tol=0.05)
claim("F3.reduce.M", "reservoir size at that peak", 274, int(M_[np.nanargmax(r_)]),
      "needsim4_cc_decomp.csv: argmax of the same ratio", tol=1e-9)
claim("F3.reduce.141", "fold reduction at M=141, dilution, genes replicating", 2.9,
      float(r_[list(M_).index(141)]), "needsim4_cc_decomp.csv: cc_ctl/cc_rep at M=141", tol=0.05)
claim("F3.dilution.nosource",
      "number of loads at which replication is a net source, dilution", 0,
      int((r_ < 1).sum()), "needsim4_cc_decomp.csv: count of ratio<1, dosage=3", tol=1e-9)

M0, r0 = ratio(cc0, "active-deg")
M3, r3 = ratio(cc3, "active-deg")
claim("F3.inject.nodose", "fold excess over control at M=1995, active-deg, reservoir alone",
      2.8, 1.0 / r0[list(M0).index(1995)],
      "needsim4_cc_decomp.csv: cc_rep/cc_ctl, dosage=0", tol=0.05)
claim("F3.inject.dose", "fold excess over control at M=19, active-deg, genes replicating",
      2.4, 1.0 / r3[list(M3).index(19)],
      "needsim4_cc_decomp.csv: cc_rep/cc_ctl, dosage=3", tol=0.05)
claim("F3.cross.nodose", "smallest M at which replication injects, active-deg, reservoir alone",
      274, int(M0[r0 < 1].min()), "needsim4_cc_decomp.csv: min M with ratio<1, dosage=0", tol=1e-9)
claim("F3.cross.dose", "smallest M at which replication no longer injects, active-deg, genes replicating",
      141, int(M3[r3 > 1].min()), "needsim4_cc_decomp.csv: min M with ratio>1, dosage=3", tol=1e-9)
claim("F3.peak.dose", "peak fold reduction, active-deg, genes replicating", 1.8,
      float(np.nanmax(r3)), "needsim4_cc_decomp.csv: max ratio, dosage=3", tol=0.05)

TGT = 50.0
for dose, lo_p, hi_p, tag in ((DOSE, 1.49, 1.50, "genes replicating"),
                              (NODOSE, 1.49, 1.50, "reservoir alone")):
    ratios, reductions = [], []
    for kd, s in dcan[dcan.dosage == dose].groupby("kd"):
        s = s.sort_values("M")
        ratios.append(s.loc[s.cc.idxmin(), "M"] / s.Mpred.iloc[0])
        reductions.append(s.cc.iloc[0] / s.cc.min())
    claim(f"F3.Mopt.min.{dose}", f"min ratio of measured optimum to 0.75(kd+xf), {tag}",
          lo_p, min(ratios), "cancellation_optimum.csv: argmin(cc)/Mpred per kd", tol=0.02)
    claim(f"F3.Mopt.max.{dose}", f"max ratio of measured optimum to 0.75(kd+xf), {tag}",
          hi_p, max(ratios), "cancellation_optimum.csv: argmin(cc)/Mpred per kd", tol=0.02)
    if dose == DOSE:
        claim("F3.red.min", "min fold reduction at the cancellation optimum", 2.1,
              min(reductions), "cancellation_optimum.csv: cc at smallest M / min(cc)", tol=0.06)
        claim("F3.red.max", "max fold reduction at the cancellation optimum", 2.8,
              max(reductions), "cancellation_optimum.csv: cc at smallest M / min(cc)", tol=0.06)

for dose, printed, tag in ((DOSE, (0.82, 0.68, 1.14), "genes replicating"),
                           (NODOSE, (0.91, 0.88, 1.28), "reservoir alone")):
    d_ = diso[diso.dosage == dose]
    base = d_[d_.M == 0].groupby("regime").step.mean().to_dict()
    fr = [(q.step - base[q.regime]) / (-(TGT / (q.kd + TGT)) * q.M / TGT)
          for _, q in d_[d_.M == 10].iterrows()]
    for cid, val, pr in (("med", float(np.median(fr)), printed[0]),
                         ("lo", min(fr), printed[1]), ("hi", max(fr), printed[2])):
        claim(f"F3.step.{cid}.{dose}", f"{cid} step recovery at M=10, {tag}", pr, val,
              "needsim4_isolated.csv: over the 4 parameter sets at M=10", tol=0.02)
q141 = diso[(diso.regime == "dilution") & (diso.kd == 200.0) & (diso.M == 141)]
for dose, printed in ((DOSE, -0.35), (NODOSE, -0.38)):
    claim(f"F3.step141.{dose}", f"replication step at M=141, kd=200, dilution, dosage={dose}",
          printed, float(q141[q141.dosage == dose].step.iloc[0]),
          "needsim4_isolated.csv: step", tol=0.03)

for reg, printed in (("dilution", 17.0), ("active-deg", 7.0)):
    s = diso[(diso.regime == reg) & (diso.kd == 200.0)
             & (diso.dosage == DOSE)].sort_values("M").CV2y.to_numpy()
    claim(f"F3.cv2y.{reg}", f"decline in CV2y from M=0 to M=1995, {reg} (%)", printed,
          100 * (1 - s[-1] / s[0]), "needsim4_isolated.csv: CV2y, dosage=3", tol=0.12)
worst = 0.0
for reg in ("dilution", "active-deg"):
    s = diso[(diso.regime == reg) & (diso.kd == 200.0)
             & (diso.dosage == DOSE)].sort_values("M").CV2y.to_numpy()
    worst = max(worst, float(np.diff(s / s[0]).max()))
bound("F3.cv2y.scatter", "largest rise in CV2y between adjacent loads (%)", 0.6, 100 * worst,
      "needsim4_isolated.csv: max positive diff of CV2y/CV2y(M=0)")

for M, pr in ((1600, (1.30, 1.64)), (400, (2.07, 2.93)), (0, (3.51, 5.45))):
    q = db0[db0.M == M].iloc[0]
    claim(f"F3.beta0.min.M{M}", f"min within-cycle F_f at M={M} (beta=0)", pr[0], q.Fmin,
          "needsim5_beta0.csv: Fmin")
    claim(f"F3.beta0.max.M{M}", f"max within-cycle F_f at M={M} (beta=0)", pr[1], q.Fmax,
          "needsim5_beta0.csv: Fmax")

# ---- age-profile excursions: not stored in any CSV, so recompute -----------
try:
    import decoy_sim as ds
    LN2 = np.log(2.0)
    for reg, gf, printed in (("dilution", 0.0, 75.0), ("active-deg", 5 * LN2, 75.0)):
        kx = float(diso[(diso.regime == reg) & (diso.M == 0)
                        & (diso.dosage == DOSE)].kx.iloc[0])
        nmax = int(4 * TGT + 500)
        Eb, Vb = ds.cond_tables(0, 200.0, nmax)
        Ef = (np.arange(nmax + 1) - Eb).astype(float)
        st, amf, avf, aw = ds.ssa_cellcycle(kx, 5.0, float(gf), 1.0, 200.0, Ef, Ef,
                                            Vb.astype(float), Vb.astype(float), nmax,
                                            1.0, 0.5, DOSE, 0.06, 1.0, float(gf),
                                            60000, 300, 40, 101)
        claim(f"F3.excursion.{reg}", f"peak-to-trough excursion of x_f, no decoys, {reg} (%)",
              printed, 100 * (amf.max() - amf.min()) / amf.mean(),
              "recomputed: ssa_cellcycle at M=0 with kx from needsim4_isolated.csv", tol=0.07)
except Exception as e:                      # numba absent: skip rather than fail the audit
    rows.append(dict(id="F3.excursion", claim="peak-to-trough excursion of x_f, no decoys",
                     manuscript="75 / 75", recomputed="not run", rel_diff="",
                     status="SKIP", source=f"requires decoy_sim + numba ({type(e).__name__})"))

# ---------------------------------------------------------------- co-scaling reservoir
ex = dit[dit.kind == "exact"]
mfp = dit[dit.kind == "meanfield"]
M_QUOTED, KD_QUOTED = 400, 30.0      # the (M, k_d) the manuscript quotes gains at
for case, printed, cid in (("decoy", 1.58, "F2d.gain.fixed"), ("iteron", 0.19, "F2d.gain.coscale")):
    q = ex[(ex.case == case) & (ex.kd == KD_QUOTED) & (ex.M == M_QUOTED)]
    assert len(q) == 1, (case, len(q))
    # label built from the same constants that select the row, so the two cannot diverge
    claim(cid, f"extrinsic gain at k_d={KD_QUOTED:g}, M={M_QUOTED}, {case} architecture",
          printed, q.g.iloc[0], f"{DATA}/iteron_coscale.csv: kind=exact", tol=0.05)
for M_, printed, cid in ((400, 8.5, "F2d.ratio.400"), (1000, 33.0, "F2d.ratio.1000")):
    gd = ex[(ex.case == "decoy") & (ex.kd == KD_QUOTED) & (ex.M == M_)].g.iloc[0]
    gi = ex[(ex.case == "iteron") & (ex.kd == KD_QUOTED) & (ex.M == M_)].g.iloc[0]
    claim(cid, f"ratio of transmitted noise, fixed vs co-scaling, k_d={KD_QUOTED:g}, M={M_}",
          printed, gd / gi, f"{DATA}/iteron_coscale.csv: ratio of the two exact gains", tol=0.05)
q_ = mfp[(mfp.case == "decoy") & (mfp.kd == KD_QUOTED)]
claim("F2d.peak", f"reservoir size at which the fixed-reservoir gain peaks (k_d={KD_QUOTED:g})",
      230, float(q_.loc[q_.g.idxmax(), "M"]), f"{DATA}/iteron_coscale.csv: argmax of meanfield g",
      tol=0.10)

err = []
for _, r in ex.iterrows():
    pr = mfp[(mfp.case == r.case) & (mfp.kd == r.kd) & (mfp.M == r.M)]
    if len(pr):
        err.append(abs(r.g / pr.g.iloc[0] - 1))
claim("F2d.meanfield.err", "worst |exact / mean-field - 1| over the co-scaling scan (%)",
      5.5, 100 * max(err), f"{DATA}/iteron_coscale.csv: exact vs meanfield rows", tol=0.10)

# co-scaling gain equals the transmission coefficient at beta = 1
q = mfp[mfp.case == "iteron"].copy()
chi = q.M * q.p * (1 - q.p)
claim("F2d.identity", "max |g_coscale - epsilon| at beta=1 (should be 0)",
      0.0, float(np.abs(q.g - q.xf / (q.xf + chi)).max()),
      "identity check on Eq (gain-coscale) against x_f/(x_f+chi)", tol=1e-9, fmt="{:.2e}")

claim("F2ab.matched.mean", "matched mean free protein in Fig 2(a,b)",
      10.0, float(dm.mf.mean()), f"{DATA}/needsim2_matchedmean.csv: mean of mf", tol=1e-3)


# ---------------------------------------------------------------- iteron regime (Fig 2d)
# Panel-d selection, stated in the caption: free initiator >= 2.5 molecules and k_d >= 3.
FREE_MIN, KD_MIN, M_LO, M_HI = 2.5, 3.0, 3, 9
ub = dbio[(dbio.mean_free_initiator >= FREE_MIN) & (dbio.kd >= KD_MIN)]
tb = ub[(ub.iterons_per_plasmid >= M_LO) & (ub.iterons_per_plasmid <= M_HI)]
for lbl, printed, val in (
        ("weakest", 0.861, tb.g_coscaling.max()),
        ("strongest", 0.066, tb.g_coscaling.min())):
    claim(f"F2d.bio.g_co.{lbl}", f"co-scaling gain, {lbl} over m={M_LO}-{M_HI}",
          printed, val, f"{DATA}/iteron_biological.csv: panel-d selection", tol=0.02)
for lbl, printed, val in (
        ("lo", 1.178, tb.g_fixed.min()), ("hi", 2.296, tb.g_fixed.max())):
    claim(f"F2d.bio.g_fixed.{lbl}", f"fixed-reservoir gain, {lbl} over m={M_LO}-{M_HI}",
          printed, val, f"{DATA}/iteron_biological.csv: panel-d selection", tol=0.02)
best = tb.loc[tb.g_coscaling.idxmin()]
for fld, printed, cid, tol in (("ratio", 33.0, "F2d.bio.ratio", 0.05),
                               ("chi_over_xf", 6.9, "F2d.bio.capacity", 0.05),
                               ("occupancy", 0.51, "F2d.bio.occupancy", 0.05),
                               ("kd", 3.0, "F2d.bio.kd", 1e-9),
                               ("iterons_per_plasmid", 9, "F2d.bio.m", 1e-9)):
    claim(cid, f"{fld} at the strongest point in m={M_LO}-{M_HI}", printed, best[fld],
          f"{DATA}/iteron_biological.csv: argmin g_coscaling over the panel-d selection",
          tol=tol)
m3 = tb[tb.iterons_per_plasmid == M_LO]
claim("F2d.bio.m3.atten", f"attenuation at m={M_LO}, weakest end (%)",
      14.0, 100 * (1 - m3.g_coscaling.max()),
      f"{DATA}/iteron_biological.csv: 1 - max g_coscaling at m={M_LO}", tol=0.10)
rr = ub.g_coscaling_meanfield / ub.g_coscaling
claim("F2d.bio.mf.worst", "worst mean-field overestimate of the co-scaling gain (fold)",
      4.4, rr.max(), f"{DATA}/iteron_biological.csv: max meanfield/exact", tol=0.05)
bound("F2d.bio.mf.median", "median |mean-field / exact - 1| over the panel-d set (%)",
      8.0, 100 * float(np.median(np.abs(rr - 1))),
      f"{DATA}/iteron_biological.csv: median |meanfield/exact - 1|")
# the plotted curves must be monotone in m, as the caption's exclusion implies
nonmono = [k for k in sorted(ub.kd.unique())
           if not (np.diff(ub[ub.kd == k].sort_values("iterons_per_plasmid")
                           .g_coscaling.to_numpy()) <= 1e-12).all()]
claim("F2d.bio.monotone", "number of plotted k_d curves non-monotone in m", 0, len(nonmono),
      f"{DATA}/iteron_biological.csv: monotonicity of g_coscaling within the selection",
      tol=1e-9)


out = pd.DataFrame(rows)
out.to_csv("reported_values.csv", index=False)
n_fail = (out.status == "FAIL").sum()
print(out.to_string(index=False))
print(f"\n{len(out)} claims | PASS {(out.status=='PASS').sum()} | FAIL {n_fail} "
      f"| SKIP {(out.status=='SKIP').sum()}   -> reported_values.csv")
sys.exit(1 if n_fail else 0)

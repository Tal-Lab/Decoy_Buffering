"""10_figure2: extrinsic noise is not buffered, and what changes that.

Reads data/needsim2_matchedmean.csv (02_sweep_extrinsic.py) and
data/iteron_biological.csv (14_scan_iteron_biological.py). Panels (a)-(c) concern a reservoir
of fixed size; panel (d) contrasts that with a reservoir that scales with the
noise source, which is the plasmid-iteron architecture.

Generated output: figures/fig2_extrinsic.png
Run from the repository root:  python scripts/10_figure2.py
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from figstyle import (apply_figure_style, panel_letter, opaque_legend,
                      check_overlaps, META_GREY, KD_COLOURS, ACCENT, ACCENT2)

apply_figure_style(sizes=(8, 7, 6))
CV2ETA = 0.01                       # modulation strength used in 02_sweep_extrinsic
dm = pd.read_csv("data/needsim2_matchedmean.csv")
dit = pd.read_csv("data/iteron_coscale.csv")

fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.4))
(ax_a, ax_b), (ax_c, ax_d) = axs

# ---- (a),(b) intrinsic/extrinsic decomposition at beta = 0 and beta = 1
for ax, beta, ttl in ((ax_a, 0.0, r"$\beta=0$: extrinsic noise passes through"),
                      (ax_b, 1.0, r"$\beta=1$: extrinsic noise is amplified")):
    s = dm[dm.beta == beta].sort_values("M")
    shared = (beta == 0.0)          # panels (a),(b) share colours: label once, in (a)
    ax.axhline(CV2ETA, ls="--", lw=0.9, color=META_GREY, zorder=1,
               label=r"$\mathrm{CV}^2_\eta$ (the source)" if shared else None)
    if beta > 0:
        ax.plot(s.M, CV2ETA / (1 - s.p) ** 2, ls="-.", lw=1.0, color="#4a4a4a",
                zorder=2, label=r"floor $\mathrm{CV}^2_\eta/(1-p)^2$")
    ax.plot(s.M, s.CV2_tot, "-o", ms=3.2, lw=1.8, color=ACCENT, zorder=5, label="total" if shared else None)
    ax.plot(s.M, s.CV2_int, "-s", ms=3.0, lw=1.2, color="#7a9fc0", zorder=4,
            label="intrinsic" if shared else None)
    ax.plot(s.M, s.CV2_ext, "-^", ms=3.0, lw=1.2, color="#3f3f3f", zorder=4,
            label="extrinsic" if shared else None)
    ax.set(xscale="log", yscale="log", xlabel="reservoir size $M$", ylim=(6e-3, 1.6))
    ax.set_title(ttl, loc="left")
ax_a.set_ylabel(r"$\mathrm{CV}^2$ of free protein")
ax_b.set_ylabel(r"$\mathrm{CV}^2$ of free protein")
opaque_legend(ax_a, loc="upper right")
opaque_legend(ax_b, loc="lower right")   # only the floor entry is new in (b)

# ---- (c) the gain saturates at 1/(1-p)
xf = 10.0
Mg = np.logspace(0, 4, 300)
for kd, c in zip((1.0, 3.0, 10.0, 30.0, 100.0), plt.cm.plasma(np.linspace(0.05, 0.78, 5))):
    p = xf / (kd + xf)
    ax_c.plot(Mg, (xf + Mg * p) / (xf + Mg * p * (1 - p)), "-", lw=1.3, color=c,
              zorder=3)
    ax_c.axhline(1 / (1 - p), ls=":", lw=0.8, color=c, zorder=1)
    ax_c.text(1.35e4, 1 / (1 - p), rf"$k_d={kd:g}$", fontsize=6, color=c,
              va="center", ha="left", clip_on=False)
ax_c.plot([], [], ls=":", lw=0.8, color=META_GREY, label=r"limit $1/(1-p)$")
s1 = dm[dm.beta == 1.0].sort_values("M")
ax_c.plot(s1.M, s1.gain, "o", ms=4, mfc="none", mec="k", mew=0.9, zorder=6,
          label=r"exact (CME), $k_d=10$")
ax_c.set(xscale="log", yscale="log", xlim=(1, 1.2e4), xlabel="reservoir size $M$",
         ylabel=r"extrinsic gain $g=\mathrm{CV}_f/\mathrm{CV}_\eta$")
ax_c.set_yticks([1, 2, 5, 10]); ax_c.set_yticklabels(["1", "2", "5", "10"])
ax_c.set_title(r"Fixed reservoir: gain saturates at $1/(1-p)$", loc="left")
opaque_legend(ax_c, loc="lower right")

# ---- (d) real iteron numbers: gain against iterons per plasmid
# Reservoir size is (copy number) x (iterons per plasmid) and both scale with
# copy number, so single-digit arrays still give a reservoir of tens of sites.
db = pd.read_csv("data/iteron_biological.csv")
# Restrict to states where the free initiator pool is resolved. Below a few
# molecules the exact gain becomes non-monotonic in m through discreteness of
# the free pool, which is a property of the small-number limit rather than of
# the mechanism; figS1 shows the same comparison where the pool is large.
FREE_MIN, KD_MIN = 2.5, 3.0
db = db[(db.mean_free_initiator >= FREE_MIN) & (db.kd >= KD_MIN)]
kds_b = sorted(db.kd.unique())
for _k in kds_b:                       # the plotted curves must be monotone in m
    _g = db[db.kd == _k].sort_values("iterons_per_plasmid").g_coscaling.to_numpy()
    assert (np.diff(_g) <= 1e-12).all(), f"g_coscaling not monotone at kd={_k}: {_g}"
lo, hi = 3, 9           # iterons per array in characterised iteron plasmids
ax_d.axvspan(lo, hi, color="#f0e6d2", zorder=0, label="typical iteron array")
ax_d.axhline(1.0, ls="--", lw=0.9, color=META_GREY, zorder=1,
             label="source transmitted unchanged")
for kd, c in zip(kds_b, ("#08306b", "#2171b5", "#6baed6")):
    for col, ls, mk in (("g_coscaling", "-", "s"), ("g_fixed", "--", "o")):
        q = db[db.kd == kd].sort_values("iterons_per_plasmid")
        ax_d.plot(q.iterons_per_plasmid, q[col], ls, lw=1.3, color=c, zorder=3)
        ax_d.plot(q.iterons_per_plasmid, q[col], mk, ms=2.8, color=c, zorder=4)
    ax_d.plot([], [], "-", lw=1.4, color=c, label=rf"$k_d={kd:g}$")
ax_d.plot([], [], "-s", lw=1.3, ms=3.4, color="#3f3f3f", label="co-scaling (iterons)")
ax_d.plot([], [], "--o", lw=1.3, ms=3.4, color="#3f3f3f", label="fixed reservoir")
ax_d.set(yscale="log", xlabel="iterons per plasmid $m$",
         ylabel=r"extrinsic gain $g=\mathrm{CV}_f/\mathrm{CV}_n$")
ax_d.set_xlim(0.4, 12.6); ax_d.set_ylim(0.011, 11.0)
ax_d.set_yticks([0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5])
ax_d.set_yticklabels(["0.02", "0.05", "0.1", "0.2", "0.5", "1", "2", "5"])
ax_d.set_title("At real iteron numbers, co-scaling still reverses the sign", loc="left")
opaque_legend(ax_d, loc="lower left", ncol=2)

for ax, L in zip((ax_a, ax_b, ax_c, ax_d), "abcd"):
    panel_letter(ax, L)
fig.tight_layout()
fig.canvas.draw()
check_overlaps(fig, "fig2")
fig.savefig("figures/fig2_extrinsic.png", dpi=300, bbox_inches="tight")
typ = db[(db.iterons_per_plasmid >= lo) & (db.iterons_per_plasmid <= hi)]
print("wrote figures/fig2_extrinsic.png | panel d rows:", len(db),
      "| typical arrays: g_co %.3f-%.3f, g_fixed %.3f-%.3f"
      % (typ.g_coscaling.min(), typ.g_coscaling.max(),
         typ.g_fixed.min(), typ.g_fixed.max()))

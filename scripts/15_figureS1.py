"""15_figureS1: the co-scaling mechanism over a wide reservoir range.

Supporting figure. The main-text panel is set at real iteron numbers, where the
free initiator pool is only a few molecules; this figure shows the same
comparison over three decades of reservoir size, and shows that the co-scaling
gain collapses onto 1/(1+beta*chi/x_f) for every affinity, so the main-text
result is the small-number end of a smooth law rather than a discreteness
artifact.

Reads data/iteron_coscale.csv (12_scan_iteron.py).
Generated output: figures/figS1_mechanism.png
Run from the repository root:  python scripts/15_figureS1.py
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from figstyle import (apply_figure_style, panel_letter, opaque_legend,
                      check_overlaps, META_GREY, KD_COLOURS)

apply_figure_style(sizes=(8, 7, 6))
dit = pd.read_csv("data/iteron_coscale.csv")
mf = dit[dit.kind == "meanfield"].copy()
ex = dit[dit.kind == "exact"].copy()
for d in (mf, ex):
    d["chi"] = d.M * d.p * (1 - d.p)
    d["chi_over_xf"] = d.chi / d.xf

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 3.1))
kds = sorted(mf.kd.unique())

# ---- (a) gain against reservoir size, both architectures
ax_a.axhline(1.0, ls="--", lw=0.9, color=META_GREY, zorder=1,
             label="source transmitted unchanged")
for kd, c in zip(kds, KD_COLOURS):
    for case, ls in (("decoy", "-"), ("iteron", "--")):
        q = mf[(mf.kd == kd) & (mf.case == case)].sort_values("M")
        ax_a.plot(q.M, q.g, ls, lw=1.4, color=c, zorder=3)
    ax_a.plot([], [], "-", lw=1.4, color=c, label=rf"$k_d={kd:g}$")
for case, mk, lab in (("decoy", "o", "fixed reservoir"),
                      ("iteron", "s", "co-scaling reservoir")):
    q = ex[ex.case == case]
    ax_a.plot(q.M, q.g, mk, ms=4, mfc="none", mec="k", mew=0.9, zorder=6, label=lab)
ax_a.set(xscale="log", yscale="log", xlabel="reservoir size $M$", ylim=(0.03, 3.2),
         ylabel=r"extrinsic gain $g=\mathrm{CV}_f/\mathrm{CV}_\eta$")
ax_a.set_yticks([0.05, 0.1, 0.2, 0.5, 1, 2])
ax_a.set_yticklabels(["0.05", "0.1", "0.2", "0.5", "1", "2"])
ax_a.set_title("Mechanism over three decades", loc="left")
opaque_legend(ax_a, loc="lower left")

# ---- (b) the co-scaling gain is a function of the capacity ratio alone
r = np.logspace(-2, 2.2, 200)
ax_b.plot(r, 1.0 / (1.0 + r), "-", lw=1.6, color="#3f3f3f", zorder=2,
          label=r"$1/(1+\beta\chi/\bar{x}_f)$")
for kd, c in zip(kds, KD_COLOURS):
    q = mf[(mf.kd == kd) & (mf.case == "iteron")].sort_values("chi_over_xf")
    ax_b.plot(q.chi_over_xf, q.g, "o", ms=2.6, color=c, zorder=4, label=rf"$k_d={kd:g}$")
q = ex[ex.case == "iteron"]
ax_b.plot(q.chi_over_xf, q.g, "s", ms=4.5, mfc="none", mec="k", mew=0.9, zorder=6,
          label="exact (CME)")
ax_b.set(xscale="log", yscale="log", xlabel=r"capacity ratio $\beta\chi/\bar{x}_f$",
         ylabel=r"co-scaling gain $g_{\rm co}$")
ax_b.set_title("Collapse onto the capacity ratio", loc="left")
opaque_legend(ax_b, loc="lower left")

for ax, L in zip((ax_a, ax_b), "ab"):
    panel_letter(ax, L)
fig.tight_layout()
fig.canvas.draw()
check_overlaps(fig, "figS1")
fig.savefig("figures/figS1_mechanism.png", dpi=300, bbox_inches="tight")

dev = np.abs(mf[mf.case == "iteron"].g -
             1.0 / (1.0 + mf[mf.case == "iteron"].chi_over_xf)).max()
print(f"wrote figures/figS1_mechanism.png | max |g_co - 1/(1+chi/xf)| = {dev:.2e}")

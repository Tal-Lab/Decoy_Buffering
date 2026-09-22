"""11_figure3: a replicating reservoir in a dividing cell.

Reads data/fig3a_age_profiles.csv (13_export_age_profiles.py),
data/needsim4_cc_decomp.csv (06_decompose_cellcycle.py) and
data/needsim4_isolated.csv (04_sweep_replication.py).

Generated output: figures/fig3_replication.png
Run from the repository root:  python scripts/11_figure3.py
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from figstyle import (apply_figure_style, panel_letter, opaque_legend,
                      check_overlaps, META_GREY, ACCENT, ACCENT2)

apply_figure_style(sizes=(8, 7, 6))
prof = pd.read_csv("data/fig3a_age_profiles.csv")
dcc = pd.read_csv("data/needsim4_cc_decomp.csv")
diso = pd.read_csv("data/needsim4_isolated.csv")
KD = 200.0
# At replication the TF gene and the target gene double along with the reservoir,
# because all three sit on the same replicon. dosage=3 is that case and is the one
# plotted; dosage=0 holds gene dosage fixed and isolates the reservoir's own effect.
DOSE, NODOSE = 3, 0

fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(8.8, 3.0))

# ---- (a) age-resolved free protein, with and without a replicating reservoir
STYLE = [("M = 0 (no reservoir)", "#3f3f3f", "-", 1.1),
         ("M = 141, replicating", ACCENT, "-", 1.9),
         ("M = 212 static (matched buffer)", "#7a9fc0", "--", 1.2),
         ("M = 141, reservoir only", "#c98a8a", "-.", 1.1)]
ax_a.axvline(0.5, ls=":", lw=0.9, color=META_GREY, zorder=1,
             label="replication (age $a/T=0.5$)")
for lab, c, ls, lw in STYLE:
    s = prof[prof.series == lab].sort_values("age_over_T")
    ax_a.plot(s.age_over_T, s.mean_xf_over_mean, ls, color=c, lw=lw,
              zorder=4 if c == ACCENT else 3, label=lab)
ax_a.set(xlabel="cell-cycle age $a/T$", ylim=(0.55, 1.62),
         ylabel=r"$\langle x_f\,|\,a\rangle$ / mean")
ax_a.set_title("Reservoir replication flattens the trajectory", loc="left")
opaque_legend(ax_a, loc="upper left")

# ---- (b) does replication raise or lower cell-cycle variance? Ratio to a
# non-replicating cell at matched cycle-average capacity and matched mean, which
# is the same reference for both curves.
ctl = dcc[dcc.dosage == NODOSE].set_index(["regime", "M"]).cc_ctl
for reg, c, mk, lab in (("dilution", ACCENT, "o", "dilution-dominated"),
                        ("active-deg", ACCENT2, "s", "active degradation")):
    for dose, ls, lw, mfc, tag in ((DOSE, "-", 1.8, c, "genes replicate too"),
                                   (NODOSE, "--", 1.0, "none", "reservoir only")):
        s = dcc[(dcc.regime == reg) & (dcc.dosage == dose) & (dcc.M > 0)].sort_values("M")
        r = [ctl.get((reg, int(m)), np.nan) for m in s.M]
        ax_b.plot(s.M, r / s.cc_rep.to_numpy(), ls + mk, ms=3.4, lw=lw, color=c,
                  mfc=mfc, zorder=4 if dose == DOSE else 3, label=f"{lab}, {tag}")
ax_b.axhline(1.0, ls=":", lw=0.9, color=META_GREY, zorder=1,
             label="no change vs static control")
ax_b.set(xscale="log", yscale="log", xlabel="reservoir size $M$",
         ylabel=r"static control $/$ replicating, $\mathrm{CV}^2_{\rm cc}$")
ax_b.set_yticks([0.4, 0.6, 1, 2, 3])
ax_b.set_yticklabels(["0.4", "0.6", "1", "2", "3"])
ax_b.set_title("Reduced or injected, depending on load", loc="left")
opaque_legend(ax_b, loc="lower left")

# ---- (c) total target-protein noise at matched mean
for reg, c, mk, lab in (("dilution", ACCENT, "o", "dilution-dominated"),
                        ("active-deg", ACCENT2, "s", "active degradation")):
    s = diso[(diso.regime == reg) & (diso.kd == KD)
             & (diso.dosage == DOSE)].sort_values("M")
    ax_c.plot(s.M.clip(lower=8), s.CV2y / s.CV2y.iloc[0], "-" + mk, ms=3.4, lw=1.8,
              color=c, zorder=4, label=lab)
mstar = float(diso[(diso.regime == "dilution") & (diso.kd == KD)
                   & (diso.dosage == DOSE)].Mstar_pred.iloc[0])
ax_c.axvline(mstar, ls="-.", lw=1.0, color="k", zorder=2,
             label=rf"predicted $M^*$ = {mstar:.0f} (trade-off)")
ax_c.axhline(1.0, ls=":", lw=0.9, color=META_GREY, zorder=1, label="value at $M=0$")
ax_c.set(xscale="log", xlabel="reservoir size $M$",
         ylabel=r"$\mathrm{CV}^2_y$ / value at $M=0$")
ax_c.set_title("No optimum near the prediction", loc="left")
opaque_legend(ax_c, loc="lower left")

for ax, L in zip((ax_a, ax_b, ax_c), "abc"):
    panel_letter(ax, L, dx=-0.16)
fig.tight_layout()
fig.canvas.draw()
check_overlaps(fig, "fig3")
fig.savefig("figures/fig3_replication.png", dpi=300, bbox_inches="tight")
print("wrote figures/fig3_replication.png | profiles:", prof.series.nunique(),
      "| cc rows:", len(dcc), f"| M* = {mstar:.0f}")

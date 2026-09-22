"""09_figure1: the thinning identity against the LNA.

Reads data/needsim1.csv (see 01_sweep_thinning.py) and writes the figure. No
simulation is run here, so the figure can be re-rendered in seconds.

Generated output: figures/fig1_thinning.png
Run from the repository root:  python scripts/09_figure1.py
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from figstyle import (apply_figure_style, panel_letter, opaque_legend,
                      check_overlaps, META_GREY)

apply_figure_style(sizes=(8, 7, 6))
d1 = pd.read_csv("data/needsim1.csv")
b0 = d1[d1.beta == 0]                      # Dey Eq (10) is derived at beta = 0

fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.2, 3.15))

# ---- (a) exact Fano excess against the Proposition 1 prediction
lim = [3e-2, 5]
axa.plot(lim, lim, ls="--", lw=0.9, color=META_GREY, zorder=1,
         label="identity holds exactly")
sc = axa.scatter(d1.thin_pred - 1, d1.Ff - 1, c=d1.p, cmap="viridis", s=15,
                 edgecolor="none", vmin=0, vmax=0.55, zorder=3)
axa.set(xscale="log", yscale="log", xlim=lim, ylim=lim,
        xlabel=r"predicted $\epsilon_{\rm thin}(F_T-1)$   [Prop. 1]",
        ylabel=r"exact $F_f-1$")
axa.set_title("Prop. 1 under-predicts once sites fill", loc="left")
opaque_legend(axa, loc="upper left")
cb = fig.colorbar(sc, ax=axa, pad=0.02)
cb.set_label("site occupancy $p$", fontsize=6)
cb.ax.tick_params(labelsize=6)

# ---- (b) relative error of the three predictors
axb.axhline(1e-2, ls=":", lw=0.9, color=META_GREY, zorder=1,
            label="1 % relative error")
for frame, col, lab, c, ms, z in [
        (b0, "lna_err", r"Dey Eq. (10), LNA ($\beta=0$)", "#4a4a4a", 11, 3),
        (d1, "hyb_err", r"identity with $\epsilon=x_f/(x_f{+}\chi)$", "#7a9fc0", 11, 4),
        (d1, "thin_err", r"Prop. 1: $\epsilon_{\rm thin}=k_d/(k_d{+}M)$", "#c1121f", 15, 5)]:
    axb.scatter(frame.p, frame[col].abs(), s=ms, color=c, label=lab, zorder=z,
                edgecolor="none", alpha=0.85 if c == "#c1121f" else 0.6)
axb.set(yscale="log", xlabel="site occupancy $p$",
        ylabel=r"$|$relative error in $F_f-1|$", xlim=(-0.03, 0.80))
axb.set_title('The "exact" identity is the least accurate', loc="left")
opaque_legend(axb, loc="lower right")

for ax, L in zip((axa, axb), "ab"):
    panel_letter(ax, L)
fig.tight_layout()
fig.canvas.draw()
check_overlaps(fig, "fig1")
fig.savefig("figures/fig1_thinning.png", dpi=300, bbox_inches="tight")
print("wrote figures/fig1_thinning.png |", len(d1), "points,", len(b0), "at beta=0")

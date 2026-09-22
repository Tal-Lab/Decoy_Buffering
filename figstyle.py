"""Shared figure styling for the manuscript figures.

Self-contained: no dependency on any external style package. Legibility rule
applied throughout the figure scripts: every reference line and every parameter
family is identified in an opaque legend, never by free-floating text or
end-of-line labels, because those collide with data and with guide lines.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

META_GREY = "#888888"
# colour-blind-safe, prints legibly in greyscale by lightness ordering
KD_COLOURS = ("#08306b", "#2171b5", "#6baed6")
ACCENT = "#c1121f"
ACCENT2 = "#2f6f9f"


def apply_figure_style(sizes=(8, 7, 6)):
    mpl.rcParams.update({
        "font.size": sizes[1],
        "axes.labelsize": sizes[1],
        "axes.titlesize": sizes[2],
        "xtick.labelsize": sizes[2],
        "ytick.labelsize": sizes[2],
        "legend.fontsize": sizes[2],
        "figure.dpi": 100,
        "axes.axisbelow": True,
        "legend.borderpad": 0.35,
        "legend.labelspacing": 0.28,
        "legend.handletextpad": 0.5,
        "legend.handlelength": 1.6,
    })


def panel_letter(ax, letter, dx=-0.13, dy=1.06):
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="right")


def opaque_legend(ax, **kw):
    """Legend with a solid white background, so guide lines cannot show through
    the label text. This is the fix for the 'lines crossing over text' defect."""
    kw.setdefault("frameon", True)
    kw.setdefault("framealpha", 1.0)
    kw.setdefault("facecolor", "white")
    kw.setdefault("edgecolor", "none")
    leg = ax.legend(**kw)
    leg.set_zorder(20)
    return leg


def check_overlaps(fig, name, ignore_substrings=()):
    """Report text-vs-text collisions. Tick labels are excluded, as are any
    labels whose text matches ignore_substrings."""
    r = fig.canvas.get_renderer()
    ticks = set()
    for ax in fig.axes:
        ticks |= set(ax.get_xticklabels(which="both")) | set(ax.get_yticklabels(which="both"))
    items = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text)
             if t.get_text().strip() and t.get_visible() and t not in ticks
             and not any(s in t.get_text() for s in ignore_substrings)]
    bad = []
    for i, (t1, b1) in enumerate(items):
        for t2, b2 in items[i + 1:]:
            if b1.overlaps(b2):
                bad.append((t1.get_text()[:28], t2.get_text()[:28]))
    print(f"[{name}] text-text overlaps: {bad if bad else 'none'}")
    return bad

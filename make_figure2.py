"""
Figure 2: blind-recovery rates, this work vs Odrzywolek SI Table S5.

Cleveland dot plot (position, not bar area). Twelve numbers; a table would
also serve. Stars mark this work's representational-depth cells.
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "snapping_v2_final.csv"
ODR_PATH = HERE / "odrzywolek_si_table_s5.csv"

DEPTHS = [2, 3, 4, 5]


def load_ours():
    by = defaultdict(list)
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            by[(r["function"], int(r["depth"]))].append(r)

    def rate(fn, d):
        rs = by[(fn, d)]
        n = sum(1 for r in rs if int(r["nan_epoch"]) <= 0)
        v = sum(1 for r in rs if r["valid_snap"] == "1")
        return 100.0 * v / n if n else 0.0

    return [rate("exp", d) for d in DEPTHS], [rate("ln", d) for d in DEPTHS]


def load_odr():
    by_d = {}
    with open(ODR_PATH) as f:
        for r in csv.DictReader(f):
            by_d[int(r["depth"])] = float(r["rate_percent"])
    return [by_d[d] for d in DEPTHS]


def _spine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(axis="y", length=0, labelsize=8)
    ax.tick_params(axis="x", length=3, width=0.6, labelsize=8)


def main():
    odr, (ours_exp, ours_ln) = load_odr(), load_ours()
    print("Odrzywolek d=2..5:", [f"{x:.1f}" for x in odr])
    print("Ours exp d=2..5:  ", [f"{x:.1f}" for x in ours_exp])
    print("Ours ln  d=2..5:  ", [f"{x:.1f}" for x in ours_ln])

    series = [
        ("Odrzywolek S5", odr, "0.45", set()),
        ("this work  exp(x)", ours_exp, "#1a1a1a", {0}),
        ("this work  ln(x)", ours_ln, "#1a1a1a", {2}),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(5.8, 4.2), sharex=True)
    y = list(range(len(DEPTHS)))[::-1]

    for ax, (name, vals, color, star_idx) in zip(axes, series):
        _spine(ax)
        ax.hlines(y, 0, vals, color="0.82", lw=0.8)
        ax.plot(vals, y, "o", color=color, ms=5.5, zorder=3)
        for i, (v, yy) in enumerate(zip(vals, y)):
            mark = " *" if i in star_idx else ""
            ax.text(v + 2.0, yy, f"{v:.1f}%{mark}", fontsize=7.5,
                    va="center", color="0.25")
        ax.set_yticks(y)
        ax.set_yticklabels([f"d = {d}" for d in DEPTHS])
        ax.set_xlim(0, 108)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.text(0, 1.08, name, transform=ax.transAxes, fontsize=8.5, va="bottom")
        ax.set_ylim(-0.6, 3.6)

    axes[-1].set_xlabel("blind-recovery rate (%)")
    fig.text(
        0.01, 0.01,
        "* = this work at representational depth (exp d=2, ln d=4). "
        "Odrzywolek targets grow with tree depth; ours are fixed. "
        "n = 32, 64, 64, 448 (S5); 20 (ours; exp d=5 = 18). "
        "snapping_v2_final.csv, odrzywolek_si_table_s5.csv.",
        fontsize=6.5, color="0.35",
    )
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    out = HERE / "figure2_rate_comparison.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

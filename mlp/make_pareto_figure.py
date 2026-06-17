"""
Area-vs-accuracy Pareto figure for the symbolic-vs-MLP study.

Reads results/mlp_pareto.csv (from mlp/synth_sweep.py) and plots logic-cell count
vs fixed-point max-error: MLP cells as a scatter/frontier, the two snapped-EML
units as starred points annotated "bit-exact". Writes notes/figure_symbolic_vs_mlp.png.

    python -m mlp.make_pareto_figure
"""

import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(os.path.dirname(HERE), "results", "mlp_pareto.csv")
OUT_PNG = os.path.join(os.path.dirname(HERE), "notes", "figure_symbolic_vs_mlp.png")


def load_rows():
    with open(CSV_PATH) as f:
        return list(csv.DictReader(f))


def numeric_lcs(r):
    """LC count as int: nextpnr ICESTORM_LC ('lcs') if placed, else the yosys
    SB_LUT4 estimate ('lc_est'); None only if neither is numeric."""
    for key in ("lcs", "lc_est"):
        try:
            return int(r[key])
        except (ValueError, KeyError, TypeError):
            continue
    return None


def is_estimate(r):
    """True if the area number is the pre-P&R yosys-stat proxy, not a placed count."""
    return r.get("area_source") in ("yosys-stat", "none")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_rows()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    for ax, func in zip(axes, ("exp", "ln")):
        mlp = [r for r in rows if r["func"] == func and r["unit"].startswith("mlp_")]
        sym = [r for r in rows if r["func"] == func and r["unit"].startswith("symbolic_")]
        for depth, marker in ((1, "o"), (2, "s")):
            pts = [r for r in mlp if int(r["depth"]) == depth and numeric_lcs(r) is not None]
            if not pts:
                continue
            pts.sort(key=lambda r: numeric_lcs(r))
            color = "C0" if depth == 1 else "C1"
            ax.plot([numeric_lcs(r) for r in pts], [float(r["fp_maxerr"]) for r in pts],
                    marker + "-", color=color, alpha=0.8, label=f"MLP depth {depth}")
            # Overlay hollow markers on synth-estimated (unplaced) points so the
            # reader can tell measured-placed area from the pre-P&R LC proxy.
            est = [r for r in pts if is_estimate(r)]
            if est:
                ax.scatter([numeric_lcs(r) for r in est],
                           [float(r["fp_maxerr"]) for r in est], marker=marker,
                           facecolors="none", edgecolors=color, s=90, zorder=4)
            for r in pts:
                tag = f"h{r['hidden']}" + ("*" if is_estimate(r) else "")
                ax.annotate(tag, (numeric_lcs(r), float(r["fp_maxerr"])),
                            fontsize=7, xytext=(3, 3), textcoords="offset points")
        for r in sym:
            ax.scatter([int(r["lcs"])], [float(r["fp_maxerr"])], marker="*", s=320,
                       color="crimson", zorder=5, edgecolor="k",
                       label="snapped EML (bit-exact)")
            ax.annotate("snapped EML\n(bit-exact)",
                        (int(r["lcs"]), float(r["fp_maxerr"])),
                        fontsize=8, color="crimson", xytext=(8, -4),
                        textcoords="offset points")
        # Direct-ROM baseline (sec 3.5): cost is in EBR, not LCs, so annotate EBR.
        # A feasible ROM is a green diamond at its LC; an EBR-infeasible ROM cannot
        # be honestly placed on an LC axis, so it is a banner note instead.
        rom = [r for r in rows if r["func"] == func and r["unit"].startswith("rom_")]
        for r in rom:
            err = float(r["fp_maxerr"])
            if r.get("area_source") == "rom-infeasible":
                ax.annotate(f"direct ROM: {r['ebr']} EBR — infeasible\n(no iCE40 has >32)",
                            (ax.get_xlim()[1], err), fontsize=7.5, color="green",
                            ha="right", va="bottom", xytext=(-4, 2),
                            textcoords="offset points")
            else:
                ax.scatter([int(r["lcs"])], [err], marker="D", s=70, color="green",
                           zorder=5, edgecolor="k", label="direct ROM (bit-exact)")
                ax.annotate(f"direct ROM ({r['ebr']} EBR)", (int(r["lcs"]), err),
                            fontsize=7.5, color="green", xytext=(6, 9),
                            textcoords="offset points")
        for lim, lbl in ((1280, "HX1K"), (5280, "UP5K"), (7680, "HX8K")):
            ax.axvline(lim, color="gray", ls=":", lw=0.8, alpha=0.6)
            ax.text(lim, ax.get_ylim()[1], lbl, fontsize=6, color="gray",
                    rotation=90, va="top", ha="right")
        ax.set_yscale("log")
        ax.set_xlabel("logic cells (UP5K, stream wrapper; * = synth est.)")
        ax.set_ylabel("fixed-point max error vs true function")
        ax.set_title(f"{func}(x)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Symbolic vs quantized-MLP function units on iCE40: area vs accuracy")
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140)
    print(f"wrote {os.path.relpath(OUT_PNG)}")


if __name__ == "__main__":
    main()

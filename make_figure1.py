"""
Figure 1: valid-snap counts by function and depth.

The paper already prints these 12 cells as Table 1. A heatmap encodes the
same small-n counts by color (area/hue, not position). This script renders
the table itself as the figure.
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "snapping_v2_final.csv"

FUNCS = ["exp", "ln", "sqrt"]
DEPTHS = [2, 3, 4, 5]
MIN_DEPTH = {"exp": 2, "ln": 4, "sqrt": 9}


def load_counts():
    by = defaultdict(list)
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            by[(r["function"], int(r["depth"]))].append(r)
    counts = {}
    for fn in FUNCS:
        for d in DEPTHS:
            rs = by[(fn, d)]
            n = sum(1 for r in rs if int(r["nan_epoch"]) <= 0)
            v = sum(1 for r in rs if r["valid_snap"] == "1")
            counts[(fn, d)] = (v, n)
    return counts


def main():
    counts = load_counts()

    fig, ax = plt.subplots(figsize=(5.8, 2.4))
    ax.axis("off")

    header = ["", "d = 2", "d = 3", "d = 4", "d = 5", "min. depth"]
    rows = []
    for fn in FUNCS:
        row = [f"{fn}(x)"]
        for d in DEPTHS:
            v, n = counts[(fn, d)]
            pct = f"{100.0 * v / n:.0f}%" if n else "—"
            mark = "*" if MIN_DEPTH[fn] == d else ""
            row.append(f"{v}/{n}  {pct}{mark}")
        row.append("9+" if MIN_DEPTH[fn] > 5 else str(MIN_DEPTH[fn]))
        rows.append(row)

    table = ax.table(
        cellText=[header] + rows,
        cellLoc="center",
        loc="center",
        colWidths=[0.14, 0.17, 0.17, 0.17, 0.17, 0.14],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.7)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("none")
        cell.set_linewidth(0)
        cell.set_facecolor("white")
        if r == 0:
            cell.set_text_props(color="0.25", fontsize=7.5)
            cell.visible_edges = "B"
            cell.set_edgecolor("0.55")
            cell.set_linewidth(0.6)
        elif r == len(rows):
            cell.visible_edges = "T"
            cell.set_edgecolor("0.75")
            cell.set_linewidth(0.4)
        if c == 0 and r > 0:
            cell._loc = "left"
            cell.set_text_props(ha="left")

    fig.text(
        0.02, 0.04,
        "* = minimum representational depth. Valid snap = correct form and "
        "post-snap MAE < 0.01. 20 seeds/cell (exp d=5: 18 after NaN). "
        "snapping_v2_final.csv.",
        fontsize=6.5, color="0.35",
    )
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    out = HERE / "figure1_heatmap_v2.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

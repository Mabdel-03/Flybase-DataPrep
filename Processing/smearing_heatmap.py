#!/usr/bin/env python
"""
Per-broad-celltype smearing heatmap across the AFCA integration sweep.

Reads the per-run smearing_metrics.csv files (written by smearing_extract.py) and
renders a cell_type x run heatmap of the normalised 2D compactness — the median
distance of a broad type's cells to its own UMAP centroid, scaled by the global
embedding radius. LOWER = the type stays a tight island; HIGHER = it smears out.

This is the actionable view behind the run-level ranking: it shows *which* cell
type smears in *which* run, so a run that wins overall but blows up one
biologically important compartment can be spotted.
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN_ORDER = ["noscale_pc50", "scale_pc50", "scale_dataset", "scale_sex",
             "scale_nn15_md0.1", "scale_hvg5000"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", default="outputs/03_Integrated/sweep")
    ap.add_argument("--outdir", default="outputs/03_Integrated/sweep/smearing_compare")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    # Discover run dirs that actually have a smearing extraction. Prefer the
    # canonical RUN_ORDER for the original sweep; otherwise fall back to natural
    # sort (so an n_pcs sweep reads pc30 < pc50 < pc75 < pc100).
    import glob
    import re

    found = {os.path.basename(os.path.dirname(os.path.dirname(p))): p
             for p in glob.glob(os.path.join(args.sweep, "*", "smearing",
                                             "smearing_metrics.csv"))}

    def natkey(s):
        return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]

    ordered = [r for r in RUN_ORDER if r in found]
    ordered += sorted((r for r in found if r not in RUN_ORDER), key=natkey)

    frames = []
    for r in ordered:
        df = pd.read_csv(found[r])
        df = df[df.scope == "celltype"][["cell_type", "n", "compactness_norm"]].copy()
        df["run"] = r
        frames.append(df)
    if not frames:
        print("ERROR: no smearing_metrics.csv found")
        return 1
    M = pd.concat(frames)
    runs = [r for r in ordered if r in M.run.unique()]
    piv = M.pivot_table(index="cell_type", columns="run", values="compactness_norm")[runs]
    size = M.groupby("cell_type")["n"].mean().sort_values(ascending=False)
    piv = piv.loc[size.index]

    os.makedirs(args.outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(1.5 * len(runs) + 3, 0.45 * len(piv) + 2))
    data = piv.values
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r",
                   vmin=0, vmax=np.nanpercentile(data, 95))
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels(runs, rotation=45, ha="right")
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels([f"{ct}  (n≈{int(size[ct]):,})" for ct in piv.index], fontsize=8)
    # annotate cells + mark per-row best (tightest) with a box
    rowmin = np.nanargmin(data, axis=1)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="black")
        ax.add_patch(plt.Rectangle((rowmin[i] - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor="blue", lw=2))
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("2D compactness (norm)  —  lower = tighter / less smeared")
    ax.set_title("Per-broad-celltype smearing across runs\n"
                 "(blue box = tightest run for that type; rows by size)",
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(args.outdir, "smearing_celltype_heatmap.png")
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[heatmap] wrote {out}")

    # also dump the pivot as csv
    piv.to_csv(os.path.join(args.outdir, "smearing_celltype_compactness.csv"))
    print(f"[heatmap] wrote {os.path.join(args.outdir, 'smearing_celltype_compactness.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""
Combine per-run smearing extractions into a toe-to-toe comparison of the broad
cell-type UMAPs across the AFCA integration sweep.

Consumes the small cached outputs that Processing/smearing_extract.py writes per
run (smearing_coords.npz + smearing_summary.json) — it never re-reads the
multi-GB h5ads. Produces, under --outdir:

  umap_broad_grid.png        : ONE figure, all runs side by side, SHARED palette
                               + SHARED point size, each panel = that run's own
                               recomputed X_umap coloured by afca_annotation_broad.
                               This is the fair, directly-comparable view the
                               existing per-run figures don't give.
  umap_broad_grid_split.png  : same grid but one faint-grey background + single
                               highlighted type is impractical for 17 types, so
                               instead this renders the SAME grid at higher point
                               size for legibility on screen.
  smearing_metrics_bars.png  : bar charts of the run-level smearing metrics
                               (silhouette_2d, knn_purity, mean_compactness,
                               gaussian_overlap) so "least smeared" is visible.
  smearing_ranking.csv       : run-level metrics + a combined smearing rank.

A shared palette is built from the UNION of broad categories across all runs so
a given cell type is the same colour in every panel — essential for spotting
which run keeps each type as a tight, separated island.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
import numpy as np


def big_palette(names: list[str]) -> dict:
    """Stable name->hex map from concatenated qualitative colormaps (72 colors).
    17 broad types fit comfortably; tiling only if a union ever exceeds 72."""
    base = []
    for cmap_name in ("tab20", "tab20b", "tab20c", "Set3"):
        cmap = matplotlib.colormaps[cmap_name]
        base.extend(to_hex(cmap(i)) for i in range(cmap.N))
    seen, uniq = set(), []
    for c in base:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return {n: uniq[i % len(uniq)] for i, n in enumerate(names)}


def load_runs(run_dirs: list[str]):
    """Load (summary dict, npz) per run dir that has a smearing/ subdir."""
    runs = []
    for d in run_dirs:
        sm = os.path.join(d, "smearing")
        npz_p = os.path.join(sm, "smearing_coords.npz")
        sum_p = os.path.join(sm, "smearing_summary.json")
        if not (os.path.isfile(npz_p) and os.path.isfile(sum_p)):
            continue
        with open(sum_p) as fh:
            summary = json.load(fh)
        npz = np.load(npz_p, allow_pickle=True)
        runs.append({"dir": d, "summary": summary, "npz": npz,
                     "label": str(npz["label"])})
    return runs


def order_runs(runs):
    """Stable, meaningful run order for the grid (control first, then scaled)."""
    pref = ["noscale_pc50", "scale_pc50", "scale_dataset", "scale_sex",
            "scale_hvg5000", "scale_nn15_md0.1",
            # n_pcs sweep variants (ascending PC count)
            "pc30", "pc50", "pc75", "pc100"]
    rank = {n: i for i, n in enumerate(pref)}
    return sorted(runs, key=lambda r: rank.get(r["label"], 999))


def render_grid(runs, palette, outpath, point_size, dpi, title_suffix=""):
    n = len(runs)
    ncols = 3 if n > 4 else min(n, 2)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 5.6 * nrows),
                             squeeze=False)
    for ax in axes.flat:
        ax.axis("off")

    # rank label per run for the title (lower combined rank = less smearing)
    for i, r in enumerate(runs):
        ax = axes[i // ncols][i % ncols]
        npz = r["npz"]
        xy = npz["xy"]
        labels = npz["labels"]
        names = list(npz["label_names"])
        colors = np.array([palette[names[c]] for c in labels])
        # draw in random order so no type is systematically painted on top
        rng = np.random.default_rng(0)
        order = rng.permutation(xy.shape[0])
        ax.scatter(xy[order, 0], xy[order, 1], s=point_size, c=colors[order],
                   linewidths=0, rasterized=True)
        s = r["summary"]
        ax.set_title(
            f"{r['label']}\n"
            f"sil2d={s['silhouette_2d']:.3f}  kNN-purity={s['knn_purity']:.3f}\n"
            f"compact={s['mean_compactness_norm']:.3f}  overlap={s['gaussian_overlap']:.3f}",
            fontsize=10,
        )
        ax.set_aspect("equal")
        ax.axis("off")

    # shared legend (union of broad types) along the bottom
    union = sorted({n for r in runs for n in list(r["npz"]["label_names"])})
    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=7,
                      markerfacecolor=palette[n], markeredgewidth=0, label=n)
               for n in union]
    ncol_leg = 4 if len(union) > 12 else 3
    fig.legend(handles=handles, loc="lower center", ncol=ncol_leg,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        "Broad cell-type UMAPs across integration runs "
        "(each panel = that run's own X_umap, shared palette)" + title_suffix,
        fontsize=14, y=0.995,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] wrote {outpath}", flush=True)


def render_metric_bars(runs, outpath, dpi):
    labels = [r["label"] for r in runs]
    # (key, nice name, higher_is_better)
    metrics = [
        ("silhouette_2d", "2D silhouette (broad)\n↑ better", True),
        ("knn_purity", "kNN label purity\n↑ better", True),
        ("mean_compactness_norm", "Mean compactness (norm)\n↓ better", False),
        ("gaussian_overlap", "Gaussian cloud overlap\n↓ better", False),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    x = np.arange(len(labels))
    for ax, (key, name, hib) in zip(axes, metrics):
        vals = [r["summary"][key] for r in runs]
        best = (max(vals) if hib else min(vals))
        colors = ["#2ca02c" if v == best else "#7f9fd0" for v in vals]
        ax.bar(x, vals, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(name, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        for xi, v in zip(x, vals):
            ax.text(xi, v, f"{v:.3f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=7)
    fig.suptitle("Run-level smearing metrics (green = best per metric)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] wrote {outpath}", flush=True)


def write_ranking(runs, outpath):
    """Combined smearing rank: average of per-metric ranks (1=best per metric)."""
    import csv

    keys = [("silhouette_2d", True), ("knn_purity", True),
            ("mean_compactness_norm", False), ("gaussian_overlap", False)]
    vals = {k: [r["summary"][k] for r in runs] for k, _ in keys}
    ranks = {}
    for k, hib in keys:
        v = np.array(vals[k], dtype=float)
        order = np.argsort(-v if hib else v)  # best first
        rk = np.empty(len(v), dtype=int)
        rk[order] = np.arange(1, len(v) + 1)
        ranks[k] = rk
    combined = np.mean([ranks[k] for k, _ in keys], axis=0)
    final_order = np.argsort(combined)

    with open(outpath, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "run", "silhouette_2d", "knn_purity",
                    "mean_compactness_norm", "gaussian_overlap",
                    "n_cells_full", "combined_rank_score"])
        for pos, i in enumerate(final_order, 1):
            r = runs[i]["summary"]
            w.writerow([pos, runs[i]["label"],
                        f"{r['silhouette_2d']:.4f}", f"{r['knn_purity']:.4f}",
                        f"{r['mean_compactness_norm']:.4f}",
                        f"{r['gaussian_overlap']:.4f}",
                        r["n_cells_full"], f"{combined[i]:.2f}"])
    print(f"[compare] wrote {outpath}", flush=True)
    # also echo a short table to stdout
    print("\n=== Smearing ranking (lower combined score = less smearing) ===")
    for pos, i in enumerate(final_order, 1):
        r = runs[i]["summary"]
        print(f"  {pos}. {runs[i]['label']:18s} "
              f"sil2d={r['silhouette_2d']:.3f} purity={r['knn_purity']:.3f} "
              f"compact={r['mean_compactness_norm']:.3f} "
              f"overlap={r['gaussian_overlap']:.3f} "
              f"score={combined[i]:.2f}")
    return [runs[i]["label"] for i in final_order]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="*",
                    help="Run directories (each containing smearing/). "
                         "Default: outputs/03_Integrated/sweep/*/")
    ap.add_argument("--outdir", default="outputs/03_Integrated/sweep/smearing_compare")
    ap.add_argument("--point-size", type=float, default=1.5)
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    run_dirs = args.run_dirs
    if not run_dirs:
        run_dirs = [d for d in glob.glob("outputs/03_Integrated/sweep/*/")
                    if os.path.isdir(os.path.join(d, "smearing"))]
    runs = load_runs(run_dirs)
    if not runs:
        print("ERROR: no run dirs with smearing/ cache found", flush=True)
        return 1
    runs = order_runs(runs)
    print(f"[compare] {len(runs)} runs: {[r['label'] for r in runs]}", flush=True)

    os.makedirs(args.outdir, exist_ok=True)
    union = sorted({n for r in runs for n in list(r["npz"]["label_names"])})
    palette = big_palette(union)

    render_grid(runs, palette, os.path.join(args.outdir, "umap_broad_grid.png"),
                point_size=args.point_size, dpi=args.dpi)
    render_grid(runs, palette,
                os.path.join(args.outdir, "umap_broad_grid_large.png"),
                point_size=args.point_size * 2.5, dpi=args.dpi,
                title_suffix=" — larger points")
    render_metric_bars(runs, os.path.join(args.outdir, "smearing_metrics_bars.png"),
                       dpi=args.dpi)
    write_ranking(runs, os.path.join(args.outdir, "smearing_ranking.csv"))
    print(f"\n[compare] all outputs in {args.outdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

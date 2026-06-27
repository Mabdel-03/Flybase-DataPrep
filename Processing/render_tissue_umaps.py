#!/usr/bin/env python
"""
Render per-tissue broad + fine cell-type UMAPs for the head / body re-embeddings.

Part (1) of the per-tissue request: each tissue was re-embedded independently
(sbatch_tissue_umap.sh -> by_tissue/<tissue>/fly_annotated.h5ad). For each tissue
this draws that tissue's OWN X_umap coloured by:
  (A) afca_annotation_broad  (17 broad classes)
  (B) afca_annotation        (163 fine types)

Broad uses the same shared palette as the cross-run comparison grids, so broad
colours are consistent everywhere. Fine uses a 72-colour concatenated palette
(scanpy's default greys-out past ~20 categories).

Reads obs + X_umap only via h5py (never materialises the matrices), so it is
fast and memory-light even on the 17 GB annotated objects.
"""
from __future__ import annotations

import argparse
import os

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
import numpy as np


def big_palette(names):
    base = []
    for cm in ("tab20", "tab20b", "tab20c", "Set3"):
        c = matplotlib.colormaps[cm]
        base += [to_hex(c(i)) for i in range(c.N)]
    seen, uniq = set(), []
    for c in base:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return {n: uniq[i % len(uniq)] for i, n in enumerate(names)}


def read_cat(obs, col):
    node = obs[col]
    cats = [c.decode() if isinstance(c, (bytes, bytearray)) else str(c)
            for c in node["categories"][:]]
    return node["codes"][:], cats


def render(xy, codes, cats, palette, title, outpath, dpi, legend_ncol,
           point_size, legend_fontsize):
    valid = codes >= 0
    xy, codes = xy[valid], codes[valid]
    colors = np.array([palette[cats[c]] for c in codes])
    rng = np.random.default_rng(0)
    order = rng.permutation(xy.shape[0])
    fig, ax = plt.subplots(figsize=(13, 11))
    ax.scatter(xy[order, 0], xy[order, 1], s=point_size, c=colors[order],
               linewidths=0, rasterized=True)
    ax.set_aspect("equal")
    ax.axis("off")
    present = [cats[c] for c in sorted(set(codes.tolist()))]
    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=7,
                      markerfacecolor=palette[n], markeredgewidth=0, label=n)
               for n in present]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=legend_fontsize, frameon=False, ncol=legend_ncol)
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[render] wrote {outpath}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--by-tissue-dir", required=True,
                    help="Dir containing <tissue>/fly_annotated.h5ad subdirs.")
    ap.add_argument("--tissues", nargs="+", default=["head", "body"])
    ap.add_argument("--broad-col", default="afca_annotation_broad")
    ap.add_argument("--fine-col", default="afca_annotation")
    ap.add_argument("--embedding", default="X_umap")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    # Build a SHARED broad palette from the union across tissues so head/body
    # broad colours match each other and the earlier grids.
    broad_union = set()
    fine_union = set()
    tissue_data = {}
    for t in args.tissues:
        f = os.path.join(args.by_tissue_dir, t, "fly_annotated.h5ad")
        with h5py.File(f, "r") as h:
            xy = h["obsm"][args.embedding][:, :2].astype(np.float32)
            bcodes, bcats = read_cat(h["obs"], args.broad_col)
            fcodes, fcats = read_cat(h["obs"], args.fine_col)
        tissue_data[t] = (xy, bcodes, bcats, fcodes, fcats)
        broad_union |= set(bcats)
        fine_union |= set(fcats)

    broad_pal = big_palette(sorted(broad_union))
    fine_pal = big_palette(sorted(fine_union))

    for t in args.tissues:
        xy, bcodes, bcats, fcodes, fcats = tissue_data[t]
        outdir = os.path.join(args.by_tissue_dir, t, "figures")
        os.makedirs(outdir, exist_ok=True)
        n = (bcodes >= 0).sum()
        # (A) broad
        render(xy, bcodes, bcats, broad_pal,
               f"{t.capitalize()} — broad cell type ({len(set(bcats))} classes) · "
               f"{n:,} cells · own X_umap (n_pcs=30, Harmony=dataset)",
               os.path.join(outdir, "umap_broad.png"),
               args.dpi, legend_ncol=1, point_size=2.0, legend_fontsize=9)
        # (B) fine
        render(xy, fcodes, fcats, fine_pal,
               f"{t.capitalize()} — fine cell type ({len(set(fcats))} types) · "
               f"{n:,} cells · own X_umap (n_pcs=30, Harmony=dataset)",
               os.path.join(outdir, "umap_fine.png"),
               args.dpi, legend_ncol=3, point_size=2.0, legend_fontsize=4)


if __name__ == "__main__":
    raise SystemExit(main())

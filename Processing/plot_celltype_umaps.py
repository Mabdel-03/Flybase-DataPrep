#!/usr/bin/env python
"""
Plot cell-type-annotated UMAPs from the *original* downloaded AFCA object,
using the authors' own published UMAP embedding (obsm["X_umap"]).

Produces, at both annotation granularities:
  * fine  (afca_annotation, 163 types)
  * broad (afca_annotation_broad, 17 classes)

This reads the atlas as-is (no re-processing) and only renders figures, so it is
safe and read-only with respect to the data object. Mirrors the repo convention
of resolving paths via config/paths.sh (FLY_INPUT_H5AD, FLY_PROCESSING_OUTPUTS).
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import numpy as np
import scanpy as sc


def _big_palette(n: int) -> list:
    """A list of n visually distinct hex colors.

    scanpy's default categorical palette only spans ~20 colors; past that it
    silently renders every point grey. We concatenate several qualitative
    colormaps (72 unique colors) and tile if even more are needed, so a
    163-category annotation gets actual colors rather than grey.
    """
    base = []
    for cmap_name in ("tab20", "tab20b", "tab20c", "Set3"):
        cmap = matplotlib.colormaps[cmap_name]
        base.extend(to_hex(cmap(i)) for i in range(cmap.N))
    # de-duplicate while preserving order
    seen, uniq = set(), []
    for c in base:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    if n <= len(uniq):
        return uniq[:n]
    # more categories than colors: tile (legend disambiguates repeats)
    return [uniq[i % len(uniq)] for i in range(n)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input",
        default=os.environ.get("FLY_INPUT_H5AD"),
        help="Path to the AFCA .h5ad (default: $FLY_INPUT_H5AD).",
    )
    ap.add_argument(
        "--outdir",
        default=os.path.join(
            os.environ.get("FLY_PROCESSING_OUTPUTS", "outputs"),
            "Original_UMAPs",
        ),
        help="Directory to write figures into.",
    )
    ap.add_argument(
        "--embedding",
        default="X_umap",
        help="obsm key to plot (default: X_umap = authors' published UMAP).",
    )
    ap.add_argument(
        "--dpi", type=int, default=200, help="Figure DPI (default: 200)."
    )
    args = ap.parse_args()

    if not args.input or not os.path.isfile(args.input):
        print(f"ERROR: input h5ad not found: {args.input!r}", file=sys.stderr)
        return 1

    os.makedirs(args.outdir, exist_ok=True)
    sc.settings.figdir = args.outdir
    sc.settings.autoshow = False

    print(f"[plot] reading {args.input}", flush=True)
    # We only need obs + the embedding; read everything but keep it simple/robust.
    adata = sc.read_h5ad(args.input)
    print(f"[plot] loaded {adata.shape[0]:,} cells x {adata.shape[1]:,} genes", flush=True)

    if args.embedding not in adata.obsm:
        print(
            f"ERROR: embedding {args.embedding!r} not in obsm "
            f"(have: {list(adata.obsm.keys())})",
            file=sys.stderr,
        )
        return 1

    # basis name scanpy expects: "X_umap" -> "umap"
    basis = args.embedding[2:] if args.embedding.startswith("X_") else args.embedding

    # (column, n_legend_per_col, legend_loc, point_size, filename)
    panels = [
        ("afca_annotation_broad", 1, "right margin", 4, "umap_broad.png"),
        ("afca_annotation", 4, "right margin", 2, "umap_fine.png"),
    ]

    for col, ncols, loc, size, fname in panels:
        if col not in adata.obs.columns:
            print(f"[plot] WARNING: obs column {col!r} missing, skipping", flush=True)
            continue
        n_cat = adata.obs[col].nunique()
        print(f"[plot] {col} ({n_cat} categories) -> {fname}", flush=True)

        # Wider canvas + smaller font for the 163-type fine legend.
        is_fine = n_cat > 30
        figsize = (18, 12) if is_fine else (12, 9)
        fontsize = 5 if is_fine else 9

        # Ensure the column is categorical and give it an explicit palette when
        # there are more categories than scanpy's default ~20 (else: all grey).
        if str(adata.obs[col].dtype) != "category":
            adata.obs[col] = adata.obs[col].astype("category")
        n_real = len(adata.obs[col].cat.categories)
        if n_real > 20:
            adata.uns[f"{col}_colors"] = _big_palette(n_real)

        fig, ax = plt.subplots(figsize=figsize)
        sc.pl.embedding(
            adata,
            basis=basis,
            color=col,
            ax=ax,
            show=False,
            size=size,
            legend_loc=loc,
            legend_fontsize=fontsize,
            frameon=False,
            title=f"AFCA head+body — {col} ({n_cat} types)",
        )
        # scanpy 1.11 has no legend_ncols; fan the legend into columns ourselves
        # so the 163-type fine legend doesn't run off the bottom of the canvas.
        leg = ax.get_legend()
        if leg is not None and ncols > 1:
            leg.set_ncols(ncols)
        out = os.path.join(args.outdir, fname)
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] wrote {out}", flush=True)

    print("[plot] done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

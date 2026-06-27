#!/usr/bin/env python
"""
Stage 4 — per-compartment subclustering (nested re-analysis).

The global Stage-3 embedding (Processing/integrate_annotate.py) spends one HVG/PCA
basis across all 17 broad compartments, so dense compartments — above all CNS
neuron (34% of cells, 74 fine types) — stay over-compressed into a single blob.
The diagnostic showed the limiting factor is the SHARED gene basis, not batch and
not head/body tissue. The fix is standard atlas practice: subset one broad
compartment and re-derive its OWN HVGs / PCA / Harmony / Leiden / UMAP, so the
basis resolves structure WITHIN the compartment.

This produces a NESTED zoom-in object; it does not modify the global Stage-3
output. Cross-compartment distances are not comparable across nested objects (the
gene basis differs) — keep the global object as the top-level map.

Constraints (same as Stage 3): the AFCA matrix is log1p(normalize_total(1e4)) with
NO raw counts, so HVG uses flavor='seurat' on log data (seurat_v3 needs counts).
HVGs are re-selected from adata.raw (the full 15,992-gene log-norm matrix the
Stage-3 annotated object preserves), NOT from the global 3000-HVG subset.

Example (from the repo root; quote the spaced bucket dirs):
  bash "0 - Data Prep/Processing/run_subcluster.sh" "CNS neuron"
  python "0 - Data Prep/Processing/subcluster_compartment.py" \
      --input "0 - Data Prep/outputs/03_Integrated/sweep/scale_dataset/fly_annotated.h5ad" \
      --compartment "CNS neuron" --resolution 2.0
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc

sc.settings.verbosity = 1


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    rr = repo_root()
    default_in = os.environ.get(
        "FLY_SUBCLUSTER_INPUT",
        str(rr / "outputs" / "03_Integrated" / "sweep" / "scale_dataset" / "fly_annotated.h5ad"),
    )
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=Path(default_in),
                   help="Stage-3 annotated h5ad (must carry full-gene .raw).")
    p.add_argument("--compartment", type=str, default="CNS neuron",
                   help="Value of --broad-col to subset (default: 'CNS neuron').")
    p.add_argument("--broad-col", type=str, default="afca_annotation_broad")
    p.add_argument("--fine-col", type=str, default="afca_annotation",
                   help="Fine label column, used for the UMAP coloring + eval.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Default: <input parent>/subclusters/<compartment slug>/.")

    # method knobs — defaults are the diagnostic's recommendation for CNS neuron
    p.add_argument("--n-hvgs", type=int, default=2000)
    p.add_argument("--n-pcs", type=int, default=50)
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--neighbor-metric", type=str, default="cosine")
    p.add_argument("--umap-min-dist", type=float, default=0.30,
                   help="Higher min_dist spreads a dense compartment out (default 0.30).")
    p.add_argument("--resolution", type=float, default=2.0,
                   help="Leiden resolution; push high to resolve many fine types.")
    p.add_argument("--scale-max-value", type=float, default=10.0)
    p.add_argument("--no-scale", dest="scale", action="store_false", default=True)
    # batch correction: correct the SAME technical batch as Stage 3 (dataset);
    # do NOT correct tissue — head/body neurons of the same type should co-embed.
    p.add_argument("--harmony-batch-key", type=str, default="dataset",
                   help="obs column for Harmony (default 'dataset'). Empty/'none' skips.")
    p.add_argument("--harmony-theta", type=float, default=2.0)
    p.add_argument("--eval", dest="eval", action="store_true", default=True)
    p.add_argument("--no-eval", dest="eval", action="store_false")
    p.add_argument("--eval-subsample", type=int, default=50000)
    p.add_argument("--min-cells", type=int, default=2000,
                   help="Refuse to subcluster a compartment smaller than this "
                        "(too few cells for a stable independent PCA).")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s).strip("_").lower()


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        return 1

    out_dir = args.output_dir or (args.input.parent / "subclusters" / _slug(args.compartment))
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    sc.settings.figdir = str(fig_dir)
    out_h5ad = out_dir / "subcluster.h5ad"
    if out_h5ad.exists() and not args.overwrite:
        print(f"[skip] {out_h5ad} exists (use --overwrite).")
        return 0

    print(f"[load] {args.input}")
    adata = ad.read_h5ad(args.input)
    if args.broad_col not in adata.obs.columns:
        print(f"ERROR: '{args.broad_col}' not in obs.", file=sys.stderr)
        return 1

    # --- subset to the compartment ---
    mask = adata.obs[args.broad_col].astype(str) == args.compartment
    n_sel = int(mask.sum())
    if n_sel == 0:
        avail = sorted(adata.obs[args.broad_col].astype(str).unique())
        print(f"ERROR: no cells for compartment '{args.compartment}'. "
              f"Available: {avail}", file=sys.stderr)
        return 1
    if n_sel < args.min_cells:
        print(f"ERROR: compartment '{args.compartment}' has only {n_sel} cells "
              f"(< --min-cells {args.min_cells}); too small for stable independent "
              f"PCA. Leave it in the global embedding.", file=sys.stderr)
        return 1

    # Rebuild the subset on the FULL-gene log-norm matrix from .raw, so HVGs are
    # re-selected within the compartment rather than from the global 3000 HVGs.
    if adata.raw is None:
        print("ERROR: input has no .raw (full-gene matrix needed to re-pick HVGs).",
              file=sys.stderr)
        return 1
    sub = adata[mask].copy()
    full = sub.raw.to_adata()                 # n_sub x 15,992 (log-norm)
    full.obs = sub.obs.copy()
    full.obsm["X_umap_global"] = sub.obsm["X_umap"].copy()  # keep the old position
    del adata, sub
    full.uns["fly_subcluster"] = {
        "compartment": args.compartment, "n_cells": int(n_sel),
        "parent": str(args.input), "n_hvgs": args.n_hvgs, "n_pcs": args.n_pcs,
        "resolution": args.resolution,
    }
    n_fine = full.obs[args.fine_col].nunique() if args.fine_col in full.obs else None
    print(f"[subset] {args.compartment}: {full.n_obs:,} cells x {full.n_vars:,} genes "
          f"({n_fine} fine types)")

    # --- compartment-internal HVGs (the key change) ---
    sc.pp.highly_variable_genes(full, flavor="seurat", n_top_genes=args.n_hvgs)
    full.raw = full
    full = full[:, full.var["highly_variable"]].copy()
    print(f"[hvg] re-selected {int(full.n_vars)} HVGs within the compartment "
          f"(flavor='seurat').")

    if args.scale:
        sc.pp.scale(full, max_value=args.scale_max_value, zero_center=True)
        print(f"[scale] z-scored {full.n_vars} HVGs (max_value={args.scale_max_value}).")

    sc.tl.pca(full, svd_solver="arpack", n_comps=args.n_pcs, use_highly_variable=True)

    # --- Harmony on the technical batch (not tissue) ---
    bk = args.harmony_batch_key
    skip_h = (not bk) or bk.lower() == "none" or bk not in full.obs.columns
    if not skip_h and full.obs[bk].nunique() < 2:
        print(f"[harmony] '{bk}' has <2 levels in this compartment; skipping.")
        skip_h = True
    if not skip_h:
        import harmonypy as hm
        print(f"[harmony] batch_key={bk}, theta={args.harmony_theta}")
        res = hm.run_harmony(full.obsm["X_pca"], full.obs, bk, theta=args.harmony_theta)
        z = res.Z_corr
        if hasattr(z, "detach"):
            z = z.detach().cpu().numpy()
        z = np.asarray(z)
        emb = z.T if z.shape[0] == args.n_pcs else z
        if emb.shape[0] != full.n_obs:
            raise RuntimeError(f"Harmony shape {emb.shape} != n_obs {full.n_obs}")
        full.obsm["X_harmony"] = emb
        rep = "X_harmony"
    else:
        print("[harmony] SKIPPED — using PCA directly.")
        rep = "X_pca"

    # --- neighbors + Leiden + UMAP (nested) ---
    sc.pp.neighbors(full, use_rep=rep, n_neighbors=args.n_neighbors,
                    n_pcs=args.n_pcs, metric=args.neighbor_metric)
    sub_key = f"leiden_sub_res{args.resolution:g}".replace(".", "_")
    sc.tl.leiden(full, key_added=sub_key, resolution=args.resolution,
                 flavor="igraph", n_iterations=2, directed=False)
    sc.tl.umap(full, min_dist=args.umap_min_dist, random_state=0)
    n_sub_clusters = full.obs[sub_key].nunique()
    print(f"[leiden] {sub_key}: {n_sub_clusters} sub-clusters at resolution "
          f"{args.resolution}")

    full.write_h5ad(out_h5ad)
    print(f"[write] {out_h5ad}")

    # --- evaluation: did the nested embedding resolve the fine types? ---
    if args.eval:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from evaluate_clustering import evaluate_object
            # Score the NESTED cluster key explicitly. The subset inherits the
            # parent's global leiden_res0_* columns in obs, so we must name our
            # own key rather than rely on the default/fallback (which would pick
            # the inherited global columns and report ARI-vs-broad = 0).
            evaluate_object(full, out_dir, neighbor_rep=rep,
                            cell_type_key=args.fine_col,
                            subsample=args.eval_subsample,
                            label=f"sub_{_slug(args.compartment)}",
                            leiden_keys=[sub_key])
        except Exception as exc:
            print(f"[eval][WARN] {type(exc).__name__}: {exc}")

    # --- figures: nested UMAP by fine type + sub-cluster, before/after ---
    if args.fine_col in full.obs.columns:
        n_cat = full.obs[args.fine_col].nunique()
        sc.pl.umap(full, color=args.fine_col, show=False,
                   legend_fontsize=4, size=4, frameon=False,
                   title=f"{args.compartment}: {args.fine_col} ({n_cat} types) — nested",
                   save="_sub_fine.png")
    sc.pl.umap(full, color=sub_key, show=False, legend_loc="on data",
               legend_fontsize=4, size=4, frameon=False,
               title=f"{args.compartment}: {sub_key}", save="_sub_clusters.png")

    # Side-by-side: global position vs nested re-embedding, colored by fine type.
    if args.fine_col in full.obs.columns and "X_umap_global" in full.obsm:
        codes = full.obs[args.fine_col].astype("category").cat.codes.to_numpy()
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 7))
        a1.scatter(full.obsm["X_umap_global"][:, 0], full.obsm["X_umap_global"][:, 1],
                   c=codes, cmap="tab20", s=2, alpha=0.4, rasterized=True)
        a1.set_title(f"Global Stage-3 UMAP\n({args.compartment} cells only)")
        a2.scatter(full.obsm["X_umap"][:, 0], full.obsm["X_umap"][:, 1],
                   c=codes, cmap="tab20", s=2, alpha=0.4, rasterized=True)
        a2.set_title(f"Nested re-embedding\n(compartment-internal HVG/PCA)")
        for a in (a1, a2):
            a.set_xticks([]); a.set_yticks([]); a.set_aspect("equal")
        fig.suptitle(f"{args.compartment}: global blob vs nested zoom "
                     f"(colored by {args.fine_col})", fontsize=13)
        fig.tight_layout()
        fig.savefig(fig_dir / "before_after_global_vs_nested.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
    print(f"[figures] saved to {fig_dir}")
    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

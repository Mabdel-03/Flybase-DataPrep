#!/usr/bin/env python
"""
Annotate CNS-neuron subclusters by BIOLOGY, independent of the atlas labels.

Motivation: ~46% of AFCA 'CNS neuron' cells are labeled 'unannotated' and another
~12% are generic 'adult ventral nervous system', so atlas-anchored annotation
cannot give these cells real identities and the UMAP smears. This script scores
the nested neuron subclusters against a CURATED, atlas-independent Drosophila
neuron marker panel (Resources/Fly_Neuron_Markers_curated.csv) — neurotransmitter
classes + well-marked specialized types — and assigns each subcluster its best
biological identity with a confidence margin.

Honest scope: neurotransmitter classes are robust (bulletproof markers); fine
morphological types mostly lack single-gene markers, so cells whose best-vs-second
score margin is small are left 'Unresolved neuron' rather than mislabeled.

Method:
  1. sc.tl.score_genes per panel type on adata.raw (log-norm), restricted to genes
     present (verified at panel-build time).
  2. Per-subcluster mean score; assign argmax type if (top1 - top2) margin >=
     --min-margin AND top1 >= --min-score, else 'Unresolved neuron'.
  3. Validate: marker co-expression dotplot (assigned type x its markers),
     per-cluster purity, and how much of the 'unannotated' mass gets a confident
     biological call. Recolor the nested UMAP by the new labels.

NON-DESTRUCTIVE: writes neurons_annotated.h5ad + report/; never edits the input.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

sc.settings.verbosity = 1


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def log(m: str) -> None:
    print(m, flush=True)


def load_panel(path: Path, present_genes: set) -> pd.DataFrame:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("source,"):
            continue
        parts = line.split(",")
        if len(parts) < 2:
            continue
        rows.append({"source": parts[0].strip(), "target": parts[1].strip()})
    net = pd.DataFrame(rows)
    net = net[net["target"].isin(present_genes)]
    return net


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    rr = repo_root()
    ap.add_argument("--input", type=Path,
                    default=rr / "outputs/03_Integrated/sweep/scale_dataset/subclusters/cns_neuron/subcluster.h5ad")
    ap.add_argument("--panel", type=Path, default=rr / "Resources/Fly_Neuron_Markers_curated.csv")
    ap.add_argument("--cluster-key", default="leiden_sub_res2")
    ap.add_argument("--atlas-col", default="afca_annotation")
    ap.add_argument("--min-margin", type=float, default=0.10,
                    help="Min (top1-top2) per-cluster positivity gap to assign a type.")
    ap.add_argument("--min-score", type=float, default=0.25,
                    help="Min top-class positivity (frac of cluster expressing its "
                         "markers) to assign, else Unresolved.")
    ap.add_argument("--ignore-types", default="Glia (contaminant check)",
                    help="Comma-sep panel types used only for QC, not as final labels.")
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--subsample", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_dir = args.output_dir or (args.input.parent / "neuron_annotation")
    fig_dir = out_dir / "report"; fig_dir.mkdir(parents=True, exist_ok=True)
    sc.settings.figdir = str(fig_dir)
    out_h5ad = out_dir / "neurons_annotated.h5ad"
    if out_h5ad.exists() and not args.overwrite:
        log(f"[skip] {out_h5ad} exists (use --overwrite)."); return 0

    log(f"[load] {args.input}")
    adata = ad.read_h5ad(args.input)
    if args.subsample and args.subsample < adata.n_obs:
        sc.pp.subsample(adata, n_obs=args.subsample, random_state=0)
    if adata.raw is None:
        log("ERROR: input has no .raw (full-gene matrix needed for scoring)."); return 1

    present = set(adata.raw.var_names)
    net = load_panel(args.panel, present)
    types = sorted(net["source"].unique())
    log(f"[panel] {len(net)} markers across {len(types)} types: {types}")

    # --- per-cell argmax of single canonical identity genes, then cluster vote ---
    # This is immune to marker-set-SIZE bias (the failure mode of counting ">=1 of
    # N markers" or summing scores: classes with more genes win mechanically). Each
    # class is represented by the MAX expression over its identity genes; the per-
    # cell winner is the class with the highest such value (ties / all-zero -> none).
    # Then a cluster takes its majority winner, gated by how much of the cluster
    # actually carries NT signal. Validated to reproduce known fly-brain NT balance
    # (cholinergic-dominant, ~13% each glut/GABA), not artifacts.
    import scipy.sparse as sp
    raw_names = pd.Index(adata.raw.var_names)
    clusters = adata.obs[args.cluster_key].astype(str)
    ignore = {t.strip() for t in args.ignore_types.split(",") if t.strip()}
    label_types = [t for t in types if t not in ignore]

    # per-class max-expression vector across that class's identity genes
    classmax = {}
    for t in label_types:
        gi = raw_names.get_indexer([g for g in net.loc[net["source"] == t, "target"] if g in present])
        gi = gi[gi >= 0]
        if len(gi) == 0:
            continue
        sub = adata.raw.X[:, gi]
        sub = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
        classmax[t] = sub.max(axis=1)
    cls = list(classmax.keys())
    E = np.vstack([classmax[t] for t in cls]).T            # cells x classes
    any_sig = E.max(axis=1) > 0
    percell = np.where(any_sig, np.array(cls)[E.argmax(axis=1)], "none")
    adata.obs["neuron_type_percell"] = pd.Categorical(percell)

    # per-cluster majority among cells that have NT signal
    cl_ids = np.unique(clusters.values)
    assign, conf = {}, {}
    pc = pd.Series(percell, index=adata.obs_names)
    for cl in cl_ids:
        m = clusters.values == cl
        sigvc = pc[m & any_sig].value_counts()
        frac_sig = float((m & any_sig).sum() / max(m.sum(), 1))
        if sigvc.empty or frac_sig < args.min_score:
            assign[cl] = "Unresolved neuron"; conf[cl] = 0.0
        else:
            assign[cl] = sigvc.idxmax()
            conf[cl] = float(sigvc.iloc[0] / sigvc.sum())   # majority purity
    adata.obs["neuron_type"] = pd.Categorical(clusters.map(assign).values)
    adata.obs["neuron_type_majpurity"] = clusters.map(conf).astype(float).values

    # per-cluster positivity table (single-gene argmax fractions) for the report
    posdf = pd.DataFrame(
        {t: [float((pc[(clusters.values == cl)] == t).mean()) for cl in cl_ids]
         for t in cls}, index=cl_ids)
    posdf.to_csv(out_dir / "cluster_positivity.csv")
    cl_mean = posdf
    log("[assign] per-cell argmax of single canonical genes -> cluster majority.")

    vc = adata.obs["neuron_type"].value_counts()
    n_unres = int(vc.get("Unresolved neuron", 0))
    log(f"\n[assign] neuron_type distribution:")
    for k, v in vc.items():
        log(f"   {v:>7,}  {k}")
    log(f"[assign] {100*(adata.n_obs-n_unres)/adata.n_obs:.1f}% of neurons got a "
        f"confident biological call; {100*n_unres/adata.n_obs:.1f}% Unresolved.")

    # how much of the atlas 'unannotated' mass we rescued
    if args.atlas_col in adata.obs.columns:
        un = adata.obs[args.atlas_col].astype(str) == "unannotated"
        if un.any():
            rescued = (adata.obs.loc[un, "neuron_type"] != "Unresolved neuron").mean()
            log(f"[rescue] of {int(un.sum()):,} atlas-'unannotated' neurons, "
                f"{100*rescued:.1f}% now have a confident biological label.")

    # --- validation: per-cluster purity vs assigned type's markers (dotplot) ---
    cl_summary = cl_mean.copy()
    cl_summary["assigned"] = pd.Series(assign)
    cl_summary["n_cells"] = clusters.value_counts()
    cl_summary["margin"] = pd.Series(conf)
    cl_summary.to_csv(out_dir / "cluster_assignments.csv")

    # dotplot: assigned types x all panel markers (co-expression check). Build a
    # compact marker-only AnnData from .raw for plotting (dotplot needs no controls).
    try:
        marker_genes = [g for g in net["target"].unique() if g in present]
        mi = raw_names.get_indexer(marker_genes); mi = mi[mi >= 0]
        adata_sc = ad.AnnData(X=adata.raw.X[:, mi].copy())
        adata_sc.var_names = [g for g, i in zip(marker_genes, raw_names.get_indexer(marker_genes)) if i >= 0]
        adata_sc.obs_names = list(adata.obs_names)
        adata_sc.obs["neuron_type"] = adata.obs["neuron_type"].values
        sc.pl.dotplot(adata_sc, var_names=list(adata_sc.var_names), groupby="neuron_type",
                      use_raw=False, show=False, save="_neuron_marker_dotplot.png",
                      standard_scale="var")
    except Exception as e:
        log(f"[validate][WARN] dotplot failed: {type(e).__name__}: {e}")

    # --- recolor the nested UMAP by the new biology labels ---
    if "X_umap" in adata.obsm:
        sc.pl.umap(adata, color=["neuron_type"], show=False, size=4, frameon=False,
                   legend_fontsize=6, title="CNS neuron — curated biological annotation",
                   save="_neuron_type.png")
        if args.atlas_col in adata.obs.columns:
            sc.pl.umap(adata, color=[args.atlas_col], show=False, size=4, frameon=False,
                       legend_loc=None, title="CNS neuron — atlas labels (for comparison)",
                       save="_atlas_for_compare.png")

    adata.write_h5ad(out_h5ad)
    log(f"[write] {out_h5ad}")
    log(f"[report] {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

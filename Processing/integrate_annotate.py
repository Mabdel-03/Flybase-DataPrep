#!/usr/bin/env python3
"""
Flybase adapted Stage-3: integration, clustering, and annotation for the AFCA
Drosophila single-nucleus atlas.

This is a fork of
  ROSMAP_Code/Transcriptomics/Processing/Tsai/Pipeline/03_integration_annotation.py
keeping its embedding/clustering/figure method (HVG -> PCA -> Harmony -> Leiden ->
UMAP) but adapting the front-end to the realities of the AFCA atlas, which the
human pipeline does NOT handle:

  * SINGLE combined file (not per-sample CellBender outputs) -> no concat/glob.
  * PRE-NORMALIZED input: X = log1p(normalize_total(counts, target_sum=1e4)),
    and there is NO raw-count store (.raw is None, .layers empty). So we must
    NOT re-run normalize_total/log1p (double-normalization) and must NOT use
    seurat_v3 HVG (it requires raw counts). resolve_counts() detects this and
    switches to a log-normalized path with flavor="seurat".
  * FLY genes: mito prefix "mt:", ribo "RpL"/"RpS", no hemoglobin. QC tags are
    parameterized, not hardcoded to human "MT-"/"RPS,RPL"/"^HB".
  * ALREADY ANNOTATED: obs["afca_annotation"] (and _broad) are peer-reviewed
    labels. We trust them (copy -> obs["cell_type"]); ORA is an optional,
    non-authoritative cross-check only run if a fly marker RDS is supplied.

Stages 1 (per-sample raw-count QC) and 2 (scDblFinder doublet removal) from the
human pipeline are intentionally NOT run: they are statistically meaningless on a
pre-filtered, normalized, single-file published atlas with no recoverable counts.
"""
from __future__ import annotations

import argparse
import gc
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
import scipy.sparse as sp
import yaml

sc.settings.verbosity = 1
sc.set_figure_params(figsize=(6, 6), frameon=False)


# ----------------------------------------------------------------------------
# Config / variant resolution (lightweight reader of config/*.yaml)
# ----------------------------------------------------------------------------
def repo_root() -> Path:
    # The "0 - Data Prep" bucket root (this file lives in <bucket>/Processing/).
    # data/, outputs/, Resources/ all hang off this.
    return Path(__file__).resolve().parents[1]


def config_dir() -> Path:
    # config/ lives at the git repo root (shared by both buckets), i.e. the
    # parent of the "0 - Data Prep" bucket. Honor an explicit override if set.
    env = os.environ.get("FLY_CONFIG_DIR")
    return Path(env) if env else repo_root().parent / "config"


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def stage3_defaults() -> dict:
    cfg = load_yaml(Path(os.environ.get("FLY_PIPELINE_YAML", str(config_dir() / "pipeline.yaml"))))
    return (cfg.get("stage3", {}) or {}).get("integration", {}) or {}


PRIMARY_ALIASES = {"primary", "canonical", "main", "official"}


def resolve_variant(requested: str) -> tuple[str, dict]:
    variants = load_yaml(Path(os.environ.get("FLY_VARIANTS_YAML", str(config_dir() / "variants.yaml"))))
    block = (variants.get("datasets", {}) or {}).get("fly", {}) or {}
    records = block.get("variants", {}) or {}
    variant_id = requested
    if requested in PRIMARY_ALIASES:
        variant_id = (variants.get("primary", {}) or {}).get("fly")
        if not variant_id:
            raise SystemExit("No primary variant declared for 'fly' in variants.yaml")
    if variant_id not in records:
        raise SystemExit(
            f"Unknown variant '{requested}'. Available: {', '.join(sorted(records))}"
        )
    return variant_id, records[variant_id]


# ----------------------------------------------------------------------------
# Arguments
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    d = stage3_defaults()
    rr = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-h5ad", type=Path,
        default=Path(os.environ.get("FLY_INPUT_H5AD", str(rr / "data" / "adata_headBody_S_v1.0.h5ad"))),
        help="Single combined AFCA atlas .h5ad.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Override output dir. Default: ${FLY_PROCESSING_OUTPUTS}/03_Integrated/<variant>.",
    )
    parser.add_argument("--variant", type=str, default="primary",
                        help="Variant id (or 'primary'). See config/variants.yaml.")

    # input handling
    parser.add_argument("--input-mode", choices=["auto", "normalized", "counts"],
                        default=d.get("input_mode", "auto"))
    parser.add_argument("--hvg-flavor", choices=["seurat", "cell_ranger", "seurat_v3"],
                        default=d.get("hvg_flavor", "seurat"))

    # organism-specific QC tags (metrics/plots only; no filtering applied)
    parser.add_argument("--mito-prefix", type=str, default=d.get("mito_prefix", "mt:"))
    parser.add_argument("--ribo-prefix", type=str,
                        default=",".join(d.get("ribo_prefix", ["RpL", "RpS"])),
                        help="Comma-separated ribosomal gene prefixes.")
    parser.add_argument("--hb-regex", type=str, default=d.get("hb_regex", ""),
                        help="Hemoglobin regex; empty disables (default for fly).")

    # embedding / clustering
    parser.add_argument("--n-pcs", type=int, default=int(d.get("n_pcs", 30)))
    parser.add_argument("--n-neighbors", type=int, default=int(d.get("n_neighbors", 30)))
    parser.add_argument("--n-hvgs", type=int, default=int(d.get("n_hvgs", 3000)))
    parser.add_argument("--neighbor-metric", type=str, default=d.get("neighbor_metric", "cosine"))
    parser.add_argument("--umap-min-dist", type=float, default=float(d.get("umap_min_dist", 0.15)))
    parser.add_argument("--harmony-theta", type=float, default=float(d.get("harmony_theta", 2.0)))
    parser.add_argument("--annotation-cluster-key", type=str,
                        default=d.get("annotation_cluster_key", "leiden_res0_5"))

    # z-scaling before PCA (standard scanpy practice; the AFCA authors AND the
    # ROSMAP Stage-3 method both skip it). Default ON via pipeline.yaml; --no-scale
    # reproduces the previous (unscaled) behavior.
    parser.add_argument("--scale", dest="scale", action="store_true",
                        default=bool(d.get("scale", True)),
                        help="z-score HVGs (sc.pp.scale) before PCA.")
    parser.add_argument("--no-scale", dest="scale", action="store_false",
                        help="Disable z-scaling (PCA on unscaled log-norm).")
    parser.add_argument("--scale-max-value", type=float,
                        default=float(d.get("scale_max_value", 10.0)),
                        help="Clip value for sc.pp.scale (default 10).")

    # adapted ROSMAP Stage-1 cell QC: percentile filter on the PRECOMPUTED obs QC
    # metrics (no raw counts needed). Default OFF — AFCA is a pre-QC'd atlas.
    parser.add_argument("--qc-filter", dest="qc_filter", action="store_true",
                        default=bool(d.get("qc_filter", False)),
                        help="Drop cells by percentile on existing obs QC metrics.")
    parser.add_argument("--no-qc-filter", dest="qc_filter", action="store_false",
                        help="Disable the adapted Stage-1 cell QC filter.")
    parser.add_argument("--qc-counts-low-pct", type=float,
                        default=float(d.get("qc_counts_low_pct", 4.5)))
    parser.add_argument("--qc-counts-high-pct", type=float,
                        default=float(d.get("qc_counts_high_pct", 96.0)))
    parser.add_argument("--qc-genes-low-pct", type=float,
                        default=float(d.get("qc_genes_low_pct", 5.0)))
    parser.add_argument("--qc-mt-pct", type=float,
                        default=float(d.get("qc_mt_pct", 10.0)),
                        help="Drop cells with pct_counts_mt above this (0 disables).")

    # quantitative cluster-quality evaluation (ported from ROSMAP
    # 03b_evaluate_correction.py). Writes cluster_quality_metrics.csv.
    parser.add_argument("--eval", dest="eval", action="store_true",
                        default=bool(d.get("eval", True)),
                        help="Compute cluster-quality metrics after clustering.")
    parser.add_argument("--no-eval", dest="eval", action="store_false",
                        help="Skip the cluster-quality evaluation.")
    parser.add_argument("--eval-subsample", type=int,
                        default=int(d.get("eval_subsample", 50000)),
                        help="Subsample N cells for silhouette/LISI (0 = all).")

    # batch / harmony — variant supplies the default; CLI overrides
    parser.add_argument("--harmony-batch-key", type=str, default=None,
                        help="obs column for Harmony. Overrides the variant's key.")
    parser.add_argument("--skip-harmony", action="store_true",
                        help="Skip Harmony; use PCA embedding directly.")

    # annotation
    parser.add_argument("--atlas-label-col", type=str, default=d.get("atlas_label_col", "afca_annotation"),
                        help="obs column with the atlas's authoritative cell-type labels.")
    parser.add_argument("--markers-rds", type=Path,
                        default=Path(os.environ.get("FLY_MARKERS_RDS", "")) if os.environ.get("FLY_MARKERS_RDS") else None,
                        help="Optional fly marker RDS for the ORA cross-check overlay.")

    # operational
    parser.add_argument("--subsample", type=int, default=0,
                        help="Randomly subsample to N cells for a quick smoke run (0 = all).")
    parser.add_argument("--subset-obs", type=str, default=None,
                        help="Restrict to cells matching an obs filter 'COLUMN=VALUE' "
                             "(e.g. 'tissue=head') BEFORE HVG/PCA, so the run re-embeds "
                             "only that subset. Applied after load, before subsample.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


# ----------------------------------------------------------------------------
# resolve_counts — the crux of pre-normalized handling
# ----------------------------------------------------------------------------
def _looks_like_counts(X) -> bool:
    """True if a matrix looks like raw integer counts (non-neg near-integers, max>>1)."""
    if sp.issparse(X):
        d = X.data
    else:
        d = np.asarray(X).ravel()
    d = d[np.isfinite(d)]
    if d.size == 0:
        return False
    nz = d[d != 0]
    if nz.size == 0:
        return False
    near_int = np.all(np.abs(nz - np.round(nz)) < 1e-6)
    return bool(near_int and nz.min() >= 0 and nz.max() > 1.0)


def _looks_like_log1p(X, target_sum: float = 10000.0) -> bool:
    """True if expm1(X) row-sums are ~constant (i.e. X = log1p(normalize_total))."""
    sub = X[:200] if X.shape[0] > 200 else X
    arr = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
    rs = np.expm1(arr).sum(axis=1)
    rs = rs[np.isfinite(rs)]
    if rs.size == 0:
        return False
    # constant to within 1% => normalize_total target reached before log1p
    return bool(rs.std() / (rs.mean() + 1e-9) < 0.01)


def resolve_counts(adata: ad.AnnData, mode: str, target_sum: float) -> dict:
    """
    Decide how to treat the expression matrix and record the decision in
    adata.uns["fly_input_mode"]. Priority:
      1. an explicit raw-counts store (.layers['counts'], .raw.X) if present;
      2. .X itself if it looks like counts;
      3. otherwise treat .X as already log-normalized (the AFCA reality).
    Returns a dict describing the resolution.
    """
    info = {"requested_mode": mode, "counts_source": None, "is_log1p": None,
            "renormalize": None, "hvg_uses_counts": None}

    if mode == "counts":
        info.update(counts_source="X(forced)", renormalize=True, hvg_uses_counts=True, is_log1p=False)
    elif mode == "normalized":
        info.update(counts_source=None, renormalize=False, hvg_uses_counts=False,
                    is_log1p=_looks_like_log1p(adata.X, target_sum))
    else:  # auto
        # 1. explicit raw layer / .raw
        raw_layer = None
        for cand in ("counts", "raw_counts", "UMIs", "spliced"):
            if cand in adata.layers and _looks_like_counts(adata.layers[cand]):
                raw_layer = cand
                break
        if raw_layer is not None:
            info.update(counts_source=f"layers['{raw_layer}']", renormalize=True,
                        hvg_uses_counts=True, is_log1p=False)
        elif adata.raw is not None and _looks_like_counts(adata.raw.X):
            info.update(counts_source="raw.X", renormalize=True, hvg_uses_counts=True, is_log1p=False)
        elif _looks_like_counts(adata.X):
            info.update(counts_source="X", renormalize=True, hvg_uses_counts=True, is_log1p=False)
        else:
            info.update(counts_source=None, renormalize=False, hvg_uses_counts=False,
                        is_log1p=_looks_like_log1p(adata.X, target_sum))

    adata.uns["fly_input_mode"] = info
    print(f"[resolve_counts] {info}")
    if info["counts_source"] is None:
        print("[resolve_counts] -> NORMALIZED-INPUT MODE: skipping normalize_total/log1p; "
              f"HVG via flavor='seurat' on log-data (is_log1p={info['is_log1p']}).")
    return info


# ----------------------------------------------------------------------------
# QC metrics (fly-aware, metrics only — no cell-dropping)
# ----------------------------------------------------------------------------
def ensure_qc_metrics(adata: ad.AnnData, mito_prefix: str, ribo_prefixes: list[str], hb_regex: str) -> None:
    needed = {"pct_counts_mt", "log1p_total_counts", "log1p_n_genes_by_counts"}
    if needed.issubset(set(adata.obs.columns)):
        print("[qc] metrics already present in obs — keeping atlas values.")
        return
    adata.var["mt"] = adata.var_names.str.startswith(mito_prefix)
    adata.var["ribo"] = adata.var_names.str.startswith(tuple(ribo_prefixes)) if ribo_prefixes else False
    qc_vars = ["mt", "ribo"]
    if hb_regex:
        adata.var["hb"] = adata.var_names.str.contains(hb_regex, regex=True)
        qc_vars.append("hb")
    sc.pp.calculate_qc_metrics(adata, qc_vars=qc_vars, inplace=True, percent_top=[20], log1p=True)
    print(f"[qc] computed metrics (mito='{mito_prefix}', ribo={ribo_prefixes}, "
          f"n_mt={int(adata.var['mt'].sum())} genes)")


def qc_filter_cells(adata: ad.AnnData, counts_low_pct: float, counts_high_pct: float,
                    genes_low_pct: float, mt_pct: float) -> ad.AnnData:
    """Adapted ROSMAP Stage-1 percentile QC, applied to the PRECOMPUTED obs QC
    metrics (no raw counts required). Mirrors 01_qc_filter.py's percentile
    outlier rule + a hard pct_counts_mt cap. Returns the filtered AnnData.

    The AFCA atlas is already QC'd upstream, so this is OFF by default; it exists
    so the user's Stage-1 thresholds can be applied on the fly metrics for parity.
    """
    n0 = adata.n_obs

    def _pct_outlier(col: str, low: float | None, high: float | None) -> np.ndarray:
        if col not in adata.obs.columns:
            print(f"[qc-filter][WARN] '{col}' not in obs; skipping that criterion.")
            return np.zeros(adata.n_obs, dtype=bool)
        v = adata.obs[col].to_numpy()
        mask = np.zeros(adata.n_obs, dtype=bool)
        if low is not None:
            mask |= v < np.percentile(v, low)
        if high is not None:
            mask |= v > np.percentile(v, high)
        return mask

    outlier = (
        _pct_outlier("log1p_total_counts", counts_low_pct, counts_high_pct)
        | _pct_outlier("log1p_n_genes_by_counts", genes_low_pct, None)
    )
    if mt_pct and mt_pct > 0 and "pct_counts_mt" in adata.obs.columns:
        outlier |= adata.obs["pct_counts_mt"].to_numpy() > mt_pct

    adata = adata[~outlier].copy()
    n_drop = n0 - adata.n_obs
    print(f"[qc-filter] dropped {n_drop:,}/{n0:,} cells "
          f"({100 * n_drop / max(n0, 1):.1f}%) -> {adata.n_obs:,} retained "
          f"(counts {counts_low_pct}/{counts_high_pct} pct, genes {genes_low_pct} pct, "
          f"mt>{mt_pct}%).")
    adata.uns["fly_qc_filter"] = {
        "counts_low_pct": counts_low_pct, "counts_high_pct": counts_high_pct,
        "genes_low_pct": genes_low_pct, "mt_pct": mt_pct,
        "n_before": int(n0), "n_after": int(adata.n_obs), "n_dropped": int(n_drop),
    }
    return adata


# ----------------------------------------------------------------------------
# Optional ORA overlay (non-authoritative; only if a fly marker set is given)
# ----------------------------------------------------------------------------
def run_ora_overlay(adata: ad.AnnData, markers_rds: Path, cluster_key: str, output_dir: Path) -> None:
    try:
        import decoupler as dc
        from rpy2.robjects import r
    except Exception as exc:  # pragma: no cover
        print(f"[ora] decoupler/rpy2 unavailable ({exc}); skipping ORA overlay.")
        return
    try:
        marker_list = r["readRDS"](str(markers_rds))
        records = []
        for ct, genes in zip(list(marker_list.names), list(marker_list)):
            for g in list(genes):
                records.append({"source": str(ct), "target": str(g), "weight": 1.0})
        markers_df = pd.DataFrame(records).drop_duplicates()
        if markers_df.empty:
            print("[ora] marker set empty; skipping.")
            return
        # decoupler API has drifted across versions; guard the call.
        if hasattr(dc, "run_ora"):
            dc.run_ora(adata, markers_df, source="source", target="target", use_raw=False)
            acts = dc.get_acts(adata, obsm_key="ora_estimate")
            ranked = dc.rank_sources_groups(acts, groupby=cluster_key, reference="rest",
                                            method="t-test_overestim_var")
        else:  # decoupler >=2 API
            dc.mt.ora(adata, net=markers_df.rename(columns={"source": "source", "target": "target"}))
            key = next((k for k in adata.obsm if "ora" in k.lower()), None)
            if key is None:
                print("[ora] no ORA estimate produced; skipping.")
                return
            acts = dc.get_acts(adata, obsm_key=key) if hasattr(dc, "get_acts") else None
            ranked = None
        if ranked is not None:
            ranked.to_csv(output_dir / "cluster_annotation_rankings.csv", index=False)
            top1 = ranked.groupby("group").head(1).set_index("group")["names"].to_dict()
            adata.obs["cell_type_ora"] = (
                adata.obs[cluster_key].astype(str).map(top1).fillna("Unassigned").astype("category")
            )
            print("[ora] wrote cluster_annotation_rankings.csv and obs['cell_type_ora'] (overlay only).")
    except Exception as exc:
        print(f"[ora] overlay failed ({exc}); continuing without it (non-critical).")


# ----------------------------------------------------------------------------
# Figures (fly-keyed fork of the human save_figures)
# ----------------------------------------------------------------------------
def save_figures(adata: ad.AnnData, output_dir: Path, cluster_key: str, batch_key: str | None) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    sc.settings.figdir = str(figures_dir)

    # PCA elbow
    if "pca" in adata.uns and "variance_ratio" in adata.uns["pca"]:
        vr = adata.uns["pca"]["variance_ratio"]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(range(1, len(vr) + 1), vr, "o-", markersize=3)
        ax.set_xlabel("Principal Component"); ax.set_ylabel("Variance Ratio")
        ax.set_title("PCA Elbow Plot")
        fig.savefig(figures_dir / "pca_elbow.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # Pre/post Harmony comparison
    color_key = batch_key if (batch_key and batch_key in adata.obs.columns) else None
    if "X_umap_pca" in adata.obsm and color_key is not None:
        vals = adata.obs[color_key].astype(str)
        uniq = sorted(vals.unique())
        cmap = plt.cm.get_cmap("tab20", max(len(uniq), 1))
        cmap_d = {b: cmap(i) for i, b in enumerate(uniq)}
        colors = [cmap_d[b] for b in vals]
        post_label = "No Correction (PCA)" if batch_key is None else f"After Harmony ({batch_key})"
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6))
        a1.scatter(adata.obsm["X_umap_pca"][:, 0], adata.obsm["X_umap_pca"][:, 1],
                   c=colors, s=1, alpha=0.3, rasterized=True)
        a1.set_title("Before Harmony (PCA)"); a1.set_aspect("equal")
        a2.scatter(adata.obsm["X_umap"][:, 0], adata.obsm["X_umap"][:, 1],
                   c=colors, s=1, alpha=0.3, rasterized=True)
        a2.set_title(post_label); a2.set_aspect("equal")
        fig.suptitle(f"Batch Integration Comparison (colored by {color_key})")
        fig.tight_layout()
        fig.savefig(figures_dir / "harmony_comparison.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # QC metrics on UMAP
    qc_cols = [c for c in ["total_counts", "n_genes_by_counts", "pct_counts_mt"] if c in adata.obs.columns]
    if qc_cols:
        sc.pl.umap(adata, color=qc_cols, show=False, save="_qc_metrics.png")

    # Multi-resolution clustering
    res_keys = [k for k in ["leiden_res0_2", "leiden_res0_5", "leiden_res1"] if k in adata.obs.columns]
    if res_keys:
        sc.pl.umap(adata, color=res_keys, show=False, save="_multi_resolution.png")

    # Integration + biology (cluster, atlas cell_type, and fly covariates)
    bio_keys = [k for k in [cluster_key, "cell_type", "tissue", "sex", "age"] if k in adata.obs.columns]
    if bio_keys:
        sc.pl.umap(adata, color=bio_keys, show=False, save="_fly_integration.png")
    if "cell_type" in adata.obs.columns:
        sc.pl.umap(adata, color=["cell_type"], legend_loc="on data",
                   legend_fontsize=4, show=False, save="_fly_celltypes.png")

    # Cluster x covariate composition
    comp_key = color_key or ("tissue" if "tissue" in adata.obs.columns else None)
    if comp_key and cluster_key in adata.obs.columns:
        ct = pd.crosstab(adata.obs[cluster_key], adata.obs[comp_key], normalize="index")
        fig, ax = plt.subplots(figsize=(max(8, len(ct) * 0.5), 6))
        ct.plot(kind="bar", stacked=True, ax=ax, width=0.85)
        ax.set_xlabel("Cluster"); ax.set_ylabel("Proportion")
        ax.set_title(f"{comp_key} composition per cluster")
        ax.legend(title=comp_key, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=6)
        fig.savefig(figures_dir / "cluster_composition.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # Cell-type proportions
    if "cell_type" in adata.obs.columns:
        counts = adata.obs["cell_type"].value_counts().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(8, max(4, len(counts) * 0.18)))
        ax.barh(range(len(counts)), counts.values, color="#4292c6")
        ax.set_yticks(range(len(counts))); ax.set_yticklabels(counts.index, fontsize=5)
        ax.set_xlabel("Number of Cells"); ax.set_title("Cell Type Proportions (atlas labels)")
        fig.savefig(figures_dir / "cell_type_proportions.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[figures] saved to {figures_dir}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    variant_id, record = resolve_variant(args.variant)

    # Resolve batch key: CLI > variant.skip_harmony > variant.harmony_batch_key.
    skip_harmony = args.skip_harmony or bool(record.get("skip_harmony", False))
    batch_key = args.harmony_batch_key or record.get("harmony_batch_key")
    if skip_harmony:
        batch_key = None

    # Output dir: CLI override else ${FLY_PROCESSING_OUTPUTS}/03_Integrated/<leaf>.
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        base = Path(os.environ.get("FLY_PROCESSING_OUTPUTS", str(repo_root() / "outputs")))
        leaf = record.get("output_leaf", variant_id)
        output_dir = base / "03_Integrated" / leaf
    output_dir.mkdir(parents=True, exist_ok=True)

    integrated_path = output_dir / "fly_integrated.h5ad"
    annotated_path = output_dir / "fly_annotated.h5ad"
    if annotated_path.exists() and not args.overwrite:
        print(f"[skip] {annotated_path} already exists (use --overwrite).")
        return

    ribo_prefixes = [p for p in args.ribo_prefix.split(",") if p]
    d = stage3_defaults()
    target_sum = float(d.get("target_sum", 10000))

    print(f"[start] variant={variant_id} batch_key={batch_key} skip_harmony={skip_harmony}")
    print(f"[load] {args.input_h5ad}")
    adata = ad.read_h5ad(args.input_h5ad)
    adata.obs_names_make_unique(); adata.var_names_make_unique()
    print(f"[load] {adata.n_obs} cells x {adata.n_vars} genes")

    # --- Optional obs subset (e.g. tissue=head): re-embed only matching cells ---
    if args.subset_obs:
        if "=" not in args.subset_obs:
            raise SystemExit(f"--subset-obs must be 'COLUMN=VALUE', got {args.subset_obs!r}")
        col, val = args.subset_obs.split("=", 1)
        col, val = col.strip(), val.strip()
        if col not in adata.obs.columns:
            raise SystemExit(f"--subset-obs column {col!r} not in obs "
                             f"(have: {list(adata.obs.columns)})")
        mask = adata.obs[col].astype(str).to_numpy() == val
        n_match = int(mask.sum())
        if n_match == 0:
            present = sorted(adata.obs[col].astype(str).unique())[:20]
            raise SystemExit(f"--subset-obs {col}={val!r} matched 0 cells "
                             f"(values present: {present})")
        n_before = adata.n_obs
        adata = adata[mask].copy()
        print(f"[subset] {col}={val}: {n_match:,}/{n_before:,} cells retained "
              f"({100 * n_match / n_before:.1f}%)")

    if args.subsample and args.subsample < adata.n_obs:
        sc.pp.subsample(adata, n_obs=args.subsample, random_state=0)
        print(f"[subsample] -> {adata.n_obs} cells (smoke mode)")

    # --- Validate / fall back the batch key against real obs columns ---
    if (not skip_harmony) and batch_key is not None and batch_key not in adata.obs.columns:
        print(f"[WARN] batch key '{batch_key}' not in obs. Available: {list(adata.obs.columns)}")
        print("[WARN] falling back to --skip-harmony.")
        skip_harmony = True; batch_key = None

    # --- Resolve how to treat the matrix (AFCA = normalized-input mode) ---
    cinfo = resolve_counts(adata, args.input_mode, target_sum)

    # --- QC metrics (fly-aware; metrics only, no filtering) ---
    ensure_qc_metrics(adata, args.mito_prefix, ribo_prefixes, args.hb_regex)

    # --- Optional adapted Stage-1 cell QC filter (off by default) ---
    if args.qc_filter:
        adata = qc_filter_cells(adata, args.qc_counts_low_pct, args.qc_counts_high_pct,
                                args.qc_genes_low_pct, args.qc_mt_pct)
    else:
        print("[qc-filter] skipped (atlas is pre-QC'd; enable with --qc-filter).")

    # --- Trust the atlas labels: afca_annotation -> cell_type ---
    if args.atlas_label_col in adata.obs.columns:
        adata.obs["cell_type"] = adata.obs[args.atlas_label_col].astype(str).astype("category")
        print(f"[annot] obs['cell_type'] set from atlas '{args.atlas_label_col}' "
              f"({adata.obs['cell_type'].nunique()} types).")
    else:
        print(f"[annot][WARN] atlas label col '{args.atlas_label_col}' absent; cell_type not set.")

    # --- Normalization (only if we resolved usable raw counts) ---
    if cinfo["renormalize"]:
        if cinfo["counts_source"] and cinfo["counts_source"].startswith("layers"):
            adata.X = adata.layers[cinfo["counts_source"].split("'")[1]].copy()
        elif cinfo["counts_source"] == "raw.X":
            adata.X = adata.raw.X.copy()
        adata.layers["counts"] = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)
        print("[norm] applied normalize_total + log1p from raw counts.")
    elif cinfo["is_log1p"] is False:
        sc.pp.log1p(adata)
        print("[norm] data was linear-normalized; applied log1p only.")
    else:
        print("[norm] input already log-normalized; no (re)normalization applied.")
    if hasattr(adata.X, "astype"):
        adata.X = adata.X.astype(np.float32)

    # --- HVG selection ---
    flavor = args.hvg_flavor
    if cinfo["hvg_uses_counts"] and "counts" in adata.layers:
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=args.n_hvgs,
                                    layer="counts", batch_key=batch_key)
    else:
        # log-normalized flavors operate on adata.X
        if flavor == "seurat_v3":
            flavor = "seurat"
            print("[hvg] forcing flavor='seurat' (no raw counts available for seurat_v3).")
        sc.pp.highly_variable_genes(adata, flavor=flavor, n_top_genes=args.n_hvgs, batch_key=batch_key)
    print(f"[hvg] flavor={flavor}, n_hvgs={args.n_hvgs}, batch_key={batch_key}, "
          f"selected={int(adata.var['highly_variable'].sum())}")

    # Keep full-gene log-normalized data as .raw for marker dotplots / ORA.
    adata.raw = adata
    adata = adata[:, adata.var["highly_variable"]].copy()
    if "counts" in adata.layers:
        del adata.layers["counts"]

    # --- z-scaling before PCA (standard scanpy clustering recipe) ---
    # Runs on the HVG subset, AFTER .raw is set (so .raw keeps the unscaled
    # full-gene log-norm matrix for dotplots/ORA). zero-centering densifies the
    # HVG block (n_obs x n_hvgs float32); fits comfortably in the SLURM budget.
    if args.scale:
        sc.pp.scale(adata, max_value=args.scale_max_value, zero_center=True)
        print(f"[scale] z-scored {adata.n_vars} HVGs (max_value={args.scale_max_value}); "
              f"PCA will run on scaled data.")
        adata.uns["fly_scale"] = {"applied": True, "max_value": args.scale_max_value}
    else:
        print("[scale] skipped (--no-scale); PCA on unscaled log-norm data.")
        adata.uns["fly_scale"] = {"applied": False, "max_value": None}

    # --- PCA ---
    sc.tl.pca(adata, svd_solver="arpack", n_comps=args.n_pcs, use_highly_variable=True)

    # --- pre-Harmony UMAP for comparison ---
    sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=args.n_neighbors, n_pcs=args.n_pcs,
                    metric=args.neighbor_metric, key_added="pca_neighbors")
    sc.tl.umap(adata, neighbors_key="pca_neighbors", min_dist=args.umap_min_dist, random_state=0)
    adata.obsm["X_umap_pca"] = adata.obsm["X_umap"].copy()
    for k in ("pca_neighbors_distances", "pca_neighbors_connectivities"):
        adata.obsp.pop(k, None)
    adata.uns.pop("pca_neighbors", None)

    # --- Harmony ---
    if not skip_harmony:
        import harmonypy as hm
        print(f"[harmony] batch_key={batch_key}, theta={args.harmony_theta}")
        res = hm.run_harmony(adata.obsm["X_pca"], adata.obs, batch_key, theta=args.harmony_theta)
        # harmonypy returns Z_corr as (n_pcs, n_cells); newer (PyTorch) builds
        # return a torch tensor. Coerce to a numpy (n_cells, n_pcs) array.
        z_corr = res.Z_corr
        if hasattr(z_corr, "detach"):       # torch.Tensor -> numpy
            z_corr = z_corr.detach().cpu().numpy()
        z_corr = np.asarray(z_corr)
        harmony_emb = z_corr.T if z_corr.shape[0] == args.n_pcs else z_corr
        if harmony_emb.shape[0] != adata.n_obs:
            raise RuntimeError(
                f"Harmony embedding shape {harmony_emb.shape} does not match "
                f"n_obs={adata.n_obs}; check harmonypy output orientation."
            )
        adata.obsm["X_harmony"] = harmony_emb
        neighbor_rep = "X_harmony"
    else:
        print("[harmony] SKIPPED — using PCA embedding directly.")
        neighbor_rep = "X_pca"
    adata.uns["harmony_params"] = {
        "batch_key": "SKIPPED" if skip_harmony else batch_key,
        "theta": None if skip_harmony else args.harmony_theta,
        "neighbor_rep": neighbor_rep,
    }

    # --- neighbors + Leiden (3 resolutions) + UMAP ---
    sc.pp.neighbors(adata, use_rep=neighbor_rep, n_neighbors=args.n_neighbors,
                    n_pcs=args.n_pcs, metric=args.neighbor_metric)
    for key, res in [("leiden_res0_2", 0.2), ("leiden_res0_5", 0.5), ("leiden_res1", 1.0)]:
        sc.tl.leiden(adata, key_added=key, resolution=res, flavor="igraph",
                     n_iterations=2, directed=False)
    sc.tl.umap(adata, min_dist=args.umap_min_dist, random_state=0)

    adata.write_h5ad(integrated_path)
    print(f"[write] {integrated_path}")

    # --- Quantitative cluster-quality metrics (ported ROSMAP evaluator) ---
    if args.eval:
        try:
            from evaluate_clustering import evaluate_object
            evaluate_object(adata, output_dir, neighbor_rep=neighbor_rep,
                            cell_type_key=args.atlas_label_col,
                            subsample=args.eval_subsample)
        except Exception as exc:  # never let eval block the pipeline write
            print(f"[eval][WARN] cluster-quality evaluation failed: "
                  f"{type(exc).__name__}: {exc}")
    else:
        print("[eval] skipped (--no-eval).")

    # --- Optional ORA cross-check overlay ---
    if args.markers_rds and Path(args.markers_rds).exists():
        run_ora_overlay(adata, Path(args.markers_rds), args.annotation_cluster_key, output_dir)
    else:
        print("[ora] no fly marker set supplied — relying on atlas labels only.")

    # --- Figures ---
    save_figures(adata, output_dir, args.annotation_cluster_key, batch_key)

    # Sanitize obsm column names ('/' breaks HDF5 write).
    for key in list(adata.obsm.keys()):
        elem = adata.obsm[key]
        if hasattr(elem, "columns") and elem.columns.astype(str).str.contains("/").any():
            elem.columns = elem.columns.astype(str).str.replace("/", "|", regex=False)
            adata.obsm[key] = elem

    adata.write_h5ad(annotated_path)
    print(f"[write] {annotated_path}")
    print(f"[done] variant={variant_id} -> {output_dir}")
    gc.collect()


if __name__ == "__main__":
    main()

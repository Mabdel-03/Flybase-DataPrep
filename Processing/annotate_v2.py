#!/usr/bin/env python
"""
Stage 5 — atlas-anchored, marker-based de novo cell-type annotation (v2).

Goal: converge on a cell-type annotation that is MORE accurate than the published
AFCA labels, by re-deriving cluster->type assignments on the superior embedding
(scale + Harmony + nested per-compartment subclustering) instead of trusting the
atlas labels directly. Ported from the user's ROSMAP method
  ROSMAP_Code/Transcriptomics/Processing/Tsai/Pipeline/03b_specificity_annotation.py
and adapted for the fly data + the installed decoupler 1.9.2 API.

Method (per granularity level):
  1. derive_markers : sc.tl.rank_genes_groups on the ATLAS labels (on full-gene
     .raw) -> a candidate marker net per type.
  2. trim_markers   : keep each type's most SPECIFIC genes (tau specificity +
     argmax-belonging + detection), truncate all types to an equal top-N. This
     removes marker-set-size bias AND purges the atlas's own bad markers.
  3. score          : decoupler 1.9.2 dc.run_aucell (rank-based, size-robust) on
     the marker-gene universe; sc.tl.score_genes fallback if decoupler fails.
  4. assign_labels  : per-cell argmax score -> per-cluster majority vote +
     confidence (top1-top2 gap, cluster majority fraction); flag low-confidence.

Hierarchy: broad labels from the GLOBAL Leiden clusters; fine labels from the
NESTED per-compartment subclusters (compartment-restricted panels). Composed into
obs['cell_type_v2'] (fine where a confident nested call exists, else broad).

Circularity caveat (honest scope): markers come from the atlas labels, so this
cannot prove absolute correctness with no orthogonal ground truth. What it CAN
defend: higher cluster-label coherence on the better embedding, higher assigned-
panel marker specificity, and that every atlas-disagreement is marker-defensible
(the assigned type outscores the atlas type on these cells). Reported in full.

NON-DESTRUCTIVE: writes fly_annotated_v2.h5ad + a report/; never modifies the
Stage-3 fly_annotated.h5ad. Atlas labels carried as obs['cell_type_atlas'].
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
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

sc.settings.verbosity = 1


def log(msg: str) -> None:
    print(msg, flush=True)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s).strip("_").lower()


# ---------------------------------------------------------------------------
# Marker derivation + specificity trimming (ported from 03b, label-based)
# ---------------------------------------------------------------------------
def derive_markers(adata: ad.AnnData, label_key: str, n_genes: int = 200) -> pd.DataFrame:
    """rank_genes_groups on the atlas labels -> long candidate net.

    Runs on adata.raw (full 15,992-gene log-norm) so rare-type markers absent from
    the global HVG set are recovered. Returns columns: source, target, score,
    logfoldchanges, pct_nz_group.
    """
    log(f"  [markers] rank_genes_groups on '{label_key}' (wilcoxon, use_raw=True, "
        f"n_genes={n_genes}) ...")
    sc.tl.rank_genes_groups(adata, groupby=label_key, method="wilcoxon",
                            use_raw=True, n_genes=n_genes, tie_correct=True, pts=True,
                            key_added=f"rgg_{label_key}")
    res = adata.uns[f"rgg_{label_key}"]
    groups = res["names"].dtype.names
    recs = []
    pts = res.get("pts")  # DataFrame genes x groups (detection fraction in group)
    for g in groups:
        names = res["names"][g]
        scores = res["scores"][g]
        lfc = res["logfoldchanges"][g]
        for gene, sc_, lf in zip(names, scores, lfc):
            d = float(pts.loc[gene, g]) if (pts is not None and gene in pts.index) else np.nan
            recs.append({"source": str(g), "target": str(gene), "score": float(sc_),
                         "logfoldchanges": float(lf), "pct_nz_group": d})
    net = pd.DataFrame(recs)
    log(f"  [markers] {net['source'].nunique()} types x ~{n_genes} candidates "
        f"= {len(net)} rows.")
    return net


def type_mean_matrix(adata: ad.AnnData, genes: list[str], label_key: str) -> pd.DataFrame:
    """Mean log-norm expression per atlas type over `genes`, from adata.raw.
    Memory-safe one-shot (indicator @ X)[:, valid] form (ported cluster_mean_matrix,
    grouped by the label instead of a cluster key)."""
    raw = adata.raw
    idx = pd.Index(raw.var_names).get_indexer(genes)
    present = [g for g, i in zip(genes, idx) if i >= 0]
    valid = idx[idx >= 0]

    labels = adata.obs[label_key].astype(str).values
    type_ids, inverse = np.unique(labels, return_inverse=True)
    n_types, n_obs = len(type_ids), adata.n_obs
    counts = np.bincount(inverse, minlength=n_types).astype(np.float64)

    ind = sp.csr_matrix(
        (np.ones(n_obs, dtype=np.float32),
         (inverse.astype(np.int64), np.arange(n_obs, dtype=np.int64))),
        shape=(n_types, n_obs),
    )
    X = raw.X
    if sp.issparse(X):
        X_csr = X if isinstance(X, sp.csr_matrix) else X.tocsr()
        sums = (ind @ X_csr).toarray()[:, valid]
    else:
        sums = (ind @ X)[:, valid]
    means = sums / counts[:, None]
    return pd.DataFrame(means, index=type_ids, columns=present)


def compute_tau(type_means: pd.DataFrame) -> pd.Series:
    """tau specificity per gene across cell types (ported verbatim from 03b).
    tau in [0,1]; 1 = perfectly specific to one type."""
    X = type_means.to_numpy(dtype=float)
    n = X.shape[0]
    col_max = X.max(axis=0)
    tau = np.full(X.shape[1], np.nan)
    nz = col_max > 0
    Xhat = X[:, nz] / col_max[nz]
    tau[nz] = (1.0 - Xhat).sum(axis=0) / (n - 1)
    return pd.Series(tau, index=type_means.columns)


def trim_markers(adata: ad.AnnData, raw_net: pd.DataFrame, label_key: str,
                 top_n: int, tau_min: float, min_detect: float,
                 tau_floor: float, tab_dir: Path, level: str) -> pd.DataFrame:
    """Build an equal-size, high-specificity panel per type (ported 03b.trim_markers,
    type_means built directly from atlas labels). Returns net source/target/weight=tau."""
    log(f"\n=== [{level}] marker specificity trimming (top_n={top_n}, "
        f"tau_min={tau_min}, min_detect={min_detect}) ===")
    cand_genes = sorted(set(raw_net["target"]) & set(adata.raw.var_names))
    type_means = type_mean_matrix(adata, cand_genes, label_key)
    type_means = type_means.reindex(sorted(type_means.index))
    tau = compute_tau(type_means)
    argmax_type = type_means.idxmax(axis=0)

    cand = raw_net[["source", "target", "pct_nz_group"]].drop_duplicates().copy()
    cand = cand.rename(columns={"source": "cell_type", "target": "gene"})
    cand["tau"] = cand["gene"].map(tau)
    cand["argmax_type"] = cand["gene"].map(argmax_type)
    cand["mean_expr_in_type"] = [
        type_means.loc[ct, g] if (ct in type_means.index and g in type_means.columns) else np.nan
        for ct, g in zip(cand["cell_type"], cand["gene"])
    ]
    detected = cand["pct_nz_group"].fillna(0) >= min_detect
    belongs = cand["argmax_type"] == cand["cell_type"]
    spec = cand["tau"] >= tau_min
    cand["keep"] = detected & belongs & spec
    cand["spec_score"] = cand["tau"] * cand["mean_expr_in_type"]

    # tau-fallback for under-filled types (relax tau to tau_floor, keep det+belong)
    counts = cand[cand["keep"]].groupby("cell_type").size()
    underfilled = [t for t in type_means.index if counts.get(t, 0) < top_n]
    if underfilled:
        relax = (cand["cell_type"].isin(underfilled) & detected & belongs
                 & (cand["tau"] >= tau_floor))
        cand.loc[relax, "keep"] = True
        log(f"  tau-fallback (floor={tau_floor}) applied to {len(underfilled)} "
            f"under-filled types.")

    kept = cand[cand["keep"]].sort_values(["cell_type", "spec_score"],
                                          ascending=[True, False])
    kept["rank"] = kept.groupby("cell_type").cumcount() + 1
    trimmed = kept[kept["rank"] <= top_n].copy()

    # per-type panel summary + quality flag
    rows = []
    for t in type_means.index:
        n_kept = int((trimmed["cell_type"] == t).sum())
        flag = ("severe_low_panel" if n_kept < 5 else
                "low_panel_quality" if n_kept < top_n else "ok")
        rows.append({"source": t, "n_kept": n_kept, "panel_quality_flag": flag})
    summary = pd.DataFrame(rows)
    summary.to_csv(tab_dir / f"trimmed_markers_summary_{level}.csv", index=False)
    n_severe = int((summary["panel_quality_flag"] == "severe_low_panel").sum())
    log(f"  panels: {len(summary)} types, {n_severe} severe_low_panel (excluded from "
        f"scoring).")

    trimmed[["cell_type", "gene", "tau", "mean_expr_in_type", "pct_nz_group", "rank"]] \
        .rename(columns={"cell_type": "source", "gene": "target"}) \
        .to_csv(tab_dir / f"trimmed_markers_{level}.csv", index=False)

    # drop severe panels (<5 genes): scoring them is noise
    good = set(summary.loc[summary["panel_quality_flag"] != "severe_low_panel", "source"])
    net = trimmed[trimmed["cell_type"].isin(good)][["cell_type", "gene", "tau"]].rename(
        columns={"cell_type": "source", "gene": "target", "tau": "weight"})
    return net


# ---------------------------------------------------------------------------
# Scoring (decoupler 1.9.2 run_aucell; score_genes fallback)
# ---------------------------------------------------------------------------
def build_score_adata(adata: ad.AnnData, net: pd.DataFrame) -> ad.AnnData:
    """Compact AnnData on the union of all trimmed marker genes (from .raw log-norm).
    AUCell ranks per cell over this shared universe so per-panel scores are
    comparable; restricting to markers bounds memory."""
    raw = adata.raw
    raw_names = pd.Index(raw.var_names)
    universe = [g for g in net["target"].unique() if g in set(raw_names)]
    idx = raw_names.get_indexer(universe)
    idx = idx[idx >= 0]
    sub = raw.X[:, idx]
    sub = sub.tocsr().astype(np.float32) if sp.issparse(sub) else np.asarray(sub, np.float32)
    s = ad.AnnData(X=sub)
    s.obs_names = list(adata.obs_names)
    s.var_names = list(raw_names[idx])
    log(f"  [score] universe = {s.n_vars} marker genes x {s.n_obs:,} cells.")
    return s


def score_panels(adata: ad.AnnData, net: pd.DataFrame, scorer: str,
                 min_n: int = 5) -> pd.DataFrame:
    """Return a cells x types score DataFrame. Tries decoupler 1.9.2 run_aucell,
    falls back to sc.tl.score_genes (per type) on the same compact AnnData."""
    score_ad = build_score_adata(adata, net)
    types = sorted(net["source"].unique())
    if scorer == "aucell":
        try:
            import decoupler as dc
            log("  [score] decoupler.run_aucell ...")
            dc.run_aucell(score_ad, net=net[["source", "target"]],
                          source="source", target="target", min_n=min_n,
                          use_raw=False, seed=0, verbose=True)
            acts = score_ad.obsm["aucell_estimate"]
            df = pd.DataFrame(np.asarray(acts), index=adata.obs_names,
                              columns=list(acts.columns) if hasattr(acts, "columns") else types)
            del score_ad; gc.collect()
            return df
        except Exception as exc:
            log(f"  [score][WARN] run_aucell failed ({type(exc).__name__}: {exc}); "
                f"falling back to score_genes.")
    # score_genes fallback (also the explicit scorer='score_genes' path)
    log("  [score] sc.tl.score_genes per type ...")
    cols = {}
    for t, panel in net.groupby("source"):
        genes = [g for g in panel["target"] if g in set(score_ad.var_names)]
        if len(genes) < min_n:
            continue
        sc.tl.score_genes(score_ad, genes, score_name="__s", use_raw=False,
                          ctrl_size=50, n_bins=25, random_state=0)
        cols[t] = score_ad.obs["__s"].to_numpy().copy()
    df = pd.DataFrame(cols, index=adata.obs_names)
    del score_ad; gc.collect()
    return df


# ---------------------------------------------------------------------------
# Assignment (ported 03b.assign_labels, generalized categories)
# ---------------------------------------------------------------------------
def assign_labels(adata: ad.AnnData, cluster_key: str, scores: pd.DataFrame,
                  level: str, gap_min: float | None, majority_min: float) -> dict:
    log(f"\n=== [{level}] assignment ({cluster_key} -> majority of per-cell argmax) ===")
    cols = list(scores.columns)
    arr = scores.to_numpy(dtype=float)
    order = np.argsort(-arr, axis=1)
    t1, t2 = order[:, 0], order[:, 1]
    top1 = arr[np.arange(arr.shape[0]), t1]
    top2 = arr[np.arange(arr.shape[0]), t2]
    percell = np.array(cols)[t1]
    gap = top1 - top2

    pre = f"cell_type_v2_{level}"
    adata.obs[f"{pre}_percell"] = pd.Categorical(percell, categories=cols)
    adata.obs[f"{pre}_gap"] = gap

    if gap_min is None:
        gdf = pd.DataFrame({"t": percell, "gap": gap})
        gap_min = float(gdf.groupby("t")["gap"].quantile(0.05).median())
    log(f"  gap_min = {gap_min:.5f}")

    clusters = adata.obs[cluster_key].astype(str).values
    pc = pd.Series(percell, index=adata.obs_names)
    maj, maj_frac = {}, {}
    for cl in pd.unique(clusters):
        vc = pc[clusters == cl].value_counts()
        maj[cl] = vc.idxmax(); maj_frac[cl] = float(vc.iloc[0] / vc.sum())
    cell_type = pd.Series(clusters, index=adata.obs_names).map(maj)
    adata.obs[pre] = pd.Categorical(cell_type.values,
                                    categories=sorted(set(maj.values())))
    adata.obs[f"{pre}_cluster_majfrac"] = pd.Series(clusters, index=adata.obs_names).map(maj_frac).values
    low = (gap < gap_min) | (adata.obs[f"{pre}_cluster_majfrac"].to_numpy() < majority_min)
    adata.obs[f"{pre}_low_confidence"] = low
    log(f"  low-confidence: {int(low.sum()):,}/{adata.n_obs:,} "
        f"({100*low.sum()/adata.n_obs:.1f}%)")
    return {"gap_min": gap_min, "cluster_majority": maj, "cluster_majority_frac": maj_frac}


def annotate_level(adata: ad.AnnData, label_key: str, cluster_key: str, level: str,
                   n_genes: int, top_n: int, tau_min: float, min_detect: float,
                   tau_floor: float, majority_min: float, min_type_cells: int,
                   scorer: str, tab_dir: Path,
                   restrict_types: set | None = None) -> tuple[pd.DataFrame, dict]:
    """Run derive -> trim -> score -> assign for one granularity level on `adata`,
    grouping clusters by `cluster_key`. restrict_types limits the atlas types
    considered (used to keep nested compartment scoring within-compartment)."""
    # exclude atlas types too small for stable markers (and out-of-compartment)
    vc = adata.obs[label_key].astype(str).value_counts()
    keep_types = set(vc[vc >= min_type_cells].index)
    if restrict_types is not None:
        keep_types &= restrict_types
    sub = adata[adata.obs[label_key].astype(str).isin(keep_types)]
    excluded = sorted(set(vc.index) - keep_types)
    if excluded:
        pd.DataFrame({"excluded_type": excluded,
                      "n_cells": [int(vc.get(t, 0)) for t in excluded]}) \
            .to_csv(tab_dir / f"excluded_too_small_{level}.csv", index=False)
        log(f"  [{level}] excluded {len(excluded)} types (<{min_type_cells} cells or "
            f"out-of-compartment) from the panel vocabulary.")

    raw_net = derive_markers(sub.copy(), label_key, n_genes=n_genes)
    trimmed = trim_markers(adata, raw_net, label_key, top_n, tau_min, min_detect,
                           tau_floor, tab_dir, level)
    scores = score_panels(adata, trimmed, scorer)
    info = assign_labels(adata, cluster_key, scores, level, None, majority_min)
    return scores, {"trimmed": trimmed, **info}


# ---------------------------------------------------------------------------
# Validation (silhouette reuse + specificity + defensibility)
# ---------------------------------------------------------------------------
def _aucell_mean_by_cluster(scores: pd.DataFrame, clusters: np.ndarray) -> pd.DataFrame:
    s = scores.copy(); s["__cl"] = clusters
    return s.groupby("__cl").mean()


def defensibility(adata: ad.AnnData, scores: pd.DataFrame, cluster_key: str,
                  v2_col: str, atlas_col: str, tab_dir: Path, level: str) -> dict:
    """For clusters whose v2 majority disagrees with the atlas majority, is the
    v2 type's mean score > the atlas type's mean score on those cells? (defensible)"""
    clusters = adata.obs[cluster_key].astype(str).values
    means = _aucell_mean_by_cluster(scores, clusters)
    df = pd.DataFrame({"cluster": clusters,
                       "v2": adata.obs[v2_col].astype(str).values,
                       "atlas": adata.obs[atlas_col].astype(str).values})
    rows = []
    for cl, g in df.groupby("cluster"):
        v2t = g["v2"].value_counts().idxmax()
        att = g["atlas"].value_counts().idxmax()
        if v2t == att:
            continue
        sv = float(means.loc[cl, v2t]) if v2t in means.columns else np.nan
        sa = float(means.loc[cl, att]) if att in means.columns else np.nan
        rows.append({"cluster": cl, "n_cells": int(len(g)), "atlas_type": att,
                     "v2_type": v2t, "score_v2": sv, "score_atlas": sa,
                     "defensible": bool(sv > sa) if np.isfinite(sv) and np.isfinite(sa) else False})
    dd = pd.DataFrame(rows)
    out = {"n_disagree_clusters": int(len(dd)), "n_defensible": 0, "frac_cells_defensible": float("nan")}
    if not dd.empty:
        dd.to_csv(tab_dir / f"disagreements_defensible_{level}.csv", index=False)
        out["n_defensible"] = int(dd["defensible"].sum())
        tot = dd["n_cells"].sum()
        out["frac_cells_defensible"] = float(dd.loc[dd["defensible"], "n_cells"].sum() / tot) if tot else float("nan")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    rr = repo_root()
    ap.add_argument("--input", type=Path,
                    default=rr / "outputs/03_Integrated/sweep/scale_dataset/fly_annotated.h5ad")
    ap.add_argument("--subcluster", action="append", default=[],
                    help="Nested object as 'name=path' (repeatable). e.g. "
                         "cns_neuron=outputs/.../subclusters/cns_neuron/subcluster.h5ad")
    ap.add_argument("--broad-col", default="afca_annotation_broad")
    ap.add_argument("--fine-col", default="afca_annotation")
    ap.add_argument("--broad-cluster-key", default="leiden_res0_2")
    ap.add_argument("--fine-cluster-key", default="leiden_sub_res2")
    ap.add_argument("--broad-top-n", type=int, default=50)
    ap.add_argument("--fine-top-n", type=int, default=30)
    ap.add_argument("--broad-tau-min", type=float, default=0.7)
    ap.add_argument("--fine-tau-min", type=float, default=0.85)
    ap.add_argument("--tau-floor", type=float, default=0.5)
    ap.add_argument("--min-detect", type=float, default=0.10)
    ap.add_argument("--min-type-cells", type=int, default=30)
    ap.add_argument("--broad-majority-min", type=float, default=0.5)
    ap.add_argument("--fine-majority-min", type=float, default=0.4)
    ap.add_argument("--n-genes", type=int, default=200)
    ap.add_argument("--scorer", choices=["aucell", "score_genes"], default="aucell")
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--trim-only", action="store_true",
                    help="Run marker derivation + trimming only, then exit.")
    ap.add_argument("--subsample", type=int, default=0, help="Smoke: subsample N cells.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_dir = args.output_dir or (args.input.parent / "annotate_v2")
    tab_dir = out_dir / "tables"; fig_dir = out_dir / "report"
    tab_dir.mkdir(parents=True, exist_ok=True); fig_dir.mkdir(parents=True, exist_ok=True)
    out_h5ad = out_dir / "fly_annotated_v2.h5ad"
    if out_h5ad.exists() and not args.overwrite and not args.trim_only:
        log(f"[skip] {out_h5ad} exists (use --overwrite)."); return 0

    log(f"[load] {args.input}")
    adata = ad.read_h5ad(args.input)
    if args.subsample and args.subsample < adata.n_obs:
        sc.pp.subsample(adata, n_obs=args.subsample, random_state=0)
        log(f"[subsample] -> {adata.n_obs} cells")
    if adata.raw is None:
        log("ERROR: input has no .raw (full-gene matrix needed for markers)."); return 1
    adata.obs["cell_type_atlas"] = adata.obs[args.fine_col].astype(str).values

    # ---- broad level on the global clusters ----
    broad_scores, broad_info = annotate_level(
        adata, args.broad_col, args.broad_cluster_key, "broad",
        args.n_genes, args.broad_top_n, args.broad_tau_min, args.min_detect,
        args.tau_floor, args.broad_majority_min, args.min_type_cells, args.scorer, tab_dir)
    if args.trim_only:
        log("[trim-only] broad panel built; exiting."); return 0

    # confusion + defensibility (broad)
    conf = pd.crosstab(adata.obs["afca_annotation_broad"].astype(str),
                       adata.obs["cell_type_v2_broad"].astype(str))
    conf.to_csv(tab_dir / "confusion_atlas_vs_v2_broad.csv")
    dfb = defensibility(adata, broad_scores, args.broad_cluster_key,
                        "cell_type_v2_broad", "afca_annotation_broad", tab_dir, "broad")
    log(f"  [broad] disagreeing clusters: {dfb['n_disagree_clusters']}, "
        f"defensible: {dfb['n_defensible']}")

    # ---- fine level on each nested subcluster object ----
    nested_meta = {}
    # Pre-initialize the global fine columns with write-safe defaults so cells
    # outside any nested compartment don't become NaN in an object column (which
    # the h5ad writer rejects). "" = no fine call; True = treat as low-confidence.
    if args.subcluster:
        adata.obs["cell_type_v2_fine"] = ""
        adata.obs["cell_type_v2_fine_low_confidence"] = True
    for spec in args.subcluster:
        name, _, path = spec.partition("=")
        npath = Path(path)
        if not npath.exists():
            log(f"[nested][WARN] {name}: {npath} not found; skipping."); continue
        log(f"\n[nested] {name}: {npath}")
        nad = ad.read_h5ad(npath)
        if nad.raw is None:
            log(f"[nested][WARN] {name} has no .raw; skipping."); continue
        # restrict fine vocabulary to types whose atlas-majority broad == this compartment
        comp = name.replace("_", " ")
        broad_of_fine = (adata.obs.groupby(args.fine_col, observed=True)["afca_annotation_broad"]
                         .agg(lambda s: s.astype(str).value_counts().idxmax()))
        comp_match = [c for c in adata.obs["afca_annotation_broad"].astype(str).unique()
                      if _slug(c) == name]
        comp_label = comp_match[0] if comp_match else comp
        restrict = set(broad_of_fine[broad_of_fine == comp_label].index.astype(str))
        log(f"  restricting fine panels to {len(restrict)} types of '{comp_label}'.")
        nad.obs["cell_type_atlas"] = nad.obs[args.fine_col].astype(str).values
        fine_scores, fine_info = annotate_level(
            nad, args.fine_col, args.fine_cluster_key, "fine",
            args.n_genes, args.fine_top_n, args.fine_tau_min, args.min_detect,
            args.tau_floor, args.fine_majority_min, args.min_type_cells, args.scorer,
            tab_dir, restrict_types=restrict)
        conf_f = pd.crosstab(nad.obs[args.fine_col].astype(str),
                             nad.obs["cell_type_v2_fine"].astype(str))
        conf_f.to_csv(tab_dir / f"confusion_atlas_vs_v2_fine_{name}.csv")
        dff = defensibility(nad, fine_scores, args.fine_cluster_key,
                            "cell_type_v2_fine", args.fine_col, tab_dir, f"fine_{name}")
        try:
            ad.settings.allow_write_nullable_strings = True
        except Exception:
            pass
        nsub = out_dir / name; nsub.mkdir(exist_ok=True)
        _sanitize_obsm(nad)
        nad.write_h5ad(nsub / "subcluster_v2.h5ad")
        nested_meta[name] = {"path": str(nsub / "subcluster_v2.h5ad"),
                             "defensibility": dff, "n_cells": int(nad.n_obs)}
        # carry fine labels back onto the global object by obs_names (intersection
        # only — the nested object may be a subset of the global one)
        common = adata.obs_names.intersection(nad.obs_names)
        adata.obs.loc[common, "cell_type_v2_fine"] = \
            nad.obs.loc[common, "cell_type_v2_fine"].astype(str).values
        adata.obs.loc[common, "cell_type_v2_fine_low_confidence"] = \
            nad.obs.loc[common, "cell_type_v2_fine_low_confidence"].astype(bool).values
        del nad; gc.collect()

    # ---- compose hierarchy ----
    # Use the fine call where a confident nested one exists (sentinel "" = none),
    # else fall back to the broad call so the map is always complete.
    broad_lab = adata.obs["cell_type_v2_broad"].astype(str)
    if "cell_type_v2_fine" in adata.obs.columns:
        fine_lab = adata.obs["cell_type_v2_fine"].astype(str)
        fine_low = adata.obs["cell_type_v2_fine_low_confidence"].astype(bool)
        use_fine = (fine_lab != "") & (~fine_low)
        composed = np.where(use_fine.to_numpy(), fine_lab.to_numpy(), broad_lab.to_numpy())
        resolution = np.where(use_fine.to_numpy(), "fine", "broad")
        # normalize the carried fine columns to write-safe dtypes
        adata.obs["cell_type_v2_fine"] = pd.Categorical(fine_lab.values)
        adata.obs["cell_type_v2_fine_low_confidence"] = fine_low.values
    else:
        composed = broad_lab.values; resolution = np.array(["broad"] * adata.n_obs)
    adata.obs["cell_type_v2"] = pd.Categorical(composed)
    adata.obs["cell_type_v2_resolution"] = pd.Categorical(resolution)
    log(f"\n[compose] cell_type_v2: {adata.obs['cell_type_v2'].nunique()} labels "
        f"({(resolution=='fine').sum():,} fine / {(resolution=='broad').sum():,} broad).")

    # ---- validation family A: coherence via the ported evaluator ----
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from evaluate_clustering import evaluate_object
        # evaluate_object writes a fixed-name cluster_quality_metrics.csv, so give
        # each run its OWN subdir, then merge the two into one side-by-side table.
        coh_rows = []
        for ck, lbl in [("cell_type_v2_broad", "v2_broad"), ("afca_annotation_broad", "atlas_broad")]:
            sub = fig_dir / f"coherence_{lbl}"
            df_c = evaluate_object(adata, sub, neighbor_rep="X_harmony", cell_type_key=ck,
                                   leiden_keys=[args.broad_cluster_key], label=lbl,
                                   subsample=50000)
            coh_rows.append(df_c)
        pd.concat(coh_rows, ignore_index=True).to_csv(
            tab_dir / "coherence_v2_vs_atlas.csv", index=False)
        log(f"[eval] coherence_v2_vs_atlas.csv written (v2 vs atlas, side by side).")
    except Exception as exc:
        log(f"[eval][WARN] coherence eval failed: {type(exc).__name__}: {exc}")

    # ---- figures: confusion heatmap + v2-vs-atlas UMAP ----
    _confusion_heatmap(conf, fig_dir / "confusion_heatmap_broad.png")
    _umap_compare(adata, ["cell_type_v2_broad", "afca_annotation_broad",
                          "cell_type_v2_broad_low_confidence"],
                  fig_dir / "umap_v2_vs_atlas_broad.png")

    # ---- write ----
    meta = {"input": str(args.input), "subclusters": nested_meta,
            "broad_defensibility": dfb,
            "n_cells": int(adata.n_obs),
            "params": {k: getattr(args, k) for k in
                       ["broad_top_n", "fine_top_n", "broad_tau_min", "fine_tau_min",
                        "min_detect", "min_type_cells", "scorer"]}}
    (tab_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    try:
        ad.settings.allow_write_nullable_strings = True
    except Exception:
        pass
    _sanitize_obsm(adata)
    adata.write_h5ad(out_h5ad)
    log(f"[write] {out_h5ad}")
    log("[done]")
    return 0


def _sanitize_obsm(adata: ad.AnnData) -> None:
    for key, val in list(adata.obsm.items()):
        if isinstance(val, pd.DataFrame) and any("/" in str(c) for c in val.columns):
            val = val.copy(); val.columns = [str(c).replace("/", "|") for c in val.columns]
            adata.obsm[key] = val


def _confusion_heatmap(conf: pd.DataFrame, out: Path) -> None:
    cn = conf.div(conf.sum(axis=1).replace(0, 1), axis=0)
    fig, ax = plt.subplots(figsize=(max(7, len(cn.columns) * 0.5), max(6, len(cn) * 0.4)))
    im = ax.imshow(cn.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(cn.columns))); ax.set_xticklabels(cn.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(cn.index))); ax.set_yticklabels(cn.index, fontsize=6)
    ax.set_xlabel("cell_type_v2"); ax.set_ylabel("atlas"); ax.set_title("Atlas vs v2 (row-norm)")
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)


def _umap_compare(adata: ad.AnnData, colors: list[str], out: Path) -> None:
    cols = [c for c in colors if c in adata.obs.columns]
    if not cols or "X_umap" not in adata.obsm:
        return
    sc.pl.umap(adata, color=cols, show=False, ncols=len(cols), size=2,
               legend_fontsize=4, frameon=False, save=None)
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close("all")


if __name__ == "__main__":
    raise SystemExit(main())

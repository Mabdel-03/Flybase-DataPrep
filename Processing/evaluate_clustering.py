#!/usr/bin/env python
"""
Quantitative cluster-quality evaluation for the Flybase integrated objects.

Ported from the ROSMAP pipeline's
  ROSMAP_Code/Transcriptomics/Processing/Tsai/Pipeline/03b_evaluate_correction.py
and adapted for the AFCA fly data:

  * cell-type silhouette (ASW) on the corrected embedding   — higher = better separation
  * batch silhouette (ASW)                                  — closer to 0 = better mixing
  * per-cluster batch entropy                               — higher = better mixing
  * ARI / NMI of each Leiden resolution vs the atlas labels — higher = clusters reproduce
    the peer-reviewed cell types (afca_annotation / afca_annotation_broad)
  * iLISI / cLISI via scib if installed (optional; skipped if absent)

Defaults target fly covariates: cell type = afca_annotation, batch auto-detected
from uns['harmony_params'] (-> sex / dataset / 'none').

Usable two ways:
  1. imported by integrate_annotate.py (evaluate_object on the in-memory AnnData), or
  2. standalone:  python "0 - Data Prep/Processing/evaluate_clustering.py" --input <fly_annotated.h5ad>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

# Leiden keys produced by integrate_annotate.py, and the atlas label columns we
# score unsupervised clusters against.
LEIDEN_KEYS = ["leiden_res0_2", "leiden_res0_5", "leiden_res1"]
ATLAS_LABEL_COLS = ["afca_annotation", "afca_annotation_broad"]
# fly batch covariates of interest for mixing metrics
FLY_BATCH_FALLBACKS = ["dataset", "sex", "age", "tissue"]


# ---------------------------------------------------------------------------
# Embedding / batch detection (ported, fly fallbacks)
# ---------------------------------------------------------------------------
def detect_batch_and_embedding(adata: ad.AnnData, batch_key: str | None,
                               embedding_key: str | None) -> tuple[str | None, str]:
    """Auto-detect the batch column and the corrected embedding from
    uns['harmony_params'], falling back to fly covariates / X_pca."""
    hp = adata.uns.get("harmony_params", {}) or {}

    if batch_key is None:
        batch_key = hp.get("batch_key", None)
        if batch_key in ("SKIPPED", None):
            batch_key = None
    if batch_key is not None and batch_key not in adata.obs.columns:
        batch_key = None
    if batch_key is None:
        for fb in FLY_BATCH_FALLBACKS:
            if fb in adata.obs.columns:
                batch_key = fb
                break

    if embedding_key is None:
        embedding_key = hp.get("neighbor_rep", "X_harmony")
    if embedding_key not in adata.obsm:
        embedding_key = "X_pca" if "X_pca" in adata.obsm else next(iter(adata.obsm))

    return batch_key, embedding_key


def subsample_idx(n_obs: int, n: int, seed: int = 42) -> np.ndarray:
    """Indices for an optional subsample (silhouette is O(n^2) memory)."""
    if n <= 0 or n_obs <= n:
        return np.arange(n_obs)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_obs, size=n, replace=False)
    idx.sort()
    return idx


# ---------------------------------------------------------------------------
# Metric primitives (ported from 03b_evaluate_correction.py)
# ---------------------------------------------------------------------------
def _silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import silhouette_score
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return float("nan")
    # silhouette_score's own sample_size keeps it cheap even on the subsample
    return float(silhouette_score(X, labels, metric="euclidean",
                                  sample_size=min(10000, len(labels)), random_state=0))


def _batch_entropy(adata: ad.AnnData, batch_key: str, cluster_key: str) -> float:
    """Mean per-cluster entropy of the batch distribution — higher = better mixing."""
    from scipy.stats import entropy
    ct = pd.crosstab(adata.obs[cluster_key], adata.obs[batch_key])
    proportions = ct.div(ct.sum(axis=1), axis=0)
    return float(proportions.apply(entropy, axis=1).mean())


def _ari_nmi(truth: np.ndarray, clusters: np.ndarray) -> tuple[float, float]:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    return (float(adjusted_rand_score(truth, clusters)),
            float(normalized_mutual_info_score(truth, clusters)))


def _try_lisi(adata: ad.AnnData, batch_key: str, cell_type_key: str,
              embedding_key: str) -> dict:
    """iLISI/cLISI via scib if available; silently skipped otherwise."""
    out: dict[str, float] = {}
    try:
        from scib.metrics import ilisi_graph, clisi_graph
    except ImportError:
        return out
    try:
        out["iLISI"] = float(ilisi_graph(adata, batch_key=batch_key,
                                         type_="embed", use_rep=embedding_key))
        if cell_type_key in adata.obs.columns:
            out["cLISI"] = float(clisi_graph(adata, label_key=cell_type_key,
                                             type_="embed", use_rep=embedding_key))
    except Exception as exc:  # numerical edge cases in scib — non-fatal
        print(f"  [eval][WARN] LISI failed: {type(exc).__name__}: {exc}")
    return out


# ---------------------------------------------------------------------------
# Core entry point (imported by the pipeline AND the CLI)
# ---------------------------------------------------------------------------
def evaluate_object(adata: ad.AnnData, output_dir: Path,
                    neighbor_rep: str | None = None,
                    batch_key: str | None = None,
                    cell_type_key: str = "afca_annotation",
                    subsample: int = 50000,
                    label: str | None = None,
                    leiden_keys: list[str] | None = None) -> pd.DataFrame:
    """Compute cluster-quality metrics and write cluster_quality_metrics.csv.

    Returns the per-Leiden-resolution metrics DataFrame and also stores it in
    adata.uns['cluster_quality_metrics'].

    leiden_keys: cluster columns to score (ARI/NMI vs labels, internal
    silhouette, batch entropy). Defaults to the Stage-3 multi-resolution keys;
    Stage-4 subclustering passes its own (e.g. ['leiden_sub_res2']). If None and
    none of the defaults are present, any obs column starting with 'leiden' is
    used so the per-cluster rows are never silently empty.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label = label or output_dir.name

    batch_key, emb_key = detect_batch_and_embedding(adata, batch_key, neighbor_rep)
    label_cols = [c for c in ATLAS_LABEL_COLS if c in adata.obs.columns]
    if cell_type_key not in label_cols and cell_type_key in adata.obs.columns:
        label_cols = [cell_type_key] + label_cols
    if leiden_keys is None:
        leiden_keys = [k for k in LEIDEN_KEYS if k in adata.obs.columns]
        if not leiden_keys:  # Stage-4 / custom keys: fall back to any leiden* col
            leiden_keys = [c for c in adata.obs.columns if c.startswith("leiden")]
    else:
        leiden_keys = [k for k in leiden_keys if k in adata.obs.columns]

    idx = subsample_idx(adata.n_obs, subsample)
    sub = adata[idx]
    emb = np.asarray(sub.obsm[emb_key])

    print(f"\n[eval] {label}: n={adata.n_obs:,} (silhouette on {len(idx):,}) "
          f"embedding={emb_key} batch={batch_key}")

    rows: list[dict] = []

    # 1. Embedding-level "is it better" metrics, independent of Leiden resolution:
    #    how well does the embedding separate the trusted atlas labels?
    emb_row: dict[str, object] = {"label": label, "scope": "<embedding>",
                                  "embedding": emb_key, "batch_key": batch_key,
                                  "n_cells": int(adata.n_obs)}
    for lab in label_cols:
        emb_row[f"celltype_silhouette_{lab}"] = _silhouette(
            emb, sub.obs[lab].astype(str).to_numpy())
    if batch_key is not None:
        emb_row["batch_silhouette"] = _silhouette(
            emb, sub.obs[batch_key].astype(str).to_numpy())
    emb_row.update(_try_lisi(adata, batch_key, cell_type_key, emb_key)
                   if batch_key is not None else {})
    rows.append(emb_row)

    # 2. Per-Leiden-resolution: ARI/NMI vs atlas labels, internal silhouette,
    #    batch entropy.
    for lk in leiden_keys:
        clusters_full = adata.obs[lk].astype(str).to_numpy()
        clusters_sub = sub.obs[lk].astype(str).to_numpy()
        row: dict[str, object] = {"label": label, "scope": lk,
                                  "embedding": emb_key, "batch_key": batch_key,
                                  "n_clusters": int(len(np.unique(clusters_full))),
                                  "n_cells": int(adata.n_obs)}
        row["leiden_silhouette"] = _silhouette(emb, clusters_sub)
        for lab in label_cols:
            ari, nmi = _ari_nmi(adata.obs[lab].astype(str).to_numpy(), clusters_full)
            row[f"ARI_{lab}"] = ari
            row[f"NMI_{lab}"] = nmi
        if batch_key is not None:
            row["batch_entropy"] = _batch_entropy(adata, batch_key, lk)
        rows.append(row)

    df = pd.DataFrame(rows)
    out = output_dir / "cluster_quality_metrics.csv"
    df.to_csv(out, index=False)
    adata.uns["cluster_quality_metrics"] = df.to_dict(orient="list")
    print(f"[eval] wrote {out}")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Flybase cluster quality.")
    p.add_argument("--input", type=Path, nargs="+", required=True,
                   help="One or more fly_integrated/annotated h5ad files.")
    p.add_argument("--labels", type=str, nargs="*", default=None,
                   help="Labels per input (default: parent dir name).")
    p.add_argument("--batch-key", type=str, default=None,
                   help="obs batch column (default: auto from harmony_params).")
    p.add_argument("--cell-type-key", type=str, default="afca_annotation")
    p.add_argument("--embedding-key", type=str, default=None,
                   help="obsm corrected embedding (default: auto).")
    p.add_argument("--subsample", type=int, default=50000,
                   help="Subsample N cells for silhouette/LISI (0 = all).")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Where to write CSVs (default: each input's parent dir).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    labels = args.labels or [p.parent.name for p in args.input]
    if len(labels) != len(args.input):
        raise SystemExit("[ERROR] number of --labels must match number of --input files")
    for in_path, label in zip(args.input, labels):
        if not in_path.exists():
            print(f"[ERROR] not found: {in_path}")
            continue
        print(f"\nLoading {in_path} ...")
        adata = ad.read_h5ad(in_path)
        out_dir = args.output_dir or in_path.parent
        evaluate_object(adata, out_dir, neighbor_rep=args.embedding_key,
                        batch_key=args.batch_key, cell_type_key=args.cell_type_key,
                        subsample=args.subsample, label=label)
    print("\nDone.")


if __name__ == "__main__":
    main()

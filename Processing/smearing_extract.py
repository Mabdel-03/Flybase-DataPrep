#!/usr/bin/env python
"""
Per-run "smearing" extractor for the AFCA integration sweep.

The user is comparing, toe-to-toe, the *broad cell-type-annotated UMAPs* across
the different Stage-3 processing runs under outputs/03_Integrated, trying to
minimise smearing — i.e. broad cell types that should sit in tight, separated
islands instead bleed across the embedding and overlap each other.

The existing per-run `umap_fly_celltypes.png` figures are each rendered with an
independent palette/scale, so they cannot be compared directly, and the existing
silhouette metrics live in the *high-dimensional* embedding (X_pca / X_harmony),
not in the 2D UMAP the user is actually looking at. This script fixes both:

For one run's fly_annotated.h5ad it reads ONLY
  * obs/afca_annotation_broad  (the 17-class broad annotation)
  * obsm/<embedding>           (default X_umap = that run's recomputed 2D UMAP)
via h5py, so it never touches the multi-GB X / layers / raw matrices. It then

  1. subsamples cells with a FIXED, run-independent seed so every run keeps the
     same number of points and the metrics/figures are comparable, and
  2. computes 2D-UMAP smearing metrics per broad cell type:
       - silhouette_2d        : silhouette of broad labels on the 2D coords
       - knn_purity           : mean fraction of each cell's k nearest UMAP
                                neighbours sharing its broad label (higher=tighter)
       - mean_compactness     : per-type spread (median dist to type centroid),
                                normalised by global embedding scale (lower=tighter)
       - bhattacharyya_overlap: mean pairwise Gaussian overlap between type
                                clouds (lower = less smearing/overlap)
  3. caches the subsampled 2D coords + integer broad labels to an NPZ so the
     combined grid figure can be rendered without re-reading the h5ad.

Outputs (into --outdir, default <run_dir>/smearing):
  smearing_coords.npz     : xy (float32 [n,2]), labels (int16), label_names, meta
  smearing_metrics.csv    : one row global + one row per broad cell type
  smearing_summary.json   : run-level scalar summary (for cross-run aggregation)

Read-only with respect to the data object. Mirrors repo path conventions.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np


# --------------------------------------------------------------------------- #
# Lightweight h5ad readers (metadata only — never materialise X/layers/raw)   #
# --------------------------------------------------------------------------- #
def read_categorical(obs_grp: h5py.Group, col: str):
    """Return (codes int array, categories list[str]) for an AnnData categorical
    obs column stored in the h5ad. Falls back gracefully for plain string/int
    columns written without the categorical {codes, categories} layout."""
    node = obs_grp[col]
    if isinstance(node, h5py.Group) and "codes" in node and "categories" in node:
        codes = node["codes"][:]
        cats = node["categories"][:]
        cats = [c.decode() if isinstance(c, (bytes, bytearray)) else str(c) for c in cats]
        return codes.astype(np.int64), cats
    # plain (non-categorical) column: factorize on the fly
    vals = node[:]
    vals = [v.decode() if isinstance(v, (bytes, bytearray)) else v for v in vals]
    cats = sorted({str(v) for v in vals})
    cat_index = {c: i for i, c in enumerate(cats)}
    codes = np.fromiter((cat_index[str(v)] for v in vals), dtype=np.int64, count=len(vals))
    return codes, cats


def load_run(path: str, ann_col: str, embedding: str):
    """Read just the broad annotation + 2D embedding from a fly_annotated.h5ad."""
    with h5py.File(path, "r") as h:
        if "obsm" not in h or embedding not in h["obsm"]:
            have = list(h["obsm"].keys()) if "obsm" in h else []
            raise KeyError(f"embedding {embedding!r} not in obsm (have: {have})")
        xy = h["obsm"][embedding][:, :2].astype(np.float32)

        obs = h["obs"]
        if ann_col not in obs:
            raise KeyError(f"obs column {ann_col!r} missing (have: {list(obs.keys())})")
        codes, cats = read_categorical(obs, ann_col)
    return xy, codes, cats


# --------------------------------------------------------------------------- #
# Smearing metrics (all on the 2D UMAP coordinates the user actually sees)     #
# --------------------------------------------------------------------------- #
def subsample(n: int, cap: int, seed: int) -> np.ndarray:
    """Deterministic subsample of indices, seed shared across runs so the same
    *count* is used everywhere (cells differ per run only because UMAPs differ)."""
    if n <= cap:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=cap, replace=False))


def knn_label_purity(xy: np.ndarray, labels: np.ndarray, k: int, seed: int,
                     sample: int) -> float:
    """Mean fraction of each query cell's k nearest UMAP neighbours that share
    its broad label. 1.0 = perfectly tight islands, ~1/n_types = fully smeared."""
    from sklearn.neighbors import NearestNeighbors

    n = xy.shape[0]
    nn = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(xy)
    rng = np.random.default_rng(seed + 7)
    q = rng.choice(n, size=min(sample, n), replace=False)
    _, idx = nn.kneighbors(xy[q])
    idx = idx[:, 1:]  # drop self
    same = labels[idx] == labels[q][:, None]
    return float(same.mean())


def per_type_compactness(xy: np.ndarray, labels: np.ndarray, cats: list[str]):
    """Per-type median distance to the type's 2D centroid, normalised by the
    global embedding RMS radius so it is comparable across runs with different
    UMAP scales. Lower = tighter island = less smearing."""
    global_scale = float(np.sqrt(((xy - xy.mean(0)) ** 2).sum(1).mean())) + 1e-9
    rows = {}
    for code, name in enumerate(cats):
        m = labels == code
        if m.sum() < 3:
            continue
        pts = xy[m]
        c = pts.mean(0)
        d = np.sqrt(((pts - c) ** 2).sum(1))
        rows[name] = {
            "n": int(m.sum()),
            "median_dist_centroid": float(np.median(d)),
            "compactness_norm": float(np.median(d) / global_scale),
            "centroid_x": float(c[0]),
            "centroid_y": float(c[1]),
            "spread_x": float(pts[:, 0].std()),
            "spread_y": float(pts[:, 1].std()),
        }
    return rows, global_scale


def bhattacharyya_overlap(xy: np.ndarray, labels: np.ndarray, cats: list[str],
                          min_n: int = 50) -> float:
    """Mean pairwise Bhattacharyya overlap between broad-type point clouds,
    each approximated as a 2D Gaussian. Higher = clouds sit on top of each other
    = more smearing. Averaged over all type pairs with enough cells."""
    means, covs, present = [], [], []
    for code, name in enumerate(cats):
        m = labels == code
        if m.sum() < min_n:
            continue
        pts = xy[m]
        means.append(pts.mean(0))
        covs.append(np.cov(pts.T) + np.eye(2) * 1e-6)
        present.append(name)
    if len(means) < 2:
        return float("nan")
    overlaps = []
    for i in range(len(means)):
        for j in range(i + 1, len(means)):
            S = 0.5 * (covs[i] + covs[j])
            dm = means[i] - means[j]
            try:
                Sinv = np.linalg.inv(S)
                term1 = 0.125 * dm @ Sinv @ dm
                detS = np.linalg.det(S)
                detij = np.sqrt(max(np.linalg.det(covs[i]) * np.linalg.det(covs[j]), 1e-30))
                term2 = 0.5 * np.log(max(detS, 1e-30) / detij)
                bc = np.exp(-(term1 + term2))  # Bhattacharyya coefficient in [0,1]
            except np.linalg.LinAlgError:
                continue
            overlaps.append(float(bc))
    return float(np.mean(overlaps)) if overlaps else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Path to a run's fly_annotated.h5ad")
    ap.add_argument("--label", default=None, help="Run label (default: parent dir name)")
    ap.add_argument("--outdir", default=None,
                    help="Output dir (default: <run_dir>/smearing)")
    ap.add_argument("--ann-col", default="afca_annotation_broad")
    ap.add_argument("--embedding", default="X_umap",
                    help="obsm key = the run's recomputed 2D UMAP (default X_umap)")
    ap.add_argument("--cap", type=int, default=120000,
                    help="Max cells to subsample (shared across runs).")
    ap.add_argument("--seed", type=int, default=0, help="Shared subsample seed.")
    ap.add_argument("--knn-k", type=int, default=30)
    ap.add_argument("--knn-sample", type=int, default=40000)
    ap.add_argument("--sil-sample", type=int, default=20000)
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        return 1
    run_dir = os.path.dirname(os.path.abspath(args.input))
    label = args.label or os.path.basename(run_dir)
    outdir = args.outdir or os.path.join(run_dir, "smearing")
    os.makedirs(outdir, exist_ok=True)

    print(f"[smear:{label}] reading {args.embedding} + {args.ann_col} from {args.input}",
          flush=True)
    xy_full, codes_full, cats = load_run(args.input, args.ann_col, args.embedding)
    n_full = xy_full.shape[0]

    # drop cells with no broad label (code < 0) before subsampling
    valid = codes_full >= 0
    xy_full, codes_full = xy_full[valid], codes_full[valid]
    # guard against non-finite UMAP coords
    finite = np.isfinite(xy_full).all(1)
    xy_full, codes_full = xy_full[finite], codes_full[finite]

    idx = subsample(xy_full.shape[0], args.cap, args.seed)
    xy = np.ascontiguousarray(xy_full[idx])
    labels = codes_full[idx]
    print(f"[smear:{label}] {n_full:,} cells -> {xy.shape[0]:,} subsampled "
          f"({len(cats)} broad categories)", flush=True)

    # ---- metrics ----------------------------------------------------------- #
    from sklearn.metrics import silhouette_score

    rng = np.random.default_rng(args.seed + 13)
    sil_idx = rng.choice(xy.shape[0], size=min(args.sil_sample, xy.shape[0]),
                         replace=False)
    # silhouette needs >=2 labels present in the sample
    sil_2d = float("nan")
    if len(np.unique(labels[sil_idx])) > 1:
        sil_2d = float(silhouette_score(xy[sil_idx], labels[sil_idx], metric="euclidean"))

    purity = knn_label_purity(xy, labels, args.knn_k, args.seed, args.knn_sample)
    comp_rows, global_scale = per_type_compactness(xy, labels, cats)
    overlap = bhattacharyya_overlap(xy, labels, cats)
    comp_vals = [r["compactness_norm"] for r in comp_rows.values()]
    mean_comp = float(np.mean(comp_vals)) if comp_vals else float("nan")
    median_comp = float(np.median(comp_vals)) if comp_vals else float("nan")

    print(f"[smear:{label}] silhouette_2d={sil_2d:.4f}  knn_purity={purity:.4f}  "
          f"mean_compactness={mean_comp:.4f}  overlap={overlap:.4f}", flush=True)

    # ---- cache coords for the combined grid render ------------------------- #
    names_present = sorted(set(cats[c] for c in np.unique(labels)))
    np.savez_compressed(
        os.path.join(outdir, "smearing_coords.npz"),
        xy=xy.astype(np.float32),
        labels=labels.astype(np.int16),
        label_names=np.array(cats, dtype=object),
        label=label,
        embedding=args.embedding,
    )

    # ---- per-type + global metrics CSV ------------------------------------- #
    import csv
    csv_path = os.path.join(outdir, "smearing_metrics.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "scope", "cell_type", "n",
                    "compactness_norm", "median_dist_centroid",
                    "spread_x", "spread_y", "centroid_x", "centroid_y"])
        w.writerow([label, "GLOBAL", "<all>", xy.shape[0],
                    mean_comp, "", "", "", "", ""])
        for name, r in sorted(comp_rows.items(), key=lambda kv: -kv[1]["n"]):
            w.writerow([label, "celltype", name, r["n"],
                        r["compactness_norm"], r["median_dist_centroid"],
                        r["spread_x"], r["spread_y"], r["centroid_x"], r["centroid_y"]])

    summary = {
        "run": label,
        "embedding": args.embedding,
        "ann_col": args.ann_col,
        "n_cells_full": int(n_full),
        "n_cells_used": int(xy.shape[0]),
        "n_broad_types": len(cats),
        "n_broad_types_present": len(names_present),
        "silhouette_2d": sil_2d,
        "knn_purity": purity,
        "mean_compactness_norm": mean_comp,
        "median_compactness_norm": median_comp,
        "gaussian_overlap": overlap,
        "global_scale": global_scale,
    }
    with open(os.path.join(outdir, "smearing_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"[smear:{label}] wrote {outdir}/{{smearing_coords.npz,"
          f"smearing_metrics.csv,smearing_summary.json}}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

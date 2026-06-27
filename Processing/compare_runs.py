#!/usr/bin/env python
"""
Aggregate and compare cluster-quality metrics across Flybase sweep runs.

Ported pattern from the ROSMAP pipeline's
  ROSMAP_Code/Transcriptomics/Processing/DeJager/Pipeline/03c_compare_corrections.py
(comparison table + per-metric bar plots), trimmed for fly: it reads the
cluster_quality_metrics.csv that evaluate_clustering.py writes per run, so it
needs no h5ad reloading and none of ROSMAP's clinical-variable R²/chi-square.

Usage (from the repo root; quote the spaced bucket dir):
  # point at sweep run directories (each containing cluster_quality_metrics.csv)
  python "0 - Data Prep/Processing/compare_runs.py" "0 - Data Prep/outputs/03_Integrated/sweep/"*/
  # or pass explicit CSVs
  python "0 - Data Prep/Processing/compare_runs.py" --csv run_a/cluster_quality_metrics.csv ...
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS_CSV = "cluster_quality_metrics.csv"


def find_csvs(paths: list[str], explicit_csv: list[str] | None) -> list[Path]:
    csvs: list[Path] = []
    for c in explicit_csv or []:
        csvs.append(Path(c))
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            cand = pp / METRICS_CSV
            if cand.exists():
                csvs.append(cand)
        elif pp.suffix == ".csv":
            csvs.append(pp)
        else:  # glob pattern
            for g in glob.glob(p):
                gp = Path(g)
                cand = gp / METRICS_CSV if gp.is_dir() else gp
                if cand.exists():
                    csvs.append(cand)
    # de-dup, keep order
    seen, out = set(), []
    for c in csvs:
        r = c.resolve()
        if r not in seen:
            seen.add(r)
            out.append(c)
    return out


def load_all(csvs: list[Path]) -> pd.DataFrame:
    frames = []
    for c in csvs:
        df = pd.read_csv(c)
        # 'label' is written by evaluate_clustering; fall back to the run dir name
        if "label" not in df.columns or df["label"].isna().all():
            df["label"] = c.parent.name
        df["run"] = c.parent.name
        frames.append(df)
    if not frames:
        raise SystemExit("No cluster_quality_metrics.csv files found.")
    return pd.concat(frames, ignore_index=True)


def best_resolution_per_run(df: pd.DataFrame, target_n: int, label_col: str) -> pd.DataFrame:
    """For each run, pick the Leiden resolution whose cluster count is closest to
    target_n (163 fine / 17 broad), and report that row's ARI for label_col."""
    ari_col = f"ARI_{label_col}"
    leiden = df[df["scope"].str.startswith("leiden", na=False)].copy()
    if leiden.empty or ari_col not in leiden.columns:
        return pd.DataFrame()
    leiden["dist_to_target"] = (leiden["n_clusters"] - target_n).abs()
    idx = leiden.groupby("run")["dist_to_target"].idxmin()
    pick = leiden.loc[idx, ["run", "scope", "n_clusters", ari_col]].reset_index(drop=True)
    pick = pick.rename(columns={"scope": f"res@~{target_n}",
                                "n_clusters": f"n_clusters@~{target_n}",
                                ari_col: f"{ari_col}@~{target_n}"})
    return pick


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare Flybase sweep runs.")
    ap.add_argument("paths", nargs="*", help="Run dirs or globs containing "
                    f"{METRICS_CSV}.")
    ap.add_argument("--csv", nargs="*", default=None, help="Explicit CSV paths.")
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/03_Integrated/sweep"),
                    help="Where to write the comparison table/plots.")
    args = ap.parse_args()

    csvs = find_csvs(args.paths, args.csv)
    if not csvs:
        raise SystemExit("No metrics CSVs found. Pass run dirs, globs, or --csv.")
    print(f"[compare] {len(csvs)} runs:")
    for c in csvs:
        print(f"  - {c}")

    df = load_all(csvs)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Full long table.
    full_out = args.output_dir / "all_runs_metrics.csv"
    df.to_csv(full_out, index=False)
    print(f"[compare] wrote {full_out}")

    # --- Embedding-level summary: the resolution-independent "is it better" view.
    emb = df[df["scope"] == "<embedding>"].copy()
    emb_metric_cols = [c for c in emb.columns
                       if c.startswith("celltype_silhouette_")
                       or c in ("batch_silhouette", "iLISI", "cLISI")]
    summary = emb[["run"] + emb_metric_cols].copy() if not emb.empty else pd.DataFrame()

    # --- Best-resolution ARI vs atlas labels (fine ~163, broad ~17).
    for target_n, lab in [(163, "afca_annotation"), (17, "afca_annotation_broad")]:
        pick = best_resolution_per_run(df, target_n, lab)
        if not pick.empty:
            summary = (pick if summary.empty
                       else summary.merge(pick, on="run", how="outer"))

    if not summary.empty:
        sum_out = args.output_dir / "comparison_summary.csv"
        summary.to_csv(sum_out, index=False)
        print(f"[compare] wrote {sum_out}")
        with pd.option_context("display.max_columns", None, "display.width", 220):
            print("\n=== Comparison summary (one row per run) ===")
            print(summary.to_string(index=False))

        # Bar plot per numeric metric.
        metric_cols = [c for c in summary.columns
                       if c != "run" and pd.api.types.is_numeric_dtype(summary[c])]
        if metric_cols:
            n = len(metric_cols)
            fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
            if n == 1:
                axes = [axes]
            for ax, col in zip(axes, metric_cols):
                vals = summary[col].to_numpy(dtype=float)
                ax.bar(range(len(vals)), vals, color="#4292c6")
                ax.set_xticks(range(len(vals)))
                ax.set_xticklabels(summary["run"].values, rotation=45, ha="right", fontsize=8)
                ax.set_title(col, fontsize=9)
                for i, v in enumerate(vals):
                    if np.isfinite(v):
                        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
            fig.suptitle("Flybase sweep — cluster-quality comparison", fontsize=13)
            fig.tight_layout()
            plot_out = args.output_dir / "comparison_summary.png"
            fig.savefig(plot_out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[compare] wrote {plot_out}")

    print("\nDone.")


if __name__ == "__main__":
    main()

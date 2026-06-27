# Flybase — AFCA Drosophila snRNA-seq processing (Data Prep)

Applies the ROSMAP `Transcriptomics` single-nucleus **Stage-3** method
(integration → batch correction → clustering → annotation) to the
**Aging Fly Cell Atlas (AFCA)** head+body single-nucleus object.

> This is the **`0 - Data Prep`** half of the repo (downloading, integrating,
> annotating, clustering, QC/diagnostics). The CELLxGENE serving platform that
> consumes these outputs lives in the sibling **`1 - Platform`** directory.
> Shared config (`config/paths.sh`, `pipeline.yaml`, `variants.yaml`) stays at the
> repo root and is sourced by both halves.

## What the data is

`0 - Data Prep/data/adata_headBody_S_v1.0.h5ad` — the AFCA stringent ("S") combined
head+body atlas from the Hongjie Li Lab (BCM), fetched from an anonymous SharePoint share.

- **566,254 nuclei × 15,992 genes**, fly gene **symbols** (`128up`, `5-HT1A`, `mt:CoI`, …).
- `obs`: `tissue` (head/body), `sex` (female/male/mix), `age` (5/30/50/70 d),
  `sex_age`, `dataset` (AFCA/FCA), and three annotation columns —
  `afca_annotation` (163 fine types), `afca_annotation_broad` (17), `fca_annotation`.
- **`X` is already log-normalized**: `X = log1p(normalize_total(counts, target_sum=1e4))`
  (verified: `expm1(X).sum(axis=1) == 10000` per cell). There is **no raw-count
  store** (`.raw` is None, `.layers` empty). QC metrics (`pct_counts_mt`, …) are
  precomputed.

## Why this is Stage-3 only

The human pipeline's Stage 1 (per-sample raw-count percentile QC) and Stage 2
(`scDblFinder` doublet removal) assume **per-sample raw CellBender counts**. The
AFCA object is a single, pre-filtered, pre-normalized, already-annotated atlas
with no recoverable counts — so those stages are statistically meaningless here
and are intentionally not run. We run an **adapted Stage 3** that:

- loads the single file (no per-sample concat),
- **auto-detects** that the matrix is already log-normalized and therefore
  **does not re-normalize** (no double `normalize_total`/`log1p`) and uses HVG
  `flavor="seurat"` instead of `seurat_v3` (which needs counts),
- computes **fly-aware** QC metrics (mito `mt:`, ribo `RpL`/`RpS`, no hemoglobin) —
  for plots only, no cell dropping,
- **trusts the atlas labels**: `afca_annotation` → `obs["cell_type"]`; ORA marker
  scoring is an *optional* non-authoritative overlay (off by default),
- runs the same **PCA → Harmony → Leiden (0.2/0.5/1.0) → UMAP** method and figure
  set as the human Stage 3, keyed on the fly covariates.

See `/orcd/data/lhtsai/001/mabdel03/ROSMAP_Code/Transcriptomics/Processing/Tsai/Pipeline/03_integration_annotation.py`
for the source method this is forked from.

## Layout

```
config/                  (repo root, shared) paths.sh + paths.local.sh.template, pipeline.yaml, variants.yaml
0 - Data Prep/
  scripts/               download_afca.sh (idempotent SharePoint fetch)
  Processing/            integrate_annotate.py (adapted Stage 3), run_integrate.sh (sbatch wrapper)
  Resources/             optional fly marker RDS for the ORA overlay
  data/                  the AFCA .h5ad (gitignored)
  outputs/               integrated/annotated objects + figures (gitignored)
  *.py                   ad-hoc diagnostics (run with CWD = "0 - Data Prep")
1 - Platform/            the CELLxGENE serving platform (separate README/spec)
```

## Quickstart

Paths below are relative to the repo root; the spaces in `0 - Data Prep` must be
quoted in the shell.

```bash
cd /orcd/data/lhtsai/001/mabdel03/Flybase
cp config/paths.local.sh.template config/paths.local.sh   # edit conda/SLURM if needed
source config/paths.sh && check_paths

# 1. Download the atlas (idempotent; ~3 GB)
bash "0 - Data Prep/scripts/download_afca.sh"

# 2a. Quick smoke run (interactive, ~minutes, no SLURM)
bash "0 - Data Prep/Processing/run_integrate.sh" no_harmony --subsample 20000

# 2b. Full run (SLURM) — primary variant corrects Harmony on sex
sbatch "0 - Data Prep/Processing/run_integrate.sh"            # == --variant primary (sex)
sbatch "0 - Data Prep/Processing/run_integrate.sh" no_harmony # sensitivity: no batch correction
sbatch "0 - Data Prep/Processing/run_integrate.sh" age        # experiment: Harmony on age
sbatch "0 - Data Prep/Processing/run_integrate.sh" dataset    # experiment: Harmony on AFCA-vs-FCA source
```

## Outputs

Per variant under `0 - Data Prep/outputs/03_Integrated/<variant>/`:
- `fly_integrated.h5ad` — embeddings + Leiden clusters (checkpoint),
- `fly_annotated.h5ad` — final object; `obs["cell_type"]` = atlas labels,
  `uns["fly_input_mode"]` records the counts/normalization decision,
- `figures/` — PCA elbow, pre/post-Harmony UMAP, QC-metric UMAPs, multi-resolution
  clusters, UMAP by cluster/cell_type/tissue/sex/age, composition bars, cell-type
  proportions.

## Variants

`config/variants.yaml`: `sex` (primary), `no_harmony` (sensitivity), `age` and
`dataset` (experiments). Switch the primary by editing the `primary: fly:` line.

## Provenance / sources

- AFCA portal: https://hongjielilab.org/afca/
- Paper: Lu et al., *Science* 2023 — https://www.science.org/doi/10.1126/science.adg0934
- Dataset card: https://huggingface.co/datasets/longevity-db/aging-fly-cell-atlas

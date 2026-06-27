#!/bin/bash
#SBATCH -J fly_tissue
#SBATCH -t 03:30:00
#SBATCH -n 16
#SBATCH --mem=180G
#SBATCH --array=0-1
#SBATCH --mail-type=END,FAIL

# Per-tissue re-embedding as a SLURM JOB ARRAY (durable; one task per tissue).
#
# Re-runs the full pipeline (HVG -> PCA -> Harmony -> UMAP -> Leiden -> annotate)
# INDEPENDENTLY on each tissue subset, so each tissue gets its own embedding that
# reflects only that tissue's structure (vs masking the shared atlas UMAP).
#
# Uses the winning config (scale_dataset = Harmony on `dataset`, z-scaling ON) at
# n_pcs=30 — the least-smeared setting from the n_pcs sweep.
#
# Array index -> tissue:  0=head, 1=body
#
# Submit from the repo root so config/paths.sh resolves REPO_ROOT correctly:
#   cd /orcd/data/lhtsai/001/mabdel03/Flybase
#   sbatch --export=ALL,REPO_ROOT="$(pwd)" -p ou_bcs_high \
#          --output="<FLY_LOGS>/tissue_%A_%a.out" --error="<FLY_LOGS>/tissue_%A_%a.err" \
#          "0 - Data Prep/Processing/sbatch_tissue_umap.sh"

set -euo pipefail

if [[ -n "${REPO_ROOT:-}" && -f "${REPO_ROOT}/config/paths.sh" ]]; then
    :
else
    _SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
fi
source "${REPO_ROOT}/config/paths.sh"

mkdir -p "${FLY_LOGS}"
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1

set +u
source "${CONDA_INIT_SCRIPT}"
conda activate "${BATCHCORR_ENV}"
set -u
if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "ERROR: failed to activate conda env: ${BATCHCORR_ENV}" >&2
    exit 1
fi

TISSUES=(head body)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
TISSUE="${TISSUES[$IDX]}"
NPCS=30

OUT="${FLY_PROCESSING_OUTPUTS}/03_Integrated/by_tissue/${TISSUE}"
mkdir -p "$(dirname "${OUT}")"

echo "[tissue:${TISSUE}] host=$(hostname) job=${SLURM_JOB_ID:-NA} array_task=${IDX} n_pcs=${NPCS}"
echo "[tissue:${TISSUE}] config=scale_dataset (variant=dataset, scale ON) -> ${OUT}"

# Idempotent resume: skip if already finished.
if [[ -f "${OUT}/fly_annotated.h5ad" ]]; then
    echo "[tissue:${TISSUE}] fly_annotated.h5ad already present — skipping (resume)."
    exit 0
fi

python "${REPO_ROOT}/0 - Data Prep/Processing/integrate_annotate.py" \
    --variant dataset \
    --n-pcs "${NPCS}" \
    --subset-obs "tissue=${TISSUE}" \
    --output-dir "${OUT}"

echo "[tissue:${TISSUE}] done -> ${OUT}"

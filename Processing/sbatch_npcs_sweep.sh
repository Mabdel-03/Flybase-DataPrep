#!/bin/bash
#SBATCH -J fly_npcs
#SBATCH -t 03:30:00
#SBATCH -n 16
#SBATCH --mem=180G
#SBATCH --array=0-3
#SBATCH --mail-type=END,FAIL

# n_pcs sweep as a SLURM JOB ARRAY — the durable way to run this.
#
# One array task per n_pcs value (indices 0..3 -> {30,50,75,100}), each an
# independent, resumable job with its own allocation. This is preferable to a
# backgrounded/`setsid` loop on an interactive node, which dies whenever the
# session or node allocation changes (observed repeatedly: node4003<->node4004).
#
# Runs the WINNING integration config (scale_dataset = Harmony on `dataset`,
# z-scaling ON); only --n-pcs varies, so any change in broad-celltype UMAP
# smearing is attributable to PC count. Idempotent: skips a value whose
# fly_annotated.h5ad already exists (so a requeued/preempted task resumes cleanly).
#
# Submit (from the repo root so config/paths.sh resolves REPO_ROOT correctly):
#   cd /orcd/data/lhtsai/001/mabdel03/Flybase
#   sbatch --export=ALL,REPO_ROOT="$(pwd)" \
#          -p ou_bcs_high \
#          --output="<FLY_LOGS>/npcs_%A_%a.out" --error="<FLY_LOGS>/npcs_%A_%a.err" \
#          "0 - Data Prep/Processing/sbatch_npcs_sweep.sh"
# (the companion submit helper resolves the log paths for you.)

set -euo pipefail

# Resolve REPO_ROOT. Under sbatch BASH_SOURCE points into the spool copy, so
# honor a pre-exported REPO_ROOT (set via --export) and fall back for bash use.
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

# n_pcs grid indexed by the array task id.
NPCS_GRID=(30 50 75 100)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
NPCS="${NPCS_GRID[$IDX]}"

SWEEP_BASE="${FLY_PROCESSING_OUTPUTS}/03_Integrated/npcs_sweep"
OUT="${SWEEP_BASE}/pc${NPCS}"
mkdir -p "${SWEEP_BASE}"

echo "[npcs:${NPCS}] host=$(hostname) job=${SLURM_JOB_ID:-NA} array_task=${IDX}"
echo "[npcs:${NPCS}] config=scale_dataset (variant=dataset, scale ON) -> ${OUT}"

# Idempotent resume: skip if this value already finished.
if [[ -f "${OUT}/fly_annotated.h5ad" ]]; then
    echo "[npcs:${NPCS}] fly_annotated.h5ad already present — skipping (resume)."
    exit 0
fi

python "${REPO_ROOT}/0 - Data Prep/Processing/integrate_annotate.py" \
    --variant dataset \
    --n-pcs "${NPCS}" \
    --output-dir "${OUT}"

echo "[npcs:${NPCS}] done -> ${OUT}"

#!/bin/bash
#SBATCH -J fly_npcs_sweep
#SBATCH -t 08:00:00
#SBATCH -n 16
#SBATCH --mem=200G
#SBATCH --mail-type=END,FAIL

# n_pcs sweep on the WINNING integration config (scale_dataset = Harmony on
# `dataset`, z-scaling ON). The only thing that varies between runs is --n-pcs,
# so any change in broad-celltype UMAP smearing is attributable to PC count.
#
# Motivation: config/pipeline.yaml sets n_pcs=50 ("bumped 30->50 for the
# 566k-cell / 163-type atlas") but the PCA elbow is still descending at PC50
# (cumulative variance only ~23% for scaled runs), so 50 was never validated.
# This sweeps n_pcs in {30,50,75,100} and feeds each into the smearing harness.
#
# Usage:
#   sbatch "0 - Data Prep/Processing/run_npcs_sweep.sh"
#   bash   "0 - Data Prep/Processing/run_npcs_sweep.sh"            # interactive (inside an alloc)
#   bash   "0 - Data Prep/Processing/run_npcs_sweep.sh" 30 50 75   # custom n_pcs list

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

# n_pcs values: CLI args override the default list.
if [[ $# -ge 1 ]]; then
    NPCS_LIST=("$@")
else
    NPCS_LIST=(30 50 75 100)
fi

SWEEP_BASE="${FLY_PROCESSING_OUTPUTS}/03_Integrated/npcs_sweep"
mkdir -p "${SWEEP_BASE}"
echo "[npcs_sweep] config=scale_dataset (variant=dataset, scale ON), n_pcs=${NPCS_LIST[*]}"
echo "[npcs_sweep] output base=${SWEEP_BASE}"

for NPCS in "${NPCS_LIST[@]}"; do
    OUT="${SWEEP_BASE}/pc${NPCS}"
    echo ""
    echo "==================== n_pcs=${NPCS} -> ${OUT} ===================="
    # winning config, only --n-pcs changes. scale is ON by pipeline.yaml default.
    python "${REPO_ROOT}/0 - Data Prep/Processing/integrate_annotate.py" \
        --variant dataset \
        --n-pcs "${NPCS}" \
        --output-dir "${OUT}"
    echo "[npcs_sweep] done n_pcs=${NPCS}"
done

echo ""
echo "[npcs_sweep] ALL integration runs complete. Now extract smearing + compare:"
echo "  for d in \"${SWEEP_BASE}\"/pc*/; do python \"${REPO_ROOT}/0 - Data Prep/Processing/smearing_extract.py\" --input \"\$d/fly_annotated.h5ad\" --cap 120000; done"

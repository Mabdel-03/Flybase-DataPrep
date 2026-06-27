#!/bin/bash
#SBATCH -J fly_subcl
#SBATCH -t 04:00:00
#SBATCH -n 16
#SBATCH --mem=200G
#SBATCH --array=0-4
#SBATCH --mail-type=END,FAIL

# Stage-4 per-compartment subclustering as a SLURM JOB ARRAY (durable).
#
# Part (2) of the per-tissue/subtype request: for each big multi-subtype broad
# compartment, subset it and re-derive its OWN HVGs/PCA/Harmony/Leiden/UMAP
# (subcluster_compartment.py), then the UMAP is coloured by the FINE label
# (afca_annotation) so sub-types resolve within the compartment.
#
# Array index -> compartment (the 5 big multi-subtype broad types):
#   0 = CNS neuron      (74 fine types, 194k cells)
#   1 = epithelial cell (24 fine, 89k)
#   2 = glial cell      (18 fine, 37k)
#   3 = sensory neuron  (14 fine, 59k)
#   4 = muscle cell     (10 fine, 86k)
#
# n_pcs here keeps subcluster_compartment.py's tuned default (50) — the global
# n_pcs=30 finding is about the WHOLE-atlas broad UMAP; a single compartment's
# fine structure is a separate basis and may want more PCs. Sweep separately if
# needed.
#
# Submit from repo root:
#   cd /orcd/data/lhtsai/001/mabdel03/Flybase
#   sbatch --export=ALL,REPO_ROOT="$(pwd)" -p ou_bcs_high \
#          --output="<FLY_LOGS>/subcl_%A_%a.out" --error="<FLY_LOGS>/subcl_%A_%a.err" \
#          "0 - Data Prep/Processing/sbatch_subcluster.sh"

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

COMPARTMENTS=("CNS neuron" "epithelial cell" "glial cell" "sensory neuron" "muscle cell")
IDX="${SLURM_ARRAY_TASK_ID:-0}"
COMP="${COMPARTMENTS[$IDX]}"

# Source object: the Stage-3 annotated atlas (carries full-gene .raw needed to
# re-select per-compartment HVGs). subcluster_compartment.py writes to
# <input parent>/subclusters/<slug>/ by default.
INPUT="${FLY_PROCESSING_OUTPUTS}/03_Integrated/sweep/scale_dataset/fly_annotated.h5ad"

echo "[subcl] host=$(hostname) job=${SLURM_JOB_ID:-NA} array_task=${IDX} compartment='${COMP}'"

python "${REPO_ROOT}/0 - Data Prep/Processing/subcluster_compartment.py" \
    --input "${INPUT}" \
    --compartment "${COMP}" \
    --fine-col afca_annotation \
    --broad-col afca_annotation_broad

echo "[subcl] done compartment='${COMP}'"

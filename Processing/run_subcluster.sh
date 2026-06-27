#!/bin/bash
#SBATCH -J fly_subcluster
#SBATCH -t 06:00:00
#SBATCH -n 16
#SBATCH --mem=200G
#SBATCH --mail-type=END,FAIL

# Stage-4 per-compartment subclustering launcher. Mirrors run_integrate.sh:
# resolve REPO_ROOT, source paths.sh, activate the BatchCorrection env, run.
#
# Usage:
#   sbatch "0 - Data Prep/Processing/run_subcluster.sh" "CNS neuron"
#   sbatch "0 - Data Prep/Processing/run_subcluster.sh" "CNS neuron" --resolution 2.5 --n-hvgs 3000
#   bash   "0 - Data Prep/Processing/run_subcluster.sh" "CNS neuron"   # interactive

set -euo pipefail

if [[ -n "${REPO_ROOT:-}" && -f "${REPO_ROOT}/config/paths.sh" ]]; then
    :
else
    _SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
fi
source "${REPO_ROOT}/config/paths.sh"

# First positional arg is the compartment; the rest pass through.
COMPARTMENT="${1:-CNS neuron}"
if [[ $# -ge 1 ]]; then shift; fi
EXTRA_ARGS=("$@")

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

echo "[run_subcluster] compartment='${COMPARTMENT}'"
python "${REPO_ROOT}/0 - Data Prep/Processing/subcluster_compartment.py" \
    --compartment "${COMPARTMENT}" \
    "${EXTRA_ARGS[@]}"

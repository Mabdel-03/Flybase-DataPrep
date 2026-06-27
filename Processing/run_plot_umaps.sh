#!/bin/bash
#SBATCH -J fly_plot_umaps
#SBATCH -t 01:00:00
#SBATCH -n 4
#SBATCH --mem=120G
#SBATCH --mail-type=END,FAIL

# Plot cell-type-annotated UMAPs from the original AFCA object (authors' UMAP).
# Read-only w.r.t. the data; just renders figures. Mirrors run_integrate.sh:
# resolve REPO_ROOT, source paths.sh, activate the BatchCorrection env, run.
#
# Usage:
#   sbatch "0 - Data Prep/Processing/run_plot_umaps.sh"
#   bash   "0 - Data Prep/Processing/run_plot_umaps.sh"        # interactive (no SLURM)

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

echo "[run_plot_umaps] input=${FLY_INPUT_H5AD}"
python "${REPO_ROOT}/0 - Data Prep/Processing/plot_celltype_umaps.py" "$@"

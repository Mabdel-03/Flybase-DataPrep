#!/bin/bash
#SBATCH -J fly_annotate_v2
#SBATCH -t 06:00:00
#SBATCH -n 16
#SBATCH --mem=200G
#SBATCH --mail-type=END,FAIL

# Stage-5 atlas-anchored de novo annotation launcher. Mirrors run_integrate.sh /
# run_subcluster.sh: resolve REPO_ROOT, source paths.sh, activate BatchCorr env.
#
# Usage:
#   sbatch "0 - Data Prep/Processing/run_annotate_v2.sh"
#   sbatch "0 - Data Prep/Processing/run_annotate_v2.sh" --scorer score_genes
#   bash   "0 - Data Prep/Processing/run_annotate_v2.sh" --subsample 40000   # interactive smoke

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

WIN="${REPO_ROOT}/0 - Data Prep/outputs/03_Integrated/sweep/scale_dataset"
echo "[run_annotate_v2] input=${WIN}/fly_annotated.h5ad"
python "${REPO_ROOT}/0 - Data Prep/Processing/annotate_v2.py" \
    --input "${WIN}/fly_annotated.h5ad" \
    --subcluster "cns_neuron=${WIN}/subclusters/cns_neuron/subcluster.h5ad" \
    --output-dir "${WIN}/annotate_v2" \
    --overwrite \
    "$@"

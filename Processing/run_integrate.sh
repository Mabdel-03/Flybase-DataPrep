#!/bin/bash
#SBATCH -J fly_integrate
#SBATCH -t 12:00:00
#SBATCH -n 16
#SBATCH --mem=200G
#SBATCH --mail-type=BEGIN,END,FAIL

# Flybase adapted Stage-3 launcher.
# Mirrors the structure of the Transcriptomics Stage-3 wrapper
# (Processing/Tsai/Pipeline/03_integration_annotation.sh): source paths.sh,
# activate the BatchCorrection env, run the integration script. One object (not
# 478 concatenations), so 200G/16 cores is ample vs the human 500G/32.
#
# Usage (from the repo root; quote the spaced bucket dir):
#   sbatch "0 - Data Prep/Processing/run_integrate.sh"                 # primary variant (sex)
#   sbatch "0 - Data Prep/Processing/run_integrate.sh" no_harmony      # a named variant
#   VARIANT=age sbatch "0 - Data Prep/Processing/run_integrate.sh"     # via env var
#   bash   "0 - Data Prep/Processing/run_integrate.sh" sex --subsample 20000 --skip-harmony  # interactive smoke

set -euo pipefail

# Resolve REPO_ROOT. Under sbatch the script is copied to /var/spool/slurmd, so
# BASH_SOURCE is unreliable; honor a pre-exported REPO_ROOT (set by the submit
# command via --export) and fall back to BASH_SOURCE for interactive/bash use.
if [[ -n "${REPO_ROOT:-}" && -f "${REPO_ROOT}/config/paths.sh" ]]; then
    :
else
    _SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
fi
source "${REPO_ROOT}/config/paths.sh"

# First positional arg (if present) is the variant; the rest pass through.
VARIANT="${1:-${VARIANT:-primary}}"
if [[ $# -ge 1 ]]; then shift; fi
EXTRA_ARGS=("$@")

mkdir -p "${FLY_LOGS}"
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1

# Activate the Stage-3 conda env (relax nounset for activate.d scripts).
set +u
source "${CONDA_INIT_SCRIPT}"
conda activate "${BATCHCORR_ENV}"
set -u
if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "ERROR: failed to activate conda env: ${BATCHCORR_ENV}" >&2
    exit 1
fi

echo "[run_integrate] variant=${VARIANT} input=${FLY_INPUT_H5AD}"
python "${REPO_ROOT}/0 - Data Prep/Processing/integrate_annotate.py" \
    --variant "${VARIANT}" \
    "${EXTRA_ARGS[@]}"

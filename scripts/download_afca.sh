#!/bin/bash
# =============================================================================
# Idempotently download the AFCA (Aging Fly Cell Atlas) head+body atlas.
#
# Source: anonymous SharePoint share from the Hongjie Li Lab (BCM).
#   .../HongjieLab/AFCA/dataToShare/afcaFca_headBody_v1.0/normalizedCounts/
#       adata_headBody_S_v1.0.h5ad
# The file is ~3.06 GiB (exactly 3,287,250,398 bytes), HDF5/AnnData, the
# stringent ("S") combined head+body object: 566,254 cells x 15,992 genes,
# X = log1p(normalize_total(counts, target_sum=1e4)).
#
# Usage:  bash scripts/download_afca.sh
# =============================================================================
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
source "${REPO_ROOT}/config/paths.sh"

# Anonymous SharePoint share token (last path segment of the :u:/g/ share URL).
TOKEN="EXgUmzr6BFJJhAHJ106pd1sBU8ksAMg4VHxkyqncy3J7Qw"
DL_URL="https://bcmedu-my.sharepoint.com/personal/u239500_bcm_edu/_layouts/15/download.aspx?share=${TOKEN}"
EXPECTED_BYTES=3287250398

OUT="${FLY_INPUT_H5AD}"
mkdir -p "$(dirname "${OUT}")"

if [[ -f "${OUT}" ]]; then
    have=$(stat -c%s "${OUT}")
    if [[ "${have}" == "${EXPECTED_BYTES}" ]]; then
        echo "[skip] ${OUT} already present and correct size (${have} bytes)."
        exit 0
    fi
    echo "[warn] ${OUT} exists but size=${have} != ${EXPECTED_BYTES}; resuming/redownloading."
fi

echo "[download] -> ${OUT}"
curl -L --fail --retry 3 --retry-delay 5 -C - \
    "${DL_URL}" \
    -o "${OUT}" \
    -w "DONE http=%{http_code} size=%{size_download} time=%{time_total}s\n"

have=$(stat -c%s "${OUT}")
if [[ "${have}" != "${EXPECTED_BYTES}" ]]; then
    echo "ERROR: downloaded size ${have} != expected ${EXPECTED_BYTES}" >&2
    exit 1
fi
# HDF5 files start with the signature \x89HDF\r\n\x1a\n
magic=$(head -c4 "${OUT}" | od -An -tx1 | tr -d ' ')
if [[ "${magic}" != "89484446" ]]; then
    echo "ERROR: ${OUT} is not a valid HDF5 file (magic=${magic})" >&2
    exit 1
fi
echo "[ok] AFCA atlas downloaded and verified: ${OUT} (${have} bytes)"

#!/usr/bin/env bash
# Smoke test: verify scLong + dependencies import correctly inside the container,
# and (if --gpu is passed) run the synthetic architecture forward-pass smoke test —
# no real checkpoint/gene metadata required. See pipeline/smoke_test_architecture.py.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIF="$SCRIPT_DIR/sclong_v1.0.0.sif"
PIPELINE_DIR="$SCRIPT_DIR/../pipeline"

if [[ ! -f "$SIF" ]]; then
    echo "ERROR: Container not found: $SIF"
    echo "Run: ./build.sh (or ./build.sh --fakeroot)"
    exit 1
fi

echo "=== Import check ==="
apptainer exec --cleanenv --env PYTHONNOUSERSITE=1 "$SIF" python -c "
import torch, torch_geometric, scanpy, anndata, scib, einops, local_attention
print('torch:          ', torch.__version__, '| cuda:', torch.version.cuda)
print('torch_geometric:', torch_geometric.__version__)
print('scanpy:         ', scanpy.__version__)
print('scib:           ', scib.__version__)
from performer_pytorch_cont.ding_models import DualEncoderSCFM
print('scLong model class import OK')
"

echo ""
echo "=== CLI help check ==="
apptainer exec --cleanenv --env PYTHONNOUSERSITE=1 \
    --bind "$PIPELINE_DIR:/pipeline" \
    "$SIF" python /pipeline/run_zero_shot_batch_integration.py --help

if [[ "${1:-}" == "--gpu" ]]; then
    echo ""
    echo "=== GPU architecture forward-pass smoke test (synthetic weights, no checkpoint needed) ==="
    apptainer exec --nv --cleanenv --env PYTHONNOUSERSITE=1 \
        --bind "$PIPELINE_DIR:/pipeline" \
        "$SIF" python /pipeline/smoke_test_architecture.py
fi

echo ""
echo "All checks passed."

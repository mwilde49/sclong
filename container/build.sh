#!/usr/bin/env bash
# Build sclong_v1.0.0.sif from apptainer.def
# Requires sudo, or --fakeroot on systems where user namespaces are enabled
# (this is the default path used in WSL2 dev — see README.md).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIF="$SCRIPT_DIR/sclong_v1.0.0.sif"

echo "Building: $SIF"
if [[ "${1:-}" == "--fakeroot" ]]; then
    apptainer build --fakeroot "$SIF" "$SCRIPT_DIR/apptainer.def"
else
    apptainer build "$SIF" "$SCRIPT_DIR/apptainer.def"
fi
echo "Done: $SIF"

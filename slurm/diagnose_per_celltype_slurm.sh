#!/bin/bash
#SBATCH --job-name=sclong-diag-percelltype
#SBATCH --output=logs/sclong_diag_percelltype_%j.out
#SBATCH --error=logs/sclong_diag_percelltype_%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=h200
# CPU-only diagnostic (pipeline/diagnose_per_celltype.py reuses a saved *_ce.npy, never
# touches the model or GPU) -- lands on the h200 partition anyway purely because that's
# the queue this project has confirmed access to; no --gpus requested. If a dedicated
# CPU-only partition is available on Juno, prefer that instead.

module load apptainer

PROJECT_ROOT=/groups/tprice/pipelines
SCRATCH_ROOT=/scratch/juno/$USER
REPO_ROOT="$PROJECT_ROOT/containers/sclong"
CONTAINER="$REPO_ROOT/container/sclong_v1.0.0.sif"
PIPELINE_DIR="$REPO_ROOT/pipeline"
DATA_ROOT="$PROJECT_ROOT/references/sclong"

EMB="$1"
if [ -z "$EMB" ]; then
    echo "Usage: sbatch diagnose_per_celltype_slurm.sh /path/to/some_variant_ce.npy"
    exit 1
fi

mkdir -p logs

apptainer exec \
    --cleanenv \
    --env PYTHONNOUSERSITE=1 \
    --bind $PROJECT_ROOT:$PROJECT_ROOT \
    --bind $SCRATCH_ROOT:$SCRATCH_ROOT \
    --bind $PIPELINE_DIR:/pipeline \
    $CONTAINER \
    python /pipeline/diagnose_per_celltype.py \
        --data-root "$DATA_ROOT" \
        --embedding "$EMB"

#!/bin/bash
#SBATCH --job-name=sclong
#SBATCH --output=logs/sclong_%j.out
#SBATCH --error=logs/sclong_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=h200
#SBATCH --gpus=1
# CORRECTED 2026-08-04 (ROADMAP.md §14.6): this default was stale from before Juno
# access was confirmed -- it pointed at partition=a30/gres=gpu:nvidia_a30:1, which this
# project has never actually verified works (a30's 24GB VRAM is also a real OOM risk at
# batch_size=48+, vs. H200's 141GB). Every real completed full-scale run (jobs
# 315836, 316352, 316699) used partition=h200. Use `--gpus=1` (generic count-based
# allocation), not a named --gres string -- we tried guessing the H200 gres name
# (gpu:nvidia_h200:1) and it failed with "Requested node configuration is not
# available"; --gpus=1 is confirmed working and sidesteps needing the exact string.

module load apptainer

PROJECT_ROOT=/groups/tprice/pipelines
SCRATCH_ROOT=/scratch/juno/$USER
WORK_ROOT=/work/$USER

REPO_ROOT="$PROJECT_ROOT/containers/sclong"
CONTAINER="$REPO_ROOT/container/sclong_v1.0.0.sif"
PIPELINE_DIR="$REPO_ROOT/pipeline"
DATA_ROOT="$PROJECT_ROOT/references/sclong"

# --- Pre-flight checks ---

if [ ! -f "$CONTAINER" ]; then
    echo "ERROR: Container not found: $CONTAINER"
    echo "scp it from the validated local build, or: cd $REPO_ROOT/container && ./build.sh --fakeroot"
    exit 1
fi

if [ ! -d "$DATA_ROOT/checkpoints" ]; then
    echo "ERROR: Data not staged: $DATA_ROOT"
    echo "rsync gene_meta/, checkpoints/, datasets/ there first — see sclong/ROADMAP.md §1."
    exit 1
fi

BATCH_SIZE=${1:-32}
N_CELLS=${2:-all}
OUTPUT_DIR=${3:-$SCRATCH_ROOT/sclong_results}
GENE_MAPPING_VERSION=${4:-}    # empty -> pipeline reads gene_meta/mappings/CURRENT
POOLING=${5:-full}             # full | measured-only
NORMALIZATION=${6:-as-shipped} # as-shipped | cpm

mkdir -p logs "$OUTPUT_DIR"

MAPPING_FLAG=()
if [ -n "$GENE_MAPPING_VERSION" ]; then
    MAPPING_FLAG=(--gene-mapping-version "$GENE_MAPPING_VERSION")
fi

echo "====================================================================="
echo "  scLong — Stage 1 (zero-shot batch integration)"
echo "  batch_size=$BATCH_SIZE n_cells=$N_CELLS pooling=$POOLING normalization=$NORMALIZATION"
echo "  gene_mapping_version=${GENE_MAPPING_VERSION:-<CURRENT pointer>}"
echo "====================================================================="

apptainer exec \
    --nv \
    --cleanenv \
    --env PYTHONNOUSERSITE=1 \
    --env MPLBACKEND=Agg \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --bind $PROJECT_ROOT:$PROJECT_ROOT \
    --bind $SCRATCH_ROOT:$SCRATCH_ROOT \
    --bind $WORK_ROOT:$WORK_ROOT \
    --bind $PIPELINE_DIR:/pipeline \
    $CONTAINER \
    python /pipeline/run_zero_shot_batch_integration.py \
        --data-root "$DATA_ROOT" \
        --output-dir "$OUTPUT_DIR" \
        --batch-size "$BATCH_SIZE" \
        --n-cells "$N_CELLS" \
        --progress-every 25 \
        --pooling "$POOLING" \
        --normalization "$NORMALIZATION" \
        "${MAPPING_FLAG[@]}"

#!/bin/bash
#SBATCH --job-name=sclong
#SBATCH --output=logs/sclong_%j.out
#SBATCH --error=logs/sclong_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=a30
#SBATCH --gres=gpu:nvidia_a30:1
# For larger batch sizes / more VRAM, swap to:
#   #SBATCH --partition=h100
#   #SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
# (same swap dconvatac-gpu documents — see DCONVATAC_HPC_GUIDE.md in hpc/)

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

mkdir -p logs "$OUTPUT_DIR"

echo "====================================================================="
echo "  scLong — Stage 1 (zero-shot batch integration) — batch_size=$BATCH_SIZE n_cells=$N_CELLS"
echo "====================================================================="

apptainer exec \
    --nv \
    --cleanenv \
    --env PYTHONNOUSERSITE=1 \
    --env MPLBACKEND=Agg \
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
        --progress-every 25

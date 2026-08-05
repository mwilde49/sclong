#!/bin/bash
#SBATCH --job-name=sclong-additive
#SBATCH --output=logs/sclong_additive_%j.out
#SBATCH --error=logs/sclong_additive_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=h200
#SBATCH --gpus=1
# Mirrors slurm/sclong_gpu_slurm_template.sh's structure (that one is proven working —
# jobs 315836/316352 both ran this way). Deliberately an explicit #!/bin/bash SCRIPT
# FILE, not `sbatch --wrap="..."` -- --wrap runs under /bin/sh (dash on this cluster),
# and `module load apptainer` sources a bash-completion file with a hyphenated function
# name that's illegal in POSIX sh, killing the job at shell-startup before any real
# command runs (see ROADMAP.md §14.6 for the full diagnosis, job 316668).

module load apptainer

PROJECT_ROOT=/groups/tprice/pipelines
SCRATCH_ROOT=/scratch/juno/$USER
WORK_ROOT=/work/$USER

REPO_ROOT="$PROJECT_ROOT/containers/sclong"
CONTAINER="$REPO_ROOT/container/sclong_v1.0.0.sif"
PIPELINE_DIR="$REPO_ROOT/pipeline"
EXPERIMENTS_DIR="$REPO_ROOT/experiments"
DATA_ROOT="$PROJECT_ROOT/references/sclong"

N_CELLS=${1:-20}
SEED=${2:-42}
GENE_MAPPING_VERSION=${3:-v2_hgnc_plus_biomart_synonyms}
POOLING=${4:-measured-only}
OUTPUT_DIR=${5:-$SCRATCH_ROOT/sclong_results/additive_pooling}

mkdir -p logs "$OUTPUT_DIR"

if [ ! -f "$CONTAINER" ]; then
    echo "ERROR: Container not found: $CONTAINER"
    exit 1
fi

echo "====================================================================="
echo "  scLong — additive get_cell_emb test"
echo "  n_cells=$N_CELLS seed=$SEED gene_mapping=$GENE_MAPPING_VERSION pooling=$POOLING"
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
    --bind $EXPERIMENTS_DIR:/experiments \
    $CONTAINER \
    python /experiments/additive_pooling_test.py \
        --data-root "$DATA_ROOT" \
        --output-dir "$OUTPUT_DIR" \
        --n-cells "$N_CELLS" \
        --seed "$SEED" \
        --gene-mapping-version "$GENE_MAPPING_VERSION" \
        --pooling "$POOLING"

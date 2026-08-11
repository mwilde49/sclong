#!/bin/bash
#SBATCH --job-name=sclong-additive
#SBATCH --output=logs/sclong_additive_%j.out
#SBATCH --error=logs/sclong_additive_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=h200
#SBATCH --gpus=1
# Mirrors slurm/sclong_gpu_slurm_template.sh's structure AND resource allocation (that
# one is proven working for real full-scale runs — jobs 315836/316352 both used
# time=24:00:00/cpus=16/mem=128G). Bumped up from this script's original
# time=02:00:00/cpus=8/mem=32G, which was sized only for the n=20 smoke test (job
# 316699) and is not safe for a full n=16,382 run computing 3 embedding variants plus
# 3x the bio-conservation (NMI/ARI/isolated-F1 Leiden) panel per run instead of 1x.
# Deliberately an explicit #!/bin/bash SCRIPT FILE, not `sbatch --wrap="..."` -- --wrap
# runs under /bin/sh (dash on this cluster), and `module load apptainer` sources a
# bash-completion file with a hyphenated function name that's illegal in POSIX sh,
# killing the job at shell-startup before any real command runs (see ROADMAP.md §14.6
# for the full diagnosis, job 316668).

module load apptainer

PROJECT_ROOT=/groups/tprice/pipelines
SCRATCH_ROOT=/scratch/juno/$USER
WORK_ROOT=/work/$USER

REPO_ROOT="$PROJECT_ROOT/containers/sclong"
CONTAINER="$REPO_ROOT/container/sclong_v1.0.0.sif"
PIPELINE_DIR="$REPO_ROOT/pipeline"
EXPERIMENTS_DIR="$REPO_ROOT/experiments"
DATA_ROOT="$PROJECT_ROOT/references/sclong"

# Default N_CELLS changed from 20 -> all: this script is being promoted from a smoke
# test to the full-scale retest itself (see ROADMAP.md §14.2 update). Pass "20"
# explicitly as $1 to reproduce the old smoke test.
N_CELLS=${1:-all}
SEED=${2:-42}
GENE_MAPPING_VERSION=${3:-v2_hgnc_plus_biomart_synonyms}
POOLING=${4:-measured-only}
# Default changed from unset(as-shipped) -> cpm: the whole point of this retest is to
# compare A/B/C under the SAME normalization as the project's 0.8550 reference config
# (job 316352), not the as-shipped normalization the original n=20 smoke test silently
# used. Pass "as-shipped" explicitly as $6 to reproduce the old smoke test's config.
NORMALIZATION=${5:-cpm}
OUTPUT_DIR=${6:-$SCRATCH_ROOT/sclong_results/additive_pooling_full}

mkdir -p logs "$OUTPUT_DIR"

if [ ! -f "$CONTAINER" ]; then
    echo "ERROR: Container not found: $CONTAINER"
    exit 1
fi

echo "====================================================================="
echo "  scLong — additive get_cell_emb test (paper Eq. 11)"
echo "  n_cells=$N_CELLS seed=$SEED gene_mapping=$GENE_MAPPING_VERSION pooling=$POOLING normalization=$NORMALIZATION"
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
        --pooling "$POOLING" \
        --normalization "$NORMALIZATION"

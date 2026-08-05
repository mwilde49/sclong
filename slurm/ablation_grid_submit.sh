#!/bin/bash
# Submits the full Stage 1 ablation grid: 2 gene-mapping x 2 pooling x 3 normalization
# = 12 combos, all full-scale (16,382 cells, ~90 min each on H200). Skips the one combo
# we already have a real, confirmed run for (v2 x measured-only x cpm = job 316352,
# ASW=0.8550 -- see ROADMAP.md §14.0). Purpose: (1) attribute the isolated_label_F1
# regression found in §14.1 to a specific lever, (2) search for a genuinely better
# config than the current "kitchen sink" default, not just explain it. See
# ROADMAP.md §14.6 item 7.
#
# `literal` normalization is deliberately excluded -- proven mathematically
# near-identical to an already-tested, already-worse variant (ROADMAP.md §14.5);
# spending 11 more GPU-hours on a 12th run of it would be wasted compute.
#
# Run from the repo root (containers/sclong):
#   bash slurm/ablation_grid_submit.sh
#
# All jobs land in $SCRATCH_ROOT/sclong_results_grid/ with self-describing filenames
# (config_tag is baked in by the pipeline already -- see run_zero_shot_batch_integration.py).
# Each is a fully independent SLURM job -- they queue and run as H200 capacity allows,
# no manual concurrency management needed.

set -euo pipefail

SCRATCH_ROOT=/scratch/juno/$USER
OUTPUT_DIR="$SCRATCH_ROOT/sclong_results_grid"
BATCH_SIZE=48

MAPPINGS=(v1_biomart_symbol_only v2_hgnc_plus_biomart_synonyms)
POOLINGS=(full measured-only)
NORMALIZATIONS=(as-shipped cpm cpm-norescale)

mkdir -p logs "$OUTPUT_DIR"

submitted=0
skipped=0
for mapping in "${MAPPINGS[@]}"; do
  for pooling in "${POOLINGS[@]}"; do
    for norm in "${NORMALIZATIONS[@]}"; do
      # Skip the one combo we already have a confirmed real run for (job 316352).
      if [ "$mapping" = "v2_hgnc_plus_biomart_synonyms" ] && [ "$pooling" = "measured-only" ] && [ "$norm" = "cpm" ]; then
        echo "SKIP (already have job 316352): map=$mapping pool=$pooling norm=$norm"
        skipped=$((skipped + 1))
        continue
      fi
      echo "SUBMIT: map=$mapping pool=$pooling norm=$norm"
      sbatch slurm/sclong_gpu_slurm_template.sh \
        "$BATCH_SIZE" all "$OUTPUT_DIR" "$mapping" "$pooling" "$norm"
      submitted=$((submitted + 1))
    done
  done
done

echo
echo "Submitted $submitted jobs, skipped $skipped (already have real data)."
echo "Watch with: squeue -u \$USER"
echo "When all complete, summarize with:"
echo "  apptainer exec --cleanenv --env PYTHONNOUSERSITE=1 \\"
echo "      --bind /groups/tprice/pipelines:/groups/tprice/pipelines \\"
echo "      --bind $SCRATCH_ROOT:$SCRATCH_ROOT \\"
echo "      --bind \$(pwd)/pipeline:/pipeline \\"
echo "      container/sclong_v1.0.0.sif \\"
echo "      python /pipeline/summarize_ablation_grid.py --grid-dir $OUTPUT_DIR"

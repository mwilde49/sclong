"""
Diagnostic for the Stage 1 full-run gap (0.8262 vs. paper's 0.9561): reloads the
already-computed embedding from a prior run() and asks scib for a per-cell-type
breakdown (return_all=True) instead of the single blended number, to see whether the
shortfall concentrates in the rare cell types excluded from earlier small-n subsamples
(epsilon=32, schwann=25, t_cell=7 cells total) or is spread evenly across all 14 types.

Does NOT re-run the model — reuses the saved embedding, so this is seconds, not ~88 minutes.

Run:
    apptainer exec --cleanenv --env PYTHONNOUSERSITE=1 \
        --bind /groups/tprice/pipelines:/groups/tprice/pipelines \
        --bind /scratch/juno/$USER:/scratch/juno/$USER \
        --bind $(pwd)/pipeline:/pipeline \
        container/sclong_v1.0.0.sif \
        python /pipeline/diagnose_per_celltype.py \
          --data-root /groups/tprice/pipelines/references/sclong \
          --embedding /scratch/juno/$USER/sclong_results/scLong_batch_cell_emb_mode_ce.npy
"""
import argparse
from pathlib import Path

import numpy as np
import scanpy as sc
import scib

p = argparse.ArgumentParser()
p.add_argument("--data-root", type=str, required=True)
p.add_argument("--embedding", type=str, required=True)
args = p.parse_args()

adata = sc.read_h5ad(str(Path(args.data_root) / "datasets" / "pancreas_scib.h5ad"))
sc.pp.filter_cells(adata, min_genes=10)
sc.pp.filter_genes(adata, min_cells=10)

emb = np.load(args.embedding)
print(f"embedding shape: {emb.shape}, adata cells after filtering: {adata.n_obs}")
assert emb.shape[0] == adata.n_obs, "embedding/adata cell count mismatch — wrong run's file?"

adata.obsm["X_scLong"] = emb

# return_all=True -> (asw, sil_means, sil_df); sil_means is already grouped by celltype
# (scib source: sil_means = sil_df.groupby("group").mean(); asw = sil_means.mean()).
# NOTE: any celltype present in only 1 batch is silently SKIPPED (not scored, not
# counted) -- "mixing" isn't measurable with nothing to mix against.
asw, sil_means, sil_df = scib.metrics.silhouette_batch(
    adata, "batch", "celltype", embed="X_scLong", metric="euclidean",
    return_all=True, verbose=False,
)
print(f"\noverall (should match the earlier run's 0.8262): {asw:.4f}\n")

counts = adata.obs["celltype"].value_counts()
sil_means = sil_means.rename(columns={"silhouette_score": "mean_batch_asw"})
sil_means["n_cells_total"] = counts.reindex(sil_means.index)
sil_means = sil_means.sort_values("mean_batch_asw")

included = set(sil_means.index)
skipped = set(counts.index) - included
print("per-celltype batch-ASW (lower = worse batch mixing for that type):")
print(sil_means.to_string())
print(f"\nSKIPPED entirely (present in only 1 batch, or other scib exclusion rule): {sorted(skipped) or 'none'}")

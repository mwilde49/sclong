"""
Bio-conservation companion to diagnose_per_celltype.py — reuses an already-computed
embedding (does NOT re-run the model) and adds NMI/ARI/isolated-label-F1 alongside batch
ASW, via the same eval_scib_metrics() the main pipeline now uses by default (see
ROADMAP.md §14.1). CPU-only, seconds not GPU-minutes, per the same "reload + assert shape
match" pattern as diagnose_per_celltype.py.

Two ways to supply cell identity:
  --obs-csv <path>   preferred, exact: the *_obs.csv this pipeline now writes alongside
                      every *_ce.npy (added same day as this script). Works for any run
                      size, including seeded subsamples.
  (omitted)           fallback for embeddings computed BEFORE the *_obs.csv fix (e.g. the
                      original 0.8262 baseline / 0.8550 combined runs): reloads the full
                      h5ad with the identical filter_cells/filter_genes calls the main
                      pipeline uses and asserts the cell COUNT matches. This only works if
                      that run used --n-cells all (no subsampling) or you know its exact
                      seed and n -- assert-by-count alone cannot prove cell IDENTITY for a
                      subsampled run, only catch a gross size mismatch. Prefer --obs-csv
                      whenever it exists.

Run (inside the container, no --nv needed -- CPU only):
    apptainer exec --cleanenv --env PYTHONNOUSERSITE=1 \
        --bind /groups/tprice/pipelines:/groups/tprice/pipelines \
        --bind /scratch/juno/$USER:/scratch/juno/$USER \
        --bind $(pwd)/pipeline:/pipeline \
        container/sclong_v1.0.0.sif \
        python /pipeline/diagnose_bio_conservation.py \
          --data-root /groups/tprice/pipelines/references/sclong \
          --embedding /scratch/juno/$USER/sclong_results/<run>_ce.npy \
          [--obs-csv /scratch/juno/$USER/sclong_results/<run>_obs.csv]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

from eval_utils import eval_scib_metrics, print_sys

p = argparse.ArgumentParser()
p.add_argument("--data-root", type=str, required=True)
p.add_argument("--embedding", type=str, required=True)
p.add_argument("--obs-csv", type=str, default=None,
                help="This run's *_obs.csv (celltype+batch), if it exists. Strongly "
                     "preferred over the full-reload fallback -- see module docstring.")
p.add_argument("--label", type=str, default=None,
                help="Free-text tag for the printed report (e.g. 'baseline_0.8262', "
                     "'combined_0.8550'). Defaults to the embedding filename.")
args = p.parse_args()

emb = np.load(args.embedding)
label = args.label or Path(args.embedding).name
print_sys(f"[{label}] embedding shape: {emb.shape}")

if args.obs_csv:
    obs = pd.read_csv(args.obs_csv, index_col=0)
    assert emb.shape[0] == len(obs), (
        f"embedding/obs row count mismatch ({emb.shape[0]} vs {len(obs)}) — wrong pair?"
    )
    for col in ("celltype", "batch"):
        assert col in obs.columns, f"--obs-csv missing required column {col!r}"
    adata = sc.AnnData(X=np.zeros((len(obs), 1), dtype=np.float32), obs=obs)
    print_sys(f"[{label}] cell identity loaded from {args.obs_csv} ({len(obs)} cells)")
else:
    print_sys(f"[{label}] WARNING: no --obs-csv given, falling back to full-h5ad reload + "
              f"assert-by-count. This can only catch a gross size mismatch, not confirm "
              f"cell IDENTITY for a subsampled run — see module docstring.")
    adata = sc.read_h5ad(str(Path(args.data_root) / "datasets" / "pancreas_scib.h5ad"))
    sc.pp.filter_cells(adata, min_genes=10)
    sc.pp.filter_genes(adata, min_cells=10)
    assert emb.shape[0] == adata.n_obs, (
        f"embedding/adata cell count mismatch ({emb.shape[0]} vs {adata.n_obs}) — wrong "
        f"run's file, or this run was subsampled (use --obs-csv instead)?"
    )

adata.obsm["X_scLong"] = emb

results = eval_scib_metrics(
    adata, batch_key="batch", label_key="celltype", embedding_key="X_scLong",
    bio_conservation=True,
)

print(f"\n=== [{label}] bio-conservation panel ({emb.shape[0]} cells) ===")
for k, v in results.items():
    print(f"  {k:22s} = {v:.4f}")
if "ASW_label/batch" in results:
    others = {k: v for k, v in results.items() if k != "ASW_label/batch"}
    if others and all(v < 0.05 for v in others.values()):
        print("\n  ^^ WARNING: bio-conservation scores are all near zero alongside a "
              "nonzero ASW_label/batch — this is the homogenization pattern flagged in "
              "ROADMAP.md §14.1 (batch mixing without real cell-type structure). Do not "
              "report the ASW number as progress without investigating this.")
print("\nDONE", flush=True)

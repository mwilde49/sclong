"""
Correctness sanity gate for the 'literal' normalization variant (see
run_zero_shot_batch_integration.py::literal_renormalize) -- CPU-only, no model/GPU
needed. Loads the full pancreas_scib.h5ad, applies the paper's literal asymmetric
formula to adata.layers['counts'], and reports what fraction of cells land at
post-transform max ~= 10.0 vs. below 10.0 (magnified) vs above (should be
structurally impossible post-clip, since clipping enforces max==10 exactly).

Also reports the as-shipped .X per-cell-max distribution for direct comparison, and a
per-batch (protocol) breakdown of raw library size, to size the depth-confound risk
flagged in the assignment (droplet vs. full-length protocols).
"""
import sys
import numpy as np
import scanpy as sc

sys.path.insert(0, "/pipeline")
from run_zero_shot_batch_integration import literal_renormalize, cpm_renormalize  # noqa: E402

adata = sc.read_h5ad("/data/datasets/pancreas_scib.h5ad")
sc.pp.filter_cells(adata, min_genes=10)
sc.pp.filter_genes(adata, min_cells=10)
print(f"n_cells={adata.n_obs} n_genes={adata.n_vars}", flush=True)

counts = adata.layers["counts"]
counts_dense = counts.toarray() if hasattr(counts, "toarray") else np.asarray(counts)

X_shipped = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
row_max_shipped = X_shipped.max(axis=1)

X_literal = literal_renormalize(counts_dense)
row_max_literal = X_literal.max(axis=1)

X_cpm = cpm_renormalize(counts_dense)
row_max_cpm = X_cpm.max(axis=1)

lib_size = counts_dense.sum(axis=1)

def summarize(name, row_max):
    at_10 = np.isclose(row_max, 10.0, atol=1e-3)
    below_10 = row_max < 10.0 - 1e-3
    above_10 = row_max > 10.0 + 1e-3
    print(f"[{name}] mean={row_max.mean():.4f} std={row_max.std():.4f} "
          f"n_at_10={at_10.sum()}/{len(row_max)} ({100*at_10.mean():.2f}%) "
          f"n_below_10={below_10.sum()} ({100*below_10.mean():.2f}%) "
          f"n_above_10={above_10.sum()} ({100*above_10.mean():.2f}%)", flush=True)

summarize("as-shipped .X", row_max_shipped)
summarize("literal (log(x/10000+1), clip/magnify)", row_max_literal)
summarize("cpm (lib-size-first, log1p, uniform rescale)", row_max_cpm)

print("\n--- per-batch raw library size (depth-confound check) ---", flush=True)
batches = adata.obs["batch"].astype(str).values
for b in sorted(set(batches)):
    mask = batches == b
    ls = lib_size[mask]
    print(f"batch={b:12s} n={mask.sum():5d} lib_size mean={ls.mean():10.1f} "
          f"median={np.median(ls):10.1f} std={ls.std():10.1f}", flush=True)

print("\n--- per-batch pre-rescale log(x/10000+1) row-max (i.e. BEFORE the "
      "clip/magnify step -- this determines which branch each cell takes) ---", flush=True)
x_prerescale = np.log(counts_dense / 10000.0 + 1.0)
row_max_prerescale = x_prerescale.max(axis=1)
for b in sorted(set(batches)):
    mask = batches == b
    rm = row_max_prerescale[mask]
    print(f"batch={b:12s} n={mask.sum():5d} pre-rescale max mean={rm.mean():.4f} "
          f"median={np.median(rm):.4f} std={rm.std():.4f} frac_exceeds_10={100*np.mean(rm>10):.2f}%", flush=True)

print("\n--- correlation check: does post-'literal'-transform row max correlate with "
      "raw library size (the depth-confound risk)? ---", flush=True)
print(f"corr(lib_size, row_max_literal) = {np.corrcoef(lib_size, row_max_literal)[0,1]:.4f}", flush=True)
print(f"corr(lib_size, row_max_cpm)     = {np.corrcoef(lib_size, row_max_cpm)[0,1]:.4f}", flush=True)

print("\nDONE", flush=True)

"""
Follow-up to sanity_gate_literal_norm.py: the row-max correlation check there was
degenerate (row_max is always exactly 10.0 after either transform, zero variance, hence
the NaN correlation). This script instead checks whether the SHAPE of each cell's
post-transform profile (not just its max) carries a depth/protocol fingerprint, which is
the actual mechanism behind the Round 2 concern -- log(x/10000+1) compresses large counts
logarithmically but treats small counts near-linearly, so a per-cell max-rescale
does NOT equalize profile shape across cells of very different sequencing depth the way
library-size normalization does.

Diagnostic: for each cell (post max-rescale-to-10), what fraction of its measured,
nonzero genes land above value 5.0 (i.e., in the upper half of the 0-10 range -- a proxy
for how "saturated/flat" vs. "peaky" the profile is). If this fraction is strongly
batch/depth-correlated under 'literal' but much less so under 'cpm', that's direct
evidence for the predicted depth confound.
"""
import sys
import numpy as np
import scanpy as sc

sys.path.insert(0, "/pipeline")
from run_zero_shot_batch_integration import literal_renormalize, cpm_renormalize  # noqa: E402

adata = sc.read_h5ad("/data/datasets/pancreas_scib.h5ad")
sc.pp.filter_cells(adata, min_genes=10)
sc.pp.filter_genes(adata, min_cells=10)

counts = adata.layers["counts"]
counts_dense = counts.toarray() if hasattr(counts, "toarray") else np.asarray(counts)
lib_size = counts_dense.sum(axis=1)
batches = adata.obs["batch"].astype(str).values

X_literal = literal_renormalize(counts_dense)
X_cpm = cpm_renormalize(counts_dense)

def frac_upper_half_nonzero(X):
    nz = counts_dense > 0
    n_nz = nz.sum(axis=1)
    n_nz = np.maximum(n_nz, 1)
    upper = (X > 5.0) & nz
    return upper.sum(axis=1) / n_nz

frac_lit = frac_upper_half_nonzero(X_literal)
frac_cpm = frac_upper_half_nonzero(X_cpm)

print("--- fraction of each cell's nonzero measured genes with transformed value > 5.0 "
      "('saturation fraction' -- higher = flatter/more-saturated profile) ---", flush=True)
print(f"{'batch':12s} {'n':>6s} {'lit_mean':>10s} {'lit_std':>8s} {'cpm_mean':>10s} {'cpm_std':>8s} {'lib_mean':>12s}",
      flush=True)
for b in sorted(set(batches)):
    mask = batches == b
    print(f"{b:12s} {mask.sum():6d} {frac_lit[mask].mean():10.4f} {frac_lit[mask].std():8.4f} "
          f"{frac_cpm[mask].mean():10.4f} {frac_cpm[mask].std():8.4f} {lib_size[mask].mean():12.1f}", flush=True)

print("\n--- correlation of log10(lib_size) with saturation fraction, across all cells ---", flush=True)
log_lib = np.log10(np.maximum(lib_size, 1))
print(f"corr(log10(lib_size), frac_upper_half_literal) = {np.corrcoef(log_lib, frac_lit)[0,1]:.4f}", flush=True)
print(f"corr(log10(lib_size), frac_upper_half_cpm)     = {np.corrcoef(log_lib, frac_cpm)[0,1]:.4f}", flush=True)

# Also: simple ANOVA-style effect size -- how much of the variance in saturation fraction
# is explained by batch identity alone (eta-squared), literal vs cpm.
def eta_squared(y, groups):
    grand_mean = y.mean()
    ss_total = ((y - grand_mean) ** 2).sum()
    ss_between = 0.0
    for g in set(groups):
        yg = y[groups == g]
        ss_between += len(yg) * (yg.mean() - grand_mean) ** 2
    return ss_between / ss_total

print("\n--- eta-squared (fraction of variance in saturation-fraction explained by batch) ---", flush=True)
print(f"literal: eta^2 = {eta_squared(frac_lit, batches):.4f}", flush=True)
print(f"cpm:     eta^2 = {eta_squared(frac_cpm, batches):.4f}", flush=True)

print("\nDONE", flush=True)

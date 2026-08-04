"""
EXPERIMENT (not part of the shipped Stage 1 pipeline): tests whether applying scLong's
own documented preprocessing -- log(x/10000+1), then per-cell rescale so the max value is
exactly 10 -- to raw counts, instead of using pancreas_scib.h5ad's .X as-is, changes the
batch-ASW result.

Context (see sclong/ROADMAP.md for full history): the pipeline has always fed adata.X
directly into the model, on the assumption the author-provided pancreas_scib.h5ad was
already preprocessed to match scLong's pretraining normalization. Checking the FULL
dataset (not a small sample) shows that assumption doesn't hold uniformly: per-cell max
values range 6.1-13.0 (std=1.33), not a tight max=10 -- only ~6% of cells actually land
near 10. This is a genuine, previously unaddressed input-distribution mismatch that could
plausibly explain a broad, fairly uniform degradation across cell types (the exact
signature the Stage 1 full-run diagnostic found) better than the gene-mapping-completeness
or masked-pooling hypotheses (both already tested), since a systematic scale shift
relative to training data affects every gene's representation, not a subset of them.

adata.layers['counts'] holds raw counts, so the real normalization can be applied
directly rather than guessed at.

Moved here from local WSL2 dev after two reproducible CUDA crashes there (likely WSL2 GPU
passthrough instability after sustained contention, not a code issue -- Juno's native
Linux GPU nodes have had zero crashes all session).

Run (inside the container, same bind pattern as run_zero_shot_batch_integration.py):
    apptainer exec --nv --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        --bind /groups/tprice/pipelines:/groups/tprice/pipelines \\
        --bind /scratch/juno/$USER:/scratch/juno/$USER \\
        --bind $(pwd)/pipeline:/pipeline --bind $(pwd)/experiments:/experiments \\
        container/sclong_v1.0.0.sif \\
        python /experiments/experiment_renormalization.py \\
          --data-root /groups/tprice/pipelines/references/sclong \\
          --n-cells 500 --batch-size 48
"""
import argparse
import pickle
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import scanpy as sc
import scib
import scipy.sparse as sparse
import torch
from tqdm import tqdm

sys.path.insert(0, "/pipeline")  # get_cell_emb.py -- same module the shipped pipeline uses
from performer_pytorch_cont.ding_models import DualEncoderSCFM  # from PYTHONPATH=/opt/scLong
from get_cell_emb import attach_get_cell_emb


def reindex_tensor_universal(tensor, index_positions, dim, filler="0", device="cpu"):
    index_tensor = torch.tensor(index_positions, device=device)
    tensor_shape = list(tensor.shape)
    tensor_shape[dim] = len(index_positions)
    index_shape = [1] * len(tensor_shape)
    index_shape[dim] = len(index_positions)
    index_tensor = index_tensor.view(*index_shape).expand(*tensor_shape)
    padder_shape = deepcopy(tensor_shape)
    padder_shape[dim] = 1
    padder = torch.zeros(padder_shape, dtype=tensor.dtype, device=device)
    expanded_tensor = torch.cat((tensor, padder), dim=dim)
    return torch.gather(expanded_tensor, dim, index_tensor)


def scLong_renormalize(counts):
    """x -> log(x/10000 + 1), then rescale each cell so its max value is exactly 10.
    counts: dense or sparse (n_cells, n_genes) raw-count matrix."""
    if sparse.issparse(counts):
        counts = counts.toarray()
    x = np.log(counts / 10000.0 + 1.0)
    row_max = x.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1.0  # avoid div-by-zero for all-zero rows
    x = x * (10.0 / row_max)
    return x.astype(np.float32)


def eval_asw(adata, embedding_key):
    if len(adata.obs["batch"].unique()) <= 1:
        return float("nan")
    return scib.metrics.silhouette_batch(
        adata, "batch", "celltype", embed=embedding_key, metric="euclidean",
        return_all=False, verbose=False,
    )


def embed(model, X, scfm_index_positions, batch_size, device, label=""):
    out = []
    for i in tqdm(range(0, X.shape[0], batch_size), desc=label):
        with torch.no_grad():
            x_batch = X[i:i + batch_size, :]
            x = torch.tensor(x_batch).to(torch.float32).to(device)
            x_scfm = reindex_tensor_universal(x, scfm_index_positions, dim=1, filler="0", device=device)
            cell_emb = model.get_cell_emb(x_scfm)
            out.append(cell_emb.cpu().numpy())
    return np.concatenate(out, axis=0)


def resolve_gene_mapping_dir(gene_meta, version):
    if version is None:
        current_ptr = gene_meta / "mappings" / "CURRENT"
        if current_ptr.exists():
            version = current_ptr.read_text().strip()
        else:
            return None  # caller falls back to the legacy flat file
    return gene_meta / "mappings" / version


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="/data")
    p.add_argument("--gene-mapping-version", type=str, default=None,
                    help="Named version under gene_meta/mappings/. Defaults to gene_meta/mappings/CURRENT")
    p.add_argument("--n-cells", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=48)
    args = p.parse_args()

    root = Path(args.data_root)
    gene_meta = root / "gene_meta"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    with open(gene_meta / "gocont_4096_48m_pretrain_1b_mix.pkl", "rb") as f:
        hp = pickle.load(f)
    hp["gene2vec_file"] = str(gene_meta / "selected_gene2vec_27k.npy")

    model = DualEncoderSCFM(**hp).to(device)
    ckpt = torch.load(str(root / "checkpoints" / "gocont_4096_48m_pretrain_1b_mix_2024-02-05_16-23-37.pth"),
                       map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    attach_get_cell_emb(model)
    print("model loaded", flush=True)

    mapping_dir = resolve_gene_mapping_dir(gene_meta, args.gene_mapping_version)
    mapping_path = (mapping_dir / "human_symbol_to_ens.txt") if mapping_dir else (gene_meta / "human_symbol_to_ens.txt")
    print(f"gene mapping: {mapping_path}", flush=True)
    with open(mapping_path) as f:
        symbol2ens = {l.split(",")[0]: l.split(",")[1].rstrip("\n") for l in f if l.split(",")[1].rstrip("\n") != "unknown"}

    adata = sc.read_h5ad(str(root / "datasets" / "pancreas_scib.h5ad"))
    sc.pp.filter_cells(adata, min_genes=10)
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.subsample(adata, n_obs=min(args.n_cells, adata.n_obs), copy=False)
    print(f"subsampled to {adata.n_obs} cells", flush=True)

    input_genes = adata.var.index.tolist()
    input_genes_mapped = [symbol2ens.get(s, f"unknown{i}") for i, s in enumerate(input_genes)]
    input_mapping = {idx: i for i, idx in enumerate(input_genes_mapped)}
    input_gene_num = len(input_genes_mapped)

    with open(gene_meta / "selected_genes_27k.txt") as f:
        scfm_genes = [l.rstrip("\n") for l in f]
    scfm_genes_pad = scfm_genes + ["PAD"]
    scfm_index_positions = [input_mapping.get(idx, input_gene_num) for idx in scfm_genes_pad]

    X_as_shipped = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    row_max_shipped = X_as_shipped.max(axis=1)
    print(f"as-shipped .X per-cell max: mean={row_max_shipped.mean():.2f} std={row_max_shipped.std():.2f}", flush=True)

    X_renorm = scLong_renormalize(adata.layers["counts"][: adata.n_obs, :])
    row_max_renorm = X_renorm.max(axis=1)
    print(f"renormalized per-cell max: mean={row_max_renorm.mean():.4f} std={row_max_renorm.std():.6f} "
          f"(should be ~10.0 / ~0.0)", flush=True)

    emb_shipped = embed(model, X_as_shipped, scfm_index_positions, args.batch_size, device, label="as-shipped")
    emb_renorm = embed(model, X_renorm, scfm_index_positions, args.batch_size, device, label="renormalized")

    adata.obsm["X_shipped"] = emb_shipped
    adata.obsm["X_renorm"] = emb_renorm

    asw_shipped = eval_asw(adata, "X_shipped")
    asw_renorm = eval_asw(adata, "X_renorm")
    print(f"\nRESULT n_cells={adata.n_obs} batch_asw_as_shipped={asw_shipped:.4f} "
          f"batch_asw_renormalized={asw_renorm:.4f}", flush=True)


if __name__ == "__main__":
    main()

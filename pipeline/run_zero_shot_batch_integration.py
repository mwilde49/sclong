"""
Patched, runnable version of upstream scLong's zero-shot-batch/scLong_zero_shot.py —
ROADMAP.md Stage 1: reproduce batch ASW = 0.96 on the scIB pancreas dataset.

Fixes applied relative to upstream (each documented inline at the fix site):
  1. model.get_cell_emb() reconstructed — see get_cell_emb.py (upstream: doesn't exist).
  2. Checkpoint loaded with map_location='cpu' first, only the state dict moved to GPU
     (upstream: torch.load(..., map_location='cuda') risks loading optimizer/scheduler
     state directly onto the GPU before we even know if it's present — confirmed present:
     keys=['epoch','model_state_dict','optimizer_state_dict','scheduler_state_dict',
     'scaler_state_dict','losses']).
  3. Sparse .h5ad input densified before torch.tensor() (upstream: crashes on CSR X).
  4. All hardcoded personal absolute paths (/home/ding.bai/..., /l/users/ding.bai/...)
     replaced with CLI arguments.
  5. batch_size promoted from a hardcoded literal to a CLI flag.
  6. pickle5 replaced with the stdlib pickle (protocol 5 has been in stdlib since 3.8).

Validated locally (WSL2, RTX 5070 Ti, 12GB VRAM) prior to containerizing:
  - Full pipeline runs end-to-end against the real checkpoint.
  - n=100 cells: ASW=0.8794 (n=20: 0.7501) — trending toward the paper's 0.9561 as sample
    size grows, consistent with a correct pipeline; the full 16,382-cell run is the real
    comparison point.
  - At batch_size=4, VRAM is fully saturated on a 12GB card (100% util, ~95MB free) with a
    per-batch time that trends upward over the run — memory-pressure allocator overhead,
    not just noise. This is the reason this container exists: Juno's A30 (24GB)/H100 (80GB)
    remove that ceiling and allow a much larger batch size.

Expects, under --data-root (default /data — bind-mount your host data dir there):
  gene_meta/selected_gene2vec_27k.npy
  gene_meta/gocont_4096_48m_pretrain_1b_mix.pkl
  gene_meta/selected_genes_27k.txt
  gene_meta/human_symbol_to_ens.txt        (reconstructed via Ensembl BioMart — not from authors)
  checkpoints/gocont_4096_48m_pretrain_1b_mix_2024-02-05_16-23-37.pth
  datasets/pancreas_scib.h5ad              (author-provided copy — has both 'tech' and 'batch'
                                             obs columns; use 'batch' — see fix-site comment below)

Run (inside the container):
    apptainer exec --nv --bind /path/to/data:/data sclong_v1.0.0.sif \
        python /pipeline/run_zero_shot_batch_integration.py --data-root /data
"""
import argparse
import pickle
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
from tqdm import tqdm

from performer_pytorch_cont.ding_models import DualEncoderSCFM  # from PYTHONPATH=/opt/scLong

from eval_utils import evaluate, print_sys
from get_cell_emb import attach_get_cell_emb


def reindex_tensor_universal(tensor, index_positions, dim, filler="0", device="cpu"):
    """Unmodified from upstream — gene-panel reindexing logic, no bugs found here."""
    index_tensor = torch.tensor(index_positions, device=device)
    tensor_shape = list(tensor.shape)
    tensor_shape[dim] = len(index_positions)
    index_shape = [1] * len(tensor_shape)
    index_shape[dim] = len(index_positions)
    index_tensor = index_tensor.view(*index_shape).expand(*tensor_shape)

    if filler == "0":
        padder_shape = deepcopy(tensor_shape)
        padder_shape[dim] = 1
        padder = torch.zeros(padder_shape, dtype=tensor.dtype, device=device)
    elif filler == "mean":
        padder = tensor.mean(dim=dim, keepdim=True)
    else:
        raise ValueError("filler should be 0 or mean!")
    expanded_tensor = torch.cat((tensor, padder), dim=dim)
    return torch.gather(expanded_tensor, dim, index_tensor)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="/data",
                    help="Root containing gene_meta/, checkpoints/, datasets/ (bind-mount target)")
    p.add_argument("--output-dir", type=str, default="/data/results")
    p.add_argument("--batch-size", type=int, default=4, help="upstream hardcoded this to 4")
    p.add_argument("--n-cells", type=str, default="all")
    p.add_argument("--output-key", type=str, default="merged_decodings")
    p.add_argument("--progress-every", type=int, default=25,
                    help="print a PROGRESS line to stdout every N batches (for log tailing)")
    args = p.parse_args()

    root = Path(args.data_root)
    gene_meta = root / "gene_meta"
    ckpt_dir = root / "checkpoints"
    data_dir = root / "datasets"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_sys(f"device: {device}")

    n_cells = 1e10 if args.n_cells == "all" else int(args.n_cells)

    hyper_params_path = gene_meta / "gocont_4096_48m_pretrain_1b_mix.pkl"
    gene2vec_path = gene_meta / "selected_gene2vec_27k.npy"
    ckpt_path = ckpt_dir / "gocont_4096_48m_pretrain_1b_mix_2024-02-05_16-23-37.pth"
    scfm_genes_path = gene_meta / "selected_genes_27k.txt"
    symbol2ens_path = gene_meta / "human_symbol_to_ens.txt"
    adata_path = data_dir / "pancreas_scib.h5ad"

    for path in [hyper_params_path, gene2vec_path, ckpt_path, scfm_genes_path, symbol2ens_path, adata_path]:
        if not path.exists():
            print_sys(f"MISSING: {path} — check the --data-root bind-mount.")
            sys.exit(1)

    with open(hyper_params_path, "rb") as f:
        scfm_hyper_params = pickle.load(f)  # fix #6: stdlib pickle, not pickle5
    scfm_hyper_params["gene2vec_file"] = str(gene2vec_path)
    print_sys(scfm_hyper_params)

    print("PROGRESS stage=instantiate_model", flush=True)
    t_stage = time.time()
    model = DualEncoderSCFM(**scfm_hyper_params).to(device)
    print(f"PROGRESS stage=instantiate_model done elapsed={time.time()-t_stage:.1f}s", flush=True)

    # fix #2: load to CPU first so we control exactly what lands on the GPU
    print("PROGRESS stage=load_checkpoint note=~12GB_from_disk_this_is_the_slow_step", flush=True)
    t_stage = time.time()
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    print(f"PROGRESS stage=load_checkpoint done elapsed={time.time()-t_stage:.1f}s "
          f"keys={list(ckpt.keys())}", flush=True)
    t_stage = time.time()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    vram_note = f"free_vram_gb={torch.cuda.mem_get_info()[0]/1e9:.2f}" if device.type == "cuda" else ""
    print(f"PROGRESS stage=move_to_gpu done elapsed={time.time()-t_stage:.1f}s {vram_note}", flush=True)

    # fix #1
    attach_get_cell_emb(model)

    with open(symbol2ens_path, "r") as f:
        symbol2ens = {
            line.split(",")[0]: line.split(",")[1].rstrip("\n")
            for line in f.readlines() if line.split(",")[1] != "unknown"
        }

    adata = sc.read_h5ad(str(adata_path))
    sc.pp.filter_cells(adata, min_genes=10)
    sc.pp.filter_genes(adata, min_cells=10)

    n_cells = int(np.min((adata.n_obs, n_cells)))
    if adata.n_obs > n_cells:
        print_sys(f"adata has {adata.n_obs} cells. Taking a subset of {n_cells}.")
        sc.pp.subsample(adata, n_obs=n_cells, copy=False)

    input_genes = adata.var.index.tolist()
    input_genes = [symbol2ens.get(symbol, f"unknown{i}") for i, symbol in enumerate(input_genes)]
    input_mapping = {idx: i for i, idx in enumerate(input_genes)}
    input_gene_num = len(input_genes)

    with open(scfm_genes_path, "r") as f:
        scfm_genes = [line.rstrip("\n") for line in f.readlines()]
        intersect_n = len(np.intersect1d(input_genes, scfm_genes))
        print_sys(f"target and scfm genes intersect: {intersect_n}")  # key sanity check, see ROADMAP.md §5

    scfm_genes_pad = scfm_genes + ["PAD"]
    scfm_seq_len = len(scfm_genes_pad)
    scfm_mapping = {idx: i for i, idx in enumerate(scfm_genes_pad)}
    scfm_index_positions = [input_mapping.get(idx, input_gene_num) for idx in scfm_genes_pad]

    cell_embeddings = []
    embedding_key = "X_scLong"
    batch_starts = list(range(0, n_cells, args.batch_size))
    n_batches = len(batch_starts)
    print(f"PROGRESS stage=embed start n_cells={n_cells} n_batches={n_batches} batch_size={args.batch_size}",
          flush=True)
    t_embed_start = time.time()
    for b_idx, i in enumerate(tqdm(batch_starts, file=sys.stderr)):
        with torch.no_grad():
            x_batch = adata.X[i:i + args.batch_size, :]
            if hasattr(x_batch, "toarray"):  # fix #3: densify sparse CSR before torch.tensor()
                x_batch = x_batch.toarray()
            x = torch.tensor(x_batch).to(torch.float32).to(device)
            x_scfm = reindex_tensor_universal(x, scfm_index_positions, dim=1, filler="0", device=device)
            cell_emb = model.get_cell_emb(x_scfm, output_key=args.output_key)
            cell_embeddings.append(cell_emb.cpu().numpy())

        if (b_idx + 1) % args.progress_every == 0 or (b_idx + 1) == n_batches:
            elapsed = time.time() - t_embed_start
            rate = elapsed / (b_idx + 1)  # seconds per batch, running average
            remaining_batches = n_batches - (b_idx + 1)
            eta_s = remaining_batches * rate
            pct = 100.0 * (b_idx + 1) / n_batches
            print(
                f"PROGRESS stage=embed batch={b_idx + 1}/{n_batches} cells={min((b_idx + 1) * args.batch_size, n_cells)}/{n_cells} "
                f"pct={pct:.1f}% elapsed={elapsed:.0f}s rate={rate:.2f}s/batch eta={eta_s:.0f}s "
                f"eta_finish={time.strftime('%H:%M:%S', time.localtime(time.time() + eta_s))}",
                flush=True,
            )

    cell_embeddings = np.concatenate(cell_embeddings, axis=0)
    adata.obsm[embedding_key] = cell_embeddings
    print(f"PROGRESS stage=embed done elapsed={time.time()-t_embed_start:.0f}s shape={cell_embeddings.shape}",
          flush=True)
    print_sys(f"cell_embeddings: {cell_embeddings.shape}")

    # batch_key="batch" matches upstream exactly. Worth the history: the public figshare
    # mirror of this dataset lacks a "batch" column (only has the finer-grained "tech",
    # 9 categories) which briefly looked like an upstream bug; the author-provided copy
    # (data/datasets/pancreas_scib.h5ad) has both, and "batch" is confirmed to be the
    # correct coarser 6-category grouping (celseq, celseq2, fluidigmc1, smartseq2, indrop,
    # smarter — the four inDrop sub-runs merged into one) matching the paper's "6 batches".
    print("PROGRESS stage=evaluate start", flush=True)
    t_stage = time.time()
    res = evaluate(
        adata_=adata, batch_key="batch", label_key=["celltype"], embedding_key=embedding_key,
        res_path=str(out_dir / "scLong_batch_cell_emb_mode.csv"),
    )
    print(f"PROGRESS stage=evaluate done elapsed={time.time()-t_stage:.0f}s", flush=True)
    print_sys(res)
    print("RESULT " + " | ".join(f"{row.metric}[{row.label}]={row.value:.4f}" for row in res.itertuples()), flush=True)
    print("RESULT target_from_paper batch_ASW=0.9561 (Nat Commun 2026, zero-shot batch integration)", flush=True)


if __name__ == "__main__":
    main()

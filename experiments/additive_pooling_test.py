"""
Tests the ADDITIVE get_cell_emb reconstruction against the two already-refuted
alternatives, on a single shared forward pass per batch (cheap: merged_decodings is
already produced either way, exp_to_out adds negligible compute):

  (A) sum_pool  = merged_decodings.sum(-1)                    current pipeline default
  (B) exp_only  = model.exp_to_out(merged_decodings).squeeze(-1)  refuted: ~0.70, matches
                                                                    the paper's own "Raw"
                                                                    baseline (see ROADMAP.md §14.2)
  (C) additive  = sum_pool + exp_only                          the live candidate
  (D) additive, measured-only-pooled (only if --pooling measured-only)

Ported into the repo from an ephemeral scratchpad harness a Round-2/3 research agent
built during the wf_d1c898a4-2d1 workflow (see ROADMAP.md §14.2) — fixed to be
CLI-parameterized and batched (the original was single-batch, n=20 only, would OOM at
n=500+), and to inherit --bio-conservation from eval_utils.py automatically (no code
change needed there — this script's evaluate() call now gets NMI/ARI/isolated-F1 for
free, closing the exact gap Round 3 flagged as a required fix before running).

IMPORTANT — before trusting any comparison to prior numbers, read ROADMAP.md §14.2:
the citation motivating this hypothesis (a "paper quote" about concatenation +
sum-pooling) is very likely NOT real -- it doesn't appear in either extracted PDF, and a
WebFetch that seemed to confirm it is judged a probable hallucination. This script is
still worth running (concat-then-sum is a well-defined, cheap thing to test on pure
engineering grounds), but do not present a positive result as "matching the paper's own
described method" without an independent, human, visual check of the actual PMC PDF
first.

Run (inside the container):
    apptainer exec --nv --cleanenv --env PYTHONNOUSERSITE=1 \
        --bind /groups/tprice/pipelines:/groups/tprice/pipelines \
        --bind /scratch/juno/$USER:/scratch/juno/$USER \
        --bind $(pwd)/pipeline:/pipeline --bind $(pwd)/experiments:/experiments \
        container/sclong_v1.0.0.sif \
        python /experiments/additive_pooling_test.py \
          --data-root /groups/tprice/pipelines/references/sclong \
          --output-dir /scratch/juno/$USER/sclong_results/additive_pooling \
          --n-cells 20 --seed 42 \
          --gene-mapping-version v2_hgnc_plus_biomart_synonyms --pooling measured-only

Escalation per ROADMAP.md §14.2 / the workflow's own decision rule: only move from
n=20 -> n=500 -> full scale if the additive variant lands meaningfully above its
same-config sum_pool baseline AND bio-conservation (NMI/ARI/isolated-F1) doesn't fall
alongside it. If it lands close to either the ~0.39 (gene-axis) or ~0.70 (exp_to_out
alone) range instead, treat get_cell_emb pooling-axis exploration as exhausted.
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
sys.path.insert(0, "/opt/scLong")  # container PYTHONPATH; harmless no-op if already there

from performer_pytorch_cont.ding_models import DualEncoderSCFM  # noqa: E402
from eval_utils import evaluate, print_sys  # noqa: E402
from run_zero_shot_batch_integration import (  # noqa: E402
    reindex_tensor_universal, resolve_gene_mapping_path,
)

p = argparse.ArgumentParser()
p.add_argument("--data-root", type=str, default="/data")
p.add_argument("--output-dir", type=str, default="/data/results/additive_pooling")
p.add_argument("--batch-size", type=int, default=32)
p.add_argument("--n-cells", type=str, default="20")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--gene-mapping-version", type=str, default="v2_hgnc_plus_biomart_synonyms")
p.add_argument("--pooling", choices=["full", "measured-only"], default="full")
args = p.parse_args()

root = Path(args.data_root)
gene_meta = root / "gene_meta"
out_dir = Path(args.output_dir)
out_dir.mkdir(parents=True, exist_ok=True)
n_cells = 1e10 if args.n_cells == "all" else int(args.n_cells)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print_sys(f"device={device}")

t0 = time.time()
with open(gene_meta / "gocont_4096_48m_pretrain_1b_mix.pkl", "rb") as f:
    hp = pickle.load(f)
hp["gene2vec_file"] = str(gene_meta / "selected_gene2vec_27k.npy")

model = DualEncoderSCFM(**hp).to(device)
ckpt = torch.load(str(gene_meta.parent / "checkpoints" /
                       "gocont_4096_48m_pretrain_1b_mix_2024-02-05_16-23-37.pth"),
                   map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
model.to(device)
for param in model.parameters():
    param.requires_grad = False
model.eval()
print_sys(f"model loaded, elapsed={time.time()-t0:.1f}s")

symbol2ens_path = resolve_gene_mapping_path(gene_meta, args.gene_mapping_version)
with open(symbol2ens_path, "r") as f:
    symbol2ens = {
        line.split(",")[0]: line.split(",")[1].rstrip("\n")
        for line in f.readlines() if line.split(",")[1] != "unknown"
    }

adata = sc.read_h5ad(str(root / "datasets" / "pancreas_scib.h5ad"))
sc.pp.filter_cells(adata, min_genes=10)
sc.pp.filter_genes(adata, min_cells=10)

n_cells = int(np.min((adata.n_obs, n_cells)))
if adata.n_obs > n_cells:
    print_sys(f"subsampling to {n_cells} cells, seed={args.seed}")
    sc.pp.subsample(adata, n_obs=n_cells, copy=False, random_state=args.seed)

input_genes = adata.var.index.tolist()
input_genes = [symbol2ens.get(symbol, f"unknown{i}") for i, symbol in enumerate(input_genes)]
input_mapping = {idx: i for i, idx in enumerate(input_genes)}
input_gene_num = len(input_genes)

with open(gene_meta / "selected_genes_27k.txt", "r") as f:
    scfm_genes = [line.rstrip("\n") for line in f.readlines()]
scfm_genes_pad = scfm_genes + ["PAD"]
scfm_index_positions = [input_mapping.get(idx, input_gene_num) for idx in scfm_genes_pad]

measured_mask = None
if args.pooling == "measured-only":
    measured_mask = torch.tensor(
        [pos != input_gene_num for pos in scfm_index_positions], dtype=torch.bool, device=device
    )
    print_sys(f"pooling=measured-only: {int(measured_mask.sum())}/{len(scfm_genes_pad)} positions kept")

X = adata.X
if hasattr(X, "toarray"):
    X = X.toarray()
X = np.asarray(X)

variant_names = ["A_sum_pool", "B_exp_to_out_alone", "C_additive"]
embeddings = {name: [] for name in variant_names}

batch_starts = list(range(0, n_cells, args.batch_size))
print_sys(f"running {len(batch_starts)} batches, batch_size={args.batch_size}, n_cells={n_cells}")
t_embed = time.time()
for i in tqdm(batch_starts, file=sys.stderr):
    with torch.no_grad():
        x_batch = X[i:i + args.batch_size, :]
        x = torch.tensor(x_batch).to(torch.float32).to(device)
        x_scfm = reindex_tensor_universal(x, scfm_index_positions, dim=1, filler="0", device=device)
        out = model(x_scfm, return_encodings=True)
        merged_decodings = out["merged_decodings"]  # (B, 27875, 200)
        exp_out = model.exp_to_out(merged_decodings).squeeze(-1)  # (B, 27875)
        sum_pool = merged_decodings.sum(dim=-1)  # (B, 27875)
        additive = sum_pool + exp_out  # (B, 27875)

        for name, emb in zip(variant_names, [sum_pool, exp_out, additive]):
            if measured_mask is not None:
                emb = emb[:, measured_mask]
            embeddings[name].append(emb.cpu().numpy())
print_sys(f"forward passes done, elapsed={time.time()-t_embed:.1f}s")

results = {}
for name in variant_names:
    emb_arr = np.concatenate(embeddings[name], axis=0)
    embedding_key = "X_test"
    adata.obsm[embedding_key] = emb_arr
    res = evaluate(
        adata_=adata, batch_key="batch", label_key=["celltype"], embedding_key=embedding_key,
        res_path=str(out_dir / f"additive_pooling__{name}__n{n_cells}_seed{args.seed}_pool-{args.pooling}.csv"),
        bio_conservation=True,
    )
    row = res[res["metric"] == "ASW_label/batch"]
    asw = float(row["value"].values[0]) if len(row) else float("nan")
    results[name] = res.set_index("metric")["value"].to_dict()
    print(f"RESULT {name}: {results[name]}", flush=True)

print(f"\n=== SUMMARY (n={n_cells}, seed={args.seed}, "
      f"gene_mapping={args.gene_mapping_version}, pooling={args.pooling}) ===")
for name, metrics in results.items():
    print(f"  {name}: " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
print("\nDecision rule (ROADMAP.md §14.2): only escalate n if C's ASW clears A's "
      "same-config baseline by more than small-n noise AND C's bio-conservation "
      "doesn't fall relative to A's.")
print("DONE", flush=True)

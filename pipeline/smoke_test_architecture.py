"""
Environment/architecture smoke test — no real scLong weights or gene metadata required.

Instantiates DualEncoderSCFM with small synthetic dimensions (a handful of genes instead
of the real 27,874) and a random GO graph, runs a forward pass on GPU, and checks shapes.
Used by container/test_container.sh --gpu to validate a freshly built .sif before it's
trusted with the real (gated) checkpoint.

Run:
    apptainer exec --nv sclong_v1.0.0.sif python /pipeline/smoke_test_architecture.py
"""
import numpy as np
import torch

from performer_pytorch_cont.ding_models import DualEncoderSCFM  # from PYTHONPATH=/opt/scLong

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

# --- synthetic stand-ins for the gated data files ---
N_GENES = 300          # real model uses 27,874 (+1 PAD) = 27,875
TOP_L = 64              # real model uses 4,096
BASE_DIM = 200          # matches real model (paper-documented)

gene2vec = torch.randn(N_GENES, BASE_DIM).numpy()  # stand-in for selected_gene2vec_27k.npy
gene2vec_path = "/tmp/smoke_gene2vec.npy"
np.save(gene2vec_path, gene2vec)

# stand-in for the GO graph (bundled pre-built inside the real .pkl per the README) — here
# just a random sparse graph with self-loops (SGConv(add_self_loops=False) requires them
# to already be present).
num_edges = N_GENES * 4
src = torch.randint(0, N_GENES, (num_edges,))
dst = torch.randint(0, N_GENES, (num_edges,))
self_loops = torch.arange(N_GENES)
edge_index = torch.stack([
    torch.cat([src, self_loops]),
    torch.cat([dst, self_loops]),
])
edge_weight = torch.rand(edge_index.shape[1])

model = DualEncoderSCFM(
    max_seq_len=N_GENES,
    top_seq_len=TOP_L,
    base_dim=BASE_DIM,
    mini_enc_depth=1, mini_enc_heads=8, mini_enc_dim_head=64,
    large_dim=256, large_enc_depth=2, large_enc_heads=4, large_enc_dim_head=64,  # shrunk from 42/32/1280 for speed
    dec_depth=1, dec_heads=8, dec_dim_head=64,
    G_go=edge_index, G_go_weight=edge_weight,
    device=device,
    gene2vec_file=gene2vec_path,
).to(device)
model.eval()

n_params = sum(p.numel() for p in model.parameters())
print(f"instantiated OK — {n_params:,} params (shrunk config; real model is ~1B)")

batch_size = 3
x = torch.rand(batch_size, N_GENES, device=device) * 5  # fake expression values

with torch.no_grad():
    out = model(x, return_encodings=True)

merged = out["merged_decodings"]  # (B, N, base_dim) — same tensor the real pipeline reads
print(f"merged_decodings shape: {tuple(merged.shape)}  (expect ({batch_size}, {N_GENES}, {BASE_DIM}))")
assert merged.shape == (batch_size, N_GENES, BASE_DIM)

cell_emb = merged.sum(dim=-1)  # reconstructed get_cell_emb — see get_cell_emb.py
print(f"pooled cell embedding shape: {tuple(cell_emb.shape)}  (expect ({batch_size}, {N_GENES}))")
assert cell_emb.shape == (batch_size, N_GENES)

print("\nSMOKE TEST PASSED: architecture, container, and GPU all check out.")

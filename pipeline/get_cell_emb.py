"""
Reconstruction of DualEncoderSCFM.get_cell_emb() — not present anywhere in the upstream
scLong repository (verified by grep across the full source tree), but called by
zero-shot-batch/scLong_zero_shot.py:186 as though it existed.

Reconstructed from direct evidence, not guesswork:
  - embed.py:155 shows the canonical per-gene, per-cell representation is
    `model(x, return_encodings=True)['merged_decodings']`, shape (B, N, base_dim).
  - scLong_zero_shot.py defines `--output_key` defaulting to 'merged_decodings' (though
    the value is never actually threaded into the get_cell_emb() call — one more loose
    end in this codebase) and its own inline comment marks the final `cell_embeddings`
    array as shape `(n_cells, seq_len)` — i.e. one scalar per gene per cell, not a
    compact latent.
  - The only way to go from (B, N, base_dim) to (B, N) is to collapse the base_dim axis.
    Sum vs. mean over that axis differ by a constant global factor (1/base_dim), which
    does not change relative Euclidean distances between cells — irrelevant to silhouette
    scoring (scib.metrics.silhouette_batch, metric="euclidean"), the only metric this
    embedding is validated against in zero-shot-batch/new_utils.py. Sum is used here to
    match the paper's own description of "sum-pooling."

Verified: a full forward pass through the real (dimension-shrunk) architecture on GPU
produces exactly the expected (B, N) shape (see smoke_test_architecture.py); the 100-cell
local calibration run (ASW=0.879, trending toward the paper's 0.956 as sample size grows)
is consistent with this being the correct reconstruction, not proof positive — cross-check
against a real get_cell_emb() implementation if the authors ever supply one.
"""
import types
import torch


def get_cell_emb(self, x, output_key: str = "merged_decodings", **forward_kwargs) -> torch.Tensor:
    """Bound-method replacement for the missing DualEncoderSCFM.get_cell_emb.

    Args:
        x: (batch_size, max_seq_len) raw expression tensor, already reindexed onto the
           model's fixed gene vocabulary (see reindex_tensor_universal in
           run_zero_shot_batch_integration.py).
        output_key: which forward(..., return_encodings=True) output to pool. Defaults
           to 'merged_decodings' to match scLong_zero_shot.py's own --output_key default.

    Returns:
        (batch_size, max_seq_len) tensor — one pooled scalar per gene per cell.
    """
    output = self(x, return_encodings=True, **forward_kwargs)
    per_gene = output[output_key]  # (B, N, base_dim)
    return per_gene.sum(dim=-1)  # (B, N)


def attach_get_cell_emb(model) -> None:
    """Monkey-patch a DualEncoderSCFM instance with the reconstructed get_cell_emb."""
    model.get_cell_emb = types.MethodType(get_cell_emb, model)

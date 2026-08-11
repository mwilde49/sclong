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

**UPDATE 2026-08-11 (ROADMAP.md §16): this reconstruction is real and useful (best of the
model-vocab-space pooling modes this pipeline has tested, see ablation grid results), but
it is NOT what the paper's Equation (11) actually specifies. A colleague's independently
written and run notebook (sclong_completed.ipynb), executed against the SAME checkpoint
this pipeline uses, implements the paper's literal recipe -- concatenate the reconstructed
expression onto E, sum-pool, but only AFTER reindexing the result back onto the dataset's
own gene ordering (filler='mean') rather than staying in the model's internal 27,875-
position vocabulary -- and scored ASW=0.9212, well above this pipeline's own best of
0.8550. See get_cell_emb_eq11_reindexed() below for the ported implementation, and
run_zero_shot_batch_integration.py's `--pooling reindex-to-dataset` to use it end to end.
Do not treat `get_cell_emb()` below as "the" reconstruction going forward -- it is one of
now four candidate pooling mechanisms, and the reindex-to-dataset one is the only one with
a real, independently-produced, near-paper-target result behind it.**
"""
import types
import torch

from reindex_utils import reindex_tensor_universal


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


def get_cell_emb_eq11_reindexed(model, x_scfm, input_index_positions, device="cpu",
                                 decode_key: str = "merged_decodings") -> torch.Tensor:
    """The paper's literal Equation (11) (Methods, "Zero-shot batch integration"),
    including the reindex-back-to-dataset-gene-space step this pipeline never implemented
    before 2026-08-11 -- ported from a colleague's independently written and run notebook
    (sclong_completed.ipynb cells 8/11), which scored ASW=0.9212 on the exact same
    checkpoint this pipeline uses. See ROADMAP.md §16 for the full comparison and
    experiments/additive_pooling_test.py's variant D for a direct A/B/C/D bake-off.

    Differs from get_cell_emb() above in two ways, both required to match Eq. 11:
      1. Concatenates the reconstructed expression x' = exp_to_out(E) onto E BEFORE
         pooling (get_cell_emb() above only ever pools a single output_key tensor, never
         the reconstruction head's own output).
      2. Reindexes the combined (still-in-model-vocabulary-space) tensor back onto the
         DATASET's own gene ordering with filler='mean' -- genes the model's 27,874-
         symbol vocabulary doesn't cover get the mean of the genes it does cover, not
         zero (this pipeline's --pooling full) and not dropped (--pooling
         measured-only) -- BEFORE summing, not after. Every other pooling mode in this
         pipeline stays in the model's own internal gene ordering throughout; this is the
         one candidate that instead lands on the dataset's own gene space.

    Args:
        model: DualEncoderSCFM instance with weights loaded, in eval mode.
        x_scfm: (batch, 27875) input already reindexed onto the model's own gene
            vocabulary with filler='0' -- identical input to every other pooling mode.
        input_index_positions: length-N list (N = the dataset's own post-QC gene count,
            in the dataset's own column order) mapping each dataset gene to its position
            in the model's scfm_genes_pad vocabulary, or len(scfm_genes_pad) if the model
            doesn't cover that gene at all. Built in run_zero_shot_batch_integration.py by
            reusing the same scfm_mapping dict already constructed there for the
            opposite-direction (dataset -> model) reindex -- see that file for the exact
            one-line construction.
        device: torch device string/object, matching x_scfm's device.
        decode_key: which forward(..., return_encodings=True) output holds E;
            'merged_decodings' matches both this project's own architecture audit
            (ding_models.py DualEncoderSCFM.forward()) and the source notebook's usage.

    Returns:
        (batch, N) tensor, N = len(input_index_positions) = the dataset's own gene
        count -- NOT max_seq_len like get_cell_emb() above. Output dimensionality here is
        tied to the dataset's biology (every column is a real, named gene, in the
        dataset's own order), not the model's fixed internal vocabulary.
    """
    output = model(x_scfm, return_encodings=True)
    dec = output[decode_key]                          # (B, 27875, base_dim) -- E
    exp_out = model.exp_to_out(dec)                    # (B, 27875, 1) -- x', same op as forward()
    combined = torch.cat([dec, exp_out], dim=-1)        # (B, 27875, base_dim+1) -- V, model gene space
    combined = reindex_tensor_universal(
        combined, input_index_positions, dim=1, filler="mean", device=device,
    )                                                    # (B, N, base_dim+1) -- V reindexed to dataset genes
    return combined.sum(dim=-1)                          # (B, N) -- v_i = sum_j V_ij, matches Eq. 11

"""
Shared gene-panel reindexing helper — factored out 2026-08-11 (ROADMAP.md §16) so
get_cell_emb.py can use filler='mean' without a circular import with
run_zero_shot_batch_integration.py (which needs get_cell_emb.py's attach_get_cell_emb,
and previously defined this function itself). Verbatim from upstream scLong's
zero-shot-batch/scLong_zero_shot.py lines 59-93 — no bugs found here on repeated
independent review, and independently re-verified byte-for-byte identical to the copy in
a colleague's own, separately-written notebook (sclong_completed.ipynb cell 8).

run_zero_shot_batch_integration.py re-exports reindex_tensor_universal from here (`from
reindex_utils import reindex_tensor_universal`) so experiments/additive_pooling_test.py's
existing `from run_zero_shot_batch_integration import reindex_tensor_universal` keeps
working unchanged.
"""
from copy import deepcopy

import torch


def reindex_tensor_universal(tensor, index_positions, dim, filler="0", device="cpu"):
    """Gather `tensor` along `dim` according to `index_positions`, after appending one
    extra filler row/column so any index_positions value == len(tensor along dim) (one
    past the end) resolves to that filler instead of a real position.

    filler='0': filler row is all zeros. Used throughout this pipeline to move a
        dataset's expression values ONTO the model's fixed 27,875-position gene
        vocabulary — genes the model doesn't know about get 0 input.
    filler='mean': filler row is the mean of `tensor` along `dim`. Used by
        get_cell_emb_eq11_reindexed (see get_cell_emb.py) to move the model's OWN
        per-gene output back OFF its internal vocabulary and onto a dataset's actual
        gene ordering — genes the model's vocabulary doesn't cover get a soft, non-zero
        imputed value instead of being silently dropped (measured-only pooling) or
        summed in as an arbitrary zero-input artifact (full pooling). This is the one
        piece of the paper's literal Equation (11) this pipeline never implemented before
        2026-08-11 — see ROADMAP.md §16.
    """
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

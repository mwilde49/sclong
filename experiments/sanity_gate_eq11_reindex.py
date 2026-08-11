"""
CPU-only correctness gate for get_cell_emb_eq11_reindexed (ROADMAP.md §16) -- no GPU, no
real model, no dataset. Follows this project's standing practice (see
sanity_gate_literal_norm.py, sanity_gate_shape_check.py) of proving a mechanism correct
on CPU, against an INDEPENDENT reference implementation, before spending any GPU time on
it.

Specifically exercises the one thing this pooling mode adds that nothing else in this
pipeline has ever used: filler='mean' reindexing -- a dataset gene not covered by the
model's vocabulary gets the MEAN of the genes the model does cover, not zero (--pooling
full) and not dropped (--pooling measured-only).

Run (no container needed -- pure torch, CPU):
    python experiments/sanity_gate_eq11_reindex.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from get_cell_emb import get_cell_emb_eq11_reindexed  # noqa: E402


class FakeModel:
    """Stand-in for DualEncoderSCFM -- implements only the two calls
    get_cell_emb_eq11_reindexed actually uses (forward(..., return_encodings=True) and
    .exp_to_out). Deterministic, no learned weights, so expected output is hand-checkable."""

    def __init__(self, merged_decodings):
        self._merged_decodings = merged_decodings

    def __call__(self, x, return_encodings=True):
        assert return_encodings
        return {"merged_decodings": self._merged_decodings}

    def exp_to_out(self, dec):
        return dec.sum(dim=-1, keepdim=True)


def brute_force_reference(merged_decodings, input_index_positions):
    """Independent, explicit-loop implementation -- shares no code with
    get_cell_emb_eq11_reindexed or reindex_utils.py -- so a bug shared between the two
    (a self-referential test) can't hide from this comparison."""
    B, M, _D = merged_decodings.shape
    exp_out = merged_decodings.sum(dim=-1, keepdim=True)
    combined = torch.cat([merged_decodings, exp_out], dim=-1)  # (B, M, D+1)
    mean_row = combined.mean(dim=1)  # (B, D+1) -- the filler='mean' row
    N = len(input_index_positions)
    out = torch.zeros(B, N)
    for b in range(B):
        for n, pos in enumerate(input_index_positions):
            row = mean_row[b] if pos >= M else combined[b, pos]
            out[b, n] = row.sum()
    return out


torch.manual_seed(42)
B, M, D = 2, 3, 2
merged_decodings = torch.tensor([
    [[1., 2.], [10., 20.], [100., 200.]],
    [[-1., 1.], [0., 0.], [5., -5.]],
])
assert merged_decodings.shape == (B, M, D)

# 4 dataset genes: two map to real, distinct model positions, one is deliberately
# unmapped (must hit the filler='mean' path -- index M is one-past-the-end), one maps to
# a third real position -- exercises both branches (real position / filler).
input_index_positions = [0, 2, M, 1]

model = FakeModel(merged_decodings)
x_dummy = torch.zeros(B, 10)  # unused by FakeModel -- get_cell_emb_eq11_reindexed never
                               # reads it directly, only passes it to model(x, ...)

got = get_cell_emb_eq11_reindexed(model, x_dummy, input_index_positions, device="cpu")
expected = brute_force_reference(merged_decodings, input_index_positions)

print("got     :", got.tolist())
print("expected:", expected.tolist())
max_abs_diff = (got - expected).abs().max().item()
print(f"max abs diff: {max_abs_diff:.2e}")

assert got.shape == (B, len(input_index_positions)), f"shape mismatch: {got.shape}"
assert torch.allclose(got, expected, atol=1e-5), "MISMATCH vs brute-force reference"

# Directly confirm the filler='mean' value landed exactly where expected (dataset gene
# index 2, the deliberately-unmapped one) -- the specific mechanism this pooling mode adds
# and the one thing no other pooling mode in this pipeline has ever exercised.
combined_b0 = torch.cat([merged_decodings[0], merged_decodings[0].sum(dim=-1, keepdim=True)], dim=-1)
filler_b0 = combined_b0.mean(dim=0)
assert torch.allclose(got[0, 2], filler_b0.sum(), atol=1e-5), "filler='mean' value not where expected"

print(
    "\nPASS -- get_cell_emb_eq11_reindexed matches an independent brute-force reference, "
    "including the filler='mean' branch for dataset genes the model's vocabulary doesn't "
    "cover. Safe to spend GPU time on --pooling reindex-to-dataset."
)

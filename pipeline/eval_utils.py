"""
Trimmed copy of upstream scLong's zero-shot-batch/new_utils.py evaluate()/
eval_scib_metrics()/print_sys() — the only three functions this pipeline needs.

Why a copy instead of importing the upstream module directly: new_utils.py does
`import pickle5 as pickle` at module level even though none of these three functions use
pickle at all. pickle5 does not build on Python >=3.12 (verified directly: `pip install
pickle5` fails with `Py_SIZE(self->stack) = x` — an lvalue-required compile error from the
Python 3.10+ removal of the Py_SIZE macro's assignability). Rather than pull in an
unbuildable, unused dependency, or hand-edit the vendored upstream mirror, the three
functions are reproduced here verbatim (only the pickle5 import dropped).
"""
import sys
from typing import Dict

import numpy as np
import pandas as pd
import scanpy as sc
import scib
from scanpy import AnnData


def print_sys(s):
    print(s, flush=True, file=sys.stderr)


def eval_scib_metrics(
    adata: AnnData,
    batch_key: str = "str_batch",
    label_key: str = "cell_type",
    embedding_key: str = "X_scGPT",
) -> Dict:
    if "neighbors" in adata.uns:
        print_sys(
            f"neighbors in adata.uns found \n {adata.uns['neighbors']} \nto make sure the "
            f"optimal clustering is calculated for the correct embedding, removing neighbors "
            f"from adata.uns.\nOverwriting calculation of neighbors with "
            f"sc.pp.neighbors(adata, use_rep={embedding_key})."
        )
        adata.uns.pop("neighbors", None)
        sc.pp.neighbors(adata, use_rep=embedding_key)
        print_sys(f"neighbors in adata.uns removed, new neighbors calculated: {adata.uns['neighbors']}")

    results_dict = dict()
    if len(adata.obs[batch_key].unique()) > 1:
        results_dict["ASW_label/batch"] = scib.metrics.silhouette_batch(
            adata, batch_key, label_key, embed=embedding_key, metric="euclidean",
            return_all=False, verbose=False,
        )

    print_sys("\n".join([f"{k}: {v:.4f}" for k, v in results_dict.items()]))
    results_dict = {k: v for k, v in results_dict.items() if not np.isnan(v)}
    return results_dict


def evaluate(adata_, batch_key=None, label_key: list = ["celltype"],
             embedding_key: str = "X_scLong", res_path: str = "") -> pd.DataFrame:
    met_df = pd.DataFrame(columns=["metric", "label", "value"])
    label_cols = [x for i, x in enumerate(label_key) if x not in label_key[:i]]
    label_cols = [x for x in label_cols if x in adata_.obs.columns]

    if len(label_cols) == 0:
        raise ValueError(f"No label columns {label_key} found in adata.obs")
    if embedding_key not in adata_.obsm.keys():
        raise ValueError(f"Embeddings {embedding_key} not found in adata.obsm")

    for label in label_cols:
        metrics = eval_scib_metrics(adata_, batch_key=batch_key, label_key=label, embedding_key=embedding_key)
        for metric in metrics.keys():
            met_df.loc[len(met_df)] = [metric, label, metrics[metric]]

    met_df.to_csv(res_path, index=False)
    ce_array = adata_.obsm[embedding_key]
    np.save(f"{res_path[:-4]}_ce.npy", ce_array)
    return met_df

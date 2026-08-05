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
    bio_conservation: bool = True,
) -> Dict:
    """TJP lab addition (2026-08-04), on top of the otherwise-verbatim upstream function
    below: batch ASW alone cannot tell real batch integration from homogenization
    (collapsing cell-type structure while incidentally mixing batches together) -- this
    was the single strongest, unanimous, cross-examination-surviving finding of the
    sclong-gap-closing-research workflow (wf_d1c898a4-2d1; see research/INDEX.md and
    ROADMAP.md §14 for the full rationale, including a live proof-of-concept: an
    alternative pooling variant landed within 0.006 of the paper's own "Raw"/no-integration
    baseline while still scoring a respectable batch ASW). When bio_conservation=True
    (default), also compute NMI/ARI (cluster-vs-label agreement, via scib's own
    cluster_optimal_resolution Leiden sweep) and isolated_labels_f1 (recovery of rare/
    single-batch cell types, e.g. this dataset's mast/schwann/epsilon groups). Standing
    policy going forward: no ASW number should be reported as "progress" unless this
    bio-conservation panel moves with it, not against it. Each block is wrapped in its own
    try/except -- a bio-conservation failure (e.g. too few cells for a stable clustering
    sweep at small n) degrades to ASW-only rather than crashing the whole run.
    """
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

    if bio_conservation:
        # Clustering-step-only PCA reduction -- standard scib practice for Leiden on
        # feature-space embeddings (scib.pp.reduce_data does the same). This is a
        # SEPARATE, unrelated question from the open "should ASW itself be computed on
        # raw or PCA-reduced dims" debate (see ROADMAP.md §14) -- it only bounds the
        # cost/memory of the neighbor graph used for clustering-based metrics below, and
        # never touches ASW_label/batch above or the headline embedding itself. Skipped
        # if the embedding is already low-dimensional (<=50 dims).
        cluster_rep = embedding_key
        emb = adata.obsm[embedding_key]
        if emb.shape[1] > 50:
            # sklearn directly on the obsm array, NOT sc.pp.pca(adata, ...) -- that call
            # operates on adata.X by default in this scanpy version, not on an arbitrary
            # obsm key, and would silently PCA-reduce the wrong matrix. Verified directly
            # (a synthetic-data smoke test caught exactly this mistake on first attempt).
            from sklearn.decomposition import PCA
            n_comps = min(50, emb.shape[0] - 1, emb.shape[1])
            cluster_rep = f"{embedding_key}_pca{n_comps}_clusteronly"
            adata.obsm[cluster_rep] = PCA(n_components=n_comps, random_state=0).fit_transform(
                np.asarray(emb)
            )
            print_sys(f"bio_conservation: PCA-reduced {embedding_key} ({emb.shape[1]}-dim) -> "
                       f"{cluster_rep} ({n_comps}-dim) for clustering only.")

        # BUG FOUND AND FIXED 2026-08-04 (see ROADMAP.md §14.2, job 316672's identical-
        # to-14-decimals NMI/ARI across 3 different embeddings): scib.metrics.
        # cluster_optimal_resolution SKIPS re-clustering whenever f"{cluster_key}_{res}"
        # already exists in adata.obs.columns (its own caching, meant for the normal
        # case of one adata per run). Any caller that reuses ONE adata object across
        # multiple evaluate()/eval_scib_metrics() calls (e.g. experiments/
        # additive_pooling_test.py, comparing several pooling variants in one process)
        # silently gets the FIRST call's stale clustering reused for every subsequent
        # variant. force=True (documented in scib's own isolated_labels_f1 docstring
        # example) makes both calls below always re-cluster on the actual current
        # embedding, regardless of what a prior call on this same adata left behind.
        try:
            cluster_key = "leiden_biocons"
            scib.metrics.cluster_optimal_resolution(
                adata, label_key=label_key, cluster_key=cluster_key,
                use_rep=cluster_rep, verbose=False, return_all=False, force=True,
            )
            results_dict["NMI_cluster/label"] = scib.metrics.nmi(adata, cluster_key, label_key)
            results_dict["ARI_cluster/label"] = scib.metrics.ari(adata, cluster_key, label_key)
        except Exception as e:
            print_sys(f"WARNING: NMI/ARI clustering metrics failed "
                       f"({type(e).__name__}: {e}) -- reporting without them.")

        try:
            results_dict["isolated_label_F1"] = scib.metrics.isolated_labels_f1(
                adata, label_key=label_key, batch_key=batch_key, embed=cluster_rep, verbose=False,
                force=True,  # same stale-cache issue as above -- see comment on cluster_optimal_resolution
            )
        except Exception as e:
            print_sys(f"WARNING: isolated_labels_f1 failed ({type(e).__name__}: {e}) -- reporting without it.")

    print_sys("\n".join([f"{k}: {v:.4f}" for k, v in results_dict.items()]))
    results_dict = {k: v for k, v in results_dict.items() if not np.isnan(v)}
    return results_dict


def evaluate(adata_, batch_key=None, label_key: list = ["celltype"],
             embedding_key: str = "X_scLong", res_path: str = "",
             bio_conservation: bool = True) -> pd.DataFrame:
    met_df = pd.DataFrame(columns=["metric", "label", "value"])
    label_cols = [x for i, x in enumerate(label_key) if x not in label_key[:i]]
    label_cols = [x for x in label_cols if x in adata_.obs.columns]

    if len(label_cols) == 0:
        raise ValueError(f"No label columns {label_key} found in adata.obs")
    if embedding_key not in adata_.obsm.keys():
        raise ValueError(f"Embeddings {embedding_key} not found in adata.obsm")

    for label in label_cols:
        metrics = eval_scib_metrics(adata_, batch_key=batch_key, label_key=label, embedding_key=embedding_key,
                                     bio_conservation=bio_conservation)
        for metric in metrics.keys():
            met_df.loc[len(met_df)] = [metric, label, metrics[metric]]

    met_df.to_csv(res_path, index=False)
    ce_array = adata_.obsm[embedding_key]
    np.save(f"{res_path[:-4]}_ce.npy", ce_array)
    # TJP lab addition (2026-08-04): persist which cells back this embedding, for every
    # run, not just full-scale ones. Without this, a cached *_ce.npy is unusable for any
    # post-hoc reanalysis (per-celltype breakdown, bio-conservation, re-scoring under a
    # different metric) unless the exact subsample is independently reproducible -- which
    # it wasn't before `sc.pp.subsample` was seeded (see run_zero_shot_batch_integration.py).
    obs_cols = [c for c in label_cols]
    if batch_key and batch_key not in obs_cols and batch_key in adata_.obs.columns:
        obs_cols.append(batch_key)
    adata_.obs[obs_cols].to_csv(f"{res_path[:-4]}_obs.csv")
    return met_df

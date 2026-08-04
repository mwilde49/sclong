# sclong — scLong Single-Cell Foundation Model Pipeline

Apptainer-packaged pipeline wrapping [scLong](https://github.com/BaiDing1234/scLong)
(Bai et al., *Nature Communications*, 2026) — a ~1B-parameter single-cell transcriptomics
foundation model. Built for the TJP lab's HPC framework
([mwilde49/hpc](https://github.com/mwilde49/hpc)) and its Juno GPU nodes, following the
same submoduled Python+Apptainer pattern as `mwilde49/dconvatac`.

**Not yet wired into the shared `hpc/` registry (`tjp-launch`/`tjp-batch`).** This repo is
usable standalone via `apptainer exec` + `sbatch`/`srun` — see below. Full framework
integration (config templates, SLURM registry, `tjp-test-suite` module) is scoped in
`sclong/ROADMAP.md` §4 in the parent planning workspace but deliberately deferred until
this pipeline is validated end-to-end on real data.

## Why this exists

scLong's own repository is a research-artifact dump — no license (permission for this
build obtained directly from the lab, 2026-08-03), no packaged dependencies, several
broken entry points (including a `model.get_cell_emb()` call to a method that doesn't
exist anywhere in the codebase). This repo is a patched, containerized, reproducible
wrapper around it. Full technical audit in the parent planning workspace's
`sclong/ROADMAP.md`.

## What it does (so far)

`pipeline/run_zero_shot_batch_integration.py` — zero-shot cell embedding + scIB batch-
integration evaluation. This is "Stage 1" of the validation plan in `ROADMAP.md` §5:
reproduce the paper's own reported batch ASW = 0.9561 on the scIB pancreas benchmark
before trusting this installation on anything else. Deterministic, no training required —
the cleanest first checkpoint that the checkpoint/environment/gene-vocabulary chain is
correct.

Validated locally (WSL2, RTX 5070 Ti, 12GB VRAM) before containerizing — see inline
comments in `pipeline/run_zero_shot_batch_integration.py` for exact numbers. **The reason
this needs to run on Juno rather than a laptop GPU:** at `batch_size=4`, a 12GB card is
already fully saturated (100% util, ~95MB free) and per-batch time trends upward over a
run — memory-pressure allocator overhead, not just slow hardware. Juno's A30 (24GB) or
H100 (80GB) remove that ceiling entirely and allow a much larger batch size.

## Repository layout

```
sclong-hpc/
├── pipeline/
│   ├── run_zero_shot_batch_integration.py   # main pipeline script
│   ├── get_cell_emb.py                      # reconstructed missing upstream method
│   ├── eval_utils.py                        # trimmed, pickle5-free copy of upstream's eval helpers
│   └── smoke_test_architecture.py           # synthetic-weights sanity check, no gated data needed
└── container/
    ├── apptainer.def         # container definition — clones BaiDing1234/scLong@41b7202 at build time
    ├── build.sh               # build sclong_v1.0.0.sif (sudo, or --fakeroot)
    └── test_container.sh      # import checks + optional --gpu smoke test
```

## Building the container

```bash
cd container
./build.sh --fakeroot     # or: sudo ./build.sh
```

The built `.sif` is excluded from git (multi-GB). Transfer it wherever it needs to run:

```bash
scp container/sclong_v1.0.0.sif juno:/scratch/juno/$USER/sclong/
```

## Data — not in this repo, not in git, ever

The container clones scLong's *code* at build time but does **not** bundle the checkpoint
or gene-vocabulary files (multi-GB, and gated behind the authors' SharePoint folder — see
`ROADMAP.md` §1). Stage these separately and bind-mount them at runtime:

```
<data-root>/
├── gene_meta/
│   ├── selected_gene2vec_27k.npy
│   ├── selected_genes_27k.txt
│   ├── gocont_4096_48m_pretrain_1b_mix.pkl
│   └── human_symbol_to_ens.txt          # reconstructed via Ensembl BioMart, not from authors
├── checkpoints/
│   └── gocont_4096_48m_pretrain_1b_mix_2024-02-05_16-23-37.pth
└── datasets/
    └── pancreas_scib.h5ad                # author-provided copy — has both 'tech' and 'batch' obs columns
```

## Running

```bash
apptainer exec --nv \
    --bind /path/to/data-root:/data \
    container/sclong_v1.0.0.sif \
    python /pipeline/run_zero_shot_batch_integration.py --data-root /data --batch-size <N>
```

Smoke-test a freshly built container first, with no data required:

```bash
cd container && ./test_container.sh --gpu
```

## SLURM (Juno)

No registered SLURM template in this repo yet (see "Why this exists" above — that's the
deferred framework-integration layer). Submit directly, following the resource pattern
`mwilde49/dconvatac` already established for GPU pipelines in this framework:

```bash
#SBATCH --partition=a30              # or h100 for larger batch sizes
#SBATCH --gres=gpu:nvidia_a30:1      # or gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
```

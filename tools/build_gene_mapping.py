"""
Build/version the human gene-symbol -> Ensembl-ID mapping scLong's pipeline needs.

Why this exists: scLong's own repo never shipped the mapping file it depends on
(human_symbol_to_ens.txt is read by the pipeline but was never in the authors' SharePoint
folder — ROADMAP.md blocker #3). Any user of this pipeline has to reconstruct one, and
mapping completeness measurably affects results: the Stage 1 full-run diagnostic
(2026-08-04) found a batch-ASW shortfall vs. the paper's reported 0.9561, spread broadly
across almost every cell type rather than concentrated in a few — the signature of a
systematically incomplete gene mapping degrading the whole embedding, not a few hard
cell types (which was the first, ruled-out hypothesis; see the per-celltype breakdown in
diagnose_per_celltype.py's output).

This tool treats "the mapping" as a versioned, auditable artifact, not a one-off file.
Each named recipe below is a fully reproducible build (documented sources, deterministic
tiered disambiguation, full per-symbol audit trail) written to its own directory under
<output-root>/<version>/. Point run_zero_shot_batch_integration.py's
--gene-mapping-version at whichever one you want to use or compare — swapping mappings
never requires touching pipeline code, and every version's raw source downloads are
cached alongside it for exact reproducibility.

Recipes:
  v1_biomart_symbol_only
      The original reconstruction (2026-08-03): Ensembl BioMart external_gene_name only,
      one tier, no synonym/previous-symbol resolution. ~42k symbols. Kept only as the
      baseline to compare against — not recommended for actual use.

  v2_hgnc_plus_biomart_synonyms
      HGNC's complete gene set (the authoritative source for human gene nomenclature —
      current symbol, previous symbol, alias symbol, all with a direct Ensembl
      cross-reference HGNC itself curates) UNIONed with Ensembl BioMart
      (external_gene_name + external_synonym, catching Ensembl-only entries HGNC hasn't
      reviewed). Six tiers by descending confidence:
        1. HGNC current symbol       4. HGNC alias symbol
        2. BioMart current symbol    5. BioMart synonym
        3. HGNC previous symbol      6. case-insensitive fallback over all of the above
      A symbol resolves at the first tier where it has ANY candidate. Within a tier, if
      a symbol has multiple distinct candidate Ensembl IDs, resolution proceeds only if
      exactly one candidate is in scLong's fixed gene vocabulary (--model-genes) — the
      one that actually matters functionally; if that still leaves 0 or 2+ candidates,
      the symbol is left unresolved and logged to ambiguous.tsv rather than guessed.

Usage:
    python build_gene_mapping.py v2_hgnc_plus_biomart_synonyms \\
        --output-root ../../data/gene_meta/mappings \\
        --model-genes ../../data/gene_meta/selected_genes_27k.txt \\
        --dataset-genes-h5ad ../../data/datasets/pancreas_scib.h5ad \\
        --set-current

    # compare against the baseline without touching CURRENT:
    python build_gene_mapping.py v1_biomart_symbol_only \\
        --output-root ../../data/gene_meta/mappings \\
        --model-genes ../../data/gene_meta/selected_genes_27k.txt \\
        --dataset-genes-h5ad ../../data/datasets/pancreas_scib.h5ad
"""
import argparse
import csv
import datetime
import io
import sys
from collections import defaultdict
from pathlib import Path

import requests

BIOMART_URL = "https://www.ensembl.org/biomart/martservice"
HGNC_URL = "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt"


# ───────────────────────── source fetchers (cached to raw/ for provenance) ─────────────────────────

def fetch_biomart(attributes, cache_path):
    if cache_path.exists():
        return cache_path.read_text()
    attrs_xml = "".join(f'<Attribute name="{a}" />' for a in attributes)
    xml_query = (
        '<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE Query>'
        '<Query virtualSchemaName="default" formatter="TSV" header="0" uniqueRows="1" '
        'count="" datasetConfigVersion="0.6"><Dataset name="hsapiens_gene_ensembl" '
        f'interface="default">{attrs_xml}</Dataset></Query>'
    )
    r = requests.get(BIOMART_URL, params={"query": xml_query}, timeout=180)
    r.raise_for_status()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(r.text)
    return r.text


def fetch_hgnc(cache_path):
    if cache_path.exists():
        return cache_path.read_text()
    r = requests.get(HGNC_URL, timeout=180)
    r.raise_for_status()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(r.text)
    return r.text


def _index_two_col_tsv(text, key_col=0, val_col=1):
    idx = defaultdict(set)
    for line in text.strip().splitlines():
        parts = line.split("\t")
        if len(parts) < 2 or not parts[key_col] or not parts[val_col]:
            continue
        idx[parts[key_col]].add(parts[val_col])
    return idx


# ───────────────────────────────── tiered resolution core ─────────────────────────────────

def add_case_insensitive_tier(tier_indexes):
    lower_idx = defaultdict(set)
    for _, idx in tier_indexes:
        for sym, ens_set in idx.items():
            lower_idx[sym.lower()].update(ens_set)
    return tier_indexes + [("case_insensitive_fallback", lower_idx)]


def resolve(tier_indexes, model_genes):
    """tier_indexes: [(tier_name, {symbol: {ensembl_id, ...}}), ...] in priority order.
    Returns (mapping, audit_rows, ambiguous_rows)."""
    all_symbols = set()
    for _, idx in tier_indexes:
        all_symbols.update(idx.keys())

    mapping, audit, ambiguous = {}, [], []
    for symbol in sorted(all_symbols):
        for tier_name, idx in tier_indexes:
            candidates = idx.get(symbol)
            if not candidates:
                continue
            if len(candidates) == 1:
                ens = next(iter(candidates))
                mapping[symbol] = ens
                audit.append({"symbol": symbol, "ensembl_id": ens, "tier": tier_name, "vocab_disambiguated": False})
            else:
                in_vocab = (candidates & model_genes) if model_genes else set()
                if len(in_vocab) == 1:
                    ens = next(iter(in_vocab))
                    mapping[symbol] = ens
                    audit.append({"symbol": symbol, "ensembl_id": ens, "tier": tier_name, "vocab_disambiguated": True})
                else:
                    ambiguous.append({"symbol": symbol, "tier": tier_name, "candidates": "|".join(sorted(candidates))})
            break  # stop at the first tier with any match, resolved or not
    return mapping, audit, ambiguous


# ───────────────────────────────────────── recipes ─────────────────────────────────────────

def build_v1_biomart_symbol_only(raw_dir, model_genes):
    text = fetch_biomart(["external_gene_name", "ensembl_gene_id"], raw_dir / "biomart_symbol.tsv")
    tier_indexes = [("biomart_current_symbol", _index_two_col_tsv(text))]
    mapping, audit, ambiguous = resolve(tier_indexes, model_genes)
    notes = {
        "sources": ["Ensembl BioMart hsapiens_gene_ensembl: external_gene_name, ensembl_gene_id"],
        "tiers": [t[0] for t in tier_indexes],
    }
    return mapping, audit, ambiguous, notes


def build_v2_hgnc_plus_biomart_synonyms(raw_dir, model_genes):
    hgnc_text = fetch_hgnc(raw_dir / "hgnc_complete_set.txt")
    biomart_name_text = fetch_biomart(["external_gene_name", "ensembl_gene_id"], raw_dir / "biomart_symbol.tsv")
    biomart_syn_text = fetch_biomart(["ensembl_gene_id", "external_synonym"], raw_dir / "biomart_synonym.tsv")

    hgnc_symbol, hgnc_prev, hgnc_alias = defaultdict(set), defaultdict(set), defaultdict(set)
    reader = csv.DictReader(io.StringIO(hgnc_text), delimiter="\t")
    for row in reader:
        if row.get("status") != "Approved":
            continue
        ens = (row.get("ensembl_gene_id") or "").strip()
        if not ens:
            continue
        sym = (row.get("symbol") or "").strip()
        if sym:
            hgnc_symbol[sym].add(ens)
        for alias in (row.get("prev_symbol") or "").split("|"):
            alias = alias.strip()
            if alias:
                hgnc_prev[alias].add(ens)
        for alias in (row.get("alias_symbol") or "").split("|"):
            alias = alias.strip()
            if alias:
                hgnc_alias[alias].add(ens)

    biomart_name = _index_two_col_tsv(biomart_name_text, key_col=0, val_col=1)
    # this query returns (ensembl_id, synonym) -- flip so synonym is the lookup key
    biomart_syn_raw = _index_two_col_tsv(biomart_syn_text, key_col=0, val_col=1)
    biomart_syn = defaultdict(set)
    for ens, synonyms in biomart_syn_raw.items():
        for syn in synonyms:
            biomart_syn[syn].add(ens)

    tier_indexes = [
        ("hgnc_current_symbol", hgnc_symbol),
        ("biomart_current_symbol", biomart_name),
        ("hgnc_previous_symbol", hgnc_prev),
        ("hgnc_alias_symbol", hgnc_alias),
        ("biomart_synonym", biomart_syn),
    ]
    tier_indexes = add_case_insensitive_tier(tier_indexes)
    mapping, audit, ambiguous = resolve(tier_indexes, model_genes)
    notes = {
        "sources": [
            f"HGNC complete gene set, status=Approved only ({HGNC_URL})",
            "Ensembl BioMart hsapiens_gene_ensembl: external_gene_name, ensembl_gene_id",
            "Ensembl BioMart hsapiens_gene_ensembl: ensembl_gene_id, external_synonym",
        ],
        "tiers": [t[0] for t in tier_indexes],
    }
    return mapping, audit, ambiguous, notes


RECIPES = {
    "v1_biomart_symbol_only": build_v1_biomart_symbol_only,
    "v2_hgnc_plus_biomart_synonyms": build_v2_hgnc_plus_biomart_synonyms,
}


# ───────────────────────────────────────── I/O + reporting ─────────────────────────────────────────

def load_gene_set(path):
    if path is None:
        return None
    with open(path) as f:
        return {line.rstrip("\n") for line in f if line.rstrip("\n")}


def load_dataset_genes(h5ad_path):
    """Mirrors run_zero_shot_batch_integration.py's preprocessing exactly (filter_cells
    then filter_genes) -- coverage stats are only meaningful if computed against the same
    gene list the pipeline actually reindexes at runtime, not the raw unfiltered h5ad."""
    if h5ad_path is None:
        return None
    import scanpy as sc
    adata = sc.read_h5ad(str(h5ad_path))
    sc.pp.filter_cells(adata, min_genes=10)
    sc.pp.filter_genes(adata, min_cells=10)
    return list(adata.var.index)


def coverage_report(mapping, model_genes, dataset_genes):
    lines = [f"Total resolved symbols in table: {len(mapping):,}"]
    if dataset_genes is not None:
        resolved = [mapping[g] for g in dataset_genes if g in mapping]
        n_resolved = len(resolved)
        lines.append(f"Dataset genes resolved to *some* Ensembl ID: {n_resolved:,} / {len(dataset_genes):,} "
                      f"({100 * n_resolved / len(dataset_genes):.1f}%)")
        if model_genes is not None:
            n_in_vocab = len(set(resolved) & model_genes)
            lines.append(f"Dataset genes resolved AND in scLong's {len(model_genes):,}-gene vocabulary: "
                          f"{n_in_vocab:,} / {len(dataset_genes):,} ({100 * n_in_vocab / len(dataset_genes):.1f}%)")
    return "\n".join(lines)


def write_outputs(version_dir, version, mapping, audit, ambiguous, notes, coverage_text):
    version_dir.mkdir(parents=True, exist_ok=True)

    with open(version_dir / "human_symbol_to_ens.txt", "w") as f:
        for symbol, ens in mapping.items():
            f.write(f"{symbol},{ens}\n")

    with open(version_dir / "audit.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["symbol", "ensembl_id", "tier", "vocab_disambiguated"], delimiter="\t")
        w.writeheader()
        w.writerows(audit)

    with open(version_dir / "ambiguous.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["symbol", "tier", "candidates"], delimiter="\t")
        w.writeheader()
        w.writerows(ambiguous)

    tier_counts = defaultdict(int)
    for row in audit:
        tier_counts[row["tier"]] += 1
    tier_lines = "\n".join(f"- `{t}`: {c:,}" for t, c in tier_counts.items())

    manifest = f"""# Gene mapping — {version}

Built: {datetime.datetime.now(datetime.timezone.utc).isoformat()}

## Sources
{chr(10).join(f"- {s}" for s in notes["sources"])}

## Resolution tiers (in priority order)
{chr(10).join(f"{i + 1}. `{t}`" for i, t in enumerate(notes["tiers"]))}

## Resolution breakdown by tier
{tier_lines}

## Coverage
{coverage_text}

## Files
- `human_symbol_to_ens.txt` — the mapping itself, `symbol,ensembl_id` per line (format the
  pipeline consumes directly).
- `audit.tsv` — every resolved symbol with which tier resolved it and whether vocabulary-aware
  disambiguation was needed.
- `ambiguous.tsv` — symbols with multiple candidate Ensembl IDs that could NOT be
  disambiguated (left unmapped rather than guessed).
- `raw/` — cached exact source downloads this build used, for reproducibility.
"""
    (version_dir / "MANIFEST.md").write_text(manifest)
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("version", choices=sorted(RECIPES.keys()))
    p.add_argument("--output-root", type=str, default="../../data/gene_meta/mappings")
    p.add_argument("--model-genes", type=str, default=None,
                   help="selected_genes_27k.txt -- enables vocabulary-aware disambiguation + coverage stats")
    p.add_argument("--dataset-genes-h5ad", type=str, default=None,
                   help="an h5ad to report real coverage against (e.g. pancreas_scib.h5ad)")
    p.add_argument("--set-current", action="store_true",
                   help="write output-root/CURRENT to point run_zero_shot_batch_integration.py at this version")
    args = p.parse_args()

    output_root = Path(args.output_root)
    version_dir = output_root / args.version
    raw_dir = version_dir / "raw"

    model_genes = load_gene_set(args.model_genes)
    dataset_genes = load_dataset_genes(args.dataset_genes_h5ad)

    print(f"Building {args.version}...", file=sys.stderr)
    mapping, audit, ambiguous, notes = RECIPES[args.version](raw_dir, model_genes)
    coverage_text = coverage_report(mapping, model_genes, dataset_genes)

    manifest = write_outputs(version_dir, args.version, mapping, audit, ambiguous, notes, coverage_text)
    print(manifest)

    if args.set_current:
        (output_root / "CURRENT").write_text(args.version + "\n")
        print(f"CURRENT -> {args.version}", file=sys.stderr)


if __name__ == "__main__":
    main()

"""
Morning-after summary for the ablation_grid_submit.sh batch (ROADMAP.md §14.6 item 7).
Reads every result CSV in --grid-dir (plus, optionally, the two pre-existing confirmed
runs from job 315836/316352) and prints one ranked comparison table: ASW alongside
NMI/ARI/isolated-F1 for every config, so a config is never judged on ASW alone (the
standing policy from ROADMAP.md §14.1).

Does not touch the model or dataset -- pure CSV parsing, seconds, no GPU needed. Can run
on the login node.

Run:
    apptainer exec --cleanenv --env PYTHONNOUSERSITE=1 \
        --bind /groups/tprice/pipelines:/groups/tprice/pipelines \
        --bind /scratch/juno/$USER:/scratch/juno/$USER \
        --bind $(pwd)/pipeline:/pipeline \
        container/sclong_v1.0.0.sif \
        python /pipeline/summarize_ablation_grid.py \
          --grid-dir /scratch/juno/$USER/sclong_results_grid \
          --baseline-csv /scratch/juno/$USER/sclong_results/scLong_batch_cell_emb_mode.csv \
          --combined-csv /scratch/juno/$USER/sclong_results_v2/scLong_batch_cell_emb_mode__map-v2_hgnc_plus_biomart_synonyms_pool-measured-only_norm-cpm.csv
"""
import argparse
import re
from pathlib import Path

import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--grid-dir", type=str, required=True)
p.add_argument("--baseline-csv", type=str, default=None,
                help="Pre-existing baseline run (job 315836), included in the table for context.")
p.add_argument("--combined-csv", type=str, default=None,
                help="Pre-existing combined run (job 316352), included in the table for context.")
args = p.parse_args()

CONFIG_TAG_RE = re.compile(r"map-(?P<map>[^_]+(?:_[^_]+)*?)_pool-(?P<pool>[^_]+(?:-only)?)_norm-(?P<norm>[^_.]+)")


def parse_config_from_filename(path):
    """Best-effort parse of the map-.../pool-.../norm-... config_tag baked into the
    filename by run_zero_shot_batch_integration.py. Falls back to the raw filename if it
    doesn't match (e.g. the pre-config_tag-era baseline file)."""
    stem = path.stem
    m = CONFIG_TAG_RE.search(stem)
    if m:
        return f"map={m.group('map')} pool={m.group('pool')} norm={m.group('norm')}"
    return stem


def load_one(path, label=None):
    df = pd.read_csv(path)
    metrics = df.set_index("metric")["value"].to_dict()
    return {
        "config": label or parse_config_from_filename(path),
        "file": path.name,
        "ASW": metrics.get("ASW_label/batch"),
        "NMI": metrics.get("NMI_cluster/label"),
        "ARI": metrics.get("ARI_cluster/label"),
        "isoF1": metrics.get("isolated_label_F1"),
    }


rows = []
seen_paths = set()  # resolved paths already loaded -- avoids double-counting if
                     # --baseline-csv/--combined-csv happen to live inside --grid-dir

if args.baseline_csv and Path(args.baseline_csv).exists():
    p_ = Path(args.baseline_csv).resolve()
    rows.append(load_one(p_, label="BASELINE (job 315836): map=legacy pool=full norm=as-shipped"))
    seen_paths.add(p_)
if args.combined_csv and Path(args.combined_csv).exists():
    p_ = Path(args.combined_csv).resolve()
    rows.append(load_one(p_, label="COMBINED (job 316352): map=v2 pool=measured-only norm=cpm"))
    seen_paths.add(p_)

grid_dir = Path(args.grid_dir)
csv_files = sorted(f for f in grid_dir.glob("*.csv") if not f.name.endswith("_obs.csv"))
if not csv_files:
    print(f"WARNING: no result CSVs found in {grid_dir} -- have the grid jobs finished?")
for f in csv_files:
    if f.resolve() in seen_paths:
        continue
    rows.append(load_one(f))

if not rows:
    print("Nothing to summarize.")
    raise SystemExit(1)

table = pd.DataFrame(rows).sort_values("ASW", ascending=False, na_position="last")
pd.set_option("display.max_colwidth", 80)
pd.set_option("display.width", 200)

print(f"\n=== Ablation grid summary ({len(table)} configs, sorted by ASW) ===\n")
print(table[["config", "ASW", "NMI", "ARI", "isoF1"]].to_string(index=False, float_format=lambda v: f"{v:.4f}" if pd.notnull(v) else "—"))

best_asw = table.iloc[0]
print(f"\nBest ASW: {best_asw['config']} ({best_asw['ASW']:.4f})")

# Flag any config whose bio-conservation trio all sit meaningfully below the best-ASW
# config's own trio -- the homogenization warning from ROADMAP.md §14.1, applied across
# the whole grid instead of just baseline-vs-combined.
if pd.notnull(best_asw.get("NMI")) and pd.notnull(best_asw.get("ARI")):
    print("\n--- bio-conservation check: any config with ASW within 0.01 of the best "
          "but NMI/ARI clearly worse? (possible homogenization, not just noise) ---")
    near_best = table[(best_asw["ASW"] - table["ASW"]).abs() < 0.01]
    for _, r in near_best.iterrows():
        if r["config"] == best_asw["config"]:
            continue
        nmi_drop = (best_asw["NMI"] - r["NMI"]) if pd.notnull(r.get("NMI")) else None
        ari_drop = (best_asw["ARI"] - r["ARI"]) if pd.notnull(r.get("ARI")) else None
        if (nmi_drop and nmi_drop > 0.02) or (ari_drop and ari_drop > 0.02):
            print(f"  {r['config']}: ASW={r['ASW']:.4f} (close to best) but NMI/ARI drop "
                  f"NMI Δ={nmi_drop:.4f} ARI Δ={ari_drop:.4f} -- inspect before trusting")

print("\nDONE")

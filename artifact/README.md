# Artifact: Leakage-Free Evaluation of ML Hardware Trojan Detection

Anonymized artifact accompanying the submission. Scope of what is reproducible
from this package alone, stated precisely:

- **Stages 4--5 (main tables and analyses): fully reproducible from the bundled
  feature matrices in `results/`.** These generate the paper's E1/E2/E3 protocol
  tables, the per-family breakdown, the PCA leakage statistics, the paired
  ΔAUC significance tests, the literature-pipeline baselines, the design ×
  library joint holdout, the 3-seed E2 decomposition, the single-feature
  structural baseline of Sec. V, and the four-library E3 matrix of Table V. This path has been
  verified in a clean directory (see `VERIFICATION.md` for the number-by-number
  comparison against the submitted tables).
- **Stages 1--3 and 6--7 (raw-data pipeline and GNN/AIG experiments) require
  external inputs** that cannot be redistributed here: the public Trust-Hub
  archives (trust-hub.org) and the OpenROAD standard-cell libraries (pinned by
  hash in `tools/lib_manifest.sha256`). They are documented and expected to run
  after fetching those inputs, but were not clean-room re-run from this package.

## Layout

```
src/      pipeline and experiment code (see below)
tools/    dataset extraction, library preparation, library provenance manifest,
          and the HW2VEC/GNN4TJ re-scoring driver (tools/hw2vec_ladder)
results/  generated features (v1 cell-type + v2 operator) and reported results
tools/audit_golden.py  degeneracy audit of every golden class (Sec. III-B, V-B)
tests/    mini Liberty fixtures so `src/test_lib_ops.py` runs self-contained
DEVIATIONS.md  every deviation of the HW2VEC/GNN4TJ re-scoring (Sec. V-F)
```

## Environment (pinned)

| Tool | Version |
|---|---|
| Yosys | 0.68 (OSS CAD Suite build 0.68+195) |
| Python | 3.11 |
| scikit-learn | 1.9.1 |
| XGBoost | 3.2.0 |
| PyTorch (CPU) | 2.14.1 |
| PyTorch Geometric | 2.8.0 |
| numpy / pandas | 2.2.6 / 2.3.3 |

All experiments fix `SEED = 20261012` (in `src/c1_protocol.py`) and use 1,000
bootstrap resamples for 95% CIs. Tool locations are overridable via environment
variables (defaults point at the authors' machine): `YOSYS` (used by
`src/run_yosys_synth.py`, `src/aig_ablation.py`) and `UNRAR` (used by
`tools/extract_trusthub.py` for the rar-based Trust-Hub archives).

## Pipeline stages and reproducibility status

| Stage | What | Input | Runs from this package? |
|---|---|---|---|
| 1. Extraction/cleaning | `tools/extract_trusthub.py` | Trust-Hub archives (external) | No -- external data |
| 2. Synthesis + features | `src/run_yosys_synth.py`, `src/batch_synth.py`, `src/batch_multilib.py`, `tools/merge_asap7_lib.py` | stage-1 HDL + Liberty libs (external, pinned) | No -- external data |
| 3. v2 operator normalization | `src/lib_ops.py`, `src/c2_normalize.py` | stage-2 feature CSVs (v1 bundled in `results/`) | Partially (replay runs on bundled v1) |
| 4. Protocol runner | `src/c1_protocol.py` | bundled feature CSVs | **Yes (verified, `VERIFICATION.md`)** |
| 5. Breakdowns & analyses | `src/family_table.py`, `src/pca_leak_fig.py`, `src/e2_subset.py` (E2 TRIT/non-TRIT decomposition), `src/lit_baselines.py`, `src/joint_holdout.py`, `src/aig_detect.py` (four-representation detection comparison) | bundled feature CSVs incl. `aigA/B_features.csv` | **Yes (verified, `VERIFICATION.md`)** |
| 6. GNN baseline | `src/netlist_graph.py`, `src/gnn_baseline.py` | stage-2 JSON netlists | No -- needs re-synthesis from raw |
| 7. AIG ablation | `src/aig_cell_models.py`, `src/aig_ablation.py` | stage-2 netlists + v1/v2 CSVs (bundled) + yosys/abc | No -- needs re-synthesis from raw |

Stage details:

1. **Extraction/cleaning** -- `tools/extract_trusthub.py` unpacks the Trust-Hub
   benchmark archives (gold/Trojan pair layouts, TRIT variant layouts, mixed
   testbench layouts), rewrites author-machine `include` paths, filters
   testbenches (name + directory), and deduplicates basenames. Note: several
   archives are rar-in-zip; an external `unrar` is required (`UNRAR` env var).
2. **Synthesis + features** -- `src/run_yosys_synth.py` (single design: clean →
   Yosys map to a Liberty library → flatten → `src/netlist_features.py`
   features) and `src/batch_synth.py` / `src/batch_multilib.py` (batch drivers;
   they expect the library files at `./libs/liberty/...`).
   `tools/merge_asap7_lib.py` merges the ASAP7 multi-library Liberty files into
   `asap7_RVT_TT_merged.lib` (for `dfflibmap`) and `asap7_RVT_TT_comb.lib`
   (combinational-only, for `abc`).
3. **v2 operator normalization** -- `src/lib_ops.py` (truth-table equivalence via
   factorized-template matching; unit tests in `src/test_lib_ops.py`, which
   falls back to the bundled mini Liberty fixtures in `tests/fixture/` when the
   real libraries are absent---force with `LIB_OPS_FIXTURE=1`) and
   `src/c2_normalize.py` (feature replay onto v2 schema).
4. **Protocol runner** -- `src/c1_protocol.py`: E1 (paired set) / E2 (full set) /
   E3 (cross-library), protocols pooled-random-5fold / group-by-variant /
   group-by-base / leave-one-family-out / leave-one-library-out, models RF /
   XGBoost / MLP, metrics + bootstrap CIs. The CI resample unit is always the
   base-design cluster, independent of the split (`run_protocol(ci_groups=...)`),
   and `boot_delta` computes paired ΔAUC CIs between protocol pairs. ⚠️ The
   module default points at the
   v1 CSV (`features_all.csv`); pass `--all-csv` explicitly to select the v2
   matrices used for the paper's tables.
5. **Breakdowns** -- `src/family_table.py` (per-family XGBoost),
   `src/pca_leak_fig.py` (PCA leakage visualization + neighbor statistics),
   `src/lit_baselines.py` (published-pipeline baselines: decision tree$+$PCA and
   $k$NN $k{=}5$), `src/joint_holdout.py` (design $\times$ library joint
   holdout), `src/e2_subset.py --seed` (E2 TRIT/non-TRIT decomposition,
   reproducible under any seed), and `src/permutation_test.py` (label-shuffle
   null distributions; deterministic, bundled, but not part of the v3
   clean-room comparison table).
6. **GNN baseline** -- `src/netlist_graph.py` (netlist → PyG graph cache),
   `src/gnn_baseline.py` (GCN under the same protocol ladder). Graph caching
   takes ~1–1.5 h on 4 cores; cross-validation ~12–15 h. Both consume the
   stage-2 JSON netlists, so this stage requires the raw-data pipeline first.
7. **AIG ablation** -- `src/aig_cell_models.py` (Liberty → Verilog functional
   models) + `src/aig_ablation.py` (AIG baselines A/B, drift + additivity
   check). Consumes stage-2 netlists plus the v1 and v2 multilib CSVs (both
   bundled in `results/`); expects them under `./experiments/synth_work/`.

## Reproducing the paper's main tables (stages 4–5, no external data)

```
python src/c1_protocol.py \
  --all-csv results/features_all_v2.csv \
  --multilib-csv results/features_multilib_v2.csv \
  --out-dir out --tag v2cc
python src/family_table.py  --all-csv results/features_all_v2.csv --out-dir out
python src/pca_leak_fig.py  --all-csv results/features_all_v2.csv
python src/e2_subset.py --all-csv results/features_all_v2.csv --out-dir out
python src/aig_detect.py --out-dir out
```

Expected outputs match `results/c1_report_v2cc.txt`, `results/e2_subset_report.txt`,
`results/aig_detect_report.txt`, and `results/family_xgb.txt` (all runs fix
`n_jobs=1`, so outputs are byte-deterministic). See `VERIFICATION.md` for the
clean-directory check.

## Library provenance (stages 2, 7)

The Liberty files are not redistributed (size and ASAP7 redistribution terms);
`tools/lib_manifest.sha256` pins the exact inputs used in the paper
(SHA-256). To fetch them: `bash tools/get_orfs_libs.sh <dest>` (sparse checkout
of OpenROAD-flow-scripts `flow/platforms/{nangate45,sky130hd,asap7}/lib`; the
paper used commit `3451ca1f56902a49baf50bb8eba345eb709cf9c8`), copy
`NangateOpenCellLibrary_typical.lib` and `sky130_fd_sc_hd__tt_025C_1v80.lib`
plus the ASAP7 RVT_TT NLDM group into `./libs/liberty/{nangate45,sky130hd,asap7}/`,
run `tools/merge_asap7_lib.py`, and verify against `lib_manifest.sha256`.

## Dataset composition audit

`results/features_all.csv` / `results/features_all_v2.csv` (1,059 rows) and
`results/features_multilib.csv` / `results/features_multilib_v2.csv` (297 rows
= 99 designs × 3 libraries) carry per-row `family / variant / source / tech /
label / group` metadata; the degeneracy audit (60 golden rows = 13 unique
feature vectors) is reproduced by `src/c1_protocol.py` itself at startup.

## Notes

- The paper's headline numbers are read from `results/*report*.txt`.
- `family_xgb.tex`-style table snippets are generated by the breakdown scripts.
- Trust-Hub archives are public (trust-hub.org) but not redistributed here;
  everything else needed for stages 4–5 is bundled.


## R7/R8 追加结果（2026-10-20）
results/ 新增：framework_holdout（跨框架迁移）、oracle_control（可解性正对照）、
probe_stability（库身份探针+归因稳定性）、e3_signflip/e3_per_group（E3 精确检验+逐组表）、
audit_demo（cone 预算差的探索性 golden-reference 启发式，24 TRIT 对）。该脚本按单元类型名称
推断算子类别，并将没有同名 golden 对应项的单元赋予零参考预算；它不是 C2 的 Liberty
真值表实现，也不是盲测定位评估，因此不作为论文的定位性能或 C2 有效性证据。src/ 新增对应脚本 r7_*.py / r8_audit_demo.py。
数据源路径已消毒为 PROJECT_ROOT（指向本 artifact 根）。种子统一 20261012，模型 n_jobs=1。

## Data provenance after the golden-class correction (2026-10-23)

Two result files in `results/` are **not** re-generated from the corrected
feature matrices, and are labelled as such:

| file | status |
|---|---|
| `gnn_results.csv` | E1 and E3 rows are current; the three `E2-full` rows were computed from the pre-correction TRIT golden class, are superseded, and are **not reported** in the paper (Table VIII reports E1/E3 only). Regenerating them needs the 12 corrected golden graph caches (`probe` recipe in `DEVIATIONS.md`) plus about 4 h of retraining per protocol. |
| `framework_holdout.csv` | regenerated from the corrected matrices (see `results/framework_holdout_report.txt`). |

`aig_detect_results.csv`, `aigA_features.csv`, `aigB_features.csv` and
`e3*` files come from the four-library matrix (`results/features_multilib*.csv`),
which is built from RTL designs keyed by variant and is unaffected by the
golden-class correction.

`e3table_4lib_ci.csv` (AUC + CI half-width) and `e3_4lib_detection.csv`
(AUC, specificity, sensitivity, balanced accuracy at the fixed 0.5
threshold, per held-out library and pooled) are the two files behind
Table V; `e3x_signflip_4lib.csv` holds the exact sign-flip test.

`features_all.csv` / `features_all_v2.csv` carry the corrected golden rows
(`tools/audit_golden.py` passes: 72 golden rows, 21 unique vectors).

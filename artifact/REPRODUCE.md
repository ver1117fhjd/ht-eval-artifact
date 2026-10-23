# Reproducing the paper's numbers from this package

Everything below runs with **Python 3.11** and the pinned stack
(`scikit-learn 1.9.1`, `XGBoost 3.2.0`, `numpy 2.2.6`, `pandas 2.3.3`).
All runs fix `n_jobs=1` and `random_state=SEED`, so outputs are byte-deterministic.

## A. Main tables and analyses (no external data)

```bash
python src/c1_protocol.py --all-csv results/features_all_v2.csv \
    --multilib-csv results/features_multilib_v2.csv --out-dir out --tag v2cd
python src/family_table.py --all-csv results/features_all_v2.csv --out-dir out
python src/pca_leak_fig.py --all-csv results/features_all_v2.csv
python src/lit_baselines.py --out-dir out
python src/joint_holdout.py --out-dir out
python src/e2_subset.py --all-csv results/features_all_v2.csv --out-dir out
python src/permutation_test.py --out-dir out
python src/r7_e3_signflip.py --out-dir out
```

Expected outputs are the files in `results/`; `VERIFICATION.md` records the
clean-directory, number-by-number comparison for the E1–E3 tables, the paired
ΔAUC tests, the literature-pipeline baselines and the joint holdout.

## B. E3 extension (needs external netlists)

The extended cross-library set re-maps 30 Trust-Hub netlists that ship as
gate-level netlists in the original Nangate45 cell names. The pipeline is:

1. `src/e3_extension/build_cell_models.py` — inventories the 58 cell names used
   by those netlists and emits functional Verilog models:
   42 combinational cells are generated from the `function` strings of the
   nangate45 Liberty; 16 cells absent from that library (AO/OA compounds,
   MUX21/MUX41, XOR3/XNOR3, buffer, isolation-AND, latches, scan flops) are
   written explicitly.
2. `src/e3_extension/verify_cell_models_comb.py` — exhaustive truth-table check
   of every combinational model against an independent reference
   (`iverilog`); `verify_cell_models_seq.py` — reset/clock-edge/scan/latch
   stimulus check of the eight sequential models. Both must print PASS for all
   cells before any synthesis is run.
3. `src/e3_extension/synth_with_cell_models.py` — one (design, library) job:
   read models + netlist, `synth`, `dfflibmap`, `abc -liberty`, `write_json`,
   feature extraction.
4. `src/e3_extension/run_e3_extension.py` — the 30 × 3 batch driver.

Inputs required (not redistributable here): the Trust-Hub archives, the
cleaned generic-synthesis work directories, and the three Liberty files pinned
in `tools/lib_manifest.sha256`.

## C. E3 extension: batch synthesis and analysis (src/e3x)

The extended cross-library set (118 labeled netlists × 3 libraries = 354 rows,
eleven base designs) is produced by:

1. `src/e3_extension/build_cell_models.py` + `verify_cell_models_*.py` — as in B.
2. `src/e3x/e3x_synth.py <family> <design> <tech>` — one (design, library) job
   (functional cell models + `synth` / `dfflibmap` / `abc -liberty` / `write_json`).
3. `src/e3x/e3x_features.py` — v1+v2 feature rows from all `features.json`.
4. `src/e3x/merge_ext.py` — merges with `results/features_multilib_v2.csv`.
5. `src/e3x/e3x_e3table.py` — E3 table numbers (v1/v2 AUC, Spec, Bal; RF/XGB).
6. `src/e3x/e3x_signflip.py` — per-group table + exact sign-flip (2^11).

Outputs shipped: `results/features_multilib_v2_ext.csv`,
`results/e3x_e3table.csv`, `results/e3x_signflip.csv`,
`results/e3x_per_group.csv`, `results/e3x_signflip_report.txt`.
The scripts carry the machine's absolute `ROOT`; set it to your checkout first.

## D. Re-scoring a published tool (HW2VEC/GNN4TJ, Sec. V-F)

Reported, not regenerated from this package (level iii of the paper's artifact
statement). The framework is an external checkout
(`git clone https://github.com/AICPS/hw2vec tools/hw2vec`); the driver is
`tools/hw2vec_ladder/`:

1. `prep_corpus.py` — E1 pairs from the cleaned netlists, **whole pairs excluded
   together** (87 pairs survive).
2. `extract_graphs.py` — AST graphs cached for the GNN4TJ model.
3. `run_ladder.py` — the four surviving rungs x 2 seeds x 200 epochs.

Every deviation from the released code (double log-softmax, feature scale,
mean readout, AST instead of DFG graphs) and its consequence is itemised in
`DEVIATIONS.md`, together with what those deviations do and do not license.
Shipped outputs: `results/ladder_results.csv`, `results/ladder_report.txt`.
The trained checkpoint and the 4.2 MB training log are not shipped.

## E. What is *not* verified from this package

Raw extraction/synthesis, the GNN stage and the AIG feature extraction need the
upstream netlists and were not clean-room re-run here; their numbers come from
the bundled `results/` files. The paper states this explicitly in the Artifact
and Reproducibility Statement.

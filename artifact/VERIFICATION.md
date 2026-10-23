# VERIFICATION — clean-directory reproduction check

## v3 check (2026-10-15, current code and reports) — COMPLETE, all green

Runner: artifact copied to a fresh directory
(`%LOCALAPPDATA%\Temp\artifact_v3_check`), bundled files only, same pinned
stack, `n_jobs=1`. Four stages (`c1_protocol` v2cd, `e2_subset --seed
20261012`, `lit_baselines`, `joint_holdout`) all exited 0. Output-by-output
comparison against the bundled `results/`:

| File | Result |
|---|---|
| `c1_results_v2cd.csv` | IDENTICAL |
| `c1_delta_auc_v2cd.csv` | IDENTICAL |
| `lit_baseline_results.csv` | IDENTICAL |
| `joint_holdout_results.csv` | IDENTICAL |
| `e2_subset_results.csv` | IDENTICAL |
| `c1_report_v2cd.txt` | identical except the data-source header line |
| `lit_baseline_report.txt` | identical except the data-source header line |
| `joint_holdout_report.txt` | identical except the data-source header line |
| `e2_subset_report.txt` | identical except the data-source header line |

Verdict: the paper's E1/E2/E3 tables (v2cd caliber: cluster-bootstrap CIs
resampled on base-design groups regardless of split), the paired ΔAUC
significance tests, the literature-pipeline baselines (decision tree+PCA,
kNN k=5), the design × library joint holdout, and the E2 TRIT/non-TRIT
decomposition all regenerate byte-for-byte from the bundled feature matrices.

Process note for reviewers: an earlier comparison pass reported false CSV
diffs caused by a text-scrubbing step that stripped CR from CRLF files; the
CSVs contain no machine paths and were restored to their pristine form before
the final comparison above.

## v2 check (2026-10-14, superseded caliber v2cc) — COMPLETE, all green

Date: 2026-10-14 (v2; supersedes the earlier v1 check below). Runner: artifact
copied to a fresh directory (`%LOCALAPPDATA%\Temp\artifact_test2`), no access to
the project tree; the only inputs were the files bundled in this package.
Interpreter: Python 3.11 with the pinned stack (scikit-learn 1.9.1, XGBoost
3.2.0, pandas 2.3.3, numpy 2.2.6). All model runs fix `n_jobs=1` and
`random_state=SEED`, so outputs are byte-deterministic by design.

## v2 check (current code and reports) — COMPLETE, all green

Runner: fresh directory, bundled files only, pinned stack, `n_jobs=1`.
All four stages exited 0 (`c1_protocol` v2cc, `e2_subset`, `aig_detect`,
`family_table`). Output-by-output comparison against the bundled `results/`:

| File | Result |
|---|---|
| `c1_report_v2cc.txt` | IDENTICAL |
| `c1_results_v2cc.csv` | IDENTICAL |
| `e2_subset_report.txt` | IDENTICAL |
| `e2_subset_results.csv` | IDENTICAL |
| `aig_detect_report.txt` | IDENTICAL |
| `aig_detect_results.csv` | IDENTICAL |
| `family_xgb.csv` | IDENTICAL |
| `family_xgb.txt` | identical except 2 header lines naming the output path (`out\` vs `experiments\c2\`); all numbers byte-identical |

Verdict: the paper's E1/E2/E3 tables, the per-family table, the E2 subset
decomposition, and the four-representation detection comparison regenerate
byte-for-byte from the bundled feature matrices. With `n_jobs=1` and the fixed
seed the pipeline is fully deterministic.

## v1 check (superseded — kept for the record)

The earlier check ran the previous code revision (row-level bootstrap,
`n_jobs=-1`) against the then-current `c1_report_v2ci.txt`: 32 of 33
protocol×model rows reproduced byte-identically and one (E1 LOFO, RF, AUC)
differed by 0.007 AUC (inside its CI), traced to multi-threaded RandomForest
non-determinism. Two consequences fed into the current revision: the final
number runs now pin `n_jobs=1`, and the CI unit is a cluster (base-design)
resample rather than a row resample.

## Known join pitfall (fixed, recorded for reviewers)

`src/aig_detect.py` joins four feature tables on
`(family, variant, source, tech)`. The AIG exports encode the label in the
variant suffix (`__TjIn`/`__TjFree`) and carry 33 unlabeled rows, while the
feature CSVs use a separate `source` column with three `TnIn` spellings. The
script normalizes both sides identically and **fails fast** if any row does not
join (an earlier revision silently produced an all-zero feature matrix that
yielded degenerate 0.5 AUCs — caught before use, see git history/notes).

## Not verified from this package (requires external inputs)

- Stages 1–3 (raw Trust-Hub extraction, synthesis, feature generation): needs
  the public Trust-Hub archives and the Liberty libraries pinned in
  `tools/lib_manifest.sha256`.
- Stage 6 (GNN) and stage 7 (AIG feature extraction itself): consume stage-2
  netlists and were not re-run from this package. Their reported numbers come
  from `results/gnn_*` and `results/aig_report.txt` / `results/aig{A,B}_features.csv`
  (the detection comparison in stage 5b does run from the bundled AIG CSVs).
- The HW2VEC/GNN4TJ re-scoring (Sec. V-F) is likewise reported, not regenerated:
  it needs an external checkout of the tool. Its deviations from the released
  code are itemised in `DEVIATIONS.md`, and its numbers come from
  `results/ladder_results.csv` / `results/ladder_report.txt`.

- Level (i) also covers the single-feature structural baseline
  (`src/signature_baseline.py` -> `results/signature_baseline.csv`): it runs
  from the bundled v2 feature matrix alone and reproduces the numbers quoted in
  Sec. V ("What the residual signal is"). Verified by running the shipped
  script from inside the package: the generated CSV is byte-identical to the
  one shipped in `results/`.

- Four-library E3 (Table V): `src/merge_4lib.py` rebuilds the 516-row matrix
  from the bundled per-library feature files and reproduces 472 labelled rows
  (11 base designs x 4 libraries). The sign-flip p-values (RF 0.28 / XGB 0.61 /
  MLP 0.38) and the CI half-widths (0.11-0.20) in the caption regenerate with
  `e3x_e3table.py`, `e3x_signflip.py`, `e3_4lib_ci.py`. Synthesis of the fourth
  library itself needs Yosys + the GF180 liberty, so those netlists are
  level (ii); their feature matrices are bundled, so the analysis is level (i).

## Golden-class extraction audit (added with the revised numbers)

The E2 golden class originally carried a single netlist: all eight TRIT
`original_designs/*.v` were synthesized in one working directory, and Yosys'
automatic top-level selection returned the largest design (`s35932`) for every
row. The twelve TRIT golden rows therefore held one identical feature vector,
and every E2 statistic measured that artifact.

`tools/audit_golden.py` is the check that catches it, and it now runs as part of
the pipeline:

    python3 tools/audit_golden.py results/features_all_v2.csv

Expected: `total golden rows=72 -> 21 unique feature vectors` and `PASS`, with
`TRIT-TC rows=8 unique_vectors=8`, `TRIT-TS rows=4 unique_vectors=4`.
Before the correction the same script reports 12 golden rows collapsing to 1
unique vector and exits non-zero.

The golden rows in `results/features_all.csv` and `results/features_all_v2.csv`
have been regenerated with one synthesis per original design
(`hierarchy -top <module>`); variants are unchanged.

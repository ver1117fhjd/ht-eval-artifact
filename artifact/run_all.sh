#!/usr/bin/env bash
# One-command reproduction of the paper's main tables (no external data needed).
# See REPRODUCE.md for what runs with external inputs and what is reported-only.
set -euo pipefail
cd "$(dirname "$0")"

python src/c1_protocol.py --all-csv results/features_all_v2.csv \
    --multilib-csv results/features_multilib_v2.csv --out-dir out --tag v2cd
python src/family_table.py --all-csv results/features_all_v2.csv --out-dir out
python src/pca_leak_fig.py --all-csv results/features_all_v2.csv
python src/lit_baselines.py --out-dir out
python src/joint_holdout.py --out-dir out
python src/e2_subset.py --all-csv results/features_all_v2.csv --out-dir out
python src/permutation_test.py --out-dir out
python src/r7_e3_signflip.py --out-dir out

echo "DONE: outputs in ./out ; compare number-by-number against results/ (VERIFICATION.md)"

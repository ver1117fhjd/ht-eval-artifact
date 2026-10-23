# Artifact: leakage-free evaluation of gate-level ML hardware-Trojan detectors

This repository is the anonymized artifact for the paper **"Leakage-Free Evaluation of
Machine-Learning Hardware Trojan Detection: A Cross-Library Protocol and Auditable
Operator Representation"** (under review).

Everything lives in [`artifact/`](artifact/):

- `artifact/README.md` — layout and the three-level reproduction contract
- `artifact/REPRODUCE.md` — step-by-step commands
- `artifact/results/` — feature matrices and result tables behind every number in the paper
- `artifact/src/` — protocol ladder (C1), operator normalisation (C2), baselines, GNN/AIG ablations
- `artifact/tools/` — `audit_golden.py` (golden-class degeneracy audit), packaging and manifests
- `artifact/DEVIATIONS.md`, `artifact/VERIFICATION.md` — deviations and verification notes

Start with `artifact/README.md`. Stages 4-5 reproduce from the bundled feature matrices
with no external downloads; the synthesis and library-mapping stages need Trust-Hub
archives and the pinned standard-cell libraries listed in `artifact/tools/lib_manifest.sha256`.

No author, institution or contact information is included, by design.

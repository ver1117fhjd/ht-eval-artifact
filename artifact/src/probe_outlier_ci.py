# -*- coding: utf-8 -*-
"""给 framework_norm_probe 里那个异常高的格子（profile-only, pairs->TRIT, XGB=0.945）
做按组 bootstrap：12 个金样行的 AUC 到底稳不稳。"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(r"PROJECT_ROOT")
spec = importlib.util.spec_from_file_location("c1", ROOT / "src/c1_protocol.py")
c1 = importlib.util.module_from_spec(spec)
sys.argv = [sys.argv[0]]
spec.loader.exec_module(c1)
c1.ALL_CSV = str(ROOT / "experiments/synth_work/features_all_v2.csv")
full = c1.load_full()
FEATS = c1.feature_cols(full)
GEOM = [c for c in FEATS if c.startswith(("comb_depth", "comb_opdepth", "fanin_", "fanout_",
                                          "n_cells", "n_ops", "n_seq", "in_width", "out_width"))]
PROFILE = [c for c in FEATS if c not in GEOM]
is_trit = full["family"].str.startswith("TRIT").values
trit = full[is_trit].reset_index(drop=True)
nt = full[~is_trit].reset_index(drop=True)

Xtr, ytr = nt[PROFILE].values.astype(float), nt["label"].values
Xte, yte, gte = trit[PROFILE].values.astype(float), trit["label"].values, trit["group"].values
for mname in ("XGB", "RF", "MLP"):
    m = c1.make_models()[mname]
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p)
    groups = np.unique(gte)
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(1000):
        pick = rng.choice(groups, len(groups), replace=True)
        idx = np.concatenate([np.where(gte == gg)[0] for gg in pick])
        if len(set(yte[idx])) < 2:
            continue
        boot.append(roc_auc_score(yte[idx], p[idx]))
    boot = np.array(boot)
    print(f"{mname}: AUC={auc:.3f}  按组bootstrap 95%CI=[{np.percentile(boot,2.5):.3f}, "
          f"{np.percentile(boot,97.5):.3f}]  跨过去=含0.5? {'YES' if np.percentile(boot,2.5)<=0.5<=np.percentile(boot,97.5) else 'no'}")

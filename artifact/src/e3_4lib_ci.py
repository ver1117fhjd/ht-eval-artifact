# -*- coding: utf-8 -*-
"""四库 E3：每折 AUC + 聚类 bootstrap 95% CI 半宽（用于 Table V caption 的 0.xx–0.xx）。"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(r"D:/10.7")
spec = importlib.util.spec_from_file_location("c1", ROOT / "src/c1_protocol.py")
c1 = importlib.util.module_from_spec(spec)
sys.argv = [sys.argv[0]]
spec.loader.exec_module(c1)

EXT = ROOT / "experiments/e3x/features_multilib_v2_ext_4lib.csv"
c1.MULTILIB_CSV = str(EXT)
df = c1.load_multilib().reset_index(drop=True)
FEATS = c1.feature_cols(df)
V1 = ["cell_class_entropy", "comb_depth_max", "comb_depth_mean", "const_bit_frac",
      "fanin_max", "fanin_mean", "fanin_std", "fanout_max", "fanout_mean", "fanout_std",
      "frac_and", "frac_buf", "frac_complex", "frac_inv", "frac_mux", "frac_or",
      "frac_other", "frac_seq", "frac_xor", "in_width", "n_cells", "n_seq", "out_width"]

y = df["label"].values
g = df["group"].values
techs = df["tech"].values
print(f"n={len(df)} pos={y.sum()} groups={df.group.nunique()} techs={sorted(set(techs))}")

rows = []
for fname, cols in (("v1", [c for c in V1 if c in df.columns]), ("v2", FEATS)):
    X = df[cols].values.astype(float)
    for mname in ("RF", "XGB"):
        for held in sorted(set(techs)):
            tr, te = techs != held, techs == held
            m = c1.make_models()[mname]
            m.fit(X[tr], y[tr])
            prob = m.predict_proba(X[te])[:, 1]
            auc = roc_auc_score(y[te], prob)
            ci = c1.boot_ci(y[te], prob, g[te])
            lo, hi = ci["ROC_AUC"][1], ci["ROC_AUC"][2]
            rows.append(dict(feat=fname, model=mname, held=held, n=int(te.sum()),
                             auc=auc, ci_lo=lo, ci_hi=hi, half=(hi - lo) / 2))
res = pd.DataFrame(rows)
res.to_csv(ROOT / "experiments/e3x/e3table_4lib_ci.csv", index=False)
print(res.pivot_table(index="held", columns=["feat", "model"], values="auc").round(3).to_string())
print("\nCI 半宽（各折）：")
print(res.pivot_table(index="held", columns=["feat", "model"], values="half").round(3).to_string())
print("\n半宽范围: %.3f – %.3f" % (res.half.min(), res.half.max()))
print("v2 半宽范围: %.3f – %.3f" % (res[res.feat == "v2"].half.min(), res[res.feat == "v2"].half.max()))

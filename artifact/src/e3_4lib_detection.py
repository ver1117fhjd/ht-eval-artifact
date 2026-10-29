# -*- coding: utf-8 -*-
"""e3_4lib_detection.py —— 四库 E3 的检测侧读数（AUC / Spec / Sens / Bal@0.5）。

补 e3_4lib_ci.py 只输出 AUC+CI 的缺口：Table V 里每库的 v2 Spec、v2 Bal，
以及 pooled 行，之前没有落盘文件。口径与 e3_4lib_ci.py 完全一致：
同一份 features_multilib_v2_ext_4lib.csv、同一个 c1.make_models()、
同样的 v1（23 列）/v2（c1.feature_cols）特征集；pooled 用
StratifiedKFold(5, shuffle, SEED)，与 c1_protocol 的 pooled 档一致。

用法: python e3_4lib_detection.py
输出: results/e3_4lib_detection.csv
"""
import importlib.util
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]   # artifact root (sanitized)
spec = importlib.util.spec_from_file_location("c1", ROOT / "src/c1_protocol.py")
c1 = importlib.util.module_from_spec(spec)
sys.argv = [sys.argv[0]]
spec.loader.exec_module(c1)

EXT = ROOT / "results/features_multilib_v2_ext_4lib.csv"
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
print("n=%d pos=%d groups=%d techs=%s" % (len(df), y.sum(), df.group.nunique(), sorted(set(techs))))


def det(y_true, prob):
    """0.5 阈值下的 Spec / Sens / Bal。Trojan=1 为正类。"""
    pred = (prob >= 0.5).astype(int)
    pos, neg = y_true == 1, y_true == 0
    sens = float((pred[pos] == 1).mean()) if pos.any() else float("nan")
    spec = float((pred[neg] == 0).mean()) if neg.any() else float("nan")
    return sens, spec, (sens + spec) / 2.0


rows = []
for fname, cols in (("v1", [c for c in V1 if c in df.columns]), ("v2", FEATS)):
    X = df[cols].values.astype(float)
    for mname in ("RF", "XGB"):
        # ---- 留一库 ----
        for held in sorted(set(techs)):
            tr, te = techs != held, techs == held
            m = c1.make_models()[mname]
            m.fit(X[tr], y[tr])
            prob = m.predict_proba(X[te])[:, 1]
            sens, spec, bal = det(y[te], prob)
            rows.append(dict(feat=fname, model=mname, held=held, n=int(te.sum()),
                             auc=float(roc_auc_score(y[te], prob)),
                             spec=spec, sens=sens, bal=bal))
        # ---- pooled 5 折（与 c1_protocol 同 splitter/seed）----
        prob = cross_val_predict(c1.make_models()[mname], X, y,
                                 cv=StratifiedKFold(5, shuffle=True,
                                                    random_state=c1.SEED),
                                 method="predict_proba")[:, 1]
        sens, spec, bal = det(y, prob)
        rows.append(dict(feat=fname, model=mname, held="pooled", n=int(len(y)),
                         auc=float(roc_auc_score(y, prob)),
                         spec=spec, sens=sens, bal=bal))

res = pd.DataFrame(rows)
out = ROOT / "results/e3_4lib_detection.csv"
res.to_csv(out, index=False)
print("\n写入", out, len(res), "行")
print(res.pivot_table(index="held", columns=["feat", "model"], values="auc").round(3).to_string())
print("\nSpec:")
print(res.pivot_table(index="held", columns=["feat", "model"], values="spec").round(2).to_string())
print("\nBal:")
print(res.pivot_table(index="held", columns=["feat", "model"], values="bal").round(2).to_string())

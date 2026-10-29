# -*- coding: utf-8 -*-
"""framework-disjoint 试探：把"插入几何"从特征里拿掉，跨框架能不能迁移？

四套特征：
  all-32        全部 v2 特征（论文基线：TRIT→pairs 0.52/0.49/0.50 近随机）
  profile-only  去掉深度/规模类（comb_depth_*, comb_opdepth_*, n_cells, n_ops, n_seq,
                in_width, out_width, fanin_*, fanout_*）——只留占比/算子画像（可部署）
  geometry-only 只留上面被去掉的那些（对照：看几何本身是否就是全部信号）
  zs-per-group  在"每个基准设计内部"做 z-score（transductive：审计场景可用，
                单网表部署不可用，标注清楚）

协议：framework-disjoint 双向（TRIT→pairs、pairs→TRIT）+ 组内 grouped-by-base 作对照。
输出：experiments/c2/framework_norm_probe.csv / .txt
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

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
print(f"特征 {len(FEATS)}：几何类 {len(GEOM)}，画像类 {len(PROFILE)}")

is_trit = full["family"].str.startswith("TRIT").values
trit = full[is_trit].reset_index(drop=True)
nt = full[~is_trit].reset_index(drop=True)


def zs_within_group(df):
    d = df.copy()
    d[FEATS] = d[FEATS].astype(float)
    for g, idx in d.groupby("group").groups.items():
        sub = d.loc[idx, FEATS].astype(float)
        mu, sd = sub.mean(), sub.std(ddof=0).replace(0, 1.0)
        d.loc[idx, FEATS] = ((sub - mu) / sd).values
    return d


def fit_eval(train, test, cols, mname):
    m = c1.make_models()[mname]
    Xtr, ytr = train[cols].values.astype(float), train["label"].values
    Xte, yte = test[cols].values.astype(float), test["label"].values
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    return roc_auc_score(yte, p)


def grouped_auc(df, cols, mname, seed=c1.SEED):
    X, y, g = df[cols].values.astype(float), df["label"].values, df["group"].values
    pred = np.full(len(df), np.nan)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups=g):
        m = c1.make_models()[mname]
        m.fit(X[tr], y[tr])
        pred[te] = m.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, pred)


rows = []
trit_zs, nt_zs = zs_within_group(trit), zs_within_group(nt)
for mname in ("RF", "XGB", "MLP"):
    for tag, cols in (("all-32", FEATS), ("profile-only", PROFILE), ("geometry-only", GEOM)):
        rows.append(dict(model=mname, feats=tag, protocol="TRIT->pairs",
                         auc=fit_eval(trit, nt, cols, mname)))
        rows.append(dict(model=mname, feats=tag, protocol="pairs->TRIT",
                         auc=fit_eval(nt, trit, cols, mname)))
        rows.append(dict(model=mname, feats=tag, protocol="TRIT grouped",
                         auc=grouped_auc(trit, cols, mname)))
        rows.append(dict(model=mname, feats=tag, protocol="pairs grouped",
                         auc=grouped_auc(nt, cols, mname)))
    # per-design z-score（transductive）
    rows.append(dict(model=mname, feats="zs-per-group", protocol="TRIT->pairs",
                     auc=fit_eval(trit_zs, nt_zs, FEATS, mname)))
    rows.append(dict(model=mname, feats="zs-per-group", protocol="pairs->TRIT",
                     auc=fit_eval(nt_zs, trit_zs, FEATS, mname)))
    rows.append(dict(model=mname, feats="zs-per-group", protocol="TRIT grouped",
                     auc=grouped_auc(trit_zs, FEATS, mname)))
    rows.append(dict(model=mname, feats="zs-per-group", protocol="pairs grouped",
                     auc=grouped_auc(nt_zs, FEATS, mname)))

res = pd.DataFrame(rows)
res.to_csv(ROOT / "experiments/c2/framework_norm_probe.csv", index=False)
piv = res.pivot_table(index=["feats", "protocol"], columns="model", values="auc").round(3)
txt = ("framework-disjoint 特征归一化试探（E2 语料；TRIT 911 行 / pairs 122 行）\n"
       "注：zs-per-group 是 transductive（用整族统计量），只在审计场景成立\n\n"
       + piv.to_string())
(ROOT / "experiments/c2/framework_norm_probe.txt").write_text(txt, encoding="utf-8")
print(txt)

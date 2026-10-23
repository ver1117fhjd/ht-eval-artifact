# -*- coding: utf-8 -*-
"""单特征结构基线：一组"框架签名"到底需要几个特征？

问题：TRIT 子集在 group-by-base 下仍 0.92-1.00，是不是说明"可迁移的检测信号"存在？
做法：只用 comb_depth_max（最大组合深度，一个标量）训练，跑同一协议阶梯，
      与 32 维全特征对比；再搬到非 TRIT 与全量集上做对照。
口径：直接 import c1_protocol，模型/分割/指标与论文主表一致（显式切 v2 数据源）。
路径：项目内与 artifact 包内都能跑（结果目录自动选择）。
输出：signature_baseline.csv / signature_baseline_report.txt
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]          # 项目根 或 artifact 根
SEEDS = [20261012, 20261013, 20261014]
SCALAR = "comb_depth_max"

CANDIDATES = [
    ROOT / "experiments/synth_work/features_all_v2.csv",
    ROOT / "results/features_all_v2.csv",
]
V2 = next((p for p in CANDIDATES if p.exists()), None)
assert V2 is not None, f"找不到 features_all_v2.csv，试过 {CANDIDATES}"
MLV = [ROOT / "experiments/synth_work/features_multilib_v2.csv",
       ROOT / "results/features_multilib_v2.csv"]
ML = next((p for p in MLV if p.exists()), None)
OUT = ROOT / "experiments/c2" if (ROOT / "experiments").exists() else ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location("c1", ROOT / "src/c1_protocol.py")
c1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c1)
# ⚠️ c1_protocol 的默认 --all-csv 是 v1；必须显式切 v2，否则数字静默偏离主表
c1.ALL_CSV = str(V2)
assert "v2" in Path(c1.ALL_CSV).name

full = c1.load_full()
FCOLS = c1.feature_cols(full)
assert len(FCOLS) == 32, len(FCOLS)
trit = full[full["family"].str.startswith("TRIT")].reset_index(drop=True)
nt = full[~full["family"].str.startswith("TRIT")].reset_index(drop=True)
assert len(full) == 1033 and len(trit) == 911 and len(nt) == 122


def run(df, cols, seed, protocol):
    X = df[cols].values.astype(float)
    y = df["label"].values
    g = df["group"].values
    pred = np.full(len(df), np.nan)
    if protocol == "grouped":
        sp = StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups=g)
    else:
        sp = StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y)
    for tr, te in sp:
        m = c1.make_models()["XGB"]
        m.fit(X[tr], y[tr])
        pred[te] = m.predict_proba(X[te])[:, 1]
    th = (pred >= 0.5).astype(int)
    return dict(AUC=roc_auc_score(y, pred),
                Spec=float((th[y == 0] == 0).mean()),
                Sens=float((th[y == 1] == 1).mean()))


rows = []
for setname, df in (("TRIT", trit), ("non-TRIT", nt), ("E2", full)):
    for protocol in ("pooled", "grouped"):
        for fname, cols in (("all-32", FCOLS), (SCALAR, [SCALAR])):
            vals = [run(df, cols, s, protocol) for s in SEEDS]
            rows.append(dict(set=setname, protocol=protocol, features=fname, n=len(df),
                             groups=df["group"].nunique(),
                             AUC=np.mean([v["AUC"] for v in vals]),
                             AUC_sd=np.std([v["AUC"] for v in vals]),
                             Spec=np.mean([v["Spec"] for v in vals]),
                             Sens=np.mean([v["Sens"] for v in vals])))

res = pd.DataFrame(rows)
res.to_csv(OUT / "signature_baseline.csv", index=False)

L = ["单特征结构基线（XGB，%d seeds 均值）：all-32 vs %s" % (len(SEEDS), SCALAR),
     "口径 = c1_protocol + %s，与论文主表一致" % V2.name,
     "模型/分割与论文一致；Spec/Sens 用固定 0.5 阈值", "",
     res.to_string(index=False, float_format=lambda v: f"{v:.3f}")]

if ML is not None:
    ml = pd.read_csv(ML)
    piv = ml.pivot_table(index=["family", "variant", "module"], columns="tech",
                         values=SCALAR).dropna()
    ratio = (piv.max(axis=1) / piv.min(axis=1).replace(0, np.nan)).dropna()
    L += ["", f"{SCALAR} 跨库漂移（99 设计×3库，max/min）：中位 {ratio.median():.2f}×，"
              f"最大 {ratio.max():.2f}×，n={len(ratio)}"]
txt = "\n".join(L)
(OUT / "signature_baseline_report.txt").write_text(txt, encoding="utf-8")
print(txt)

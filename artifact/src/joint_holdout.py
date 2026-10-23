# -*- coding: utf-8 -*-
"""joint_holdout.py —— R3/R4 整改 P1-7：design × library 联合留出
============================================================
E3 的 LOLO 只隔离库轴：训练侧仍见过测试设计在其他库下的版本。
本脚本做严格联合留出：test = 设计 d @ 库 L 的每一行；
train = {其他设计} × {其他两个库}（设计轴和库轴同时隔离）。

每行得到一个 out-of-sample 预测，再按 base design 聚类算 pooled 指标
+ cluster bootstrap CI——这是"从没见过的设计+没见过的库"的保守口径。

用法：
  python src/joint_holdout.py
输出：
  experiments/c2/joint_holdout_report.txt / joint_holdout_results.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

import c1_protocol as c1

ROOT = Path(__file__).resolve().parents[1]
_ap = argparse.ArgumentParser()
_ap.add_argument("--all-csv",
                 default=str(ROOT / "experiments" / "synth_work" / "features_all_v2.csv"))
_ap.add_argument("--multilib-csv",
                 default=str(ROOT / "experiments" / "synth_work" / "features_multilib_v2.csv"))
_ap.add_argument("--out-dir", default=str(ROOT / "experiments" / "c2"))
_args = _ap.parse_args()
c1.ALL_CSV = _args.all_csv
c1.MULTILIB_CSV = _args.multilib_csv
OUT = Path(_args.out_dir)
OUT.mkdir(parents=True, exist_ok=True)


def make_models():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier
    return {
        "RF": RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                     random_state=c1.SEED, n_jobs=1),
        "XGB": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                             subsample=0.9, colsample_bytree=0.9,
                             random_state=c1.SEED, n_jobs=1,
                             eval_metric="logloss"),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                          max_iter=500, n_iter_no_change=30,
                          random_state=c1.SEED)),
    }


def main():
    df = c1.load_multilib().reset_index(drop=True)
    X = df[c1.feature_cols(df)].values
    y = df["label"].values
    groups = df["group"].values       # base design
    techs = df["tech"].values
    n = len(df)
    print(f"[joint] n={n} pos={int(y.sum())} designs={len(set(groups))} "
          f"libs={sorted(set(techs))}", flush=True)

    rows = []
    for mname, model in make_models().items():
        prob = np.zeros(n)
        for i in range(n):
            tr = (groups != groups[i]) & (techs != techs[i])
            te = np.zeros(n, dtype=bool)
            te[i] = True
            model.fit(X[tr], y[tr])
            prob[i] = model.predict_proba(X[te])[:, 1][0]
        m = c1.metrics(y, prob)
        ci = c1.boot_ci(y, prob, groups)
        rows.append({
            "protocol": "design-x-library joint holdout", "model": mname,
            "n": n, "n_pos": int(y.sum()),
            "AP": m["AP"], "AP_lo": ci["AP"][1], "AP_hi": ci["AP"][2],
            "ROC_AUC": m["ROC_AUC"],
            "AUC_lo": ci["ROC_AUC"][1], "AUC_hi": ci["ROC_AUC"][2],
            "F1": m["F1"], "Sens": m["Sens"], "Spec": m["Spec"],
            "Spec_lo": ci["Spec"][1], "Spec_hi": ci["Spec"][2],
            "BalAcc": m["BalAcc"], "Gmean": m["Gmean"]})
        print(f"  {mname}: AUC={m['ROC_AUC']:.3f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "joint_holdout_results.csv", index=False)

    lines = ["design × library 联合留出（R3/R4 整改 P1-7）",
             f"数据源: {c1.MULTILIB_CSV}",
             "test=设计d@库L 每行；train=其他设计×其他两库（设计轴+库轴双隔离）",
             "CI=cluster bootstrap 1000，聚类=base design", ""]
    for _, r in out.iterrows():
        lines.append(
            f"{r['model']:4s} n={r['n']:4d} "
            f"AUC={r['ROC_AUC']:.3f} [{r['AUC_lo']:.3f},{r['AUC_hi']:.3f}] "
            f"AP={r['AP']:.3f} [{r['AP_lo']:.3f},{r['AP_hi']:.3f}] "
            f"Spec={r['Spec']:.3f} [{r['Spec_lo']:.3f},{r['Spec_hi']:.3f}] "
            f"Bal={r['BalAcc']:.3f}")
    txt = "\n".join(lines)
    (OUT / "joint_holdout_report.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""e2_subset.py —— R2 整改 P0-1：E2 分解分析（评审 A "最大硬伤" 的正解）。

问题：E2 group-by-base AUC 0.79-0.87 为什么没塌到随机？
假说：高分全部由 TRIT（单一自动插桩框架）扛；非 TRIT 部分（=E1 全部 122 行）塌到随机。
做法：E2 按 TRIT / 非 TRIT 拆子集，各跑 group-by-base（对照 pooled），
     cluster bootstrap（按 base design 重采）95% CI，n_jobs=1 钉死确定性。
输出：experiments/c1/e2_subset_report.txt + e2_subset_results.csv
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
import c1_protocol as c1

_ap = argparse.ArgumentParser()
_ap.add_argument("--all-csv", default="results/features_all_v2.csv")
_ap.add_argument("--out-dir", default="out")
_ap.add_argument("--seed", type=int, default=c1.SEED)   # P1-6 三种子补跑用
_args = _ap.parse_args()
c1.ALL_CSV = _args.all_csv
OUT = _args.out_dir
os.makedirs(OUT, exist_ok=True)
SEED = _args.seed
N_BOOT = 1000


def make_models():
    return {
        "RF": RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                     random_state=SEED, n_jobs=1),
        "XGB": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                             subsample=0.9, colsample_bytree=0.9,
                             random_state=SEED, n_jobs=1, eval_metric="logloss"),
        "MLP": make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(64, 32),
                                           activation="relu", max_iter=500,
                                           n_iter_no_change=30,
                                           random_state=SEED)),
    }


def cluster_boot_ci(y, prob, groups, n=N_BOOT, seed=SEED):
    """按 base design（组）重采的 bootstrap 95% CI。"""
    y = np.asarray(y); prob = np.asarray(prob); groups = np.asarray(groups)
    uniq = np.unique(groups)
    gidx = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    out = {}
    for name in ("ROC_AUC", "Spec", "Sens", "BalAcc", "Gmean"):
        vals = []
        for _ in range(n):
            pick = rng.integers(0, len(uniq), len(uniq))
            idx = np.concatenate([gidx[uniq[p]] for p in pick])
            if len(np.unique(y[idx])) < 2:
                continue
            vals.append(c1.metrics(y[idx], prob[idx])[name])
        out[name] = (float(np.nanmean(vals)), float(np.nanpercentile(vals, 2.5)),
                     float(np.nanpercentile(vals, 97.5)))
    return out


def run_one(tag, df, prot, use_groups):
    X = df[c1.feature_cols(df)].values
    y = df["label"].values
    groups = df["group"].values
    rows = []
    cv = (StratifiedGroupKFold(5, shuffle=True, random_state=SEED) if use_groups
          else StratifiedKFold(5, shuffle=True, random_state=SEED))
    for mname, model in make_models().items():
        prob = cross_val_predict(model, X, y, cv=cv, groups=groups if use_groups else None,
                                 method="predict_proba")[:, 1]
        m = c1.metrics(y, prob)
        ci = cluster_boot_ci(y, prob, groups)
        rows.append({"subset": tag, "protocol": prot, "model": mname,
                     "n": len(y), "n_pos": int(y.sum()),
                     "n_groups": len(set(groups)),
                     "AUC": m["ROC_AUC"], "AUC_lo": ci["ROC_AUC"][1], "AUC_hi": ci["ROC_AUC"][2],
                     "Spec": m["Spec"], "Spec_lo": ci["Spec"][1], "Spec_hi": ci["Spec"][2],
                     "Sens": m["Sens"], "BalAcc": m["BalAcc"], "Gmean": m["Gmean"]})
        print(f"[{tag}] {prot:22s} {mname:4s} AUC={m['ROC_AUC']:.3f} "
              f"[{ci['ROC_AUC'][1]:.3f},{ci['ROC_AUC'][2]:.3f}] Spec={m['Spec']:.3f}", flush=True)
    return rows


def main():
    df2 = c1.load_full()
    is_trit = df2["family"].str.startswith("TRIT")
    sub_trit = df2[is_trit].copy()
    sub_non = df2[~is_trit].copy()
    print(f"[data] E2={len(df2)} | TRIT={len(sub_trit)} pos={sub_trit.label.sum()} "
          f"groups={sub_trit.group.nunique()} | non-TRIT={len(sub_non)} pos={sub_non.label.sum()} "
          f"groups={sub_non.group.nunique()}", flush=True)

    rows = []
    for tag, sub in (("E2-TRIT-only", sub_trit), ("E2-non-TRIT(=E1)", sub_non)):
        rows += run_one(tag, sub, "pooled-random-5fold", use_groups=False)
        rows += run_one(tag, sub, "group-by-base-5fold", use_groups=True)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "e2_subset_results.csv"), index=False)
    lines = [f"E2 子集分解（TRIT vs 非 TRIT），cluster bootstrap n=1000, seed={SEED}",
             "数据源: features_all_v2.csv", ""]
    for _, r in out.iterrows():
        lines.append(f"{r['subset']:18s} {r['protocol']:22s} {r['model']:4s} n={r['n']:4d} "
                     f"AUC={r['AUC']:.3f} [{r['AUC_lo']:.3f},{r['AUC_hi']:.3f}] "
                     f"Spec={r['Spec']:.3f} [{r['Spec_lo']:.3f},{r['Spec_hi']:.3f}] "
                     f"Sens={r['Sens']:.3f} Bal={r['BalAcc']:.3f}")
    txt = "\n".join(lines)
    open(os.path.join(OUT, "e2_subset_report.txt"), "w", encoding="utf-8").write(txt)
    print("\n" + txt)


if __name__ == "__main__":
    main()

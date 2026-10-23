# -*- coding: utf-8 -*-
"""permutation_test.py —— R5-M5：标签置换检验（AUC=chance 论断的 null 分布）
================================================================
主假设（论文声明的三个）对应的置换检验：
  H1 variant→base 掉落（E1）
  H2 pooled→base 掉落（E1/E2）
  H3 LOLO 特异性摆动（E3）
本脚本给 null 分布：打乱标签后在 group-by-base / pooled 协议下重跑，
得到 AUC 的零分布，检验观测值是否可由"标签无信息"解释。

用法：D:/10.7/.venv/Scripts/python.exe src/permutation_test.py [--n-perm 100]
输出：experiments/c2/perm_test_report.txt / perm_test_results.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict

import c1_protocol as c1

ROOT = Path(__file__).resolve().parents[1]
_ap = argparse.ArgumentParser()
_ap.add_argument("--all-csv",
                 default=str(ROOT / "experiments" / "synth_work" / "features_all_v2.csv"))
_ap.add_argument("--out-dir", default=str(ROOT / "experiments" / "c2"))
_ap.add_argument("--n-perm", type=int, default=100)
_args = _ap.parse_args()
c1.ALL_CSV = _args.all_csv
OUT = Path(_args.out_dir)
OUT.mkdir(parents=True, exist_ok=True)
SEED = c1.SEED


def auc_of(y, prob):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else float("nan")


def run_split(model, X, y, groups, grouped):
    if grouped:
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
        prob = cross_val_predict(model, X, y, cv=cv, groups=groups,
                                 method="predict_proba")[:, 1]
    else:
        cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
        prob = cross_val_predict(model, X, y, cv=cv,
                                 method="predict_proba")[:, 1]
    return prob


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for exp, loader, models in (
            ("E1-pairs", c1.load_pairs, ("XGB", "RF")),
            ("E2-full", c1.load_full, ("XGB",))):
        df = loader()
        X = df[c1.feature_cols(df)].values
        y = df["label"].values
        groups = df["group"].values
        print(f"[{exp}] n={len(y)} perms={_args.n_perm}", flush=True)
        for mname in models:
            model = c1.make_models()[mname]
            obs = {}
            for grouped in (False, True):
                obs[grouped] = auc_of(y, run_split(model, X, y, groups, grouped))
            nulls = {False: [], True: []}
            for p in range(_args.n_perm):
                y_perm = rng.permutation(y)
                for grouped in (False, True):
                    prob = run_split(model, X, y_perm, groups, grouped)
                    nulls[grouped].append(auc_of(y_perm, prob))
                if (p + 1) % 20 == 0:
                    print(f"  {mname} perm {p + 1}/{_args.n_perm}", flush=True)
            for grouped, label in ((False, "pooled"), (True, "group-by-base")):
                arr = np.array(nulls[grouped])
                pval = float((np.sum(np.abs(arr - 0.5) >= abs(obs[grouped] - 0.5)) + 1)
                             / (len(arr) + 1))
                rows.append({"experiment": exp, "model": mname, "protocol": label,
                             "observed_AUC": obs[grouped],
                             "null_mean": float(arr.mean()),
                             "null_lo": float(np.percentile(arr, 2.5)),
                             "null_hi": float(np.percentile(arr, 97.5)),
                             "p_value": pval})
                print(f"  {mname} {label}: obs={obs[grouped]:.3f} "
                      f"null=[{np.percentile(arr, 2.5):.3f},{np.percentile(arr, 97.5):.3f}] "
                      f"p={pval:.4f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "perm_test_results.csv", index=False)
    lines = ["permutation test（R5-M5，打乱标签 null 分布）",
             f"数据源: {c1.ALL_CSV}",
             f"n_perm={_args.n_perm}, seed={SEED}, 5-fold CV 同主协议",
             "p 值=|null-0.5| >= |obs-0.5| 的双侧比例（+1 校正）", ""]
    for _, r in out.iterrows():
        lines.append(
            f"{r['experiment']:10s} {r['model']:4s} {r['protocol']:14s} "
            f"obs={r['observed_AUC']:.3f} "
            f"null=[{r['null_lo']:.3f},{r['null_hi']:.3f}] (mean {r['null_mean']:.3f}) "
            f"p={r['p_value']:.4f}")
    txt = "\n".join(lines)
    (OUT / "perm_test_report.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    sys.exit(main())

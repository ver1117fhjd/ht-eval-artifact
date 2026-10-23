# -*- coding: utf-8 -*-
"""lit_baselines.py —— R3/R4 整改 P1-5：文献管线实例化
========================================================
把"已发表检测器管线"在我们协议阶梯上实例化，封住
"你们一个已发表管线都没复现过"的攻击面：

  DT+PCA  = [13] Chattopadhyay et al., Discover Electronics 2025 的 Case-I
            管线本体（decision tree + PCA 降维；其 NN 增强是图邻域，非分类器）
  KNN5    = [5]  Whitten et al., J. Electron. Test. 2026 的 case-based
            管线本体（kNN, k=5）

协议= c1_protocol 同一套（pooled / grp-variant / grp-base / LOFO / E3 LOLO），
CI 聚类恒按 base design（R4 口径），特征=v2 算子特征。

用法：
  python src/lit_baselines.py
输出：
  experiments/c2/lit_baseline_results.csv / lit_baseline_report.txt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (
    StratifiedKFold, StratifiedGroupKFold, LeaveOneGroupOut,
)

import c1_protocol as c1

# ⚠️ 必须显式切数据源：c1 模块默认指向 v1（skill pitfall）
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
SEED = c1.SEED


def make_models():
    return {
        "DT+PCA": make_pipeline(
            PCA(n_components=0.9, random_state=SEED),
            DecisionTreeClassifier(random_state=SEED)),
        "KNN5": make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    }


def main():
    rows = []
    # ---------------- E1 / E2 ----------------
    for exp, loader in (("E1-pairs", c1.load_pairs), ("E2-full", c1.load_full)):
        df = loader()
        X = df[c1.feature_cols(df)].values
        y = df["label"].values
        groups = df["group"].values            # base design（CI 聚类键）
        vgroups = df.apply(c1.variant_pair, axis=1).values
        fam_groups = df["family"].values
        print(f"[{exp}] n={len(df)} pos={int(y.sum())} groups={len(set(groups))}",
              flush=True)
        for mname, model in make_models().items():
            r, _ = c1.run_protocol("pooled-random-5fold", X, y, groups, mname,
                                   model,
                                   StratifiedKFold(5, shuffle=True,
                                                  random_state=SEED),
                                   use_groups=False, ci_groups=groups)
            rows.append((exp, r))
            r, _ = c1.run_protocol("group-by-variant-5fold", X, y, vgroups,
                                   mname, model,
                                   StratifiedGroupKFold(5, shuffle=True,
                                                        random_state=SEED),
                                   use_groups=True, ci_groups=groups)
            rows.append((exp, r))
            r, _ = c1.run_protocol("group-by-base-5fold", X, y, groups, mname,
                                   model,
                                   StratifiedGroupKFold(5, shuffle=True,
                                                        random_state=SEED),
                                   use_groups=True, ci_groups=groups)
            rows.append((exp, r))
            if exp == "E1-pairs":
                r, _ = c1.run_protocol("leave-one-family-out", X, y,
                                       fam_groups, mname, model,
                                       LeaveOneGroupOut(), use_groups=True,
                                       ci_groups=groups)
                rows.append((exp, r))

    # ---------------- E3（LOLO + pooled 对照）----------------
    df3 = c1.load_multilib()
    X3 = df3[c1.feature_cols(df3)].values
    y3 = df3["label"].values
    g3 = df3["group"].values
    techs = df3["tech"].values
    print(f"[E3-multilib] n={len(df3)} pos={int(y3.sum())}", flush=True)
    for mname, model in make_models().items():
        for held in sorted(set(techs)):
            tr = techs != held
            te = ~tr
            model.fit(X3[tr], y3[tr])
            prob = np.zeros(len(y3))
            prob[te] = model.predict_proba(X3[te])[:, 1]
            m = c1.metrics(y3[te], prob[te])
            ci = c1.boot_ci(y3[te], prob[te], g3[te])
            rows.append(("E3-multilib", {
                "protocol": f"leave-one-library-out({held})", "model": mname,
                "n": int(te.sum()), "n_pos": int(y3[te].sum()),
                "AP": m["AP"], "AP_lo": ci["AP"][1], "AP_hi": ci["AP"][2],
                "ROC_AUC": m["ROC_AUC"],
                "AUC_lo": ci["ROC_AUC"][1], "AUC_hi": ci["ROC_AUC"][2],
                "F1": m["F1"], "Sens": m["Sens"], "Spec": m["Spec"],
                "Spec_lo": ci["Spec"][1], "Spec_hi": ci["Spec"][2],
                "BalAcc": m["BalAcc"], "Gmean": m["Gmean"]}))
        r, _ = c1.run_protocol("pooled-random-5fold", X3, y3, g3, mname, model,
                               StratifiedKFold(5, shuffle=True,
                                              random_state=SEED),
                               use_groups=False, ci_groups=g3)
        rows.append(("E3-multilib", r))

    flat = [dict(experiment=e, **r) for e, r in rows]
    out = pd.DataFrame(flat)
    out.to_csv(OUT / "lit_baseline_results.csv", index=False)

    lines = ["文献管线实例化（R3/R4 整改 P1-5）",
             f"数据源: {c1.ALL_CSV} / {c1.MULTILIB_CSV}",
             "DT+PCA=[13] Case-I 管线；KNN5=[5] case-based 管线（k=5）",
             "CI=cluster bootstrap 1000，聚类=base design（R4 口径）", ""]
    for exp in out["experiment"].unique():
        lines.append(f"==== {exp} ====")
        sub = out[out["experiment"] == exp]
        for prot in sub["protocol"].unique():
            for _, r in sub[sub["protocol"] == prot].iterrows():
                lines.append(
                    f"{prot:32s} {r['model']:7s} n={r['n']:4d} "
                    f"AUC={r['ROC_AUC']:.3f} [{r['AUC_lo']:.3f},{r['AUC_hi']:.3f}] "
                    f"Spec={r['Spec']:.3f} Bal={r['BalAcc']:.3f}")
        lines.append("")
    txt = "\n".join(lines)
    (OUT / "lit_baseline_report.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    sys.exit(main())

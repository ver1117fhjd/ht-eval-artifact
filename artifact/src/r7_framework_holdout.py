# -*- coding: utf-8 -*-
"""r7_framework_holdout.py —— R6/R7 P0：framework-disjoint 评估层级
================================================================
评审共识：按 base design 留出后 TRIT 子集仍 0.92-1.00 = 同插入框架特征。
本脚本给出真正的 framework-disjoint 读数（train/test 来自不同插入框架）：

  A. 双向跨框架迁移：train TRIT → test 非 TRIT；train 非 TRIT → test TRIT
  B. leave-one-framework-out：框架块 = TRIT（TC+TS 合并）+ 各非 TRIT 家族，
     每折留出一个框架块，报告逐折 + macro

输出：experiments/c2/framework_holdout.csv / framework_holdout_report.txt
口径：RF/XGB/MLP，n_jobs=1（XGB 线程敏感教训），cluster bootstrap CI
（cluster=base design），数据源显式钉 features_all_v2.csv。
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

import c1_protocol as c1

c1.ALL_CSV = str(ROOT / "experiments" / "synth_work" / "features_all_v2.csv")
c1.MULTILIB_CSV = str(ROOT / "experiments" / "synth_work" / "features_multilib_v2.csv")
OUT = ROOT / "experiments" / "c2"
OUT.mkdir(parents=True, exist_ok=True)
SEED = c1.SEED


def make_models1():
    ms = c1.make_models()
    for m in ms.values():
        try:
            m.set_params(n_jobs=1)
        except ValueError:
            pass
    return ms


def fmt(m, ci):
    return (f"AP={m['AP']:.3f} [{ci['AP'][1]:.3f},{ci['AP'][2]:.3f}] "
            f"AUC={m['ROC_AUC']:.3f} [{ci['ROC_AUC'][1]:.3f},{ci['ROC_AUC'][2]:.3f}] "
            f"F1={m['F1']:.3f} Sens={m['Sens']:.3f} Spec={m['Spec']:.3f} "
            f"Bal={m['BalAcc']:.3f} Gm={m['Gmean']:.3f}")


def main():
    df = c1.load_full()
    X = df[c1.feature_cols(df)].values
    y = df["label"].values
    groups = df["group"].values
    is_trit = df["family"].str.startswith("TRIT").values
    # 框架块：TRIT 合一，其余按家族
    fw = np.where(is_trit, "TRIT", df["family"].values)

    rows = []
    lines = ["R7 framework-disjoint 补丁实验",
             f"数据源: {c1.ALL_CSV}（v2，显式钉死）",
             f"seed={SEED}, n_jobs=1, cluster bootstrap（cluster=base design）",
             f"总行数 {len(df)}；TRIT 块 {int(is_trit.sum())} 行 "
             f"(pos={int(y[is_trit].sum())})；非 TRIT {int((~is_trit).sum())} 行 "
             f"(pos={int(y[~is_trit].sum())})",
             f"框架块: TRIT + {sorted(set(df.loc[~is_trit, 'family']))}",
             ""]

    # ---------- A. 双向跨框架迁移 ----------
    lines.append("==== A. 双向跨框架迁移（train 一个框架 → test 另一框架）====")
    for mname, model in make_models1().items():
        for direction, tr_mask, te_mask in (
                ("train-TRIT->test-nonTRIT", is_trit, ~is_trit),
                ("train-nonTRIT->test-TRIT", ~is_trit, is_trit)):
            model.fit(X[tr_mask], y[tr_mask])
            prob = model.predict_proba(X[te_mask])[:, 1]
            m = c1.metrics(y[te_mask], prob)
            ci = c1.boot_ci(y[te_mask], prob, groups[te_mask])
            n_tr, n_te = int(tr_mask.sum()), int(te_mask.sum())
            n_tr_pos, n_te_pos = int(y[tr_mask].sum()), int(y[te_mask].sum())
            n_te_grp = len(set(groups[te_mask]))
            rec = {"experiment": "cross-framework", "protocol": direction,
                   "model": mname, "n_train": n_tr, "n_train_pos": n_tr_pos,
                   "n_test": n_te, "n_test_pos": n_te_pos,
                   "n_test_groups": n_te_grp, **{k: m[k] for k in
                   ("AP", "ROC_AUC", "F1", "Sens", "Spec", "BalAcc", "Gmean")},
                   "AUC_lo": ci["ROC_AUC"][1], "AUC_hi": ci["ROC_AUC"][2],
                   "Spec_lo": ci["Spec"][1], "Spec_hi": ci["Spec"][2]}
            rows.append(rec)
            lines.append(
                f"{direction:28s} {mname:4s} train={n_tr}({n_tr_pos}+) "
                f"test={n_te}({n_te_pos}+) test_groups={n_te_grp}  {fmt(m, ci)}")

    # ---------- B. leave-one-framework-out ----------
    lines.append("")
    lines.append("==== B. leave-one-framework-out（每折留出一个框架块）====")
    from sklearn.model_selection import LeaveOneGroupOut
    logo = LeaveOneGroupOut()
    macro_store = {}
    for mname, model in make_models1().items():
        fold_aucs = []
        for held in sorted(set(fw)):
            tr = fw != held
            te = ~tr
            model.fit(X[tr], y[tr])
            prob = np.zeros(len(y))
            prob[te] = model.predict_proba(X[te])[:, 1]
            m = c1.metrics(y[te], prob[te])
            ci = c1.boot_ci(y[te], prob[te], groups[te])
            fold_aucs.append(m["ROC_AUC"])
            rows.append({"experiment": "leave-one-framework-out",
                         "protocol": f"holdout({held})", "model": mname,
                         "n_train": int(tr.sum()), "n_train_pos": int(y[tr].sum()),
                         "n_test": int(te.sum()), "n_test_pos": int(y[te].sum()),
                         "n_test_groups": len(set(groups[te])),
                         **{k: m[k] for k in
                            ("AP", "ROC_AUC", "F1", "Sens", "Spec", "BalAcc", "Gmean")},
                         "AUC_lo": ci["ROC_AUC"][1], "AUC_hi": ci["ROC_AUC"][2],
                         "Spec_lo": ci["Spec"][1], "Spec_hi": ci["Spec"][2]})
            lines.append(
                f"holdout({held:12s}) {mname:4s} test={int(te.sum()):4d}"
                f"({int(y[te].sum()):4d}+) groups={len(set(groups[te])):2d}  {fmt(m, ci)}")
        macro = float(np.mean(fold_aucs))
        macro_store[mname] = macro
        lines.append(f"  {mname} macro-AUC over {len(fold_aucs)} framework folds = {macro:.3f}")

    lines.append("")
    lines.append("口径注：非 TRIT 各家族各有自己的插入来源，'框架块'按来源合并 TRIT-TC/TS"
                 "（共享 TRIT 插入框架）；语料只含两大类插入来源，措辞 bounded by corpus "
                 "structure。逐折 AUC 由模型族平均仅作汇总，不作显著性主张。")
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "framework_holdout.csv", index=False)
    txt = "\n".join(lines)
    (OUT / "framework_holdout_report.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    sys.exit(main())

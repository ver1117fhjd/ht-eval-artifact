# -*- coding: utf-8 -*-
"""
family_table.py —— P2：XGB 单模型逐 family 表（HOST 评审整改 P2 项）
====================================================================

动机：评审要求"逐 family 全量表"，10 页放不下三模型全量 → 压成 XGB 单模型
家族表。每行一个家族，对比两种评价口径下该家族的表现：
  - pooled（泄漏口径）：pooled 随机 5 折的预测按家族切片 —— 该家族的兄弟变体
    大概率在训练侧，虚高。
  - LOFO（诚实口径）：leave-one-family-out，该家族整体留出 —— 真泛化。

与主表同环境同 seed（SEED=20261012，make_models 同参数），pooled 总体行
可与 c1_report_v2ci 的 E1/E2 pooled 行对账。

用法：
  ./.venv/Scripts/python.exe src/family_table.py
    [--all-csv experiments/synth_work/features_all_v2.csv]
    [--out-dir experiments/c2]
输出：
  family_xgb.csv   每 (实验, 家族) 一行
  family_xgb.tex   IEEE 三线表 LaTeX 片段（直接抄进 paper.tex）
  family_xgb.txt   可读报告（含总体对账行）
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c1_protocol as c1  # noqa: E402  parse_known_args 已保证可安全 import


def per_family(df, y, prob_pooled, prob_lofo, exp):
    """把逐行预测按家族切片算指标。"""
    rows = []
    for fam in sorted(df["family"].unique()):
        mask = (df["family"] == fam).values
        mp = c1.metrics(y[mask], prob_pooled[mask])
        ml = c1.metrics(y[mask], prob_lofo[mask])
        rows.append({
            "experiment": exp, "family": fam,
            "n": int(mask.sum()), "n_pos": int(y[mask].sum()),
            "pooled_AUC": mp["ROC_AUC"], "pooled_AP": mp["AP"],
            "lofo_AUC": ml["ROC_AUC"], "lofo_AP": ml["AP"],
        })
    return rows


def fmt(v):
    return "---" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.2f}"


def main():
    import argparse
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-csv",
                    default=str(root / "experiments/synth_work/features_all_v2.csv"))
    ap.add_argument("--out-dir", default=str(root / "experiments/c2"))
    args = ap.parse_args()
    # 让 c1_protocol 的数据源也指到同一份（load_pairs/load_full 用模块级 ALL_CSV）
    c1.ALL_CSV = args.all_csv
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_rows = []
    reports = []

    for exp, loader in (("E1", c1.load_pairs), ("E2", c1.load_full)):
        df = loader().reset_index(drop=True)
        X = df[c1.feature_cols(df)].values
        y = df["label"].values
        fams = df["family"].values
        model = c1.make_models()["XGB"]

        # pooled 随机 5 折（与主表 pooled-random-5fold 完全一致）
        prob_pooled = cross_val_predict(
            model, X, y,
            cv=StratifiedKFold(5, shuffle=True, random_state=c1.SEED),
            method="predict_proba")[:, 1]
        # LOFO：按家族留出
        prob_lofo = cross_val_predict(
            model, X, y, cv=LeaveOneGroupOut(), groups=fams,
            method="predict_proba")[:, 1]

        all_rows.extend(per_family(df, y, prob_pooled, prob_lofo, exp))

        # 总体对账行（应与 c1_report_v2ci 的 XGB pooled / LOFO 行一致）
        mt_p = c1.metrics(y, prob_pooled)
        mt_l = c1.metrics(y, prob_lofo)
        reports.append(
            f"[{exp}] n={len(y)} pos={int(y.sum())} families={df['family'].nunique()}\n"
            f"  总体 pooled: AP={mt_p['AP']:.3f} AUC={mt_p['ROC_AUC']:.3f}\n"
            f"  总体 LOFO : AP={mt_l['AP']:.3f} AUC={mt_l['ROC_AUC']:.3f}")

    # ---- 写 CSV ----
    import csv
    cols = ["experiment", "family", "n", "n_pos",
            "pooled_AUC", "pooled_AP", "lofo_AUC", "lofo_AP"]
    csv_path = out / "family_xgb.csv"

    def cell(v):
        if isinstance(v, float):
            return "---" if np.isnan(v) else f"{v:.3f}"
        return v

    with open(csv_path, "w", newline="\n", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: cell(r[k]) for k in cols})

    # ---- 写 LaTeX 片段 ----
    e1 = {r["family"]: r for r in all_rows if r["experiment"] == "E1"}
    e2 = {r["family"]: r for r in all_rows if r["experiment"] == "E2"}
    fams = sorted(set(e1) | set(e2))
    tex = ["\\begin{table}[t]", "\\centering",
           "\\caption{Per-family XGBoost performance (v2 features) under the "
           "leaky pooled protocol vs.\\ leave-one-family-out (LOFO). Pooled "
           "scores slice the predictions of the pooled 5-fold run, where "
           "sibling variants of the same base design occur on both sides of "
           "the split; LOFO holds out the entire family. Even family-level "
           "LOFO is not leak-free for TRIT: TRIT-TC and TRIT-TS share eight "
           "base designs, so a held-out TC fold still sees same-design TS "
           "variants in training. Families with fewer than six rows (B19, "
           "S15850, VGA\\_LCD) are noise-dominated and excluded from the "
           "macro averages.}",
           "\\label{tab:family}", "\\small",
           "\\begin{tabular}{lcccccccc}", "\\toprule",
           "& \\multicolumn{4}{c}{E1 (paired, balanced)} & "
           "\\multicolumn{4}{c}{E2 (full, imbalanced)} \\\\",
           "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
           "Family & \\#pos & AUC$_{pool}$ & AUC$_{LOFO}$ & AP$_{LOFO}$ & "
           "\\#pos & AUC$_{pool}$ & AUC$_{LOFO}$ & AP$_{LOFO}$ \\\\",
           "\\midrule"]
    for fam in fams:
        a, b = e1.get(fam), e2.get(fam)
        cells = []
        for r in (a, b):
            if r is None:
                cells += ["---"] * 4
            else:
                cells += [str(r["n_pos"]), fmt(r["pooled_AUC"]),
                          fmt(r["lofo_AUC"]), fmt(r["lofo_AP"])]
        esc = fam.replace("_", "\\_")
        tex.append(esc + " & " + " & ".join(cells) + " \\\\")
    # macro 行（n>=6 家族，剔除噪声小家族）
    tex.append("\\midrule")
    e1b = [r for r in e1.values() if r["n"] >= 6]
    e2b = [r for r in e2.values() if r["n"] >= 6]

    def macro(rs, key):
        vals = [r[key] for r in rs if not np.isnan(r[key])]
        return fmt(float(np.mean(vals))) if vals else "---"

    tex.append("Macro ($n\\geq 6$) & --- & " +
               " & ".join([macro(e1b, "pooled_AUC"), macro(e1b, "lofo_AUC"),
                           macro(e1b, "lofo_AP")]) +
               " & --- & " +
               " & ".join([macro(e2b, "pooled_AUC"), macro(e2b, "lofo_AUC"),
                           macro(e2b, "lofo_AP")]) +
               " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    tex_path = out / "family_xgb.tex"
    tex_path.write_text("\n".join(tex) + "\n", encoding="utf-8")

    report = "\n".join(reports) + f"\n\nCSV: {csv_path}\nTeX: {tex_path}\n"
    (out / "family_xgb.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

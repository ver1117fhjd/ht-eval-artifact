# -*- coding: utf-8 -*-
"""r7_probe_stability.py —— R7：库身份探针 + 归因稳定性
========================================================
  ① 库身份探针：用 v1 / v2 特征预测网表来自哪个工艺库（297 行，3 类），
    LeaveOneGroupOut（组=基准设计）——v2 显著低于 v1 = "模型学库外观"
    从 3 点 std 变成 n=297 的可测证据。
  ② 归因稳定性：每库单独训练 XGB 预测木马标签，permutation importance
    排序在三库两两 Kendall τ——v2 操作子语义固定 ⇒ 归因跨库可对账。

输出：experiments/c2/probe_stability.csv / probe_stability_report.txt
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import c1_protocol as c1

SYNTH = ROOT / "experiments" / "synth_work"
OUT = ROOT / "experiments" / "c2"
OUT.mkdir(parents=True, exist_ok=True)
SEED = c1.SEED

META = {"family", "variant", "source", "module", "tech", "label", "group"}


def feat_cols(df):
    return [c for c in df.columns if c not in META]


def load_ml(path):
    df = pd.read_csv(path)
    df["group"] = df.apply(c1.base_design, axis=1)
    return df


def probe(X, y, groups, tag):
    """LeaveOneGroupOut 3 类库识别。"""
    logo = LeaveOneGroupOut()
    y = np.asarray(y)
    pred = np.zeros(len(y), dtype=int)
    for tr, te in logo.split(X, y, groups):
        clf = XGBClassifier(random_state=SEED, n_jobs=1, eval_metric="mlogloss")
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
    acc = accuracy_score(y, pred)
    bal = balanced_accuracy_score(y, pred)
    f1 = f1_score(y, pred, average="macro")
    # 置乱对照
    rng = np.random.default_rng(SEED)
    y_sh = rng.permutation(y)
    pred_sh = np.zeros(len(y), dtype=int)
    for tr, te in logo.split(X, y_sh, groups):
        clf = XGBClassifier(random_state=SEED, n_jobs=1, eval_metric="mlogloss")
        clf.fit(X[tr], y_sh[tr])
        pred_sh[te] = clf.predict(X[te])
    acc_sh = accuracy_score(y_sh, pred_sh)
    return acc, bal, f1, acc_sh


def attr_stability(df, cols, label_filter, tag):
    """每库 permutation importance，两两 Kendall τ + 同库 split-half 天花板。"""
    sub = df[df["source"].replace({"TnIn": "TjIn"}).isin(label_filter)]
    imps = {}
    ceilings = {}
    for tech in sorted(sub["tech"].unique()):
        d = sub[sub["tech"] == tech]
        X = d[cols].values
        y = (d["source"].replace({"TnIn": "TjIn"}).values == "TjIn").astype(int)
        if len(set(y)) < 2:
            continue
        fold_imps = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        for tr, te in skf.split(X, y):
            clf = XGBClassifier(random_state=SEED, n_jobs=1)
            clf.fit(X[tr], y[tr])
            r = permutation_importance(clf, X[te], y[te], scoring="roc_auc",
                                       n_repeats=10, random_state=SEED)
            fold_imps.append(r.importances_mean)
        imps[tech] = np.mean(fold_imps, axis=0)
        # split-half 天花板：前 2/3 折 vs 后 2/3 折的 τ（指标噪声上界）
        a = np.mean(fold_imps[:3], axis=0)
        b = np.mean(fold_imps[2:], axis=0)
        ceilings[tech] = kendalltau(a, b)[0]
    taus = {}
    techs = sorted(imps)
    for i in range(len(techs)):
        for j in range(i + 1, len(techs)):
            tau, p = kendalltau(imps[techs[i]], imps[techs[j]])
            taus[(techs[i], techs[j])] = (tau, p)
    return imps, taus, ceilings


def main():
    lines = ["R7 库身份探针 + 归因稳定性",
             f"seed={SEED} n_jobs=1", ""]
    rows = []

    v1 = load_ml(SYNTH / "features_multilib.csv")
    v2 = load_ml(SYNTH / "features_multilib_v2.csv")
    print(f"[load] v1 {v1.shape} v2 {v2.shape}")

    # ---------- ① 库身份探针 ----------
    lines.append("==== ① 库身份探针（LeaveOneGroupOut，组=基准设计）====")
    lines.append(f"样本 n={len(v2)}（3 库），组数={v2['group'].nunique()}，"
                 f"机会水平=0.333")
    for tag, df in (("v1", v1), ("v2", v2)):
        cols = feat_cols(df)
        X = df[cols].values
        classes = sorted(df["tech"].unique())
        enc = {c: i for i, c in enumerate(classes)}
        y = df["tech"].map(enc).values
        g = df["group"].values
        acc, bal, f1, acc_sh = probe(X, y, g, tag)
        rows.append({"experiment": "lib-probe", "features": tag,
                     "acc": acc, "bal_acc": bal, "macro_f1": f1,
                     "shuffled_acc": acc_sh})
        lines.append(f"  {tag}: acc={acc:.3f} bal_acc={bal:.3f} "
                     f"macro_f1={f1:.3f}（置乱对照 {acc_sh:.3f}）")

    # ---------- ② 归因稳定性 ----------
    lines.append("")
    lines.append("==== ② 归因稳定性（每库 XGB permutation importance，Kendall τ）====")
    for tag, df in (("v1", v1), ("v2", v2)):
        cols = feat_cols(df)
        imps, taus, ceilings = attr_stability(df, cols, ["TjIn", "TjFree"], tag)
        taus_all = [t for t, _ in taus.values()]
        mean_tau = float(np.mean(taus_all)) if taus_all else float("nan")
        mean_ceil = float(np.mean(list(ceilings.values())))
        rows.append({"experiment": "attr-stability", "features": tag,
                     "mean_kendall_tau": mean_tau,
                     "split_half_ceiling": mean_ceil,
                     "pairs": "; ".join(f"{a}-{b}: {t:.3f}(p={p:.3f})"
                                        for (a, b), (t, p) in taus.items())})
        lines.append(f"  {tag}: mean Kendall τ = {mean_tau:.3f}，"
                     f"同库 split-half 天花板 = {mean_ceil:.3f}")
        for (a, b), (t, p) in taus.items():
            lines.append(f"     {a:10s} vs {b:10s} τ={t:+.3f} (p={p:.3f})")
        # 顶层归因对照（对账用）
        top = np.argsort(-imps[sorted(imps)[0]])[:5]
        lines.append(f"     top-5 特征（{sorted(imps)[0]}）: "
                     + ", ".join(cols[i] for i in top))

    lines.append("")
    lines.append("口径注：探针 3 类机会水平 0.333；归因 τ 越高=跨库归因越可对账。"
                 "v2 若探针准确率低于 v1 且 τ 高于 v1，即 C2 '库外观↓+可审计归因' 双证据。")
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "probe_stability.csv", index=False)
    txt = "\n".join(lines)
    (OUT / "probe_stability_report.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    sys.exit(main())

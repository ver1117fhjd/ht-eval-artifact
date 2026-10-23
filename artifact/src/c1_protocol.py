# -*- coding: utf-8 -*-
"""
c1_protocol.py —— C1 贡献：无泄漏评价协议 + 泄漏虚高实证
=========================================================

论文贡献点 C1：对 ML 硬件木马检测，给出无泄漏评价协议
（按基准设计分组 + leave-one-family-out），并实证"兄弟变体泄漏"
（同一基准设计的金样/木马变体随机分到训练/测试两侧）导致指标虚高。

数据：
  experiments/synth_work/features_all.csv        （1059 设计，generic 综合）
  experiments/synth_work/features_multilib.csv   （99 设计 ×3 工艺库）

三组实验：
  E1  TjIn/TjFree 配对集（121 行，平衡，12 家族）
      pooled 随机 5 折  vs  按基准设计分组 5 折  vs  leave-one-family-out
  E2  全量集（1059 行，含 TRIT 899 木马变体 / 12 金样，不平衡）
      pooled 随机 5 折  vs  按基准设计分组 5 折
      （不平衡 → 主报 AP/AUPRC，不报 accuracy，见 skill pitfalls）
  E3  跨工艺库 leave-one-library-out（multilib，C1 的"跨库"半边）

指标：Average Precision（主）、ROC-AUC、F1@0.5、Sens/Spec、BalAcc、Gmean、
bootstrap 95% CI。不平衡实验（E2）主报 ROC-AUC + Spec + BalAcc/Gmean。
模型：RandomForest / XGBoost / MLP（sklearn 接口；MLP 带 StandardScaler 流水线）。

用法：
  python src/c1_protocol.py
输出：
  experiments/c1/c1_results.csv      每个 (实验, 协议, 模型) 一行
  experiments/c1/c1_report.txt       可读汇总（进论文的数字都从这抄）
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    GroupKFold, StratifiedKFold, StratifiedGroupKFold,
    cross_val_predict, LeaveOneGroupOut,
)
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
SYNTH = ROOT / "experiments" / "synth_work"
OUT = ROOT / "experiments" / "c1"

# 命令行可换数据源/输出目录（C2 重放用：--all-csv ..._v2.csv --out-dir experiments/c2）
import argparse as _argparse
_ap = _argparse.ArgumentParser()
_ap.add_argument("--all-csv", default=str(SYNTH / "features_all.csv"))
_ap.add_argument("--multilib-csv", default=str(SYNTH / "features_multilib.csv"))
_ap.add_argument("--out-dir", default=str(OUT))
_ap.add_argument("--tag", default="")
_args, _unknown = _ap.parse_known_args()   # parse_known_args：允许被 gnn_baseline 等安全 import
ALL_CSV, MULTILIB_CSV = _args.all_csv, _args.multilib_csv
OUT = Path(_args.out_dir)
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20261012
N_BOOT = 1000

# 特征列 = 排除元数据后的所有数值列
META_COLS = {"family", "variant", "source", "module", "tech", "label", "group"}


def feature_cols(df):
    return [c for c in df.columns if c not in META_COLS]


def base_design(row):
    """基准设计 key：剥掉木马编号后缀。
    'AES-T100' -> 'AES'；TRIT 的 source 'c2670_T000' -> 'c2670'；
    TRIT 的金样 source 'original_designs/c2670.v' -> 'c2670'。"""
    fam, var, src = row["family"], str(row["variant"]), str(row["source"])
    if fam.startswith("TRIT"):
        s = src.split("/")[-1]                      # c2670_T000 / c2670.v
        s = re.sub(r"\.v$", "", s)
        s = re.sub(r"_T\d+$", "", s)                # 剥 _T000
        # R15 修正：金样文件叫 s13207scan.v，变体叫 s13207_Tnnn，是同一设计的补丁。
        # 不剥 "scan" 会把同一设计拆成两组（金样单独成组），组内仍有兄弟泄漏。
        if s.endswith("scan"):
            s = s[:-4]
        return fam.split("-")[0] + ":" + s          # TRIT:c2670（TC/TS 两种布局归同一基准组）
    # 配对集：AES-T100 -> AES（同家族同基准的多触发器变体共享金样）
    return fam + ":" + re.sub(r"-T\d+$", "", var)


def variant_pair(row):
    """中间粒度分组：同一 (家族, 变体) 的金样+木马对不拆散，
    但同基准不同触发器变体仍可跨折 —— 泄漏梯度的中间档。"""
    return f"{row['family']}:{row['variant']}"


def load_pairs():
    """E1：TjIn/TjFree 配对集。TnIn 是提取脚本留下的拼写错误，归为 TjIn。"""
    df = pd.read_csv(ALL_CSV)
    src = df["source"].replace({"TnIn": "TjIn"})
    m = src.isin(["TjIn", "TjFree"])
    df = df[m].copy()
    df["label"] = (src[m] == "TjIn").astype(int)
    df["group"] = df.apply(base_design, axis=1)
    return df


def load_full():
    """E2：全量集。TRIT：source 含 _TNNN = 木马，original_designs/ = 金样；
    其余家族只用有 TjIn/TjFree 标注的行（B19 只有木马无金样 → 丢弃）。"""
    df = pd.read_csv(ALL_CSV)
    src = df["source"].replace({"TnIn": "TjIn"})
    is_trit = df["family"].str.startswith("TRIT")
    is_pair = src.isin(["TjIn", "TjFree"])
    df = df[is_trit | is_pair].copy()
    s = src[is_trit | is_pair]
    lab = np.where(is_trit[is_trit | is_pair],
                   s.str.contains(r"_T\d+").astype(int),
                   (s == "TjIn").astype(int))
    df["label"] = lab
    df["group"] = df.apply(base_design, axis=1)
    return df


def load_multilib():
    """E3：multilib（99 设计 ×3 库）。"""
    df = pd.read_csv(MULTILIB_CSV)
    src = df["source"].replace({"TnIn": "TjIn"})
    m = src.isin(["TjIn", "TjFree"])
    df = df[m].copy()
    df["label"] = (src[m] == "TjIn").astype(int)
    df["group"] = df.apply(base_design, axis=1)   # 跨库也按基准设计分组
    return df


def make_models():
    return {
        "RF": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=1),
        "XGB": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9,
            random_state=SEED, n_jobs=1,
            eval_metric="logloss"),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                          max_iter=500, n_iter_no_change=30,
                          random_state=SEED)),
    }


def metrics(y, prob):
    pred = (prob >= 0.5).astype(int)
    neg = y == 0
    pos = y == 1
    sens = float((pred[pos] == 1).mean()) if pos.any() else float("nan")
    spec = float((pred[neg] == 0).mean()) if neg.any() else float("nan")
    if np.isnan(sens) or np.isnan(spec):
        bal = gmean = float("nan")
    else:
        bal = (sens + spec) / 2
        gmean = float(np.sqrt(sens * spec))
    return {
        "AP": average_precision_score(y, prob),
        "ROC_AUC": roc_auc_score(y, prob) if len(np.unique(y)) > 1 else float("nan"),
        "F1": f1_score(y, pred, zero_division=0),
        "Sens": sens,
        "Spec": spec,
        "BalAcc": bal,
        "Gmean": gmean,
    }


def boot_ci(y, prob, groups, n=N_BOOT, seed=SEED):
    """对预测结果做 bootstrap 95% CI（cluster 重采样：按 base design 组重采，
    兄弟变体不独立，样本级重采样会把 CI 做窄——R2 整改 P0-4）。
    ⚠️ groups 参数必须传 base-design 聚类键，与 split 用的分组无关
    （R4 整改：group-by-variant/LOFO 的 CI 之前误用 split 组重采样）。"""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); prob = np.asarray(prob); groups = np.asarray(groups)
    uniq = np.unique(groups)
    gidx = {g: np.where(groups == g)[0] for g in uniq}
    out = {}
    for name in ("AP", "ROC_AUC", "F1", "Sens", "Spec", "BalAcc", "Gmean"):
        vals = []
        for _ in range(n):
            pick = rng.integers(0, len(uniq), len(uniq))
            idx = np.concatenate([gidx[uniq[p]] for p in pick])
            if len(np.unique(y[idx])) < 2:
                continue
            vals.append(metrics(y[idx], prob[idx])[name])
        out[name] = (float(np.nanmean(vals)), np.nanpercentile(vals, 2.5),
                     np.nanpercentile(vals, 97.5))
    return out


def boot_delta(y, prob_a, prob_b, groups, n=N_BOOT, seed=SEED):
    """ΔAUC = AUC(a) − AUC(b) 的 paired cluster bootstrap 95% CI（R4 整改 P0-3）。
    两个协议在同一行集上出 OOF 预测；对同一 cluster 重采样取差值——
    边际 CI 重叠时差值 CI 仍可判显著性。groups 同样必须是 base-design 键。"""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    prob_a = np.asarray(prob_a); prob_b = np.asarray(prob_b)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    gidx = {g: np.where(groups == g)[0] for g in uniq}
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([gidx[uniq[p]] for p in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], prob_a[idx]) -
                    roc_auc_score(y[idx], prob_b[idx]))
    return (float(np.nanmean(vals)), float(np.nanpercentile(vals, 2.5)),
            float(np.nanpercentile(vals, 97.5)))


def run_protocol(name, X, y, groups, model_name, model, splitter, use_groups,
                 ci_groups=None):
    """ci_groups：CI 用的聚类键（base design）；与 split 用的 groups 分离。
    不传时退回 groups（此时要求 groups 本身就是 base-design 键）。"""
    if use_groups:
        prob = cross_val_predict(model, X, y, cv=splitter, groups=groups,
                                 method="predict_proba")[:, 1]
    else:
        prob = cross_val_predict(model, X, y, cv=splitter,
                                 method="predict_proba")[:, 1]
    m = metrics(y, prob)
    ci = boot_ci(y, prob, groups if ci_groups is None else ci_groups)
    return {"protocol": name, "model": model_name,
            "n": len(y), "n_pos": int(y.sum()),
            "AP": m["AP"], "AP_lo": ci["AP"][1], "AP_hi": ci["AP"][2],
            "ROC_AUC": m["ROC_AUC"],
            "AUC_lo": ci["ROC_AUC"][1], "AUC_hi": ci["ROC_AUC"][2],
            "F1": m["F1"],
            "Sens": m["Sens"], "Spec": m["Spec"],
            "Spec_lo": ci["Spec"][1], "Spec_hi": ci["Spec"][2],
            "BalAcc": m["BalAcc"], "Gmean": m["Gmean"]}, prob


def main():
    rows = []
    probs = {}
    opreds = {}   # (exp, model) -> (y, {"pooled":..,"variant":..,"base":..})，ΔAUC 用

    # ---------------- E1：配对集（平衡） ----------------
    df = load_pairs()
    X = df[feature_cols(df)].values
    y = df["label"].values
    groups = df["group"].values
    vgroups = df.apply(variant_pair, axis=1).values
    print(f"[E1] pairs: n={len(df)} pos={y.sum()} groups={len(set(groups))} "
          f"families={df['family'].nunique()}")
    gold = df[df["label"] == 0]
    n_uniq = len(gold.drop_duplicates(subset=feature_cols(gold)))
    print(f"[audit] 金样重复度: {len(gold)} 行金样 = {n_uniq} 个唯一特征向量")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    lofo = LeaveOneGroupOut()
    # LOFO 用家族当分组
    fam_groups = df["family"].values

    for mname, model in make_models().items():
        r, prob_pool = run_protocol("pooled-random-5fold", X, y, groups, mname,
                                    model,
                                    StratifiedKFold(5, shuffle=True,
                                                   random_state=SEED),
                                    use_groups=False, ci_groups=groups)
        rows.append(("E1-pairs", r))
        r, prob_var = run_protocol("group-by-variant-5fold", X, y, vgroups,
                                   mname, model,
                                   StratifiedGroupKFold(5, shuffle=True,
                                                        random_state=SEED),
                                   use_groups=True, ci_groups=groups)
        rows.append(("E1-pairs", r))
        r, prob_base = run_protocol("group-by-base-5fold", X, y, groups, mname,
                                    model, sgkf, use_groups=True, ci_groups=groups)
        rows.append(("E1-pairs", r))
        r, prob = run_protocol("leave-one-family-out", X, y, fam_groups, mname,
                               model, lofo, use_groups=True, ci_groups=groups)
        rows.append(("E1-pairs", r))
        probs[("E1", mname)] = (y, prob)
        opreds[("E1-pairs", mname)] = (
            y, {"pooled": prob_pool, "variant": prob_var, "base": prob_base})

    # ---------------- E2：全量（不平衡，含 TRIT） ----------------
    df2 = load_full()
    X2 = df2[feature_cols(df2)].values
    y2 = df2["label"].values
    g2 = df2["group"].values
    v2 = df2.apply(variant_pair, axis=1).values
    print(f"[E2] full: n={len(df2)} pos={y2.sum()} neg={(1-y2).sum()} "
          f"groups={len(set(g2))} families={df2['family'].nunique()}")

    for mname, model in make_models().items():
        r, prob_pool = run_protocol("pooled-random-5fold", X2, y2, g2, mname,
                                    model,
                                    StratifiedKFold(5, shuffle=True,
                                                   random_state=SEED),
                                    use_groups=False, ci_groups=g2)
        rows.append(("E2-full", r))
        r, prob_var = run_protocol("group-by-variant-5fold", X2, y2, v2, mname,
                                   model,
                                   StratifiedGroupKFold(5, shuffle=True,
                                                        random_state=SEED),
                                   use_groups=True, ci_groups=g2)
        rows.append(("E2-full", r))
        r, prob_base = run_protocol("group-by-base-5fold", X2, y2, g2, mname,
                                    model,
                                    StratifiedGroupKFold(5, shuffle=True,
                                                         random_state=SEED),
                                    use_groups=True, ci_groups=g2)
        rows.append(("E2-full", r))
        probs[("E2", mname)] = (y2, prob)
        opreds[("E2-full", mname)] = (
            y2, {"pooled": prob_pool, "variant": prob_var, "base": prob_base})

    # ---------------- E3：跨工艺库 LOLO ----------------
    df3 = load_multilib()
    X3 = df3[feature_cols(df3)].values
    y3 = df3["label"].values
    g3 = df3["group"].values
    techs = df3["tech"].values
    print(f"[E3] multilib: n={len(df3)} pos={y3.sum()} "
          f"groups={len(set(g3))} techs={sorted(set(techs))}")

    for mname, model in make_models().items():
        for held in sorted(set(techs)):
            tr = techs != held
            te = ~tr
            model.fit(X3[tr], y3[tr])
            prob = np.zeros(len(y3))
            prob[te] = model.predict_proba(X3[te])[:, 1]
            # 测试侧只在留出的库上算指标
            m = metrics(y3[te], prob[te])
            ci = boot_ci(y3[te], prob[te], g3[te])
            rows.append(("E3-multilib", {
                "protocol": f"leave-one-library-out({held})", "model": mname,
                "n": int(te.sum()), "n_pos": int(y3[te].sum()),
                "AP": m["AP"], "AP_lo": ci["AP"][1], "AP_hi": ci["AP"][2],
                "ROC_AUC": m["ROC_AUC"],
                "AUC_lo": ci["ROC_AUC"][1], "AUC_hi": ci["ROC_AUC"][2],
                "F1": m["F1"],
                "Sens": m["Sens"], "Spec": m["Spec"],
                "Spec_lo": ci["Spec"][1], "Spec_hi": ci["Spec"][2],
                "BalAcc": m["BalAcc"], "Gmean": m["Gmean"]}))
        # 对照：同库内随机 5 折（3 库混在一起，泄漏兄弟行）
        r, _ = run_protocol("pooled-random-5fold", X3, y3, g3, mname, model,
                            StratifiedKFold(5, shuffle=True, random_state=SEED),
                            use_groups=False)
        rows.append(("E3-multilib", r))

    # ---------------- ΔAUC：对差值做 paired cluster bootstrap（R4 整改）--------
    tag = f"_{_args.tag}" if _args.tag else ""
    delta_rows = []
    for (exp, mname), (yy, dd) in opreds.items():
        gg = groups if exp == "E1-pairs" else g2
        for comp, (a, b) in (("pooled-minus-base", ("pooled", "base")),
                             ("variant-minus-base", ("variant", "base"))):
            dm, dlo, dhi = boot_delta(yy, dd[a], dd[b], gg)
            delta_rows.append({"experiment": exp, "model": mname,
                               "contrast": comp,
                               "dAUC": dm, "d_lo": dlo, "d_hi": dhi})
    ddf = pd.DataFrame(delta_rows)
    ddf.to_csv(OUT / f"c1_delta_auc{tag}.csv", index=False)

    # ---------------- 汇总输出 ----------------
    flat = [dict(experiment=e, **r) for e, r in rows]
    out = pd.DataFrame(flat)
    out = out[[ "experiment", "protocol", "model", "n", "n_pos",
               "AP", "AP_lo", "AP_hi", "ROC_AUC", "F1",
               "AUC_lo", "AUC_hi",
               "Sens", "Spec", "Spec_lo", "Spec_hi", "BalAcc", "Gmean"]]
    tag = f"_{_args.tag}" if _args.tag else ""
    out.to_csv(OUT / f"c1_results{tag}.csv", index=False)

    lines = []
    lines.append("C1 无泄漏评价协议 —— 结果汇总")
    lines.append(f"数据源: {ALL_CSV} / {MULTILIB_CSV}")
    lines.append(f"seed={SEED}, bootstrap n={N_BOOT}（cluster=base design，"
                 f"split 组与 CI 组分离——R4 整改口径），"
                 f"指标=AP/AUC/F1/Sens/Spec/BalAcc/Gmean")
    lines.append("")
    for exp in out["experiment"].unique():
        lines.append(f"==== {exp} ====")
        sub = out[out["experiment"] == exp]
        for prot in sub["protocol"].unique():
            for _, r in sub[sub["protocol"] == prot].iterrows():
                lines.append(
                    f"{prot:32s} {r['model']:4s} n={r['n']:4d} pos={r['n_pos']:4d} "
                    f"AP={r['AP']:.3f} [{r['AP_lo']:.3f},{r['AP_hi']:.3f}] "
                    f"AUC={r['ROC_AUC']:.3f} [{r['AUC_lo']:.3f},{r['AUC_hi']:.3f}] "
                    f"F1={r['F1']:.3f} "
                    f"Sens={r['Sens']:.3f} Spec={r['Spec']:.3f} "
                    f"Bal={r['BalAcc']:.3f} Gm={r['Gmean']:.3f}")
        lines.append("")
    lines.append("==== ΔAUC（paired cluster bootstrap，R4 整改 P0-3）====")
    for _, r in ddf.iterrows():
        lines.append(
            f"{r['experiment']:10s} {r['model']:4s} {r['contrast']:20s} "
            f"dAUC={r['dAUC']:+.3f} [{r['d_lo']:+.3f},{r['d_hi']:+.3f}]")
    lines.append("")
    txt = "\n".join(lines)
    (OUT / f"c1_report{tag}.txt").write_text(txt, encoding="utf-8")
    print(txt)
    print(f"\n[written] {OUT}\\c1_results{tag}.csv, c1_report{tag}.txt")


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""aig_detect.py —— R2 整改 P0-2：AIG vs v2 检测性能对比（评审 A 第 2 条）。

四表示（v1 cell-type / v2 operator / AIG-A direct / AIG-B re-norm）在 E3 阶梯
（pooled random 5-fold + LOLO×3）同协议比 AUC/Spec，cluster bootstrap CI。
数据源：experiments/synth_work/features_multilib{,_v2}.csv + experiments/c2/aig{A,B}_features.csv
输出：experiments/c2/aig_detect_report.txt + aig_detect_results.csv
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
import c1_protocol as c1

_ap = argparse.ArgumentParser()
_ap.add_argument("--v1-csv", default="results/features_multilib.csv")
_ap.add_argument("--v2-csv", default="results/features_multilib_v2.csv")
_ap.add_argument("--aig-a-csv", default="results/aigA_features.csv")
_ap.add_argument("--aig-b-csv", default="results/aigB_features.csv")
_ap.add_argument("--out-dir", default="out")
_args = _ap.parse_args()
c1.MULTILIB_CSV = _args.v2_csv
OUT = _args.out_dir
os.makedirs(OUT, exist_ok=True)
SEED = c1.SEED
N_BOOT = 1000
KEY = ["family", "variant", "source", "tech"]
META = set(KEY + ["module", "fanin", "fanout"])


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
    y = np.asarray(y); prob = np.asarray(prob); groups = np.asarray(groups)
    uniq = np.unique(groups)
    gidx = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    out = {}
    for name in ("ROC_AUC", "Spec", "BalAcc"):
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


def feat_cols(df):
    cols = []
    for c in df.columns:
        if c in META or c in KEY:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def norm_keys(df):
    """键归一（两侧同规则）：variant 剥 __TjIn/__TjFree/__TnIn 后缀（source 缺失时
    从后缀补），source 统一 TnIn→TjIn（提取脚本拼写遗留）。"""
    df = df.copy()
    if df["variant"].str.contains("__").any():
        suf = df["variant"].str.extract(r"__(TjIn|TnIn|TjFree)$", expand=False)
        df["variant"] = df["variant"].str.replace(r"__(TjIn|TnIn|TjFree)$", "", regex=True)
        df["source"] = df["source"].fillna(suf)
    df["source"] = df["source"].replace({"TnIn": "TjIn"})
    return df


def load_rep(path, base):
    """按 KEY 对齐到 E3 的 264 行基准框（两侧都过 norm_keys）。"""
    df = norm_keys(pd.read_csv(path))
    df = df.dropna(subset=["source"])
    b = norm_keys(base)
    m = b[KEY].merge(df, on=KEY, how="left", suffixes=("", "_rep"))
    cols = feat_cols(df)
    missing = int(m[cols].isna().all(axis=1).sum())
    if missing:
        raise SystemExit(f"[FATAL] {path}: {missing}/{len(m)} rows failed to join -- "
                         f"refusing to run on an all-zero matrix")
    return m, cols, missing


def main():
    base = c1.load_multilib()          # 264 行（88 网表×3 库），带 label/group
    y = base["label"].values
    groups = base["group"].values
    techs = base["tech"].values
    print(f"[data] E3 n={len(base)} pos={y.sum()} groups={len(set(groups))}", flush=True)

    reps = {
        "v1-celltype": _args.v1_csv,
        "v2-operator": _args.v2_csv,
        "AIG-A": _args.aig_a_csv,
        "AIG-B": _args.aig_b_csv,
    }
    rows = []
    for rep, path in reps.items():
        m, cols, missing = load_rep(path, base)
        print(f"[{rep}] features={len(cols)} missing_rows={missing}", flush=True)
        X = m[cols].fillna(0).values
        for mname, model in make_models().items():
            # pooled random 5-fold
            prob = cross_val_predict(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                                     method="predict_proba")[:, 1]
            mm = c1.metrics(y, prob)
            ci = cluster_boot_ci(y, prob, groups)
            rows.append(dict(rep=rep, protocol="pooled-random-5fold", model=mname,
                             AUC=mm["ROC_AUC"], AUC_lo=ci["ROC_AUC"][1], AUC_hi=ci["ROC_AUC"][2],
                             Spec=mm["Spec"], BalAcc=mm["BalAcc"]))
            print(f"[{rep}] pooled {mname:4s} AUC={mm['ROC_AUC']:.3f} Spec={mm['Spec']:.3f}", flush=True)
            # LOLO
            for held in sorted(set(techs)):
                tr = techs != held; te = ~tr
                model.fit(X[tr], y[tr])
                p = model.predict_proba(X[te])[:, 1]
                mm = c1.metrics(y[te], p)
                ci = cluster_boot_ci(y[te], p, groups[te])
                rows.append(dict(rep=rep, protocol=f"LOLO({held})", model=mname,
                                 AUC=mm["ROC_AUC"], AUC_lo=ci["ROC_AUC"][1], AUC_hi=ci["ROC_AUC"][2],
                                 Spec=mm["Spec"], BalAcc=mm["BalAcc"]))
                print(f"[{rep}] LOLO({held:9s}) {mname:4s} AUC={mm['ROC_AUC']:.3f} "
                      f"Spec={mm['Spec']:.3f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "aig_detect_results.csv"), index=False)
    # 分模型跨库离散度（LOLO 三折的 std / range）
    lines = ["AIG vs v2 检测对比（E3 阶梯），cluster bootstrap n=1000, seed=20261012",
             "数据源: features_multilib.csv / features_multilib_v2.csv / aigA_features.csv / aigB_features.csv", ""]
    for _, r in out.iterrows():
        lines.append(f"{r['rep']:12s} {r['protocol']:22s} {r['model']:4s} "
                     f"AUC={r['AUC']:.3f} [{r['AUC_lo']:.3f},{r['AUC_hi']:.3f}] "
                     f"Spec={r['Spec']:.3f} Bal={r['BalAcc']:.3f}")
    lines.append("")
    lines.append("== LOLO 分模型离散度（3 折 AUC 的 std / range）==")
    for rep in reps:
        for mname in ("RF", "XGB", "MLP"):
            sub = out[(out.rep == rep) & (out.model == mname) & out.protocol.str.startswith("LOLO")]
            a = sub["AUC"].values
            lines.append(f"{rep:12s} {mname:4s} std={a.std():.4f} range={a.max()-a.min():.3f} "
                         f"vals={np.round(a,3).tolist()}")
    txt = "\n".join(lines)
    open(os.path.join(OUT, "aig_detect_report.txt"), "w", encoding="utf-8").write(txt)
    print("\n" + txt)


if __name__ == "__main__":
    main()

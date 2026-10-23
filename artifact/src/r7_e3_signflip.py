# -*- coding: utf-8 -*-
"""r7_e3_signflip.py —— R7：E3 精确 sign-flip 检验 + 逐组表
============================================================
5 个独立 base-design 组下 cluster bootstrap CI 反保守（<10 clusters）。
补：逐组 AUC（pooled vs LOLO）+ 2^5=32 枚举精确 sign-flip 检验
（H0：pooled 与 LOLO 的逐组 AUC 差对称于 0）+ 逐组样本/正负/Spec 表。

输出：experiments/c2/e3_signflip.csv / e3_per_group.csv / e3_signflip_report.txt
"""
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

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


def sign_flip_p(deltas):
    """精确双侧 sign-flip：|mean(sign*d)| >= |mean(d)| 的比例。"""
    d = np.asarray(deltas, float)
    obs = abs(d.mean())
    n = len(d)
    cnt = 0
    tot = 0
    for signs in itertools.product((1.0, -1.0), repeat=n):
        v = abs(np.mean(d * np.array(signs)))
        tot += 1
        if v >= obs - 1e-12:
            cnt += 1
    return cnt / tot


def main():
    df = c1.load_multilib().reset_index(drop=True)
    X = df[c1.feature_cols(df)].values
    y = df["label"].values
    g = df["group"].values
    techs = df["tech"].values
    lines = ["R7 E3 精确 sign-flip 检验 + 逐组表",
             f"数据源: {c1.MULTILIB_CSV}（v2，显式钉死）seed={SEED} n_jobs=1",
             f"n={len(df)} pos={int(y.sum())} 组数={len(set(g))} "
             f"组={sorted(set(g))} 库={sorted(set(techs))}",
             ""]
    per_group_rows = []
    sig_rows = []

    for mname, model in make_models1().items():
        # pooled OOF（同库随机 5 折）
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
        prob_pool = cross_val_predict(model, X, y, cv=skf, method="predict_proba")[:, 1]
        # LOLO OOF
        prob_lolo = np.zeros(len(y))
        lolo_specs = {}
        for held in sorted(set(techs)):
            tr = techs != held
            te = ~tr
            model.fit(X[tr], y[tr])
            prob_lolo[te] = model.predict_proba(X[te])[:, 1]
            yh = (prob_lolo[te] >= 0.5).astype(int)
            neg = y[te] == 0
            lolo_specs[held] = float((yh[neg] == 0).mean()) if neg.any() else float("nan")

        deltas = []
        lines.append(f"==== {mname} ====")
        for grp in sorted(set(g)):
            m = g == grp
            both = len(set(y[m])) == 2
            auc_p = roc_auc_score(y[m], prob_pool[m]) if both else float("nan")
            auc_l = roc_auc_score(y[m], prob_lolo[m]) if both else float("nan")
            yh = (prob_lolo[m] >= 0.5).astype(int)
            neg = y[m] == 0
            pos = y[m] == 1
            spec = float((yh[neg] == 0).mean()) if neg.any() else float("nan")
            sens = float((yh[pos] == 1).mean()) if pos.any() else float("nan")
            per_group_rows.append({
                "model": mname, "group": grp, "n": int(m.sum()),
                "n_pos": int(y[m].sum()), "n_neg": int((1 - y[m]).sum()),
                "auc_pooled": auc_p, "auc_lolo": auc_l,
                "spec_lolo@0.5": spec, "sens_lolo@0.5": sens})
            if both:
                deltas.append(auc_p - auc_l)
            lines.append(f"  {grp:12s} n={int(m.sum()):3d} "
                         f"(+{int(y[m].sum())}/-{int((1-y[m]).sum()):3d}) "
                         f"AUC pooled={auc_p:.3f} LOLO={auc_l:.3f} "
                         f"Spec@0.5={spec:.3f} Sens@0.5={sens:.3f}")
        d = np.array(deltas)
        p = sign_flip_p(d) if len(d) else float("nan")
        sig_rows.append({"model": mname, "n_groups_with_both_classes": len(d),
                         "mean_delta_auc": float(d.mean()) if len(d) else float("nan"),
                         "signflip_p": p,
                         "lolo_specs": "; ".join(f"{k}:{v:.2f}"
                                                 for k, v in lolo_specs.items())})
        lines.append(f"  ΔAUC(pooled−LOLO) 逐组: "
                     + ", ".join(f"{x:+.3f}" for x in d))
        lines.append(f"  精确 sign-flip（2^{len(d)}={2**len(d)} 枚举）p = {p:.4f}")
        lines.append("")

    lines.append("口径注：H0=逐组 ΔAUC 对称于 0；n_groups=5，2^5=32 枚举为精确检验，"
                 "不依赖大样本近似；cluster bootstrap CI（<10 clusters 反保守）"
                 "只作区间参考，显著性以此检验为准。")
    pd.DataFrame(per_group_rows).to_csv(OUT / "e3_per_group.csv", index=False)
    pd.DataFrame(sig_rows).to_csv(OUT / "e3_signflip.csv", index=False)
    txt = "\n".join(lines)
    (OUT / "e3_signflip_report.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    sys.exit(main())

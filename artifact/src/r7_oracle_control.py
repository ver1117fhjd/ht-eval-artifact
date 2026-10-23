# -*- coding: utf-8 -*-
"""r7_oracle_control.py —— R7 P0-3：可解性正对照（oracle 特征）
================================================================
堵"near chance 是协议坏了还是你特征不行"的标准攻击：
  ① TRIT 插入日志 oracle：从 log.txt TROJAN STATS/TROJAN BODY 解析特权特征
    （木马体门数/触发信号数/效应信号数/类型/是否时序），金样=全零，
    group-by-base 协议下跑 RF/XGB/MLP —— 可分=协议在有信号时测得出。
  ② E1 配对 oracle：特权配对差分特征（x_TjIn − x_TjFree，金样=零向量），
    group-by-base 下跑 —— 验证配对集协议同样有测量能力。
  ③ 标签置乱对照：E1 grouped 下置乱标签 → 应 ≈ 随机（协议无虚高）。

输出：experiments/c2/oracle_control.csv / oracle_control_report.txt
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


def parse_log(path):
    """log.txt → oracle 特征 dict。解析失败返回 None。"""
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")
    feats = {}
    m = re.search(r"Number of Trojans:\s*(\d+)", txt)
    feats["o_n_trojans"] = int(m.group(1)) if m else 1
    m = re.search(r"Type:\s*(\w+)", txt)
    t = (m.group(1).lower() if m else "combinational")
    feats["o_is_seq"] = 1.0 if "seq" in t else 0.0
    m = re.search(r"Activation Condition:.*?\n(.*?)\*{3,}", txt, re.S)
    trig = m.group(1) if m else ""
    feats["o_n_trigger"] = float(len(trig.split()))
    m = re.search(r"Effect:.*?\n(.*?)Activation", txt, re.S)
    eff = m.group(1) if m else ""
    feats["o_n_effect"] = float(len(eff.split()))
    m = re.search(r"TROJAN BODY:\s*\n(.*?)\n-{5,}", txt, re.S)
    body = m.group(1) if m else ""
    n_inst = len(re.findall(r"\)\s*;", body))
    feats["o_n_body_cells"] = float(n_inst)
    return feats if n_inst > 0 else None


def key_of(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def load_trit_oracle(df):
    """TRIT 行 ← log.txt oracle 特征；金样/解析失败=零。"""
    logs = {}
    for p in (ROOT / "data" / "Trust-Hub" / "chip-level-trojan").rglob("log.txt"):
        pf = parse_log(p)
        if pf:
            logs[key_of(p.parent.name)] = pf
    print(f"[oracle] parsed {len(logs)} TRIT logs")
    cols = ["o_n_trojans", "o_is_seq", "o_n_trigger", "o_n_effect",
            "o_n_body_cells"]
    O = np.zeros((len(df), len(cols)))
    hit = 0
    for i, row in df.iterrows():
        # ⚠️ 每变体的目录名在 source 列（'c2670_T000'），variant 列只有家族名
        k = key_of(row["source"])
        if re.search(r"_T\d+$", str(row["source"])) and k in logs:
            pf = logs[k]
            O[i] = [pf[c] for c in cols]
            hit += 1
    print(f"[oracle] matched {hit}/{len(df)} rows to logs")
    return O, cols


def main():
    from sklearn.model_selection import StratifiedGroupKFold
    rows = []
    lines = ["R7 可解性正对照（oracle 特征）",
             f"数据源: {c1.ALL_CSV}（v2，显式钉死）seed={SEED} n_jobs=1",
             ""]

    # ---------- ① TRIT 插入日志 oracle ----------
    df2 = c1.load_full()
    is_trit = df2["family"].str.startswith("TRIT").values
    dft = df2[is_trit].reset_index(drop=True)
    Xt, ocols = load_trit_oracle(dft)
    yt = dft["label"].values
    gt = dft["group"].values
    lines.append("==== ① TRIT log oracle（特权特征），group-by-base ====")
    lines.append(f"TRIT 子集 n={len(dft)} pos={int(yt.sum())} "
                 f"groups={len(set(gt))} 特征={ocols}")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for mname, model in make_models1().items():
        from sklearn.model_selection import cross_val_predict
        prob = cross_val_predict(model, Xt, yt, cv=sgkf, groups=gt,
                                 method="predict_proba")[:, 1]
        m = c1.metrics(yt, prob)
        ci = c1.boot_ci(yt, prob, gt)
        rows.append({"experiment": "trit-log-oracle", "protocol": "group-by-base",
                     "model": mname, **{k: m[k] for k in
                     ("AP", "ROC_AUC", "F1", "Sens", "Spec", "BalAcc", "Gmean")},
                     "AUC_lo": ci["ROC_AUC"][1], "AUC_hi": ci["ROC_AUC"][2]})
        lines.append(f"  {mname:4s} AP={m['AP']:.3f} AUC={m['ROC_AUC']:.3f} "
                     f"[{ci['ROC_AUC'][1]:.3f},{ci['ROC_AUC'][2]:.3f}] "
                     f"Sens={m['Sens']:.3f} Spec={m['Spec']:.3f} "
                     f"Bal={m['BalAcc']:.3f}")

    # ---------- ② E1 配对差分 oracle ----------
    lines.append("")
    lines.append("==== ② E1 配对差分 oracle（特权配对信息），group-by-base ====")
    df1 = c1.load_pairs().reset_index(drop=True)
    Xf = df1[c1.feature_cols(df1)].values
    y1 = df1["label"].values
    g1 = df1["group"].values
    # 配对键：(family, variant)。TjIn 行 = x_in − x_free；TjFree 行 = 0
    Xo = np.zeros_like(Xf)
    pairkey = list(zip(df1["family"], df1["variant"]))
    src = df1["source"].replace({"TnIn": "TjIn"}).values
    from collections import defaultdict
    idx = defaultdict(dict)
    for i, k in enumerate(pairkey):
        idx[k][src[i]] = i
    npairs = 0
    for k, d in idx.items():
        if "TjIn" in d and "TjFree" in d:
            Xo[d["TjIn"]] = Xf[d["TjIn"]] - Xf[d["TjFree"]]
            npairs += 1
    lines.append(f"E1 n={len(df1)} pos={int(y1.sum())} groups={len(set(g1))} "
                 f"配对数={npairs}（差分特征 32 维）")
    sgkf1 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    from sklearn.model_selection import cross_val_predict
    for mname, model in make_models1().items():
        prob = cross_val_predict(model, Xo, y1, cv=sgkf1, groups=g1,
                                 method="predict_proba")[:, 1]
        m = c1.metrics(y1, prob)
        ci = c1.boot_ci(y1, prob, g1)
        rows.append({"experiment": "e1-pair-diff-oracle", "protocol": "group-by-base",
                     "model": mname, **{k: m[k] for k in
                     ("AP", "ROC_AUC", "F1", "Sens", "Spec", "BalAcc", "Gmean")},
                     "AUC_lo": ci["ROC_AUC"][1], "AUC_hi": ci["ROC_AUC"][2]})
        lines.append(f"  {mname:4s} AP={m['AP']:.3f} AUC={m['ROC_AUC']:.3f} "
                     f"[{ci['ROC_AUC'][1]:.3f},{ci['ROC_AUC'][2]:.3f}] "
                     f"Sens={m['Sens']:.3f} Spec={m['Spec']:.3f} "
                     f"Bal={m['BalAcc']:.3f}")

    # ---------- ③ 标签置乱对照 ----------
    lines.append("")
    lines.append("==== ③ 标签置乱对照（E1 grouped，1 seed）应 ≈ 随机 ====")
    rng = np.random.default_rng(SEED)
    y_shuf = rng.permutation(y1)
    for mname, model in make_models1().items():
        prob = cross_val_predict(model, Xf, y_shuf, cv=sgkf1, groups=g1,
                                 method="predict_proba")[:, 1]
        m = c1.metrics(y_shuf, prob)
        rows.append({"experiment": "e1-shuffled-labels", "protocol": "group-by-base",
                     "model": mname, **{k: m[k] for k in
                     ("AP", "ROC_AUC", "F1", "Sens", "Spec", "BalAcc", "Gmean")},
                     "AUC_lo": np.nan, "AUC_hi": np.nan})
        lines.append(f"  {mname:4s} AUC={m['ROC_AUC']:.3f} Bal={m['BalAcc']:.3f}")

    lines.append("")
    lines.append("解读口径：①/② 可分 = 协议在特权信号存在时测得出，"
                 "near-chance 是语料/特征下限而非协议 bug；③ ≈0.5 = grouped 协议无虚高。")
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "oracle_control.csv", index=False)
    txt = "\n".join(lines)
    (OUT / "oracle_control_report.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    sys.exit(main())

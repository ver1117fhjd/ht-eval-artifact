# -*- coding: utf-8 -*-
"""把 gf180 的两份新数据并入 E3 数据集，得到四库版。

项目布局：experiments/e3x/... + experiments/synth_work/...
artifact 布局：results/...（同名文件）
输出：features_multilib_v2_ext_4lib.csv（516 行；标注子集 472 行）
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = (ROOT / "paper").exists()          # 项目根有 paper/，artifact 包没有
if PROJECT:
    A = ROOT / "experiments/e3x/features_multilib_v2_ext.csv"
    B = ROOT / "experiments/synth_work/features_multilib_gf180_v2.csv"
    C = ROOT / "experiments/e3x/features_multilib_ext_gf180.csv"
    OUT = ROOT / "experiments/e3x/features_multilib_v2_ext_4lib.csv"
else:
    A = ROOT / "results/features_multilib_v2_ext.csv"
    B = ROOT / "results/features_multilib_gf180_v2.csv"
    C = ROOT / "results/features_multilib_ext_gf180.csv"
    OUT = ROOT / "results/features_multilib_v2_ext_4lib.csv"

for p in (A, B, C):
    if not p.exists():
        sys.exit(f"缺输入: {p}")

a, b, c = (pd.read_csv(p) for p in (A, B, C))
print("A", a.shape, "B", b.shape, "C", c.shape)

feat_cols = [x for x in a.columns if x not in ("family", "variant", "source", "tech", "module")]
for name, df in (("B", b), ("C", c)):
    missing = [x for x in feat_cols if x not in df.columns]
    extra = [x for x in df.columns if x not in a.columns]
    print(f"{name}: 缺列 {len(missing)} 多余列 {len(extra)}")
    for x in missing:
        df[x] = 0.0

out = pd.concat([a, b[a.columns], c[a.columns]], ignore_index=True)
print("\n合并后:", out.shape)
print(out.groupby("tech").size().to_string())

assert len(out) == 516, len(out)
assert set(out.tech) == {"asap7", "gf180", "nangate45", "sky130hd"}, set(out.tech)
out.to_csv(OUT, index=False)
print(f"\n写出 {OUT}")

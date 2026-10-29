# -*- coding: utf-8 -*-
"""GF180 第四个库：把 99 个 RTL 设计的 gf180 综合结果重放成 v1+v2 特征行。

复用 c2_normalize 的 job_rows（与三库表完全同口径），只换库、只写新文件，
不碰 features_multilib_v2.csv（保护已发表的数字）。
输出：experiments/synth_work/features_multilib_gf180_v2.csv
"""
import csv
import importlib.util
import sys
from pathlib import Path

ROOT = Path(r"PROJECT_ROOT")
spec = importlib.util.spec_from_file_location("c2n", ROOT / "src/c2_normalize.py")
c2 = importlib.util.module_from_spec(spec)
sys.argv = [sys.argv[0]]          # 防止模块级参数解析被外部参数干扰
spec.loader.exec_module(c2)

GF180_LIB = r"PROJECT_ROOT\libs\liberty\gf180\gf180mcu_fd_sc_mcu7t5v0__tt_025C_1v80.lib"
c2.LIBS["gf180"] = GF180_LIB

jobs = c2.find_rtl_designs()
print(f"RTL 设计 {len(jobs)} 个，重放 gf180 特征", flush=True)
ops = c2.make_ops_map("gf180")
rows = []
missing = []
for i, (fam, variant, label, _d) in enumerate(jobs, 1):
    n = c2.job_rows(fam, variant, label, "gf180", ops, rows)
    if not n:
        missing.append(f"{fam}/{variant}{'::' + label if label else ''}")
    if i % 25 == 0:
        print(f"  [{i}/{len(jobs)}] rows={len(rows)}", flush=True)

out = ROOT / "experiments/synth_work/features_multilib_gf180_v2.csv"
cols = ["family", "variant", "source", "tech", "module"] + sorted(
    {k for r in rows for k in r} - {"family", "variant", "source", "tech", "module"})
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)
print(f"\n写出 {out} : {len(rows)} 行，{len(cols)} 列")
if missing:
    print(f"缺综合结果 {len(missing)} 个:", missing[:8])

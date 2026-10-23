# -*- coding: utf-8 -*-
"""
c2_normalize.py —— C2 验证：算子归一化特征重放 + 跨库漂移对照
==============================================================

对 synth_work 里存档的每个 netlist.json 重提 v2 特征（cell 类占比 + 算子空间
特征），产出：
  experiments/synth_work/features_all_v2.csv        （generic 全量，行对齐 v1）
  experiments/synth_work/features_multilib_v2.csv   （99 设计 ×3 库，行对齐 v1）
  experiments/c2/c2_drift_report.txt                （跨库漂移：v1 vs v2 对照）
  experiments/c2/c2_env.txt                         （工具版本钉死）

元数据与 v1 完全同源（复用 batch_synth.find_designs 的作业清单），保证行可对齐。
用法：./.venv/Scripts/python.exe src/c2_normalize.py [--limit N] [--techs ...]
"""
import os
import re
import sys
import csv
import json
import argparse
from collections import defaultdict

SRC = os.path.dirname(os.path.abspath(__file__))
WORK = r"./experiments\synth_work"
OUT = r"./experiments\c2"
sys.path.insert(0, SRC)
from batch_synth import find_designs, flatten_features  # noqa: E402
from batch_multilib import find_rtl_designs  # noqa: E402
from netlist_features import module_features  # noqa: E402
from lib_ops import load_lib_ops, generic_cell_ops  # noqa: E402

ROOT = r"./data\Trust-Hub\chip-level-trojan"
LIBS = {
    "nangate45": r"./libs\liberty\nangate45\NangateOpenCellLibrary_typical.lib",
    "sky130hd": r"./libs\liberty\sky130hd\sky130_fd_sc_hd__tt_025C_1v80.lib",
    "asap7": r"./libs\liberty\asap7\asap7_RVT_TT_merged.lib",
}


def make_ops_map(tech=None):
    ops = dict(generic_cell_ops())
    if tech:
        ops.update(load_lib_ops(LIBS[tech]))
    return ops


def job_rows(fam, variant, label, tech, ops_map, rows):
    """一个 (设计, 工艺) 组合 → 合并 v1+v2 特征行。"""
    outdir = os.path.join(WORK, fam, variant + ("__" + label if label else "")
                          + (f"__{tech}" if tech else ""))
    fj = os.path.join(outdir, "features.json")
    nj = os.path.join(outdir, "netlist.json")
    if not (os.path.isfile(fj) and os.path.isfile(nj)):
        return 0
    with open(fj, encoding="utf-8") as f:
        meta = json.load(f)
    modules = list(meta.get("features", {}).keys())
    with open(nj, encoding="utf-8") as f:
        netlist = json.load(f)
    n = 0
    for mod in modules:
        m = netlist.get("modules", {}).get(mod)
        if m is None:
            continue
        feat = module_features(m, ops_map=ops_map)
        row = flatten_features({mod: feat})
        row.update({"family": fam, "variant": variant, "source": label,
                    "module": mod})
        if tech:
            row["tech"] = tech
        rows.append(row)
        n += 1
    return n


def write_csv(rows, path, meta_cols):
    cols = meta_cols + sorted({k for r in rows for k in r if k not in meta_cols})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


# ---------------------------------------------------------------- 漂移对照

FRAC_OLD = ["frac_and", "frac_or", "frac_xor", "frac_inv", "frac_mux",
            "frac_complex", "frac_buf", "frac_seq"]
FRAC_NEW = ["frac_op_and", "frac_op_or", "frac_op_xor", "frac_op_not",
            "frac_op_mux"]
SCALAR = ["comb_depth_max", "comb_opdepth_max"]


def drift_report(csv_path):
    """同一设计跨库的特征漂移（max/min 倍率），v1 vs v2 对照。"""
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    groups = defaultdict(dict)  # (family,variant,source) -> tech -> row
    for r in rows:
        key = (r["family"], r["variant"], r["source"])
        groups[key][r.get("tech", "?")] = r
    groups = {k: v for k, v in groups.items() if len(v) >= 3}

    def ratio(vals, eps=1e-4):
        vals = [max(float(v), eps) for v in vals]
        return max(vals) / min(vals)

    stats = defaultdict(list)
    for key, bytech in groups.items():
        techs = sorted(bytech)
        for feat in FRAC_OLD + FRAC_NEW + SCALAR:
            vals = [bytech[t].get(feat, "") for t in techs]
            if any(v in ("", None) for v in vals):
                continue
            stats[feat].append(ratio(vals))

    lines = [f"C2 跨库漂移对照（{len(groups)} 个设计 × 3 库，max/min 倍率；越接近 1 越好）",
             f"数据: {csv_path}", ""]
    import statistics
    lines.append(f"{'特征':24s} {'中位漂移':>10s} {'最大漂移':>10s}  组")
    for feat in FRAC_OLD:
        if feat in stats:
            v = stats[feat]
            lines.append(f"v1 {feat:20s} {statistics.median(v):10.2f} {max(v):10.2f}  {len(v)}")
    for feat in FRAC_NEW:
        if feat in stats:
            v = stats[feat]
            lines.append(f"v2 {feat:20s} {statistics.median(v):10.2f} {max(v):10.2f}  {len(v)}")
    for feat in SCALAR:
        if feat in stats:
            v = stats[feat]
            lines.append(f"   {feat:21s} {statistics.median(v):10.2f} {max(v):10.2f}  {len(v)}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-multilib", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    # ---- generic 全量 ----
    jobs = []
    for dirpath, dirnames, _f in os.walk(ROOT):
        if os.path.basename(dirpath).lower() != "extracted":
            continue
        fam = os.path.relpath(dirpath, ROOT).split(os.sep)[0]
        variant = os.path.basename(os.path.dirname(dirpath))
        for label, d in find_designs(dirpath):
            jobs.append((fam, variant, label))
    jobs.sort()
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"generic 重放: {len(jobs)} 设计", flush=True)

    ops_generic = make_ops_map(None)
    rows_all = []
    for i, (fam, variant, label) in enumerate(jobs, 1):
        if i % 100 == 0:
            print(f"  [{i}/{len(jobs)}] ...", flush=True)
        job_rows(fam, variant, label, None, ops_generic, rows_all)
    n1 = write_csv(rows_all, os.path.join(WORK, "features_all_v2.csv"),
                   ["family", "variant", "source", "module"])
    print(f"features_all_v2.csv: {n1} 行")

    # ---- multilib ----
    if not args.skip_multilib:
        mjobs = find_rtl_designs()
        if args.limit:
            mjobs = mjobs[:args.limit]
        print(f"multilib 重放: {len(mjobs)} 设计 × 3 库", flush=True)
        ops_maps = {t: make_ops_map(t) for t in LIBS}
        rows_ml = []
        for i, (fam, variant, label, _d) in enumerate(mjobs, 1):
            if i % 50 == 0:
                print(f"  [{i}/{len(mjobs)}] ...", flush=True)
            for tech in LIBS:
                job_rows(fam, variant, label, tech, ops_maps[tech], rows_ml)
        n2 = write_csv(rows_ml, os.path.join(WORK, "features_multilib_v2.csv"),
                       ["family", "variant", "source", "tech", "module"])
        print(f"features_multilib_v2.csv: {n2} 行")

        txt = drift_report(os.path.join(WORK, "features_multilib_v2.csv"))
        with open(os.path.join(OUT, "c2_drift_report.txt"), "w", encoding="utf-8") as f:
            f.write(txt + "\n")
        print(txt)

    import platform
    with open(os.path.join(OUT, "c2_env.txt"), "w", encoding="utf-8") as f:
        f.write(f"python {platform.python_version()}\nlib_ops: 真值表模板匹配 "
                f"(MAX_CATALOG_VARS=6)\nseed 见 c1_protocol.py\n")


if __name__ == "__main__":
    main()

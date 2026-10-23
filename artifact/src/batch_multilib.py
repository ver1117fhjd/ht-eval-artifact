#!/usr/bin/env python3
"""batch_multilib.py —— 对 RTL 设计做三工艺库（nangate45 / sky130hd / asap7）综合，
产出跨库特征表（C1 跨库协议 / C2 库无关特征的核心实验数据）。

只跑纯 RTL 设计（generic 跑里 n_blackbox_stubs == 0）：门级网表已绑定原库，无法重映射。
用法：python batch_multilib.py [--family RS232] [--techs nangate45,sky130hd] [--limit N]
"""
import os
import sys
import csv
import json
import argparse
import subprocess

SRC = os.path.dirname(os.path.abspath(__file__))
WORK = r"./experiments\synth_work"
sys.path.insert(0, SRC)
from batch_synth import find_designs, flatten_features  # noqa: E402

TECHS = {
    "nangate45": (r"./libs\liberty\nangate45\NangateOpenCellLibrary_typical.lib", None),
    "sky130hd": (r"./libs\liberty\sky130hd\sky130_fd_sc_hd__tt_025C_1v80.lib", None),
    "asap7": (r"./libs\liberty\asap7\asap7_RVT_TT_merged.lib",
              r"./libs\liberty\asap7\asap7_RVT_TT_comb.lib"),
}


def find_rtl_designs(family=None):
    """从 generic 结果里筛出纯 RTL 设计（黑盒桩为 0）。"""
    jobs = []
    for dirpath, dirnames, _f in os.walk(r"./data\Trust-Hub\chip-level-trojan"):
        if os.path.basename(dirpath).lower() != "extracted":
            continue
        fam = os.path.relpath(dirpath, r"./data\Trust-Hub\chip-level-trojan").split(os.sep)[0]
        if family and fam != family:
            continue
        variant = os.path.basename(os.path.dirname(dirpath))
        for label, d in find_designs(dirpath):
            gj = os.path.join(WORK, fam, variant + ("__" + label if label else ""), "features.json")
            if not os.path.isfile(gj):
                continue
            try:
                if json.load(open(gj, encoding="utf-8")).get("n_blackbox_stubs", 1) != 0:
                    continue
            except Exception:
                continue
            jobs.append((fam, variant, label, d))
    return sorted(jobs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default=None)
    ap.add_argument("--techs", default="nangate45,sky130hd,asap7")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    techs = [t.strip() for t in args.techs.split(",") if t.strip() in TECHS]

    jobs = find_rtl_designs(args.family)
    if args.limit:
        jobs = jobs[: args.limit]
    total = len(jobs) * len(techs)
    print(f"RTL 设计 {len(jobs)} 个 × {len(techs)} 库 = {total} 次综合")

    rows, done, failed = [], 0, []
    k = 0
    for fam, variant, label, d in jobs:
        tag = f"{fam}/{variant}" + (f"::{label}" if label else "")
        for tech in techs:
            k += 1
            outdir = os.path.join(WORK, fam,
                                  variant + ("__" + label if label else "") + f"__{tech}")
            fj = os.path.join(outdir, "features.json")
            if not os.path.isfile(fj):
                print(f"[{k}/{total}] {tag} @ {tech} ...", flush=True)
                lib, lib_abc = TECHS[tech]
                cmd = [sys.executable, os.path.join(SRC, "run_yosys_synth.py"), d,
                       "--out", outdir, "--label", label or variant,
                       "--liberty", lib]
                if lib_abc:
                    cmd += ["--liberty-abc", lib_abc]
                try:
                    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
                    if not os.path.isfile(fj):
                        failed.append(f"{tag}@{tech}")
                        print("    失败:", (r.stderr or r.stdout)[-200:].replace("\n", " "))
                        continue
                except subprocess.TimeoutExpired:
                    failed.append(f"{tag}@{tech}")
                    print("    超时")
                    continue
            try:
                data = json.load(open(fj, encoding="utf-8"))
                for mod, feat in data.get("features", {}).items():
                    row = flatten_features({mod: feat})
                    row.update({"family": fam, "variant": variant, "source": label,
                                "tech": tech, "module": mod})
                    rows.append(row)
                done += 1
            except Exception as e:
                failed.append(f"{tag}@{tech}")
                print("    读取失败:", e)

    cols = ["family", "variant", "source", "tech", "module"] + sorted(
        {k2 for r in rows for k2 in r
         if k2 not in ("family", "variant", "source", "tech", "module")})
    out_csv = os.path.join(WORK, "features_multilib.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\n完成: {done} 次, 失败 {len(failed)}")
    if failed:
        print("失败清单:", failed[:15])
    print(f"跨库汇总表: {out_csv}  （{len(rows)} 行）")


if __name__ == "__main__":
    main()

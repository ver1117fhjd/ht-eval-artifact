#!/usr/bin/env python3
"""batch_synth.py —— 批量跑 Trust-Hub 全部设计的综合 + 特征提取，汇总成一张大表。

布局适配（普查 17 家族后的规则）：
  - TjIn / TjFree 叶目录 → 各自成设计（金样/木马对）
  - original_designs / golden 类叶目录 → 每个 .v 独立成设计（多基准混放）
  - *_T### / *-T### 叶目录 → 各自成设计（TRIT 式变体）
  - 其余 → 合并为一个设计（run_yosys_synth 内部按 basename 去重）
用法：python batch_synth.py [--limit N] [--family RS232] [--force]
"""
import os
import re
import sys
import csv
import json
import argparse
import subprocess

SRC = os.path.dirname(os.path.abspath(__file__))
WORK = r"./experiments\synth_work"
ROOT = r"./data\Trust-Hub\chip-level-trojan"
sys.path.insert(0, SRC)
from run_yosys_synth import DIR_EXCL, FILE_TB, HDL_EXT  # noqa: E402

VARIANT_DIR_RE = re.compile(r"(?i)[_-]t\d+$")


def leaf_source_dirs(extracted):
    """extracted 下直接含（过滤后）.v 的目录。"""
    leaves = []
    for dirpath, dirnames, filenames in os.walk(extracted):
        dirnames[:] = [d for d in dirnames if not DIR_EXCL.match(d)]
        for fn in filenames:
            low = fn.lower()
            if os.path.splitext(low)[1] in HDL_EXT and not fn.endswith("~") \
                    and not FILE_TB.search(low) and low.endswith((".v", ".sv")):
                leaves.append(dirpath)
                break
    return sorted(set(leaves))


def find_designs(extracted):
    """返回 [(label, root_dir)]。"""
    leaves = leaf_source_dirs(extracted)
    if not leaves:
        return []
    names = [os.path.basename(d).lower() for d in leaves]

    # 1) TjIn / TjFree 对
    if any(n in ("tjin", "tjfree") for n in names):
        return [(os.path.basename(d), d) for d in leaves]

    # 2) original/golden 类目录 → 每文件一个设计；其余叶目录各自成设计
    if any(re.search(r"(?i)original|golden|nominal|clean", n) for n in names):
        out = []
        for d in leaves:
            n = os.path.basename(d).lower()
            if re.search(r"(?i)original|golden|nominal|clean", n):
                for fn in sorted(os.listdir(d)):
                    if fn.lower().endswith((".v", ".sv")) and not FILE_TB.search(fn.lower()):
                        out.append((f"{os.path.basename(d)}/{fn}", d))
            else:
                out.append((os.path.basename(d), d))
        return out

    # 3) 变体目录（xxx_T400 式）各自成设计
    if any(VARIANT_DIR_RE.search(n) for n in names):
        return [(os.path.basename(d), d) for d in leaves]

    # 4) 常规：整棵树一个设计
    return [("", extracted)]


def flatten_features(feat):
    row = {}
    for _mod, f in feat.items():
        for k, v in f.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    row[f"{k}_{k2}"] = v2
            else:
                row[k] = v
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--family", default=None)
    ap.add_argument("--force", action="store_true", help="重跑已有结果")
    args = ap.parse_args()

    jobs = []
    for dirpath, dirnames, _filenames in os.walk(ROOT):
        if os.path.basename(dirpath).lower() != "extracted":
            continue
        fam = os.path.relpath(dirpath, ROOT).split(os.sep)[0]
        if args.family and fam != args.family:
            continue
        variant = os.path.basename(os.path.dirname(dirpath))
        for label, d in find_designs(dirpath):
            jobs.append((fam, variant, label, d))
    jobs.sort()
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"待跑设计: {len(jobs)}")

    rows, done, failed = [], 0, []
    for i, (fam, variant, label, d) in enumerate(jobs, 1):
        tag = f"{fam}/{variant}" + (f"::{label}" if label else "")
        outdir = os.path.join(WORK, fam, variant + ("__" + label if label else ""))
        fj = os.path.join(outdir, "features.json")
        if os.path.isfile(fj) and not args.force:
            pass  # 复用
        else:
            print(f"[{i}/{len(jobs)}] {tag} ...", flush=True)
            try:
                r = subprocess.run(
                    [sys.executable, os.path.join(SRC, "run_yosys_synth.py"), d,
                     "--out", outdir, "--label", label or variant],
                    capture_output=True, text=True, timeout=1800)
                if not os.path.isfile(fj):
                    failed.append(tag)
                    print("    失败:", (r.stderr or r.stdout)[-200:].replace("\n", " "))
                    continue
            except subprocess.TimeoutExpired:
                failed.append(tag)
                print("    超时")
                continue
        try:
            data = json.load(open(fj, encoding="utf-8"))
            for mod, feat in data.get("features", {}).items():
                row = flatten_features({mod: feat})
                row.update({"family": fam, "variant": variant,
                            "source": label, "module": mod})
                rows.append(row)
            done += 1
        except Exception as e:
            failed.append(tag)
            print("    读取失败:", e)

    cols = ["family", "variant", "source", "module"] + sorted(
        {k for r in rows for k in r
         if k not in ("family", "variant", "source", "module")})
    out_csv = os.path.join(WORK, "features_all.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\n完成: {done} 设计, 失败 {len(failed)}")
    if failed:
        print("失败清单:", failed[:15])
    print(f"汇总表: {out_csv}  （{len(rows)} 行）")


if __name__ == "__main__":
    main()

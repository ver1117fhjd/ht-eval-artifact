#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""aig_ablation.py —— P1 AIG 归一化消融（外部评审建议「为什么不用 AIG」）。

两个 AIG 基线（与 v2 算子特征对比）：
  A = 技术无关逻辑直映：generic 网表（已存档）→ dfflibmap -liberty <lib> → abc -g AND
  B = 已映射网表再归一化：tech 网表 → 单元功能模型展开 → flatten → abc -g AND
评价维度：① 跨库漂移（与 c2_drift_report 同口径：max/min 倍率，中位/最大）
          ② 分解可加性探针（AOI21 的 AIG 节点数 vs and+or+inv 之和）
          ③ 设计间区分度（唯一组合特征向量数 / 跨设计标准差）

输出：experiments/c2/aigA_features.csv / aigB_features.csv / aig_report.txt
      每个目录留 aigA_features.json / aigB_features.json 作为断点续跑标记。

用法: python aig_ablation.py [--workers 4] [--limit N] [--only A|B|probe|report]
"""
import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)

ROOT = r"./"
WORK = os.path.join(ROOT, "experiments", "synth_work")
OUT = os.path.join(ROOT, "experiments", "c2")
MODELS_DIR = os.path.join(OUT, "aig_models")
YOSYS = os.environ.get("YOSYS", r"<local-path>")  # override via env var
TECHS = ("asap7", "nangate45", "sky130hd")
FEATS = ["frac_and", "frac_inv", "frac_seq", "frac_other",
         "comb_depth_max", "comb_depth_mean", "n_cells", "n_seq"]


# ---------------------------------------------------------------- 工具

def parse_synth_ys(ys_path):
    """取 dfflibmap/abc 的 liberty 路径（A/B 复用同库，保证口径一致）。"""
    dfflib = abclib = None
    for line in open(ys_path, encoding="utf-8", errors="replace"):
        line = line.strip()
        m = re.match(r"dfflibmap\s+-liberty\s+(\S+)", line)
        if m:
            dfflib = m.group(1)
        m = re.match(r"abc\s+-liberty\s+(\S+)", line)
        if m:
            abclib = m.group(1)
    return dfflib, abclib


def tech_dirs():
    out = []
    for t in TECHS:
        out += sorted(glob.glob(os.path.join(WORK, "*", "*__" + t)))
    return out


def meta_from_dir(techdir):
    base = os.path.basename(techdir)            # AES-T1000__TjIn__nangate45
    fam = os.path.basename(os.path.dirname(techdir))
    stem, tech = base.rsplit("__", 1)           # AES-T1000__TjIn, nangate45
    parts = stem.split("__")
    source = parts[1] if len(parts) > 1 else ""
    return {"family": fam, "variant": stem, "source": source, "tech": tech}


def run_yosys(script_path, log_path):
    r = subprocess.run([YOSYS, "-s", script_path, "-l", log_path],
                       capture_output=True, text=True, timeout=3600)
    return r.returncode, (r.stdout or "")[-2000:]


def extract_one(json_path):
    from netlist_features import extract
    out = extract(json_path)
    if not out:
        raise RuntimeError("no module in %s" % json_path)
    if len(out) == 1:
        return list(out.items())[0]
    return max(out.items(), key=lambda kv: kv[1].get("n_cells", 0))


# ---------------------------------------------------------------- 单任务

def job_one(args):
    techdir, mode, models_v = args
    tag = "aig%s" % mode
    feat_json = os.path.join(techdir, tag + "_features.json")
    if os.path.exists(feat_json):
        return techdir, mode, "skip"
    dfflib, abclib = parse_synth_ys(os.path.join(techdir, "synth.ys"))
    net_json = os.path.join(techdir, tag + "_netlist.json")
    ys = os.path.join(techdir, tag + ".ys")
    log = os.path.join(techdir, tag + "_yosys.log")
    q = lambda p: p.replace("\\", "/")
    if mode == "A":
        gen_json = os.path.join(os.path.dirname(techdir),
                                os.path.basename(techdir).rsplit("__", 1)[0],
                                "netlist.json")
        if not os.path.exists(gen_json):
            return techdir, mode, "no-generic-netlist"
        lines = ["read_json %s" % q(gen_json),
                 "dfflibmap -liberty %s" % dfflib,
                 "simplemap", "opt", "abc -g AND", "opt", "flatten", "opt",
                 "write_json %s" % q(net_json), "stat"]
    else:
        src_json = os.path.join(techdir, "netlist.json")
        if not os.path.exists(src_json):
            return techdir, mode, "no-tech-netlist"
        lines = ["read_json %s" % q(src_json),
                 "read_verilog %s" % q(models_v),
                 "flatten", "opt", "simplemap", "opt", "abc -g AND", "opt_clean",
                 "write_json %s" % q(net_json), "stat"]
    with open(ys, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    try:
        rc, tail = run_yosys(ys, log)
    except Exception as e:
        return techdir, mode, "yosys-exc:%s" % e
    if rc != 0 or not os.path.exists(net_json):
        return techdir, mode, "yosys-rc%d" % rc
    try:
        mod, feats = extract_one(net_json)
    except Exception as e:
        return techdir, mode, "extract-exc:%s" % e
    with open(feat_json, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"module": mod, "features": feats}, f)
    return techdir, mode, "ok"


# ---------------------------------------------------------------- 汇总/分析

def load_rows(mode):
    rows = []
    for td in tech_dirs():
        fj = os.path.join(td, "aig%s_features.json" % mode)
        if not os.path.exists(fj):
            continue
        d = json.load(open(fj, encoding="utf-8"))
        row = meta_from_dir(td)
        row["module"] = d["module"]
        row.update(d["features"])
        rows.append(row)
    return rows


def write_csv(rows, path):
    meta = ["family", "variant", "source", "tech", "module"]
    keys = sorted({k for r in rows for k in r} - set(meta))
    cols = meta + keys
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return len(rows)


def drift_stats(rows, feats):
    """与 c2_normalize.drift_report 同口径：组内 max/min（clip 1e-4），组间中位/最大。"""
    groups = {}
    for r in rows:
        key = (r["family"], r["variant"], r["source"])
        groups.setdefault(key, {})[r["tech"]] = r
    groups = {k: v for k, v in groups.items() if len(v) >= 3}
    out = {}
    for feat in feats:
        vals = []
        for g in groups.values():
            vs = []
            ok = True
            for t in sorted(g):
                v = g[t].get(feat)
                if v in (None, ""):
                    ok = False
                    break
                vs.append(max(float(v), 1e-4))
            if ok and min(vs) > 0:
                vals.append(max(vs) / min(vs))
        if vals:
            import statistics
            out[feat] = (statistics.median(vals), max(vals), len(vals))
    return len(groups), out


def csv_rows(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


def degeneracy_stats(rows, comp_feats, label):
    """设计间区分度：唯一组合向量数（按 design 聚合 3 库取 nangate45 行）+ 跨设计 std。"""
    import statistics
    one = [r for r in rows if r.get("tech") == "nangate45"]
    if not one:
        one = rows
    vecs, stds = set(), {}
    for r in one:
        vecs.add(tuple(round(float(r.get(f, 0) or 0), 4) for f in comp_feats))
    for f in comp_feats:
        vs = [float(r.get(f, 0) or 0) for r in one]
        if len(vs) > 1:
            stds[f] = statistics.pstdev(vs)
    return "%s: n_design=%d unique_vectors=%d std=%s" % (
        label, len(one), len(vecs),
        " ".join("%s=%.4f" % (k, v) for k, v in sorted(stds.items())))


PROBE_FUNCS = {
    "and2":  ("A B", "A&B"),
    "or2":   ("A B", "A|B"),
    "inv1":  ("A", "!A"),
    "xor2":  ("A B", "A^B"),
    "AOI21": ("A B1 B2", "!((B1&B2)|A)"),
    "AOI211": ("A B C D E", "!((A&B)|(C&D)|E)"),
    "OAI211": ("A B C D E", "!((A|B)&(C|D)&E)"),
    "mux2":  ("S A B", "(S&B)|(!S&A)"),
}


def probe():
    """分解可加性探针：复合单元的 AIG 节点数 vs 其基础算子分解之和。"""
    import tempfile
    res = {}
    tmp = os.path.join(OUT, "aig_probe")
    os.makedirs(tmp, exist_ok=True)
    for name, (ins, expr) in PROBE_FUNCS.items():
        v = os.path.join(tmp, name + ".v")
        j = os.path.join(tmp, name + ".json")
        lg = os.path.join(tmp, name + ".log")
        with open(v, "w", encoding="utf-8", newline="\n") as f:
            f.write("module %s(%s, Y);\ninput %s;\noutput Y;\nassign Y = (%s);\nendmodule\n"
                    % (name, ins.replace(" ", ", "), ins.replace(" ", ", "), expr))
        ys = os.path.join(tmp, name + ".ys")
        with open(ys, "w", encoding="utf-8", newline="\n") as f:
            f.write("read_verilog %s\nhierarchy -top %s\nsynth -top %s\n"
                    "abc -g AND\nopt_clean\nwrite_json %s\nltp -noff\n"
                    % (v.replace("\\", "/"), name, name, j.replace("\\", "/")))
        rc, tail = run_yosys(ys, lg)
        n_and = n_not = 0
        depth = None
        if os.path.exists(j):
            data = json.load(open(j))
            cells = data["modules"][name]["cells"]
            for c in cells.values():
                t = c["type"]
                if t == "$_AND_":
                    n_and += 1
                elif t == "$_NOT_":
                    n_not += 1
        for line in open(lg, encoding="utf-8", errors="replace"):
            if "topological path" in line:
                nums = re.findall(r"(\d+)", line)
                if nums:
                    depth = int(nums[-1])
        res[name] = (n_and, n_not, n_and + n_not, depth)
    return res


def report():
    lines = ["AIG 归一化消融报告（评审建议：为什么不用 AIG）",
             "口径：漂移=max/min 倍率（clip 1e-4）中位/最大，与 c2_drift_report 一致",
             ""]
    # 漂移对照
    import c2_normalize  # noqa
    v1 = csv_rows(os.path.join(WORK, "features_multilib.csv"))
    v2 = csv_rows(os.path.join(WORK, "features_multilib_v2.csv"))
    a_rows, b_rows = load_rows("A"), load_rows("B")
    write_csv(a_rows, os.path.join(OUT, "aigA_features.csv"))
    write_csv(b_rows, os.path.join(OUT, "aigB_features.csv"))
    drift_feats = ["frac_and", "frac_inv", "frac_seq", "comb_depth_max"]
    lines.append("== ① 跨库漂移（99 设计 × 3 库）==")
    for label, rows in (("v1 单元类", v1), ("v2 算子(v2 列)", v2),
                        ("AIG-A 直映", a_rows), ("AIG-B 再归一", b_rows)):
        fts = [f for f in drift_feats if any(f in r for r in rows)]
        if label.startswith("v2"):
            fts = ["frac_op_and", "frac_op_or", "frac_op_not", "frac_op_xor",
                   "frac_op_mux", "comb_opdepth_max"]
        n, st = drift_stats(rows, fts)
        lines.append("-- %s（%d 组）" % (label, n))
        for f, (med, mx, cnt) in sorted(st.items()):
            lines.append("   %-18s 中位 %-8.3f 最大 %-9.3f (%d)" % (f, med, mx, cnt))
    # 可加性
    lines.append("")
    lines.append("== ② 分解可加性探针（AIG 节点数 and+not，深度=ltp）==")
    pr = probe()
    for name, (a, b, s, d) in sorted(pr.items()):
        lines.append("   %-8s and=%-3d not=%-3d nodes=%-3d depth=%s"
                     % (name, a, b, s, d))
    aoi = pr.get("AOI21")
    if aoi and all(k in pr for k in ("and2", "or2", "inv1")):
        parts = pr["and2"][2] + pr["or2"][2] + pr["inv1"][2]
        lines.append("   >>> AOI21 nodes=%d vs and+or+inv 之和=%d %s"
                     % (aoi[2], parts,
                        "（相等=可加）" if aoi[2] == parts else "（不等=违反 Definition 可加性）"))
    # 区分度
    lines.append("")
    lines.append("== ③ 设计间区分度（组合特征）==")
    lines.append("   " + degeneracy_stats(v1, ["frac_and", "frac_or", "frac_xor",
                                               "frac_inv", "frac_mux"], "v1"))
    lines.append("   " + degeneracy_stats(v2, ["frac_op_and", "frac_op_or", "frac_op_xor",
                                               "frac_op_not", "frac_op_mux"], "v2"))
    lines.append("   " + degeneracy_stats(a_rows, ["frac_and", "frac_inv"], "AIG-A"))
    lines.append("   " + degeneracy_stats(b_rows, ["frac_and", "frac_inv"], "AIG-B"))
    txt = "\n".join(lines)
    with open(os.path.join(OUT, "aig_report.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", choices=["A", "B", "probe", "report"], default=None)
    args = ap.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    if args.only in (None, "A", "B"):
        # 生成/复用单元功能模型（B 用）
        from aig_cell_models import build_models
        models_v = {}
        for t in TECHS:
            dfflib, abclib = parse_synth_ys(
                sorted(glob.glob(os.path.join(WORK, "*", "*__" + t, "synth.ys")))[0])
            out_v = os.path.join(MODELS_DIR, t + ".v")
            if not os.path.exists(out_v):
                mods, stats = build_models(abclib)
                with open(out_v, "w", encoding="utf-8", newline="\n") as f:
                    f.write("\n\n".join(mods) + "\n")
                print("models %s: %s -> %d modules" % (t, os.path.basename(abclib), len(mods)))
            models_v[t] = out_v

        jobs = []
        for td in tech_dirs():
            t = td.rsplit("__", 1)[1]
            for mode in ("A", "B"):
                if args.only and args.only != mode:
                    continue
                jobs.append((td, mode, models_v[t]))
        if args.limit:
            jobs = jobs[:args.limit]
        print("jobs:", len(jobs))
        ok = fail = skip = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for td, mode, status in ex.map(job_one, jobs):
                if status == "ok":
                    ok += 1
                elif status == "skip":
                    skip += 1
                else:
                    fail += 1
                    print("FAIL", mode, os.path.basename(td), status)
        print("done ok=%d skip=%d fail=%d" % (ok, skip, fail))
    if args.only in (None, "probe", "report"):
        report()


if __name__ == "__main__":
    main()

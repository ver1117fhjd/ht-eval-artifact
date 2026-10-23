# -*- coding: utf-8 -*-
"""r8_audit_demo.py —— R8 C13：可审计性工作流演示
================================================
把 C2 的 "auditability" 从性质论证变成演示：
给定金样参考，逐单元 fanin-cone(2) 算子预算差 → 全单元偏差排名 →
top-k 偏差区域与 log.txt TROJAN BODY 真实木马体对账（hit@k / MRR / 随机基线）。

评分完全基于算子预算（库无关口径），不看单元名差分；单元名只用于金样对齐
和事后评真。输出：experiments/c2/audit_demo.csv / audit_demo_report.txt
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "c2"
OUT.mkdir(parents=True, exist_ok=True)
TRIT_ROOT = ROOT / "data" / "Trust-Hub" / "chip-level-trojan"

INST_RE = re.compile(r"(\w+)\s+(\w+)\s*\((.*?)\)\s*;", re.S)
PORT_RE = re.compile(r"\.(\w+)\(([^)]*)\)")
OUT_PORTS = {"Q", "QN", "Z", "Y", "OUT", "SUM", "COUT", "O"}
OPS = ("and", "or", "not", "xor", "mux")


def cell_ops(celltype, n_in):
    t = celltype.lower()
    m = re.match(r"(and|or|nand|nor|xor|xnor|buf|i1|inv|hi1|lo1|mux|aoi|oai)", t)
    fam = m.group(1) if m else "unk"
    n = max(n_in - 1, 1) if n_in else 1
    ops = dict.fromkeys(OPS, 0)
    if fam == "and":
        ops["and"] = n
    elif fam == "nand":
        ops["and"] = n
        ops["not"] = 1
    elif fam == "or":
        ops["or"] = n
    elif fam == "nor":
        ops["or"] = n
        ops["not"] = 1
    elif fam == "xor":
        ops["xor"] = n
    elif fam == "xnor":
        ops["xor"] = n
        ops["not"] = 1
    elif fam in ("i1", "inv"):
        ops["not"] = 1
    elif fam == "mux":
        ops["mux"] = 1
    elif fam in ("buf", "hi1", "lo1"):
        pass
    else:
        return None
    return ops


def parse_netlist(path):
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")
    cells = {}
    for m in INST_RE.finditer(txt):
        ctype, cname, portstr = m.group(1), m.group(2), m.group(3)
        if ctype.lower() in ("module", "primitive") or "." not in portstr:
            continue
        ins, outs = [], []
        for pm in PORT_RE.finditer(portstr):
            port, net = pm.group(1).upper(), pm.group(2).strip()
            if port in OUT_PORTS:
                outs.append(net)
            else:
                ins.append(net)
        ops = cell_ops(ctype, len(ins))
        if ops is None:
            continue
        cells[cname] = {"type": ctype, "ins": ins, "outs": outs, "ops": ops}
    return cells


def fanin_cone_profiles(cells):
    """profile[c] = ops(c) + 深度 2 fanin 内唯一单元的 ops 之和。"""
    by_out = {}
    for c, d in cells.items():
        for o in d["outs"]:
            by_out[o] = c
    prof = {}
    for c, d in cells.items():
        acc = dict(d["ops"])
        seen = {c}
        frontier = [by_out.get(n) for n in d["ins"]]
        frontier = [x for x in frontier if x]
        for _ in range(2):
            nxt = []
            for f in frontier:
                if f in seen:
                    continue
                seen.add(f)
                for k in OPS:
                    acc[k] += cells[f]["ops"][k]
                nxt += [by_out.get(n) for n in cells[f]["ins"]]
            frontier = [x for x in nxt if x]
        prof[c] = acc
    return prof


def l1(a, b):
    return sum(abs(a[k] - b[k]) for k in OPS)


def parse_body(logpath):
    txt = Path(logpath).read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"TROJAN BODY:\s*\n(.*?)\n-{5,}", txt, re.S)
    if not m:
        return set()
    names = set()
    for line in m.group(1).splitlines():
        mm = re.match(r"\s*\w+\s+(\w+)\s*\(", line)
        if mm:
            names.add(mm.group(1))
    return names


def expected_random_hit(k, N, B):
    """随机排名 hit@k 期望（至少命中一个 body 单元）。"""
    if B <= 0 or N <= 0:
        return float("nan")
    miss = 1.0
    for i in range(k):
        miss *= max(N - B - i, 0) / max(N - i, 1)
    return 1.0 - miss


def main():
    # 找配对：original_designs/<base>.v vs <base>_TNNN/<base>_TNNN.v
    pairs = []
    for fam_dir in sorted(TRIT_ROOT.rglob("original_designs")):
        parent = fam_dir.parent
        golds = {p.stem: p for p in fam_dir.glob("*.v")}
        for vdir in sorted(parent.iterdir()):
            mm = re.match(r"(.+)_T(\d+)$", vdir.name)
            if not mm or not vdir.is_dir():
                continue
            base = mm.group(1)
            vfile = vdir / f"{vdir.name}.v"
            log = vdir / "log.txt"
            if base in golds and vfile.exists() and log.exists():
                pairs.append((vdir.name, golds[base], vfile, log))
    print(f"[pairs] found {len(pairs)} TRIT variant/golden pairs")

    # 控制运行时间：每个基准设计取前 6 个变体
    seen_base = {}
    sel = []
    for name, g, v, lg in pairs:
        b = name.rsplit("_T", 1)[0]
        if seen_base.get(b, 0) < 6:
            sel.append((name, g, v, lg))
            seen_base[b] = seen_base.get(b, 0) + 1
    print(f"[pairs] selected {len(sel)} pairs over {len(seen_base)} base designs")

    rows = []
    lines = ["R8 C13 审计工作流演示：cone(2) 算子预算差 → top-k vs 真实木马体",
             "评分=算子预算 L1 差（库无关），评真=TROJAN BODY 实例名；随机基线=期望 hit@k",
             ""]
    hits = {1: 0, 5: 0, 10: 0}
    rands = {1: 0.0, 5: 0.0, 10: 0.0}
    mrr_sum = 0.0
    n_ok = 0
    for name, gpath, vpath, logpath in sel:
        gcells = parse_netlist(gpath)
        vcells = parse_netlist(vpath)
        body = parse_body(logpath)
        if not vcells or not body:
            continue
        gprof = fanin_cone_profiles(gcells)
        vprof = fanin_cone_profiles(vcells)
        scores = []
        for c, d in vcells.items():
            gp = gprof.get(c, dict.fromkeys(OPS, 0))
            scores.append((l1(vprof[c], gp), c))
        scores.sort(key=lambda x: (-x[0], x[1]))
        N = len(scores)
        B = len(body & set(vcells))
        if B == 0:
            continue
        first_rank = next((i + 1 for i, (s, c) in enumerate(scores) if c in body), None)
        rec = {"pair": name, "n_cells": N, "n_body": B,
               "first_body_rank": first_rank, "mrr": 1.0 / first_rank if first_rank else 0.0}
        for k in (1, 5, 10):
            topk = {c for _, c in scores[:k]}
            hit = 1 if (topk & body) else 0
            rec[f"hit@{k}"] = hit
            hits[k] += hit
            rands[k] += expected_random_hit(k, N, B)
        rows.append(rec)
        mrr_sum += rec["mrr"]
        n_ok += 1
        lines.append(f"{name:18s} cells={N:5d} body={B:2d} first_body_rank={first_rank:4d} "
                     f"hit@1={rec['hit@1']} hit@5={rec['hit@5']} hit@10={rec['hit@10']} "
                     f"rand@5={expected_random_hit(5, N, B):.3f}")
    lines.append("")
    lines.append(f"==== 汇总（{n_ok} pairs）====")
    for k in (1, 5, 10):
        lines.append(f"hit@{k}: {hits[k]}/{n_ok} = {hits[k]/max(n_ok,1):.3f} "
                     f"（随机基线均值 {rands[k]/max(n_ok,1):.3f}）")
    lines.append(f"MRR = {mrr_sum/max(n_ok,1):.3f}")
    lines.append("")
    lines.append("口径注：评分不含'单元是否新增'这类名字级信息——新增单元的锥预算差"
                 "自然非零，触发锥内的共享单元因锥预算变化同样被顶上来；"
                 "单元名仅用于事后评真。演示设置=有金样参考的审计（lab/可信孪生场景）。")
    import pandas as pd
    pd.DataFrame(rows).to_csv(OUT / "audit_demo.csv", index=False)
    txt = "\n".join(lines)
    (OUT / "audit_demo_report.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    sys.exit(main())

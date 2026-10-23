# -*- coding: utf-8 -*-
"""
netlist_graph.py —— 门级网表 → 图数据（GNN 基线用）
========================================================

节点 = 标准单元，边 = 驱动单元 → 负载单元（双向存，GCN 用无向消息传递）。
节点特征（8 维，库无关，与 C2 算子归一化同一套查表 lib_ops.cell_ops）：
  [0..4] 单元算子画像 frac(and,or,xor,not,mux)（真值表模板匹配，分解可加）
  [5]    other/未知单元占比
  [6]    时序单元标志
  [7]    log1p(输入引脚数)

缓存：experiments/gnn/graphs/<graph_key>.npz  (x=[N,8] float32, edge_index=[2,E] int64)
graph_key = family__variant__source__tech|generic__module（与特征 CSV 四列精确对齐）。

用法：
  python src/netlist_graph.py            # 建全量缓存（generic + 三库）
  python src/netlist_graph.py --limit 20 # 冒烟
"""

import argparse
import json
import os
import sys

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)

from netlist_features import parse_module, is_seq          # noqa: E402
from lib_ops import cell_ops                               # noqa: E402
from c2_normalize import make_ops_map, WORK, LIBS          # noqa: E402

GRAPH_DIR = os.path.join(os.path.dirname(WORK), "gnn", "graphs")
OPS_VOCAB = ["and", "or", "xor", "not", "mux"]
NFEAT = 8


def graph_key(family, variant, source, tech, module):
    parts = [family, variant, source, tech or "generic", module]
    return "__".join(str(p).replace("/", "_").replace("\\", "_") for p in parts)


def node_feat(ops_map, ctype, conns, dirs):
    info = cell_ops(ops_map, ctype)
    vec = np.zeros(NFEAT, dtype=np.float32)
    if info is None:
        vec[5] = 1.0                       # 未知单元：全部计入 other
        return vec
    ops = info.get("ops", {}) or {}
    total = sum(ops.values())
    if total > 0:
        for i, op in enumerate(OPS_VOCAB):
            vec[i] = ops.get(op, 0) / total
        vec[5] = max(0.0, 1.0 - float(vec[:5].sum()))
    else:
        vec[5] = 1.0
    if info.get("seq") or is_seq(ctype):
        vec[6] = 1.0
    n_in = 0
    for pin, bits in conns.items():
        d = dirs.get(pin)
        if d is None:
            d = "output" if pin.upper().split("[")[0] in ("Y", "Q", "Z", "O", "OUT") else "input"
        if d == "input":
            n_in += len(bits)
    vec[7] = np.log1p(n_in)
    return vec


def build_graph(mod, ops_map):
    """yosys JSON module dict → (x, edge_index)。"""
    cells, drivers, sinks = parse_module(mod)
    n = len(cells)
    if n == 0:
        return None
    idx = {name: i for i, (name, _, _, _) in enumerate(cells)}
    x = np.stack([node_feat(ops_map, ctype, conns, dirs)
                  for _, ctype, conns, dirs in cells])
    edges = set()
    for b, users in sinks.items():
        if not isinstance(b, int):
            continue
        d = drivers.get(b)
        if d in (None, "PORT_IN", "PORT_OUT"):
            continue
        di = idx.get(d)
        if di is None:
            continue
        for c, _pin in users:
            ci = idx.get(c)
            if ci is None or ci == di:
                continue
            edges.add((di, ci))
    if not edges:
        ei = np.zeros((2, 0), dtype=np.int64)
    else:
        e = np.array(sorted(edges), dtype=np.int64).T
        ei = np.concatenate([e, e[::-1]], axis=1)   # 双向
    return x, ei


def build_one(family, variant, label, tech, ops_map, stats):
    outdir = os.path.join(WORK, family, variant + ("__" + label if label else "")
                          + (f"__" + tech if tech else ""))
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
    for mod_name in modules:
        m = netlist.get("modules", {}).get(mod_name)
        if m is None:
            continue
        key = graph_key(family, variant, label, tech, mod_name)
        cache = os.path.join(GRAPH_DIR, key + ".npz")
        if os.path.isfile(cache):
            n += 1
            continue
        g = build_graph(m, ops_map)
        if g is None:
            continue
        x, ei = g
        np.savez_compressed(cache, x=x, edge_index=ei,
                            key=np.array(key))
        stats["nodes"] += x.shape[0]
        stats["edges"] += ei.shape[1]
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--techs", default="all",
                    help='逗号分隔：generic,nangate45,sky130hd,asap7 或 all')
    args = ap.parse_args()
    os.makedirs(GRAPH_DIR, exist_ok=True)

    from batch_synth import find_designs                     # noqa: E402
    from batch_multilib import find_rtl_designs              # noqa: E402
    ROOT = r"./data\Trust-Hub\chip-level-trojan"

    want_generic = args.techs in ("all", "generic")
    techs = [] if args.techs == "generic" else (
        list(LIBS) if args.techs == "all" else args.techs.split(","))

    jobs = []
    if want_generic:
        for dirpath, _dirnames, _f in os.walk(ROOT):
            if os.path.basename(dirpath).lower() != "extracted":
                continue
            fam = os.path.relpath(dirpath, ROOT).split(os.sep)[0]
            variant = os.path.basename(os.path.dirname(dirpath))
            for label, _d in find_designs(dirpath):
                jobs.append((fam, variant, label, None))
        jobs.sort()
    mjobs = find_rtl_designs()
    for fam, variant, label, _d in mjobs:
        for tech in techs:
            jobs.append((fam, variant, label, tech))
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"建图任务: {len(jobs)} 个 (设计,工艺) 组合", flush=True)

    ops_maps = {None: make_ops_map(None)}
    for t in techs:
        ops_maps[t] = make_ops_map(t)

    stats = {"nodes": 0, "edges": 0}
    done = 0
    for i, (fam, variant, label, tech) in enumerate(jobs, 1):
        done += build_one(fam, variant, label, tech, ops_maps[tech], stats)
        if i % 50 == 0:
            print(f"  [{i}/{len(jobs)}] 累计节点 {stats['nodes']:,} "
                  f"边 {stats['edges']:,}", flush=True)
    print(f"完成: {done} 图, 累计 {stats['nodes']:,} 节点 / {stats['edges']:,} 边, "
          f"缓存目录 {GRAPH_DIR}", flush=True)


if __name__ == "__main__":
    main()

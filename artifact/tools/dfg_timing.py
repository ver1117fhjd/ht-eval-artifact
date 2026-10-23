# -*- coding: utf-8 -*-
"""实测 hw2vec 的 DFG 抽取耗时（"换 DFG 重跑"到底几天还是几小时的唯一判据）。
用法: timeout 240 python dfg_timing.py <design_dir>
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, r"D:/10.7/tools/hw2vec")
from hw2vec.hw2graph import HW2GRAPH
from hw2vec.config import Config

d = Path(sys.argv[1])
YAML = r"D:/10.7/tools/hw2vec/examples/example_gnn4tj.yaml"
t0 = time.time()
cfg = Config(["--yaml_path", YAML])
cfg.graph_type = sys.argv[2] if len(sys.argv) > 2 else "DFG"
g = HW2GRAPH(cfg)
p = g.preprocess(Path(d))
dt_pre = time.time() - t0
nx = g.process(p)
dt = time.time() - t0
nodes = nx.number_of_nodes() if hasattr(nx, "number_of_nodes") else len(nx)
edges = nx.number_of_edges() if hasattr(nx, "number_of_edges") else "?"
print(f"OK {d.name} graph={cfg.graph_type} preprocess={dt_pre:.1f}s total={dt:.1f}s nodes={nodes} edges={edges}")

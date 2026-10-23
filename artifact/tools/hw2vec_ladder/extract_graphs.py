# -*- coding: utf-8 -*-
"""extract_graphs.py —— HW2GRAPH 抽图（可断点续跑）
================================================
逐项目 DFG 抽取 → 单项目 pkl 缓存（graphs_cache/<project>.pkl），
已存在即跳过——pyverilog 龟速不怕中断。全部完成后合并
hw2vec_e1_graphs.pkl（hw2vec DataProcessor 格式）+ 对齐 manifest。

用法：python extract_graphs.py [--limit N]
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
HW2VEC = HERE.parents[1] / "hw2vec"
sys.path.insert(0, str(HW2VEC))

import pandas as pd  # noqa: E402

HW2VEC_ARGS = ["--yaml_path", str(HW2VEC / "examples" / "example_gnn4tj.yaml"),
               "--raw_dataset_path", str(HERE.parents[1] / "hw2vec_corpus_e1"),
               "--graph_type", "DFG",
               "--data_pkl_path", str(HERE.parents[1] / "hw2vec_e1_graphs.pkl")]
# ⚠️ DFG（数据流分析）在 AES 级设计上 >1h/设计，不可行——默认 AST 图型
# （hw2vec 一等模式；GNN4TJ 两种图型都支持），DFG 留小样做次要对照。
GT = None  # 由 main() 里 --graph_type 覆盖
from hw2vec.config import Config  # noqa: E402
from hw2vec.hw2graph import HW2GRAPH, DataProcessor  # noqa: E402

CACHE = HERE.parents[1] / "hw2vec_graphs_cache"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--graph_type", default="AST", choices=["AST", "DFG"])
    args = ap.parse_args()
    HW2VEC_ARGS[HW2VEC_ARGS.index("--graph_type") + 1] = args.graph_type
    suffix = args.graph_type.lower()
    HW2VEC_ARGS[HW2VEC_ARGS.index("--data_pkl_path") + 1] = str(
        HERE.parents[1] / f"hw2vec_e1_graphs_{suffix}.pkl")
    CACHE.mkdir(exist_ok=True)
    cache_dir = HERE.parents[1] / f"hw2vec_graphs_cache_{suffix}"
    cache_dir.mkdir(exist_ok=True)
    cfg = Config(HW2VEC_ARGS)
    hg = HW2GRAPH(cfg)
    projects = sorted(hg.find_hw_project_folders(), key=str)
    if args.limit:
        projects = projects[:args.limit]
    print(f"[extract] {len(projects)} projects, cache={CACHE}", flush=True)
    for p in projects:
        name = p.parent.name + "__" + p.name
        out = cache_dir / (name + ".pkl")
        if out.exists() and pickle.loads(out.read_bytes()) is not None:
            continue
        t0 = time.time()
        try:
            g = hg.code2graph(p)
            if g is None:
                print(f"[skip-null] {name}", flush=True)
                out.write_bytes(pickle.dumps(None))
                continue
            g.name = name  # 对齐键（覆盖 hw2vec 的 TjFree/TjIn 命名）
            out.write_bytes(pickle.dumps(g))
            print(f"[ok] {name} nodes={g.number_of_nodes()} "
                  f"{time.time()-t0:.0f}s", flush=True)
        except Exception as e:
            print(f"[FAIL] {name}: {type(e).__name__}: {e}", flush=True)
            out.write_bytes(pickle.dumps(None))

    # 合并成 hw2vec DataProcessor 缓存
    data_proc = DataProcessor(Config(HW2VEC_ARGS))
    names = []
    for p in projects:
        name = p.parent.name + "__" + p.name
        g = pickle.loads((cache_dir / (name + ".pkl")).read_bytes())
        if g is None:
            continue
        data_proc.process(g)
        names.append(name)
    data_proc.cache_graph_data(Path(HW2VEC_ARGS[-1]))
    pd.Series(names, name="project").to_csv(
        HERE.parents[1] / f"hw2vec_e1_order_{suffix}.csv", index=False)
    print(f"[merge] {len(names)} graphs cached, order saved", flush=True)


if __name__ == "__main__":
    main()

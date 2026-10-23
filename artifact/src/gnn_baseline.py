# -*- coding: utf-8 -*-
"""
gnn_baseline.py —— GNN 基线：图神经网络在无泄漏协议下的表现
==============================================================

与 c1_protocol.py 完全同一套数据划分/指标/模型无关协议，只是把
表格特征 RF/XGB/MLP 换成「网表图 + GCN」：
  节点=单元（8 维库无关算子画像，见 netlist_graph.py）
  边=驱动→负载（双向），读写 GCNConv 3 层 + 全局 mean|max 池化 + 线性头。

实验：
  E1 配对集：pooled / group-by-variant / group-by-base / LOFO
  E2 全量（不平衡）：pooled / group-by-base（pos_weight 平衡 BCE）
  E3 跨库：pooled / leave-one-library-out ×3

用法：
  python src/gnn_baseline.py [--epochs 20] [--budget 300000]
输出：
  experiments/gnn/gnn_results.csv / gnn_report.txt

依赖：torch 2.14.1+cpu, torch-geometric 2.8（2026-10-12 装）。
"""

import argparse
import os
import sys
import time

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)

# c1_protocol 模块级 argparse 只认自己的参数，import 前先把 argv 清干净
_argv = sys.argv[:]
sys.argv = sys.argv[:1]
import c1_protocol as cp                                      # noqa: E402
sys.argv = _argv

import torch                                                    # noqa: E402
import torch.nn.functional as F                                 # noqa: E402
from torch_geometric.data import Batch, Data                    # noqa: E402
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool  # noqa: E402

import os as _os                                                   # noqa: E402
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
SYNTH = _os.environ.get("SYNTH_DIR", _os.path.join(_R, "experiments", "synth_work"))
GRAPH_DIR = _os.environ.get("GRAPH_DIR", _os.path.join(_R, "experiments", "gnn", "graphs"))
OUT_DIR = _os.environ.get("GNN_OUT_DIR", _os.path.join(_R, "experiments", "gnn"))

SEED = 20261012


# ------------------------------------------------------------------ 图加载

def load_graphs_for(df, tech_col=None):
    """df 行 → 图列表（1:1 对齐）；缺失缓存的行剔除并返回掩码。"""
    from netlist_graph import graph_key
    keep, datas = [], []
    for i, row in df.iterrows():
        key = graph_key(row["family"], row["variant"], row["source"],
                        row[tech_col] if tech_col else None, row["module"])
        p = os.path.join(GRAPH_DIR, key + ".npz")
        if not os.path.isfile(p):
            continue
        z = np.load(p)
        x = torch.from_numpy(z["x"]).float()
        ei = torch.from_numpy(z["edge_index"]).long()
        if ei.numel() and int(ei.max()) >= x.shape[0]:
            continue
        datas.append(Data(x=x, edge_index=ei))
        keep.append(i)
    return datas, np.array(keep, dtype=np.int64)


# ------------------------------------------------------------------ 模型

class GCN(torch.nn.Module):
    def __init__(self, nfeat=8, hidden=64, n_layers=3, dropout=0.2):
        super().__init__()
        self.convs = torch.nn.ModuleList(
            [GCNConv(nfeat, hidden)] +
            [GCNConv(hidden, hidden) for _ in range(n_layers - 1)])
        self.dropout = dropout
        self.head = torch.nn.Linear(2 * hidden, 1)

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = F.relu(conv(x, ei))
            x = F.dropout(x, p=self.dropout, training=self.training)
        g = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], 1)
        return self.head(g).squeeze(-1)          # logits


# ------------------------------------------------------------------ 训练/预测

def pack_batches(datas, idxs, budget, shuffle, rng):
    order = list(idxs)
    if shuffle:
        rng.shuffle(order)
    batch, n = [], 0
    for i in order:
        g = datas[i]
        if batch and n + g.num_nodes > budget:
            yield batch
            batch, n = [], 0
        batch.append(i)
        n += g.num_nodes
    if batch:
        yield batch


def fit_predict(train_idx, test_idx, datas, y, epochs, budget, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = GCN()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    n_pos = float(y[train_idx].sum())
    n_neg = float(len(train_idx) - n_pos)
    pw = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32)
    crit = torch.nn.BCEWithLogitsLoss(pos_weight=pw)

    model.train()
    for ep in range(epochs):
        t0 = time.time()
        tot = 0.0
        for idxs in pack_batches(datas, train_idx, budget, True, rng):
            batch = Batch.from_data_list([datas[i] for i in idxs])
            logits = model(batch)
            tgt = torch.tensor(y[idxs], dtype=torch.float32)
            loss = crit(logits, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idxs)
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"      ep{ep:02d} loss={tot/len(train_idx):.4f} "
                  f"({time.time()-t0:.1f}s)", flush=True)

    model.eval()
    prob = np.zeros(len(test_idx))
    with torch.no_grad():
        pos = 0
        for idxs in pack_batches(datas, test_idx, budget, False, rng):
            batch = Batch.from_data_list([datas[i] for i in idxs])
            logits = model(batch)
            prob[pos:pos + len(idxs)] = torch.sigmoid(logits).numpy()
            pos += len(idxs)
    return prob


def run_protocol_cv(name, datas, y, groups, splitter, use_groups, epochs, budget):
    """5 折或 LeaveOneGroupOut 语义的 CV，返回 pooled 预测概率。"""
    from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedGroupKFold
    prob = np.zeros(len(y))
    if isinstance(splitter, LeaveOneGroupOut):
        splits = splitter.split(np.zeros(len(y)), y, groups)
    elif use_groups:
        splits = splitter.split(np.zeros(len(y)), y, groups)
    else:
        splits = splitter.split(np.zeros(len(y)), y)
    for k, (tr, te) in enumerate(splits):
        print(f"    fold{k}: train={len(tr)} test={len(te)}", flush=True)
        prob[te] = fit_predict(tr, te, datas, y, epochs, budget, SEED + k)
    return prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--budget", type=int, default=300000,
                    help="每个 batch 的节点预算")
    ap.add_argument("--exps", default="E1,E2,E3")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    # 论文数字用 v2（算子归一化）特征表定义的数据集行
    cp.ALL_CSV = os.path.join(SYNTH, "features_all_v2.csv")
    cp.MULTILIB_CSV = os.path.join(SYNTH, "features_multilib_v2.csv")

    from sklearn.model_selection import (LeaveOneGroupOut, StratifiedGroupKFold,
                                        StratifiedKFold)
    rows = []
    probs = {}

    # ---------------- E1 ----------------
    if "E1" in args.exps:
        df = cp.load_pairs().reset_index(drop=True)
        datas, keep = load_graphs_for(df)
        df = df.iloc[keep].reset_index(drop=True)
        y = df["label"].values
        print(f"[E1] n={len(df)} pos={y.sum()} graphs={len(datas)}", flush=True)
        specs = [
            ("pooled-random-5fold", StratifiedKFold(5, shuffle=True, random_state=SEED), False),
            ("group-by-variant-5fold", StratifiedGroupKFold(5, shuffle=True, random_state=SEED), True),
            ("group-by-base-5fold", StratifiedGroupKFold(5, shuffle=True, random_state=SEED), True),
            ("leave-one-family-out", LeaveOneGroupOut(), True),
        ]
        gvar = df.apply(cp.variant_pair, axis=1).values
        gbase = df["group"].values
        gfam = df["family"].values
        for name, spl, ug in specs:
            groups = gfam if name == "leave-one-family-out" else (
                gvar if name == "group-by-variant-5fold" else gbase)
            print(f"  [E1] {name}", flush=True)
            prob = run_protocol_cv(name, datas, y, groups, spl, ug,
                                   args.epochs, args.budget)
            m = cp.metrics(y, prob)
            ci = cp.boot_ci(y, prob, gbase)   # CI 聚类恒为 base design（R4 口径）
            rows.append(dict(experiment="E1-pairs", protocol=name, model="GCN",
                             n=len(y), n_pos=int(y.sum()),
                             AP=m["AP"], AP_lo=ci["AP"][1], AP_hi=ci["AP"][2],
                             ROC_AUC=m["ROC_AUC"], F1=m["F1"], Sens=m["Sens"],
                             Spec=m["Spec"], BalAcc=m["BalAcc"], Gmean=m["Gmean"]))

    # ---------------- E2 ----------------
    if "E2" in args.exps:
        df = cp.load_full().reset_index(drop=True)
        datas, keep = load_graphs_for(df)
        df = df.iloc[keep].reset_index(drop=True)
        y = df["label"].values
        print(f"[E2] n={len(df)} pos={y.sum()} graphs={len(datas)}", flush=True)
        specs = [
            ("pooled-random-5fold", StratifiedKFold(5, shuffle=True, random_state=SEED), False),
            ("group-by-variant-5fold", StratifiedGroupKFold(5, shuffle=True, random_state=SEED), True),
            ("group-by-base-5fold", StratifiedGroupKFold(5, shuffle=True, random_state=SEED), True),
        ]
        gvar = df.apply(cp.variant_pair, axis=1).values
        gbase = df["group"].values
        for name, spl, ug in specs:
            groups = gvar if name == "group-by-variant-5fold" else gbase
            print(f"  [E2] {name}", flush=True)
            prob = run_protocol_cv(name, datas, y, groups, spl, ug,
                                   args.epochs, args.budget)
            m = cp.metrics(y, prob)
            ci = cp.boot_ci(y, prob, gbase)   # CI 聚类恒为 base design（R4 口径）
            rows.append(dict(experiment="E2-full", protocol=name, model="GCN",
                             n=len(y), n_pos=int(y.sum()),
                             AP=m["AP"], AP_lo=ci["AP"][1], AP_hi=ci["AP"][2],
                             ROC_AUC=m["ROC_AUC"], F1=m["F1"], Sens=m["Sens"],
                             Spec=m["Spec"], BalAcc=m["BalAcc"], Gmean=m["Gmean"]))

    # ---------------- E3 ----------------
    if "E3" in args.exps:
        df = cp.load_multilib().reset_index(drop=True)
        datas, keep = load_graphs_for(df, tech_col="tech")
        df = df.iloc[keep].reset_index(drop=True)
        y = df["label"].values
        techs = df["tech"].values
        gbase = df["group"].values   # CI 聚类恒为 base design（R4 口径）
        print(f"[E3] n={len(df)} pos={y.sum()} graphs={len(datas)}", flush=True)
        for held in sorted(set(techs)):
            tr = np.where(techs != held)[0]
            te = np.where(techs == held)[0]
            print(f"  [E3] LOLO({held}) train={len(tr)} test={len(te)}", flush=True)
            prob = fit_predict(tr, te, datas, y, args.epochs, args.budget, SEED)
            m = cp.metrics(y[te], prob)
            ci = cp.boot_ci(y[te], prob, gbase[te])
            rows.append(dict(experiment="E3-multilib",
                             protocol=f"leave-one-library-out({held})", model="GCN",
                             n=len(te), n_pos=int(y[te].sum()),
                             AP=m["AP"], AP_lo=ci["AP"][1], AP_hi=ci["AP"][2],
                             ROC_AUC=m["ROC_AUC"], F1=m["F1"], Sens=m["Sens"],
                             Spec=m["Spec"], BalAcc=m["BalAcc"], Gmean=m["Gmean"]))
        print("  [E3] pooled", flush=True)
        prob = run_protocol_cv("pooled-random-5fold", datas, y, None,
                               StratifiedKFold(5, shuffle=True, random_state=SEED),
                               False, args.epochs, args.budget)
        m = cp.metrics(y, prob)
        ci = cp.boot_ci(y, prob, gbase)
        rows.append(dict(experiment="E3-multilib", protocol="pooled-random-5fold",
                         model="GCN", n=len(y), n_pos=int(y.sum()),
                         AP=m["AP"], AP_lo=ci["AP"][1], AP_hi=ci["AP"][2],
                         ROC_AUC=m["ROC_AUC"], F1=m["F1"], Sens=m["Sens"],
                         Spec=m["Spec"], BalAcc=m["BalAcc"], Gmean=m["Gmean"]))

    # ---------------- 输出 ----------------
    import pandas as pd
    out = pd.DataFrame(rows)
    cols = ["experiment", "protocol", "model", "n", "n_pos",
            "AP", "AP_lo", "AP_hi", "ROC_AUC", "F1", "Sens", "Spec", "BalAcc", "Gmean"]
    out = out[cols]
    out.to_csv(os.path.join(OUT_DIR, "gnn_results.csv"), index=False)

    lines = ["GCN 基线 —— 无泄漏协议下的图神经网络（8 维算子画像节点特征）",
             f"seed={SEED}, epochs={args.epochs}, node_budget={args.budget}, "
             f"GCNConv x3 hidden=64, lr=2e-3", ""]
    for exp in out["experiment"].unique():
        lines.append(f"==== {exp} ====")
        sub = out[out["experiment"] == exp]
        for _, r in sub.iterrows():
            lines.append(
                f"{r['protocol']:32s} GCN  n={r['n']:4d} pos={r['n_pos']:4d} "
                f"AP={r['AP']:.3f} [{r['AP_lo']:.3f},{r['AP_hi']:.3f}] "
                f"AUC={r['ROC_AUC']:.3f} F1={r['F1']:.3f} "
                f"Sens={r['Sens']:.3f} Spec={r['Spec']:.3f} "
                f"Bal={r['BalAcc']:.3f} Gm={r['Gmean']:.3f}")
        lines.append("")
    txt = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "gnn_report.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt, flush=True)


if __name__ == "__main__":
    sys.exit(main())

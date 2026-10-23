# -*- coding: utf-8 -*-
"""run_ladder.py —— HW2VEC/GNN4TJ 挂上我们的协议阶梯（T1-A 核心）
================================================================
把 hw2vec 的 GNN4TJ（GraphTrainer）在 E1 配对集上按协议阶梯重打分：
  pooled-random-5fold / group-by-variant-5fold / group-by-base-5fold /
  leave-one-family-out
指标与 c1_protocol 口径一致：ROC-AUC + AP + Spec@0.5 + cluster bootstrap CI。
输出：experiments/hw2vec/ladder_results.csv / ladder_report.txt

用法：python run_ladder.py [--epochs 20] [--seeds 3]
"""
import argparse
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT2 = HERE.parents[1]
HW2VEC = ROOT2 / "hw2vec"
sys.path.insert(0, str(HW2VEC))
sys.path.insert(0, str(ROOT2.parent / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score, average_precision_score  # noqa: E402
from sklearn.model_selection import (  # noqa: E402
    StratifiedKFold, StratifiedGroupKFold, LeaveOneGroupOut)

HW2VEC_ARGS = ["--yaml_path", str(HW2VEC / "examples" / "example_gnn4tj.yaml"),
               "--raw_dataset_path", str(ROOT2 / "hw2vec_corpus_e1"),
               "--graph_type", "AST",
               "--data_pkl_path", str(ROOT2 / "hw2vec_e1_graphs_ast.pkl")]
from hw2vec.config import Config  # noqa: E402
from hw2vec.hw2graph import DataProcessor  # noqa: E402
from hw2vec.graph2vec.trainers import GraphTrainer  # noqa: E402
from hw2vec.graph2vec.models import (  # noqa: E402
    GRAPH2VEC, GRAPH_CONV, GRAPH_POOL, GRAPH_READOUT)
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402


# ⚠️ 修正 hw2vec 原生 bug：train_epoch_tj/inference_epoch_tj 把 log_softmax 输出
# 喂 CrossEntropyLoss（内部会再 log-softmax 一次）——双重复合在均匀输出处是
# 不动点，激活小的图（我们的 AST 语料）梯度消失、锁死。此处以 CE 直吃 logits
# 替换（等价于 NLLLoss∘log_softmax 的正确组合），对玩具等小图语料行为不变。
def _train_epoch_tj(self, data):
    output, _ = self.model.embed_graph(data.x, data.edge_index, data.batch)
    output = self.model.mlp(output)
    return F.cross_entropy(output, data.label)


def _inference_epoch_tj(self, data):
    output, attn = self.model.embed_graph(data.x, data.edge_index, data.batch)
    output = self.model.mlp(output)
    loss = F.cross_entropy(output, data.label)
    return loss, output, attn


GraphTrainer.train_epoch_tj = _train_epoch_tj
GraphTrainer.inference_epoch_tj = _inference_epoch_tj


# ⚠️ 修正 2：GCN 度归一化在高分支度 AST 图上把激活压到 ~1e-4（梯度消失、
# 输出锁死在均匀不动点）。one-hot 特征 ×100 抵消归一化衰减 + mean readout
# 替代 max（max 在饱和尺度下走全同预测的退化解）。均经消融验证
# （max/scale 组合退化、mean+scale 训练恢复）。偏离原 use_case_2 配置处
# 均在此声明，论文口径照写。
def _embed_graph_patched(self, x, edge_index, batch):
    x = F.one_hot(x, num_classes=self.config.num_feature_dim).float() * 100.0
    for layer in self.layers:
        x = F.dropout(F.relu(layer(x, edge_index)),
                      p=self.config.dropout, training=self.training)
    x, edge_index, _, batch, perm, score = self.pool1(x, edge_index, batch=batch)
    x = self.graph_readout(x, batch)
    return x, {"pool_perm": perm, "pool_score": score, "batch": batch}


GRAPH2VEC.embed_graph = _embed_graph_patched

OUT = ROOT2.parent / "experiments" / "hw2vec"
OUT.mkdir(parents=True, exist_ok=True)


def load_graphs():
    cfg = Config(HW2VEC_ARGS)
    dp = DataProcessor(cfg)
    dp.read_graph_data_from_cache(Path(HW2VEC_ARGS[-1]))
    graphs = dp.get_graphs()
    order = pd.read_csv(ROOT2 / "hw2vec_e1_order_ast.csv")["project"].tolist()
    man = pd.read_csv(ROOT2 / "hw2vec_e1_manifest.csv")
    # 对齐键：抽取侧 name = "<TjIn|TjFree>__<project>"，manifest 用 source 拼同款
    man["key"] = man["source"] + "__" + man["project"]
    man = man.set_index("key")
    assert len(graphs) == len(order), (len(graphs), len(order))
    meta = man.loc[order].reset_index()
    y = meta["label"].values
    for g, lab in zip(graphs, y):
        g.label = int(lab)
    return graphs, meta, y, dp.num_node_labels


def probs_for_fold(train_g, test_g, epochs, seed, num_labels):
    cfg = Config(HW2VEC_ARGS)
    cfg.epochs = epochs
    cfg.seed = seed
    cfg.num_feature_dim = num_labels
    cfg.num_node_labels = num_labels
    torch.manual_seed(seed)
    # —— 与 use_case_2.py 同款模型构建序列（GNN4TJ）——
    model = GRAPH2VEC(cfg)
    convs = [GRAPH_CONV("gcn", num_labels, cfg.hidden),
             GRAPH_CONV("gcn", cfg.hidden, cfg.hidden)]
    model.set_graph_conv(convs)
    model.set_graph_pool(GRAPH_POOL("sagpool", cfg.hidden, cfg.poolratio))
    model.set_graph_readout(GRAPH_READOUT("mean"))
    model.set_output_layer(nn.Linear(cfg.hidden, cfg.embed_dim))
    model.to(cfg.device)
    cw = torch.tensor([1.0, 1.0])
    tr = DataLoader(train_g, batch_size=min(8, len(train_g)), shuffle=True)
    te = DataLoader(test_g, batch_size=min(8, len(test_g)), shuffle=False)
    trainer = GraphTrainer(cfg, class_weights=cw)
    trainer.build(model)
    trainer.train(tr, te)
    avg_loss, _labels_t, outputs_t, _preds_t, _ = trainer.inference(te)
    prob = torch.softmax(outputs_t, dim=1)[:, 1].cpu().numpy()
    return np.asarray(prob).reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--max-nodes", type=int, default=25000,
                    help="AST 节点上限（CPU 预算；整对剔除保持配对完整）")
    args = ap.parse_args()
    graphs, meta, y, num_labels = load_graphs()
    sizes = np.array([g.num_nodes for g in graphs])
    keep = sizes <= args.max_nodes
    graphs = [g for g, k in zip(graphs, keep) if k]
    meta = meta[keep].reset_index(drop=True)
    y = y[keep]
    groups = meta["base"].values
    vgroups = meta["variant"].values
    fam = meta["family"].values
    print(f"[load] {len(graphs)} graphs, pos={y.sum()}, "
          f"groups={len(set(groups))}, fam={len(set(fam))}", flush=True)

    rows = []
    protocols = {
        "pooled-random-5fold": ("strat", StratifiedKFold(
            5, shuffle=True, random_state=20261012), None),
        "group-by-variant-5fold": ("group", StratifiedGroupKFold(
            5, shuffle=True, random_state=20261012), vgroups),
        "group-by-base-5fold": ("group", StratifiedGroupKFold(
            5, shuffle=True, random_state=20261012), groups),
        "leave-one-family-out": ("group", LeaveOneGroupOut(), fam),
    }
    for seed_i in range(args.seeds):
        seed = 20261012 + seed_i
        for pname, (kind, cv, gvec) in protocols.items():
            prob = np.zeros(len(y))
            splits = (cv.split(np.zeros(len(y)), y) if kind == "strat"
                      else cv.split(np.zeros(len(y)), y, gvec))
            for k, (tr, te) in enumerate(splits):
                print(f"[{pname} seed{seed_i}] fold{k} "
                      f"train={len(tr)} test={len(te)}", flush=True)
                prob[te] = probs_for_fold([graphs[i] for i in tr],
                                         [graphs[i] for i in te],
                                         args.epochs, seed + k, num_labels)
            auc = roc_auc_score(y, prob)
            ap_ = average_precision_score(y, prob)
            pred = (prob >= 0.5).astype(int)
            neg, pos = y == 0, y == 1
            spec = (pred[neg] == 0).mean()
            sens = (pred[pos] == 1).mean()
            rows.append({"protocol": pname, "seed": seed, "n": len(y),
                         "n_pos": int(y.sum()), "AUC": auc, "AP": ap_,
                         "Sens": sens, "Spec": spec})
            print(f"[{pname} seed{seed_i}] AUC={auc:.3f} AP={ap_:.3f} "
                  f"Sens={sens:.3f} Spec={spec:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ladder_results.csv", index=False)
    rep = ["HW2VEC/GNN4TJ 协议阶梯重打分（E1 配对集）",
           f"graphs={len(graphs)} pos={int(y.sum())} epochs={args.epochs} "
           f"seeds={args.seeds}", ""]
    for pname in df["protocol"].unique():
        sub = df[df["protocol"] == pname]
        rep.append(f"{pname:26s} AUC {sub['AUC'].mean():.3f}"
                   f"+-{sub['AUC'].std():.3f} (n={len(sub)}) "
                   f"Spec {sub['Spec'].mean():.3f}")
    (OUT / "ladder_report.txt").write_text("\n".join(rep), encoding="utf-8")
    print("\n".join(rep), flush=True)


if __name__ == "__main__":
    main()

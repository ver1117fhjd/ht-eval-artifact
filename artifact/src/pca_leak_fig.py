# -*- coding: utf-8 -*-
"""
pca_leak_fig.py —— P2：PCA 泄漏可视化（sibling 聚簇跨 split）数据 + 草稿图
===========================================================================

评审 P2 项："t-SNE 泄漏可视化"裁决为改用 PCA（确定性）。
产出：
  experiments/c2/pca_e1.csv    每行 (family, variant, source, label, base, PC1, PC2)
  experiments/c2/pca_stats.txt 聚簇量化证据（NN same-base 率 / 簇内外距离比 / 方差解释）
  fig/fig4_pca_draft.png       草稿版（参照用，最终她手画）

量化口径：
  - 特征标准化（StandardScaler）后 PCA 到 2 维；
  - NN same-base 率：每行在全特征空间找最近邻，统计同基准设计的比例
    （随机基线 = sum_g (n_g/N)^2，即随机两行同组概率）；
  - 簇内外距离比：同基准行对的平均欧氏距离 vs 不同基准行对。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c1_protocol as c1  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "c2"
FIG = ROOT / "paper" / "host2027" / "fig"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-csv",
                    default=str(ROOT / "experiments/synth_work/features_all_v2.csv"))
    args = ap.parse_args()
    c1.ALL_CSV = args.all_csv
    OUT.mkdir(parents=True, exist_ok=True)

    df = c1.load_pairs().reset_index(drop=True)
    X = df[c1.feature_cols(df)].values.astype(float)
    y = df["label"].values
    base = df["group"].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    Z = pca.fit_transform(Xs)

    out = pd.DataFrame({
        "family": df["family"].values,
        "variant": df["variant"].values,
        "source": df["source"].values,
        "label": y,
        "base": base,
        "PC1": Z[:, 0], "PC2": Z[:, 1],
    })
    csv_path = OUT / "pca_e1.csv"
    out.to_csv(csv_path, index=False, lineterminator="\n")

    # ---- 量化证据 ----
    # NN same-base 率（全特征空间，标准化后欧氏）
    from scipy.spatial.distance import cdist
    D = cdist(Xs, Xs)
    np.fill_diagonal(D, np.inf)
    nn = D.argmin(axis=1)
    same = (base[nn] == base).mean()
    sizes = pd.Series(base).value_counts().values
    N = len(base)
    chance = float(((sizes / N) ** 2).sum())

    # 簇内 / 簇外平均距离（上三角）
    iu = np.triu_indices(N, k=1)
    pair_same = base[iu[0]] == base[iu[1]]
    d = D[iu]
    intra = d[pair_same].mean()
    inter = d[~pair_same].mean()

    # 金样→木马配对距离（同 base 的 gold vs trojan）中位数 vs 随机对
    rng = np.random.default_rng(c1.SEED)
    gold_idx = np.where(y == 0)[0]
    pair_d = []
    for g in gold_idx:
        sib = np.where((base == base[g]) & (y == 1))[0]
        for s in sib:
            pair_d.append(D[g, s])
    pair_d = np.array(pair_d)
    rand_d = d[rng.choice(len(d), size=len(pair_d), replace=False)]

    lines = [
        f"E1 pairs: N={N} (gold={int((y==0).sum())}, trojan={int((y==1).sum())}), "
        f"base designs={len(sizes)}",
        f"PCA explained variance: PC1={pca.explained_variance_ratio_[0]:.3f}, "
        f"PC2={pca.explained_variance_ratio_[1]:.3f}, "
        f"cum={pca.explained_variance_ratio_.sum():.3f}",
        f"Nearest-neighbor same-base rate (full feature space): "
        f"{same:.3f}  (random chance={chance:.3f}, lift={same/chance:.1f}x)",
        f"Mean distance: intra-base={intra:.2f} vs inter-base={inter:.2f} "
        f"(ratio={intra/inter:.3f})",
        f"Gold->trojan sibling pairs (n={len(pair_d)}): median dist={np.median(pair_d):.2f} "
        f"vs random pairs median={np.median(rand_d):.2f}",
    ]
    txt = "\n".join(lines) + "\n"
    (OUT / "pca_stats.txt").write_text(txt, encoding="utf-8")
    print(txt)

    # ---- 草稿图（参照用）----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 8, "font.family": "serif"})
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.6), dpi=150)
        # (a) 按标签着色：看不出结构
        for lab, mk, c in ((0, "o", "#1f77b4"), (1, "^", "#d62728")):
            m = y == lab
            axes[0].scatter(Z[m, 0], Z[m, 1], s=14, marker=mk, c=c,
                            label="golden" if lab == 0 else "trojan", alpha=.8)
        axes[0].set_title("(a) colored by label")
        axes[0].legend(frameon=False, fontsize=7)
        # (b) 按基准设计着色：兄弟聚簇
        bases = sorted(set(base))
        cmap = plt.get_cmap("tab20")
        for i, b in enumerate(bases):
            m = base == b
            axes[1].scatter(Z[m, 0], Z[m, 1], s=14, c=[cmap(i % 20)], label=b)
        axes[1].set_title("(b) colored by base design")
        for ax in axes:
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        fig.tight_layout()
        FIG.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG / "fig4_pca_draft.png")
        print(f"draft fig: {FIG / 'fig4_pca_draft.png'}")
    except Exception as e:  # 草稿图失败不挡数据交付
        print(f"draft fig skipped: {e}")

    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()

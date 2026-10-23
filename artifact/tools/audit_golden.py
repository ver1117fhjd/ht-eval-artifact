#!/usr/bin/env python3
"""audit_golden.py —— 金样类退化审计（把论文 §III-B / §V-B 的检查做成可跑脚本）。

对每一组（base design）：统计金样行数、特征向量唯一数、最深/最常见重复；
若某组金样行 > 1 但唯一向量 = 0/1，或各组金样指向同一个 module，
则判为「金样类被解析错」。

用法：python3 audit_golden.py results/features_all_v2.csv
退出码：0 = 通过；1 = 发现退化
"""
import sys
import pandas as pd

META = {"family", "variant", "source", "module", "label", "tech", "group"}


def main(csv_path):
    df = pd.read_csv(csv_path)
    feat = [c for c in df.columns if c not in META]
    gold = df[df.get("label", 0) == 0] if "label" in df.columns else df[
        df["source"].astype(str).str.contains("TjFree|original_designs")]
    bad = 0
    print(f"file={csv_path}  rows={len(df)}  golden rows={len(gold)}")
    mod = gold["module"].astype(str).value_counts()
    if len(mod) == 1 and len(gold) > 1:
        print(f"  !! every golden row resolves to module {mod.index[0]!r} "
              f"({len(gold)} rows) — top-level selection failed")
        bad += 1
    for key, g in gold.groupby(gold["variant"].astype(str)):
        u = len(g[feat].round(6).drop_duplicates())
        flag = "  <-- degenerate" if (len(g) > 1 and u <= 1) else ""
        print(f"  {key:22s} rows={len(g):4d} unique_vectors={u:4d}{flag}")
        if len(g) > 1 and u <= 1:
            bad += 1
    total_u = len(gold[feat].round(6).drop_duplicates())
    print(f"total golden rows={len(gold)} -> {total_u} unique feature vectors")
    print("PASS" if bad == 0 else f"FAIL ({bad} degenerate golden groups)")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1
                  else "results/features_all_v2.csv"))

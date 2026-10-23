# -*- coding: utf-8 -*-
"""prep_corpus.py —— E1 配对集 → hw2vec 目录布局（T1-A 第一步）
================================================================
从 features_all_v2.csv 取 E1 标注对（source ∈ TjIn/TjFree），定位 Trust-Hub
源目录（.../<variant>/extracted/<variant>/src/{TjIn,TjFree}/*.v），
清洗后复制到 tools/hw2vec_corpus_e1/{TjIn,TjFree}/<family>__<variant>/。

清洗：① 空端口补位（pyverilog 不吃位置空端口）；② 测试台过滤（文件名双启发）。
输出：tools/hw2vec_corpus_e1/ + manifest.csv
"""
import re
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TRUST = ROOT / "data" / "Trust-Hub"
OUT = ROOT / "tools" / "hw2vec_corpus_e1"
FEATURES = ROOT / "experiments" / "synth_work" / "features_all_v2.csv"


def fix_empty_ports(txt):
    n = [0]

    def rep1(m):
        n[0] += 1
        return m.group(1) + "hw2vec_nc_%d " % n[0] + m.group(2)
    txt = re.sub(r"(\d+'[bBoOhHdD])\s+([0-9a-fA-FxXzZ_]+)", r"\1\2", txt)
    txt = re.sub(r"(\(\s*)(,\s*)", rep1, txt)
    txt = re.sub(r"(,\s*)(,\s*)", rep1, txt)
    txt = re.sub(r"(,\s*)(\)\s*;)", rep1, txt)
    return txt, n[0]


def inline_includes(txt, d, depth=0):
    """内联 `include（ivlpp 按 CWD 解析、不认文件相对路径；顺带模糊匹配
    wbconmaxdefines.v ↔ wb_conmax_defines.v 这类下划线差异）。"""
    if depth > 4:
        return txt

    def rep(m):
        name = m.group(1)
        cands = [d / name, d / Path(name).name]
        if not any(c.exists() for c in cands):
            norm = re.sub(r"[^a-z0-9]", "", Path(name).name.lower())
            for f in d.iterdir():
                if f.is_file() and re.sub(r"[^a-z0-9]", "", f.name.lower()) == norm:
                    cands.append(f)
                    break
        for c in cands:
            if c.exists():
                sub = c.read_text(encoding="utf-8", errors="ignore")
                return inline_includes(sub, d, depth + 1)
        return f"// hw2vec-prep: include {name} NOT FOUND"
    return re.sub(r'`include\s+"([^"]+)"', rep, txt)


def is_testbench(path):
    stem = path.stem.lower()
    return (stem.startswith("test") or "_tb" in stem or stem.startswith("tb")
            or "testbench" in stem)


def find_src_dir(variant, label):
    """定位 .../<variant>/extracted/<variant>/src/<label>（容错大小写/层级）。"""
    hits = list(TRUST.rglob(f"*/{variant}/extracted/{variant}/src/{label}"))
    if not hits:
        hits = [p for p in TRUST.rglob(label)
                if variant in str(p.parent.parent)]
    return hits[0] if hits else None


def main():
    df = pd.read_csv(FEATURES)
    src = df["source"].replace({"TnIn": "TjIn"})
    rows = df[src.isin(["TjIn", "TjFree"])].copy()
    rows["src"] = rows["source"].replace({"TnIn": "TjIn"})
    print(f"[prep] E1 labeled rows: {len(rows)}")

    if OUT.exists():
        shutil.rmtree(OUT)
    manifest = []
    miss = 0
    for _, r in rows.iterrows():
        variant, label = str(r["variant"]), str(r["src"])
        d = find_src_dir(variant, label)
        if d is None:
            print(f"[MISS] {r['family']} {variant} {label}")
            miss += 1
            continue
        # 工具边界：hw2vec/pyverilog 只吃 Verilog——含 .vhd 的设计（MC8051）整体排除，
        # 严禁"部分 .v 抽图"混入（会造出残缺设计图，口径不干净）
        if list(d.glob("*.vhd")):
            print(f"[VHDL-skip] {r['family']} {variant} {label}")
            miss += 1
            continue
        proj = f"{r['family']}__{variant}"
        dst = OUT / label / proj
        dst.mkdir(parents=True, exist_ok=True)
        nfiles = 0
        includes = set()
        for v in sorted(d.glob("*.v")):
            if is_testbench(v):
                continue
            txt = v.read_text(encoding="utf-8", errors="ignore")
            txt, k = fix_empty_ports(txt)
            txt = inline_includes(txt, d)
            (dst / v.name).write_text(txt, encoding="utf-8")
            nfiles += 1
            includes |= set(re.findall(r'`include\s+"([^"]+)"', txt))
        # include 目标文件一并复制（inc.h / wbconmaxdefines.v 等"作者机器私货"）
        for inc in includes:
            src_inc = d / inc
            if not src_inc.exists():
                src_inc = d / Path(inc).name
            if src_inc.exists():
                shutil.copy2(src_inc, dst / src_inc.name)
            else:
                print(f"[inc-miss] {proj} {inc}")
        manifest.append({"project": proj, "family": r["family"],
                         "variant": variant, "source": label,
                         "label": 1 if label == "TjIn" else 0,
                         "base": re.sub(r"[-_]T\d+$", "", variant),
                         "n_files": nfiles})
    mdf = pd.DataFrame(manifest)
    mdf.to_csv(OUT.parent / "hw2vec_e1_manifest.csv", index=False)
    print(f"[prep] projects={len(mdf)} missing={miss}")
    print(mdf.groupby("family").size().to_string())


if __name__ == "__main__":
    main()

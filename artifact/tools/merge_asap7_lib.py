#!/usr/bin/env python3
"""merge_asap7_lib.py v2 —— 把 asap7 各类单元库真正合并成单 library 的 .lib。

v1 教训：只抠 cell 块会丢文件尾部的 lu_table_template 等定义（cell 引用悬空，
ABC 拒读）。v2 以第一个文件全文为骨架，其余文件的"未重复顶层组"整体补到末尾。
输出：asap7_RVT_TT_merged.lib（全量，供 dfflibmap）+ asap7_RVT_TT_comb.lib（纯组合，供 abc）。
"""
import re

BASE = r"./libs\liberty\asap7"
ALL_FILES = [
    f"{BASE}/AO.lib",
    f"{BASE}/INVBUF.lib",
    f"{BASE}/OA.lib",
    f"{BASE}/SEQ.lib",
    f"{BASE}/SIMPLE.lib",
    f"{BASE}/asap7sc7p5t_DFFHQNH2V2X_RVT_TT_nldm_FAKE.lib",
    f"{BASE}/asap7sc7p5t_DFFHQNV2X_RVT_TT_nldm_FAKE.lib",
    f"{BASE}/asap7sc7p5t_DFFHQNV4X_RVT_TT_nldm_FAKE.lib",
]
COMB_FILES = [f"{BASE}/{c}.lib" for c in ("AO", "INVBUF", "OA", "SIMPLE")]

GROUP_RE = re.compile(
    r"\b(cell|lu_table_template|power_lut_template|operating_conditions|"
    r"wire_load|type|base_curves)\s*\(\s*([^)]*?)\s*\)\s*\{")


def top_groups(src):
    """抽取所有顶层组: [(kind, name, text)]。"""
    out = []
    for m in GROUP_RE.finditer(src):
        i = src.index("{", m.end() - 1)
        depth, j = 0, i
        while j < len(src):
            if src[j] == "{":
                depth += 1
            elif src[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        out.append((m.group(1), m.group(2).strip(), src[m.start():j + 1]))
    return out


def merge(files, out_path):
    first = open(files[0], encoding="utf-8", errors="replace").read()
    base = first[: first.rfind("}")]          # 骨架 = 第一个文件去掉最后的收尾 }
    seen = {(k, n) for k, n, _t in top_groups(first)}
    extra = []
    for f in files[1:]:
        for kind, name, text in top_groups(
                open(f, encoding="utf-8", errors="replace").read()):
            if (kind, name) in seen:
                continue
            seen.add((kind, name))
            extra.append(text)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(base + "\n" + "\n".join(extra) + "\n}\n")
    print(f"{out_path.split(chr(92))[-1]}: {len(seen)} 组（补入 {len(extra)}）")


merge(ALL_FILES, f"{BASE}/asap7_RVT_TT_merged.lib")
merge(COMB_FILES, f"{BASE}/asap7_RVT_TT_comb.lib")
print("done")

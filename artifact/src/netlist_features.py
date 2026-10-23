#!/usr/bin/env python3
"""netlist_features.py —— HT-Trojan 论文 C2 贡献：库无关网表特征提取。

输入：Yosys `write_json` 产出的门级网表（综合到任意工艺库后）。
输出：每个 module 一行特征向量（JSON / CSV），对单元库命名不敏感。

设计原则（对应论文 C2"库无关特征"）：
  1. 单元名只归一化到"基础逻辑类"（and/or/xor/xnor/inv/buf/mux/complex/seq/other），
     不依赖具体库的命名（Nangate_AND2_X1 / sky130_fd_sc_hd__and2_1 / $_AND_ 都归 and）。
  2. 拓扑特征（扇入/扇出/组合深度）基于位连接图，与库无关。
  3. 行为占比（各基础类占比、常量位占比）做归一化，跨库可比。

纯标准库实现。用法：
  python3 netlist_features.py netlist.json [--module TOP] [--out features.json]
"""
import json
import re
import sys
import math
import argparse
from collections import defaultdict

sys.path.insert(0, __file__.rsplit("\\", 1)[0] if "\\" in __file__ else ".")
try:
    from lib_ops import cell_ops as _lib_cell_ops, generic_cell_ops
except ImportError:  # 单独分发时允许只用 v1 特征
    _lib_cell_ops = None
    generic_cell_ops = None

# ---------------------------------------------------------------- 基础类归一化

# 时序单元提示词：覆盖 Yosys($_DFF_*)/Nangate(DFF*)/ASAP7(DFFHQN,DHLHLL)/
# sky130(dfxtp/dfrtp/dfstp/dfbbp/dlxtp) 等主流命名
SEQ_HINTS = ("dff", "dlatch", "latch", "dfxt", "dfrt", "dfst", "dfbb", "dfxb",
             "dlxt", "dhl", "dll", "lat")
CLASS_HINTS = (
    ("and", ("and", "nand")),
    ("or", ("or", "nor")),
    ("xor", ("xor", "xnor")),
    ("inv", ("inv", "not")),
    ("buf", ("buf", "clkbuf", "dly")),
    ("mux", ("mux", "mx2", "mx4")),
)


def base_class(cell_type: str) -> str:
    """把任意库的单元名归一化到基础逻辑类。

    TODO(C2 论文核心): 复合门跨库粒度不一致——sky130 一个 a2111oi 在 Nangate
    会被拆成 and/or/inv 三个单元，直接比 frac_complex/n_cells 会被库粒度污染。
    需要"复合门分解归一化"（按逻辑功能把 AOI/OAI/ANDNOT 等折算成基础算子计数），
    这是 C2 贡献的设计点之一，动笔前查 prior art（JKSU 2023 用过单元基础类型 one-hot）。
    """
    t = cell_type.lower()
    # Yosys 内部原语：$_AND_ $_MUX_ $_DFF_P_ ...
    core = t[1:-1] if t.startswith("$_") and t.endswith("_") else t
    if any(h in core for h in SEQ_HINTS):
        return "seq"
    # 复合单元（AOI/OAI/AOI22...）优先判 complex：同时含多种运算
    kinds = set()
    for cls, hints in CLASS_HINTS:
        if any(h in core for h in hints):
            kinds.add(cls)
    if "xor" in kinds:
        return "xor"
    if len(kinds) >= 2:
        return "complex"
    if kinds:
        return kinds.pop()
    # AOI/OAI 复合门：Nangate(AOI211/OAI221)、sky130(a2111oi/o21ai/a21boi)
    if "aoi" in core or "oai" in core or re.search(r"[ao]\d+[oi]", core):
        return "complex"
    return "other"


def is_seq(cell_type: str) -> bool:
    return base_class(cell_type) == "seq"


# ---------------------------------------------------------------- 网表解析


def parse_module(mod: dict):
    """返回 (cells, drivers, sinks)：
    cells:  [(name, type, {pin: [bits]}, {pin: dir})]
    drivers: {bit: (cell_name or 'PORT') }  谁驱动这个 bit
    sinks:   {bit: [(cell_name, pin)]}       谁消费这个 bit
    """
    cells = []
    drivers = {}
    sinks = defaultdict(list)

    for pname, pinfo in mod.get("ports", {}).items():
        direction = pinfo.get("direction", "input")
        for b in pinfo.get("bits", []):
            if direction == "output":
                drivers.setdefault(b, "PORT_OUT")   # 模块输出是"消费者"端
                sinks.setdefault(b, []).append(("PORT_OUT", pname))
            else:
                drivers.setdefault(b, "PORT_IN")    # 模块输入是驱动源

    for cname, cinfo in mod.get("cells", {}).items():
        ctype = cinfo.get("type", "?")
        conns = cinfo.get("connections", {})
        dirs = cinfo.get("port_directions", {})
        cells.append((cname, ctype, conns, dirs))

        for pin, bits in conns.items():
            d = dirs.get(pin)
            if d is None:
                # 兜底启发式：Y/Q/Z/O/OUT 视作输出，其余输入
                d = "output" if pin.upper().split("[")[0] in ("Y", "Q", "Z", "O", "OUT") else "input"
            if d == "output":
                for b in bits:
                    if isinstance(b, int):
                        drivers.setdefault(b, cname)
            else:
                for b in bits:
                    if isinstance(b, int):
                        sinks[b].append((cname, pin))
    return cells, drivers, sinks


# ---------------------------------------------------------------- 特征计算


def module_features(mod: dict, ops_map=None) -> dict:
    cells, drivers, sinks = parse_module(mod)
    n_cells = len(cells)
    if n_cells == 0:
        return {"n_cells": 0}

    cls_count = defaultdict(int)
    for _, ctype, _, _ in cells:
        cls_count[base_class(ctype)] += 1

    in_width = sum(len(p["bits"]) for p in mod.get("ports", {}).values() if p.get("direction") == "input")
    out_width = sum(len(p["bits"]) for p in mod.get("ports", {}).values() if p.get("direction") == "output")

    # 常量位占比：出现在连接里的 0/1（Yosys JSON 里常量是字符串 "0"/"1"）
    const_bits = 0
    total_conn_bits = 0
    for _, _, conns, _ in cells:
        for bits in conns.values():
            for b in bits:
                total_conn_bits += 1
                if isinstance(b, str):
                    const_bits += 1

    # 扇入/扇出（只统计 int bit；常量不计）
    fanin = {}
    fanout = defaultdict(int)
    for cname, ctype, conns, dirs in cells:
        fi = 0
        for pin, bits in conns.items():
            d = dirs.get(pin, "input" if pin.upper().split("[")[0] not in ("Y", "Q", "Z", "O", "OUT") else "output")
            if d == "input":
                fi += sum(1 for b in bits if isinstance(b, int))
        fanin[cname] = fi
    for b, users in sinks.items():
        if isinstance(b, int):
            for cname, _pin in users:
                if cname != "PORT_OUT":
                    fanout[cname] += 1

    def _stats(d):
        vals = list(d.values()) or [0]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        return {"mean": round(mean, 3), "max": max(vals), "std": round(math.sqrt(var), 3)}

    # 组合深度：时序单元输出 / 模块输入 = 深度 0 源；组合逻辑逐层传播
    depth = {}
    pending = list(cells)
    for _ in range(len(cells) + 1):  # 最多迭代 n 轮（链长上界）
        progress = False
        remain = []
        for cname, ctype, conns, dirs in cells:
            if cname in depth:
                continue
            if is_seq(ctype):
                depth[cname] = 0
                progress = True
                continue
            preds = []
            ok = True
            for pin, bits in conns.items():
                d = dirs.get(pin, "input" if pin.upper().split("[")[0] not in ("Y", "Q", "Z", "O", "OUT") else "output")
                if d != "input":
                    continue
                for b in bits:
                    if not isinstance(b, int):
                        continue
                    drv = drivers.get(b)
                    if drv in ("PORT_IN", "PORT_OUT", None):
                        continue
                    if drv in depth:
                        preds.append(depth[drv])
                    else:
                        ok = False
            if ok:
                depth[cname] = (max(preds) + 1) if preds else 0
                progress = True
            else:
                remain.append(cname)
        if not progress:
            for cname in remain:
                depth.setdefault(cname, 0)  # 环/未知 → 0
            break

    comb_depths = [depth[c] for c, t, _, _ in cells if not is_seq(t)] or [0]

    n_seq = cls_count.get("seq", 0)
    n_comb = n_cells - n_seq
    feat = {
        "n_cells": n_cells,
        "n_seq": n_seq,
        "in_width": in_width,
        "out_width": out_width,
        "const_bit_frac": round(const_bits / total_conn_bits, 4) if total_conn_bits else 0.0,
        "frac_seq": round(n_seq / n_cells, 4),
        "fanin": _stats(fanin),
        "fanout": _stats(fanout),
        "comb_depth_max": max(comb_depths),
        "comb_depth_mean": round(sum(comb_depths) / len(comb_depths), 3),
    }
    # 各基础类占比（库无关的核心特征）
    for cls in ("and", "or", "xor", "inv", "buf", "mux", "complex", "other"):
        feat[f"frac_{cls}"] = round(cls_count.get(cls, 0) / n_cells, 4)
    # 类分布熵（结构复杂度指标）
    ent = 0.0
    for c in cls_count.values():
        p = c / n_cells
        ent -= p * math.log2(p)
    feat["cell_class_entropy"] = round(ent, 4)
    # ---- v2：算子空间特征（C2 库无关归一化；ops_map=None 时跳过） ----
    if ops_map is not None and _lib_cell_ops is not None:
        op_counts = defaultdict(int)
        cell_opdepth = {}
        unk = 0
        seq_outs = set()
        for cname, ctype, conns, dirs in cells:
            info = _lib_cell_ops(ops_map, ctype)
            if info is None:
                unk += 1
                cell_opdepth[cname] = 1
                continue
            if info["seq"]:
                seq_outs.add(cname)
                cell_opdepth[cname] = 0
                continue
            for k, v in info["ops"].items():
                op_counts[k] += v
            cell_opdepth.setdefault(cname, info["depth"])
        n_ops = sum(op_counts.values())
        feat["n_ops"] = n_ops
        feat["unk_cell_frac"] = round(unk / n_cells, 4)
        for op in ("not", "and", "or", "xor", "mux"):
            feat[f"frac_op_{op}"] = round(op_counts.get(op, 0) / n_ops, 4) if n_ops else 0.0
        # 算子深度：时序输出/模块输入=0 源；组合单元累加其因式分解深度
        opdepth = {}
        pending = list(cells)
        for _ in range(len(cells) + 1):
            progress = False
            remain = []
            for cname, ctype, conns, dirs in cells:
                if cname in opdepth:
                    continue
                if cname in seq_outs:
                    opdepth[cname] = 0
                    progress = True
                    continue
                preds = []
                ok2 = True
                for pin, bits in conns.items():
                    d2 = dirs.get(pin, "input" if pin.upper().split("[")[0]
                                  not in ("Y", "Q", "Z", "O", "OUT") else "output")
                    if d2 != "input":
                        continue
                    for b in bits:
                        if not isinstance(b, int):
                            continue
                        drv = drivers.get(b)
                        if drv in ("PORT_IN", "PORT_OUT", None):
                            continue
                        if drv in opdepth:
                            preds.append(opdepth[drv])
                        else:
                            ok2 = False
                if ok2:
                    base = max(preds) if preds else 0
                    opdepth[cname] = base + cell_opdepth.get(cname, 1)
                    progress = True
                else:
                    remain.append(cname)
            if not progress:
                for cname in remain:
                    opdepth.setdefault(cname, cell_opdepth.get(cname, 1))
                break
        comb_opd = [opdepth[c] for c, t, _, _ in cells if not is_seq(t)] or [0]
        feat["comb_opdepth_max"] = max(comb_opd)
        feat["comb_opdepth_mean"] = round(sum(comb_opd) / len(comb_opd), 3)
    return feat


def extract(json_path: str, module: str = None, ops_map=None) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    modules = data.get("modules", {})
    targets = [module] if module else list(modules.keys())
    out = {}
    for m in targets:
        if m in modules:
            out[m] = module_features(modules[m], ops_map=ops_map)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("netlist", help="Yosys write_json 产出的 .json")
    ap.add_argument("--module", default=None, help="只提取指定模块（默认全部）")
    ap.add_argument("--out", default=None, help="输出文件（默认打印 stdout）")
    args = ap.parse_args()
    feats = extract(args.netlist, args.module)
    text = json.dumps(feats, indent=2, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"written: {args.out} ({len(feats)} modules)")
    else:
        print(text)


if __name__ == "__main__":
    main()

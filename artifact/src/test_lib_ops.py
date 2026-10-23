# -*- coding: utf-8 -*-
"""test_lib_ops.py —— lib_ops 单元测试（C2 核心不变性）。
核心断言：同一布尔功能 ⇒ 同一算子计数，且分解可加
（AOI21 格 == and+or+inv 三个格之和 == {and:1,or:1,not:1}）。
跑法：python src/test_lib_ops.py"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_ops import (cell_op_profile, load_lib_ops, generic_cell_ops,
                     stub_cell_ops, cell_ops)

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIBS = {
    "nangate45": os.environ.get(
        "NANGATE_LIB",
        os.path.join(_R, "libs", "liberty", "nangate45",
                     "NangateOpenCellLibrary_typical.lib")),
    "sky130hd": os.environ.get(
        "SKY130_LIB",
        os.path.join(_R, "libs", "liberty", "sky130hd",
                     "sky130_fd_sc_hd__tt_025C_1v80.lib")),
    "asap7": os.environ.get(
        "ASAP7_LIB",
        os.path.join(_R, "libs", "liberty", "asap7", "asap7_RVT_TT_merged.lib")),
}

# ---- 迷你 Liberty fixture：真库缺失时（如 artifact 开箱）跑基础测试 ----
_FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            os.pardir, "tests", "fixture")
_FIXTURE_LIBS = {
    "nangate45": os.path.join(_FIXTURE_DIR, "nangate45_mini.lib"),
    "sky130hd": os.path.join(_FIXTURE_DIR, "sky130_mini.lib"),
    "asap7": os.path.join(_FIXTURE_DIR, "asap7_mini.lib"),
}


def resolve_libs():
    """真库在就用真库（严格模式，>100 cells）；否则用 fixture（>3 cells）。
    环境变量 LIB_OPS_FIXTURE=1 可强制 fixture 模式（验证用）。"""
    force = os.environ.get("LIB_OPS_FIXTURE") == "1"
    if not force and all(os.path.exists(p) for p in LIBS.values()):
        return LIBS, 100
    if all(os.path.exists(p) for p in _FIXTURE_LIBS.values()):
        print("[fixture mode] 真实工艺库缺失/被禁用，使用 tests/fixture 迷你库")
        return _FIXTURE_LIBS, 3
    return LIBS, 100

def prof(expr):
    return cell_op_profile(expr)

def check(name, cond):
    print(("PASS" if cond else "FAIL"), name)
    return cond

def main():
    ok = True
    # ---- 基本形状（因式分解约定） ----
    ok &= check("AND2 = 1 and", prof("A1 & A2")[0] == {"and": 1})
    ok &= check("NAND2 = and+not", prof("!(A1 & A2)")[0] == {"and": 1, "not": 1})
    ok &= check("AND3 = 2 and", prof("A & B & C")[0] == {"and": 2})
    ok &= check("XOR2 = 1 xor", prof("A ^ B")[0] == {"xor": 1})
    ok &= check("XOR3 = 2 xor", prof("A ^ B ^ C")[0] == {"xor": 2})
    ok &= check("XNOR2 = xor+not", prof("!(A ^ B)")[0] == {"xor": 1, "not": 1})
    ok &= check("MUX = 1 mux", prof("(S&B)|(!S&A)")[0] == {"mux": 1})
    ok &= check("INV = 1 not", prof("!A")[0] == {"not": 1})
    ok &= check("BUF = 0 op", prof("A")[0] == {})
    ok &= check("OR3 = 2 or", prof("A + B + C")[0] == {"or": 2})

    # ---- 同一功能、三种代数写法 → 同一计数（C2 核心不变性） ----
    v1 = prof("!(A | (B1 & B2))")            # Nangate AOI21 写法
    v2 = prof("(!A1&!B1) | (!A2&!B1)")       # sky130 写法
    v3 = prof("(!A1 * !B) + (!A2 * !B)")     # asap7 写法
    print("  AOI21 三种写法:", v1, v2, v3)
    ok &= check("AOI21 三种代数写法计数一致",
                v1[0] == v2[0] == v3[0] == {"and": 1, "or": 1, "not": 1})
    ok &= check("AOI21 分解可加 == and+or+inv 三格之和",
                v1[0] == {"and": 1, "or": 1, "not": 1} and v1[1] == 3)
    ok &= check("OAI21 双写法一致", prof("!(A & (B1 | B2))")[0] ==
                prof("(!A1&!A2) + (!B)")[0] == {"or": 1, "and": 1, "not": 1})
    # 输入取反（内置反相引脚）应补 not
    vi = prof("(!A)&B")
    ok &= check("输入取反补 1 not", vi[0] == {"and": 1, "not": 1})

    # ---- 三库真解析 + 跨库不变性 ----
    maps = {}
    libs, min_cells = resolve_libs()
    for name, path in libs.items():
        maps[name] = load_lib_ops(path)
        print(f"  {name}: {len(maps[name])} cells parsed")
        ok &= check(f"{name} 解析非空", len(maps[name]) > min_cells)

    def get(lib, cell):
        return maps[lib].get(cell)

    a = get("nangate45", "AOI21_X1")
    b = get("sky130hd", "sky130_fd_sc_hd__a21oi_1")
    c = get("asap7", "AOI21xp33_ASAP7_75t_R")
    print("  三库 AOI21 ops:", dict(a["ops"]), dict(b["ops"]), dict(c["ops"]))
    ok &= check("AOI21 跨库一致 (Nangate vs sky130)",
                dict(a["ops"]) == dict(b["ops"]) == {"and": 1, "or": 1, "not": 1})
    ok &= check("AOI21 跨库一致 (Nangate vs asap7)",
                dict(a["ops"]) == dict(c["ops"]))
    ok &= check("AOI21 三库深度一致", a["depth"] == b["depth"] == c["depth"] == 3)

    a = get("nangate45", "OAI21_X1")
    b = get("sky130hd", "sky130_fd_sc_hd__o21ai_1")
    c = get("asap7", "OAI21xp33_ASAP7_75t_R")
    ok &= check("OAI21 跨库一致", dict(a["ops"]) == dict(b["ops"]) ==
                dict(c["ops"]) == {"or": 1, "and": 1, "not": 1})

    # 更复杂形状也应跨库一致
    for nang, sky, asap in (("AOI22_X1", "sky130_fd_sc_hd__a22oi_1", "AOI22xp5_ASAP7_75t_R"),
                            ("AOI211_X1", "sky130_fd_sc_hd__a211oi_1", "AOI211xp5_ASAP7_75t_R")):
        aa, bb, cc = get("nangate45", nang), get("sky130hd", sky), get("asap7", asap)
        if aa and bb and cc:
            ok &= check(f"{nang} 跨库一致", dict(aa["ops"]) == dict(bb["ops"]) == dict(cc["ops"]))
        else:
            print(f"  (skip {nang}: 缺 cell)", aa and dict(aa["ops"]), bb and dict(bb["ops"]), cc and dict(cc["ops"]))

    # 时序单元：DFF 不计组合算子
    for lib, cell in (("nangate45", "DFF_X1"), ("sky130hd", "sky130_fd_sc_hd__dfxtp_1"),
                      ("asap7", "DFFHQx4_ASAP7_75t_R")):
        info = get(lib, cell)
        ok &= check(f"{lib} {cell} 标记 seq 且 0 组合算子",
                    info is not None and info["seq"] and sum(info["ops"].values()) == 0)

    # ---- yosys 原语 ----
    g = generic_cell_ops()
    ok &= check("$_NAND_ = and+not", g["$_NAND_"]["ops"] == {"and": 1, "not": 1})
    ok &= check("$_MUX_ = mux", g["$_MUX_"]["ops"] == {"mux": 1})
    ok &= check("$_BUF_ = 0 op", g["$_BUF_"]["ops"] == {})
    ok &= check("$_AOI4_ = 2and+1or+1not", g["$_AOI4_"]["ops"] ==
                {"and": 2, "or": 1, "not": 1})
    ok &= check("$_ANDNOT_ = and+not", g["$_ANDNOT_"]["ops"] == {"and": 1, "not": 1})
    ok &= check("generic 与 Nangate 同功能同计数",
                g["$_NAND_"]["ops"] == dict(get("nangate45", "NAND2_X1")["ops"]))

    # ---- 黑盒桩兜底（TRIT 命名） ----
    ok &= check("stub nor2s1 = or+not", stub_cell_ops("nor2s1")["ops"] ==
                {"or": 1, "not": 1})
    ok &= check("stub nnd4s1 = 3and+1not", stub_cell_ops("nnd4s1")["ops"] ==
                {"and": 3, "not": 1})
    ok &= check("stub xor2s1 = xor", stub_cell_ops("xor2s1")["ops"] == {"xor": 1})
    ok &= check("stub i1s1 = not", stub_cell_ops("i1s1")["ops"] == {"not": 1})
    ok &= check("stub hi1s1 = 0 op", stub_cell_ops("hi1s1")["ops"] == {})
    ok &= check("cell_ops 查表优先", cell_ops(maps["nangate45"], "AOI21_X1") is not None)
    ok &= check("未知类型返回 None", cell_ops({}, "weird_cell_99") is None)

    print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())

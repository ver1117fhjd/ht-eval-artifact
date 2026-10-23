#!/usr/bin/env python3
"""netlist_features.py 单元测试：手工构造 yosys JSON 结构，验证解析与特征计算。"""
import os
import sys
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from netlist_features import base_class, module_features, extract  # noqa: E402

# ---- base_class 归一化：跨库命名都归到基础类 ----
assert base_class("NangateOpenCellLibrary_AND2_X1") == "and"
assert base_class("sky130_fd_sc_hd__and2_1") == "and"
assert base_class("$_AND_") == "and"
assert base_class("$_DFF_P_") == "seq"
assert base_class("sky130_fd_sc_hd__dfxtp_1") == "seq"
assert base_class("NangateOpenCellLibrary_XOR2_X1") == "xor"
assert base_class("$_MUX_") == "mux"
assert base_class("sky130_fd_sc_hd__a2111oi_1") == "complex"

# ---- 手工网表：PI -> AND -> XOR -> MUX -> PO，MUX 受 DFF 反馈控制 ----
# bits: 3=clk 4=a 5=b 6=y(PO) 7=t1 8=t2 9=q
mod = {
    "ports": {
        "clk": {"direction": "input", "bits": [3]},
        "a": {"direction": "input", "bits": [4]},
        "b": {"direction": "input", "bits": [5]},
        "y": {"direction": "output", "bits": [6]},
    },
    "cells": {
        "U1": {"type": "$_AND_",
               "connections": {"A": [4], "B": [5], "Y": [7]},
               "port_directions": {"A": "input", "B": "input", "Y": "output"}},
        "U2": {"type": "NangateOpenCellLibrary_XOR2_X1",
               "connections": {"A": [7], "B": [4], "Y": [8]},
               "port_directions": {"A": "input", "B": "input", "Y": "output"}},
        "U3": {"type": "sky130_fd_sc_hd__mux2_1",
               "connections": {"A0": [8], "A1": [5], "S": [9], "X": [6]},
               "port_directions": {"A0": "input", "A1": "input", "S": "input", "X": "output"}},
        "U4": {"type": "$_DFF_P_",
               "connections": {"C": [3], "D": [6], "Q": [9]},
               "port_directions": {"C": "input", "D": "input", "Q": "output"}},
    },
    "netnames": {},
}

feat = module_features(mod)
assert feat["n_cells"] == 4, feat
assert feat["n_seq"] == 1, feat
assert feat["frac_seq"] == 0.25, feat
assert feat["in_width"] == 3 and feat["out_width"] == 1, feat
assert feat["frac_and"] == 0.25 and feat["frac_xor"] == 0.25 and feat["frac_mux"] == 0.25, feat
assert feat["comb_depth_max"] == 2, feat  # U1=0, U2=1, U3=2
assert feat["fanin"]["mean"] > 0, feat

# ---- extract() 全流程（写临时文件再读回）----
data = {"creator": "test", "modules": {"top": mod}}
with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
    json.dump(data, f)
    tmp = f.name
try:
    out = extract(tmp, "top")
    assert out["top"]["n_cells"] == 4
finally:
    os.unlink(tmp)

print("netlist_features: ALL TESTS PASSED")

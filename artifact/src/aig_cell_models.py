#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""aig_cell_models.py —— 从 liberty 生成组合单元的 Verilog 功能模型（AIG 基线 B 用）。

把 .lib 里每个组合单元的 function 表达式转成 assign 语句的 Verilog module，
让 yosys 能把已映射网表里的复合门 flatten 展开成逻辑，再 `abc -g AND` 归一到 AIG。
时序单元（含 ff/latch）不生成模型（保持黑盒，AIG 只归一化组合逻辑）。

用法: python aig_cell_models.py <lib.lib> <out_models.v>
输出统计打印到 stdout：生成/跳过数量及跳过原因。
"""
import os
import re
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)

from lib_ops import (CELL_RE, PIN_RE, DIR_RE, FUNC_RE, _block_end,  # noqa: E402
                     CONST_ENV, STATE_TOKENS, FF_RE)

VERILOG_KEYWORDS = {
    "and", "or", "not", "nand", "nor", "xor", "xnor", "buf",
    "bufif0", "bufif1", "notif0", "notif1", "pmos", "nmos", "cmos",
    "tran", "tranif0", "tranif1", "rtran", "rtranif0", "rtranif1",
    "supply0", "supply1", "tri", "module", "endmodule", "input",
    "output", "inout", "wire", "reg", "assign", "begin", "end",
    "case", "endcase", "default", "if", "else", "for", "while",
    "function", "endfunction", "task", "endtask", "parameter",
    "localparam", "integer", "real", "time", "genvar", "generate",
    "endgenerate", "initial", "always", "posedge", "negedge",
    "specify", "endspecify", "defparam", "table", "endtable",
}

IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*")
OK_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")
CONST_RE = re.compile(r"\b(" + "|".join(sorted(CONST_ENV)) + r")\b")


def is_ok_name(name):
    return bool(OK_NAME_RE.match(name)) and name.lower() not in VERILOG_KEYWORDS


def translate_expr(expr):
    """liberty function 表达式 → Verilog 表达式。* → & 、+ → |（asap7 代数写法）；
    电源 token 按常量求值。返回 (verilog_expr, ids) 或 None（含未知标识符）。"""
    e = CONST_RE.sub(lambda m: "1'b1" if CONST_ENV[m.group(1)] else "1'b0", expr)
    e = e.replace("*", "&").replace("+", "|")
    ids = set(IDENT_RE.findall(e))
    return e, ids


def build_models(lib_path):
    text = open(lib_path, encoding="utf-8", errors="replace").read()
    modules = []
    stats = {"total": 0, "seq": 0, "badsig": 0, "badname": 0, "nofunc": 0,
             "state": 0, "unknown_id": 0, "ok": 0}
    for m in CELL_RE.finditer(text):
        name = m.group(1)
        stats["total"] += 1
        brace = text.index("{", m.end() - 1)
        end = _block_end(text, brace)
        body = text[m.end():end]
        if FF_RE.search(body):
            stats["seq"] += 1
            continue
        if not is_ok_name(name):
            stats["badname"] += 1
            continue
        inputs, outputs = [], []   # (pin, expr)
        skip = None
        for pm in PIN_RE.finditer(body):
            pbrace = body.index("{", pm.end() - 1)
            pend = _block_end(body, pbrace)
            pbody = body[pm.end():pend]
            pname = pm.group(1)
            dm = DIR_RE.search(pbody)
            if not dm:
                skip = "badsig"
                break
            d = dm.group(1)
            if d == "input":
                if not is_ok_name(pname):
                    skip = "badname"
                    break
                inputs.append(pname)
            elif d == "output":
                fm = FUNC_RE.search(pbody)
                if not fm:
                    continue
                expr = fm.group(1)
                ids = set(IDENT_RE.findall(expr))
                if ids & STATE_TOKENS:
                    skip = "state"
                    break
                e, eids = translate_expr(expr)
                outputs.append((pname, e, eids))
            else:
                # internal/inout：电源脚已被 PIN_RE 的 \b 过滤大半；剩的按名字给常量或放弃
                if pname not in CONST_ENV:
                    skip = "badsig"
                    break
        if skip:
            stats[skip] += 1
            continue
        if not outputs:
            stats["nofunc"] += 1
            continue
        in_set = set(inputs)
        skip = None
        for pname, e, eids in outputs:
            if not is_ok_name(pname):
                skip = "badname"
                break
            if eids - in_set:
                skip = "unknown_id"
                break
        if skip:
            stats[skip] += 1
            continue
        lines = ["module %s(%s);" % (name, ", ".join(inputs + [p for p, _, _ in outputs]))]
        if inputs:
            lines.append("input %s;" % ", ".join(inputs))
        lines.append("output %s;" % ", ".join(p for p, _, _ in outputs))
        for pname, e, _ in outputs:
            lines.append("assign %s = (%s);" % (pname, e))
        lines.append("endmodule")
        modules.append("\n".join(lines))
        stats["ok"] += 1
    return modules, stats


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    lib_path, out_path = sys.argv[1], sys.argv[2]
    modules, stats = build_models(lib_path)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("// auto-generated from %s by aig_cell_models.py\n" % os.path.basename(lib_path))
        f.write("\n\n".join(modules) + "\n")
    print("lib=%s -> %s" % (lib_path, out_path))
    print("stats:", stats)


if __name__ == "__main__":
    main()

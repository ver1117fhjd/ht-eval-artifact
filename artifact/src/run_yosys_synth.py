#!/usr/bin/env python3
"""run_yosys_synth.py —— HT 篇综合流水线驱动：Trust-Hub 基准 → Yosys → 库无关特征。

工程处理：
  - `include 原作者机器绝对路径 → 拷贝工作副本并改写为本目录 include
  - 排除测试台（bench/tb/test/sim/models 目录，test*/bench*/tb* 文件名）
  - 同名文件去重（多份拷贝只取一份）
  - yosys 脚本写文件传递（yosys -s），绕开 Windows 命令行 32K 限制
用法：
  python run_yosys_synth.py <设计根目录> [--liberty xxx.lib] [--out 输出目录] [--label 标签]
"""
import os
import re
import sys
import json
import shutil
import argparse
import subprocess

YOSYS = os.environ.get("YOSYS", r"<local-path>")  # override via env var
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from netlist_features import extract  # noqa: E402

INC_RE = re.compile(r'(`include\s+")([^"]+)(")')
DIR_EXCL = re.compile(
    r"(?i)^(bench|benchmarks?|tb|test|tests|testbench|sim|simulation|stim|stimulus|wave|models?|backup|bak|doc|docs)$")
FILE_TB = re.compile(r"(?i)^(test|bench|tb|stim|stimulus|wave)|(_tb|_test|tb_)\.|testbench")
HDL_EXT = (".v", ".sv", ".vh", ".h", ".inc")

VERILOG_KEYWORDS = {
    "if", "else", "always", "assign", "module", "endmodule", "wire", "reg", "input",
    "output", "inout", "begin", "end", "case", "casex", "casez", "endcase",
    "function", "endfunction", "task", "endtask", "generate", "endgenerate",
    "for", "while", "initial", "parameter", "localparam", "integer", "genvar",
    "posedge", "negedge", "or", "and", "not", "buf", "primitive", "table",
}


def _strip_comments(src):
    src = re.sub(r"//[^\n]*", "", src)
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return src


def _pin_dir(pin):
    """标准单元引脚方向启发式：输出名 → output，其余 → input。"""
    p = pin.lstrip("\\").upper()
    if re.match(r"^(Y|Q|QN|O|OUT|X|Z|COUT|SUM|CO|N)\d*$", p):
        return "output"
    return "input"


def gen_blackbox_stubs(workdir):
    """扫描工作副本：实例化了但没定义的模块 → 生成 (* blackbox *) 桩文件。
    门级网表直接调用工艺库单元（xor2s1/NOR2X0/...），包里不带库定义时靠它过 hierarchy。"""
    defined, used = set(), {}
    stale = os.path.join(workdir, "blackbox_stubs.v")
    if os.path.isfile(stale):
        os.remove(stale)  # 上次运行的残留会自我污染，先清
    for fn in sorted(os.listdir(workdir)):
        if os.path.splitext(fn)[1].lower() not in (".v", ".sv") \
                or fn == "blackbox_stubs.v":
            continue
        src = _strip_comments(open(os.path.join(workdir, fn),
                                  encoding="utf-8", errors="replace").read())
        for m in re.finditer(r"\bmodule\s+([A-Za-z_\\][\w$]*)", src):
            defined.add(m.group(1))
        for m in re.finditer(
                r"^\s*([A-Za-z_\\][^\s(#]*)\s+(?:[A-Za-z_\\][^\s(]*|\[[^\]]*\])\s*\(",
                src, re.M):
            name = m.group(1)
            if name.lower() in VERILOG_KEYWORDS:
                continue
            start = m.end() - 1
            depth, i = 0, start
            while i < len(src):
                if src[i] == "(":
                    depth += 1
                elif src[i] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            ports = re.findall(r"\.\s*([A-Za-z_\\][\w$]*)\s*\(", src[start:i + 1])
            if ports:
                used.setdefault(name, set()).update(ports)

    missing = {k: v for k, v in used.items() if k not in defined}
    if not missing:
        return 0
    lines = []
    for name, ports in sorted(missing.items()):
        ins = sorted(p for p in ports if _pin_dir(p) == "input")
        outs = sorted(p for p in ports if _pin_dir(p) == "output")
        decls = []
        if ins:
            decls.append("input " + ", ".join(ins))
        if outs:
            decls.append("output " + ", ".join(outs))
        lines.append(f"(* blackbox *) module {name}(" + ", ".join(decls) + "); endmodule")
    with open(os.path.join(workdir, "blackbox_stubs.v"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return len(missing)


def sanitize_includes(workdir):
    n = 0
    for f in os.listdir(workdir):
        p = os.path.join(workdir, f)
        if os.path.splitext(f)[1].lower() not in HDL_EXT or not os.path.isfile(p):
            continue
        src = open(p, encoding="utf-8", errors="replace").read()

        def repl(m):
            nonlocal n
            base = os.path.basename(m.group(2).replace("\\", "/"))
            if base != m.group(2):
                n += 1
            return m.group(1) + base + m.group(3)

        new = INC_RE.sub(repl, src)
        if new != src:
            open(p, "w", encoding="utf-8").write(new)
    return n


def collect_sources(root):
    """收集 root 下（递归）的 HDL 文件，排除测试台，按 basename 去重。"""
    vs, hs, seen = [], [], set()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not DIR_EXCL.match(d)]
        for fn in sorted(filenames):
            low = fn.lower()
            if os.path.splitext(low)[1] not in HDL_EXT:
                continue
            if fn.endswith("~") or low.endswith((".bak", ".do")):
                continue
            if FILE_TB.search(low):
                continue
            if fn in seen:          # 多份同名拷贝只取一份
                continue
            seen.add(fn)
            p = os.path.join(dirpath, fn)
            if low.endswith((".v", ".sv")):
                vs.append(p)
            else:
                hs.append(p)
    return sorted(vs), sorted(hs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="设计根目录")
    ap.add_argument("--liberty", default=None)
    ap.add_argument("--liberty-abc", default=None,
                    help="abc 用的 .lib（默认同 --liberty；asap7 需纯组合合并库）")
    ap.add_argument("--out", default=None, help="输出目录")
    ap.add_argument("--label", default="", help="设计标签（如 TjIn / TjFree / s13207_T400）")
    args = ap.parse_args()

    label = args.label or os.path.basename(os.path.normpath(args.root))
    workdir = args.out or os.path.join(r"./experiments\synth_work", label)
    os.makedirs(workdir, exist_ok=True)

    vs, hs = collect_sources(args.root)
    if not vs:
        print("没找到 .v 文件:", args.root)
        sys.exit(1)

    for p in vs + hs:
        shutil.copy(p, os.path.join(workdir, os.path.basename(p)))
    n_fixed = sanitize_includes(workdir)
    n_stubs = gen_blackbox_stubs(workdir)
    print(f"设计 {label}: {len(vs)} .v + {len(hs)} 头文件；include 改写 {n_fixed} 处，"
          f"黑盒桩 {n_stubs} 个")

    wd = workdir.replace("\\", "/")
    vfiles = " ".join(f"{wd}/{os.path.basename(p)}" for p in vs)
    jpath = f"{wd}/netlist.json"
    sfile = os.path.join(workdir, "synth.ys")
    lines = []
    if n_stubs:
        lines.append(f"read_verilog {wd}/blackbox_stubs.v")
    lines += [f"read_verilog -I{wd} {vfiles}", "hierarchy -auto-top", "synth"]
    if args.liberty:
        abc_lib = args.liberty_abc or args.liberty
        lines += [f"dfflibmap -liberty {args.liberty}",
                  f"abc -liberty {abc_lib}", "opt"]
    lines += ["flatten", "opt", f"write_json {jpath}", "stat"]
    open(sfile, "w", encoding="utf-8").write("\n".join(lines) + "\n")

    print("运行 Yosys ...")
    r = subprocess.run([YOSYS, "-s", sfile], capture_output=True, text=True, timeout=1800)
    open(os.path.join(workdir, "yosys.log"), "w", encoding="utf-8").write(
        r.stdout + "\n" + r.stderr)
    if not os.path.isfile(jpath):
        print("Yosys 失败:", (r.stderr or r.stdout)[-300:].replace("\n", " "))
        sys.exit(2)

    feats = {m: f for m, f in extract(jpath).items() if f.get("n_cells", 0) > 0}
    json.dump({"label": label, "root": args.root, "liberty": args.liberty,
               "n_blackbox_stubs": n_stubs, "features": feats},
              open(os.path.join(workdir, "features.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    for mod, f in feats.items():
        print(f"  模块 {mod}: n_cells={f.get('n_cells')} seq={f.get('n_seq')} "
              f"depth={f.get('comb_depth_max')}")
    print("完成:", workdir)


if __name__ == "__main__":
    main()

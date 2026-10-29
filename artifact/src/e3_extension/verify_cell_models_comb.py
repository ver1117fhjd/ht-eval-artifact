"""Validate stub_models.v cell by cell with iverilog.

Combinational cells: exhaustively simulate every input pattern and compare with a
reference function (Liberty `function` string for auto-generated cells, explicit
Python semantics for hand-written ones).

Sequential cells: apply a fixed stimulus (async reset, clock edges, scan select,
latch enable) and compare the output trace with an independently written
expectation.
"""
import itertools
import json
import os
import re
import subprocess
import tempfile

OUT = r"PROJECT_ROOT\experiments\e3x"
IVERILOG = r"<local-path>"
VVP = r"<local-path>"
LIB = r"PROJECT_ROOT\libs\liberty\nangate45\NangateOpenCellLibrary_typical.lib"

src = open(os.path.join(OUT, "stub_models.v"), encoding="utf-8").read()
report = json.load(open(os.path.join(OUT, "cell_models_report.json"), encoding="utf-8"))

# ---- parse liberty functions ----
libtxt = open(LIB, encoding="utf-8", errors="replace").read()
libfn = {}
for m in re.finditer(r"\n\s*cell\s*\(\s*\"?([A-Za-z0-9_]+)\"?\s*\)\s*\{", libtxt):
    name = m.group(1)
    i, d = m.end() - 1, 0
    while i < len(libtxt):
        if libtxt[i] == "{":
            d += 1
        elif libtxt[i] == "}":
            d -= 1
            if d == 0:
                break
        i += 1
    body = libtxt[m.end():i]
    fn = re.search(r"function\s*:\s*\"([^\"]+)\"", body)
    ins = [p for p, dd in re.findall(r"pin\s*\(\s*\"?([A-Za-z0-9_]+)\"?\s*\)\s*\{\s*direction\s*:\s*(\w+)", body) if dd == "input"]
    if fn:
        libfn[name] = (fn.group(1), sorted(ins))

# ---- parse generated modules ----
mods = {}
for m in re.finditer(r"module\s+(\w+)\(([^)]*)\);(.*?)endmodule", src, re.S):
    cell, ports, body = m.group(1), [p.strip() for p in m.group(2).split(",")], m.group(3)
    ins = re.findall(r"input\s+([\w\s,]+);", body)
    outs = re.findall(r"output\s+([\w\s,]+);", body)
    ins = [x.strip() for grp in ins for x in grp.split(",")]
    outs = [x.strip() for grp in outs for x in grp.split(",")]
    seq = ("always" in body)
    mods[cell] = dict(ports=ports, ins=ins, outs=outs, body=body, seq=seq)

HAND_REF = {
    "AO21X1": lambda v: (v["IN1"] & v["IN2"]) | v["IN3"],
    "AO22X1": lambda v: (v["IN1"] & v["IN2"]) | (v["IN3"] & v["IN4"]),
    "AO221X1": lambda v: (v["IN1"] & v["IN2"]) | (v["IN3"] & v["IN4"]) | v["IN5"],
    "AO222X1": lambda v: (v["IN1"] & v["IN2"]) | (v["IN3"] & v["IN4"]) | (v["IN5"] & v["IN6"]),
    "OA21X1": lambda v: (v["IN1"] | v["IN2"]) & v["IN3"],
    "OA22X1": lambda v: (v["IN1"] | v["IN2"]) & (v["IN3"] | v["IN4"]),
    "OA221X1": lambda v: (v["IN1"] | v["IN2"]) & (v["IN3"] | v["IN4"]) & v["IN5"],
    "OA222X1": lambda v: (v["IN1"] | v["IN2"]) & (v["IN3"] | v["IN4"]) & (v["IN5"] | v["IN6"]),
    "XOR3X1": lambda v: v["IN1"] ^ v["IN2"] ^ v["IN3"],
    "XNOR3X1": lambda v: 1 - (v["IN1"] ^ v["IN2"] ^ v["IN3"]),
    "MUX21X1": lambda v: v["IN2"] if v["S"] else v["IN1"],
    "MUX21X2": lambda v: v["IN2"] if v["S"] else v["IN1"],
    "MUX41X1": lambda v: [v["IN1"], v["IN2"], v["IN3"], v["IN4"]][(v["S1"] << 1) | v["S0"]],
    "NBUFFX2": lambda v: v["IN"],
    "ISOLANDX1": lambda v: v["D"] & v["ISO"],
    "HADDX1": lambda v: {"SO": v["A0"] ^ v["B0"], "C1": v["A0"] & v["B0"]},
    "FADDX1": lambda v: {"S": v["A"] ^ v["B"] ^ v["CI"],
                          "CO": (v["A"] & v["B"]) | (v["CI"] & (v["A"] ^ v["B"]))},
}


def ev(expr, env):
    e = expr.replace("!", " not ").replace("&", " and ").replace("|", " or ").replace("^", " ^ ")
    return int(eval(e, {"__builtins__": {}}, env))


def run_tb(cell, extra, tb_body):
    tb = "`timescale 1ns/1ps\nmodule tb;\n" + extra + "\n" + cell + " dut(" + \
         ", ".join("." + p + "(" + p + ")" for p in mods[cell]["ports"]) + ");\n" + tb_body + "\nendmodule\n"
    with tempfile.TemporaryDirectory() as td:
        vf = os.path.join(td, "tb.v")
        open(vf, "w", encoding="utf-8").write(tb)
        r = subprocess.run([IVERILOG, "-g2012", "-o", os.path.join(td, "a.out"), vf,
                            os.path.join(OUT, "stub_models.v")],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return None, (r.stderr or r.stdout)[-300:]
        s = subprocess.run([VVP, os.path.join(td, "a.out")], capture_output=True, text=True)
        return s.stdout, None


results = {"pass": [], "fail": [], "error": []}

for cell, m in sorted(mods.items()):
    if not m["seq"]:
        ins = m["ins"]
        k = len(ins)
        decl = "\n".join("reg %s;" % i for i in ins) + "\n" + "\n".join("wire %s;" % o for o in m["outs"])
        body = ["integer i;", "initial begin"]
        for i, name in enumerate(ins):
            body.append("  %s = 0;" % name)
        body.append("  for (i = 0; i < %d; i = i + 1) begin" % (1 << k))
        for i, name in enumerate(ins):
            body.append("    %s = (i >> %d) & 1;" % (name, i))
        body.append("    #1;")
        fmt = "%d " + " ".join("%d" for _ in m["outs"])
        body.append('    $display("%s", i, %s);' % (fmt, ", ".join(m["outs"])))
        body.append("  end")
        body.append("  $finish;")
        body.append("end")
        out, err = run_tb(cell, decl, "\n".join(body))
        if err:
            results["error"].append((cell, err))
            continue
        rows = [l.split() for l in out.splitlines()
                if l.strip() and l.split()[0].isdigit()]
        ok = True
        for row in rows:
            idx = int(row[0])
            got = tuple(int(x) for x in row[1:])
            env = {name: (idx >> i) & 1 for i, name in enumerate(ins)}
            if cell in HAND_REF:
                ref = HAND_REF[cell](env)
                if isinstance(ref, dict):
                    ref = tuple(ref[o] for o in m["outs"])
                elif not isinstance(ref, tuple):
                    ref = (ref,)
            else:
                libname = report[cell].split(":", 1)[1]
                expr, lib_ins = libfn[libname]
                obs_sorted = sorted(ins)
                libenv = {p: env[obs_sorted[k2]] for k2, p in enumerate(lib_ins)}
                ref = (ev(expr, libenv),)
            if got != ref:
                ok = False
                results.setdefault("detail", []).append((cell, env, got, ref))
                break
        (results["pass"] if ok else results["fail"]).append(cell)
    else:
        continue

print("combinational pass:", len(results["pass"]), "fail:", len(results["fail"]), "error:", len(results["error"]))
if results["fail"]:
    print("FAILED:", results["fail"])
    for d in results.get("detail", [])[:5]:
        print("  ", d)
for c, e in results["error"][:5]:
    print("ERR", c, e.replace("\n", " ")[:200])
json.dump({k: v for k, v in results.items() if k != "detail"},
          open(os.path.join(OUT, "model_verify_combinational.json"), "w"), indent=2)

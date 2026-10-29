"""Generate Verilog functional models for the 58 Trust-Hub (Nangate-original) cells.

Sources:
  * auto: our nangate45 Liberty `function` string, pin names remapped to the
    observed instance pins (sorted Liberty inputs -> IN1..INn, Z->Q, ZN->QN);
  * hand: standard cell semantics for cells absent from our Liberty, plus all
    sequential cells (DFF/latch/scan), written explicitly below.

Outputs stub_models.v and a JSON report of which cell came from where.
"""
import json
import os
import re
from collections import defaultdict

OUT = r"PROJECT_ROOT\experiments\e3x"
inv = json.load(open(os.path.join(OUT, "cell_inventory.json"), encoding="utf-8"))
LIB = r"PROJECT_ROOT\libs\liberty\nangate45\NangateOpenCellLibrary_typical.lib"
libtxt = open(LIB, encoding="utf-8", errors="replace").read()

cells_lib = {}
for m in re.finditer(r"\n\s*cell\s*\(\s*\"?([A-Za-z0-9_]+)\"?\s*\)\s*\{", libtxt):
    name = m.group(1)
    i, depth = m.end() - 1, 0
    while i < len(libtxt):
        if libtxt[i] == "{":
            depth += 1
        elif libtxt[i] == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    body = libtxt[m.end():i]
    pins = re.findall(r"pin\s*\(\s*\"?([A-Za-z0-9_]+)\"?\s*\)\s*\{\s*direction\s*:\s*(\w+)", body)
    fn = re.search(r"function\s*:\s*\"([^\"]+)\"", body)
    cells_lib[name] = {"pins": pins, "function": fn.group(1) if fn else None}

# ---------------- hand-written models (standard cell semantics) -------------
HAND = {
    "AO21X1": "assign Q = (IN1 & IN2) | IN3;",
    "AO22X1": "assign Q = (IN1 & IN2) | (IN3 & IN4);",
    "AO221X1": "assign Q = (IN1 & IN2) | (IN3 & IN4) | IN5;",
    "AO222X1": "assign Q = (IN1 & IN2) | (IN3 & IN4) | (IN5 & IN6);",
    "OA21X1": "assign Q = (IN1 | IN2) & IN3;",
    "OA22X1": "assign Q = (IN1 | IN2) & (IN3 | IN4);",
    "OA221X1": "assign Q = (IN1 | IN2) & (IN3 | IN4) & IN5;",
    "OA222X1": "assign Q = (IN1 | IN2) & (IN3 | IN4) & (IN5 | IN6);",
    "XOR3X1": "assign Q = IN1 ^ IN2 ^ IN3;",
    "XNOR3X1": "assign Q = ~(IN1 ^ IN2 ^ IN3);",
    "MUX21X1": "assign Q = S ? IN2 : IN1;",
    "MUX21X2": "assign Q = S ? IN2 : IN1;",
    "MUX41X1": "assign Q = S1 ? (S0 ? IN4 : IN3) : (S0 ? IN2 : IN1);",
    "NBUFFX2": "assign Q = IN;",
    "HADDX1": "assign SO = A0 ^ B0; assign C1 = A0 & B0;",
    "FADDX1": "assign S = A ^ B ^ CI; assign CO = (A & B) | (CI & (A ^ B));",
    "ISOLANDX1": "assign Q = D & ISO;",
    "LSDNX1": "always_latch begin Q = D; end",
    "LSDNENX1": "always_latch begin if (!ENB) Q = D; end",
    "DFFX2": "always @(posedge CLK) Q <= D; assign QN = ~Q;",
    "DFFNX2": "always @(negedge CLK) Q <= D;",
    "DFFARX1": "always @(posedge CLK or negedge RSTB) if (!RSTB) Q <= 1'b0; else Q <= D; assign QN = ~Q;",
    "LARX1": "always_latch begin if (!RSTB) Q = 1'b0; else Q = D; end assign QN = ~Q;",
    "SDFFX1": "always @(posedge CLK) Q <= SE ? SI : D; assign QN = ~Q;",
    "SDFFARX1": "always @(posedge CLK or negedge RSTB) if (!RSTB) Q <= 1'b0; else Q <= SE ? SI : D; assign QN = ~Q;",
}

DIR = {"input": "input", "output": "output", "inout": "inout"}


def auto_model(cell, libname):
    info = cells_lib[libname]
    if not info["function"]:
        return None
    ins = sorted([p for p, d in info["pins"] if d == "input"])
    outs = [p for p, d in info["pins"] if d == "output"]
    if len(outs) != 1:
        return None
    expr = info["function"]
    expr = expr.replace("\\", "").replace(" ", "")
    expr = expr.replace("!", "~").replace("&", "&").replace("|", "|")
    # rename Liberty inputs to observed IN1..INn (sorted order)
    mapping = {}
    for k, p in enumerate(ins, start=1):
        mapping[p] = "IN%d" % k
    # token-safe rename
    for p in sorted(mapping, key=len, reverse=True):
        expr = re.sub(r"\b" + re.escape(p) + r"\b", mapping[p], expr)
    out = outs[0]
    obs = inv[cell]["pins_observed"]
    if "Q" in obs:
        target = "Q"
    elif "QN" in obs:
        target = "QN"
    else:
        return None
    # NOTE: this library keeps the `ZN` pin name even for positive functions,
    # so polarity comes from the function string, never from the pin name.
    body = "assign %s = %s;" % (target, expr)
    if "QN" in obs and target == "Q":
        body += " assign QN = ~Q;"
    return dict(body=body, source="liberty:" + libname)


models = {}


def hand_outputs(cell, obs):
    if cell.startswith("FADD"):
        return [p for p in obs if p in ("S", "CO")]
    if cell.startswith("HADD"):
        return [p for p in obs if p in ("SO", "C1")]
    return [p for p in obs if p in ("Q", "QN")]


for cell, meta in inv.items():
    obs = meta["pins_observed"]
    if cell in HAND:
        outs = hand_outputs(cell, obs)
        ins = [p for p in obs if p not in outs]
        models[cell] = dict(ins=ins, outs=outs, body=HAND[cell], source="hand")
        continue
    libname = meta["lib_cell"]
    m = auto_model(cell, libname) if libname else None
    if m:
        outs = [p for p in obs if p in ("Q", "QN")]
        ins = [p for p in obs if p not in outs]
        body = m["body"]
        # auto_model emitted positional names IN1..INn; restore the observed ones
        for k, name in enumerate(sorted(ins), start=1):
            body = re.sub(r"\bIN%d\b" % k, name, body)
        models[cell] = dict(ins=ins, outs=outs, body=body, source=m["source"])
    else:
        models[cell] = None

missing = [c for c, m in models.items() if not m]
print("modelled:", len([c for c, m in models.items() if m]), "missing:", missing)

# ---------------- emit Verilog -------------
lines = ["// Functional models for Trust-Hub Nangate-original cells.",
         "// Combinational models are derived from the nangate45 Liberty `function`",
         "// strings; sequential / compound-only cells are written explicitly.",
         "`timescale 1ns/1ps", ""]
for cell in sorted(models):
    m = models[cell]
    if not m:
        continue
    ports = m["ins"] + m["outs"]
    seq = "always" in m["body"]
    regd = {"Q"} if seq else set()
    dirs = ["input " + i for i in m["ins"]] +            [("output reg " if o in regd else "output ") + o for o in m["outs"]]
    lines.append("module %s(%s);" % (cell, ", ".join(ports)))
    lines.append("  " + "; ".join(dirs) + ";")
    lines.append("  " + m["body"])
    lines.append("endmodule")
    lines.append("")

open(os.path.join(OUT, "stub_models.v"), "w", encoding="utf-8").write("\n".join(lines))
json.dump({c: (m["source"] if m else None) for c, m in models.items()},
          open(os.path.join(OUT, "cell_models_report.json"), "w", encoding="utf-8"), indent=2)
print("wrote", os.path.join(OUT, "stub_models.v"))

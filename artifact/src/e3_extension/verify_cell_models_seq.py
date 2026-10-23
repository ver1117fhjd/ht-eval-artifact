"""Sequential-cell behaviour check for the generated models (iverilog)."""
import os
import subprocess
import tempfile

OUT = r"D:\10.7\experiments\e3x"
IVERILOG = r"E:\oss-cad-suite\bin\iverilog.exe"
VVP = r"E:\oss-cad-suite\bin\vvp.exe"
MODELS = os.path.join(OUT, "stub_models.v")

CASES = {
    "DFFX2": dict(
        ports=["CLK", "D", "Q", "QN"],
        ins=["CLK", "D"],
        stim=["CLK=0; D=1; #1; $show;",
              "CLK=1; #1; $show;",
              "D=0; #1; $show;",
              "CLK=0; #1; $show;",
              "CLK=1; #1; $show;"],
        expect=[("x", "x"), ("1", "0"), ("1", "0"), ("1", "0"), ("0", "1")]),
    "DFFNX2": dict(
        ports=["CLK", "D", "Q"],
        ins=["CLK", "D"],
        stim=["CLK=1; D=1; #1; $show;",
              "CLK=0; #1; $show;",
              "D=0; #1; $show;",
              "CLK=1; #1; $show;",
              "CLK=0; #1; $show;"],
        expect=[("x",), ("1",), ("1",), ("1",), ("0",)]),
    "DFFARX1": dict(
        ports=["CLK", "D", "Q", "QN", "RSTB"],
        ins=["CLK", "D", "RSTB"],
        stim=["RSTB=0; CLK=0; D=1; #1; $show;",
              "RSTB=1; #1; $show;",
              "CLK=1; #1; $show;",
              "RSTB=0; #1; $show;"],
        expect=[("0", "1"), ("0", "1"), ("1", "0"), ("0", "1")]),
    "LARX1": dict(
        ports=["CLK", "D", "Q", "QN", "RSTB"],
        ins=["CLK", "D", "RSTB"],
        stim=["RSTB=0; D=1; #1; $show;",
              "RSTB=1; #1; $show;",
              "D=0; #1; $show;",
              "RSTB=0; #1; $show;"],
        expect=[("0", "1"), ("1", "0"), ("0", "1"), ("0", "1")]),
    "SDFFX1": dict(
        ports=["CLK", "D", "Q", "QN", "SE", "SI"],
        ins=["CLK", "D", "SE", "SI"],
        stim=["CLK=0; D=1; SE=0; SI=0; #1; CLK=1; #1; $show;",
              "CLK=0; D=0; SE=0; SI=1; #1; CLK=1; #1; $show;",
              "CLK=0; D=0; SE=1; SI=1; #1; CLK=1; #1; $show;",
              "CLK=0; SE=1; SI=0; #1; CLK=1; #1; $show;"],
        expect=[("1", "0"), ("0", "1"), ("1", "0"), ("0", "1")]),
    "SDFFARX1": dict(
        ports=["CLK", "D", "Q", "QN", "RSTB", "SE", "SI"],
        ins=["CLK", "D", "RSTB", "SE", "SI"],
        stim=["RSTB=0; CLK=0; D=1; SE=0; SI=1; #1; $show;",
              "RSTB=1; CLK=1; #1; $show;",
              "RSTB=0; #1; $show;"],
        expect=[("0", "1"), ("1", "0"), ("0", "1")]),
    "LSDNX1": dict(
        ports=["D", "Q"],
        ins=["D"],
        stim=["D=1; #1; $show;", "D=0; #1; $show;"],
        expect=[("1",), ("0",)]),
    "LSDNENX1": dict(
        ports=["D", "ENB", "Q"],
        ins=["D", "ENB"],
        stim=["ENB=0; D=1; #1; $show;",
              "D=0; #1; $show;",
              "ENB=1; D=1; #1; $show;",
              "D=0; #1; $show;",
              "ENB=0; #1; $show;"],
        expect=[("1",), ("0",), ("0",), ("0",), ("0",)]),
}

ok_all = True
for cell, c in CASES.items():
    decl = "\n".join("reg %s;" % i for i in c["ins"])
    outs = [p for p in c["ports"] if p not in c["ins"]]
    body = ["initial begin"]
    for st in c["stim"]:
        body.append("  " + st.replace("$show",
                                      '$display("OUT %s", %s)' % (
                                          " ".join("%b" for _ in outs), ", ".join(outs))))
    body += ["  $finish;", "end"]
    tb = ("`timescale 1ns/1ps\nmodule tb;\n" + decl + "\n" +
          cell + " dut(" + ", ".join("." + p + "(" + p + ")" for p in c["ports"]) + ");\n" +
          "\n".join(body) + "\nendmodule\n")
    with tempfile.TemporaryDirectory() as td:
        vf = os.path.join(td, "tb.v")
        open(vf, "w", encoding="utf-8").write(tb)
        r = subprocess.run([IVERILOG, "-g2012", "-o", os.path.join(td, "a.out"), vf, MODELS],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(cell, "COMPILE FAIL", (r.stderr or r.stdout).replace("\n", " ")[:200])
            ok_all = False
            continue
        s = subprocess.run([VVP, os.path.join(td, "a.out")], capture_output=True, text=True)
        got = [tuple(l.split()[1:]) for l in s.stdout.splitlines() if l.startswith("OUT")]
    exp = [tuple(x) for x in c["expect"]]
    good = got == exp
    ok_all &= good
    print(f"{cell:10} {'PASS' if good else 'FAIL'}  got={got}  exp={exp}")

print("ALL PASS" if ok_all else "SOME FAILED")

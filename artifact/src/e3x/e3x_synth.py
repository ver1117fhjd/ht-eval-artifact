"""Synthesise one (design, library) pair from the pre-cleaned generic workdir,
using functional cell models instead of blackbox stubs.

Usage: python e3x_synth.py <family> <design_dir> <tech> [--out-root DIR]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = r"D:\10.7"
WORK = os.path.join(ROOT, "experiments", "synth_work")
E3X = os.path.join(ROOT, "experiments", "e3x")
MODELS = os.path.join(E3X, "stub_models.v")
YOSYS = r"E:\oss-cad-suite\bin\yosys.exe"

sys.path.insert(0, os.path.join(ROOT, "paper", "host2027", "artifact", "src"))
from run_yosys_synth import extract  # noqa: E402

LIBS = {
    "nangate45": (r"D:\10.7\libs\liberty\nangate45\NangateOpenCellLibrary_typical.lib", None),
    "sky130hd": (r"D:\10.7\libs\liberty\sky130hd\sky130_fd_sc_hd__tt_025C_1v80.lib", None),
    "asap7": (r"D:\10.7\libs\liberty\asap7\asap7_RVT_TT_merged.lib",
              r"D:\10.7\libs\liberty\asap7\asap7_RVT_TT_comb.lib"),
    # GF180MCU 7t5v0 TT（源单元用 stub_models.v 的功能模型，与目标库无关，
    # 所以换库不需要新建单元模型）
    "gf180": (r"D:\10.7\libs\liberty\gf180\gf180mcu_fd_sc_mcu7t5v0__tt_025C_1v80.lib", None),
}


def run(family, design, tech, out_root=None):
    src = os.path.join(WORK, family, design)
    out_root = out_root or os.path.join(E3X, "synth")
    outdir = os.path.join(out_root, family, design + "__" + tech)
    fj = os.path.join(outdir, "features.json")
    if os.path.isfile(fj):
        print("SKIP (exists)", outdir)
        return json.load(open(fj, encoding="utf-8"))
    os.makedirs(outdir, exist_ok=True)

    vfiles = []
    for fn in sorted(os.listdir(src)):
        if os.path.splitext(fn)[1].lower() in (".v", ".sv") and fn != "blackbox_stubs.v":
            shutil.copy(os.path.join(src, fn), os.path.join(outdir, fn))
            vfiles.append(fn)
    lib, lib_abc = LIBS[tech]
    wd = outdir.replace("\\", "/")
    sfile = os.path.join(outdir, "synth.ys")
    lines = ["read_verilog -sv %s" % MODELS.replace("\\", "/"),
             "read_verilog -I%s %s" % (wd, " ".join("%s/%s" % (wd, v) for v in vfiles)),
             "hierarchy -auto-top",
             "synth",
             "dfflibmap -liberty %s" % lib.replace("\\", "/"),
             "abc -liberty %s" % (lib_abc or lib).replace("\\", "/"),
             "opt", "flatten", "opt",
             "write_json %s/netlist.json" % wd,
             "stat"]
    open(sfile, "w", encoding="utf-8").write("\n".join(lines) + "\n")

    t0 = time.time()
    r = subprocess.run([YOSYS, "-s", sfile], capture_output=True, text=True, timeout=7200)
    dt = time.time() - t0
    open(os.path.join(outdir, "yosys.log"), "w", encoding="utf-8").write(
        r.stdout + "\n" + r.stderr)

    jpath = os.path.join(outdir, "netlist.json")
    if not os.path.isfile(jpath):
        print("FAIL", family, design, tech, "%.1fs" % dt,
              (r.stderr or r.stdout)[-200:].replace("\n", " "))
        return None
    feats = {m: f for m, f in extract(jpath).items() if f.get("n_cells", 0) > 0}
    payload = {"label": design, "tech": tech, "lib": lib,
               "models": MODELS, "features": feats,
               "runtime_s": round(dt, 1)}
    json.dump(payload, open(fj, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    for mod, f in feats.items():
        print("OK", family, design, tech, "%.1fs" % dt,
              "cells=%s seq=%s unk=%.4f" % (f.get("n_cells"), f.get("n_seq"),
                                            f.get("unk_cell_frac", -1)))
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("family")
    ap.add_argument("design")
    ap.add_argument("tech")
    ap.add_argument("--out-root", default=None)
    a = ap.parse_args()
    run(a.family, a.design, a.tech, a.out_root)

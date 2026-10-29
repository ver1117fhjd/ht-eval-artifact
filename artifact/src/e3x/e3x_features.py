"""Build v1+v2 feature rows for the E3 extension from e3x synthesis outputs.

Mirrors c2_normalize.job_rows: module_features() returns both the v1 cell-type
features and the v2 operator-profile features, so one row carries both.
"""
import argparse
import csv
import json
import os
import sys

ROOT = r"PROJECT_ROOT"
SRC = os.path.join(ROOT, "paper", "host2027", "artifact", "src")
sys.path.insert(0, SRC)
from netlist_features import module_features  # noqa: E402
from lib_ops import load_lib_ops, generic_cell_ops  # noqa: E402
from batch_synth import flatten_features  # noqa: E402

E3X = os.path.join(ROOT, "experiments", "e3x")
SYNTH = os.path.join(E3X, "synth")
OUT_CSV = os.path.join(E3X, "features_multilib_ext.csv")
LIBS = {
    "nangate45": r"PROJECT_ROOT\libs\liberty\nangate45\NangateOpenCellLibrary_typical.lib",
    "sky130hd": r"PROJECT_ROOT\libs\liberty\sky130hd\sky130_fd_sc_hd__tt_025C_1v80.lib",
    "asap7": r"PROJECT_ROOT\libs\liberty\asap7\asap7_RVT_TT_merged.lib",
    "gf180": r"PROJECT_ROOT\libs\liberty\gf180\gf180mcu_fd_sc_mcu7t5v0__tt_025C_1v80.lib",
}
META = ["family", "variant", "source", "tech", "module"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_CSV)
    a = ap.parse_args()

    rows = []
    ops_cache = {}
    for fam in sorted(os.listdir(SYNTH)):
        fdir = os.path.join(SYNTH, fam)
        if not os.path.isdir(fdir):
            continue
        for d in sorted(os.listdir(fdir)):
            odir = os.path.join(fdir, d)
            fj, nj = os.path.join(odir, "features.json"), os.path.join(odir, "netlist.json")
            if not (os.path.isfile(fj) and os.path.isfile(nj)):
                continue
            meta = json.load(open(fj, encoding="utf-8"))
            tech = meta["tech"]
            design = meta["label"]                       # e.g. s15850-T100__TjFree
            variant, label = design.split("__")
            if tech not in ops_cache:
                ops = dict(generic_cell_ops())
                ops.update(load_lib_ops(LIBS[tech]))
                ops_cache[tech] = ops
            ops = ops_cache[tech]
            netlist = json.load(open(nj, encoding="utf-8"))
            for mod in meta.get("features", {}):
                m = netlist.get("modules", {}).get(mod)
                if m is None:
                    continue
                feat = module_features(m, ops_map=ops)
                row = flatten_features({mod: feat})
                row.update({"family": fam, "variant": variant, "source": label,
                            "tech": tech, "module": mod})
                rows.append(row)
                print("row", fam, variant, label, tech, mod,
                      "n_cells=%s seq=%s unk=%.4f" % (feat.get("n_cells"), feat.get("n_seq"),
                                                      feat.get("unk_cell_frac", -1)))
                break   # top module only, matching the 99-design table convention

    cols = META + sorted({k for r in rows for k in r if k not in META})
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print("wrote", a.out, len(rows), "rows")


if __name__ == "__main__":
    main()

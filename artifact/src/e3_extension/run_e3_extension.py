"""Batch runner for the E3 extension: 30 netlists x 3 libraries."""
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e3x_synth import run  # noqa: E402

E3X = r"PROJECT_ROOT\experiments\e3x"
LOG = os.path.join(E3X, "batch.log")

JOBS = []
for fam, pre, vs in [
    ("S15850", "s15850-T{}", ["100"]),
    ("S35932", "s35932-T{}", ["100", "200", "300"]),
    ("S38417", "s38417-T{}", ["100", "200", "300"]),
    ("S38584", "s38584-T{}", ["100", "200", "300"]),
    ("B15", "B15-T{}", ["100", "200", "300", "400"]),
    ("VGA_LCD", "vga_lcd-T{}", ["100"]),
]:
    for v in vs:
        for lab in ("TjFree", "TjIn"):
            JOBS.append((fam, pre.format(v) + "__" + lab))

# smallest first; VGA_LCD (69k cells) last
order = {"S15850": 0, "S35932": 1, "S38417": 2, "S38584": 3, "B15": 4, "VGA_LCD": 9}
JOBS.sort(key=lambda j: order[j[0]])

TECHS = ["nangate45", "sky130hd", "asap7"]
total = len(JOBS) * len(TECHS)
done = failed = 0
t0 = time.time()


def log(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


log("START %d jobs (%d designs x %d libs)" % (total, len(JOBS), len(TECHS)))
for fam, design in JOBS:
    for tech in TECHS:
        try:
            res = run(fam, design, tech)
        except Exception:
            res = None
            log("EXC %s %s %s: %s" % (fam, design, tech, traceback.format_exc()[-400:]))
        if res is None:
            failed += 1
            log("FAIL %s/%s @%s" % (fam, design, tech))
        else:
            done += 1
            cells = list(res["features"].values())[0].get("n_cells")
            log("OK %s/%s @%s cells=%s %.0fs (%d/%d, fail=%d)" % (
                fam, design, tech, cells, res["runtime_s"], done + failed, total, failed))
log("DONE done=%d failed=%d elapsed=%.1f min" % (done, failed, (time.time() - t0) / 60))

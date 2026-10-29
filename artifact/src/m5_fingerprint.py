# -*- coding: utf-8 -*-
"""
Golden-class degeneracy: fine-grained fingerprint re-check (M5 audit).

For every golden row of the E1 paired set, compute four fingerprint families:
  (a) the 32-D v1+v2 feature vector (results/features_all_v2.csv),
  (b) structural scalars (n_cells, n_seq, depth, widths),
  (c) the exact cell-type multiset from the synthesized netlist JSON,
  (d) a netlist-level hash (cell multiset + port widths).

Claim checked (Sec. III-B): the golden rows collapse to a handful of unique
fingerprints at EVERY granularity -- the degeneracy is a corpus property
(repeated golden netlists), not a property of coarse features.

Reproducibility tier (ii): needs the Yosys synthesis workdirs (one directory per
run, containing netlist.json) from the raw-synthesis stage. Set:
  M5_FEATURES     path to features_all_v2.csv   (default: results/features_all_v2.csv)
  M5_SYNTH_WORK   path to the synth_work tree   (REQUIRED: family/variant__source/netlist.json)
  M5_OUT          output directory              (default: results)

Falsifier (pre-registered): if the finest fingerprint yields >20 unique vectors
among the 60 rows, the "corpus property" claim must be weakened; <=15 keeps it.
"""
import csv, hashlib, json, os, sys
from collections import Counter

CSV = os.environ.get("M5_FEATURES", os.path.join("results", "features_all_v2.csv"))
ROOT = os.environ.get("M5_SYNTH_WORK", "")
OUT = os.environ.get("M5_OUT", "results")
if not ROOT or not os.path.isdir(ROOT):
    sys.exit("M5_SYNTH_WORK not set or not a directory (need synthesis workdirs)")

rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
print("total rows:", len(rows))

TRIT = ("TRIT-TC", "TRIT-TS", "TRIT")
e1 = [r for r in rows if r["family"] not in TRIT]
golden = [r for r in e1 if r["source"] == "TjFree"]
print("E1 rows:", len(e1), " golden rows:", len(golden))
assert len(golden) == 60, "golden rows != 60"

cols32 = [c for c in rows[0].keys() if c not in
          ("family", "variant", "source", "module", "label", "base_design", "tech",
           "n_blackbox", "notes")]
print("feature cols:", len(cols32))


def fine_fp(path, module):
    d = json.load(open(path, encoding="utf-8"))
    mods = d["modules"]
    m = mods.get(module) or mods[list(mods)[0]]
    cnt = Counter(c["type"] for c in m.get("cells", {}).values())
    ports = tuple(sorted((p, len(v["bits"])) for p, v in m.get("ports", {}).items()))
    key = tuple(sorted(cnt.items())) + (("PORTS", ports),)
    h = hashlib.md5(repr(key).encode()).hexdigest()[:12]
    return cnt, h, sum(cnt.values())


def uniq_count(vals):
    return len(set(vals))


by_key = {}
csv_rows = []
for r in golden:
    fam, var, src, mod = r["family"], r["variant"], r["source"], r["module"]
    cand = [os.path.join(ROOT, fam, f"{var}__{src}", "netlist.json"),
            os.path.join(ROOT, fam, var, "netlist.json"),
            os.path.join(ROOT, fam, f"{var}__{src}__generic", "netlist.json")]
    path = next((p for p in cand if os.path.exists(p)), None)
    v32 = tuple(r[c] for c in cols32)
    struct = (r.get("n_cells", ""), r.get("n_seq", ""), r.get("comb_depth_max", ""),
              r.get("in_width", ""), r.get("out_width", ""))
    if path is None:
        print("UNMAPPED:", fam, var, src)
        continue
    cnt, h, ncell = fine_fp(path, mod)
    csv_cells = int(float(r["n_cells"]))
    assert abs(ncell - csv_cells) <= max(5, 0.02 * csv_cells), \
        f"cell count mismatch {fam}/{var}: json={ncell} csv={csv_cells}"
    by_key[(fam, var, src)] = (v32, struct, tuple(sorted(cnt.items())), h)
    csv_rows.append((fam, var, src, h, ncell))

n = len(by_key)
aes = {k: v for k, v in by_key.items() if k[0] == "AES"}
print(f"mapped: {n}  (AES subset: {len(aes)})")

lines = []
for name, idx in (("32-dim v1+v2", 0), ("struct scalars", 1),
                  ("fine cell-type multiset", 2), ("netlist hash", 3)):
    u_all = uniq_count(v[idx] for v in by_key.values())
    u_aes = uniq_count(v[idx] for v in aes.values())
    lines.append((name, u_all, u_aes))
    print(f"{name:26s} unique among {n} golden rows = {u_all:3d}   (AES 27 rows = {u_aes})")

fine_u = lines[2][1]
verdict = ("PASS <=15: collapse is a corpus property" if fine_u <= 15 else
           ("FAIL >20: claim must be weakened" if fine_u > 20 else
            "BORDERLINE 16-20: report counts and weaken"))
print("VERDICT:", verdict)

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "m5_fingerprint.txt"), "w", encoding="utf-8") as f:
    f.write("M5 strong-feature degeneracy re-check (golden rows of E1 paired set)\n")
    f.write("source: features_all_v2.csv (golden-class corrected) + per-run netlist.json\n")
    f.write(f"golden rows mapped: {n}\n\n")
    for name, u_all, u_aes in lines:
        f.write(f"{name:26s} unique: {u_all:3d} / {n}   (AES: {u_aes} / {len(aes)})\n")
    f.write("\nVERDICT: " + verdict + "\n")
    f.write("\nper-row hashe<local-path>")
    for fam, var, src, h, nc in sorted(csv_rows):
        f.write(f"{fam},{var},{src},{h},{nc}\n")
print("written:", os.path.join(OUT, "m5_fingerprint.txt"))

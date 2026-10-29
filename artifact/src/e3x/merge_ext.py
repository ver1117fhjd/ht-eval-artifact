"""Merge the original 264-row multilib table with the E3-extension rows, and
report column compatibility before analysis."""
import pandas as pd
from pathlib import Path

ROOT = Path(r"PROJECT_ROOT")
ORIG = ROOT / "experiments" / "synth_work" / "features_multilib_v2.csv"
EXT = ROOT / "experiments" / "e3x" / "features_multilib_ext.csv"
OUT = ROOT / "experiments" / "e3x" / "features_multilib_v2_ext.csv"

a = pd.read_csv(ORIG)
b = pd.read_csv(EXT)
print("original", a.shape, "extension", b.shape)

feat_a = [c for c in a.columns if c not in ("family", "variant", "source", "tech", "module")]
feat_b = [c for c in b.columns if c not in ("family", "variant", "source", "tech", "module")]
only_a = sorted(set(feat_a) - set(feat_b))
only_b = sorted(set(feat_b) - set(feat_a))
print("features only in original:", only_a)
print("features only in extension:", only_b)

# Use the original feature set; missing columns in the extension must be zero-safe
for c in only_a:
    b[c] = 0.0
cols = list(a.columns) + [c for c in only_b if c not in a.columns]
a = a.reindex(columns=cols, fill_value=0.0)
b = b.reindex(columns=cols, fill_value=0.0)
m = pd.concat([a, b], ignore_index=True)

print("merged", m.shape)
print("tech counts:", m["tech"].value_counts().to_dict())
print("families:", sorted(m["family"].unique()))

def base(v):
    import re
    return re.sub(r"[-_]T\d+$", "", str(v))

m["_base"] = m["variant"].map(base)
print("base designs:", m["_base"].nunique(), sorted(m["_base"].unique()))
print("rows per base x tech:")
print(m.groupby(["_base", "tech"]).size().to_string())
m.drop(columns=["_base"]).to_csv(OUT, index=False)
print("wrote", OUT)

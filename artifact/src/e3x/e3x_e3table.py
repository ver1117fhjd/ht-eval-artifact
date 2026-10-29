"""Rebuild the paper's E3 table (v1 AUC / v2 AUC / v2 Spec / v2 Bal, RF/XGB)
on the EXTENDED cross-library set (11 base designs), matching the original
c1_protocol E3 convention: LOLO per held-out library + leaky pooled 5-fold.

v1 = the 23 cell-type features; v2 = all feature columns (v1 superset, the
project's existing convention)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier
import re

ROOT = Path(r"PROJECT_ROOT")
import os
EXT = Path(os.environ.get("E3_EXT",
           ROOT / "experiments" / "e3x" / "features_multilib_v2_ext.csv"))
OUT = ROOT / "experiments" / "e3x"
SEED = 20261012
META = {"family", "variant", "source", "tech", "module"}
V1 = ["cell_class_entropy", "comb_depth_max", "comb_depth_mean", "const_bit_frac",
      "fanin_max", "fanin_mean", "fanin_std", "fanout_max", "fanout_mean", "fanout_std",
      "frac_and", "frac_buf", "frac_complex", "frac_inv", "frac_mux", "frac_or",
      "frac_other", "frac_seq", "frac_xor", "in_width", "n_cells", "n_seq", "out_width"]


def base(v):
    v = re.sub(r"__Tj(In|Free)$", "", str(v))
    return re.sub(r"[-_]T\d+$", "", v).lower()


def models():
    return {
        "RF": RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                     random_state=SEED, n_jobs=1),
        "XGB": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                             subsample=0.9, colsample_bytree=0.9, random_state=SEED,
                             n_jobs=1, eval_metric="logloss"),
    }


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    neg, pos = y == 0, y == 1
    sens = float((pred[pos] == 1).mean()) if pos.any() else float("nan")
    spec = float((pred[neg] == 0).mean()) if neg.any() else float("nan")
    bal = (sens + spec) / 2 if not (np.isnan(sens) or np.isnan(spec)) else float("nan")
    auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    return auc, sens, spec, bal


def main():
    df = pd.read_csv(EXT)
    src = df["source"].replace({"TnIn": "TjIn"})
    m = src.isin(["TjIn", "TjFree"])
    df = df[m].copy()
    df["label"] = (src[m] == "TjIn").astype(int)
    df["group"] = df["variant"].map(base)

    feat_all = [c for c in df.columns if c not in META | {"label", "group"}]
    feat_v1 = [c for c in V1 if c in df.columns]
    techs = sorted(df["tech"].unique())
    rows = []
    for fname, cols in (("v1", feat_v1), ("v2", feat_all)):
        X, y, tech = df[cols].values, df["label"].values, df["tech"].values
        for mname, mdl in models().items():
            for held in techs:
                tr, te = tech != held, tech == held
                mdl.fit(X[tr], y[tr])
                p = mdl.predict_proba(X[te])[:, 1]
                auc, sens, spec, bal = metrics(y[te], p)
                rows.append(dict(feat=fname, model=mname, held=held, n=int(te.sum()),
                                 auc=auc, sens=sens, spec=spec, bal=bal))
            p = cross_val_predict(mdl, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                                  method="predict_proba")[:, 1]
            auc, sens, spec, bal = metrics(y, p)
            rows.append(dict(feat=fname, model=mname, held="pooled", n=len(y),
                             auc=auc, sens=sens, spec=spec, bal=bal))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "e3x_e3table.csv", index=False)
    print("n=%d groups=%d techs=%s" % (len(df), df["group"].nunique(), techs))
    print(res.to_string(index=False))
    # paper-format: v1 AUC / v2 AUC / v2 Spec / v2 Bal  (RF/XGB)
    print("\n=== paper format (RF/XGB) ===")
    for held in ["asap7", "nangate45", "sky130hd", "pooled"]:
        vals = {}
        for feat in ("v1", "v2"):
            for mdl in ("RF", "XGB"):
                r = res[(res.feat == feat) & (res.model == mdl) & (res.held == held)]
                vals[(feat, mdl)] = r.iloc[0]["auc"]
        spec = {mdl: res[(res.feat == "v2") & (res.model == mdl) & (res.held == held)].iloc[0]["spec"]
                for mdl in ("RF", "XGB")}
        bal = {mdl: res[(res.feat == "v2") & (res.model == mdl) & (res.held == held)].iloc[0]["bal"]
               for mdl in ("RF", "XGB")}
        print("%-10s v1AUC %.3f/%.3f  v2AUC %.3f/%.3f  v2Spec %.2f/%.2f  v2Bal %.2f/%.2f" % (
            held, vals[("v1", "RF")], vals[("v1", "XGB")], vals[("v2", "RF")], vals[("v2", "XGB")],
            spec["RF"], spec["XGB"], bal["RF"], bal["XGB"]))


if __name__ == "__main__":
    sys.exit(main())

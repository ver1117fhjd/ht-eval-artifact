"""Exact sign-flip test for the EXTENDED cross-library set (11 base designs).

Same procedure as r7_e3_signflip.py, but on features_multilib_v2_ext.csv and with
2^11 = 2048 sign patterns.
"""
import itertools
import sys
from pathlib import Path

ROOT = Path(r"D:\10.7")
SRC = ROOT / "paper" / "host2027" / "artifact" / "src"
sys.path.insert(0, str(SRC))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import c1_protocol as c1

import os
EXT = Path(os.environ.get("E3_EXT",
           ROOT / "experiments" / "e3x" / "features_multilib_v2_ext.csv"))
c1.MULTILIB_CSV = str(EXT)
OUT = ROOT / "experiments" / "e3x"
SEED = c1.SEED


def sign_flip_p(deltas):
    d = np.asarray(deltas, float)
    obs = abs(d.mean())
    cnt = tot = 0
    for signs in itertools.product((1.0, -1.0), repeat=len(d)):
        tot += 1
        if abs(np.mean(d * np.array(signs))) >= obs - 1e-12:
            cnt += 1
    return cnt / tot


def make_models1():
    ms = c1.make_models()
    for m in ms.values():
        try:
            m.set_params(n_jobs=1)
        except ValueError:
            pass
    return ms


def main():
    df = c1.load_multilib().reset_index(drop=True)
    X = df[c1.feature_cols(df)].values
    y = df["label"].values
    g = df["group"].values
    techs = df["tech"].values
    groups = sorted(set(g))
    lines = ["E3 EXTENDED: exact sign-flip test + per-group table",
             f"source: {EXT}  seed={SEED}  n_jobs=1",
             f"n={len(df)} pos={int(y.sum())} groups={len(groups)} techs={sorted(set(techs))}",
             ""]
    per_group, sig = [], []
    for mname, model in make_models1().items():
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
        prob_pool = cross_val_predict(model, X, y, cv=skf, method="predict_proba")[:, 1]
        prob_lolo = np.zeros(len(y))
        lolo_specs = {}
        for held in sorted(set(techs)):
            tr = techs != held
            te = ~tr
            model.fit(X[tr], y[tr])
            prob_lolo[te] = model.predict_proba(X[te])[:, 1]
            yh = (prob_lolo[te] >= 0.5).astype(int)
            neg = y[te] == 0
            lolo_specs[held] = float((yh[neg] == 0).mean()) if neg.any() else float("nan")
        deltas = []
        lines.append(f"==== {mname} ====")
        for grp in groups:
            m = g == grp
            both = len(set(y[m])) == 2
            auc_p = roc_auc_score(y[m], prob_pool[m]) if both else float("nan")
            auc_l = roc_auc_score(y[m], prob_lolo[m]) if both else float("nan")
            yh = (prob_lolo[m] >= 0.5).astype(int)
            neg, pos = y[m] == 0, y[m] == 1
            spec = float((yh[neg] == 0).mean()) if neg.any() else float("nan")
            sens = float((yh[pos] == 1).mean()) if pos.any() else float("nan")
            per_group.append({"model": mname, "group": grp, "n": int(m.sum()),
                              "n_pos": int(y[m].sum()), "n_neg": int((1 - y[m]).sum()),
                              "auc_pooled": auc_p, "auc_lolo": auc_l,
                              "spec_lolo@0.5": spec, "sens_lolo@0.5": sens})
            if both:
                deltas.append(auc_p - auc_l)
            lines.append(f"  {grp:14s} n={int(m.sum()):4d} (+{int(y[m].sum())}/-{int((1-y[m]).sum()):4d}) "
                         f"AUC pooled={auc_p:.3f} LOLO={auc_l:.3f} Spec@0.5={spec:.3f} Sens@0.5={sens:.3f}")
        d = np.array(deltas)
        p = sign_flip_p(d) if len(d) else float("nan")
        sig.append({"model": mname, "n_groups_with_both_classes": len(d),
                    "mean_delta_auc": float(d.mean()) if len(d) else float("nan"),
                    "signflip_p": p,
                    "lolo_specs": "; ".join(f"{k}:{v:.2f}" for k, v in lolo_specs.items())})
        lines.append(f"  per-group delta AUC(pooled-LOLO): " + ", ".join(f"{x:+.3f}" for x in d))
        lines.append(f"  exact sign-flip (2^{len(d)}={2**len(d)}) p = {p:.5f}")
        lines.append("")
    pd.DataFrame(per_group).to_csv(OUT / "e3x_per_group.csv", index=False)
    pd.DataFrame(sig).to_csv(OUT / "e3x_signflip.csv", index=False)
    txt = "\n".join(lines)
    (OUT / "e3x_signflip_report.txt").write_text(txt, encoding="utf-8")
    print(txt[-3000:])


if __name__ == "__main__":
    sys.exit(main())

"""Closure comparison (harness §5 verbatim) with a REAL reconvergence eps + the §5c capped-horizon
variant, per cell x delta x slice (PRE 08-27..31 | FRESH 09-01..04 | OOS 09-05..08).
  closure : first row after entry with |fair-mid| < close_eps (run-time eps); closed% = share of trades
            that reconverge before settle; close_s = median seconds to reconverge; moves signed by direction
  capped  : moves at min(closure, cap) — every trade measured, at the cap if it never closed
Usage: closure_cells2.py LABEL=trades.csv [...]   env MIN_N (30)"""
import os
import sys

import numpy as np
import pandas as pd

MIN_N = int(os.environ.get("MIN_N", "30"))
COLS = ["model", "delta", "ticker", "date", "side_yes", "gap", "net", "close_s", "close_dmid", "close_dfair",
        "cap_s", "cap_dmid", "cap_dfair"]


def moves(R, dm, df):
    s = np.where(R["side_yes"].to_numpy().astype(bool), 1.0, -1.0)
    mk = (100 * s * R[dm].to_numpy()).mean(); fr = (100 * s * R[df].to_numpy()).mean()
    return mk, fr, 100 * mk / (abs(mk) + abs(fr) + 1e-9)


def evt(R):
    e = R.groupby("ticker")["net"].mean()
    t = e.mean() / (e.std(ddof=1) / np.sqrt(len(e))) if len(e) > 3 and e.std() > 0 else 0.0
    return 100 * e.mean(), t


for arg in sys.argv[1:]:
    lab, path = arg.split("=", 1)
    df = pd.read_csv(path, usecols=COLS)
    if os.environ.get("SPLIT"):                     # SPLIT=YYYY-MM-DD -> two slices: before ("A") and from ("B") that date
        df["slice"] = np.where(df.date < os.environ["SPLIT"], "A", "B")
    else:
        df["slice"] = np.where(df.date < "2026-09-01", "PRE", np.where(df.date < "2026-09-05", "FRESH", "OOS"))
    print("=" * 160)
    print(lab)
    print("%-20s %-6s %-5s | %6s %6s | %-34s | %-34s | %s" % ("cell", "delta", "slice", "n", "gap0", "RECONVERGE(<0.1c): closed% med s  mkt  fair share",
                                                            "CAPPED@5s: closed<5s%  mkt  fair share", "net/tr (t)"))
    for (m, dl), g in df.groupby(["model", "delta"]):
        for sl in (("A", "B") if os.environ.get("SPLIT") else ("PRE", "FRESH", "OOS")):
            h = g[g["slice"] == sl]
            if len(h) < MIN_N:
                continue
            c = h[h["close_s"].notna()]
            g0 = 100 * h["gap"].mean()
            if len(c) >= 10:
                mk, fr, sh = moves(c, "close_dmid", "close_dfair")
                rec = "%5.0f%%  %6.1fs  %+5.2f %+5.2f %4.0f%%" % (100 * len(c) / len(h), c["close_s"].median(), mk, fr, sh)
            else:
                rec = "%5.0f%%  (n<10)" % (100 * len(c) / len(h))
            k = h[h["cap_s"].notna()]
            if len(k) >= 10:
                mk2, fr2, sh2 = moves(k, "cap_dmid", "cap_dfair")
                cap = "%5.0f%%  %+5.2f %+5.2f %4.0f%%" % (100 * (k["cap_s"] < 5.0).mean(), mk2, fr2, sh2)
            else:
                cap = "--"
            ev, t = evt(h)
            print("%-20s %-6.4f %-5s | %6d %+5.2fc | %-34s | %-34s | %+5.1fc (%+4.1f)" % (m, dl, sl, len(h), g0, rec, cap, ev, t))
        print()

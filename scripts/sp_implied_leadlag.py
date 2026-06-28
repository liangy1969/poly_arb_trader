#!/usr/bin/env python3
"""ES (CME) vs Kalshi-implied S&P level lead-lag.

Kalshi KXINXU is a strike ladder of P(close>K). The market's implied fair level is
the K where P=0.5, interpolated from the ladder. We reconstruct that level on a grid
and cross-correlate its log-returns against ES log-returns: peak lag < 0 => ES LEADS.

  py scripts/sp_implied_leadlag.py --grid 200ms --span-ms 2000
"""
import argparse, glob, json, re
import numpy as np
import pandas as pd


def _series(rows):
    s = pd.Series([v for _, v in rows], index=pd.to_datetime([t for t, _ in rows], utc=True))
    return s[~s.index.duplicated(keep="last")].sort_index()


def _dglob(dir, venue, date):
    if date:
        return glob.glob(f"{dir}/stream=book/venue={venue}/date={date}/events.jsonl")
    return glob.glob(f"{dir}/stream=book/venue={venue}/**/events.jsonl", recursive=True)


def load_future(dir, inst, date=""):
    rows = []
    for f in _dglob(dir, "databento", date):
        for ln in open(f, encoding="utf-8"):
            try:
                e = json.loads(ln); b = e["payload"]["Book"]
            except Exception:
                continue
            if b.get("instrument") != inst or not b["bids"] or not b["asks"]:
                continue
            rows.append((e["ts_ns"], (b["bids"][0][0] + b["asks"][0][0]) / 2))
    return _series(rows)


def load_strikes(dir, series, event="", date=""):
    d = {}
    for f in _dglob(dir, "kalshi", date):
        for ln in open(f, encoding="utf-8"):
            try:
                e = json.loads(ln); b = e["payload"]["Book"]
            except Exception:
                continue
            inst = b.get("instrument", "")
            if series not in inst or not inst.endswith(".YES") or not b["bids"] or not b["asks"]:
                continue
            if event and event not in inst:
                continue
            m = re.search(r"-T(\d+\.?\d*)", inst)
            if not m:
                continue
            d.setdefault(float(m.group(1)), []).append((e["ts_ns"], (b["bids"][0][0] + b["asks"][0][0]) / 2))
    return d


def implied_level(strikes, grid):
    ks = sorted(strikes)
    cols = {k: _series(strikes[k]).resample(grid).last() for k in ks}
    P = pd.DataFrame(cols).sort_index().ffill()
    K = np.array(ks, float)
    Pv = P.to_numpy()
    out = np.full(len(P), np.nan)
    order = np.argsort(K)
    Ks = K[order]
    for i in range(len(P)):
        pp = Pv[i][order]
        valid = ~np.isnan(pp)
        if valid.sum() < 2:
            continue
        kk, p = Ks[valid], pp[valid]            # strike asc; P should be desc
        above = np.where(p >= 0.5)[0]
        below = np.where(p < 0.5)[0]
        if len(above) == 0 or len(below) == 0:
            continue
        j = above[-1]
        if j + 1 >= len(kk):
            continue
        p0, p1, k0, k1 = p[j], p[j + 1], kk[j], kk[j + 1]
        out[i] = k0 if p0 == p1 else k0 + (k1 - k0) * (p0 - 0.5) / (p0 - p1)
    return pd.Series(out, index=P.index).interpolate(limit=3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/leadlag-cme-collect")
    ap.add_argument("--future", default="databento.ES", help="CME future instrument, e.g. databento.ES or databento.CL")
    ap.add_argument("--series", default="KXINXU", help="Kalshi series, e.g. KXINXU or KXWTI")
    ap.add_argument("--grid", default="200ms")
    ap.add_argument("--span-ms", type=int, default=2000)
    ap.add_argument("--event", default="", help="restrict to one settlement event, e.g. KXINXU-26JUN25H1300")
    ap.add_argument("--date", default="", help="restrict to one date partition, e.g. 2026-06-26")
    a = ap.parse_args()
    step = pd.Timedelta(a.grid).total_seconds() * 1000
    lags = list(range(-round(a.span_ms / step), round(a.span_ms / step) + 1))

    es = load_future(a.dir, a.future, a.date)
    strikes = load_strikes(a.dir, a.series, a.event, a.date)
    print(f"{a.future} vs {a.series}  event={a.event or 'ALL'}  fut ticks={len(es):,}  strikes={len(strikes)}  span={(es.index[-1]-es.index[0]).total_seconds()/3600:.2f}h  grid={a.grid}")
    L = implied_level(strikes, a.grid)
    eg = es.resample(a.grid).last().ffill()
    j = pd.concat({"es": eg, "L": L}, axis=1).dropna()
    print(f"joined grid points={len(j):,}  basis ES-L={(j['es']-j['L']).mean():+.1f} pts  L coverage={L.notna().mean()*100:.0f}%")

    er = np.log(j["es"]).diff().to_numpy()
    lr = np.log(j["L"]).diff().to_numpy()
    res = {}
    for Lg in lags:
        y = np.roll(lr, -Lg)
        x = er
        if Lg >= 0:
            x, y = x[1:len(x) - Lg if Lg else None], y[1:len(y) - Lg if Lg else None]
        else:
            x, y = x[1 - Lg:], y[1 - Lg:]
        m = ~(np.isnan(x) | np.isnan(y))
        x, y = x[m], y[m]
        if len(x) > 30 and x.std() > 0 and y.std() > 0:
            res[Lg] = float(np.corrcoef(x, y)[0, 1])
    if not res:
        print("insufficient overlap"); return
    pk = max(res, key=res.get)
    print(f"\npeak corr={res[pk]:+.3f} at lag={pk*step:+.0f}ms  (negative => ES leads Kalshi)")
    print("lag(ms): corr  —")
    for Lg in sorted(res):
        if abs(Lg * step) <= 1000:
            bar = "#" * int(abs(res[Lg]) * 60)
            print(f"  {Lg*step:+6.0f}: {res[Lg]:+.3f} {bar}")


if __name__ == "__main__":
    main()

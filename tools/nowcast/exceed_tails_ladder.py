"""Tails round trip (entry mid<0.10 or >0.90, 1 s classifier, top-q |s| cut, re-arm, entry at the ask/1-bid + fee, exit by
crossing the spread at 1 s + fee) with the entry filled at the quotes 0/50/100/150 ms after the signal row (from the 50 ms
sampler) and 200 ms (dataset). Cost decomposition + per-trade table. Usage: exceed_tails_diag2.py X:q [X:q ...]"""
import sys, os
COMBOS = [(c.split(":")[0], float(c.split(":")[1])) for c in (sys.argv[1:] or ["1:0.02"])]
import numpy as np, pandas as pd
SP = r"C:\Users\fatli\AppData\Local\Temp\claude\e--poly-crypto-trader\0ed64f57-c300-45f3-b675-113fb239783c\scratchpad"
exec(open(os.path.join(SP, "exceed_strat_realistic.py")).read().split("VA = load_split")[0])
LAGS = [0, 50, 100, 150]


def load_split2(days, off):
    D = load_split(days, off); TS, NM = [], []
    for i, day in enumerate(days):
        d = np.load(os.path.join(SP, "midmove_v10", day + ".npz")); sel = (d["ctx"][:, 1] >= 60) & (d["ctx"][:, 1] <= 300)
        names = np.array([str(x) for x in d["names"]]); TS.append(d["ts"][sel]); NM.append(names[d["tk"][sel]])
    D["ts"] = np.concatenate(TS); D["name"] = np.concatenate(NM); return D


VA = load_split2(VAL_DAYS, 0); TE = load_split2(TEST_DAYS, 100)
SAMP = {}
for day in VAL_DAYS + TEST_DAYS:
    fn = [f for f in (f"data/samples/{day}.csv.gz", f"data/samples/{day}.csv") if os.path.exists(f)][0]
    S = pd.read_csv(fn, usecols=["ts_ms", "ticker", "ybid", "yask"], dtype={"ts_ms": np.int64, "ticker": str, "ybid": np.float64, "yask": np.float64}, on_bad_lines="skip")
    S = S[(S.yask > S.ybid)]; S = S.sort_values(["ticker", "ts_ms"])
    SAMP[day] = {tk: (g.ts_ms.to_numpy(), g.ybid.to_numpy(), g.yask.to_numpy()) for tk, g in S.groupby("ticker")}
    print("loaded", day, len(S), flush=True)


def quotes_at(day, name, ts, lag):
    g = SAMP[day].get(name)
    if g is None:
        return np.nan, np.nan
    t, b, a = g; j = np.searchsorted(t, ts + lag)
    if j >= len(t) or t[j] > ts + lag + 150:
        return np.nan, np.nan
    return b[j], a[j]


def trades(D, s, days, cut):
    tails = (D["mid"] < 0.10) | (D["mid"] > 0.90)
    idx = throttle((np.abs(s) >= cut) & tails, np.abs(s) < cut / 2, D, "rearm")
    side = np.where(s[idx] >= 0, 1, -1); day = np.array(days)[D["mkt"][idx] // 1000 - D["mkt"][idx].min() // 1000]
    mid0, b0, a0 = D["mid"][idx], D["bid"][idx], D["ask"][idx]
    m1, b1, a1 = D["fut"][idx, 1, 0], D["fut"][idx, 1, 1], D["fut"][idx, 1, 2]
    pout = np.where(side > 0, b1, 1 - a1); fout = FEE * pout * (1 - pout)
    v0 = np.where(side > 0, mid0, 1 - mid0); v1 = np.where(side > 0, m1, 1 - m1)
    ent = {}
    ent["row"] = np.where(side > 0, a0, 1 - b0)
    ent[200] = np.where(side > 0, D["fut"][idx, 0, 2], 1 - D["fut"][idx, 0, 1])
    for lag in LAGS:
        qb, qa = zip(*[quotes_at(day[i], D["name"][idx][i], D["ts"][idx][i], lag) for i in range(len(idx))])
        qb, qa = np.array(qb), np.array(qa); ent[lag] = np.where(side > 0, qa, 1 - qb)
    net = {k: 100 * (pout - fout - p - FEE * p * (1 - p)) for k, p in ent.items()}
    return dict(day=day, mkt=D["mkt"][idx] % 1000, tte=D["tte"][idx], side=side, s=s[idx], mid0=mid0, spr0=100 * (a0 - b0), gross=100 * (v1 - v0),
                half_in=100 * (ent["row"] - v0), fin=100 * FEE * ent["row"] * (1 - ent["row"]), half_out=100 * (v1 - pout), fout=100 * fout, ent=ent, net=net,
                settle=100 * (np.where(side > 0, D["settle"][idx], 1 - D["settle"][idx]) - ent["row"] - FEE * ent["row"] * (1 - ent["row"])))


def clus(v, mkt):
    ok = np.isfinite(v); v, mkt = v[ok], mkt[ok]
    if len(v) == 0:
        return np.nan, np.nan, 0
    um, inv = np.unique(mkt, return_inverse=True); m = np.bincount(inv, v) / np.bincount(inv)
    return v.mean(), (m.mean() / (m.std(ddof=1) / np.sqrt(len(m))) if len(m) > 3 and m.std() > 0 else np.nan), len(v)


for X, q in COMBOS:
    p = np.load(os.path.join(EX, f"std_mid5_x{X}_s0.npz")); sv = p["p_va"][:, 0] - p["p_va"][:, 1]; st = p["p_te"][:, 0] - p["p_te"][:, 1]
    cut = np.quantile(np.abs(sv), 1 - q)
    print("\n##### tails, 1 s classifier X=%sc, top %g%% (|s| >= %.3f), re-arm; taker exit at 1 s (+fee); entry at the ask/1-bid (+fee) filled at lag L after the signal row" % (X, 100 * q, cut))
    for name, D, s, days in (("VAL", VA, sv, VAL_DAYS), ("TEST", TE, st, TEST_DAYS)):
        T = trades(D, s, days, cut); ok = np.isfinite(T["net"]["row"]); mk = T["day"].astype(object) + T["mkt"].astype(str)
        d0 = T["ent"][0] - T["ent"]["row"]
        print("=== %s: %d trades; sampler-at-0ms vs row entry price: mean |diff| %.3fc (%d matched)" % (name, ok.sum(), 100 * np.nanmean(np.abs(d0)), np.isfinite(d0).sum()))
        print("  gross mid move 0->1 s %+.2fc | entry ask-mid %+.2fc + fee %.2fc | exit mid1-bid1 %+.2fc + fee %.2fc | spread at entry mean %.2fc median %.2fc" % (
            T["gross"][ok].mean(), T["half_in"][ok].mean(), T["fin"][ok].mean(), T["half_out"][ok].mean(), T["fout"][ok].mean(), T["spr0"][ok].mean(), np.median(T["spr0"][ok])))
        print("  %-10s | %-28s | %s" % ("entry lag", "NET c/trade (clustered t, n)", "entry slip vs row (mean c) | net>0"))
        for k in ["row", 0, 50, 100, 150, 200]:
            m, t, n = clus(T["net"][k], mk); slip = 100 * np.nanmean(T["ent"][k] - T["ent"]["row"]); pos = np.nanmean(T["net"][k] > 0)
            print("  %-10s | %+6.2fc  (t %+5.2f, n=%3d)      | %+5.2fc | %3.0f%%" % (str(k) + (" ms" if k != "row" else ""), m, t, n, slip, 100 * pos))
        m, t, n = clus(T["settle"], mk); print("  %-10s | %+6.2fc  (t %+5.2f, n=%3d)      | hold-to-settle, row entry (median %+.2fc)" % ("settle", m, t, n, np.nanmedian(T["settle"])))
        for d in days:
            mm = ok & (T["day"] == d)
            if mm.sum():
                print("    %s n=%3d  net row %+6.2fc  100ms %+6.2fc  200ms %+6.2fc  gross %+5.2fc" % (d, mm.sum(), T["net"]["row"][mm].mean(), np.nanmean(T["net"][100][mm]), np.nanmean(T["net"][200][mm]), T["gross"][mm].mean()))
        lo, hi = ok & (T["mid0"] < 0.1), ok & (T["mid0"] > 0.9)
        print("    low tail n=%d net row %+.2fc 100ms %+.2fc | high tail n=%d net row %+.2fc 100ms %+.2fc | YES n=%d row %+.2fc 100ms %+.2fc | NO n=%d row %+.2fc 100ms %+.2fc" % (
            lo.sum(), T["net"]["row"][lo].mean(), np.nanmean(T["net"][100][lo]), hi.sum(), T["net"]["row"][hi].mean(), np.nanmean(T["net"][100][hi]),
            (ok & (T["side"] > 0)).sum(), T["net"]["row"][ok & (T["side"] > 0)].mean(), np.nanmean(T["net"][100][ok & (T["side"] > 0)]), (ok & (T["side"] < 0)).sum(), T["net"]["row"][ok & (T["side"] < 0)].mean(), np.nanmean(T["net"][100][ok & (T["side"] < 0)])))
        if name == "TEST" and q <= 0.02:
            print("  per trade: day mkt tte side mid0 spr0 | entry px row/100ms/200ms | gross | net row/100ms/200ms | settle | s")
            for i in np.flatnonzero(ok):
                print("    %s %3d %4.0f %-3s %.3f %4.1f | %.3f %.3f %.3f | %+6.2f | %+6.2f %+6.2f %+6.2f | %+6.2f | %+.3f" % (
                    T["day"][i], T["mkt"][i], T["tte"][i], "YES" if T["side"][i] > 0 else "NO", T["mid0"][i], T["spr0"][i], T["ent"]["row"][i], T["ent"][100][i], T["ent"][200][i],
                    T["gross"][i], T["net"]["row"][i], T["net"][100][i], T["net"][200][i], T["settle"][i], T["s"][i]))

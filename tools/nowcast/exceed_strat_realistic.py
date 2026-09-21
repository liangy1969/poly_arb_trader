"""Realistic taker round trip: entry at ask/bid (row or 200 ms later), exit by CROSSING the spread at the
horizon (YES sold at bid_h, NO sold at 1-ask_h) and paying the taker fee again. Horizons with quotes: 1 s, 5 s, 30 s.
Also the maker-exit variant (exit at the horizon mid, no exit fee) with lat200 entry, for the ladder.
"""
import numpy as np, os, sys, time
sys.argv = [sys.argv[0]]
SP = r"C:\Users\fatli\AppData\Local\Temp\claude\e--poly-crypto-trader\0ed64f57-c300-45f3-b675-113fb239783c\scratchpad"
EX = r"E:\poly\crypto_trader\data\nowcast\exceed"
VAL_DAYS = ["2026-08-27","2026-08-28","2026-08-29","2026-08-30","2026-08-31"]
TEST_DAYS = ["2026-09-01","2026-09-02","2026-09-03","2026-09-04"]
HS = ["mid5","mid25","mid150"]; HLAB = {"mid5":"1s","mid25":"5s","mid150":"30s"}
HK = {"mid5":1,"mid25":2,"mid150":3}
XS = ["1","1.5","2","3","5"]; QS = [0.005,0.01,0.02,0.05,0.10]; FEE = 0.07

def load_split(days, off):
    out = {k:[] for k in ["mid","tte","spread","bid","ask","fut","settle","mkt"]}
    for i,day in enumerate(days):
        d = np.load(os.path.join(SP,"midmove_v10",day+".npz"))
        ctx = d["ctx"]; sel = (ctx[:,1]>=60)&(ctx[:,1]<=300)
        c = ctx[sel]
        out["mid"].append(c[:,0]); out["tte"].append(c[:,1]); out["spread"].append(c[:,3]); out["bid"].append(c[:,4]); out["ask"].append(c[:,5])
        out["fut"].append(d["fut"][sel].reshape(-1,4,3)); out["settle"].append(d["settle"][sel])
        out["mkt"].append((off+i)*1000 + d["tk"][sel].astype(np.int64))
    return {k:np.concatenate(v) for k,v in out.items()}

def throttle(q, r, D, mode):
    idx = np.flatnonzero(q)
    if mode == "all" or len(idx)==0: return idx
    if mode == "one":
        _, first = np.unique(D["mkt"][idx], return_index=True); return idx[np.sort(first)]
    mstart = np.ones(len(q), bool); mstart[1:] = D["mkt"][1:] != D["mkt"][:-1]
    seg = np.cumsum(r | mstart); _, first = np.unique(seg[idx], return_index=True); return idx[np.sort(first)]

def stats(net, mkt, side):
    ok = np.isfinite(net); net = net[ok]; mkt = mkt[ok]; side = side[ok]; n = len(net)
    if n == 0: return dict(n=0, nm=0, mean=np.nan, t=np.nan, yes=np.nan)
    um, inv = np.unique(mkt, return_inverse=True); m = np.bincount(inv, net)/np.bincount(inv); nm = len(um)
    t = m.mean()/(m.std(ddof=1)/np.sqrt(nm)) if nm>=3 and m.std(ddof=1)>0 else np.nan
    return dict(n=n, nm=nm, mean=net.mean(), t=t, yes=(side>0).mean())

def entry_price(D, idx, side, fill):
    if fill == "row": a = D["ask"][idx]; b = D["bid"][idx]
    else:             a = D["fut"][idx,0,2]; b = D["fut"][idx,0,1]      # quotes 200 ms later
    p = np.where(side>0, a, 1.0-b); return p, FEE*p*(1-p)

def exit_value(D, idx, side, h, kind):
    k = HK[h]
    if kind == "mid":     # maker-style exit at the horizon mid, no fee (optimistic)
        ev = D["fut"][idx,k,0]; return np.where(side>0, ev, 1.0-ev)
    if kind == "taker":   # cross the spread at the horizon, pay fee again
        b = D["fut"][idx,k,1]; a = D["fut"][idx,k,2]
        v = np.where(side>0, b, 1.0-a); return v - FEE*v*(1-v)
    if kind == "settle":
        s = D["settle"][idx]; return np.where(side>0, s, 1.0-s)

VA = load_split(VAL_DAYS,0); TE = load_split(TEST_DAYS,100)
lines = ["# Realistic ladder for the directional taker (std model). net cents/contract; t = event-clustered.",
         "# entry: row = ask/bid at the row; lat = ask/bid 200 ms later. exit: mid = horizon mid no fee (maker-out, optimistic); taker = horizon bid/ask + fee; settle.",
         ""]
recs = []
for h in HS:
    for X in XS:
        p = np.load(os.path.join(EX, f"std_{h}_x{X}_s0.npz"))
        sv = p["p_va"][:,0]-p["p_va"][:,1]; st = p["p_te"][:,0]-p["p_te"][:,1]
        for q in QS:
            cut = np.quantile(np.abs(sv), 1-q)
            for mode in ("one","rearm"):
                iv = throttle(np.abs(sv)>=cut, np.abs(sv)<cut/2, VA, mode); it = throttle(np.abs(st)>=cut, np.abs(st)<cut/2, TE, mode)
                sidev = np.where(sv[iv]>=0,1,-1); sidet = np.where(st[it]>=0,1,-1)
                for fill in ("row","lat"):
                    pv_, fv_ = entry_price(VA, iv, sidev, fill); pt_, ft_ = entry_price(TE, it, sidet, fill)
                    for ek in ("mid","taker","settle"):
                        nv = (exit_value(VA, iv, sidev, h, ek) - pv_ - fv_)*100; nt = (exit_value(TE, it, sidet, h, ek) - pt_ - ft_)*100
                        a = stats(nv, VA["mkt"][iv], sidev); b = stats(nt, TE["mkt"][it], sidet)
                        recs.append(dict(h=h,X=X,q=q,cut=cut,mode=mode,fill=fill,exit=ek,v=a,t=b))
def fmt(r):
    a=r["v"]; b=r["t"]
    return (f"{HLAB[r['h']]:>3s} X{r['X']:<3s} q{r['q']*100:>4.1f}% {r['mode']:<5s} entry={r['fill']:<3s} exit={r['exit']:<6s} | VAL n{a['n']:>5d} m{a['nm']:>4d} net{a['mean']:>+7.2f}c t{a['t']:>+6.2f} | TEST n{b['n']:>5d} m{b['nm']:>4d} net{b['mean']:>+7.2f}c t{b['t']:>+6.2f} yes{b['yes']:.2f}")
for r in recs: lines.append(fmt(r))
lines.append(""); lines.append("# Summary: count of cells (of %d per entry/exit combo) with VAL t>2, TEST t>2, both; and best VAL cell per combo:" % (len(recs)//6))
for fill in ("row","lat"):
    for ek in ("mid","taker","settle"):
        c = [r for r in recs if r["fill"]==fill and r["exit"]==ek and r["v"]["nm"]>=20 and np.isfinite(r["v"]["t"]) and np.isfinite(r["t"]["t"])]
        vt = np.array([r["v"]["t"] for r in c]); tt = np.array([r["t"]["t"] for r in c])
        b = max(c, key=lambda r: r["v"]["t"])
        lines.append(f"  entry={fill} exit={ek:<6s}: cells {len(c)}  VAL t>2 {int((vt>2).sum()):3d}  TEST t>2 {int((tt>2).sum()):3d}  both {int(((vt>2)&(tt>2)).sum()):3d}  median VAL t {np.median(vt):+.2f} TEST t {np.median(tt):+.2f}")
        lines.append("     best VAL: " + fmt(b))
with open(os.path.join(SP,"exceed_strat_realistic.txt"),"w") as fh: fh.write("\n".join(lines)+"\n")
print("\n".join(lines[-16:]))
print("# top VAL cell 1s X1 q0.5% rearm, all combos:")
for r in recs:
    if r["h"]=="mid5" and r["X"]=="1" and r["q"]==0.005 and r["mode"]=="rearm": print(fmt(r))
print("# 1s X1.5 q0.5% rearm:")
for r in recs:
    if r["h"]=="mid5" and r["X"]=="1.5" and r["q"]==0.005 and r["mode"]=="rearm": print(fmt(r))

"""Diagnostics on the top VAL-selected cells: per-day net, entry-price buckets, signed mid/quote path
0.2 s -> settle, and how much of the 1 s move is already gone by 200 ms."""
import numpy as np, os
SP = r"C:\Users\fatli\AppData\Local\Temp\claude\e--poly-crypto-trader\0ed64f57-c300-45f3-b675-113fb239783c\scratchpad"
EX = r"E:\poly\crypto_trader\data\nowcast\exceed"
VAL_DAYS = ["2026-08-27","2026-08-28","2026-08-29","2026-08-30","2026-08-31"]
TEST_DAYS = ["2026-09-01","2026-09-02","2026-09-03","2026-09-04"]
FEE = 0.07

def load_split(days, off):
    out = {k:[] for k in ["mid","tte","spread","bid","ask","fut","m3","settle","mkt","day"]}
    for i,day in enumerate(days):
        d = np.load(os.path.join(SP,"midmove_v10",day+".npz")); f = np.load(os.path.join(SP,"midmove_v10_fut",day+".fut.npz"))
        ctx = d["ctx"]; sel = (ctx[:,1]>=60)&(ctx[:,1]<=300); c = ctx[sel]
        out["mid"].append(c[:,0]); out["tte"].append(c[:,1]); out["spread"].append(c[:,3]); out["bid"].append(c[:,4]); out["ask"].append(c[:,5])
        out["fut"].append(d["fut"][sel].reshape(-1,4,3)); out["m3"].append(f["mid3s"][sel]); out["settle"].append(d["settle"][sel])
        out["mkt"].append((off+i)*1000 + d["tk"][sel].astype(np.int64)); out["day"].append(np.full(sel.sum(), i))
    return {k:np.concatenate(v) for k,v in out.items()}

def throttle(q, r, D, mode):
    idx = np.flatnonzero(q)
    if mode == "all" or len(idx)==0: return idx
    if mode == "one":
        _, first = np.unique(D["mkt"][idx], return_index=True); return idx[np.sort(first)]
    mstart = np.ones(len(q), bool); mstart[1:] = D["mkt"][1:] != D["mkt"][:-1]
    seg = np.cumsum(r | mstart); _, first = np.unique(seg[idx], return_index=True); return idx[np.sort(first)]

def clus(net, mkt):
    ok = np.isfinite(net); net=net[ok]; mkt=mkt[ok]
    if len(net)==0: return (0,0,np.nan,np.nan)
    um, inv = np.unique(mkt, return_inverse=True); m = np.bincount(inv, net)/np.bincount(inv)
    t = m.mean()/(m.std(ddof=1)/np.sqrt(len(um))) if len(um)>=3 and m.std(ddof=1)>0 else np.nan
    return (len(net), len(um), net.mean(), t)

VA = load_split(VAL_DAYS,0); TE = load_split(TEST_DAYS,100)
def score(h,X):
    p = np.load(os.path.join(EX, f"std_{h}_x{X}_s0.npz")); return p["p_va"][:,0]-p["p_va"][:,1], p["p_te"][:,0]-p["p_te"][:,1]

CELLS = [("dir 1s X1 q0.5% rearm", "mid5","1",0.005,"rearm",None),
         ("dir 1s X1.5 q0.5% rearm","mid5","1.5",0.005,"rearm",None),
         ("dir 1s X1 q2% rearm",   "mid5","1",0.02,"rearm",None),
         ("tails 1s X1 q10% rearm","mid5","1",0.10,"rearm","tails"),
         ("agree1s5s 5s X1 q0.5% rearm","mid25","1",0.005,"rearm","agree")]
out = []
for name,h,X,q,mode,filt in CELLS:
    sv, st = score(h,X); cut = np.quantile(np.abs(sv),1-q)
    qv = np.abs(sv)>=cut; qt = np.abs(st)>=cut
    if filt=="tails": qv &= (VA["mid"]<0.10)|(VA["mid"]>0.90); qt &= (TE["mid"]<0.10)|(TE["mid"]>0.90)
    if filt=="agree":
        s1v,s1t = score("mid5",X); c1 = np.quantile(np.abs(s1v),1-q)
        qv &= (np.sign(s1v)==np.sign(sv))&(np.abs(s1v)>=c1); qt &= (np.sign(s1t)==np.sign(st))&(np.abs(s1t)>=c1)
    out.append(""); out.append(f"===== {name}  cut={cut:.3f}")
    for D,S,label,days in ((VA,sv,"VAL",VAL_DAYS),(TE,st,"TEST",TEST_DAYS)):
        idx = throttle(np.abs(S)>=cut if filt is None else (qv if label=="VAL" else qt), np.abs(S)<cut/2, D, mode)
        side = np.where(S[idx]>=0,1,-1)
        mid = D["mid"][idx]; fut = D["fut"][idx]
        price = np.where(side>0, D["ask"][idx], 1-D["bid"][idx]); fee = FEE*price*(1-price)
        price_lat = np.where(side>0, fut[:,0,2], 1-fut[:,0,1])
        hk = {"mid5":1,"mid25":2}[h]
        exit_h = np.where(side>0, fut[:,hk,0], 1-fut[:,hk,0])
        net_h = (exit_h - price - fee)*100
        net_lat = (exit_h - price_lat - FEE*price_lat*(1-price_lat))*100
        net_set = (np.where(side>0, D["settle"][idx], 1-D["settle"][idx]) - price - fee)*100
        net_30 = (np.where(side>0, fut[:,3,0], 1-fut[:,3,0]) - price - fee)*100
        # signed moves (in trade direction), cents
        mv = {lab:(np.where(side>0, v, -v) - np.where(side>0, mid, -mid))*100 for lab,v in
              (("0.2s",fut[:,0,0]),("1s",fut[:,1,0]),("3s",D["m3"][idx]),("5s",fut[:,2,0]),("30s",fut[:,3,0]),("settle",D["settle"][idx]))}
        slip = (price_lat - price)*100      # fill price 200 ms later minus fill at the row, in trade direction
        out.append(f"-- {label}: n={len(idx)} markets={len(np.unique(D['mkt'][idx]))} YES share={np.mean(side>0):.2f} mean mid at entry={mid.mean():.3f} mean spread={D['spread'][idx].mean()*100:.2f}c mean fee={fee.mean()*100:.2f}c")
        out.append("   signed mid path (c): " + "  ".join(f"{k} {np.nanmean(v):+.2f}" for k,v in mv.items()) + f" | fill slippage @200ms {np.nanmean(slip):+.2f}c (share of 1s mid move already in the 200ms quote: {np.nanmean(slip)/max(np.nanmean(mv['1s']),1e-9):.0%})")
        out.append(f"   frac rows with 200ms mid move >= 1c in trade dir: {np.nanmean(mv['0.2s']>=0.99):.2f};  frac 1s move >=1c: {np.nanmean(mv['1s']>=0.99):.2f};  frac 1s move <= -1c: {np.nanmean(mv['1s']<=-0.99):.2f}")
        for lab,net in (("mark@h row-fill",net_h),("mark@h lat200-fill",net_lat),("mark@30s row-fill",net_30),("settle row-fill",net_set)):
            n,nm,m,t = clus(net, D["mkt"][idx]); out.append(f"   {lab:<20s} n{n:>5d} m{nm:>4d} net{m:>+7.2f}c t{t:>+6.2f}")
        # per day
        out.append("   per day (mark@h | settle):  " + "  ".join(
            f"{days[d][5:]} n{(D['day'][idx]==d).sum():>3d} {clus(net_h[D['day'][idx]==d], D['mkt'][idx][D['day'][idx]==d])[2]:+.2f}c/t{clus(net_h[D['day'][idx]==d], D['mkt'][idx][D['day'][idx]==d])[3]:+.1f} | {clus(net_set[D['day'][idx]==d], D['mkt'][idx][D['day'][idx]==d])[2]:+.1f}c"
            for d in range(len(days))))
        # price buckets
        edges = [0,0.10,0.30,0.70,0.90,1.01]
        rowsb = []
        for lo,hi in zip(edges[:-1],edges[1:]):
            b = (mid>=lo)&(mid<hi)
            if b.sum()==0: continue
            n,nm,m,t = clus(net_h[b], D["mkt"][idx][b]); n2,nm2,m2,t2 = clus(net_lat[b], D["mkt"][idx][b]); n3,nm3,m3,t3 = clus(net_set[b], D["mkt"][idx][b])
            rowsb.append(f"[{lo:.1f},{hi:.1f}) n{n:>4d} m{nm:>3d} spread{D['spread'][idx][b].mean()*100:.2f}c mark@h {m:+.2f}c t{t:+.1f} | lat200 {m2:+.2f}c t{t2:+.1f} | settle {m3:+.1f}c t{t3:+.1f}")
        out.append("   by entry mid: " + "\n                 ".join(rowsb))
        # side split
        for sname, sm in (("YES",side>0),("NO",side<0)):
            if sm.sum()==0: continue
            n,nm,m,t = clus(net_h[sm], D["mkt"][idx][sm]); n3,nm3,m3,t3 = clus(net_set[sm], D["mkt"][idx][sm])
            out.append(f"   side {sname}: n{n:>4d} mark@h {m:+.2f}c t{t:+.1f} | settle {m3:+.1f}c t{t3:+.1f} | mean mid {mid[sm].mean():.3f}")
with open(os.path.join(SP,"exceed_strat_diag.txt"),"w") as fh: fh.write("\n".join(out)+"\n")
print("\n".join(out))

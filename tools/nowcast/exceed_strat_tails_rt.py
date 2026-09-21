"""Tails family (entry mid <0.10 or >0.90) under the realistic ladder: row/lat200 entry x mid/taker/settle exit, 1 s and 5 s horizons."""
import numpy as np, os
exec(open(os.path.join(r"C:\Users\fatli\AppData\Local\Temp\claude\e--poly-crypto-trader\0ed64f57-c300-45f3-b675-113fb239783c\scratchpad","exceed_strat_realistic.py")).read().split("VA = load_split")[0])
def fmt(r):
    a=r["v"]; b=r["t"]
    return (f"{HLAB[r['h']]:>3s} X{r['X']:<3s} q{r['q']*100:>4.1f}% {r['mode']:<5s} entry={r['fill']:<3s} exit={r['exit']:<6s} | VAL n{a['n']:>5d} m{a['nm']:>4d} net{a['mean']:>+7.2f}c t{a['t']:>+6.2f} | TEST n{b['n']:>5d} m{b['nm']:>4d} net{b['mean']:>+7.2f}c t{b['t']:>+6.2f} yes{b['yes']:.2f}")
VA = load_split(VAL_DAYS,0); TE = load_split(TEST_DAYS,100)
tv =(VA["mid"]<0.10)|(VA["mid"]>0.90); tt = (TE["mid"]<0.10)|(TE["mid"]>0.90)
lines = ["# Tails (mid<0.10 or >0.90) realistic ladder, throttle=rearm"]
recs = []
for h in ("mid5","mid25"):
    for X in XS:
        p = np.load(os.path.join(EX, f"std_{h}_x{X}_s0.npz")); sv = p["p_va"][:,0]-p["p_va"][:,1]; st = p["p_te"][:,0]-p["p_te"][:,1]
        for q in QS:
            cut = np.quantile(np.abs(sv), 1-q)
            iv = throttle((np.abs(sv)>=cut)&tv, np.abs(sv)<cut/2, VA, "rearm"); it = throttle((np.abs(st)>=cut)&tt, np.abs(st)<cut/2, TE, "rearm")
            sidev = np.where(sv[iv]>=0,1,-1); sidet = np.where(st[it]>=0,1,-1)
            for fill in ("row","lat"):
                pv_, fv_ = entry_price(VA, iv, sidev, fill); pt_, ft_ = entry_price(TE, it, sidet, fill)
                for ek in ("mid","taker","settle"):
                    nv = (exit_value(VA, iv, sidev, h, ek) - pv_ - fv_)*100; nt = (exit_value(TE, it, sidet, h, ek) - pt_ - ft_)*100
                    recs.append(dict(h=h,X=X,q=q,cut=cut,mode="rearm",fill=fill,exit=ek,v=stats(nv, VA["mkt"][iv], sidev),t=stats(nt, TE["mkt"][it], sidet)))
for r in recs: lines.append(fmt(r))
lines.append("")
for fill in ("row","lat"):
    for ek in ("mid","taker","settle"):
        c = [r for r in recs if r["fill"]==fill and r["exit"]==ek and r["v"]["nm"]>=20 and np.isfinite(r["v"]["t"]) and np.isfinite(r["t"]["t"])]
        vt = np.array([r["v"]["t"] for r in c]); tt_ = np.array([r["t"]["t"] for r in c]); b = max(c, key=lambda r: r["v"]["t"])
        lines.append(f"  tails entry={fill} exit={ek:<6s}: cells {len(c)}  VAL t>2 {int((vt>2).sum()):3d}  TEST t>2 {int((tt_>2).sum()):3d}  both {int(((vt>2)&(tt_>2)).sum()):3d}  median VAL t {np.median(vt):+.2f} TEST t {np.median(tt_):+.2f}")
        lines.append("     best VAL: " + fmt(b))
with open(os.path.join(SP,"exceed_strat_tails_rt.txt"),"w") as fh: fh.write("\n".join(lines)+"\n")
print("\n".join(lines[-13:]))

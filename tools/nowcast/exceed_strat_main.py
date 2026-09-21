"""Offline strategy study on saved exceedance-model predictions (KXBTC15M).

Row study: taker fills at the row ask/bid (no latency, no chase) unless the
fill variant says otherwise; fee = 0.07*p*(1-p) per contract at traded price p.
Selection on VAL (2026-08-27..31) only; TEST (09-01..04) is report-only.
"""
import numpy as np, os, sys, time, csv
SP = r"C:\Users\fatli\AppData\Local\Temp\claude\e--poly-crypto-trader\0ed64f57-c300-45f3-b675-113fb239783c\scratchpad"
EX = r"E:\poly\crypto_trader\data\nowcast\exceed"
VAL_DAYS = ["2026-08-27","2026-08-28","2026-08-29","2026-08-30","2026-08-31"]
TEST_DAYS = ["2026-09-01","2026-09-02","2026-09-03","2026-09-04"]
HS = ["mid5","mid15","mid25","mid150"]          # 1s, 3s, 5s, 30s
HLAB = {"mid5":"1s","mid15":"3s","mid25":"5s","mid150":"30s"}
XS = ["1","1.5","2","3","5"]
QS = [0.005,0.01,0.02,0.05,0.10]
FEE = 0.07

def load_split(days, off):
    out = {k:[] for k in ["mid","tte","spread","bid","ask","f0b","f0a","m1","m3","m5","m30","settle","mkt","ts","day"]}
    for i,day in enumerate(days):
        d = np.load(os.path.join(SP,"midmove_v10",day+".npz"))
        f = np.load(os.path.join(SP,"midmove_v10_fut",day+".fut.npz"))
        ctx = d["ctx"]; sel = (ctx[:,1]>=60)&(ctx[:,1]<=300)
        c = ctx[sel]; fu = d["fut"][sel].reshape(-1,4,3)
        out["mid"].append(c[:,0]); out["tte"].append(c[:,1]); out["spread"].append(c[:,3])
        out["bid"].append(c[:,4]); out["ask"].append(c[:,5])
        out["f0b"].append(fu[:,0,1]); out["f0a"].append(fu[:,0,2])
        out["m1"].append(fu[:,1,0]); out["m5"].append(fu[:,2,0]); out["m30"].append(fu[:,3,0])
        out["m3"].append(f["mid3s"][sel].astype(np.float32))
        out["settle"].append(d["settle"][sel])
        out["mkt"].append((off+i)*1000 + d["tk"][sel].astype(np.int64))
        out["ts"].append(d["ts"][sel]); out["day"].append(np.full(sel.sum(), off+i, dtype=np.int32))
    D = {k:np.concatenate(v) for k,v in out.items()}
    order = np.lexsort((D["ts"], D["mkt"]))
    D["inorder"] = bool(np.all(order == np.arange(len(order))))
    return D

def load_pred(kind, h, X):
    p = np.load(os.path.join(EX, f"{kind}_{h}_x{X}_s0.npz"))
    return p["p_va"].astype(np.float64), p["p_te"].astype(np.float64), p["okva"], p["okte"]

# ---------------------------------------------------------------- helpers
def exit_mid(D, key):
    return {"mid5":D["m1"],"mid15":D["m3"],"mid25":D["m5"],"mid150":D["m30"],"30s":D["m30"]}[key]

def throttle(q, r, D, mode):
    """q: qualifying rows (bool), r: re-arm rows (bool). Rows are time-ordered within market."""
    idx = np.flatnonzero(q)
    if mode == "all" or len(idx)==0: return idx
    if mode == "one":
        _, first = np.unique(D["mkt"][idx], return_index=True)
        return idx[np.sort(first)]
    if mode == "rearm":
        mstart = np.ones(len(q), bool); mstart[1:] = D["mkt"][1:] != D["mkt"][:-1]
        seg = np.cumsum(r | mstart)
        _, first = np.unique(seg[idx], return_index=True)
        return idx[np.sort(first)]
    raise ValueError(mode)

def net_cents(D, idx, side, exitv, fill="row"):
    """side +1 YES / -1 NO; exitv = YES value at exit (mid or settle) over all rows."""
    if fill == "row":
        a = D["ask"][idx]; b = D["bid"][idx]; feek = FEE
    elif fill == "lat":          # fill at the quotes 200 ms later (latency proxy)
        a = D["f0a"][idx]; b = D["f0b"][idx]; feek = FEE
    elif fill == "maker":        # OPTIMISTIC: a resting order at the row bid/ask is assumed filled, no fee
        a = D["bid"][idx]; b = D["ask"][idx]; feek = 0.0
    price = np.where(side>0, a, 1.0-b)
    fee = feek*price*(1.0-price)
    ev = exitv[idx]
    val = np.where(side>0, ev, 1.0-ev)
    return (val - price - fee)*100.0

def stats(net, mkt, side):
    ok = np.isfinite(net)
    net = net[ok]; mkt = mkt[ok]; side = side[ok]
    n = len(net)
    if n == 0: return dict(n=0, nm=0, mean=np.nan, t=np.nan, yes=np.nan, tot=0.0)
    um, inv = np.unique(mkt, return_inverse=True)
    m = np.bincount(inv, net)/np.bincount(inv)
    nm = len(um)
    t = m.mean()/(m.std(ddof=1)/np.sqrt(nm)) if nm>=3 and m.std(ddof=1)>0 else np.nan
    return dict(n=n, nm=nm, mean=net.mean(), t=t, yes=(side>0).mean(), tot=net.sum())

def run_cell(D, q, r, side, exitv, mode, fill="row"):
    idx = throttle(q, r, D, mode)
    if len(idx)==0: return stats(np.array([]), np.array([]), np.array([]))
    net = net_cents(D, idx, side[idx], exitv, fill)
    return stats(net, D["mkt"][idx], side[idx])

# ---------------------------------------------------------------- main
t0 = time.time()
VA = load_split(VAL_DAYS, 0); TE = load_split(TEST_DAYS, 100)
print("VAL rows", len(VA["mid"]), "in order", VA["inorder"], "| TEST rows", len(TE["mid"]), "in order", TE["inorder"])
assert VA["inorder"] and TE["inorder"], "rows not time-ordered within market; would need permutation"
print("VAL markets", len(np.unique(VA["mkt"])), "TEST markets", len(np.unique(TE["mkt"])))
print("spread dist VAL (c):", np.percentile(VA["spread"]*100,[10,50,90]).round(2), "TEST:", np.percentile(TE["spread"]*100,[10,50,90]).round(2))

pv,pt,okv,okt = load_pred("std","mid5","1")
lab = (VA["m1"] - VA["mid"]) >= 0.0099
print("align check std 1s x1 VAL: mean P(up)|up=%.3f  P(up)|not=%.3f  base rate=%.4f  okva==finite(m1): %s" % (
    pv[okv & lab,0].mean(), pv[okv & ~lab,0].mean(), lab[okv].mean(), np.array_equal(okv, np.isfinite(VA["m1"]))))

PRED = {}
for kind in ["std","ctx"]:
    for h in HS:
        for X in XS:
            PRED[(kind,h,X)] = load_pred(kind,h,X)
print("preds loaded", time.time()-t0)

rows = []
def rec(fam, h, X, q, cut, exitk, mode, fill, sv, st):
    rows.append(dict(fam=fam, h=h, X=X, q=q, cut=cut, exit=exitk, mode=mode, fill=fill,
                     v_n=sv["n"], v_nm=sv["nm"], v_mean=sv["mean"], v_t=sv["t"], v_yes=sv["yes"], v_tot=sv["tot"],
                     t_n=st["n"], t_nm=st["nm"], t_mean=st["mean"], t_t=st["t"], t_yes=st["yes"], t_tot=st["tot"]))

def exits_for(h):
    ex = [("h", h)]
    if h != "mid150": ex.append(("30s","30s"))
    ex.append(("settle","settle"))
    return ex

def exit_arrays(D, key):
    return D["settle"] if key=="settle" else exit_mid(D, key)

def dir_family(fam, kind, filt_va=None, filt_te=None, fill="row", agree=False, ctxdis=False, modes=("all","one","rearm")):
    for h in HS:
        for X in XS:
            pv,pt,okv,okt = PRED[(kind,h,X)]
            sv_ = pv[:,0]-pv[:,1]; st_ = pt[:,0]-pt[:,1]
            for q in QS:
                cut = np.quantile(np.abs(sv_), 1-q)
                qv = np.abs(sv_)>=cut; qt = np.abs(st_)>=cut
                rv = np.abs(sv_)<cut/2; rt = np.abs(st_)<cut/2
                sidev = np.where(sv_>=0,1,-1); sidet = np.where(st_>=0,1,-1)
                if filt_va is not None: qv = qv & filt_va; qt = qt & filt_te
                if agree:   # 1s and 5s scores must agree in sign and both exceed their own same-quantile cut
                    if h != "mid25": continue
                    p1v,p1t,_,_ = PRED[(kind,"mid5",X)]
                    s1v = p1v[:,0]-p1v[:,1]; s1t = p1t[:,0]-p1t[:,1]
                    c1 = np.quantile(np.abs(s1v), 1-q)
                    qv = qv & (np.sign(s1v)==np.sign(sv_)) & (np.abs(s1v)>=c1)
                    qt = qt & (np.sign(s1t)==np.sign(st_)) & (np.abs(s1t)>=c1)
                if ctxdis:  # context-only model must NOT fire in the same direction at its own same-quantile cut
                    cv,ct,_,_ = PRED[("ctx",h,X)]
                    csv_ = cv[:,0]-cv[:,1]; cst_ = ct[:,0]-ct[:,1]
                    cc = np.quantile(np.abs(csv_), 1-q)
                    qv = qv & ((sidev*csv_) < cc); qt = qt & ((sidet*cst_) < cc)
                for ek, ekey in exits_for(h):
                    exv = exit_arrays(VA, ekey); ext = exit_arrays(TE, ekey)
                    for mode in modes:
                        sv = run_cell(VA, qv, rv, sidev, exv, mode, fill)
                        st = run_cell(TE, qt, rt, sidet, ext, mode, fill)
                        rec(fam, h, X, q, cut, ek, mode, fill, sv, st)

def single_family(fam, kind):
    for h in HS:
        for X in XS:
            pv,pt,okv,okt = PRED[(kind,h,X)]
            for q in QS:
                pool = np.concatenate([pv[:,0], pv[:,1]])
                cut = np.quantile(pool, 1-q)
                qv = (pv[:,0]>=cut)|(pv[:,1]>=cut); qt = (pt[:,0]>=cut)|(pt[:,1]>=cut)
                rv = np.maximum(pv[:,0],pv[:,1])<cut/2; rt = np.maximum(pt[:,0],pt[:,1])<cut/2
                sidev = np.where(pv[:,0]>=pv[:,1],1,-1); sidet = np.where(pt[:,0]>=pt[:,1],1,-1)
                for ek, ekey in exits_for(h):
                    exv = exit_arrays(VA, ekey); ext = exit_arrays(TE, ekey)
                    for mode in ("all","one","rearm"):
                        sv = run_cell(VA, qv, rv, sidev, exv, mode)
                        st = run_cell(TE, qt, rt, sidet, ext, mode)
                        rec(fam, h, X, q, cut, ek, mode, "row", sv, st)

print("running families...")
dir_family("dir", "std")
dir_family("ctxdir", "ctx")                       # context-only control, own quantile cuts
dir_family("dir_lat200", "std", fill="lat", modes=("one","rearm"))
dir_family("dir_maker", "std", fill="maker", modes=("one","rearm"))
single_family("single", "std")
single_family("ctxsingle", "ctx")
tailv = (VA["mid"]<0.10)|(VA["mid"]>0.90); tailt = (TE["mid"]<0.10)|(TE["mid"]>0.90)
dir_family("tails", "std", tailv, tailt, modes=("one","rearm"))
atmv = (VA["mid"]>=0.30)&(VA["mid"]<=0.70); atmt = (TE["mid"]>=0.30)&(TE["mid"]<=0.70)
dir_family("atm", "std", atmv, atmt, modes=("one","rearm"))
tv = VA["spread"]<=0.0101; tt = TE["spread"]<=0.0101
dir_family("tight1c", "std", tv, tt, modes=("one","rearm"))
dir_family("agree1s5s", "std", agree=True, modes=("all","one","rearm"))
dir_family("ctxdisagree", "std", ctxdis=True, modes=("one","rearm"))
latev = VA["tte"]<=120; latet = TE["tte"]<=120
dir_family("late120", "std", latev, latet, modes=("one","rearm"))
print("cells", len(rows), "elapsed", time.time()-t0)

# ---------------------------------------------------------------- output
fields = ["fam","h","X","q","cut","exit","mode","fill","v_n","v_nm","v_mean","v_t","v_yes","v_tot","t_n","t_nm","t_mean","t_t","t_yes","t_tot"]
with open(os.path.join(SP,"exceed_strat_grid.csv"),"w",newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore"); w.writeheader()
    for r in rows: w.writerow({k:(f"{r[k]:.4f}" if isinstance(r[k],float) else r[k]) for k in fields})

def fmt(r):
    return (f"{r['fam']:<11s} {HLAB[r['h']]:>3s} X{r['X']:<3s} q{r['q']*100:>4.1f}% cut{r['cut']:.3f} {r['exit']:<6s} {r['mode']:<5s} {r['fill']:<5s} | "
            f"VAL n{r['v_n']:>6d} m{r['v_nm']:>4d} net{r['v_mean']:>+7.2f}c t{r['v_t']:>+6.2f} yes{r['v_yes']:>4.2f} | "
            f"TEST n{r['t_n']:>6d} m{r['t_nm']:>4d} net{r['t_mean']:>+7.2f}c t{r['t_t']:>+6.2f} yes{r['t_yes']:>4.2f}")

lines = []
lines.append("# Full grid (all families). net = exit value - entry cost - fee, cents/contract; t = event-clustered (per-market means).")
lines.append("# Fill row = ask/bid at the row (no latency/chase); lat = quotes 200 ms later; maker = OPTIMISTIC resting fill at bid/ask, no fee.")
for r in rows: lines.append(fmt(r))
with open(os.path.join(SP,"exceed_strat_full.txt"),"w") as fh: fh.write("\n".join(lines)+"\n")

# ---------------------------------------------------------------- selection on VAL
MINM = 20
summ = []
summ.append("# Best VAL-selected cell per family (selection = max VAL event-clustered t, VAL markets >= %d); TEST is report-only." % MINM)
summ.append("# Also shown: the same cell under the other throttles, and the ctx control at the same (h,X,q,exit,mode).")
fams = []
for r in rows:
    if r["fam"] not in fams: fams.append(r["fam"])
def find(fam, h, X, q, exitk, mode):
    for r in rows:
        if (r["fam"],r["h"],r["X"],r["q"],r["exit"],r["mode"])==(fam,h,X,q,exitk,mode): return r
best = {}
for fam in fams:
    cand = [r for r in rows if r["fam"]==fam and r["v_nm"]>=MINM and np.isfinite(r["v_t"])]
    if not cand: continue
    b = max(cand, key=lambda r: r["v_t"])
    best[fam] = b
    summ.append(""); summ.append(f"== {fam}: best by VAL t"); summ.append(fmt(b))
    for mode in ("all","one","rearm"):
        o = find(fam, b["h"], b["X"], b["q"], b["exit"], mode)
        if o is not None and mode != b["mode"]: summ.append("   alt-throttle " + fmt(o))
    ctrl = "ctxsingle" if fam.endswith("single") else "ctxdir"
    if fam not in ("ctxdir","ctxsingle"):
        c = find(ctrl, b["h"], b["X"], b["q"], b["exit"], b["mode"])
        if c is not None: summ.append("   ctx-control  " + fmt(c))
    b2 = max(cand, key=lambda r: r["v_mean"])
    if b2 is not b: summ.append("   best-by-VAL-mean " + fmt(b2))
    both = [r for r in cand if r["v_t"]>2 and (r["t_t"]>2)]
    pos_v = [r for r in cand if r["v_t"]>2]
    tts = [r['t_t'] for r in cand if np.isfinite(r['t_t'])]
    summ.append(f"   cells: {len(cand)} eligible; VAL t>+2: {len(pos_v)}; VAL t>+2 AND TEST t>+2: {len(both)}; "
                f"median VAL t {np.median([r['v_t'] for r in cand]):+.2f}, median TEST t {np.median(tts) if tts else float('nan'):+.2f}")
    for r in both[:10]: summ.append("   BOTH>2 " + fmt(r))

for mode in ("rearm","one","all"):
    summ.append(""); summ.append(f"# Directional taker (std) grid, throttle={mode}: h in {{1s,5s}} x X x cut; exit mark-to-h | hold-to-settle. ctx control in brackets.")
    summ.append(f"{'h':>3s} {'X':>4s} {'q':>5s} {'cut':>6s} | {'VAL n':>6s} {'mkts':>4s} {'mark net':>8s} {'t':>6s} {'settle net':>10s} {'t':>6s} | {'TEST n':>6s} {'mkts':>4s} {'mark net':>8s} {'t':>6s} {'settle net':>10s} {'t':>6s} | ctx [mark VALt/TESTt, settle VALt/TESTt] ctx n")
    for h in ("mid5","mid25"):
        for X in XS:
            for q in QS:
                a = find("dir",h,X,q,"h",mode); s = find("dir",h,X,q,"settle",mode)
                ca = find("ctxdir",h,X,q,"h",mode); cs = find("ctxdir",h,X,q,"settle",mode)
                summ.append(f"{HLAB[h]:>3s} {X:>4s} {q*100:>4.1f}% {a['cut']:>6.3f} | {a['v_n']:>6d} {a['v_nm']:>4d} {a['v_mean']:>+8.2f} {a['v_t']:>+6.2f} {s['v_mean']:>+10.2f} {s['v_t']:>+6.2f} | "
                            f"{a['t_n']:>6d} {a['t_nm']:>4d} {a['t_mean']:>+8.2f} {a['t_t']:>+6.2f} {s['t_mean']:>+10.2f} {s['t_t']:>+6.2f} | "
                            f"[{ca['v_t']:>+5.2f}/{ca['t_t']:>+5.2f}, {cs['v_t']:>+5.2f}/{cs['t_t']:>+5.2f}] {ca['v_n']}/{ca['t_n']}")

summ.append(""); summ.append("# Census over ALL cells (std families only, VAL markets>=20):")
allc = [r for r in rows if not r["fam"].startswith("ctx") and r["v_nm"]>=MINM and np.isfinite(r["v_t"]) and np.isfinite(r["t_t"])]
vt = np.array([r["v_t"] for r in allc]); tt_ = np.array([r["t_t"] for r in allc])
summ.append(f"  n cells {len(allc)}; VAL t>2: {(vt>2).sum()}; TEST t>2: {(tt_>2).sum()}; both: {((vt>2)&(tt_>2)).sum()}; corr(VAL t, TEST t) = {np.corrcoef(vt,tt_)[0,1]:+.3f}")
summ.append(f"  VAL mean net>0: {(np.array([r['v_mean'] for r in allc])>0).sum()}; TEST mean net>0: {(np.array([r['t_mean'] for r in allc])>0).sum()}")
allctx = [r for r in rows if r["fam"].startswith("ctx") and r["v_nm"]>=MINM and np.isfinite(r["v_t"]) and np.isfinite(r["t_t"])]
vt2 = np.array([r["v_t"] for r in allctx]); tt2 = np.array([r["t_t"] for r in allctx])
summ.append(f"  ctx control: n cells {len(allctx)}; VAL t>2: {(vt2>2).sum()}; TEST t>2: {(tt2>2).sum()}; both: {((vt2>2)&(tt2>2)).sum()}")
summ.append(""); summ.append("# Cells with VAL t>2 sorted by TEST t (report-only; NOT a selection):")
for r in sorted([r for r in allc if r["v_t"]>2], key=lambda r:-r["t_t"])[:15]: summ.append("  "+fmt(r))
summ.append(""); summ.append("# Gross diagnostic: mean signed mid move (c, in trade direction) on qualifying rows, dir std, all rows, q=1%; vs half-spread + fee at mid:")
for h in HS:
    for X in XS:
        pv,pt,_,_ = PRED[("std",h,X)]
        for DD,name,pp in ((VA,"VAL",pv),(TE,"TEST",pt)):
            s_ = pp[:,0]-pp[:,1]; cut = np.quantile(np.abs(pv[:,0]-pv[:,1]), 0.99)
            q = np.abs(s_)>=cut; side = np.where(s_>=0,1,-1)
            mv = (exit_mid(DD,h)-DD["mid"])*side*100; mv30 = (DD["m30"]-DD["mid"])*side*100; mst = (DD["settle"]-DD["mid"])*side*100
            hs = (DD["spread"]/2*100)
            summ.append(f"  {HLAB[h]:>3s} X{X:<3s} {name:<4s} n{q.sum():>5d} mid move@h {np.nanmean(mv[q]):+.2f}c  @30s {np.nanmean(mv30[q]):+.2f}c  to settle {np.nanmean(mst[q]):+.2f}c | half-spread {hs[q].mean():.2f}c + fee@mid {(FEE*DD['mid'][q]*(1-DD['mid'][q])*100).mean():.2f}c | mid {DD['mid'][q].mean():.2f}")
with open(os.path.join(SP,"exceed_strat_summary.txt"),"w") as fh: fh.write("\n".join(summ)+"\n")
print("\n".join(summ))
print("done", time.time()-t0)
